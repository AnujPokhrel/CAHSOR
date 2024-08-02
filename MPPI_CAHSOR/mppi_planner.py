#!/usr/bin/env python3
#Python Basics
import time
import sys
import rospy
import numpy as np
sys.path.append("/home/user_name/catkin_ws/src/experiments")

#torch
import torch
import torch.nn as nn
import torchvision.transforms.functional as F

#ROS imports
import rospy
from nav_msgs.msg import Path, Odometry 
from geometry_msgs.msg import PoseStamped, Pose, Twist
from sensor_msgs.msg import CompressedImage
from ublox_msgs.msg import NavPVT

#local helpers
import DataProcessingPipeline.Helpers.image_processing as image_processing
from MPPI_CAHSOR.MPPIHelpers.utilities import utils
import pickle
from TRON.model.model import DownstreamNetwork
from TRON.model.tron_model import IMUEncoder, VisionSpeedEncoder

MAX_VEL = 4.8
MIN_VEL = -1.0

class mppi_planner:
    def __init__(self, T, K, sigma=[0.5, 0.1], _lambda=0.5):
        #Robot Limits
        self.device = torch.device('cuda')
        self.max_vel = MAX_VEL 
        self.min_vel = MIN_VEL
        self.max_del = 0.51
        self.min_del = -0.51
        self.robot_length = 0.84
        self.dtype = torch.float32                                                                  # Data type

        #External class definations
        self.util = utils(K)

        #variables 
        self.goal_tensor = None                                                                     # Goal tensor
        self.lasttime = None     
        
        self.rep_size = 512
        self.device = torch.device('cuda')
        
        load_dir = 'path_to_model/model.pth' 
        checkpoint = torch.load(load_dir, map_location=self.device)
        visual_encoder = VisionSpeedEncoder(latent_size=self.rep_size)
        
        imu_encoder = IMUEncoder(latent_size=self.rep_size)

        self.cahsor_model = DownstreamNetwork(visual_encoder, imu_encoder, rep_size=self.rep_size,)# modality='VS')
        self.cahsor_model.load_state_dict(checkpoint['model'], strict=False)
        self.cahsor_model.eval()
        self.cahsor_model.to(self.device)
        with open("path_to_stats/stats.pkl", 'rb') as f:
            self.stats = pickle.load(f)
        
        self.stats['cmd_vel_mean'] = torch.tensor(self.stats['cmd_vel_mean'])[-6:].cuda()
        self.stats['cmd_vel_std'] = torch.tensor(self.stats['cmd_vel_std'])[-6:].cuda()
        self.stats['velocity_mean'] = torch.tensor(self.stats['velocity_mean']).cuda()
        self.stats['velocity_std'] = torch.tensor(self.stats['velocity_std']).cuda()
        self.stats['res_vel_omega_roll_slde_bump_mean'] = torch.tensor(self.stats['res_vel_omega_roll_slde_bump_mean']).cuda()
        self.stats['res_vel_omega_roll_slde_bump_std'] = torch.tensor(self.stats['res_vel_omega_roll_slde_bump_std']).cuda()
        
        self.goals = self.util.read_waypoints_to_xy("path_to_helpers/MPPIHelpers/xy_coordinates.csv", self.dtype)
        self.goal_tensor = torch.Tensor([5.0, 0.0, 0.0,
                                        0.0, 0.0, -0.278824]).type(self.dtype)
        
        # print("Loading:", model_name)                                                               
        # print("Model:\n",self.model)
        print("Torch Datatype:", self.dtype)

        #------MPPI variables and constants----
        #Parameters
        self.T = T                                                                                  # Length of rollout horizon
        self.K = K                                                                                  # Number of sample rollouts
        self.dt = 0.5                   
        self._lambda = float(_lambda)                                                               # 
        self.sigma = torch.Tensor(sigma).type(self.dtype).expand(self.T, self.K, 2).cuda()               # (T,K,2)
        self.inv_sigma = 1/ self.sigma[0,0,:]                                                       # (2, )
        self.nn_scale = [0.8014116190759601, 0.650365580673219, 0.9859552932867042, 0.9883425253246529]

        self.robot_pose = Pose()                                                                    # Robot pose
        self.odom_msg = Odometry()
        self.ctrlmsg = Twist()
        self.noise = torch.Tensor(self.T, self.K, 2).type(self.dtype).cuda()                              # (T,K,2)
        self.poses = torch.zeros(self.K, self.T, 6).type(self.dtype).cuda()                           # (K,T,6)
        self.fpose = torch.Tensor(self.K, 6).type(self.dtype)                                    # (K,6)
        self.last_pose = None
        self.at_goal = False 
        self.curr_pose = torch.zeros(6).type(self.dtype)
        self.pose_dot = None
        self.image_init = False
        self.difference_from_goal = 0.0
        
        #Cost variables
        self.running_cost = torch.zeros(self.K).type(self.dtype)                                    # (K, )
        self.ctrl_cost = torch.Tensor(self.K, 2).type(self.dtype).cuda()                                   # (K,2)
        self.ctrl_change = torch.Tensor(self.T,2).type(self.dtype).cuda()                                  # (T,2)
        self.euclidian_distance = torch.Tensor(self.K).type(self.dtype)                             # (K, )
        self.dist_to_goal = torch.Tensor(self.K, 6).type(self.dtype)                                # (K, )  
        
        #costs for hunter
        self.bumpiness_cost = None
        self.rolling_cost = None
        self.sliding_cost = None

        self.recent_controls = np.zeros((3,2))
        self.control_i = 0
        self.msgid = 0
        self.speed = 0
        self.steering_angle = 0
        self.prev_ctrl = None
        self.ctrl = torch.zeros((T,2)).cuda()
        self.rate_ctrl = 0

        self.previous_nano = 0
        self.image_stack = torch.zeros((765, 3, 64, 64), dtype=torch.float32).cuda()
        self.image_poses = torch.zeros((765, 2)).cuda()
        self.n_gps = 0
        self.gps_pose_batch = np.zeros((21, 2))
        self.pvt_initialized = False
        self.gpsx_offset = 0.0
        self.gpsy_offset = 0.0
        self.gps_heading = 0.0
        self.last_iTOW = Odometry().header.stamp
        self.print_counter = 0
        self.prev_print_counter = 0
        self.hunter_linear_vel = 0.0
        self.previous_nano = 0
        self.gps_velocity = torch.zeros(2).type(self.dtype).cuda()
        self.cmd_vel = torch.zeros((self.K, 6)).type(self.dtype).cuda()
        self.steer_joy = 0.0
        self.times_mppi_called = 0

        #publisher / Subscribers
        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_cb, queue_size=1)
        rospy.Subscriber('/rgb_publisher/color/image/compressed', CompressedImage, self.image_callback, queue_size=10)
        rospy.Subscriber('/f9p_rover/navpvt', NavPVT, self.gps_cb, queue_size=1)
        
        self.cont_ctrl = True
        self.viz = True
        self.num_viz_paths = 50
        if self.K < self.num_viz_paths:
            self.num_viz_paths = self.K
        
        self.path_pub = rospy.Publisher("/mppi/paths", Path, queue_size = self.num_viz_paths)
        self.central_path_pub = rospy.Publisher("/mppi/path_center", Path, queue_size = 1)
        self.pose_pub = rospy.Publisher("/mppi/pose", PoseStamped, queue_size=1)
        self.goal_pub = rospy.Publisher("/mppi/goal", PoseStamped, queue_size=1)
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        print("MPPI initialization complete")

    def hunter_callback(self, msg):
        self.hunter_linear_vel = msg.linear_velocity

    def image_callback(self, image):
        self.last_image = image
        if self.image_init == False:
            self.image_init = True

    #update the image grid with new images and their poses
    def update_image(self):
        self.image_stack = torch.roll(self.image_stack, -51, 0)
        self.image_poses = torch.roll(self.image_poses, -51, 0)

        temp = torch.tensor(image_processing.bev_image(self.last_image)[200:444]/255).type(self.dtype).cuda().permute(2,0,1)
        curr_pose = self.curr_pose.clone()
        relative_change = torch.zeros((51, 6))
        relative_change[:, 0] = 3
        for each in range(51):
            self.image_stack[-each] = F.resize(temp[:, :, (13 + (each * 21)): ((183+13) + (each * 21))], size=(64,64))
            
            relative_change[-each, 1] = (each-25) * 0.1

        curr_pose = curr_pose.repeat(51, 1)
        self.image_poses[-51:] = self.util.to_world_torch(curr_pose, relative_change)[:, :2].cuda()
    
    def goal_cb(self, msg):
        print("Goal received")
        r,p, y = self.util.quaternion_to_rpy(msg.pose.orientation)

        goal_tensor_new = torch.Tensor([msg.pose.position.x,
                                    msg.pose.position.y,
                                    msg.pose.position.z,
                                    r, p, y]).type(self.dtype)
        self.goal_tensor = goal_tensor_new
        self.at_goal = False
        print("Current Pose: ", self.last_pose)
        print("SETTING Goal: ", self.goal_tensor.cpu().numpy())

    #update the immediate goal
    def set_goals(self):
        g_x = self.goals[0, 0] - self.gpsx_offset
        g_y = self.goals[0, 1] - self.gpsy_offset
        self.goal_tensor = torch.Tensor([g_x, g_y, 0.0,
                                        0.0, 0.0, 0.0]).type(self.dtype)
        print("Goal Set")

    #function to calculate costs for MPPI
    def cost(self, pose, goal, ctrl, noise,  t, bump_cost, slide_cost, roll_cost):
        self.fpose.copy_(pose)
        self.dist_to_goal.copy_(self.fpose).sub_(goal)
        self.dist_to_goal[:,5] = self.util.clamp_angle_tensor_(self.dist_to_goal[:,5])
        
        xy_to_goal = self.dist_to_goal[:,:2]
        self.euclidian_distance = torch.norm(xy_to_goal, p=2, dim=1)
        cost = self.euclidian_distance.cpu().numpy()
        euclidian_distance_squared = self.euclidian_distance.pow(2)

        theta_distance_to_goal = (self.dist_to_goal[:,5] % (2 * np.pi)) #**2

        st = noise + ctrl
        th = st[:,0].cpu().tolist()
        bp = bump_cost.squeeze().cpu().tolist()
        mx1 = np.argmax(bp)
        mx2 = np.argmin(bp)
        th1 = th[mx1]
        th2 = th[mx2]

        self.ctrl_cost.copy_(ctrl).mul_(self._lambda).mul_(self.inv_sigma).mul_(noise).mul_(0.5)
        running_cost_temp = self.ctrl_cost.abs_().sum(dim=1)
        self.running_cost.copy_(running_cost_temp)
        self.running_cost.add_(euclidian_distance_squared).add_(roll_cost.squeeze().cpu() *10).add_(bump_cost.squeeze().cpu() * 100).add_(roll_cost.squeeze().cpu() * 100)

    #get control input to send to the robot
    def get_control(self):
        # Apply the first control values, and shift your control trajectory
        run_ctrl = self.ctrl[0].clone()
        
        # shift all controls forward by 1, with last control replicated
        self.ctrl = torch.roll(self.ctrl, shifts=-1, dims=0)
        return run_ctrl

    #ackermann model implementation for generating the robot poses
    def ackermann_model(self, nn_input):
        """
        Calculates the change in pose (x, y, theta) for a batch of vehicles using the Ackermann steering model.
        
        Parameters:
        velocity (torch.Tensor): Tensor of shape (batch_size,) representing the velocity of each vehicle.
        steering (torch.Tensor): Tensor of shape (batch_size,) representing the steering angle of each vehicle.
        wheelbase (float): The distance between the front and rear axles of the vehicles.
        dt (float): Time step for the simulation.

        Returns:
        torch.Tensor: Tensor of shape (batch_size, 3) representing the change in pose (dx, dy, dtheta) for each vehicle.
        """
        # Ensure the velocity and steering tensors have the same batch size
        
        velocity = nn_input[:, 5] 
        steering = nn_input[:, 6]
        wheelbase = 0.56
        dt = 0.15 #hardcoded

        # Calculate the change in orientation (dtheta)
        dtheta = (velocity / wheelbase) * torch.tan(steering) * dt 

        # Calculate change in x and y coordinates
        dx = velocity * torch.cos(dtheta) * dt
        dy = velocity * torch.sin(dtheta) * dt

        # Stack the changes in x, y, and theta into one tensor
        pose_change = torch.stack((dx, dy, dtheta), dim=1)

        return pose_change

    #main mppi wrapper function
    def mppi(self, init_pose, init_inp, gps_vel):
        # init_pose (6, ) [x, y, z, r, p, y]
        
        self.times_mppi_called += 1        
        t0 = time.time()
        dt = self.dt

        self.running_cost.zero_()                                                  # Zero running cost
        pose = init_pose.repeat(self.K, 1).cuda()                                 # Repeat the init pose to sample size 
        nn_input = init_inp.repeat(self.K, 1).cuda()                               # Repeat the init input to sample size
        torch.normal(0, self.sigma, out=self.noise)                                # Generate noise based on the sigma

        # Loop the forward calculation till the horizon
        gps_vel_t = self.gps_velocity.unsqueeze(0).repeat(self.K, 1) 
        cmd_vel_rollouts_t = self.cmd_vel.clone()
        for t in range(self.T):

            nn_input[:, 5:7] = (self.ctrl[t] + self.noise[t])      # Add noise to previous control input
            nn_input[:,5].clamp_(self.min_vel, self.max_vel)               # Clamp control velocity
            nn_input[:,6].clamp_(self.min_del, self.max_del)               # Clamp control steering
            cmd_vel_rollouts_t[:, -2:] = nn_input[:, 5:7]

            o = self.ackermann_model(nn_input)
            
            pose = self.util.se2_transform_to_world(pose, o)

            self.poses[:,t, [0,1,5]] = pose

            self.images_for_poses = self.util.get_images_for_poses(self.poses[:, t, :], self.image_stack, self.image_poses, gps_vel, dt, t)

            gps_vel_t = (gps_vel_t - self.stats['velocity_mean']) / (self.stats['velocity_std'] + 0.000006)
            cmd_vel_rollouts_t = (cmd_vel_rollouts_t - self.stats['cmd_vel_mean'].cuda()) / (self.stats['cmd_vel_std'].cuda() + 0.000006)

            with torch.no_grad():
                roll_cost, slide_cost, bump_cost = self.cahsor_model(self.images_for_poses, gps_vel_t, cmd_vel_rollouts_t)

            roll_cost = (roll_cost * (self.stats['res_vel_omega_roll_slde_bump_std'][2].cuda() + 0.000006)) + self.stats['res_vel_omega_roll_slde_bump_mean'][2]
            slide_cost = (slide_cost * (self.stats['res_vel_omega_roll_slde_bump_std'][3].cuda() + 0.000006)) + self.stats['res_vel_omega_roll_slde_bump_mean'][3]
            bump_cost = (bump_cost * (self.stats['res_vel_omega_roll_slde_bump_std'][4].cuda() + 0.000006)) + self.stats['res_vel_omega_roll_slde_bump_mean'][4]

            cmd_vel_rollouts_t = torch.roll(cmd_vel_rollouts_t, -2, 1)
            self.cost(self.poses[:, t, :], self.goal_tensor, self.ctrl[t], self.noise[t], t, bump_cost, slide_cost, roll_cost)


        # MPPI weighing 
        self.running_cost -= torch.min(self.running_cost)
        self.running_cost /= -self._lambda
        torch.exp(self.running_cost, out=self.running_cost)
        weights = (self.running_cost / torch.sum(self.running_cost)).cuda()

        weights = weights.unsqueeze(1).expand(self.T, self.K, 2)
        weights_temp = weights.mul(self.noise)
        self.ctrl_change.copy_(weights_temp.sum(dim=1))
        self.ctrl += self.ctrl_change
        self.ctrl[:,0].clamp_(self.min_vel, self.max_vel)
        self.ctrl[:,1].clamp_(self.min_del, self.max_del)
        self.ctrl
        
        print("MPPI: %4.5f ms" % ((time.time()-t0)*1000.0))
        return self.poses

    def gps_cb(self, msg):
        new_nano = msg.nano
        if abs(new_nano - self.previous_nano) > 100:
            self.previous_nano = new_nano
            self.n_gps += 1
            
            gps_xy_raw = self.util.to_utm(msg.lat * 1e-7, msg.lon * 1e-7)
            if self.n_gps == 1:
                self.gpsx_offset = gps_xy_raw[0]
                self.gpsy_offset = gps_xy_raw[1]
                print("Offset set")
                self.set_goals()
                return
            
            gps_xy_raw[0] -= self.gpsx_offset
            gps_xy_raw[1] -= self.gpsy_offset

            if self.n_gps < 21:
                self.gps_pose_batch = np.roll(self.gps_pose_batch, -1, axis=0)
                self.gps_pose_batch[-1, 0] = gps_xy_raw[0]
                self.gps_pose_batch[-1, 1] = gps_xy_raw[1]
                return
            
            self.gps_pose_batch = np.roll(self.gps_pose_batch, -1, axis=0)
            self.gps_pose_batch[-1, 0] = gps_xy_raw[0]
            self.gps_pose_batch[-1, 1] = gps_xy_raw[1]
            gps_x = np.convolve(self.gps_pose_batch[:, 0], np.ones(3)/3, mode='valid')[-1]
            gps_y = np.convolve(self.gps_pose_batch[:, 1], np.ones(3)/3, mode='valid')[-1]
            yaw = self.util.convert_gps_to_xy_angles(np.array([msg.heading * 1e-5]))[0]

            if self.last_pose is None:
                self.last_pose = torch.Tensor([gps_x,
                                    gps_y,
                                    0, 0, 0, yaw]).type(self.dtype)
                self.last_iTOW = msg.iTOW *1e-3
                self.pvt_initialized = True
                return

            dt = self.last_iTOW - msg.iTOW *1e-3        

            self.curr_pose = torch.Tensor([gps_x,
                                    gps_y,
                                    0, 0, 0, yaw]).type(self.dtype)

            if dt == 0.0:
                return
            
            P_last = self.last_pose[[0,1,5]].numpy()
            P_curr = self.curr_pose[[0,1,5]].numpy()

            dx_dy_dw = self.util.non_batch_2d_transform_to_robot(P_last, P_curr)

            self.gps_velocity[0] = dx_dy_dw[0] / dt 
            self.gps_velocity[1] = dx_dy_dw[1] / dt 

            self.difference_from_goal = np.sqrt(((self.curr_pose.cpu())[0] - (self.goal_tensor.cpu())[0])**2 + ((self.curr_pose.cpu())[1] - (self.goal_tensor.cpu())[1])**2)
            
            self.update_image()
        
            if self.difference_from_goal < 2.3:
                self.min_vel = -0.25 
                self.max_vel = 3.0
            else:
                self.min_vel = MIN_VEL
                self.max_vel = MAX_VEL

            self.last_pose = self.curr_pose

    def mppi_cb(self, curr_pose, pose_dot):
        if curr_pose is None or self.goal_tensor is None or not self.pvt_initialized:
            return

        r, p, y = (self.curr_pose.cpu())[3], (self.curr_pose.cpu())[4], (self.curr_pose.cpu())[5]
        nn_input = torch.Tensor([0,0,0, np.sin(y), np.cos(y), 0.0, 0.0]).type(self.dtype)
        
        self.pose_pub.publish(self.util.particle_to_posestamped(curr_pose, 'odom'))
        self.goal_pub.publish(self.util.particle_to_posestamped(self.goal_tensor, 'odom'))

        #only calling the mppi function with x,y and w pose. 3DoF
        poses = self.mppi(curr_pose[[0, 1, 5]], nn_input, self.gps_velocity)

        run_ctrl = None
        if not self.cont_ctrl:
            run_ctrl = self.get_control().cpu().numpy()
            self.recent_controls[self.control_i] = run_ctrl
            self.control_i = (self.control_i + 1) % self.recent_controls.shape[0]
            pub_control = self.recent_controls.mean(0)
            self.speed = pub_control[0]
            self.steering_angle = pub_control[1]
        if self.viz:
            self.visualize(poses)
    
    # Publish some paths to RVIZ to visualize rollouts
    def visualize(self, poses):
        frame_id = 'odom'
        strt = int((self.K - self.num_viz_paths)/2)
        if self.path_pub.get_num_connections() > 0:
            for i in range(strt, strt + self.num_viz_paths):
                pa = Path()
                pa.header = self.util.make_header(frame_id)
                pa.poses = list(map(self.util.particle_to_posestamped, poses[i,:,:], [frame_id]*self.T))
                self.path_pub.publish(pa)
        if self.central_path_pub.get_num_connections() > 0:
            pa = Path()
            pa.header = self.util.make_header(frame_id)
            pa.poses = list(map(self.util.particle_to_posestamped, poses[int(self.K/2),:,:], [frame_id]*self.T))
            self.central_path_pub.publish(pa)

    #check if the robot is already at a goal or assigned waypoint
    def check_at_goal(self):
        if self.at_goal or self.last_pose is None:
            return

        XY_THRESHOLD = 1.0
        THETA_THRESHOLD = np.radians(180) 
        difference_from_goal = (self.last_pose - self.goal_tensor).abs()
        print(f"{difference_from_goal =}")
        xy_distance_to_goal = difference_from_goal[:2].norm()
        theta_distance_to_goal = difference_from_goal[5] % (2 * np.pi)
        if xy_distance_to_goal < XY_THRESHOLD:
            print(f'Waypoint achieved {self.goals[0]}')
            if self.goals.shape[0] == 1:
                print("All the waypoints achieved" )
                self.at_goal = True
                self.speed = 0
                self.steering_angle = 0
            else:
                self.goals = self.goals[1:,]
                self.set_goals()

    #send control to the robot
    def send_controls(self):
        if not self.at_goal:
            if self.cont_ctrl:
                run_ctrl = self.get_control()
                if self.prev_ctrl is None:
                    self.prev_ctrl = run_ctrl
                speed = self.prev_ctrl[0] * .5 + run_ctrl[0] * .5
                steer = self.prev_ctrl[1] * .5 + run_ctrl[1] * .5
                self.prev_ctrl = (speed, steer)
            else:
                speed = self.speed
                steer = self.steering_angle
        else:
            speed = 0
            steer = 0
        if speed != 0:
            print("Speed:", speed, "Steering:", steer)

        self.cmd_vel = self.cmd_vel.roll(-2, 1)
        self.cmd_vel[:, -2] = speed
        self.cmd_vel[:, -1] = self.steer_joy  

        self.ctrlmsg.angular.z = steer
        self.ctrlmsg.linear.x = speed 
        self.cmd_pub.publish(self.ctrlmsg)

        self.msgid += 1

if __name__ == '__main__':
 
  T = 8
  K = 255
  sigma = [0.15, 0.06] 
  _lambda = .12

  rospy.init_node("mppi_control", anonymous=True) # Initialize the node
  
  # run with ROS
  mp = mppi_planner(T, K, sigma, _lambda)
  rate = rospy.Rate(5)
  while not rospy.is_shutdown():
    mp.check_at_goal()
    mp.mppi_cb(mp.curr_pose, mp.pose_dot)
    mp.send_controls()
    rate.sleep()
