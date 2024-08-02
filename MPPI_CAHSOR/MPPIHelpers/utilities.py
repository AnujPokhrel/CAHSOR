import numpy as np
import math
import torch
from geometry_msgs.msg import PoseStamped, Quaternion
from std_msgs.msg import Header
import rospy
import utm
import pandas as pd
from scipy.spatial.transform import Rotation as R

#Class for general functions
class utils:
    def __init__(self, batch_size=255):
        self.queue_size = 0
        self.camera_parameter = torch.zeros((4,6))
        self.camera_parameter[:, :2] = torch.tensor([[3,1], [3,-1], [5,5], [5,-5]])
        #initialize all the zeros and other values
        self.batch_size = batch_size
        self.curr_z_angle = torch.zeros((batch_size, 3)).cuda()
        self.prev_z_angle = torch.zeros((batch_size, 3)).cuda()
        self.prev_pos_transform = torch.zeros(batch_size, 3, 3).cuda()
        self.robot_dims = torch.Tensor([[0.5, 0.33, 0],
                                        [0.5, -0.33, 0],
                                        [-0.5, -0.33, 0],
                                        [-0.5, 0.33, 0]]).cuda().repeat(batch_size, 1, 1).transpose(1,2)
        self.patchs_t = torch.zeros((batch_size, 64, 64, 3)).cuda()


        self.CENTER_t = torch.Tensor([635, 205]).unsqueeze(0).repeat(batch_size, 4, 1).cuda()
        self.image_dims = np.float32([[0,0], [255,0], [255, 255], [0, 255]])

    def rmap(self, value, from_min, from_max, to_min, to_max):
        # Calculate the range of the input value
        from_range = from_max - from_min

        # Calculate the range of the output value
        to_range = to_max - to_min

        # Scale the input value to the output range
        mapped_value = (value - from_min) * (to_range / from_range) + to_min

        return mapped_value

    # Convert quaternion to yaw angle (in radians)
    def quaternion_to_yaw(self, quaternion):
        quaternion_norm = math.sqrt(quaternion.x**2 + quaternion.y**2 + quaternion.z**2 + quaternion.w**2)
        if (quaternion_norm == 0):
            return 0.0
        quaternion.x /= quaternion_norm
        quaternion.y /= quaternion_norm
        quaternion.z /= quaternion_norm
        quaternion.w /= quaternion_norm

        yaw = math.atan2(2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
                         1.0 - 2.0 * (quaternion.y**2 + quaternion.z**2))

        return yaw
    
    # Convert quaternion to roll angle (in radians)
    def quaternion_to_roll(self, quaternion):
        quaternion_norm = math.sqrt(quaternion.x**2 + quaternion.y**2 + quaternion.z**2 + quaternion.w**2)
        if quaternion_norm == 0:
            return 0.0
        quaternion.x /= quaternion_norm
        quaternion.y /= quaternion_norm
        quaternion.z /= quaternion_norm
        quaternion.w /= quaternion_norm

        roll = math.atan2(2.0 * (quaternion.y * quaternion.z + quaternion.w * quaternion.x),
                        1.0 - 2.0 * (quaternion.x**2 + quaternion.y**2))

        return roll

    # Convert quaternion to pitch angle (in radians)
    def quaternion_to_pitch(self, quaternion):
        quaternion_norm = math.sqrt(quaternion.x**2 + quaternion.y**2 + quaternion.z**2 + quaternion.w**2)
        if quaternion_norm == 0:
            return 0.0
        quaternion.x /= quaternion_norm
        quaternion.y /= quaternion_norm
        quaternion.z /= quaternion_norm
        quaternion.w /= quaternion_norm

        pitch = math.asin(2.0 * (quaternion.w * quaternion.y - quaternion.z * quaternion.x))

        return pitch
    
    # Convert quaternion to roll, pitch, and yaw angles
    def quaternion_to_rpy(self, quaternion):
        qw = quaternion.w
        qx = quaternion.x
        qy = quaternion.y
        qz = quaternion.z
        

        # Roll (x-axis rotation)
        sinr_cosp = 2.0 * (qw * qx + qy * qz)
        cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2.0 * (qw * qy - qz * qx)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)  # Use 90 degrees if out of range
        else:
            pitch = math.asin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    # Convert yaw angle (in radians) to quaternion
    def yaw_to_quaternion(self, yaw):
        quaternion = Quaternion()
        quaternion.x = 0.0
        quaternion.y = 0.0
        quaternion.z = math.sin(yaw / 2.0)
        qw = quaternion.w
        qx = quaternion.x
        qy = quaternion.y
        qz = quaternion.z
        

        # Roll (x-axis rotation)
        sinr_cosp = 2.0 * (qw * qx + qy * qz)
        cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2.0 * (qw * qy - qz * qx)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)  # Use 90 degrees if out of range
        else:
            pitch = math.asin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    # Convert roll, pitch, and yaw angles to quaternion
    def rpy_to_quaternion(self, roll, pitch, yaw):
        quaternion = Quaternion()
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        quaternion.x = sr * cp * cy - cr * sp * sy
        quaternion.y = cr * sp * cy + sr * cp * sy
        quaternion.z = cr * cp * sy - sr * sp * cy
        quaternion.w = cr * cp * cy + sr * sp * sy

        return quaternion
    
    def clamp_angle(self, angles):
        angles += np.pi
        angles %= (2 * np.pi)
        angles -= np.pi
        return angles
    
    def clamp_angle_tensor_(self, angles):
        angles += np.pi
        torch.remainder(angles, 2*np.pi, out=angles)
        angles -= np.pi
        return angles

    def get_dist(self, start_pose, goal_pose):
        return math.sqrt((goal_pose.position.x - start_pose.position.x)**2 + (goal_pose.position.y - start_pose.position.y)**2)

    # Create a PoseStamped message from a Pose message
    def create_pose_stamped(self, pose):
        pose_stamped = PoseStamped()
        pose_stamped.pose = pose
        pose_stamped.header.stamp = rospy.Time.now()
        pose_stamped.header.frame_id = 'odom'  # Replace 'world' with your desired frame ID

        return pose_stamped

    def map_value(self, value, from_min, from_max, to_min, to_max):
        # Calculate the range of the input value
        from_range = from_max - from_min

        # Calculate the range of the output value
        to_range = to_max - to_min

        # Scale the input value to the output range
        mapped_value = (value - from_min) * (to_range / from_range) + to_min

        return mapped_value
    
    def make_header(self, frame_id, stamp=None):
        if stamp == None:
            stamp = rospy.Time.now()
        header = Header()
        header.stamp = stamp
        header.frame_id = frame_id
        return header

    def particle_to_posestamped(self, particle, frame_id):
        pose = PoseStamped()
        pose.header = self.make_header(frame_id)
        pose.pose.position.x = particle[0].item()
        pose.pose.position.y = particle[1].item()
        pose.pose.position.z = particle[2].item()
        pose.pose.orientation = self.rpy_to_quaternion(particle[3].item(), particle[4].item(), particle[5].item())
        return pose
 
    #convert lat and long msg from gps to utm co-ordinates
    def to_utm(self, lat, long):        
        return list(utm.from_latlon(lat, long))
    
    def to_utm_np_batch(self, lat_lon):
        utm_batch = []
        for each in lat_lon:
            list_data = list(utm.from_latlon(each[0], each[1]))
            utm_batch.append(list_data[0:2])
        
        return utm_batch 

    #convert xy co-ordinates to lat, long
    def to_latlong(self, northing, easting, zone, type):
        val_tuple = utm.to_latlon(northing, easting, zone, type)
        return [round(x,7) for x in val_tuple]   

    #takes gps waypoints and convertes them to a tensor of xy utm coordinates 
    def read_waypoints_to_xy(self, filename, dtype):
        df = pd.read_csv(filename)
        for index, each in enumerate(df.values):
            df.values[index] = self.to_utm(float(each[0]), float(each[1]))[0:2]
        
        return torch.tensor(df.values).type(dtype)

    #transform to robot frame in SE2 using torch    
    def se2_transform_to_robot(self, p1_batch, p2_batch):
        if p1_batch.shape != p2_batch.shape or p1_batch.shape[-1] != 3:
            raise ValueError("Both batches must be of the same shape and contain 3 elements per pose")

        # Extract components
        x1, y1, theta1 = p1_batch[:, 0], p1_batch[:, 1], p1_batch[:, 2]
        x2, y2, theta2 = p2_batch[:, 0], p2_batch[:, 1], p2_batch[:, 2]

        # Construct SE2 matrices
        zeros = torch.zeros_like(x1)
        ones = torch.ones_like(x1)
        T1 = torch.stack([torch.stack([torch.cos(theta1), -torch.sin(theta1), x1]),
                            torch.stack([torch.sin(theta1),  torch.cos(theta1), y1]),
                            torch.stack([zeros, zeros, ones])], dim=-1).permute(1,2,0)

        T2 = torch.stack([torch.stack([torch.cos(theta2), -torch.sin(theta2), x2]),
                            torch.stack([torch.sin(theta2),  torch.cos(theta2), y2]),
                            torch.stack([zeros, zeros, ones])], dim=-1).permute(1,2,0)

        # Inverse of T1 and transformation
        T1_inv = torch.inverse(T1)
        tf2_mat = torch.matmul(T2, T1_inv)

        # Extract transformed positions and angles
        transform = torch.matmul(T1_inv, torch.cat((p2_batch[:,:2], ones.unsqueeze(-1)), dim=1).unsqueeze(2)).squeeze()

        #added t the shape condition for checking the rollouts
        if transform.shape[0] == 3:
            transform[2] = torch.atan2(tf2_mat[:, 1, 0], tf2_mat[:, 0, 0]) 
        else:
            transform[:, 2] = torch.atan2(tf2_mat[:, 1, 0], tf2_mat[:, 0, 0])
        
        return transform
    
    #changing to world frame in SE2 
    def se2_transform_to_world(self, p1_batch, p2_batch):
        # Extract components
        x1, y1, theta1 = p1_batch[:, 0], p1_batch[:, 1], p1_batch[:, 2]
        x2, y2, theta2 = p2_batch[:, 0], p2_batch[:, 1], p2_batch[:, 2]

        # Construct SE2 matrices
        zeros = torch.zeros_like(x1)
        ones = torch.ones_like(x1)
        T1 = torch.stack([torch.stack([torch.cos(theta1), -torch.sin(theta1), x1]),
                            torch.stack([torch.sin(theta1),  torch.cos(theta1), y1]),
                            torch.stack([zeros, zeros, ones])], dim=-1).permute(1,2,0)

        T2 = torch.stack([torch.stack([torch.cos(theta2), -torch.sin(theta2), x2]),
                            torch.stack([torch.sin(theta2),  torch.cos(theta2), y2]),
                            torch.stack([zeros, zeros, ones])], dim=-1).permute(1,2,0)

        # Inverse of T1 and transformation
        tf2_mat = torch.matmul(T2, T1)

        # Extract transformed positions and angles
        transform = torch.matmul(T1, torch.cat((p2_batch[:,:2], ones.unsqueeze(-1)), dim=1).unsqueeze(2)).squeeze()
        transform[:, 2] = torch.atan2(tf2_mat[:, 1, 0], tf2_mat[:, 0, 0])
        
        return transform
    
    def euler_to_rotation_matrix(self, euler_angles):
        """ Convert Euler angles to a rotation matrix """
        # Compute sin and cos for Euler angles
        cos = torch.cos(euler_angles)
        sin = torch.sin(euler_angles)
        zero = torch.zeros_like(euler_angles[:, 0])
        one = torch.ones_like(euler_angles[:, 0])
        # Constructing rotation matrices (assuming 'xyz' convention for Euler angles)
        R_x = torch.stack([one, zero, zero, zero, cos[:, 0], -sin[:, 0], zero, sin[:, 0], cos[:, 0]], dim=1).view(-1, 3, 3)
        R_y = torch.stack([cos[:, 1], zero, sin[:, 1], zero, one, zero, -sin[:, 1], zero, cos[:, 1]], dim=1).view(-1, 3, 3)
        R_z = torch.stack([cos[:, 2], -sin[:, 2], zero, sin[:, 2], cos[:, 2], zero, zero, zero, one], dim=1).view(-1, 3, 3)

        return torch.matmul(torch.matmul(R_z, R_y), R_x)
    
    def extract_euler_angles_from_se3_batch(self, tf3_matx):
        # Validate input shape
        if tf3_matx.shape[1:] != (4, 4):
            raise ValueError("Input tensor must have shape (batch, 4, 4)")

        # Extract rotation matrices
        rotation_matrices = tf3_matx[:, :3, :3]

        # Initialize tensor to hold Euler angles
        batch_size = tf3_matx.shape[0]
        euler_angles = torch.zeros((batch_size, 3), device=tf3_matx.device, dtype=tf3_matx.dtype)

        # Compute Euler angles
        euler_angles[:, 0] = torch.atan2(rotation_matrices[:, 2, 1], rotation_matrices[:, 2, 2])  # Roll
        euler_angles[:, 1] = torch.atan2(-rotation_matrices[:, 2, 0], torch.sqrt(rotation_matrices[:, 2, 1] ** 2 + rotation_matrices[:, 2, 2] ** 2))  # Pitch
        euler_angles[:, 2] = torch.atan2(rotation_matrices[:, 1, 0], rotation_matrices[:, 0, 0])  # Yaw

        return euler_angles
    
    #convert robot pose to the world pose in SE3
    def to_world_torch(self, Robot_frame, P_relative):
        """ Assemble a batch of SE3 homogeneous matrices from a batch of 6DOF poses """
        batch_size = Robot_frame.shape[0]
        ones = torch.ones_like(P_relative[:, 0])
        transform = torch.zeros_like(Robot_frame)
        T1 = torch.zeros((batch_size, 4, 4), device=Robot_frame.device, dtype=Robot_frame.dtype)
        T2 = torch.zeros((batch_size, 4, 4), device=P_relative.device, dtype=P_relative.dtype)

        R1 = self.euler_to_rotation_matrix(Robot_frame[:, 3:])
        R2 = self.euler_to_rotation_matrix(P_relative[:, 3:])
        
        T1[:, :3, :3] = R1
        T2[:, :3, :3] = R2
        T1[:, :3,  3] = Robot_frame[:, :3]
        T2[:, :3,  3] = P_relative[:, :3]
        T1[:,  3,  3] = 1
        T2[:,  3,  3] = 1 

        T_tf = torch.matmul(T2, T1)
        
        transform[:, :3] = torch.matmul(T1, torch.cat((P_relative[:,:3], ones.unsqueeze(-1)), dim=1).unsqueeze(2)).squeeze()[:, :3]
        transform[:, 3:] = self.extract_euler_angles_from_se3_batch(T_tf)
        return transform
    
    def to_robot_6_dof(self, p1_batch, p2_batch):
        # Ensure input dimensions are correct
        if p1_batch.shape != p2_batch.shape or p1_batch.shape[1] != 6:
            raise ValueError("Input batches must be of the same shape and contain 6 elements per pose")

        n = p1_batch.shape[0]
        transformations = np.zeros((n, 6))
        # Extract and reshape translation components
        t1 = p1_batch[:, :3].reshape(n, 3, 1)
        t2 = p2_batch[:, :3].reshape(n, 3, 1)
        R1 = R.from_euler('xyz', p1_batch[:, 3:6]).as_matrix()
        R2 = R.from_euler('xyz', p2_batch[:, 3:6]).as_matrix()
        id = np.array([[[0,0,0,1]]]).repeat(n, axis=0)
        T1 = np.concatenate([np.concatenate((R1, t1), axis=2), id], axis=1)
        T2 = np.concatenate([np.concatenate((R2, t2), axis=2), id], axis=1)
        # Process rotations
        for i in range(n):
            T1_inv = np.linalg.inv(T1[i])
            transformations[i, :3] = (T1_inv @ T2[i] @ np.array([0, 0, 0, 1]))[:3]
            t_inv = T2[i] @ T1_inv
            transformations[i, 3:6] = R.from_matrix(t_inv[:-1, :-1]).as_euler('xyz')
        return transformations

    #convert the gps anges which start from due north to start from due east 
    def convert_gps_to_xy_angles(self, angle_np):
        angle_np = 360 - angle_np
        angle_np = angle_np + 90
        angle_np[angle_np > 360] = angle_np[angle_np > 360] - 360
        angle_np[angle_np < 0] = angle_np[angle_np < 0] + 360
        return np.radians(angle_np)
    
    def euler_to_rotation_matrix(self, euler_angles):
        """ Convert Euler angles to a rotation matrix """
        # Compute sin and cos for Euler angles
        cos = torch.cos(euler_angles)
        sin = torch.sin(euler_angles)
        zero = torch.zeros_like(euler_angles[:, 0])
        one = torch.ones_like(euler_angles[:, 0])
        # Constructing rotation matrices (assuming 'xyz' convention for Euler angles)
        R_x = torch.stack([one, zero, zero, zero, cos[:, 0], -sin[:, 0], zero, sin[:, 0], cos[:, 0]], dim=1).view(-1, 3, 3)
        R_y = torch.stack([cos[:, 1], zero, sin[:, 1], zero, one, zero, -sin[:, 1], zero, cos[:, 1]], dim=1).view(-1, 3, 3)
        R_z = torch.stack([cos[:, 2], -sin[:, 2], zero, sin[:, 2], cos[:, 2], zero, zero, zero, one], dim=1).view(-1, 3, 3)

        return torch.matmul(torch.matmul(R_z, R_y), R_x)

    #transformation to the robot frame using numpy and no batches
    def non_batch_2d_transform_to_robot(self, p1, p2):
        tf = np.zeros(3)
        T1 = np.array([[np.cos(p1[2]), -np.sin(p1[2]), p1[0]],
                    [np.sin(p1[2]),  np.cos(p1[2]), p1[1]],
                    [0,               0,               1]])

        T2 = np.array([[np.cos(p2[2]), -np.sin(p2[2]), p2[0]],
                    [np.sin(p2[2]),  np.cos(p2[2]), p2[1]],
                    [0,               0,               1]])

        T1_inv = np.linalg.inv(T1)
        # print("x and y coordinates: ", T1.dot(T2.dot(np.array([0, 0, 1])))[:2] )
        tf2_mat = T2 @ T1_inv

        tf[:2] = T1_inv.dot(T2.dot(np.array([0, 0, 1])))[:2]
        tf[2] = np.arctan2(tf2_mat[1,0], tf2_mat[0,0])

        # print("theta: ", np.arctan2(tf2_mat[1,0], tf2_mat[0,0]))
        return tf

    #gets the poses,stack of images and the poses of those image
    #calcualtes th l2 norm between the poses and the poses of the images to get the nearest image
    def get_images_for_poses(self, poses, image_stack, image_pose):
        poses = poses[:, :2].unsqueeze(1)
        distances = torch.norm(poses - image_pose.unsqueeze(0), dim=2)
        
        min_distances, min_indices = torch.min(distances, dim=1)
        return image_stack[min_indices]
    
    #takes gps waypoints and convertes them to a tensor of xy utm coordinates 
    def read_waypoints_to_xy(self, filename, dtype):
        df = pd.read_csv(filename)
        for index, each in enumerate(df.values):
            df.values[index] = self.to_utm(float(each[0]), float(each[1]))[0:2]
        
        return torch.tensor(df.values).type(dtype)