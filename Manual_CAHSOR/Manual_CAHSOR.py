#!/usr/bin/env python3
import rospy
from ublox_msgs.msg import NavPVT
from geometry_msgs.msg import Twist
from Helpers.utilities import utils as uts
from  Helpers.deployment_data_calculation import DataCalculation
from sensor_msgs.msg import Imu as IMU, MagneticField, Joy
import torch
import numpy as np
from deployment_calling_model import ModelClass
from Helpers.find_current_patch import FindCurrentPatch as image_processor
from DataProcessingPipeline.Helpers.sensor_fusion import imu_processor
import gc
import torch

class HeadingCalculator:
    def __init__(self):
        gc.collect()

        #initialize helper files
        self.batch_size = 100
        self.image = image_processor()
        self.data_calc = DataCalculation(self.batch_size)
        self.uts = uts()
        self.gps_pose_batch = np.zeros((21, 2))
        self.last_cmd_vel_to_model = np.zeros(2)
        self.pure_gps_vel = 0.0 

        #imu initialization
        #vals to pass to model
        self.imu_processor = imu_processor()
        self.imu_counter = 0

        #gps vars initialization        
        self.pvt_initialized = 0
        self.previous_nano = 0
        self.gpsx_offset = 0.0
        self.gpsy_offset = 0.0
        self.n_gps = 0
        self.last_pose = np.array([0.0, 0.0, 0.0, 0.0])

        #twist and twist stamped initialization
        self.twist = Twist()
        self.model_speed = [0.0, 0.0, 0.0, 0.0] 
 
        #callbacks
        self.pvt_sub = rospy.Subscriber('/f9p_rover/navpvt', NavPVT, self.pvt_callback, queue_size=10)

        self.cmd_vel_sub = rospy.Subscriber('/joy', Joy, self.cmd_vel_callback, queue_size=10)
        self.mag_data_sub = rospy.Subscriber('/magnetometer', MagneticField, self.mag_callback, queue_size=200)
        self.imu_sub = rospy.Subscriber('/imu', IMU, self.imu_callback, queue_size=200)

        #publishers
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.model_output = rospy.Publisher('/imu_dash', IMU, queue_size=1)

        self.model_publisher = IMU() 

        #grab the new model weights
        self.model = ModelClass(self.batch_size)
        print("model loaded successfully")

       

    def cmd_vel_callback(self, data):
        self.data_calc.cmd_vel_msg = np.roll(self.data_calc.cmd_vel_msg, -2, axis=0)
        self.data_calc.cmd_vel_msg[-2] = data.axes[0] * 4.8
        self.data_calc.cmd_vel_msg[-1] = - data.axes[1] / 2


    def model_callback(self):
        if self.pvt_initialized and self.image.patch_underneath is not None:

            self.actual_cmd_vel, self.last_cmd_vel_to_model, combinatation_cmd_vel = self.data_calc.process_cmd_vel()
            self.actual_vel_msg, un_vel_msg = self.data_calc.process_vel_msg()

            self.actual_accl_msg = self.data_calc.process_accl_msg()
            self.actual_gyro_msg = self.data_calc.process_gyro_msg()
            self.image_patches = self.data_calc.process_image(self.image.patch_underneath)

            self.model_speed, stats = self.model.forward(self.image_patches, self.actual_accl_msg, self.actual_gyro_msg, self.actual_vel_msg, self.actual_cmd_vel, self.data_calc.cmd_vel_msg[-2:], combinatation_cmd_vel, un_vel_msg)

            self.model_data_publishing(stats, combinatation_cmd_vel, un_vel_msg)
    
    def model_data_publishing(self, stats, combinatation_cmd_vel, un_vel_msg): 
            
            if self.model_speed == None:
                self.twist.linear.x = 0 
                self.twist.angular.z = 0.0
            else:
                if abs(self.twist.linear.x) < abs(self.model_speed[0]):
                    self.twist.linear.x = self.twist.linear.x * 0.7 + 0.3 * self.model_speed[0]
                else:
                    self.twist.linear.x = self.twist.linear.x * 0.6 + 0.4 * self.model_speed[0]
                
                if abs(self.twist.angular.z) < abs(self.model_speed[1]):
                    self.twist.angular.z = self.twist.angular.z * 0.7 + 0.3 * self.model_speed[1]
                else:
                    self.twist.angular.z = self.twist.angular.z * 0.2 + 0.8 * self.model_speed[1]
                

            self.cmd_vel_pub.publish(self.twist)
            self.model_publisher.angular_velocity.x =  stats[4] * 20 
            self.model_publisher.angular_velocity.y = stats[5] * 20 
            self.model_publisher.linear_acceleration.x = stats[2] * 20 
            self.model_publisher.linear_acceleration.y = stats[3] * 20 
            self.model_publisher.linear_acceleration.z = stats[6]
            self.model_publisher.orientation.x = self.last_cmd_vel_to_model[0] - self.twist.linear.x 
            self.model_publisher.orientation.y = self.last_cmd_vel_to_model[1] - self.twist.angular.z  
            self.model_publisher.orientation.z = stats[0] * 5
            self.model_publisher.orientation.w = stats[1] * 5
            self.model_publisher.header.seq += 1
            self.model_publisher.header.stamp = rospy.Time.now()
            self.model_output.publish(self.model_publisher)


    def pvt_callback(self, data):
        new_nano = data.nano
        if abs(new_nano - self.previous_nano) > 100:
            self.previous_nano = new_nano
            self.n_gps += 1
            
            gps_xy_raw = self.uts.to_utm(data.lat * 1e-7, data.lon * 1e-7)
            if self.n_gps == 1:
                self.gpsx_offset = gps_xy_raw[0]
                self.gpsy_offset = gps_xy_raw[1]
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
            gps_x = np.convolve(self.gps_pose_batch[:, 0], np.ones(5)/5, mode='valid')[-1]
            gps_y = np.convolve(self.gps_pose_batch[:, 1], np.ones(5)/5, mode='valid')[-1]

            x_diff = gps_x - self.gps_pose_batch[-2, 0]
            y_diff = gps_y - self.gps_pose_batch[-2, 1]

            if self.n_gps == 22:
                self.last_pose = np.array([gps_x, gps_y, self.uts.convert_gps_to_xy_angles(np.array([data.heading * 1e-5]))[0], data.iTOW * 1e-3])
                self.pvt_initialized = True
                return
            
            curr_xyh = np.array([gps_x, gps_y, self.uts.convert_gps_to_xy_angles(np.array([data.heading * 1e-5]))[0], data.iTOW * 1e-3])
            dx_dy_dw = self.uts.non_batch_2d_transform_to_robot(self.last_pose[:3], curr_xyh[:3])

            dt = (curr_xyh[3] - self.last_pose[3] + 0.000000001)
            self.data_calc.velocity_msgs[0] =  dx_dy_dw[0] / dt 
            self.data_calc.velocity_msgs[1] =  dx_dy_dw[1] / dt
            change = self.gps_pose_batch[-2] - self.gps_pose_batch[-1]
            speed = np.sqrt(change[0]**2 + change[1]**2) / dt
            
            
            self.pure_gps_vel = data.gSpeed * 1e-3
            self.last_pose = curr_xyh
            
    def imu_callback(self, data):
        # this calculates the yaw from the imu data
        self.imu_processor.imu_update(data)
        self.imu_counter += 1
        if self.imu_counter <= 500:
            self.imu_processor.beta = 0.8
        else:
            self.imu_processor.beta = 0.05

        self.imu_processor.imu_update(data)

        self.data_calc.accel_msgs = np.roll(self.data_calc.accel_msgs, -1, axis=0)
        self.data_calc.gyro_msgs = np.roll(self.data_calc.gyro_msgs, -1, axis=0)
        self.data_calc.accel_msgs[-1 ,0] = data.linear_acceleration.x
        self.data_calc.accel_msgs[-1 ,1] = data.linear_acceleration.y
        self.data_calc.accel_msgs[-1 ,2] = data.linear_acceleration.z
        self.data_calc.gyro_msgs[-1, 0] = data.angular_velocity.x
        self.data_calc.gyro_msgs[-1, 1] = data.angular_velocity.y
        self.data_calc.gyro_msgs[-1, 2] = data.angular_velocity.z

    def mag_callback(self, data):
        self.imu_processor.mag_update(data)

    def cleanup_model(self):
        if self.model:
            del self.model
            self.model.to("cpu")
            torch.cuda.empty_cache()
            self.model = None
            print("PyTorch model cleaned up successfully")

if __name__ == '__main__':
    rospy.init_node('HeadingCalculator')
    try:
        HeadingCalculatorNode = HeadingCalculator()
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            HeadingCalculatorNode.model_callback()
            rate.sleep()
    
    except KeyboardInterrupt:
        rospy.loginfo('Shutting down')
        HeadingCalculator.cleanup_model()