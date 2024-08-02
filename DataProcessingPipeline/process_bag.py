#!/usr/bin/env python3
import os.path
import copy
from logging import root
import pickle
import numpy as np
import cv2
from termcolor import cprint
import Helpers.image_processing as image_processing
import argparse
from Helpers.sensor_fusion import imu_processor
import rosbag
import threading
import Helpers.data_calculation as data_calculation

class ListenRecordData:
    def __init__(self, bag):
        self.data = [] 

        self.odom_msgs = np.zeros((200, 3), dtype=np.float32)
        self.gyro_msgs = np.zeros((400, 3), dtype=np.float32)
        self.accel_msgs = np.zeros((400, 3), dtype=np.float32)        

        self.velocity_msgs = np.zeros((5, 2), dtype=np.float32)
        self.cmd_vel_history = np.zeros((10,2), dtype=np.float32)
        self.roll_pitch_yaw = np.zeros((400, 3), dtype=np.float32)
        self.cmd_vel = None
        self.image = None
        self.odom = None
        self.counter = 0
        self.imu_processor = imu_processor()
        bag = rosbag.Bag(bag)
        self.imu_dash_msg = None
        self.joy_msg = None
        
        self.imu_counter = 0
        self.previous_nano = 0
        self.hunter_msg = [0.0, 0.0, 0.0, 0.0]

        #gps_stuff
        self.previous_itow = 0
        self.previous_pose = np.zeros(3)

        velocity = '/f9p_rover/navpvt'
        image = '/rgb_publisher/color/image/compressed'
        cmd_vel = '/stamped_cmd_vel' 
        imu = '/imu'
        odom = '/odom'
        mag = '/magnetometer'
        hunter_msg = 'hunter_status'
        joy = '/joy'
        imu_dash = '/imu_dash'

        all_topics = [velocity, image, cmd_vel, imu, odom, mag, hunter_msg, joy, imu_dash]
        self.msg_data = {
            'image_msg': [],
            'odom_msg': [],
            'odom_1sec_msg': [],
            'velocity_msg': [],
            'just_velocity_msg': [],
            'cmd_vel_msg': [],
            'time_stamp': [],
            'accel_msg': [],
            'roll_pitch_yaw': [],
            'gyro_msg': [],
            'hunter_msg': [],
            'lat_lon_heading_msg': [],
            'imu_dash_stats': [],
            'joy': []
            }

        self.callback_bag(bag, velocity, image, cmd_vel, imu, odom, mag, hunter_msg, joy, imu_dash,  all_topics)

    def callback_bag(self, bag, velocity, image, cmd_vel, imu, odom, mag, hunter_msg, joy, imu_dash, all_topics):
        for topic, msg, t in bag.read_messages(topics=all_topics):
            if topic == cmd_vel:
                self.cmd_vel_callback(msg)
            elif topic == odom:
                self.odom_callback(msg)
            elif topic == imu:
                self.imu_callback(msg)
            elif topic == image:
                self.image_callback(msg)
            elif topic == mag:
                self.mag_callback(msg)
            elif topic == hunter_msg:
                self.hunter_callback(msg)
            elif topic == velocity:
                self.callback(msg)
            elif topic == joy:
                self.joy_callback(msg)
            elif topic == imu_dash:
                self.imu_dash_callback(msg)
    
    def cmd_vel_callback(self, msg):
        self.cmd_vel_history = np.roll(self.cmd_vel_history, -1, axis=0)
        self.cmd_vel_history[-1] = np.array([msg.twist.linear.x, msg.twist.angular.z])
        self.cmd_vel = msg

    def image_callback(self, msg):
        self.image = msg
    
    def hunter_callback(self, msg):
        self.hunter_msg = [msg.linear_velocity, msg.steering_angle, msg.motor_states[1]['rpm'], msg.motor_states[2]['rpm']]

    def callback(self, velocity):
        new_nano = velocity.nano
        if abs(new_nano - self.previous_nano) > 100 and self.hunter_msg[0] >= 0.0:
            self.previous_nano = new_nano
            print('Received messages :: ', self.counter, ' ___')
            self.velocity_msgs = np.roll(self.velocity_msgs, -1, axis=0)
            self.velocity_msgs[-1] = np.array([velocity.velN * 1e-3, velocity.velE * 1e-3])
            cmd_vel = self.cmd_vel_history
            odom_msg = self.odom_msgs
            if self.cmd_vel is not None and self.image is not None and self.odom is not None:
                self.msg_data['image_msg'].append(self.image)
                self.msg_data['odom_msg'].append(self.odom)
                self.msg_data['odom_1sec_msg'].append(self.odom_msgs.flatten())
                self.msg_data['accel_msg'].append(self.accel_msgs.flatten())
                self.msg_data['gyro_msg'].append(self.gyro_msgs.flatten())
                self.msg_data['roll_pitch_yaw'].append(self.roll_pitch_yaw.flatten())
                self.msg_data['velocity_msg'].append(self.velocity_msgs.flatten())
                self.msg_data['just_velocity_msg'].append([velocity.velN * 1e-3, velocity.velE * 1e-3])
                self.msg_data['time_stamp'].append(self.cmd_vel.header.stamp.to_sec())
                self.msg_data['cmd_vel_msg'].append(cmd_vel.flatten())  #[cmd_vel.twist.linear.x, cmd_vel.twist.angular.z])
                self.msg_data['hunter_msg'].append(self.hunter_msg)
                self.msg_data['lat_lon_heading_msg'].append([velocity.lat * 1e-7, velocity.lon * 1e-7, velocity.heading * 1e-5, velocity.iTOW * 1e-3])
                self.msg_data['imu_dash_stats'].append(self.imu_dash_msg)
                self.msg_data['joy'].append(self.joy_msg)
                self.counter += 1

    def imu_dash_callback(self, msg):
        self.imu_dash_msg = msg

    def joy_callback(self, msg):
        self.joy_msg = msg

    def odom_callback(self, msg):
        self.odom = msg
        self.odom_msgs = np.roll(self.odom_msgs, -1, axis=0)
        msg = msg.twist.twist
        self.odom_msgs[-1] = np.array([msg.linear.x, msg.linear.y, msg.angular.z])
    
    def imu_callback(self, msg):
        if self.imu_counter <= 600:
            self.imu_processor.beta = 0.8  
            self.imu_counter += 1
        else:
            self.imu_processor.beta = 0.05 
            
        self.imu_processor.imu_update(msg)
        self.roll_values = self.imu_processor.roll
        self.gyro_msgs = np.roll(self.gyro_msgs, -1, axis=0)
        self.accel_msgs = np.roll(self.accel_msgs, -1, axis=0)
        self.gyro_msgs[-1] = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
        self.accel_msgs[-1] = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
        
        self.roll_pitch_yaw = np.roll(self.roll_pitch_yaw, -1, axis=0)
        self.roll_pitch_yaw[-1] = np.radians(np.array([self.imu_processor.roll, self.imu_processor.pitch, self.imu_processor.heading]))
    
    def mag_callback(self, msg):
        self.imu_processor.mag_update(msg)
        
    def save_data(self, msg_data, pickle_file_name, data_folder, just_the_name):
        data = {}

        # process front camera image
        print('Processing front camera image')
        data['res_vel_omega_roll_slde_bump'], data['triplets'] = data_calculation.process_resultant_vel(self.msg_data)
        data_length = len(data['res_vel_omega_roll_slde_bump'])

        data['cmd_vel_msg'] = msg_data['cmd_vel_msg'][:data_length]        
        data['odom_1sec_msg'] = msg_data['odom_1sec_msg'][:data_length]
        data['odom'] = data_calculation.process_odom_vel_data(self.msg_data)
        data['velocity_msg'], data['poses'] = data_calculation.process_transformation_vel_msg(self.msg_data)
        data['velocity_msg'] = data['velocity_msg'][:data_length].tolist()
        data['poses'] = data['poses'][:data_length].tolist()
        data['accel_msg'] = msg_data['accel_msg'][:data_length]
        data['gyro_msg'] = msg_data['gyro_msg'][:data_length]
        data['time_stamp'] = msg_data['time_stamp'][:data_length]
        data['roll_pitch_yaw'] = msg_data['roll_pitch_yaw'][:data_length]
        # data['imu_dash_stats'] = msg_data['imu_dash_stats'][:data_length]
        # data['joy'] = msg_data['joy'][:data_length]
        
        # data['hunter_msgs'] = msg_data['hunter_msg'][:data_length]

        data['patches'], data['patches_found'] = image_processing.process_bev_image_and_patches(self.msg_data)

        diff_length = len(data['patches']) - data_length
        if diff_length > 0:
            for i in range(diff_length):
                data['patches'].popitem()
                data['patches_found'].popitem()
                
        del msg_data['image_msg']
        del msg_data['odom_msg']
        del data['odom_1sec_msg']
        del data['odom']
        assert(data_length) == len(data['patches'].keys())
        patches = []
        sorted_keys = sorted(data['patches'].keys())
        for i in range(len(sorted_keys)):
            patches.append(data['patches'][sorted_keys[i]])
        data['patches'] = patches
        cprint('data length: '+str(data_length), 'green', attrs=['bold'])
        assert(len(data['patches_found']) == data_length)
        assert(len(data['patches']) == data_length)
        assert(len(data['velocity_msg']) == data_length)
        assert(len(data['cmd_vel_msg']) == data_length)
        assert(len(data['accel_msg']) == data_length)
        assert(len(data['gyro_msg']) == data_length)
        assert(len(data['time_stamp']) == data_length)    
        assert(len(data['roll_pitch_yaw']) == data_length)
        # assert(len(data['hunter_msgs']) == data_length)
        assert(len(data['res_vel_omega_roll_slde_bump']) == data_length)
        if len(data['gyro_msg']) > 0:
            cprint('Saving data...{}'.format(len(data['cmd_vel_msg'])), 'yellow')
            cprint(f"the keys in data are {data.keys()}", 'red')

            path = '{}.pkl'.format(just_the_name) 
            data['patches_path'] = []
            black_image = np.zeros((256, 256, 3), dtype=np.uint8)
            for index, each in enumerate(data['patches']):
                
                #save each patch in a file under the images folder
                path_of_image = os.path.join(data_folder, 'images_' + just_the_name, '{}.jpg'.format(index))
                data['patches_path'].append(path_of_image)
                if each is not None:
                    cv2.imwrite(path_of_image, each[0])
                else:
                    cv2.imwrite(path_of_image, black_image)

            del data['patches']        
            pickle.dump(data, open(path, 'wb'))
            cprint('Saved data successfully ', 'yellow', attrs=['blink'])

        return True
  
def threading_function(each, folder, just_the_name):
    data_recorder = ListenRecordData(each)

    if len(data_recorder.msg_data['image_msg']) > 0:
        data_saver = data_recorder.save_data(copy.deepcopy(data_recorder.msg_data),
                             each, folder, just_the_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #insted of file input doing a folder input 

    parser.add_argument('-f', '--file', type=str, default='data/aa', help='file to save data to')
    parser.add_argument('-b', '--folder', type=str, default='data/aa', help='folder containing the rosbags to process')
    args = parser.parse_args()
    save_data_path = args.file
    pickle_file_names = []

    if not os.path.exists(args.folder):
        cprint(args.bag, 'red', attrs=['bold'])
        raise FileNotFoundError('ROS bag folder not found')
    else:
        list_of_bags = os.listdir(args.folder)

    threading_array = []
    for each in list_of_bags:
        just_the_name = each.split('.')[0]
        
        os.makedirs(args.folder + '/images_' + just_the_name, exist_ok=True) 
        each = os.path.join(args.folder, each)
        threading_array.append(threading.Thread(target = threading_function, args =(each, args.folder, just_the_name,)))
        threading_array[-1].start()
        print(f"each is : {each} thread: {threading_array[-1].name}")

    for each in threading_array:
        each.join()
    
    exit(0)
        

