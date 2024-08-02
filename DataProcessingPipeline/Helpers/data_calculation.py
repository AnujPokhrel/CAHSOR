import numpy as np
from scipy.signal import savgol_filter
import math
import os
import sys
sys.path.append('/home/robotixx_user/CAHSOR')
from MPPI_CAHSOR.MPPIHelpers.utilities import utils as transformation_util

def process_rpy_msg(msg_data):
    rpy_data = np.array(msg_data['roll_pitch_yaw']).reshape(len(msg_data['roll_pitch_yaw']), 200, 3)[:,-20:,:]
    x = np.clip(rpy_data[:, :, 0], -10, 10)
    y = np.clip(rpy_data[:, :, 1], -10, 10)
    z = np.clip(rpy_data[:, :, 2], -10, 10)

    x = np.median(np.convolve(x.flatten(), np.ones(7)/7, mode='same').reshape(x.shape[0], 20), axis=1)
    y = np.median(np.convolve(y.flatten(), np.ones(7)/7, mode='same').reshape(y.shape[0], 20), axis=1)
    z = np.median(np.convolve(z.flatten(), np.ones(7)/7, mode='same').reshape(z.shape[0], 20), axis=1)
    dx = np.zeros(x.shape[0])
    dy = np.zeros(y.shape[0])
    dz = np.zeros(z.shape[0])
    dx[1:] = np.clip(np.convolve(np.diff(x, axis=0), np.ones(7)/7, mode='same') * 18 ,-1,1) 
    dy[1:] = np.clip(np.convolve(np.diff(y, axis=0), np.ones(7)/7, mode='same') * 26,-1,1)
    dz[1:] = np.clip(np.convolve(np.diff(z, axis=0), np.ones(7)/7, mode='same') * 10 ,-1,1)
    dz[dz>0.1] = 0.02
    dz[dz<-0.1] = -0.02
    x = np.clip(x,-10,10) * 3 
    y = np.clip(y,-10,10) * 4
    z = np.clip(z,-10,10) * 0.3

    return np.array([x, y, z, dx, dy, dz]).T

def process_cmd_vel(msg_data):
    cmd_vel = np.array(msg_data['cmd_vel_msg']).reshape(len(msg_data['cmd_vel_msg']), 10, 2)
    linear_vel = cmd_vel[:, :, 0]
    angular_vel = cmd_vel[:, :, 1]
    linear_vel = np.median(np.convolve(linear_vel.flatten(), np.ones(11)/11, mode='same').reshape(len(msg_data['cmd_vel_msg']), 10), axis=1) / 4.8
    angular_vel = np.median(np.convolve(angular_vel.flatten(), np.ones(11)/11, mode='same').reshape(len(msg_data['cmd_vel_msg']), 10), axis=1) * 2
    return np.array([linear_vel, angular_vel]).T

def process_vel_msg(msg_data):
    vel_msg = np.array(msg_data['velocity_msg'])
    sav_x = savgol_filter(vel_msg[:, 0], 11, 3, axis=0)
    sav_y = savgol_filter(vel_msg[:, 1], 11, 3, axis=0)

    sav_diff_x = np.zeros(sav_x.shape)
    sav_diff_y = np.zeros(sav_y.shape)

    sav_diff_x[1:] = np.diff(sav_x, axis=0) * 0.9
    sav_diff_y[1:] = np.diff(sav_y, axis=0) * 0.9

    sav_x = sav_x * 0.227
    sav_y = sav_y * 0.227
    return np.array([sav_x, sav_y, sav_diff_x, sav_diff_y]).T


def process_accl_msg(msg_data):
    acc_data = np.array(msg_data['accel_msg']).reshape(len(msg_data['accel_msg']), 400, 3)[:,-20:,:]
    x = np.clip(acc_data[:, :, 0], -10, 10)
    y = np.clip(acc_data[:, :, 1], -10, 10)
    z = np.clip(acc_data[:, :, 2]-10.1, -10, 10)

    # x1 = np.median(np.convolve(x, np.ones(7)/7, mode='same').reshape(len(data['accel_msg']), 20), axis=1)
    x = np.median(np.convolve(x.flatten(), np.ones(7)/7, mode='same').reshape(x.shape[0], 20), axis=1)
    y = np.median(np.convolve(y.flatten(), np.ones(7)/7, mode='same').reshape(y.shape[0], 20), axis=1)
    z = np.median(np.convolve(z.flatten(), np.ones(7)/7, mode='same').reshape(z.shape[0], 20), axis=1)
    dx = np.zeros(x.shape[0])
    dy = np.zeros(y.shape[0])
    dz = np.zeros(z.shape[0])
    dx[1:] = np.convolve(np.diff(x, axis=0), np.ones(7)/7, mode='same') / 2
    dy[1:] = np.convolve(np.diff(y, axis=0), np.ones(7)/7, mode='same') / 2
    dz[1:] = np.convolve(np.diff(z, axis=0), np.ones(7)/7, mode='same') / 1.2
    x = np.clip(x,-10,10) / 10
    y = np.clip(y, -10,10) / 9
    z = np.clip(z, -10,10) / 7
    #here we go
    return np.array([x, y, z, dx, dy, dz]).T

def process_gyro_msg(msg_data):
    gyro_data = np.array(msg_data['gyro_msg']).reshape(len(msg_data['gyro_msg']), 400, 3)[:,-20:,:]
    x = np.clip(gyro_data[:, :, 0], -10, 10)
    y = np.clip(gyro_data[:, :, 1], -10, 10)
    z = np.clip(gyro_data[:, :, 2], -10, 10)
    #here we go
    # x1 = np.median(np.convolve(x, np.ones(7)/7, mode='same').reshape(len(data['accel_msg']), 20), axis=1)
    x = np.median(np.convolve(x.flatten(), np.ones(7)/7, mode='same').reshape(x.shape[0], 20), axis=1)
    y = np.median(np.convolve(y.flatten(), np.ones(7)/7, mode='same').reshape(y.shape[0], 20), axis=1)
    z = np.median(np.convolve(z.flatten(), np.ones(7)/7, mode='same').reshape(z.shape[0], 20), axis=1)
    dx = np.zeros(x.shape[0])
    dy = np.zeros(y.shape[0])
    dz = np.zeros(z.shape[0])
    dx[1:] = np.clip(np.convolve(np.diff(x, axis=0), np.ones(7)/7, mode='same') * 3.8,-1,1)
    dy[1:] = np.clip(np.convolve(np.diff(y, axis=0), np.ones(7)/7, mode='same') * 2.2,-1,1)
    dz[1:] = np.clip(np.convolve(np.diff(z, axis=0), np.ones(7)/7, mode='same') * 2,-1,1)
    x = np.clip(x,-10,10) / 2 
    y = np.clip(y,-10,10) / 2
    z = np.clip(z,-10,10) / 2.4


    return np.array([x, y, z, dx, dy, dz]).T

def process_resultant_vel(msg_data):
    '''The function takes in the msg_data and returns the resultant velocity, omega, roll and sliding velocity
        the results are taken 0.4 second from the current time stamp,
    '''
    msg_data['res_vel_omega_roll_slide'] = []
    msg_data['triplets'] = []
    bumpiness_data = calculate_bumpiness(msg_data)
    roll_data = calculate_roll(msg_data)
    slide_data, poses = process_transformation_vel_msg(msg_data)
    slide_data = slide_data[:, 1]
    for i in range(len(msg_data['velocity_msg'])):
        triplet_this_round = [0, 0, 0]
        #find vel from time 0.5 sec after from the msg_data
        time_now = msg_data['time_stamp'][i]
        time_after = time_now + 0.4
        #find the index in the msg_data with the time_after
        index = np.where(np.array(msg_data['time_stamp']) >= time_after)[0]
        #index returns a list of indices, so we take the first one and go one before that first one
        if len(index) > 0:
            index = index[0] - 1
            gyro_mean = np.mean(np.reshape(msg_data['gyro_msg'][index][-150:], (50,3)), axis = 0)
            velocity_mean = np.mean(np.reshape(msg_data['velocity_msg'][index], (5,2)), axis = 0)

            roll = roll_data[index]
            bumpiness = bumpiness_data[index] 
            yaw = gyro_mean[2]

            # slide = 0.0 # velocity_mean[1]
            slide = slide_data[index] 
            g_speed = math.sqrt(velocity_mean[0]**2 + velocity_mean[1]**2)
            msg_data['res_vel_omega_roll_slide'].append([g_speed, yaw, roll, slide, bumpiness])
            if roll > 0.2:
                triplet_this_round[0] = 1
            if slide > 0.2:
                triplet_this_round[1] = 1
            if bumpiness > 0.2:
                triplet_this_round[2] = 1
            msg_data['triplets'].append(triplet_this_round)
            

    return msg_data['res_vel_omega_roll_slide'], msg_data['triplets']

def calculate_roll(data):
    vel_msg = process_vel_msg(data)[:, 2:4]
    gyro_msg = np.abs(process_gyro_msg(data)[:, 3:5])
    vel_msg = np.max(np.abs(vel_msg), axis=1)
    gyro_msg = np.max(np.abs(gyro_msg), axis=1)
    vel_msg_mask = process_vel_msg(data)[:, 0:2]
    vel_msg_mask = np.linalg.norm(vel_msg_mask, axis=1)
    gyro_msg_mask = gyro_msg
    gyro_msg_mask[vel_msg_mask < 0.7] = 0.2

    res = np.multiply(gyro_msg_mask, vel_msg)
    res = np.clip(res, 0, 0.5) * 2
    return res

def calculate_bumpiness(data):
    accl_z = abs(process_accl_msg(data)[:, 5])
    gyro_xy = process_gyro_msg(data)[:, 0:2]

    gyro_prod = np.abs(np.max(gyro_xy, axis=1))
    gyro_filtered = np.convolve(gyro_prod, np.ones(7)/7, mode='same')
    accl_filtered = np.convolve(accl_z, np.ones(7)/7, mode='same')

    return np.clip(np.multiply(gyro_filtered, accl_filtered) * 6.5, 0, 1)

def process_odom_vel_data(data):
    odoms = []
    for i in range(len(data['odom_msg'])-1):
        odom_now = data['odom_msg'][i]
        odom_now = np.array([odom_now.twist.twist.linear.x, odom_now.twist.twist.linear.y, odom_now.twist.twist.angular.z])
        if i>len(data['odom_msg'])-6:
            odom_next = data['odom_msg'][i+1]
        else:
            odom_next = data['odom_msg'][i+5] # assuming a delay of 0.2s
        odom_next = np.array([odom_next.twist.twist.linear.x, odom_next.twist.twist.linear.y, odom_next.twist.twist.angular.z])
        odoms.append(np.hstack((odom_now, odom_next)))
    return odoms

#process all the msgs to find velocity in robot frame
def process_transformation_vel_msg(msg_data):
    #convert array to numpy first
    t_util = transformation_util()
    gps_np = np.array(msg_data['lat_lon_heading_msg'])
    poses_np = np.zeros((len(msg_data['lat_lon_heading_msg']), 6))

    #get the utm version of the lat lon coordinates
    poses_np [:, :2] = np.array(t_util.to_utm_np_batch(msg_data['lat_lon_heading_msg']))                               #convert the first two lat and long to utms
    poses_np [:, 5] = t_util.convert_gps_to_xy_angles(gps_np[:, 2])


    #subtract the first value from the rest of the values
    poses_np[:, 0] = poses_np[:, 0] - poses_np[0, 0] 
    poses_np[:, 1] = poses_np[:, 1] - poses_np[0, 1]

    #apply convolution to smooth the data
    x_vals = np.convolve(poses_np[:, 0], np.ones(11)/11, mode='same')
    y_vals = np.convolve(poses_np[:, 1], np.ones(11)/11, mode='same')
    # heading = np.convolve(poses_np[:, 2], np.ones(3)/3, mode='same')

    #set the xy vals back to the poses_np
    poses_np[:, 0] = x_vals
    poses_np[:, 1] = y_vals
    # poses_np[:, 2] = heading

    #generate the transformations back
    # out_poses = t_util.se2_transforms_np_to_robot(poses_np[:-1, :3], poses_np[1:, :3])
    out_poses = t_util.to_robot_6_dof(poses_np[:-1], poses_np[1:])

    dts = gps_np[1:, 3] - gps_np[:-1, 3]
    # dts = 0.1
    # pdb.set_trace()
    out_poses[:, 0] = out_poses[:, 0] / dts
    out_poses[:, 1] = out_poses[:, 1] / dts

    return out_poses[:, :2], poses_np[:, :2]

