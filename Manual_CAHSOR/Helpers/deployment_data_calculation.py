import numpy as np
import cv2
import pickle
from scipy.signal import periodogram 
import torch

class DataCalculation():
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.frequency_rate = 200
        self.cmd_vel_msg = np.zeros(20, dtype=np.float32)
        self.accel_msgs = np.zeros((400,3), dtype=np.float32)
        self.gyro_msgs= np.zeros((400,3), dtype=np.float32)
        self.velocity_msgs = np.zeros((2), dtype=np.float32)
        self.device = torch.device('cuda')

        with open("path_to_stats/stats.pkl", 'rb') as f:
            self.stats = pickle.load(f)  

    def process_cmd_vel(self):
        combination_cmd_vel = torch.tensor(DataCalculation.create_cmd_vel_combinations(self.cmd_vel_msg, self.batch_size), dtype=torch.float32, device=self.device)
        cmd_vel_t = torch.tensor(self.cmd_vel_msg, dtype=torch.float32, device=self.device).repeat(self.batch_size, 1)
        
        cmd_vel_t[:, -2:] = combination_cmd_vel
        cmd_vel_t = (cmd_vel_t - self.stats['cmd_vel_mean'].cuda()) / (self.stats['cmd_vel_std'].cuda() + 0.000006)

        return cmd_vel_t[:, -6:], self.cmd_vel_msg[-2:], combination_cmd_vel

    def process_vel_msg(self):
        vel_msg = (self.velocity_msgs - self.stats['velocity_mean'].numpy()) / (self.stats['velocity_std'].numpy() + 0.000006)
        vel_msg_t = torch.tensor(vel_msg, dtype=torch.float32, device=self.device).repeat(self.batch_size, 1)
        return vel_msg_t, self.velocity_msgs

    def process_accl_msg(self):
        accel_msg = torch.from_numpy(self.accel_msgs).float()
        accel_msg = accel_msg.view(2, self.frequency_rate, 3)
        accel_msg = (accel_msg - self.stats['accel_mean']) / (self.stats['accel_std'] + 0.000006)
        accel_msg = periodogram(accel_msg.view(2 * self.frequency_rate , 3), fs=self.frequency_rate, axis=0)[1]
        accel_t = torch.tensor(accel_msg, dtype=torch.float32, device=self.device).repeat(self.batch_size, 1, 1).flatten(1)
        return accel_t

    def process_gyro_msg(self):
        gyro_msg = torch.from_numpy(self.gyro_msgs).float()
        gyro_msg = gyro_msg.view(2, self.frequency_rate, 3)
        gyro_msg = (gyro_msg - self.stats['gyro_mean']) / (self.stats['gyro_std'] + 0.000006)
        gyro_msg = periodogram(gyro_msg.view(2 * self.frequency_rate , 3), fs=self.frequency_rate, axis=0)[1]
        gyro_t = torch.tensor(gyro_msg, dtype=torch.float32, device=self.device).repeat(self.batch_size, 1, 1).flatten(1)
        return gyro_t
    
    def process_image(self, image_patch):
        patch_underneath = cv2.resize(image_patch, (64, 64))
        patch_underneath = np.transpose(patch_underneath, (2,0,1))/255.0        
        image_tensor = torch.tensor(patch_underneath, dtype=torch.float32).to(self.device).repeat(self.batch_size, 1, 1, 1)
        return image_tensor

    #create a combination of command velocities to pass to the cahsor model
    @staticmethod
    def create_cmd_vel_combinations(input_cmd_vel, batch_size):
        x_linspace = np.linspace(input_cmd_vel[-2]/4, input_cmd_vel[-2], num=int(batch_size/5))
        z_linspace = np.linspace(0, input_cmd_vel[-1], num=int((batch_size*5)/batch_size))

        comb_array = np.array(np.meshgrid(x_linspace, z_linspace)).T.reshape(-1,2)
        return comb_array