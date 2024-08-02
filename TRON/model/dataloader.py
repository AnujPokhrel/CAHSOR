from pathlib import Path
import copy
import pickle

import albumentations as A
import numpy as np
import cv2
from scipy.signal import periodogram
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def merge(base_dict: dict, new_dict: dict):
    """Merges two dictionary together

    base_dict (dict): The base dictionary to be updated
    new_dict (dict): The new data to be added to the base dictionary
    """
    # assert base_dict is None, "Base dictionary cannot be None"
    assert (
        base_dict.keys() == new_dict.keys()
    ), "The two dictionaries must have the same keys"
    for key in base_dict.keys():
        if key == 'patches_found':
            continue
        base_dict[key].extend(new_dict[key])

    return base_dict


def imread(address: str):
    img = cv2.imread(address, cv2.IMREAD_COLOR)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return np.array(img)


class TronDataset(Dataset):
    def __init__(self, root: str, stats: str, resize: tuple[int, int]=(64, 64), frequency_rate: int = 200, seed: int=42):
        torch.manual_seed(seed)
        self.resize = resize
        self.frequency_rate = frequency_rate  # this is IMU frequency rate
        files = list(Path(root).glob("*.pkl"))
        self.data = dict()
        for file in files:
            with file.open("rb") as f:
                data = pickle.load(f)
            if bool(self.data):
                self.data = merge(self.data, data)
            else:
                self.data = data
        # load stats
        self.stats = None
        with open(stats, 'rb') as f:
            self.stats = pickle.load(f)

        # define a transformation
        self.transform =  A.Compose([
            A.Flip(always_apply=False, p=0.5),
            A.ShiftScaleRotate(always_apply=False, p=0.75, shift_limit_x=(-0.1, 0.1), shift_limit_y=(-0.1, 0.1),
                               scale_limit=(-0.1, 2.0), rotate_limit=(-21, 21), interpolation=0, border_mode=0,
                               value=(0, 0, 0), mask_value=None, rotate_method='largest_box'),
            A.Perspective(always_apply=False, p=0.5, scale=(0.025, 0.25), keep_size=1, pad_mode=0, pad_val=(0, 0, 0),
                          mask_pad_val=0, fit_output=0, interpolation=3),
        ])

    def __len__(self):
        return len(self.data['time_stamp'])

    def __getitem__(self, idx):
        img = imread(Path(self.data['patches_path'][idx]).resolve().as_posix().replace("png", "jpg"))#.permute(1, 2, 0).numpy()
        label = 0
        if 'concrete' in self.data['patches_path'][idx] or 'Nov4' in self.data['patches_path'][idx]:
            label = 0 # 'concrete'
        elif 'grass' in self.data['patches_path'][idx]:
            label = 1 # 'grass'
        elif 'rocks' in self.data['patches_path'][idx]:
            label = 2 # 'rocks'
        patch1 = self.transform(image=copy.deepcopy(img))['image']
        patch2 = self.transform(image=copy.deepcopy(img))['image']
        # normalize the image patches and cast to torch tensor
        patch1 = torch.tensor(np.asarray(patch1, dtype=np.float32) / 255.0).permute(2, 0, 1)
        patch2 = torch.tensor(np.asarray(patch2, dtype=np.float32) / 255.0).permute(2, 0, 1)

        patch1 = transforms.Resize(self.resize, antialias=True)(patch1)
        patch2 = transforms.Resize(self.resize, antialias=True)(patch2)

        velocity_msg = (torch.tensor(self.data['velocity_msg'][idx]).float() - self.stats['velocity_mean']) / (self.stats['velocity_std'] + 0.000006)

        accel_msg = torch.tensor(self.data['accel_msg'][idx])
        gyro_msg = torch.tensor(self.data['gyro_msg'][idx])

        accel_msg = accel_msg.view(2, self.frequency_rate, 3)
        gyro_msg = gyro_msg.view(2, self.frequency_rate, 3)

        accel_msg = (accel_msg - self.stats['accel_mean']) / (self.stats['accel_std'] + 0.000006)
        gyro_msg = (gyro_msg - self.stats['gyro_mean']) / (self.stats['gyro_std'] + 0.000006)

        accel_msg = periodogram(accel_msg.view(2 * self.frequency_rate, 3), fs=self.frequency_rate, axis=0)[1]
        gyro_msg = periodogram(gyro_msg.view(2 * self.frequency_rate, 3), fs=self.frequency_rate, axis=0)[1]

        return patch1, patch2, torch.from_numpy(accel_msg).float(), torch.from_numpy(gyro_msg).float(), velocity_msg.float(), label


class MovingRobotDataset(Dataset):
    def __init__(self, root: str, stats: str, frequency_rate: int = 200, resize: tuple[int, int]=(224, 224)):
        files = list(Path(root).glob("*.pkl"))
        self.resize = resize
        self.frequency_rate = frequency_rate
        self.samples = dict()
        for file in files:
            with file.open("rb") as f:
                data = pickle.load(f)
            if bool(self.samples):
                self.samples = merge(self.samples, data)
            else:
                self.samples = data

        # define a transformation
        self.transform = A.Compose([
            A.Flip(always_apply=False, p=0.5),
            A.ShiftScaleRotate(always_apply=False, p=0.75, shift_limit_x=(-0.1, 0.1), shift_limit_y=(-0.1, 0.1),
                               scale_limit=(-0.1, 2.0), rotate_limit=(-21, 21), interpolation=0, border_mode=0,
                               value=(0, 0, 0), mask_value=None, rotate_method='largest_box'),
            A.Perspective(always_apply=False, p=0.5, scale=(0.025, 0.25), keep_size=1, pad_mode=0,
                          pad_val=(0, 0, 0),
                          mask_pad_val=0, fit_output=0, interpolation=3),
            A.RandomBrightnessContrast(brightness_limit=1, contrast_limit=1, p=8.0)
            # A.CLAHE(always_apply=True),
            # A.ColorJitter(always_apply=True),
        ])
        # load stats
        self.stats = None
        with open(stats, 'rb') as f:
            self.stats = pickle.load(f)

    def __len__(self):
        return len(self.samples['velocity_msg'])

    def __getitem__(self, idx):
        patch = imread(self.samples['patches_path'][idx].replace("png", "jpg"))
        patch = self.transform(image=copy.deepcopy(patch))['image']
        # normalize the image patches and cast to torch tensor
        patch = torch.tensor(np.asarray(patch, dtype=np.float32) / 255.0).permute(2, 0, 1)

        patch = transforms.Resize(self.resize, antialias=True)(patch)

        velocity_msg = self.generate_tensor(self.samples['velocity_msg'][idx])
        cmd_vel_msg = self.generate_tensor(self.samples['cmd_vel_msg'][idx])
        accel_msg = self.generate_tensor(self.samples['accel_msg'][idx])
        gyro_msg = self.generate_tensor(self.samples['gyro_msg'][idx])
        res_vel_omega_roll_slde_bump = self.generate_tensor(self.samples['res_vel_omega_roll_slde_bump'][idx])
        accel_msg = accel_msg.view(2, self.frequency_rate, 3)
        gyro_msg = gyro_msg.view(2, self.frequency_rate, 3)


        velocity_msg = (velocity_msg - self.stats['velocity_mean']) / (self.stats['velocity_std'] + 0.000006)
        cmd_vel_msg = (cmd_vel_msg - self.stats['cmd_vel_mean']) / (self.stats['cmd_vel_std'] + 0.000006)
        accel_msg = (accel_msg - self.stats['accel_mean']) / (self.stats['accel_std'] + 0.000006)
        gyro_msg = (gyro_msg - self.stats['gyro_mean']) / (self.stats['gyro_std'] + 0.000006)
        res_vel_omega_roll_slde_bump = (res_vel_omega_roll_slde_bump - self.stats['res_vel_omega_roll_slde_bump_mean']) / (self.stats['res_vel_omega_roll_slde_bump_std'] + 0.000006)
        # applying PSD filter
        accel_msg = periodogram(accel_msg.view(2 * self.frequency_rate, 3), fs=self.frequency_rate, axis=0)[1]
        gyro_msg = periodogram(gyro_msg.view(2 * self.frequency_rate, 3), fs=self.frequency_rate, axis=0)[1]

        cmd_vel_msg = cmd_vel_msg[-6:]
        res_vel_omega_roll_slde_bump = res_vel_omega_roll_slde_bump[2:]

        return (
            patch,  # [3, 64, 64]
            velocity_msg,  # [2]
            cmd_vel_msg,  # [6]
            torch.from_numpy(accel_msg).float().flatten(),  # [603]
            torch.from_numpy(gyro_msg).float().flatten(),  # [603]
            res_vel_omega_roll_slde_bump  # [3]
        )

    def generate_tensor(self, data):
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data).float()
        elif isinstance(data, list):
            return torch.tensor(data).float()
        elif isinstance(data, tuple):
            return torch.tensor(data).float()
        elif isinstance(data, torch.Tensor):
            return data.float()


if __name__ == "__main__":
    pass