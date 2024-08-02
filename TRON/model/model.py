import torch
import torch.nn as nn


class DownstreamNetwork(nn.Module):
    def __init__(self, backbone=None, imu_encoder=None, rep_size: int = 64, modality: str = 'VSI', p=0.2):
        super().__init__()
        # image encoder
        self.backbone = backbone
        self.imu_encoder = imu_encoder  # this is required for CAHSOR
        # encode accelometer data 3 x 201
        if imu_encoder is None:
            self.accel_encoder = nn.Sequential(  # For end-to-end
                nn.Flatten(),
                nn.Linear(201 * 3, 128), nn.Mish(),
                nn.Dropout(p),
                nn.Linear(128, rep_size // 2),
            )
            self.gyro_encoder = nn.Sequential(  # for end-to-end
                nn.Flatten(),
                nn.Linear(201 * 3, 128), nn.Mish(),
                nn.Dropout(p),
                nn.Linear(128, rep_size // 2),
            )

        # encode command to robot (2D)
        self.cmd_encoder = nn.Sequential(
            nn.LayerNorm(6),
            nn.Linear(6, 128),
            nn.Mish(),
            nn.Linear(128, rep_size),
            nn.Dropout(0.1),
        )

        # output 5 values, linear vel, angular vel, bumpiness
        self.fc_slide = nn.Sequential(
            nn.LayerNorm(len(modality) * rep_size),  # output of prev layers = 4 * 64 + 512
            nn.Linear(len(modality) * rep_size, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1),
        )
        self.fc_bump = nn.Sequential(
            nn.LayerNorm(len(modality) * rep_size),  # output of prev layers = 4 * 64 + 512
            nn.Linear(len(modality) * rep_size, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1),
        )
        self.fc_roll = nn.Sequential(
            nn.LayerNorm(len(modality) * rep_size),  # output of prev layers = 4 * 64 + 512
            nn.Linear(len(modality) * rep_size, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        patch: torch.Tensor,
        velocity_msg: torch.Tensor,
        cmd_vel_msg: torch.Tensor,
        accel_msg: torch.Tensor=None,
        gyro_msg: torch.Tensor=None,
    ):
        # define z
        z = []
        img = self.backbone(patch, velocity_msg)
        z.append(img)
        if accel_msg is not None and gyro_msg is not None:
            if self.imu_encoder:
                imu = self.imu_encoder(accel_msg, gyro_msg)
                z.append(imu)
            else:
                accel = self.accel_encoder(accel_msg)
                gyro = self.gyro_encoder(gyro_msg)
                imu = torch.cat([accel, gyro], dim=-1)
                z.append(imu)
        cmd = self.cmd_encoder(cmd_vel_msg)
        z.append(cmd)
        # late fusion
        z = torch.cat(z, dim=-1)

        roll_hat = self.fc_roll(z)
        slide_hat = self.fc_slide(z)
        bump_hat = self.fc_bump(z)

        return roll_hat, slide_hat, bump_hat