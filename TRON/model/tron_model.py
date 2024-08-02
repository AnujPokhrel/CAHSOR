import torch
from torch import nn
import torch.nn.functional as F


class VisionSpeedEncoder(nn.Module):
    def __init__(self, latent_size=64, l2_normalize=True, p=0.3):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1, bias=False), nn.Mish(), nn.BatchNorm2d(8),
            # output shape : (batch_size, 8, 64, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),  # output shape : (batch_size, 8, 32, 32),
        )

        self.skipblock = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1, bias=False), nn.Mish(), nn.BatchNorm2d(8),
            # output shape : (batch_size, 8, 32, 32),
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1, bias=False), nn.Mish(), nn.BatchNorm2d(8),
            # output shape : (batch_size, 8, 32, 32),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1, bias=False), nn.Mish(), nn.BatchNorm2d(16),
            # output shape : (batch_size, 16, 32, 32),
            nn.MaxPool2d(kernel_size=2, stride=2),  # output shape : (batch_size, 16, 16, 16),
        )

        self.skipblock2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False), nn.Mish(), nn.BatchNorm2d(16),
            # output shape : (batch_size, 16, 16, 16),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False), nn.Mish(), nn.BatchNorm2d(16),
            # output shape : (batch_size, 16, 16, 16),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False), nn.Mish(), nn.BatchNorm2d(32),
            # output shape : (batch_size, 32, 16, 16),
            nn.AvgPool2d(kernel_size=2, stride=2),  # output shape : (batch_size, 32, 8, 8),
        )

        self.skipblock3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False), nn.Mish(), nn.BatchNorm2d(32),
            # output shape : (batch_size, 32, 8, 8),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False), nn.Mish(), nn.BatchNorm2d(32),
            # output shape : (batch_size, 32, 8, 8),
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=3, padding=1, bias=False), nn.Mish(), nn.BatchNorm2d(64),
            # output shape : (batch_size, 64, 2, 2),
        )

        self.skipblock4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False), nn.Mish(), nn.BatchNorm2d(64),
            # output shape : (batch_size, 64, 2, 2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False), nn.Mish(), nn.BatchNorm2d(64),
            # output shape : (batch_size, 64, 2, 2),
        )

        self.patch_encoder = nn.Linear(256, latent_size)

        self.vel_encoder = nn.Sequential(  # input shape : (batch_size, 2)
            nn.Flatten(),
            nn.Linear(2, 128), nn.Mish(),
            nn.Dropout(p),
            nn.Linear(128, latent_size),
        )

        self.fc = nn.Sequential(
            nn.Linear(2 * latent_size, latent_size), nn.Mish(),
            nn.Linear(latent_size, latent_size)
        )

        self.l2_normalize = l2_normalize

    def forward(self, patch, vel):
        x = self.block1(patch)
        x = self.skipblock(x) + x
        x = self.block2(x)
        x = self.skipblock2(x) + x
        x = self.block3(x)
        x = self.skipblock3(x) + x
        x = self.block4(x)
        x = self.skipblock4(x) + x
        x = x.reshape(x.size(0), -1)  # flattened to (batch_size, 256)

        x = self.patch_encoder(x)

        vel = self.vel_encoder(vel)

        out = torch.cat([x, vel], dim=1)
        out = self.fc(out)

        # normalize
        if self.l2_normalize:
            out = F.normalize(out, dim=-1)

        return out


class IMUEncoder(nn.Module):
    def __init__(self, latent_size=64, p=0.2, l2_normalize=True):
        super(IMUEncoder, self).__init__()

        self.accel_encoder = nn.Sequential(  # input shape : (batch_size, 1, 603)
            nn.Flatten(),
            nn.Linear(201 * 3, 128), nn.Mish(),
            nn.Dropout(p),
            nn.Linear(128, latent_size // 2),
        )

        self.gyro_encoder = nn.Sequential(  # input shape : (batch_size, 1, 900)
            nn.Flatten(),
            nn.Linear(201 * 3, 128), nn.Mish(),
            nn.Dropout(p),
            nn.Linear(128, latent_size // 2),
        )

        self.fc = nn.Sequential(
            nn.Linear(2 * latent_size // 2, latent_size), nn.Mish(),
            nn.Linear(latent_size, latent_size)
        )

        self.l2_normalize = l2_normalize

    def forward(self, accel, gyro):
        accel = self.accel_encoder(accel)
        gyro = self.gyro_encoder(gyro)

        nonvis_features = self.fc(torch.cat([accel, gyro], dim=1))

        # normalize the features
        if self.l2_normalize: nonvis_features = F.normalize(nonvis_features, dim=-1)

        return nonvis_features


class TronModel(nn.Module):
    def __init__(self, visual_encoder, imu_encoder, projector):
        super().__init__()
        self.visual_encoder = visual_encoder
        self.imu_encoder = imu_encoder
        self.projector = projector

    def forward(self, patch1, patch2, acc, gyro, vel):
        v_encoded_1 = self.visual_encoder(patch1, vel)
        v_encoded_1 = F.normalize(v_encoded_1, dim=-1)
        v_encoded_2 = self.visual_encoder(patch2, vel)
        v_encoded_2 = F.normalize(v_encoded_2, dim=-1)

        i_encoded = self.imu_encoder(acc, gyro)

        zv1 = self.projector(v_encoded_1)
        zv2 = self.projector(v_encoded_2)
        zi = self.projector(i_encoded)

        return zv1, zv2, zi, v_encoded_1, v_encoded_2, i_encoded

    def barlow_loss(self, z1, z2):
        B = z1.shape[0]
        z1 = (z1 - z1.mean(dim=0)) / z1.std(dim=0)
        z2 = (z2 - z2.mean(dim=0)) / z2.std(dim=0)
        c = z1.T @ z2
        c.div_(B)
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + 0.0051 * off_diag

        return loss

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

