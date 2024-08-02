import gc
from pathlib import Path
from datetime import datetime
import sys
import argparse
import json

try:
    sys.path.append(str(Path(".").resolve()))
except:
    raise RuntimeError("Can't append root directory of the project to the path")

from rich import print
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from model.model import DownstreamNetwork
from model.tron_model import VisionSpeedEncoder, IMUEncoder
from model.dataloader import MovingRobotDataset
from utils.nn import check_grad_norm, op_counter
from utils.io import save_checkpoint, load_checkpoint
from utils.helpers import get_conf, timeit


class Learner:
    def __init__(self, cfg_dir: str):
        # load config file and initialize the logger and the device
        self.cfg = get_conf(cfg_dir)
        self.cfg.directory.model_name = self.cfg.logger.experiment_name
        self.cfg.directory.model_name += f"-{self.cfg.model.rep_size}-{datetime.now():%m-%d-%H-%M}"
        self.cfg.logger.experiment_name = self.cfg.directory.model_name
        self.cfg.directory.save = str(
            Path(self.cfg.directory.save) / self.cfg.directory.model_name
        )
        torch.random.manual_seed(self.cfg.train_params.seed)
        torch.cuda.manual_seed(self.cfg.train_params.seed)
        torch.cuda.manual_seed_all(self.cfg.train_params.seed)
        self.device = self.init_device()
        # creating dataset interface and dataloader for trained data
        self.data, self.val_data = self.init_dataloader()
        # create model and initialize its weights and move them to the device
        self.model = self.init_model()
        # initialize the optimizer
        self.optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), **self.cfg.adamw)
        # print(f"Number of parameters: {sum([x.numel() for x in self.model.parameters() if x.requires_grad == True])}")
        # define loss function
        self.criterion = torch.nn.MSELoss()
        # if resuming, load the checkpoint
        self.if_resume()

    def train(self):
        """Trains the model"""
        epoch_loss = {
            "roll": [],
            "slide": [],
            "bump": [],
            "combined": []
        }

        while self.epoch <= self.cfg.train_params.epochs:
            running_loss = {
                "roll": [],
                "slide": [],
                "bump": [],
                "combined": []
            }

            bar = tqdm(
                self.data,
                desc=f"Epoch {self.epoch:03}/{self.cfg.train_params.epochs:03}, training: ",
                total=len(self.data),
            )
            for data in bar:
                self.iteration += 1
                (loss, loss_roll, loss_slide, loss_bump, grad_norm), t_train = self.forward_batch(data)
                t_train /= self.data.batch_size
                running_loss["combined"].append(loss)
                running_loss["roll"].append(loss_roll)
                running_loss["slide"].append(loss_slide)
                running_loss["bump"].append(loss_bump)

                bar.set_postfix(loss=loss, roll=loss_roll, slide=loss_slide, bump=loss_bump, Grad=grad_norm, Time=t_train)

            bar.close()


            # validate on val set
            # val_loss, t = self.validate()
            # t /= len(self.val_data.dataset)

            # average loss for an epoch
            self.e_loss.append(np.mean(running_loss['combined']))  # epoch loss
            epoch_loss['roll'].append(np.mean(running_loss['roll']))  # epoch loss
            epoch_loss['slide'].append(np.mean(running_loss['slide']))  # epoch loss
            epoch_loss['bump'].append(np.mean(running_loss['bump']))  # epoch loss
            epoch_loss['combined'].append(np.mean(running_loss['combined']))  # epoch loss

            print(
                f"{datetime.now():%Y-%m-%d %H:%M:%S} Epoch {self.epoch:03}, "
                + f"Iteration {self.iteration:05} summary: train Loss: "
                + f"[green]{self.e_loss[-1]:.2f}[/green] || "
                + f"roll Loss: [green]{np.mean(running_loss['roll']):.2f}[/green] || "
                + f"slide Loss: [green]{np.mean(running_loss['slide']):.2f}[/green] || "
                + f"bump Loss: [green]{np.mean(running_loss['bump']):.2f}[/green] \n"
            )

            if self.epoch % self.cfg.train_params.save_every == 0 or (
                self.e_loss[-1] < self.best
                and self.epoch >= self.cfg.train_params.start_saving_best
            ):
                self.save()

            gc.collect()
            self.epoch += 1

        data = next(iter(self.data))
        patch = data[0].to(device=self.device)
        vel = data[1].to(device=self.device)
        cmd = data[2].to(device=self.device)
        accel = data[3].to(device=self.device)
        gyro = data[4].to(device=self.device)
        accel = None # for the final model
        gyro = None # for the final model
        macs, params = op_counter(self.model, sample=(patch, vel, cmd, accel, gyro))
        print("macs = ", macs, " | params = ", params)
        with open(Path(self.cfg.directory.save) / "loss.json", "w") as f:
            json.dump(epoch_loss, f)
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - Training is DONE!")

    @timeit
    def forward_batch(self, data):
        """Forward pass of a batch"""
        self.model.train()
        # move data to device
        patch = data[0].to(device=self.device)
        vel = data[1].to(device=self.device)
        cmd = data[2].to(device=self.device)
        accel = data[3].to(device=self.device)
        gyro = data[4].to(device=self.device)
        res_vel_omega_roll_slde_bump = data[5].to(device=self.device)

        roll_hat, _, _ = self.model(patch, vel, cmd)
        _, slide_hat, _ = self.model(patch, vel, cmd)
        _, _, bump_hat = self.model(patch, vel, cmd)

        loss_roll = self.criterion(roll_hat, res_vel_omega_roll_slde_bump[:, 0:1])
        loss_slide = self.criterion(slide_hat, res_vel_omega_roll_slde_bump[:, 1:2])
        loss_bump = self.criterion(bump_hat, res_vel_omega_roll_slde_bump[:, 2:])

        loss = 0.4 * loss_roll + 0.3 * loss_slide + 0.4 * loss_bump

        self.optimizer.zero_grad()
        loss.backward()
        # update
        self.optimizer.step()
        # gradient clipping
        if self.cfg.train_params.grad_clipping > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.train_params.grad_clipping
            )
        # check grad norm for debugging
        grad_norm = check_grad_norm(self.model)
        loss = loss_roll + loss_slide + loss_bump

        return loss.detach().item(), loss_roll.detach().item(), loss_slide.detach().item(), loss_bump.detach().item(), grad_norm

    @timeit
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        running_loss = []
        bar = tqdm(
            self.val_data,
            desc=f"Epoch {self.epoch:03}/{self.cfg.train_params.epochs:03}, validating",
        )
        for data in bar:
            # move data to device
            patch = data[0].to(device=self.device)
            vel = data[1].to(device=self.device)
            cmd = data[2].to(device=self.device)
            accel = data[3].to(device=self.device)
            gyro = data[4].to(device=self.device)
            y = data[5].to(device=self.device)

            # forward

            roll_hat, _, _ = self.model(patch, vel, cmd)
            _, slide_hat, _ = self.model(patch, vel, cmd)
            _, _, bump_hat = self.model(patch, vel, cmd)

            loss_roll = self.criterion(roll_hat, y[:, 0:1])
            loss_slide = self.criterion(slide_hat, y[:, 1:2])
            loss_bump = self.criterion(bump_hat, y[:, 2:])

            loss = loss_roll + loss_slide + loss_bump

            running_loss.append(loss.item())
            bar.set_postfix(loss=loss.item(), roll=loss_roll.item(), slide=loss_slide.item(), bump=loss_bump.item())

        bar.close()
        # average loss
        loss = np.mean(running_loss)
        return loss

    def init_model(self):
        """Initializes the model"""
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - INITIALIZING the model!")
        load_dir = self.cfg.model.load_pretrain
        checkpoint = load_checkpoint(load_dir, self.device)
        # visual_encoder = models.get_model('resnet18', weights='DEFAULT')
        visual_encoder = VisionSpeedEncoder(latent_size=self.cfg.model.rep_size)
        visual_encoder.load_state_dict(checkpoint['visual_encoder'])
        # freezing the visual encoder
        for name, child in visual_encoder.named_parameters():
            child.requires_grad_(False)
        visual_encoder.eval()
        # initialize the pretrained IMU encoder
        imu_encoder = IMUEncoder(latent_size=self.cfg.model.rep_size)
        imu_encoder.load_state_dict(checkpoint['imu_encoder'])
        # freezing the imu encoder
        for name, child in imu_encoder.named_parameters():
            child.requires_grad_(False)
        imu_encoder.eval()

        model = DownstreamNetwork(visual_encoder, imu_encoder, rep_size=self.cfg.model.rep_size, modality=self.cfg.model.modality)

        if (
            "cuda" in str(self.device)
            and self.cfg.train_params.device.split(":")[1] == "a"
        ):
            model = torch.nn.DataParallel(model)

        model = model.to(device=self.device)
        return model

    def init_optimizer(self):
        """Initializes the optimizer and learning rate scheduler"""
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - INITIALIZING the optimizer!")
        if self.cfg.train_params.optimizer.lower() == "adamw":
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), **self.cfg.adamw)

        elif self.cfg.train_params.optimizer.lower() == "adam":
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), **self.cfg.adam)

        elif self.cfg.train_params.optimizer.lower() == "rmsprop":
            optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, self.model.parameters()), **self.cfg.rmsprop)

        elif self.cfg.train_params.optimizer.lower() == "sgd":
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), **self.cfg.sgd)

        else:
            raise ValueError(
                f"Unknown optimizer {self.cfg.train_params.optimizer}"
                + "; valid optimizers are 'adamw, 'adam', 'sgd' and 'rmsprop'."
            )

        # initialize the learning rate scheduler
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.cfg.train_params.epochs
        )
        return optimizer, lr_scheduler

    def init_device(self):
        """Initializes the device"""
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - INITIALIZING the device!")
        is_cuda_available = torch.cuda.is_available()
        device = self.cfg.train_params.device

        if "cpu" in device:
            print(f"Performing all the operations on CPU.")
            return torch.device(device)

        elif "cuda" in device:
            if is_cuda_available:
                device_idx = device.split(":")[1]
                if device_idx == "a":
                    print(
                        f"Performing all the operations on CUDA; {torch.cuda.device_count()} devices."
                    )
                    self.cfg.dataloader.batch_size *= torch.cuda.device_count()
                    return torch.device(device.split(":")[0])
                else:
                    print(f"Performing all the operations on CUDA device {device_idx}.")
                    return torch.device(device)
            else:
                print("CUDA device is not available, falling back to CPU!")
                return torch.device("cpu")
        else:
            raise ValueError(f"Unknown {device}!")

    def init_dataloader(self):
        """Initializes the dataloaders"""
        print(
            f"{datetime.now():%Y-%m-%d %H:%M:%S} - INITIALIZING the train and val dataloaders!"
        )
        dataset = MovingRobotDataset(**self.cfg.dataset)
        data = DataLoader(dataset, **self.cfg.dataloader)
        # creating dataset interface and dataloader for val data
        val_dataset = MovingRobotDataset(**self.cfg.val_dataset)

        self.cfg.dataloader.update({"shuffle": False})  # for val dataloader
        val_data = DataLoader(val_dataset, **self.cfg.dataloader)

        print(
            f"Training consists of {len(dataset)} samples, and validation consists of {len(val_dataset)} samples."
        )

        return data, val_data

    def if_resume(self):
        if self.cfg.logger.resume:
            # load checkpoint
            print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - LOADING checkpoint!!!")
            save_dir = self.cfg.directory.load
            checkpoint = load_checkpoint(save_dir, self.device)
            self.model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.epoch = checkpoint["epoch"] + 1
            self.e_loss = checkpoint["e_loss"]
            self.iteration = checkpoint["iteration"] + 1
            self.best = checkpoint["best"]
            print(
                f"{datetime.now():%Y-%m-%d %H:%M:%S} "
                + f"LOADING checkpoint was successful, start from epoch {self.epoch}"
                + f" and loss {self.best}"
            )
        else:
            self.epoch = 1
            self.iteration = 0
            self.best = np.inf
            self.e_loss = []

    def save(self, name=None):
        model = self.model
        if isinstance(self.model, torch.nn.DataParallel):
            model = model.module

        checkpoint = {
            "time": str(datetime.now()),
            "epoch": self.epoch,
            "iteration": self.iteration,
            "model": model.state_dict(),
            "model_name": type(model).__name__,
            "optimizer": self.optimizer.state_dict(),
            "optimizer_name": type(self.optimizer).__name__,
            # "lr_scheduler": self.lr_scheduler.state_dict(),
            "best": self.best,
            "e_loss": self.e_loss,
        }

        if name is None:
            save_name = f"{self.cfg.directory.model_name}_{self.epoch}"

        else:
            save_name = name

        if self.e_loss[-1] < self.best:
            self.best = self.e_loss[-1]
            checkpoint["best"] = self.best
            save_checkpoint(checkpoint, True, self.cfg.directory.save, save_name)
        else:
            save_checkpoint(checkpoint, False, self.cfg.directory.save, save_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", default="./conf/config", type=str)
    args = parser.parse_args()
    cfg_path = args.conf
    learner = Learner(cfg_path)
    learner.train()
