import gc
from pathlib import Path
from datetime import datetime
import sys
import argparse

try:
    sys.path.append(str(Path(".").resolve()))
except:
    raise RuntimeError("Can't append root directory of the project to the path")

from rich import print
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model.dataloader import TronDataset
from model.tron_model import IMUEncoder, VisionSpeedEncoder, TronModel
from utils.nn import check_grad_norm, op_counter
from utils.io import save_checkpoint, load_checkpoint
from utils.helpers import get_conf, timeit


matplotlib.use('Agg')


class Learner:
    def __init__(self, cfg_dir: str):
        # load config file and initialize the logger and the device
        self.cfg = get_conf(cfg_dir)
        self.cfg.directory.model_name = self.cfg.train_params.experiment_name
        self.cfg.directory.model_name += f"-{self.cfg.model.rep_size}-{datetime.now():%m-%d-%H-%M}"
        self.cfg.train_params.experiment_name = self.cfg.directory.model_name
        self.cfg.directory.save = str(
            Path(self.cfg.directory.save) / self.cfg.directory.model_name
        )
        # self.logger = self.init_logger()
        self.device = self.init_device()
        # creating dataset interface and dataloader for trained data
        self.data = self.init_dataloader()
        # create model and initialize its weights and move them to the device
        self.model = self.init_model()
        # initialize the optimizer
        self.optimizer = self.init_optimizer()
        # if resuming, load the checkpoint
        self.if_resume()

    def train(self):
        """Trains the model"""
        while self.epoch <= self.cfg.train_params.epochs:
            running_loss = []
            running_loss_vpt_inv = []
            running_loss_vi = []

            bar = tqdm(
                self.data,
                desc=f"Epoch {self.epoch:03}/{self.cfg.train_params.epochs:03}, training: ",
            )
            for data in bar:
                self.iteration += 1
                (loss, loss_vpt_inv, loss_vi, grad_norm), t_train = self.forward_batch(data)
                t_train /= self.data.batch_size
                running_loss.append(loss)
                running_loss_vpt_inv.append(loss_vpt_inv)
                running_loss_vi.append(loss_vi)

                bar.set_postfix(loss=loss, loss_vpt_inv=loss_vpt_inv, loss_vi=loss_vi, Grad=grad_norm, Time=t_train)

            # evaluate the data
            with torch.no_grad():
                self.model.eval()
                patch1, patch2, acc, gyro, vel, label = data  # only checking on the last batch
                patch1 = patch1.to(device=self.device)
                patch2 = patch2.to(device=self.device)
                acc = acc.to(device=self.device)
                gyro = gyro.to(device=self.device)
                vel = vel.to(device=self.device)
                # returns: zv1, zv2, zi, v_encoded_1, v_encoded_2, i_encoded
                _, _, _, v_encoded_1, _, _ = self.model(patch1, patch2, acc, gyro, vel)
                X_embedded = TSNE(n_components=2, learning_rate='auto', init = 'random', perplexity = 3).fit_transform(v_encoded_1.detach().cpu().numpy())
                labels = {0: "concrete:black", 1: 'grass:green', 2: 'rocks:gray'}
                colors = {0: "black", 1: 'green', 2: 'gray'}
                for i in range(X_embedded.shape[0]):
                    plt.scatter(X_embedded[i, 0], X_embedded[i, 1], c=colors[label[i].item()])

                plt.grid(True)
                plt.legend(list(labels.values()))
                plt.savefig(f'tsne/sample_E{self.epoch:03}.jpg', format='jpg')
                plt.clf()
                plt.close()

            bar.close()

            # average loss for an epoch
            self.e_loss.append(np.mean(running_loss))  # epoch loss
            print(
                f"{datetime.now():%Y-%m-%d %H:%M:%S} Epoch {self.epoch:03}, "
                + f"Iteration {self.iteration:05} summary: train Loss: {self.e_loss[-1]:.2f} || "
                + f"VPT Loss: {np.mean(running_loss_vpt_inv):.2f} || "
                + f"VI Loss: {np.mean(running_loss_vi):.2f}"
            )

            if self.epoch % self.cfg.train_params.save_every == 0 or (
                self.e_loss[-1] < self.best
                and self.epoch >= self.cfg.train_params.start_saving_best
            ):
                self.save()

            gc.collect()
            self.epoch += 1

        patch1, patch2, acc, gyro, vel, _ = next(iter(self.data))

        patch1 = patch1.to(device=self.device)
        patch2 = patch2.to(device=self.device)
        acc = acc.to(device=self.device)
        gyro = gyro.to(device=self.device)
        vel = vel.to(device=self.device)
        macs, params = op_counter(self.model, sample=(patch1, patch2, acc, gyro, vel))
        print("macs = ", macs, " | params = ", params)
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - Training is DONE!")

    @timeit
    def forward_batch(self, data):
        """Forward pass of a batch"""
        self.model.train()
        # move data to device
        patch1, patch2, acc, gyro, vel, _ = data
        patch1 = patch1.to(device=self.device)
        patch2 = patch2.to(device=self.device)
        acc = acc.to(device=self.device)
        gyro = gyro.to(device=self.device)
        vel = vel.to(device=self.device)
        zv1, zv2, zi, _, _, _ = self.model(patch1, patch2, acc, gyro, vel)

        # compute viewpoint invariance
        loss_vpt_inv = self.model.barlow_loss(zv1, zv2)
        # compute visual-inertial
        loss_vi = self.model.barlow_loss(zv1, zi)  + 0.5 * self.model.barlow_loss(zv2, zi)

        loss = self.cfg.model.l1_coeff * loss_vpt_inv + (1.0 - self.cfg.model.l1_coeff) * loss_vi

        # forward, backward
        self.optimizer.zero_grad()
        loss.backward()
        # gradient clipping
        if self.cfg.train_params.grad_clipping > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.train_params.grad_clipping
            )
        # update
        self.optimizer.step()

        # check grad norm for debugging
        grad_norm = check_grad_norm(self.model)

        return loss.detach().cpu().item(), loss_vpt_inv.detach().item(), loss_vi.detach().item(), grad_norm

    def init_model(self):
        """Initializes the model"""
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - INITIALIZING the model with {self.cfg.model.rep_size} representation dim!")
        # encoder architecture
        visual_encoder = VisionSpeedEncoder(latent_size=self.cfg.model.rep_size).to(self.device)
        imu_encoder = IMUEncoder(latent_size=self.cfg.model.rep_size).to(self.device)
        # projector head
        projector = nn.Sequential(
            nn.Linear(self.cfg.model.rep_size, self.cfg.model.projection_dim), nn.PReLU(),
            nn.Linear(self.cfg.model.projection_dim, self.cfg.model.projection_dim)
        ).to(self.device)

        model = TronModel(visual_encoder, imu_encoder, projector)

        return model.to(self.device)

    def init_optimizer(self):
        """Initializes the optimizer and learning rate scheduler"""
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - INITIALIZING the optimizer!")
        if self.cfg.train_params.optimizer.lower() == "adamw":
            optimizer = optim.AdamW(self.model.parameters(), **self.cfg.adamw)

        elif self.cfg.train_params.optimizer.lower() == "rmsprop":
            optimizer = optim.RMSprop(self.model.parameters(), **self.cfg.rmsprop)

        elif self.cfg.train_params.optimizer.lower() == "sgd":
            optimizer = optim.SGD(self.model.parameters(), **self.cfg.sgd)

        else:
            raise ValueError(
                f"Unknown optimizer {self.cfg.train_params.optimizer}"
                + "; valid optimizers are 'adam' and 'rmsprop'."
            )
        return optimizer

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
        dataset = TronDataset(**self.cfg.dataset)
        data = DataLoader(dataset, **self.cfg.dataloader)
        print(
            f"{datetime.now():%Y-%m-%d %H:%M:%S} - Training consists of {len(dataset)} samples."
        )

        return data

    def if_resume(self):
        if self.cfg.train_params.resume:
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
        checkpoint = {
            "time": str(datetime.now()),
            "epoch": self.epoch,
            "iteration": self.iteration,
            "visual_encoder": self.model.visual_encoder.state_dict(),
            "imu_encoder": self.model.imu_encoder.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "optimizer_name": type(self.optimizer).__name__,
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
    parser.add_argument("--conf", default="./conf/config_tron", type=str)
    args = parser.parse_args()
    cfg_path = args.conf
    learner = Learner(cfg_path)
    learner.train()
