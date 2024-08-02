import pickle
import torch
from TRON.model.model import DownstreamNetwork
from TRON.model.tron_model import VisionSpeedEncoder, IMUEncoder


class ModelClass:
    def __init__(self, batch_size):
        self.rep_size = 512
        self.device = torch.device('cuda')
        
        load_dir = 'path_to_model/model.pth' 
        checkpoint = torch.load(load_dir, map_location=self.device)
        visual_encoder = VisionSpeedEncoder(latent_size=self.rep_size)
        
        imu_encoder = IMUEncoder(latent_size=self.rep_size)

        self.model = DownstreamNetwork(visual_encoder, imu_encoder, rep_size=self.rep_size)
        self.model.load_state_dict(checkpoint['model'], strict=False)
        self.model.eval()
        self.model.to(self.device)

        self.batch_size = batch_size
        self.roll_slide_bump_threshold = torch.tensor([0.28, 0.07, 0.07], dtype=torch.float).to(self.device)

        self.frame = 0
        self.last_vels_passed = 100
        self.none_passed = 0

        with open("path_to_stats/stats.pkl", 'rb') as f:
            self.stats = pickle.load(f)

    @torch.no_grad() 
    def forward(self, patch_underneath, accel_msgs, gyro_msgs, vx_vy_vz, input_cmd_vel, actual_cmd_vel, combinatation_cmd_vel, actual_vel_msg):
        self.frame += 1 
        print("=" * 40, self.frame, "=" * 40, self.last_vels_passed)

        roll, slide, bump = self.model(patch_underneath, vx_vy_vz, input_cmd_vel, None, None)

        roll = (roll * (self.stats['res_vel_omega_roll_slde_bump_std'][2].cuda() + 0.000006)) + self.stats['res_vel_omega_roll_slde_bump_mean'][2]
        slide = (slide * (self.stats['res_vel_omega_roll_slde_bump_std'][3].cuda() + 0.000006)) + self.stats['res_vel_omega_roll_slde_bump_mean'][3]
        bump = (bump * (self.stats['res_vel_omega_roll_slde_bump_std'][4].cuda() + 0.000006)) + self.stats['res_vel_omega_roll_slde_bump_mean'][4]

        max_bump = torch.max(bump).item()
        min_bump = torch.min(bump).item()
        max_roll = torch.max(roll).item()
        min_roll = torch.min(roll).item()
        max_slide = torch.max(slide).item()
        min_slide = torch.min(slide).item()

        cmd_vel_to_execute = self.apply_threshold(slide, combinatation_cmd_vel, actual_cmd_vel, vx_vy_vz) 
        return cmd_vel_to_execute, [max_roll, min_roll, max_slide, min_slide, max_bump, min_bump, self.last_vels_passed]

    def apply_threshold(self, results, combination_cmd_vel, actual_cmd_vel):
        threshold_mask = results <= self.roll_slide_bump_threshold
        filtered_mask = torch.all(threshold_mask, dim=1)

        cmd_vel_filtered = combination_cmd_vel[filtered_mask]
        self.last_vels_passed = cmd_vel_filtered.shape[0]

        if self.last_vels_passed > 0:
            self.none_passed = 0
            actual_cmd_vel = torch.tensor(actual_cmd_vel, dtype=torch.float32, device=self.device).unsqueeze(0).repeat(self.last_vels_passed, 1)[:, -2:] #.reshape(2, self.last_vels_passed).T
            cmd_vel_filtered = cmd_vel_filtered[:, -2:]
            difference_squared = (cmd_vel_filtered - actual_cmd_vel)**2
            error = difference_squared[:, 0] * 1 + difference_squared[:, 1]

            return cmd_vel_filtered[torch.argmin(error)].cpu().tolist()
        else:
            print("#####################")
            self.none_passed += 1

            return None 


