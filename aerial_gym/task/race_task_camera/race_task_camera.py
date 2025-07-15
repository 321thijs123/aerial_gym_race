from aerial_gym.sim.sim_builder import SimBuilder
from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_rotation_6d, quaternion_to_matrix, matrix_to_euler_angles, quaternion_apply
import torch
import numpy as np

from aerial_gym.utils.math import *

from aerial_gym.utils.logging import CustomLogger

from gym.spaces import Dict, Box

from aerial_gym.task.race_task_sim2real_end_to_end.race_task_sim2real_end_to_end import (
    RaceTaskSim2RealEndToEnd,
)

logger = CustomLogger("race_task_camera")

from PIL import Image
from pathlib import Path
import imageio

class RaceTaskCamera(RaceTaskSim2RealEndToEnd):
    def __init__(
        self, task_config, seed=None, num_envs=None, headless=None, device=None, use_warp=None
    ):
        use_warp = False # Force warp off to use RGB camera
        super().__init__(task_config, seed, num_envs, headless, device, use_warp)

        self.episode = torch.zeros(num_envs, device=self.device, dtype=torch.int)

        Path(f"camera_output").mkdir(parents=True, exist_ok=True)
        self.writer = None

    def step(self, actions):
        return_dict = super().step(actions)

        rgba = self.sim_env.global_tensor_dict["rgb_pixels"][0, 0]  # shape: (H, W, 4)
        
        img = rgba.detach().cpu().numpy().astype(np.uint8)[..., :3]  # (H, W, 3)
        # im = Image.fromarray(img)
        # im.save(f"camera_output//episode_{self.episode[0].item()}/{self.counter}.jpeg")

        self.writer.append_data(img)

        return return_dict
    
    def reset_idx(self, env_ids):
        self.episode[env_ids] += 1

        if self.writer is not None:
            self.writer.close()

        # Path(f"camera_output/episode_{self.episode[0].item()}").mkdir(parents=True, exist_ok=True)
        self.writer = imageio.get_writer(f'camera_output/episode_{self.episode[0].item()}.mp4', fps=100)

        return super().reset_idx(env_ids)
