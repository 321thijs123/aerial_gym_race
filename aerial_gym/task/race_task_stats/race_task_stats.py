from aerial_gym.sim.sim_builder import SimBuilder
from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_rotation_6d, quaternion_to_matrix, matrix_to_euler_angles, quaternion_apply
import torch
import numpy as np

from aerial_gym.utils.math import *

from aerial_gym.utils.logging import CustomLogger

from gym.spaces import Dict, Box

from aerial_gym.task.race_task.race_task import (
    RaceTask,
)

logger = CustomLogger("race_task_stats")

class RaceTaskStats(RaceTask):
    def __init__(
        self, task_config, seed=None, num_envs=None, headless=None, device=None, use_warp=None
    ):
        super().__init__(task_config, seed, num_envs, headless, device, use_warp)

        self.stats = {
            "passes": torch.zeros(
                (self.sim_env.num_envs, ), device=self.device, requires_grad=False
            ),
            "terminations": torch.zeros(
                (self.sim_env.num_envs, ), device=self.device, requires_grad=False
            ),
            "truncations": torch.zeros(
                (self.sim_env.num_envs, ), device=self.device, requires_grad=False
            )
        }
    
    def reset_idx(self, env_ids):
        sum_dict = {}

        for key, stat in self.stats.items():
            sum_dict[key] = torch.sum(stat).item()
            print(key, sum_dict[key])

        denom = sum_dict["terminations"] + sum_dict["passes"]
        if denom != 0.0:
            print("gate crash rate: ",  sum_dict["terminations"] / denom)
        
        denom = sum_dict["terminations"] + sum_dict["truncations"]
        if denom != 0.0:
            print("episode crash rate: ",  sum_dict["terminations"] / denom)
            print("passes per episode: ",  sum_dict["passes"] / denom)

        return super().reset_idx(env_ids)

    def process_obs_for_task(self):
        super().process_obs_for_task()

        self.stats["terminations"] += self.terminations
        self.stats["truncations"] += self.truncations
    
    def update_gates(self):
        gate_passings = super().update_gates()

        self.stats["passes"] += gate_passings

        return gate_passings
