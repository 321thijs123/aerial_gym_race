from typing import Any
from aerial_gym.utils.math import *

from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger("asset_manager")
logger.setLevel("DEBUG")


class AssetManager:
    def __init__(self, global_tensor_dict, num_keep_in_env):
        self.init_tensors(global_tensor_dict, num_keep_in_env)

    def init_tensors(self, global_tensor_dict, num_keep_in_env):
        self.env_asset_state_tensor = global_tensor_dict["env_asset_state_tensor"]
        self.asset_min_state = global_tensor_dict["asset_min_state_ratio"]
        self.asset_max_state = global_tensor_dict["asset_max_state_ratio"]
        self.env_bounds_min = (
            global_tensor_dict["env_bounds_min"]
            .unsqueeze(1)
            .expand(-1, self.env_asset_state_tensor.shape[1], -1)
        )
        self.env_bounds_max = (
            global_tensor_dict["env_bounds_max"]
            .unsqueeze(1)
            .expand(-1, self.env_asset_state_tensor.shape[1], -1)
        )
        self.num_keep_in_env = num_keep_in_env

    def prepare_for_sim(self):
        self.reset(self.num_keep_in_env)
        logger.warning(f"Number of obstacles to be kept in the environment: {self.num_keep_in_env}")

    def pre_physics_step(self, actions):
        pass

    def post_physics_step(self):
        pass

    def step(self, actions):
        pass
        # Implement this function if needed.
        # this functionality can do speciic things with the environment assets on stepping.
        # nothing really needs to be done for static environments.
        # if force needs to be applied, it should be done in the other classes and it's
        # better to leave this class to manipulate the state tensors.

    def reset(self, num_obstacles_per_env):
        self.reset_idx(torch.arange(self.env_asset_state_tensor.shape[0], device=self.env_asset_state_tensor.device), num_obstacles_per_env)

    def reset_idx(self, env_ids, num_obstacles_per_env=0):
        if num_obstacles_per_env < self.num_keep_in_env:
            logger.info(
                "Number of obstacles required in the environment by the \
                  code is lesser than the minimum number of obstacles that the environment configuration specifies."
            )
            num_obstacles_per_env = self.num_keep_in_env

        sampled_asset_states = torch_rand_float_tensor(self.asset_min_state[env_ids, :, :], self.asset_max_state[env_ids, :, :])

        min_dist = 2.0

        for i in range(1,num_obstacles_per_env):
            while True:
                distances = sampled_asset_states[:, :i, 0:3] - sampled_asset_states[:, i, 0:3].unsqueeze(1)
                too_close = torch.any(torch.norm(distances, dim=2) < min_dist, dim=1)

                if torch.sum(too_close) == 0:
                    break

                too_close_ids = env_ids[too_close]

                sampled_asset_states[too_close, i, 0:3] = torch_rand_float_tensor(self.asset_min_state[too_close_ids, i, 0:3], self.asset_max_state[too_close_ids, i, 0:3])

        self.env_asset_state_tensor[env_ids, :, 0:3] = sampled_asset_states[:, :, 0:3]
        self.env_asset_state_tensor[env_ids, :, 3:7] = quat_from_euler_xyz_tensor(
            sampled_asset_states[:, :, 3:6]
        )
        
        if self.env_asset_state_tensor.size(dim=1) > 1:
            self.env_asset_state_tensor[env_ids, -1, :3] = self.env_asset_state_tensor[env_ids, -2, :3]
            self.env_asset_state_tensor[env_ids, -1, 2] += 1.8

        # put those obstacles not needed in the environment outside
        self.env_asset_state_tensor[env_ids, num_obstacles_per_env:, 0:3] = -1000.0
