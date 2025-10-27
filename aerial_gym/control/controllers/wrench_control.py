import torch
import pytorch3d.transforms as p3d_transforms
from aerial_gym.utils.math import *


from aerial_gym.control.controllers.base_lee_controller import *


class WrenchController(BaseLeeController):
    def __init__(self, config, num_envs, device):
        super().__init__(config, num_envs, device)

    def init_tensors(self, global_tensor_dict=None):
        super().init_tensors(global_tensor_dict)

    def update(self, command_actions):
        self.reset_commands()
        self.wrench_command[:, 2] = command_actions[:, 0]
        self.wrench_command[:, 3:] = command_actions[:,1:]

        return self.wrench_command
