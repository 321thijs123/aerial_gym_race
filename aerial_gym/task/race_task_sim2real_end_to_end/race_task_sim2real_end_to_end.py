from aerial_gym.task.base_task import BaseTask
from aerial_gym.sim.sim_builder import SimBuilder
from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_rotation_6d, quaternion_to_matrix, matrix_to_euler_angles, quaternion_apply
import torch
import numpy as np

from aerial_gym.utils.math import *

from aerial_gym.utils.logging import CustomLogger

from gym.spaces import Dict, Box

logger = CustomLogger("race_task")


def dict_to_class(dict):
    return type("ClassFromDict", (object,), dict)


class RaceTaskSim2RealEndToEnd(BaseTask):
    def __init__(
        self, task_config, seed=None, num_envs=None, headless=None, device=None, use_warp=None
    ):
        # overwrite the params if user has provided them
        if seed is not None:
            task_config.seed = seed
        if num_envs is not None:
            task_config.num_envs = num_envs
        if headless is not None:
            task_config.headless = headless
        if device is not None:
            task_config.device = device
        if use_warp is not None:
            task_config.use_warp = use_warp

        super().__init__(task_config)
        self.device = self.task_config.device
        # set the each of the elements of reward parameter to a torch tensor
        for key in self.task_config.reward_parameters.keys():
            self.task_config.reward_parameters[key] = torch.tensor(
                self.task_config.reward_parameters[key], device=self.device
            )
        logger.info("Building environment for race task.")
        logger.info(
            "\nSim Name: {},\nEnv Name: {},\nRobot Name: {}, \nController Name: {}".format(
                self.task_config.sim_name,
                self.task_config.env_name,
                self.task_config.robot_name,
                self.task_config.controller_name,
            )
        )
        logger.info(
            "\nNum Envs: {},\nUse Warp: {},\nHeadless: {}".format(
                self.task_config.num_envs,
                self.task_config.use_warp,
                self.task_config.headless,
            )
        )

        self.sim_env = SimBuilder().build_env(
            sim_name=self.task_config.sim_name,
            env_name=self.task_config.env_name,
            robot_name=self.task_config.robot_name,
            controller_name=self.task_config.controller_name,
            args=self.task_config.args,
            device=self.device,
            num_envs=self.task_config.num_envs,
            use_warp=self.task_config.use_warp,
            headless=self.task_config.headless,
        )

        self.actions = torch.zeros(
            (self.sim_env.num_envs, self.task_config.action_space_dim),
            device=self.device,
            requires_grad=False,
        )
        self.prev_actions = torch.zeros_like(self.actions)
        self.action_history = torch.zeros(
            (self.sim_env.num_envs, self.task_config.action_space_dim*10), device=self.device, requires_grad=False)
        #self.action_history[:, 2] = 0.344

        self.counter = 0

        self.target_position = torch.zeros(
            (self.sim_env.num_envs, 3), device=self.device, requires_grad=False
        )

        self.num_gates = self.sim_env.obstacle_manager.obstacle_position.size(dim=1)

        self.gates_sin_cos = torch.zeros(
            (self.sim_env.num_envs, self.num_gates, 2), device=self.device, requires_grad=False
        )

        self.gate_idx = torch.zeros(
            (self.sim_env.num_envs,), device=self.device, requires_grad=False, dtype=torch.long
        )

        # Get the dictionary once from the environment and use it to get the observations later.
        # This is to avoid constant retuning of data back anf forth across functions as the tensors update and can be read in-place.
        self.obs_dict = self.sim_env.get_obs()
        self.obs_dict["num_obstacles_in_env"] = self.num_gates
        self.terminations = self.obs_dict["crashes"]
        self.truncations = self.obs_dict["truncations"]
        self.rewards = torch.zeros(self.truncations.shape[0], device=self.device)
        self.prev_position = torch.zeros_like(self.obs_dict["robot_position"])

        self.prev_pos_error = torch.zeros((self.sim_env.num_envs, 3), device=self.device, requires_grad=False)

        # self.observation_space = Dict(
        #     {"observations": Box(low=-1.0, high=1.0, shape=(self.task_config.observation_space_dim,), dtype=np.float32)}
        # )
        self.action_space = Box(
            low=-1.0,
            high=1.0,
            shape=(self.task_config.action_space_dim,),
            dtype=np.float32,
        )
        # self.action_transformation_function = self.sim_env.robot_manager.robot.action_transformation_function

        self.num_envs = self.sim_env.num_envs

        # Currently only the "observations" are sent to the actor and critic.
        # The "priviliged_obs" are not handled so far in sample-factory

        self.task_obs = {
            "observations": torch.zeros(
                (self.sim_env.num_envs, self.task_config.observation_space_dim),
                device=self.device,
                requires_grad=False,
            ),
            "priviliged_obs": torch.zeros(
                (
                    self.sim_env.num_envs,
                    self.task_config.privileged_observation_space_dim,
                ),
                device=self.device,
                requires_grad=False,
            ),
            "collisions": torch.zeros(
                (self.sim_env.num_envs, 1), device=self.device, requires_grad=False
            ),
            "rewards": torch.zeros(
                (self.sim_env.num_envs, 1), device=self.device, requires_grad=False
            ),
        }

    def close(self):
        self.sim_env.delete_env()

    def reset(self):
        return self.reset_idx(torch.arange(self.num_envs, device=self.device))

    def reset_idx(self, env_ids):
        # self.target_position[:, 0:3] = 0.0  # torch.rand_like(self.target_position) * 10.0
        self.infos = {}
        self.sim_env.reset_idx(env_ids)
        self.action_history[env_ids] = 0.0
        self.gate_idx[env_ids] = 0
        self.target_position[env_ids, :] = self.sim_env.obstacle_manager.obstacle_position[env_ids, 0, :]
        self.prev_actions[env_ids] = 0.0
        self.prev_pos_error[env_ids] = self.target_position[env_ids] - self.obs_dict["robot_position"][env_ids]

        gates_quat = self.sim_env.obstacle_manager.obstacle_orientation[env_ids, :, :][:,:,[3, 0, 1, 2]]

        qw = gates_quat[:, :, 0]
        qz = gates_quat[:, :, 3]

        sin_yaw = 2.0 * qw * qz
        cos_yaw = qw * qw - qz * qz

        self.gates_sin_cos[env_ids, :, 0] = sin_yaw
        self.gates_sin_cos[env_ids, :, 1] = cos_yaw

        return self.get_return_tuple()

    def render(self):
        return None

    def handle_action_history(self, actions):
        old_action_history = self.action_history.clone()
        self.action_history[:, self.task_config.action_space_dim:] = old_action_history[:, :-self.task_config.action_space_dim]
        self.action_history[:, :self.task_config.action_space_dim] = actions

    def step(self, actions):
        self.counter += 1
        self.actions = self.task_config.process_actions_for_task(
            actions, self.task_config.action_limit_min, self.task_config.action_limit_max
        )
        self.prev_position[:] = self.obs_dict["robot_position"].clone()

        self.sim_env.step(actions=self.actions)

        gate_passings = self.update_gates()

        self.rewards[:], self.terminations[:] = self.compute_rewards_and_crashes(self.obs_dict, gate_passings)

        if self.task_config.return_state_before_reset == True:
            return_tuple = self.get_return_tuple()

        self.truncations[:] = torch.where(
            self.sim_env.sim_steps > self.task_config.episode_len_steps, 1, 0
        )

        reset_envs = self.sim_env.post_reward_calculation_step()
        if len(reset_envs) > 0:
            self.reset_idx(reset_envs)

        self.infos = {}  # self.obs_dict["infos"]

        if self.task_config.return_state_before_reset == False:
            return_tuple = self.get_return_tuple()

        self.prev_actions = self.actions.clone()
        self.prev_pos_error = self.target_position - self.obs_dict["robot_position"]

        return return_tuple

    def get_return_tuple(self):
        self.process_obs_for_task()
        return (
            self.task_obs,
            self.rewards,
            self.terminations,
            self.truncations,
            self.infos,
        )

    def process_obs_for_task(self):
        or_quat = self.obs_dict["robot_orientation"][:,[3, 0, 1, 2]]
        # or_euler = matrix_to_euler_angles(quaternion_to_matrix(or_quat), "ZYX")[:, [2, 1, 0]]
        or_matr = quaternion_to_matrix(or_quat)
        or_yaw = matrix_to_euler_angles(or_matr, "ZYX")[:, 0]

        # Altitude, orientation, linear velocity, angular velocity
        self.task_obs["observations"][:, 0] = self.obs_dict["robot_position"][:,2]
        self.task_obs["observations"][:, 1:7] = matrix_to_rotation_6d(or_matr)
        self.task_obs["observations"][:, 7:10] = self.obs_dict["robot_linvel"]
        # self.task_obs["observations"][:, 7:10] = self.obs_dict["robot_body_linvel"]
        self.task_obs["observations"][:, 10:13] = self.obs_dict["robot_body_angvel"]
        
        # Gates
        for i in range(self.num_gates):
            this_gate_idx = (self.gate_idx + i) % self.num_gates
            this_gate_pos = self.sim_env.obstacle_manager.obstacle_position[torch.arange(self.num_envs), this_gate_idx, :]

            self.task_obs["observations"][:, (13 + 5*i):(16 + 5*i)] = this_gate_pos - self.obs_dict["robot_position"]
            # self.task_obs["observations"][:, (13 + 5*i):(16 + 5*i)] = quat_rotate_inverse(
            #     self.obs_dict["robot_vehicle_orientation"],
            #     (this_gate_pos - self.obs_dict["robot_position"]),
            # )

            self.task_obs["observations"][:, (16 + 5*i)] = self.gates_sin_cos[torch.arange(self.num_envs), this_gate_idx, 0]
            self.task_obs["observations"][:, (17 + 5*i)] = self.gates_sin_cos[torch.arange(self.num_envs), this_gate_idx, 1]

        self.task_obs["rewards"] = self.rewards
        self.task_obs["terminations"] = self.terminations
        self.task_obs["truncations"] = self.truncations
        
    def update_gates(self):
        gate_passings = get_gate_passings(
            self.prev_pos_error, 
            self.target_position - self.obs_dict["robot_position"], 
            self.gates_sin_cos[torch.arange(self.num_envs), self.gate_idx]
        )

        if (gate_passings[0]):
            print(f"Gate {self.gate_idx[0].item()} passed")

        self.gate_idx = (self.gate_idx + gate_passings.long()) % self.num_gates
        self.target_position = self.sim_env.obstacle_manager.obstacle_position[torch.arange(self.num_envs), self.gate_idx, :]

        return gate_passings

    def compute_rewards_and_crashes(self, obs_dict, gate_passings):
        return compute_reward(
            self.target_position - self.obs_dict["robot_position"],
            obs_dict["robot_orientation"],
            obs_dict["robot_linvel"],
            obs_dict["robot_body_angvel"],
            obs_dict["crashes"],
            self.actions.clone(),
            self.prev_actions,
            self.prev_pos_error,
            self.task_config.crash_dist,
            gate_passings
        )

@torch.jit.script
def get_gate_passings(P0, P1, gates_sin_cos, width=1.0, height=1.0):
    # type: (Tensor, Tensor, Tensor, float, float) -> Tensor
    sin_yaw = gates_sin_cos[:,0]
    cos_yaw = gates_sin_cos[:,1]

    P0_local = torch.empty_like(P0)
    P1_local = torch.empty_like(P1)

    P0_local[:,0] = P0[:,0] * cos_yaw - P0[:,1] * sin_yaw
    P0_local[:,1] = P0[:,0] * sin_yaw + P0[:,1] * cos_yaw
    P0_local[:,2] = P0[:,2]

    P1_local[:,0] = P1[:,0] * cos_yaw - P1[:,1] * sin_yaw
    P1_local[:,1] = P1[:,0] * sin_yaw + P1[:,1] * cos_yaw
    P1_local[:,2] = P1[:,2]

    P_diff = P1_local - P0_local

    P_hit = P1_local - P_diff / P_diff[:,[0]] * P1_local[:,[0]]

    in_bounds = torch.logical_and(
        torch.abs(P_hit[:,1]) < width / 2,
        torch.abs(P_hit[:,2]) < height / 2
    )

    through_plane = torch.ne(torch.sign(P0_local[:,0]), torch.sign(P1_local[:,0]))

    return torch.logical_and(in_bounds, through_plane)

@torch.jit.script
def exp_func(x, gain, exp):
    # type: (Tensor, float, float) -> Tensor
    return gain * torch.exp(-exp * x * x)


@torch.jit.script
def exp_penalty_func(x, gain, exp):
    # type: (Tensor, float, float) -> Tensor
    return gain * (torch.exp(-exp * x * x) - 1)

@torch.jit.script
# type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor) -> Tuple[Tensor, Tensor]
def compute_reward(
                     pos_error,
                     quats,
                     linvels_err,
                     angvels_err,
                     crashes,
                     action_input,
                     prev_action,
                     prev_pos_error,
                     crash_dist,
                     gate_passings):

    target_dist = torch.norm(pos_error[:, :3], dim=1)

    # Reward for moving towards gate
    prev_target_dist = torch.norm(prev_pos_error, dim=1)
    closer_by_dist = prev_target_dist - target_dist

    crashes[:] = torch.where(target_dist > crash_dist, torch.ones_like(crashes), crashes)

    towards_gate_reward = closer_by_dist * torch.logical_not(gate_passings)

    # Reward for looking at gate
    cam_pitch = -torch.pi/6
    cam_ax_b = torch.tensor([torch.cos(cam_pitch), 0.0, -torch.sin(cam_pitch)])
    cam_ax_w = quaternion_apply(quats[:,[3, 0, 1, 2]], cam_ax_b)

    gate_dir = pos_error / torch.norm(pos_error, dim=1, keepdim=True)

    look_reward = torch.clamp(torch.sum(cam_ax_w * gate_dir, dim=1) * towards_gate_reward, min=0)

    # Reward for not rotating
    smooth_reward = -torch.abs(angvels_err[:,2])

    reward = gate_passings * 200 + 15 * towards_gate_reward + look_reward * 5 + smooth_reward * 0.2 + 0.01 

    return reward, crashes


