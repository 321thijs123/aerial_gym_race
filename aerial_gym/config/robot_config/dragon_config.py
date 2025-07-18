import numpy as np

from aerial_gym import AERIAL_GYM_DIRECTORY

from aerial_gym.config.sensor_config.camera_config.fpv_camera_config import (
    FpvCameraConfig,
)
from aerial_gym.config.sensor_config.lidar_config.base_lidar_config import (
    BaseLidarConfig,
)
from aerial_gym.config.sensor_config.lidar_config.osdome_64_config import OSDome_64_Config
from aerial_gym.config.sensor_config.imu_config.base_imu_config import BaseImuConfig


class DragonCfg:

    class init_config:
        # init_state tensor is of the format [ratio_x, ratio_y, ratio_z, roll_radians, pitch_radians, yaw_radians, 1.0 (for maintaining shape), vx, vy, vz, wx, wy, wz]
        min_init_state = [
            -1.0,
            -1.0,
            1.0,
            -np.pi/6,
            -np.pi/6,
            -np.pi,
            1.0,
            -0.5,
            -0.5,
            -0.5,
            -0.5,
            -0.5,
            -0.5,
        ]
        max_init_state = [
            1.0,
            1.0,
            2.0,
            np.pi/6,
            np.pi/6,
            np.pi,
            1.0,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
        ]

    class sensor_config:
        enable_camera = False
        camera_config = FpvCameraConfig

        enable_lidar = False
        lidar_config = BaseLidarConfig  # OSDome_64_Config

        enable_imu = False
        imu_config = BaseImuConfig

    class disturbance:
        enable_disturbance = False 
        prob_apply_disturbance = 0.02
        max_force_and_torque_disturbance = [0.001, 0.001, 0.001, 0.00004, 0.00004, 0.00004]#[0.1, 0.1, 0.1, 0.004, 0.004, 0.004]

    class damping:
        linvel_linear_damping_coefficient = [0.0, 0.0, 0.0]  # along the body [x, y, z] axes
        linvel_quadratic_damping_coefficient = [0.0, 0.0, 0.0]  # along the body [x, y, z] axes
        angular_linear_damping_coefficient = [0.0, 0.0, 0.0]  # along the body [x, y, z] axes
        angular_quadratic_damping_coefficient = [0.0, 0.0, 0.0]  # along the body [x, y, z] axes

    class gyro_effects:
        rotor_inertia = 2224.04e-9
        rotor_axis = [0., 0., 1.]

    class robot_asset:
        file = "dragon.urdf"
        name = "dragon"  # actor name
        base_link_name = "base_link"
        disable_gravity = False
        collapse_fixed_joints = False  # merge bodies connected by fixed joints.
        fix_base_link = False  # fix the base of the robot
        collision_mask = 0  # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = False  # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = True  # Some .obj meshes must be flipped from y-up to z-up
        density = 0.000001
        angular_damping = 0.01
        linear_damping = 0.01
        max_angular_velocity = 100.0
        max_linear_velocity = 100.0
        armature = 0.001

        semantic_id = 0
        per_link_semantic = False

        min_state_ratio = [
            -1.0,
            -1.0,
            -1.0,
            0,
            0,
            -np.pi,
            1.0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]  # [x, y, z, roll_rad, pitch_rad, yaw_rad, 1.0, vx, vy, vz, wx, wy, wz]
        max_state_ratio = [
            1.0,
            1.0,
            1.0,
            0,
            0,
            np.pi,
            1.0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]  # [x, y, z, roll_rad, pitch_rad, yaw_rad, 1.0, vx, vy, vz, wx, wy, wz]

        color = None
        semantic_masked_links = {}
        keep_in_env = True  # this does nothing for the robot

        min_position_ratio = None
        max_position_ratio = None

        min_euler_angles = [-np.pi, -np.pi, -np.pi]
        max_euler_angles = [np.pi, np.pi, np.pi]

        place_force_sensor = True  # set this to True if IMU is desired
        force_sensor_parent_link = "base_link"
        force_sensor_transform = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]  # [x, y, z, qx, qy, qz, qw]

        use_collision_mesh_instead_of_visual = False  # does nothing for the robot
        
        asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/robots/dragon"
        

    class control_allocator_config:
        force_application_level = "motor_link"
        num_motors = 4
        application_mask = [6,7,8,9]
        motor_directions = [1, -1, 1, -1]
        allocation_matrix = [[ 0., 0., 0., 0.],
                             [ 0., 0., 0., 0.],
                             [ 1., 1., 1., 1.],
                             [-0.16, -0.16, 0.16, 0.16],
                             [-0.16, 0.16, 0.16, -0.16],
                             [-0.01, 0.01, -0.01, 0.01]]

        class motor_model_config:
            use_rps = True
            motor_thrust_constant_min = 1.32581395e-8
            motor_thrust_constant_max = 1.32581395e-8
            motor_time_constant_increasing_min = 0.047
            motor_time_constant_increasing_max = 0.047
            motor_time_constant_decreasing_min = 0.047
            motor_time_constant_decreasing_max = 0.047
            max_thrust = 1466.9 / 1000 * 9.81
            min_thrust = 0.0
            max_thrust_rate = 100000.0
            thrust_to_torque_ratio = 0.01
            use_discrete_approximation = True
            integration_scheme = "rk4" #"euler"

class DragonWithCameraCfg(DragonCfg):
    class sensor_config(DragonCfg.sensor_config):
        enable_camera = True
