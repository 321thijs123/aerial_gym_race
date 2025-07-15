from aerial_gym.config.sensor_config.base_sensor_config import BaseSensorConfig
import numpy as np


class FpvCameraConfig(BaseSensorConfig):
    num_sensors = 1

    sensor_type = "camera"

    height = 960
    width = 1280
    horizontal_fov_deg = 122.000
    max_range = 10.0    # Far plane
    min_range = 0.05    # Near plane

    return_pointcloud = False
    pointcloud_in_world_frame = False
    segmentation_camera = False
    depth_camera = False

    nominal_position = [0.07263, 0.0, 0.0]
    nominal_orientation_euler_deg = [0.0, -30.0, 0.0]

    use_collision_geometry = False

    # Currently not implemented for color cameras
    class sensor_noise:
        enable_sensor_noise = False
        pixel_dropout_prob = 0.01
        pixel_std_dev_multiplier = 0.01
