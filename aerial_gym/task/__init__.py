from aerial_gym.task.position_setpoint_task.position_setpoint_task import (
    PositionSetpointTask,
)

from aerial_gym.task.position_setpoint_task_sim2real.position_setpoint_task_sim2real import (
    PositionSetpointTaskSim2Real,
)

from aerial_gym.task.position_setpoint_task_sim2real_end_to_end.position_setpoint_task_sim2real_end_to_end import (
    PositionSetpointTaskSim2RealEndToEnd,
)

from aerial_gym.task.position_setpoint_task_acceleration_sim2real.position_setpoint_task_acceleration_sim2real import (
    PositionSetpointTaskAccelerationSim2Real,
)

from aerial_gym.task.race_task_sim2real_end_to_end.race_task_sim2real_end_to_end import (
    RaceTaskSim2RealEndToEnd,
)

from aerial_gym.task.race_task_camera.race_task_camera import (
    RaceTaskCamera,
)

from aerial_gym.task.navigation_task.navigation_task import NavigationTask

from aerial_gym.config.task_config.position_setpoint_task_config import (
    task_config as position_setpoint_task_config,
)

from aerial_gym.config.task_config.position_setpoint_task_sim2real_config import (
    task_config as position_setpoint_task_sim2real_config,
)

from aerial_gym.config.task_config.position_setpoint_task_sim2real_end_to_end_config import (
    task_config as position_setpoint_task_sim2real_end_to_end_config,
)

from aerial_gym.config.task_config.race_task_sim2real_end_to_end_config import (
    task_config as race_task_sim2real_end_to_end_config,
)

from aerial_gym.config.task_config.race_task_camera_config import (
    task_config as race_task_camera_config,
)

from aerial_gym.config.task_config.position_setpoint_task_acceleration_sim2real_config import (
    task_config as position_setpoint_task_acceleration_sim2real_config,
)

from aerial_gym.config.task_config.navigation_task_config import (
    task_config as navigation_task_config,
)

from aerial_gym.registry.task_registry import task_registry


task_registry.register_task(
    "position_setpoint_task", PositionSetpointTask, position_setpoint_task_config
)
task_registry.register_task(
    "position_setpoint_task_sim2real",
    PositionSetpointTaskSim2Real,
    position_setpoint_task_sim2real_config,
)

task_registry.register_task(
    "position_setpoint_task_sim2real_end_to_end",
    PositionSetpointTaskSim2RealEndToEnd,
    position_setpoint_task_sim2real_end_to_end_config,
)

task_registry.register_task(
    "race_task_sim2real_end_to_end",
    RaceTaskSim2RealEndToEnd,
    race_task_sim2real_end_to_end_config,
)

task_registry.register_task(
    "race_task_camera",
    RaceTaskCamera,
    race_task_camera_config,
)

task_registry.register_task(
    "position_setpoint_task_acceleration_sim2real",
    PositionSetpointTaskAccelerationSim2Real,
    position_setpoint_task_acceleration_sim2real_config,
)

task_registry.register_task("navigation_task", NavigationTask, navigation_task_config)


from aerial_gym.task.position_setpoint_task_reconfigurable.position_setpoint_task_reconfigurable import (
    PositionSetpointTaskReconfigurable,
)

from aerial_gym.config.task_config.position_setpoint_task_config_reconfigurable import (
    task_config as position_setpoint_task_config_reconfigurable,
)

from aerial_gym.task.position_setpoint_task_morphy.position_setpoint_task_morphy import (
    PositionSetpointTaskMorphy,
)

from aerial_gym.config.task_config.position_setpoint_task_morphy_config import (
    task_config as position_setpoint_task_config_morphy,
)


task_registry.register_task(
    "position_setpoint_task_reconfigurable",
    PositionSetpointTaskReconfigurable,
    position_setpoint_task_config_reconfigurable,
)

task_registry.register_task(
    "position_setpoint_task_morphy",
    PositionSetpointTaskMorphy,
    position_setpoint_task_config_morphy,
)


## Uncomment this to use custom tasks

# from aerial_gym.task.custom_task.custom_task import CustomTask
# task_registry.register_task("custom_task", CustomTask, custom_task.task_config)
