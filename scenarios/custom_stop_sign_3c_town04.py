# custom_stop_sign_3c_town04.py
from gym.envs.registration import register
from macad_gym.carla.multi_env import MultiCarlaEnv


class StopSign3CarTown04Custom(MultiCarlaEnv):
    """
    3-car straight setup in Town04 for map generalization testing.
    """

    def __init__(self):
        # Spawn points for Town04 - straight road section
        SPAWN_POINTS = [
            [-490.0, -10.0, 0.28],   # car1
            [-470.0, -10.0, 0.28],   # car2
            [-450.0, -10.0, 0.28],   # car3
        ]

        starts = SPAWN_POINTS

        # Drive ~60m straight along the same road for each car
        ends = [
            [sp[0] + 60.0, sp[1], sp[2]]
            for sp in SPAWN_POINTS
        ]

        # Scenario config
        SCENARIO = {
            "max_steps": 800,
            "actors": {
                "car1": {
                    "start": starts[0],
                    "end": ends[0],
                    "spawn_point_idx": 0,
                },
                "car2": {
                    "start": starts[1],
                    "end": ends[1],
                    "spawn_point_idx": 1,
                },
                "car3": {
                    "start": starts[2],
                    "end": ends[2],
                    "spawn_point_idx": 2,
                },
            },
        }

        CONFIGS = {
            "scenarios": SCENARIO,

            "env": {
                "server_map": "/Game/Carla/Maps/Town04",
                "render": True,
                "render_x_res": 800,
                "render_y_res": 600,
                "x_res": 168,
                "y_res": 168,
                "framestack": 1,
                "discrete_actions": True,
                "squash_action_logits": False,
                "verbose": False,
                "use_depth_camera": False,
                "send_measurements": False,
                "enable_planner": True,
                "spectator_loc": [SPAWN_POINTS[1][0], SPAWN_POINTS[1][1] + 6.0, 9.0],
                "sync_server": True,
                "fixed_delta_seconds": 0.05,
            },

            "actors": {
                a: {
                    "type": "vehicle_4W",
                    "enable_planner": True,
                    "convert_images_to_video": False,
                    "early_terminate_on_collision": True,
                    "reward_function": "corl2017",
                    "manual_control": False,
                    "auto_control": False,
                    "camera_type": "rgb",
                    "collision_sensor": "on",
                    "lane_sensor": "on",
                    "log_images": False,
                    "log_measurements": False,
                    "render": True,
                    "x_res": 168,
                    "y_res": 168,
                    "use_depth_camera": False,
                    "send_measurements": False,
                }
                for a in ["car1", "car2", "car3"]
            },
        }

        super().__init__(CONFIGS)


# Register Gym ID for Town04 scenario
register(
    id="HomoNcomIndePOIntrxMASS3CTWN4-v0",
    entry_point="custom_stop_sign_3c_town04:StopSign3CarTown04Custom",
)

