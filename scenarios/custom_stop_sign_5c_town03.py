# custom_stop_sign_5c_town03.py
from gym.envs.registration import register
from macad_gym.carla.multi_env import MultiCarlaEnv


class StopSign5CarTown03Custom(MultiCarlaEnv):
    """
    5-car straight eastbound (lane-follow) setup in Town03.

    We only specify (x, y, z); yaw is taken from the road waypoint,
    exactly like in the working 3-car setup.
    """

    def __init__(self):
        # Spawn points chosen from your Town03 dump (same road).
        # IMPORTANT: no yaw here, so CARLA uses lane orientation.
        SPAWN_POINTS = [
            [129.13, 5.37, 0.28],   # car1
            [149.59, 5.37, 0.28],   # car2
            [166.38, 5.37, 0.28],   # car3
            [183.29, 5.37, 0.28],   # car4 (moved to same lane y)
            [201.25, 5.37, 0.28],   # car5 (moved to same lane y)
        ]

        starts = SPAWN_POINTS

        # Drive ~60m straight along the same road for each car.
        # Again, no yaw here; lane direction is taken from the map.
        ends = [
            [sp[0] + 60.0, sp[1], sp[2]]
            for sp in SPAWN_POINTS
        ]

        # Scenario config that MultiCarlaEnv._load_scenario expects:
        SCENARIO = {
            "max_steps": 800,
            "actors": {
                "car1": {
                    "start": starts[0],
                    "end": ends[0],
                    "spawn_point_idx": 19,   # kept for bookkeeping; spawn code ignores it
                },
                "car2": {
                    "start": starts[1],
                    "end": ends[1],
                    "spawn_point_idx": 17,
                },
                "car3": {
                    "start": starts[2],
                    "end": ends[2],
                    "spawn_point_idx": 185,
                },
                "car4": {
                    "start": starts[3],
                    "end": ends[3],
                    "spawn_point_idx": 23,
                },
                "car5": {
                    "start": starts[4],
                    "end": ends[4],
                    "spawn_point_idx": 21,
                },
            },
        }

        CONFIGS = {
            "scenarios": SCENARIO,

            "env": {
                "server_map": "/Game/Carla/Maps/Town03",
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
                # Spectator near the middle car (car3)
                "spectator_loc": [SPAWN_POINTS[2][0], SPAWN_POINTS[2][1] + 6.0, 9.0],
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
                for a in ["car1", "car2", "car3", "car4", "car5"]
            },
        }

        super().__init__(CONFIGS)


# Register a stable Gym ID for this 5-car Town03 scenario
register(
    id="HomoNcomIndePOIntrxMASS5CTWN3-v0",
    entry_point="custom_stop_sign_5c_town03:StopSign5CarTown03Custom",
)
