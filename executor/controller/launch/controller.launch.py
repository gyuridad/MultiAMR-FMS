from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, PushRosNamespace

def generate_launch_description():
    ns = LaunchConfiguration("namespace")
    robot_name = LaunchConfiguration("robot_name")

    # ✅ 추가
    low_v_thr = LaunchConfiguration("battery_low_voltage_threshold")

    # ✅ FollowAruco용 (추가)
    flask_port = LaunchConfiguration("flask_port")
    flask_enable = LaunchConfiguration("flask_enable")

    # ✅ MapWall params (추가)
    wall_radius_m = LaunchConfiguration("wall_radius_m")
    wall_stride = LaunchConfiguration("wall_stride")
    wall_publish_period = LaunchConfiguration("wall_publish_period_sec")
    wall_occ_thresh = LaunchConfiguration("wall_occ_thresh")
    wall_max_points = LaunchConfiguration("wall_max_points")

    # ✅ BatteryBridge params (추가)
    battery_cmd = LaunchConfiguration("battery_cmd")
    battery_period = LaunchConfiguration("battery_period_sec")

    return LaunchDescription([
        DeclareLaunchArgument(
            "namespace",
            default_value="pinky1",
            description="ROS namespace (pinky1/pinky2/pinky3)"
        ),
        DeclareLaunchArgument(
            "robot_name",
            default_value=ns,
            description="Robot name for RobotState.robot_name (defaults to namespace)"
        ),

        # ✅ 추가: 기본값 7.5, 실행 시 덮어쓰기 가능
        DeclareLaunchArgument(
            "battery_low_voltage_threshold",
            default_value="7.5",
            description="RTB low voltage threshold (V)"
        ),

        # ✅ FollowAruco용 인자 추가
        DeclareLaunchArgument(
            "flask_enable",
            default_value="true",
            description="Enable Flask debug server for FollowAruco"
        ),
        DeclareLaunchArgument(
            "flask_port",
            default_value="5001",
            description="Flask port for FollowAruco"
        ),

        # ✅ MapWall 런치 인자(추가)
        DeclareLaunchArgument("wall_radius_m", default_value="2.0", description="MapWall local radius (m)"),
        DeclareLaunchArgument("wall_stride", default_value="2", description="MapWall stride sampling"),
        DeclareLaunchArgument("wall_publish_period_sec", default_value="0.5", description="MapWall publish period (sec)"),
        DeclareLaunchArgument("wall_occ_thresh", default_value="50", description="MapWall occupancy threshold"),
        DeclareLaunchArgument("wall_max_points", default_value="5000", description="MapWall max points"),

        # ✅ BatteryBridge 런치 인자(추가)
        DeclareLaunchArgument(
            "battery_cmd",
            default_value="/home/pinky/ap/check_battery_cli.py",
            description="Battery CLI script path"
        ),
        DeclareLaunchArgument(
            "battery_period_sec",
            default_value="1.0",
            description="Battery polling period (sec)"
        ),

        # ✅ 카메라 TCP 서버(단일 리소스) - namespace 밖 권장
        Node(
            package="sensors",
            executable="image_socket",
            name="image_socket",
            output="screen",
        ),

        GroupAction([
            PushRosNamespace(ns),

            # ✅ BatteryBridge 추가 (namespaced topics: /pinky1/battery_soc, /pinky1/battery_voltage, /pinky1/charger_connected)
            Node(
                package="sensors",                 # ⚠️ battery_bridge가 들어있는 패키지로 바꿔줘
                executable="battery_publisher",       # ⚠️ setup.py entry_point 이름으로 맞춰줘
                name="battery_publisher",
                output="screen",
                parameters=[{
                    "cmd": battery_cmd,
                    "period_sec": battery_period,
                    # 아래 토픽은 "상대토픽"으로 두는 게 제일 안전 (namespace 자동 적용)
                    "charger_topic": "charger_connected",
                    "soc_topic": "battery_soc",
                    "volt_topic": "battery_voltage",
                }],
            ),

            # ✅ (추가) MapWall XY Publisher (로봇 도메인에서 실행, namespaced 출력)
            Node(
                package="sensors",
                executable="map_wall_xy_near_robot",   # setup.py entry_point 이름
                name="map_wall_xy_near_robot",
                output="screen",
                parameters=[{
                    # 입력
                    "map_topic": "/map",                # 로봇 도메인의 /map 사용
                    "map_frame": "map",
                    "base_frame": "base_link",

                    # 출력 (namespaced 자동으로 /pinky1/debug/walls_xy_json 됨)
                    "json_topic": "debug/walls_xy_json",

                    # 튜닝
                    "radius_m": wall_radius_m,
                    "stride": wall_stride,
                    "publish_period_sec": wall_publish_period,
                    "occ_thresh": wall_occ_thresh,
                    "max_points": wall_max_points,
                }],
            ),

            # ✅ 1) PID Mover + VisionAvoid (MoveToPID ActionServer)
            Node(
                package="actions",
                executable="goal_mover_launch_visionavoid",  # ✅ 새 entry_point
                name="pidmover",
                output="screen",
                parameters=[{
                    # 기본
                    "cmd_topic": "cmd_vel",
                    "map_frame": "map",
                    "base_frame": "base_link",
                    "obstacle_topic": "obstacle_detected",
                    "action_name": "actions/move_to_pid",
                    "control_period_sec": 0.02,

                    # ✅ Vision Avoid params (네가 패치한 GoalMover 파라미터명과 일치해야 함)
                    "vision_avoid_topic": "vision/avoid_dir_json",  # namespaced: /pinky1/vision/avoid_dir_json
                    "vision_fresh_sec": 0.6,
                    "avoid_trigger_dist_m": 0.25,
                    "avoid_turn_deg": 45.0,

                    "avoid_go_dist_m": 0.18,
                    "avoid_go_tol_m": 0.02,
                    "avoid_go_speed": 0.12,
                    "avoid_go_max_sec": 2.0,

                    "avoid_stop_hold_sec": 0.15,
                    "avoid_cooldown_sec": 1.0,
                    "avoid_yaw_tol_deg": 6.0,
                }],
                remappings=[
                    ("cmd_vel", "/cmd_vel"),   # ✅ /pinky1/cmd_vel -> /cmd_vel
                ],
            ),

            # ✅ 2) FollowAruco(ActionServer) - odom topic 사용 (추가됨)
            Node(
                package="actions",
                executable="follow_aruco_launch",
                name="follow_aruco_server",
                output="screen",
                parameters=[{
                    "cmd_vel_topic": "cmd_vel",              # 상대 토픽 (namespace 자동)
                    "odom_topic": "odom",                    # odom 사용
                    "action_name": "actions/follow_aruco",   # 상대 액션명 (namespace 자동)

                    "tcp_host": "127.0.0.1",
                    "tcp_port": 9001,

                    "flask_enable": flask_enable,
                    "flask_port": flask_port,
                }],
                remappings=[
                    ("cmd_vel", "/cmd_vel"),  # ✅ 필요하면 유지(GoalMover와 동일 철학)
                    ("odom", "/odom"),
                ],
            ),

            # ✅ 2) Controller 
            Node(
                package="controller",
                executable="controller_domainbridge",  # <- 토픽 기반 entry_point 이름으로 맞추기
                name="controller",
                output="screen",
                parameters=[{
                    "robot_name": robot_name,

                    # namespaced topics
                    "state_topic": "robot_state",
                    "result_topic": "mission_result",
                    "battery_voltage_topic": "battery_voltage",

                    # 상대 토픽
                    # "mission_request_topic": "mission_request",   # 상대토픽 /pinky1/mission_request
                    # "mission_cancel_topic": "mission_cancel",

                    # GLOBAL topics for domain_bridge (절대 토픽으로 변경)
                    "mission_request_topic": "/traffic/mission_request",   
                    "mission_cancel_topic": "/traffic/mission_cancel",

                    # 내부 액션은 그대로
                    "move_to_pid_action_name": "actions/move_to_pid",
                    "follow_aruco_action_name": "actions/follow_aruco",

                    # battery / rtb
                    "battery_low_voltage_threshold": low_v_thr,
                    "battery_watch_period_sec": 1.0,
                    "battery_low_hold_sec": 2.0,

                    # waypoints
                    "waypoint_snap_max_dist": 0.5,
                    "wp_timeout_sec": 60.0,
                    "final_timeout_sec": 120.0,
                }],
            ),
        ]),
    ])
    
# 절대 토픽(/traffic/...)으로 둔 이유
#     “상위 제어는 하나의 채널로 전체 로봇에 방송”이 편함
#     System2(오케스트레이터)가 미션을 발행할 때,
#         한 토픽으로 다 보내고
#         메시지 안의 "robot": "pinky1" 필드로 수신 로봇이 필터링하게 하면 돼.

# # 실행 예시
# ros2 launch controller controller.launch.py battery_low_voltage_threshold:=7.8

# ros2 launch controller controller.launch.py battery_low_voltage_threshold:=7.8 flask_port:=5001
