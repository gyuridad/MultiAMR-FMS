#!/usr/bin/env python3
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node

def generate_launch_description():
    # ✅ venv python 절대경로로 고정(가장 안정적)
    # - 만약 VIRTUAL_ENV를 항상 export 한다면 환경변수 기반으로 바꿔도 됨
    VENV_PY = "/home/addinedu/venv/pinky1542/bin/python3"

    MAP_TO_OBJ = "/home/addinedu/pinky_pro/install/pinky_vision/lib/pinky_vision/map_to_obj"
    AVOID_PUB  = "/home/addinedu/pinky_pro/install/pinky_vision/lib/pinky_vision/avoid_publisher"

    # ---------------------------
    # 1) traffic_manager
    # ---------------------------
    traffic_manager_node = Node(
        package="traffic_manager",
        executable="traffic_manager",
        name="traffic_manager",
        output="screen",
        emulate_tty=True,
    )

    # ---------------------------
    # 2) orchestrator
    # ---------------------------
    orchestrator_node = Node(
        package="orchestrator",
        executable="orchestrator",
        name="orchestrator",
        output="screen",
        emulate_tty=True,
    )

    # ---------------------------
    # 3) map_to_obj for pinky1
    # ---------------------------
    map_to_obj_pinky1 = ExecuteProcess(
        cmd=[
            VENV_PY, MAP_TO_OBJ,
            "--ros-args",
            "-p", "scale_s:=0.21",
            "-p", "robot_name:=pinky1",
            "-p", "tcp_host:=192.168.0.45",
            "-p", "tcp_port:=9001",
        ],
        output="screen",
        emulate_tty=True,
    )

    # ---------------------------
    # 4) map_to_obj for pinky2
    # ---------------------------
    map_to_obj_pinky2 = ExecuteProcess(
        cmd=[
            VENV_PY, MAP_TO_OBJ,
            "--ros-args",
            "-p", "scale_s:=0.21",
            "-p", "robot_name:=pinky2",
            "-p", "tcp_host:=192.168.0.42",
            "-p", "tcp_port:=9001",
        ],
        output="screen",
        emulate_tty=True,
    )

    # ---------------------------
    # 5) avoid_publisher
    #   ⚠️ 이것도 로봇별 토픽을 쓰면 로봇 수만큼 띄워야 함
    # ---------------------------
    avoid_pub_pinky1 = ExecuteProcess(
        cmd=[
            VENV_PY, AVOID_PUB,
            "--ros-args",
            "-p", "yolo_topic:=/pinky1/yolo_target_map",
            "-p", "walls_topic:=/pinky1/debug/walls_xy_json",
            "-p", "out_topic:=/pinky1/vision/avoid_dir_json",
            "-p", "trigger_dist_m:=0.8",
            "-p", "step_m:=0.35",
            "-p", "wall_clear_min_m:=0.25",
            "-p", "prefer_away_from_object_gain:=0.15",
        ],
        output="screen",
        emulate_tty=True,
    )

    avoid_pub_pinky2 = ExecuteProcess(
        cmd=[
            VENV_PY, AVOID_PUB,
            "--ros-args",
            "-p", "yolo_topic:=/pinky2/yolo_target_map",
            "-p", "walls_topic:=/pinky2/debug/walls_xy_json",
            "-p", "out_topic:=/pinky2/vision/avoid_dir_json",
            "-p", "trigger_dist_m:=0.8",
            "-p", "step_m:=0.35",
            "-p", "wall_clear_min_m:=0.25",
            "-p", "prefer_away_from_object_gain:=0.15",
        ],
        output="screen",
        emulate_tty=True,
    )

    return LaunchDescription([
        traffic_manager_node,
        orchestrator_node,

        map_to_obj_pinky1,
        map_to_obj_pinky2,

        avoid_pub_pinky1,
        avoid_pub_pinky2,
    ])


# ros2 launch orchestrator orchestrator.launch.py