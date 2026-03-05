from launch import LaunchDescription
from launch_ros.actions import Node

def make_static_tf(name, x, y, z, qx, qy, qz, qw, parent="map", child="aruco_600"):
    return Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name=name,
        arguments=[
            str(x), str(y), str(z),
            str(qx), str(qy), str(qz), str(qw),
            parent, child
        ],
        output="screen",
    )

def generate_launch_description():
    return LaunchDescription([
        make_static_tf(
            "aruco_600_static",
            0.056921, -0.021959, 0.247497,
            0.996459, 0.001276, 0.013589, -0.082960,
            parent="map", child="aruco_600"
        ),
        make_static_tf(
            "aruco_601_static",
            1.234000, -0.500000, 0.290000,
            0.0, 0.0, 0.0, 1.0,
            parent="map", child="aruco_601"
        ),
    ])
