from setuptools import find_packages, setup

package_name = 'sensors'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='pinky',
    maintainer_email='pinky@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
                'image_publisher = sensors.image_publisher:main',
                'aruco_publisher = sensors.aruco_vision:main',
                'static_transform_publisher = sensors.static_transform_publisher:main',
                'lidar_publisher = sensors.lidar_obs_detector:main',
		        'battery_publisher = sensors.battery_publisher:main',
		        'image_socket = sensors.image_socket_server:main',
		        'static_layer_dumper = sensors.static_layer_dumper:main',
                'map_wall_xy_near_robot = sensors.map_wall_xy_near_robot:main',
        ],
    },
)
