from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'actions'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
	(os.path.join('share', 'actions', 'launch'), glob('launch/*.launch.py')),
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
		'goal_mover_action = actions.goal_mover_action:main',
		'goal_mover_simple = actions.goal_mover_simple:main',
		'follow_aruco = actions.follow_aruco:main',
		'follow_aruco_action = actions.follow_aruco_actionserver:main',
        'follow_aruco_launch = actions.follow_aruco_launch:main',
		'marker_amcl_corrector = actions.marker_amcl_corrector:main',
		'goal_mover_near_odom = actions.goal_mover_near_odom_approach:main',
		'goal_mover_obs_stop = actions.goal_mover_obs_stop:main',
		'nav2_obs_avoid = actions.nav2_obstacle_avoid:main',
		'goal_mover_obs_avoid = actions.goal_mover_obs_avoid:main',
		'goal_mover_obs_avoid_action = actions.goal_mover_obs_avoid_actionclient:main',
        'goal_mover_launch = actions.goal_mover_launch:main',
        'goal_mover_launch_visionavoid = actions.goal_mover_launch_visionavoid:main',
        ],
    },
)
