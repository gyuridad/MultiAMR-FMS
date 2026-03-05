from setuptools import find_packages, setup
import glob

package_name = 'controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/controller.launch.py']),
        ('share/' + package_name + '/launch', glob.glob('launch/*.launch.py')),
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
                'controller = controller.controller:main',
		        'controller_basic = controller.controller_basic:main',
                'controller_returntohome = controller.controller_returntohome:main',
                'controller_followaruco = controller.controller_followaruco:main',
                'controller_launch = controller.controller_launch:main',
                'controller_domainbridge = controller.controller_domainbridge:main',

        ],
    },
)
