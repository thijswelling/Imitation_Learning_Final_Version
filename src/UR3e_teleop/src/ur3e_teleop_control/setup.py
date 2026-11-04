from setuptools import setup
import os
from glob import glob

package_name = 'ur3e_teleop_control'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='thijs',
    maintainer_email='thijs@todo.todo',
    description='UR3e 6-DOF Teleoperation Control with 3D Systems Touch',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'task_space_controller = ur3e_teleop_control.task_space_controller:main',
            'robotiq_bridge = ur3e_teleop_control.robotiq_bridge:main',
            'touch_publisher = ur3e_teleop_control.touch_publisher:main',
            'trial_logger = ur3e_teleop_control.trial_logger:main',
            'inference_node = ur3e_teleop_control.inference_node:main',
        ],
    },
)
