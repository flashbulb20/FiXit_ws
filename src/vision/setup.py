from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'vision'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'models'), 
            glob('models/*.pt')),
        (os.path.join('share', package_name, 'calibration'),
            glob('calibration/*.npy') + glob('calibration/*.json')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hyunj',
    maintainer_email='hyunj@todo.todo',
    description='Vision package for robot manipulation',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "ros_yolo = vision.ros_yolo:main",
            "move_robot = vision.move_robot:main",
            "open_grip = vision.open_grip:main",
            "close_grip = vision.close_grip:main",
            "go_home = vision.go_home:main",
            "test_vision = vision.vision_detector:main",
            "vision_node = vision.ros_vision_node:main",
            "cmd_vision = vision.cmd_vision:main",
        ],
    },
)
