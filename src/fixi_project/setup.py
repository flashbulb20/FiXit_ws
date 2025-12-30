from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'fixi_project'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/models', ['fixi_project/hello_rokey_8332_32.tflite']),
        ('share/' + package_name, ['fixi_project/.env']),
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
            "yolo_test = fixi_project.yolo_test:main",
            "move_robot = fixi_project.move_robot:main",
            "open_grip = fixi_project.open_grip:main",
            "close_grip = fixi_project.close_grip:main",
            "go_home = fixi_project.go_home:main",
            "detector = fixi_project.vision_detector:main",
            "vision_node = fixi_project.vision_node:main",
            "robot_node = fixi_project.robot_node:main",
            "main_controller = fixi_project.main_controller:main",
            "voice_node = fixi_project.voice_node:main",
        ],
    },
)