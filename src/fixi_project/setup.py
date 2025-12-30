from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'fixi_project'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'models'), 
            glob('models/*.pt')),
        (os.path.join('share', package_name, 'calibration'),
            glob('calibration/*.npy') + glob('calibration/*.json')),
        ('share/' + package_name + '/models', ['fixi_project/hello_rokey_8332_32.tflite']),
        ('share/' + package_name, ['fixi_project/.env']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='flashbulb',
    maintainer_email='ikjoo2000@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'fixi_main = fixi_project.main_controller:main',
            'fixi_robot = fixi_project.robot_controller:main',
            'fixi_vision = fixi_project.vision_controller:main',
            'fixi_voice = fixi_project.voice_controller:main'
        ],
    },
)
