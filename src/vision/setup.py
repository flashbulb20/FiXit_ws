from setuptools import find_packages, setup

package_name = 'vision'

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
    maintainer='hyunj',
    maintainer_email='hyunj@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            "ros_yolo = vision.ros_yolo_finger:main",
            "ros_move = vision.ros_move:main",
            "open_grip = vision.open_grip:main",
            "close_grip = vision.close_grip:main",
            "go_home = vision.go_home:main"
        ],
    },
)
