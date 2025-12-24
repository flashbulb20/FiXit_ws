from setuptools import find_packages, setup

package_name = 'voice_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/models', ['voice_control/hello_rokey_8332_32.tflite']),
        ('share/' + package_name, ['voice_control/.env']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mh',
    maintainer_email='mh@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'voice_cmd_node = voice_control.voice_cmd_node:main',
        ],
    },
)
