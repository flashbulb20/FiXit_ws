import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    pkg_name = 'fixi_project'

    # Vision 모델 경로 설정 (기본값)
    default_model_path = os.path.expanduser('~/FiXit_ws/src/fixi_project/models/result_5.pt')
    
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value=default_model_path,
        description='Path to the YOLO model file'
    )

    # 1. 로봇 컨트롤러 노드
    robot_node = Node(
        package=pkg_name,
        executable='fixi_robot',      # setup.py에 등록한 이름
        name='robot_listener',
        output='screen',
        emulate_tty=True
    )

    # 2. 비전 컨트롤러 노드
    vision_node = Node(
        package=pkg_name,
        executable='fixi_vision',     # setup.py에 등록한 이름
        name='command_vision_node',
        output='screen',
        arguments=['--model', LaunchConfiguration('model_path')]
    )

    # 3. 음성 컨트롤러 노드
    voice_node = Node(
        package=pkg_name,
        executable='fixi_voice',      # setup.py에 등록한 이름
        name='vision_pointing_voice_node',
        output='screen'
    )

    # 4. 메인 컨트롤러 노드
    main_node = Node(
        package=pkg_name,
        executable='fixi_main',       # setup.py에 등록한 이름
        name='main_controller',
        output='screen'
    )

    return LaunchDescription([
        model_path_arg,
        robot_node,
        vision_node,
        voice_node,
        main_node
    ])