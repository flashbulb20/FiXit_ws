import cv2
import mediapipe as mp
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose  # Pose 메시지 추가
import DR_init
import time
import os

# 전역 변수로 선언 (main에서 임포트 후 할당)
posx = posj = movel = movej = get_current_posx = get_robot_state = set_tool = set_tcp = None

# 로봇 설정
ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
ROBOT_TOOL = "Tool Weight"
ROBOT_TCP = "GripperDA_v1"
VEL_APPROACH = 100
ACC = 100

class DoosanFingerPointingController(Node):
    def __init__(self, yolo_model):
        super().__init__('doosan_finger_pointing')
        
        # 1. 토픽 구독 설정 (/vision/target_pose)
        self.subscription = self.create_subscription(
            Pose,
            '/vision/target_pose',
            self.target_pose_callback,
            10
        )
        self.get_logger().info('Target-pose subscription complete')

        # MediaPipe & YOLO & RealSense 초기화 (기존 코드와 동일)
        self.init_sensors(yolo_model)
        
        # 로봇 초기화
        self.initialize_doosan_robot()
        
        # 캘리브레이션 및 설정
        self.camera_offset = np.array([500.0, 0.0, 300.0])
        self.last_move_time = time.time()
        self.move_cooldown = 3.0
        self.safe_height = 150.0

    def target_pose_callback(self, msg):
        """토픽을 통해 받은 좌표로 로봇 이동"""
        # 메시지에서 x, y, z 추출 (단위: mm 가정)
        # 만약 토픽 데이터가 미터(m) 단위라면 1000을 곱해야 합니다.
        target_3d = np.array([msg.position.x, msg.position.y, msg.position.z])
        
        self.get_logger().info(f'Received Topic Pose: x={target_3d[0]:.1f}, y={target_3d[1]:.1f}, z={target_3d[2]:.1f}')
        
        # 기존 이동 함수 호출 (Label은 'Topic_Target'으로 지정)
        self.move_robot_to_target(target_3d, "Topic_Target")

    def init_sensors(self, yolo_model):
        """센서 및 AI 모델 초기화 부분 분리"""
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.7)
        self.yolo = YOLO(yolo_model)
        
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        profile = self.pipeline.start(self.config)
        
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.fx, self.fy, self.cx, self.cy = intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy
        self.align = rs.align(rs.stream.color)

    def initialize_doosan_robot(self):
        try:
            set_tool(ROBOT_TOOL)
            set_tcp(ROBOT_TCP)
            movej(posj([0, 0, 90, 0, 90, 0]), vel=VEL_APPROACH, acc=ACC)
            self.get_logger().info('Robot Ready')
        except Exception as e:
            self.get_logger().error(f'Robot init failed: {e}')

    # ... [pixel_to_3d, camera_to_robot_frame, move_robot_to_target 등 기존 함수 유지] ...
    # (중복 방지를 위해 상세 로직은 기존 소스코드와 동일하게 유지한다고 가정합니다)

    def move_robot_to_target(self, target_3d, object_label):
        """기존 코드의 이동 로직 그대로 사용"""
        current_time = time.time()
        if current_time - self.last_move_time < self.move_cooldown:
            return False
        
        robot_pos = self.camera_to_robot_frame(target_3d)
        
        try:
            target_pose = posx([robot_pos[0], robot_pos[1], robot_pos[2], -180.0, 0.0, 0.0])
            safe_pose = posx([robot_pos[0], robot_pos[1], robot_pos[2] + self.safe_height, -180.0, 0.0, 0.0])
            
            movel(safe_pose, vel=VEL_APPROACH, acc=ACC)
            movel(target_pose, vel=VEL_APPROACH, acc=ACC)
            time.sleep(1.0)
            movel(safe_pose, vel=VEL_APPROACH, acc=ACC)
            
            self.last_move_time = current_time
            return True
        except Exception as e:
            self.get_logger().error(f'Move failed: {e}')
            return False

    def camera_to_robot_frame(self, camera_pos):
        # 기존 좌표 변환 로직
        robot_x = self.camera_offset[0] + camera_pos[2]
        robot_y = self.camera_offset[1] - camera_pos[0]
        robot_z = self.camera_offset[2] - camera_pos[1]
        return np.array([robot_x, robot_y, robot_z])

    def run(self):
        """메인 루프: 이미지 처리와 ROS2 스핀 병행"""
        try:
            while rclpy.ok():
                frames = self.pipeline.wait_for_frames()
                # ... [영상 처리 및 손가락 포인팅 로직 유지] ...
                
                # ROS2 콜백 처리를 위해 spin_once 실행
                rclpy.spin_once(self, timeout_sec=0.001)
        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()

def main(args=None):
    global posx, posj, movel, movej, get_current_posx, get_robot_state, set_tool, set_tcp
    
    rclpy.init(args=args)
    
    # 1. 노드 먼저 생성
    node = rclpy.create_node("doosan_finger_control", namespace=ROBOT_ID)
    DR_init.__dsr__node = node
    DR_init.__dsr__id = ROBOT_ID
    DR_init.__dsr__model = ROBOT_MODEL
    
    # 2. 노드 생성 후 로봇 라이브러리 임포트 (오류 방지 핵심)
    from DSR_ROBOT2 import (
        posx, posj, movel, movej, 
        get_current_posx, get_robot_state,
        set_tool, set_tcp
    )
    
    # 모델 경로 설정
    model_path = os.path.expanduser("~/FiXit_ws/src/vision/vision/third_result.pt")
    
    try:
        controller = DoosanFingerPointingController(model_path)
        controller.run()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        rclpy.shutdown()

if __name__ == "__main__":
    main()