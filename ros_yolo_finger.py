import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation
import numpy as np
import json
import os
import time
from collections import deque

# AI 및 카메라 노드 임포트
from ultralytics import YOLO
import mediapipe as mp
from .realsense import ImgNode

import DR_init

# 로봇 설정
ROBOT_ID = "dsr01"

class FingerTargetPublisher(Node):
    def __init__(self, yolo_model, npy_path, json_path):
        super().__init__('finger_target_publisher')

        # 1. 캘리브레이션 및 오프셋 로드
        self.gripper2cam = np.load(npy_path)
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                if 'poses' in data and len(data['poses']) > 0:
                    self.fine_offset = np.array(data['poses'][0][:3]) 
                else:
                    self.fine_offset = np.array([0, 0, 0])
        except Exception as e:
            self.get_logger().warn(f"JSON load failed: {e}")
            self.fine_offset = np.zeros(3)

        # 2. ImgNode 초기화 및 카메라 파라미터 획득
        self.img_node = ImgNode()
        rclpy.spin_once(self.img_node, timeout_sec=1.0)
        self.intrinsics = self.img_node.get_camera_intrinsic()
        # ImgNode에서 정의된 depth_scale을 가져옵니다.
        self.depth_scale = self.img_node.depth_scale if hasattr(self.img_node, 'depth_scale') else 0.001

        # 3. AI 모델 초기화
        self.yolo = YOLO(yolo_model)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

        # 4. 발행 설정 및 상태 관리
        pose_qos = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        self.pose_pub = self.create_publisher(PoseStamped, '/vision/target_pose', pose_qos)
        
        self.hit_buffer = deque(maxlen=5) # 5회 연속 적중 확인용
        self.published = False
        
        # 5. 메인 루프 타이머 (30fps)
        self.create_timer(1 / 30.0, self.main_loop)
        self.get_logger().info("Finger-Object Tracker with ImgNode Started")

    def get_camera_pos(self, u, v, z):
        """픽셀 좌표와 Depth를 3D 카메라 좌표로 변환"""
        camera_x = (u - self.intrinsics["ppx"]) * z / self.intrinsics["fx"]
        camera_y = (v - self.intrinsics["ppy"]) * z / self.intrinsics["fy"]
        camera_z = z
        return np.array([camera_x, camera_y, camera_z])

    def get_robot_pose_matrix(self, x, y, z, rx, ry, rz):
        """로봇 포즈를 변환 행렬로 변환"""
        R = Rotation.from_euler("ZYZ", [rx, ry, rz], degrees=True).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [x, y, z]
        return T

    def transform_to_base(self, camera_coords):
        from DSR_ROBOT2 import get_current_posx
        
        # 1. 카메라 좌표 (mm)
        coord = np.append(np.array(camera_coords), 1)
        
        # 2. 로봇 현재 포즈 획득 및 변환 행렬 생성
        curr_pos = get_current_posx()[0]
        base2gripper = self.get_robot_pose_matrix(*curr_pos)
        
        # 3. Base 좌표 계산 (Base = Base2Gripper * Gripper2Cam * P_camera)
        base_coord = (base2gripper @ self.gripper2cam @ coord)[:3]
        
        # 4. JSON에서 가져온 추가 오프셋(fine_offset) 더하기
        return base_coord + self.fine_offset

    def main_loop(self):
        # 1. ImgNode로부터 최신 프레임 획득
        rclpy.spin_once(self.img_node, timeout_sec=0.01)
        color_img = self.img_node.get_color_frame()
        depth_img = self.img_node.get_depth_frame()

        if color_img is None or depth_img is None:
            return

        # 2. YOLO 객체 탐지
        results = self.yolo(color_img, conf=0.6, verbose=False)
        detections = [{"bbox": box.xyxy[0].cpu().numpy().astype(int)} for box in results[0].boxes]

        # 3. MediaPipe 손가락 인식
        res = self.hands.process(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))
        
        if res.multi_hand_landmarks:
            lms = res.multi_hand_landmarks[0]
            
            # 검지 끝(8)과 손목(0) 좌표 추출
            u_tip, v_tip = int(lms.landmark[8].x * 640), int(lms.landmark[8].y * 480)
            u_wrist, v_wrist = int(lms.landmark[0].x * 640), int(lms.landmark[0].y * 480)

            # depth 값 획득 및 3D 변환 (mm 단위)
            z_tip = depth_img[v_tip, u_tip] * self.depth_scale * 1000.0
            z_wrist = depth_img[v_wrist, u_wrist] * self.depth_scale * 1000.0
            
            if z_tip > 0 and z_wrist > 0:
                p_tip = self.get_camera_pos(u_tip, v_tip, z_tip)
                p_wrist = self.get_camera_pos(u_wrist, v_wrist, z_wrist)
                ray = (p_tip - p_wrist) / np.linalg.norm(p_tip - p_wrist)

                for det in detections:
                    x1, y1, x2, y2 = det["bbox"]
                    u_obj, v_obj = (x1 + x2) // 2, (y1 + y2) // 2
                    z_obj = depth_img[v_obj, u_obj] * self.depth_scale * 1000.0
                    
                    if z_obj > 0:
                        p_obj = self.get_camera_pos(u_obj, v_obj, z_obj)
                        v_vec = p_obj - p_tip
                        proj = np.dot(v_vec, ray)
                        
                        # 가리키는 방향에 있고 거리가 30mm 이내인 경우
                        if proj > 0 and np.linalg.norm(v_vec - proj * ray) < 30.0:
                            self.hit_buffer.append(True)
                            if len(self.hit_buffer) >= 5 and not self.published:
                                base_target = self.transform_to_base(p_obj)
                                self.publish_pose(base_target)
                                self.published = True
                            return
        
        self.hit_buffer.clear()
        self.published = False

    def publish_pose(self, pos):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base"
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = map(float, pos)
        self.pose_pub.publish(msg)
        self.get_logger().info(f"Published Target Base Coord: {pos.round(2)}")

def main(args=None):
    rclpy.init(args=args)
    
    # 1. 로봇 노드 초기화 및 라이브러리 연동 준비
    node = rclpy.create_node("vision_interface_node", namespace=ROBOT_ID)
    DR_init.__dsr__node = node
    
    # 2. DSR_ROBOT2 임포트 (DR_init 설정 후 수행)
    try:
        from DSR_ROBOT2 import get_current_posx
    except ImportError as e:
        print(f"Error importing DSR_ROBOT2: {e}")
        return

    # 3. 파일 경로 설정 및 노드 실행
    path = os.path.expanduser("~/FiXit_ws/src/fixi_project/fixi_project")
    vision_node = FingerTargetPublisher(
        os.path.join(path, "third_result.pt"),
        os.path.join(path, "T_gripper2camera.npy"),
        os.path.join(path, "calibrate_data.json")
    )
    
    try:
        rclpy.spin(vision_node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()

if __name__ == "__main__":
    main()