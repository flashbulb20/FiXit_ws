import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation
import numpy as np
import json
import os
from collections import deque
from rclpy.executors import MultiThreadedExecutor

# AI 및 카메라 노드 임포트
from ultralytics import YOLO
import mediapipe as mp
from vision.realsense import ImgNode

import DR_init

# 로봇 설정
ROBOT_ID = "dsr01"

class FingerTargetPublisher(Node):
    def __init__(self, yolo_model, npy_path, json_path):
        super().__init__('finger_target_publisher')
        print("=== FingerTargetPublisher 초기화 시작 ===")

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

        # 2. ImgNode 초기화 (멀티스레드에서 동작하므로 spin_once 제거)
        self.img_node = ImgNode()
        self.intrinsics = self.img_node.get_camera_intrinsic()
        self.depth_scale = self.img_node.depth_scale if hasattr(self.img_node, 'depth_scale') else 0.001

        # 3. AI 모델 초기화
        self.yolo = YOLO(yolo_model)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

        # 4. 발행 설정
        pose_qos = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        self.pose_pub = self.create_publisher(PoseStamped, '/vision/target_pose', pose_qos)
        
        self.hit_buffer = deque(maxlen=5)
        self.published = False
        
        # 5. 타이머 설정 (루프 속도를 15fps 정도로 조절하여 연산 부하 감소)
        self.create_timer(1 / 15.0, self.main_loop)
        print("=== FingerTargetPublisher 초기화 완료 ===")

    def get_camera_pos(self, u, v, z):
        camera_x = (u - self.intrinsics["ppx"]) * z / self.intrinsics["fx"]
        camera_y = (v - self.intrinsics["ppy"]) * z / self.intrinsics["fy"]
        return np.array([camera_x, camera_y, z])

    def get_robot_pose_matrix(self, x, y, z, rx, ry, rz):
        R = Rotation.from_euler("ZYZ", [rx, ry, rz], degrees=True).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [x, y, z]
        return T

    def transform_to_base(self, camera_coords):
        from DSR_ROBOT2 import get_current_posx
        coord = np.append(np.array(camera_coords), 1)
        curr_pos = get_current_posx()[0]
        base2gripper = self.get_robot_pose_matrix(*curr_pos)
        base_coord = (base2gripper @ self.gripper2cam @ coord)[:3]
        return base_coord + self.fine_offset

    def main_loop(self):
        # 1. 프레임 획득 (이미지 노드가 별도 스레드에서 돌고 있으므로 바로 가져옴)
        color_img = self.img_node.get_color_frame()
        depth_img = self.img_node.get_depth_frame()

        if self.intrinsics is None:
            self.get_logger().warn("카메라 파라미터(Intrinsics) 수신 대기 중...", once=True)
            return
        
        if color_img is None or depth_img is None:
            print("이미지 수신 대기 중...")
            return

        print("=== 루프 동작 중 (이미지 획득 성공) ===")

        # 2. YOLO 및 MediaPipe 처리
        results = self.yolo(color_img, conf=0.6, verbose=False)
        detections = [{"bbox": box.xyxy[0].cpu().numpy().astype(int)} for box in results[0].boxes]
        res = self.hands.process(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))
        
        if res.multi_hand_landmarks:
            lms = res.multi_hand_landmarks[0]
            u_tip, v_tip = int(lms.landmark[8].x * 640), int(lms.landmark[8].y * 480)
            u_wrist, v_wrist = int(lms.landmark[0].x * 640), int(lms.landmark[0].y * 480)

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
        print(f"=== 좌표 발행 완료: {pos} ===")

def main(args=None):
    rclpy.init(args=args)
    
    # 로봇 노드 초기화
    node = rclpy.create_node("vision_interface_node", namespace=ROBOT_ID)
    DR_init.__dsr__node = node
    
    try:
        from DSR_ROBOT2 import get_current_posx
    except ImportError:
        return

    path = os.path.expanduser("~/FiXit_ws/src/vision/vision")
    vision_node = FingerTargetPublisher(
        os.path.join(path, "third_result.pt"),
        os.path.join(path, "T_gripper2camera.npy"),
        os.path.join(path, "calibrate_data.json")
    )

    # MultiThreadedExecutor 설정
    executor = MultiThreadedExecutor()
    executor.add_node(vision_node)
    executor.add_node(vision_node.img_node)

    print("=== MultiThreadedExecutor 시작 ===")
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()

if __name__ == "__main__":
    main()