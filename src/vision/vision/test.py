import cv2
import mediapipe as mp
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
from scipy.spatial.transform import Rotation
import json
import os
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from collections import deque

import DR_init

ROBOT_ID = "dsr01"

class TargetOnlyPublisher(Node):
    def __init__(self, yolo_model, npy_path, json_path):
        super().__init__('target_only_publisher')
        
        # 1. ROS2 Publisher 설정
        from rclpy.qos import QoSProfile, QoSReliabilityPolicy
        pose_qos = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        self.pose_pub = self.create_publisher(PoseStamped, '/vision/target_pose', pose_qos)

        # 2. 파일 로드 (Matrix & Offset)
        self.gripper2cam = np.load(npy_path)
        with open(json_path, 'r') as f:
            self.fine_offset = np.array(json.load(f).get('translation', [0, 0, 0]))

        # 3. AI 및 카메라 초기화 (yolo_finger.py 기반)
        self.yolo = YOLO(yolo_model)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.init_camera()

        self.detections = []
        self.hit_buffer = deque(maxlen=5)
        self.published = False

    def init_camera(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        profile = self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.intrinsics = {'fx': intr.fx, 'fy': intr.fy, 'ppx': intr.ppx, 'ppy': intr.ppy}
        self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

    # --- 좌표 변환 및 보조 함수들 (기존 로직 유지) ---
    def pixel_to_3d(self, u, v, depth_img):
        if not (0 <= u < 640 and 0 <= v < 480): return None
        z = depth_img[v, u] * self.depth_scale * 1000.0
        if z <= 0: return None
        return np.array([(u - self.intrinsics['ppx']) * z / self.intrinsics['fx'],
                         (v - self.intrinsics['ppy']) * z / self.intrinsics['fy'], z])

    def transform_to_base(self, cam_point):
        # 전역 변수로 임포트된 get_current_posx 사용
        curr_pos = get_current_posx()[0] 
        R = Rotation.from_euler("ZYZ", curr_pos[3:], degrees=True).as_matrix()
        T = np.eye(4); T[:3,:3] = R; T[:3,3] = curr_pos[:3]
        base = (T @ self.gripper2cam @ np.append(cam_point, 1))[:3]
        return base + self.fine_offset

    def run(self):
        try:
            while rclpy.ok():
                frames = self.pipeline.wait_for_frames()
                aligned = self.align.process(frames)
                color_img = np.asanyarray(aligned.get_color_frame().get_data())
                depth_img = np.asanyarray(aligned.get_depth_frame().get_data())

                # 1. YOLO 탐지
                results = self.yolo(color_img, conf=0.6, verbose=False)
                self.detections = [{"bbox": b.xyxy[0].cpu().numpy().astype(int), "cls": int(b.cls[0])} for b in results[0].boxes]

                # 2. 손 감지 및 Pointing Ray 계산
                res = self.hands.process(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))
                
                pointed_object = None
                if res.multi_hand_landmarks:
                    lms = res.multi_hand_landmarks[0]
                    # 검지 끝(8)과 손목(0)
                    p_tip = self.pixel_to_3d(int(lms.landmark[8].x*640), int(lms.landmark[8].y*480), depth_img)
                    p_wrist = self.pixel_to_3d(int(lms.landmark[0].x*640), int(lms.landmark[0].y*480), depth_img)

                    if p_tip is not None and p_wrist is not None:
                        ray = (p_tip - p_wrist) / np.linalg.norm(p_tip - p_wrist)
                        
                        # 3. 가리킨 물체 찾기 (pointed_object 정의)
                        for det in self.detections:
                            x1, y1, x2, y2 = det["bbox"]
                            obj_3d = self.pixel_to_3d((x1+x2)//2, (y1+y2)//2, depth_img)
                            if obj_3d is None: continue
                            
                            v = obj_3d - p_tip
                            proj = np.dot(v, ray)
                            if proj > 0 and np.linalg.norm(v - proj * ray) < 30.0:
                                pointed_object = obj_3d
                                break

                # 4. 좌표 발행
                if pointed_object is not None and not self.published:
                    base_pos = self.transform_to_base(pointed_object)
                    msg = PoseStamped()
                    msg.header.stamp = self.get_clock().now().to_msg()
                    msg.header.frame_id = "base"
                    msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = map(float, base_pos)
                    self.pose_pub.publish(msg)
                    self.published = True
                    self.get_logger().info(f"Published Target: {base_pos}")

                if cv2.waitKey(1) & 0xFF == 27: break
                rclpy.spin_once(self, timeout_sec=0.001)
        finally:
            self.pipeline.stop()

def main(args=None):
    # 중요: 전역 함수로 사용하기 위해 global 선언
    global get_current_posx 
    
    rclpy.init(args=args)
    
    # 1. 임시 노드 생성 및 DR_init 설정
    temp_node = rclpy.create_node("temp_interface_node", namespace=ROBOT_ID)
    DR_init.__dsr__node = temp_node 
    
    # 2. 노드 설정 후에 라이브러리 임포트 (이 순서가 틀리면 AttributeError 발생)
    from DSR_ROBOT2 import get_current_posx 

    base_path = os.path.expanduser("~/FiXit_ws/src/vision/vision")
    vision_node = TargetOnlyPublisher(
        os.path.join(base_path, "third_result.pt"),
        os.path.join(base_path, "T_gripper2camera.npy"),
        os.path.join(base_path, "calibrate_data.json")
    )
    
    vision_node.run()
    rclpy.shutdown()

if __name__ == "__main__":
    main()