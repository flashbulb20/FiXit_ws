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
from geometry_msgs.msg import Pose
import DR_init

# 로봇 ID 설정 (현재 위치를 받아오기 위해 필요)
ROBOT_ID = "dsr01"

class FinalTargetPublisher(Node):
    def __init__(self, yolo_model, npy_path, json_path):
        super().__init__('final_target_publisher')
        
        # 1. ROS2 Publisher 설정
        self.pose_pub = self.create_publisher(Pose, '/vision/target_pose', 10)
        
        # 2. 파일 로드 (Matrix & Offset)
        self.load_calibration_files(npy_path, json_path)
        
        # 3. AI 및 카메라 초기화
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.yolo = YOLO(yolo_model)
        self.init_camera()

        self.get_logger().info('Eye-in-Hand Target Publisher Online')

    def load_calibration_files(self, npy_path, json_path):
        # npy 로드
        try:
            self.gripper2cam = np.load(npy_path)
            self.get_logger().info(f"Loaded npy: {npy_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to load npy: {e}")
            exit()

        # json 로드
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                self.fine_offset = np.array(data.get('translation', [0.0, 0.0, 0.0]))
            self.get_logger().info(f"Loaded json offset: {self.fine_offset}")
        except:
            self.fine_offset = np.array([0.0, 0.0, 0.0])

    def init_camera(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        profile = self.pipeline.start(config)
        
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.intrinsics = {'fx': intr.fx, 'fy': intr.fy, 'ppx': intr.ppx, 'ppy': intr.ppy}
        self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        self.align = rs.align(rs.stream.color)

    def get_robot_pose_matrix(self, x, y, z, rx, ry, rz):
        R = Rotation.from_euler("ZYZ", [rx, ry, rz], degrees=True).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [x, y, z]
        return T

    def transform_to_base(self, camera_coords):
        """Camera -> Gripper -> Base 변환 + JSON 보정"""
        coord = np.append(np.array(camera_coords), 1)
        
        # 실시간 로봇 위치 획득
        try:
            curr_pos = get_current_posx()[0]
            base2gripper = self.get_robot_pose_matrix(*curr_pos)
            
            # 행렬 연산
            base2cam = base2gripper @ self.gripper2cam
            target_in_base = np.dot(base2cam, coord)[:3]
            
            # JSON 미세 보정 (mm 단위 가정)
            final_target = target_in_base + self.fine_offset
            return final_target
        except:
            return None

    def pixel_to_3d(self, u, v, depth_image):
        z = depth_image[v, u] * self.depth_scale * 1000.0 # mm
        if z == 0: return None
        x = (u - self.intrinsics['ppx']) * z / self.intrinsics['fx']
        y = (v - self.intrinsics['ppy']) * z / self.intrinsics['fy']
        return np.array([x, y, z])

    def run(self):
        try:
            while rclpy.ok():
                frames = self.pipeline.wait_for_frames()
                aligned = self.align.process(frames)
                color_img = np.asanyarray(aligned.get_color_frame().get_data())
                depth_img = np.asanyarray(aligned.get_depth_frame().get_data())

                results = self.yolo(color_img, conf=0.6, verbose=False)
                detections = []
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    detections.append({'bbox': (x1, y1, x2, y2)})

                res_hands = self.hands.process(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))
                if res_hands.multi_hand_landmarks:
                    lms = res_hands.multi_hand_landmarks[0]
                    p_tip = self.pixel_to_3d(int(lms.landmark[8].x*640), int(lms.landmark[8].y*480), depth_img)
                    p_wrist = self.pixel_to_3d(int(lms.landmark[0].x*640), int(lms.landmark[0].y*480), depth_img)

                    if p_tip is not None and p_wrist is not None:
                        ray = (p_tip - p_wrist) / np.linalg.norm(p_tip - p_wrist)
                        
                        for det in detections:
                            u, v = (det['bbox'][0] + det['bbox'][2]) // 2, (det['bbox'][1] + det['bbox'][3]) // 2
                            obj_3d = self.pixel_to_3d(u, v, depth_img)
                            
                            if obj_3d is not None:
                                dist = np.linalg.norm(np.cross(obj_3d - p_tip, ray))
                                
                                if dist < 100.0: # 10cm 오차 이내
                                    # 최종 로봇 베이스 좌표 계산
                                    final_base_target = self.transform_to_base(obj_3d)
                                    
                                    if final_base_target is not None:
                                        msg = Pose()
                                        msg.position.x = float(final_base_target[0])
                                        msg.position.y = float(final_base_target[1])
                                        msg.position.z = float(final_base_target[2])
                                        self.pose_pub.publish(msg)
                                        
                                        cv2.putText(color_img, "PUBLISHING TARGET", (10, 30), 1, 1, (0, 255, 0), 2)
                                        break

                cv2.imshow("Target Publisher (Eye-in-Hand)", color_img)
                if cv2.waitKey(1) & 0xFF == 27: break
                rclpy.spin_once(self, timeout_sec=0.001)
        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()

def main(args=None):
    global get_current_posx
    rclpy.init(args=args)
    
    # 로봇 위치를 가져오기 위한 노드 초기화
    node = rclpy.create_node("target_pub_node", namespace=ROBOT_ID)
    DR_init.__dsr__node = node
    from DSR_ROBOT2 import get_current_posx

    base_path = os.path.expanduser("~/FiXit_ws/src/vision/vision")
    controller = FinalTargetPublisher(
        yolo_model=os.path.join(base_path, "third_result.pt"),
        npy_path=os.path.join(base_path, "T_gripper2camera.npy"),
        json_path=os.path.join(base_path, "calibrate_data.json")
    )
    controller.run()
    rclpy.shutdown()

if __name__ == "__main__":
    main()