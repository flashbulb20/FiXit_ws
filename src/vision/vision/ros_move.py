import cv2
import mediapipe as mp
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
from scipy.spatial.transform import Rotation
import json
import os
import time

import rclpy
from rclpy.node import Node
import DR_init

# ==========================================
# 로봇 설정 (DSR-01, M0609)
# ==========================================
ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
VEL_APPROACH, ACC = 80, 80

class FinalEyeInHandController(Node):
    def __init__(self, yolo_model, npy_path, json_path):
        super().__init__('final_eye_in_hand_controller')
        
        # 1. 파일 로드 (둘 다 포함)
        self.load_files(npy_path, json_path)
        
        # 2. AI 모델 초기화
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.yolo = YOLO(yolo_model)
        
        # 3. 카메라 초기화
        self.init_camera()
        
        # 4. 로봇 초기화 및 상태 변수
        self.initialize_robot()
        self.last_move_time = time.time()
        self.move_cooldown = 5.0
        self.safe_height = 100.0  # mm (목표물 위쪽 대기 높이)

    def load_files(self, npy_path, json_path):
        """npy(행렬)와 json(오프셋)을 모두 로드"""
        # (1) npy 로드: Gripper to Camera 변환 행렬
        try:
            self.gripper2cam = np.load(npy_path)
            self.get_logger().info(f"Loaded Matrix: {npy_path}")
        except Exception as e:
            self.get_logger().error(f"NPY missing: {e}")
            exit()

        # (2) json 로드: 미세 보정용 Translation Offset
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                self.fine_offset = np.array(data.get('translation', [0.0, 0.0, 0.0]))
            self.get_logger().info(f"Loaded Offset: {self.fine_offset}")
        except Exception as e:
            self.get_logger().warn(f"JSON missing or error: {e}. Using zero offset.")
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

    def initialize_robot(self):
        try:
            # 시작 시 홈 포지션 이동
            movej(posj([0, 0, 90, 0, 90, 0]), vel=VEL_APPROACH, acc=ACC)
        except Exception as e:
            self.get_logger().error(f"Robot connection failed: {e}")

    def get_robot_pose_matrix(self, x, y, z, rx, ry, rz):
        """로봇 포즈(posx)를 4x4 행렬로 변환"""
        R = Rotation.from_euler("ZYZ", [rx, ry, rz], degrees=True).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [x, y, z]
        return T

    def transform_to_base(self, camera_coords):
        """npy 행렬 연산 + json 미세 보정 결합"""
        # 1. 카메라 3D 점 -> 동차 좌표
        coord = np.append(np.array(camera_coords), 1) 
        
        # 2. 현재 로봇 포즈 행렬 (Base -> Gripper)
        curr_pos = get_current_posx()[0]
        base2gripper = self.get_robot_pose_matrix(*curr_pos)
        
        # 3. 전체 행렬 연산 (Base -> Gripper -> Camera)
        base2cam = base2gripper @ self.gripper2cam
        target_in_base = np.dot(base2cam, coord)[:3]
        
        # 4. JSON의 미세 보정값 적용 (최종 좌표)
        final_target = target_in_base + self.fine_offset
        return final_target

    def pixel_to_3d(self, u, v, depth_image):
        z = depth_image[v, u] * self.depth_scale * 1000.0 # mm 단위
        if z == 0: return None
        x = (u - self.intrinsics['ppx']) * z / self.intrinsics['fx']
        y = (v - self.intrinsics['ppy']) * z / self.intrinsics['fy']
        return np.array([x, y, z])

    def move_robot_sequence(self, camera_pos):
        if time.time() - self.last_move_time < self.move_cooldown:
            return

        # 1. 최종 베이스 좌표 계산
        robot_coord = self.transform_to_base(camera_pos)
        
        try:
            curr_pos = get_current_posx()[0]
            # 안전 높이 및 목표 위치 설정
            safe_pose = posx([robot_coord[0], robot_coord[1], robot_coord[2] + self.safe_height, curr_pos[3], curr_pos[4], curr_pos[5]])
            target_pose = posx([robot_coord[0], robot_coord[1], robot_coord[2], curr_pos[3], curr_pos[4], curr_pos[5]])

            self.get_logger().info(f"Moving to Base: {robot_coord.round(2)}")
            movel(safe_pose, vel=VEL_APPROACH, acc=ACC)
            movel(target_pose, vel=VEL_APPROACH, acc=ACC)
            wait(1.5)
            movel(safe_pose, vel=VEL_APPROACH, acc=ACC)
            
            self.last_move_time = time.time()
        except Exception as e:
            self.get_logger().error(f"Robot control error: {e}")

    def run(self):
        try:
            while rclpy.ok():
                frames = self.pipeline.wait_for_frames()
                aligned = self.align.process(frames)
                color_img = np.asanyarray(aligned.get_color_frame().get_data())
                depth_img = np.asanyarray(aligned.get_depth_frame().get_data())

                # 1. YOLO 객체 검출
                results = self.yolo(color_img, conf=0.6, verbose=False)
                detections = []
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    detections.append({'bbox': (x1, y1, x2, y2)})

                # 2. 손가락 지시(Pointing) 확인
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
                                vec_to_obj = obj_3d - p_tip
                                dist = np.linalg.norm(np.cross(vec_to_obj, ray))
                                
                                # 지시 방향 10cm 이내면 타겟 확정
                                if dist < 100.0:
                                    cv2.rectangle(color_img, (det['bbox'][0], det['bbox'][1]), (det['bbox'][2], det['bbox'][3]), (0, 255, 0), 2)
                                    self.move_robot_sequence(obj_3d)
                                    break

                cv2.imshow("Final Eye-in-Hand Control", color_img)
                if cv2.waitKey(1) & 0xFF == 27: break
                rclpy.spin_once(self, timeout_sec=0.001)
        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()

def main(args=None):
    global posx, posj, movel, movej, get_current_posx, wait
    rclpy.init(args=args)
    
    node = rclpy.create_node("pointing_controller", namespace=ROBOT_ID)
    DR_init.__dsr__node = node
    
    # DSR 라이브러리 임포트
    from DSR_ROBOT2 import posx, posj, movel, movej, get_current_posx, wait
    from DR_common2 import posx, posj

    # 경로 설정
    base_p = os.path.expanduser("~/FiXit_ws/src/vision/vision")
    model_p = os.path.join(base_p, "third_result.pt")
    npy_p = os.path.join(base_p, "T_gripper2camera.npy")
    json_p = os.path.join(base_p, "calibrate_data.json")

    controller = FinalEyeInHandController(model_p, npy_p, json_p)
    controller.run()
    rclpy.shutdown()

if __name__ == "__main__":
    main()