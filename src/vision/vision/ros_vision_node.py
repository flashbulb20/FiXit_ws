import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation
import numpy as np
import json
import os
import tempfile
from collections import deque
import cv2

from vision.vision_detector import ObjectDetector, HandDetector, PointingAnalyzer


class DummyImgNode:    
    def __init__(self, use_webcam=True, camera_id=0):
        self.intrinsics = {
            'fx': 600.0, 'fy': 600.0,
            'ppx': 320.0, 'ppy': 240.0
        }
        self.depth_scale = 0.001
        
        if use_webcam:
            self.cap = cv2.VideoCapture(camera_id)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            print("✓ 웹캠 사용")
        else:
            import pyrealsense2 as rs
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            profile = self.pipeline.start(config)
            
            intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            self.intrinsics = {'fx': intr.fx, 'fy': intr.fy, 'ppx': intr.ppx, 'ppy': intr.ppy}
            self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
            self.align = rs.align(rs.stream.color)
            print("✓ RealSense 사용")
        
        self.use_webcam = use_webcam
    
    def get_camera_intrinsic(self):
        return self.intrinsics
    
    def get_color_frame(self):
        if self.use_webcam:
            ret, frame = self.cap.read()
            return frame if ret else None
        else:
            frames = self.pipeline.wait_for_frames()
            aligned = self.align.process(frames)
            return np.asanyarray(aligned.get_color_frame().get_data())
    
    def get_depth_frame(self):
        if self.use_webcam:
            return np.full((480, 640), 500, dtype=np.uint16)
        else:
            frames = self.pipeline.wait_for_frames()
            aligned = self.align.process(frames)
            return np.asanyarray(aligned.get_depth_frame().get_data())


class DummyRobot:
    """테스트용 더미 로봇"""
    @staticmethod
    def get_current_posx():
        return [[400.0, 0.0, 600.0, 0.0, 180.0, 0.0]]


class VisionNode(Node):
    """Vision ROS2 노드"""
    
    def __init__(self, model_path, npy_path, json_path, img_node):
        super().__init__('vision_node')
        
        # 카메라
        self.img_node = img_node
        self.intrinsics = self.img_node.get_camera_intrinsic()
        self.depth_scale = self.img_node.depth_scale
        
        # CV 모듈
        self.object_detector = ObjectDetector(model_path, conf_threshold=0.6)
        self.hand_detector = HandDetector()
        self.pointing_analyzer = PointingAnalyzer(distance_threshold=50.0)
        
        # 캘리브레이션
        self.gripper2cam = np.load(npy_path)
        with open(json_path, 'r') as f:
            data = json.load(f)
            self.fine_offset = np.array(data['poses'][0][:3]) if 'poses' in data else np.zeros(3)
        
        # Publisher
        qos = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        self.pub = self.create_publisher(PoseStamped, '/vision/target_pose', qos)
        
        # 상태
        self.hit_buffer = deque(maxlen=5)
        self.last_published = None
        
        # 타이머
        self.create_timer(1/15.0, self.loop)
        
        self.get_logger().info("✓ Vision 노드 초기화 완료")
    
    def get_robot_pose_matrix(self, x, y, z, rx, ry, rz):
        R = Rotation.from_euler("ZYZ", [rx, ry, rz], degrees=True).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [x, y, z]
        return T
    
    def camera_to_base(self, cam_coords, robot_pose):
        from DSR_ROBOT2 import get_current_posx
        
        coord = np.append(cam_coords, 1)
        curr_pose = get_current_posx()[0]
        base2gripper = self.get_robot_pose_matrix(*curr_pose)
        base_coord = (base2gripper @ self.gripper2cam @ coord)[:3]
        return base_coord + self.fine_offset
    
    def loop(self):
        # 프레임 획득
        color_img = self.img_node.get_color_frame()
        depth_img = self.img_node.get_depth_frame()
        
        if color_img is None or depth_img is None:
            return
        
        # 객체 검출
        detections = self.object_detector.detect(color_img)
        
        if detections:
            names = [d.class_name for d in detections]
            self.get_logger().info(f"검출: {names}", throttle_duration_sec=2.0)
        
        # 손 검출
        hand = self.hand_detector.detect(color_img)
        if not hand:
            self.hit_buffer.clear()
            self.last_published = None
            return
        
        # 3D 정보 업데이트
        self.pointing_analyzer.update_3d_info(
            hand, detections, depth_img, self.intrinsics, self.depth_scale
        )
        
        # 가리키는 객체 찾기
        pointed = self.pointing_analyzer.find_pointed_object(hand, detections)
        
        if pointed:
            self.get_logger().info(f"가리킴: {pointed.class_name}", throttle_duration_sec=1.0)
            self.hit_buffer.append(pointed.class_name)
            
            # 5회 연속 같은 객체
            if (len(self.hit_buffer) >= 2 and 
                len(set(self.hit_buffer)) == 1 and
                self.last_published != pointed.class_name):
                
                base_coords = self.camera_to_base(pointed.center_3d, None)
                
                msg = PoseStamped()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = "base"
                msg.pose.position.x = float(base_coords[0])
                msg.pose.position.y = float(base_coords[1])
                msg.pose.position.z = float(base_coords[2])
                
                self.pub.publish(msg)
                self.get_logger().info(f"✓ 발행: {pointed.class_name} at {base_coords.round(1)}")
                self.last_published = pointed.class_name
        else:
            self.hit_buffer.clear()
            self.last_published = None


def main(args=None):
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="테스트 모드")
    parser.add_argument("--webcam", action="store_true", help="웹캠 사용")
    default_model = os.path.expanduser("~/FiXit_ws/src/vision/models/third_result.pt")
    parser.add_argument("--model", type=str, default=default_model, help="YOLO 모델 경로")
    
    # ROS 인자와 분리
    import sys
    custom_args = [arg for arg in (args or sys.argv[1:]) if arg.startswith('--')]
    parsed = parser.parse_args(custom_args)
    
    rclpy.init(args=args)
    
    # === 테스트 모드 ===
    if parsed.test:
        print("\n" + "="*50)
        print("  테스트 모드: 로봇 없이 독립 테스트")
        print("="*50 + "\n")
        
        # 더미 로봇 주입
        import sys
        sys.modules['DSR_ROBOT2'] = DummyRobot
        
        # 더미 캘리브레이션
        dummy_npy = tempfile.NamedTemporaryFile(suffix='.npy', delete=False)
        dummy_json = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        
        dummy_matrix = np.eye(4)
        dummy_matrix[:3, 3] = [50, 0, 100]
        np.save(dummy_npy.name, dummy_matrix)
        json.dump({'poses': [[10.0, 5.0, -5.0, 0, 0, 0]]}, dummy_json)
        
        dummy_npy.close()
        dummy_json.close()
        
        # 더미 카메라
        img_node = DummyImgNode(use_webcam=parsed.webcam)
        
        # 노드 실행
        node = VisionNode(parsed.model, dummy_npy.name, dummy_json.name, img_node)
        
        try:
            print("실행 중... (Ctrl+C로 종료)\n")
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            os.unlink(dummy_npy.name)
            os.unlink(dummy_json.name)
            rclpy.shutdown()
        
        return
    
    # === 프로덕션 모드 (미구현) ===
    print("프로덕션 모드는 --test 옵션을 사용하세요")
    rclpy.shutdown()


if __name__ == "__main__":
    main()