import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation
import numpy as np
import os
from collections import deque

from vision.vision_detector import ObjectDetector, HandDetector, PointingAnalyzer


class VisionNode(Node):    
    def __init__(self, model_path, npy_path, img_node):
        super().__init__('vision_node')
        
        # 카메라
        self.img_node = img_node
        self.intrinsics = self.img_node.get_camera_intrinsic()
        
        # RealSense depth는 이미 mm 단위 (depth_scale 사용 안 함)
        self.depth_scale = 1.0
        self.get_logger().info(f"✓ depth_scale: {self.depth_scale} (RealSense는 이미 mm 단위)")
        
        # CV 모듈
        self.object_detector = ObjectDetector(model_path, conf_threshold=0.6)
        self.hand_detector = HandDetector()
        self.pointing_analyzer = PointingAnalyzer(distance_threshold=50.0)
        
        # 캘리브레이션 (NPY만 사용)
        self.gripper2cam = np.load(npy_path)
        self.get_logger().info(f"✓ 캘리브레이션 로드: {npy_path}")
        
        # Publisher
        self.pub = self.create_publisher(PoseStamped, '/vision/target_pose', 10)
        
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
        return base_coord
    
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
            if (len(self.hit_buffer) >= 5 and 
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
    
    # 커맨드라인 인자 파싱
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="테스트 모드 (로봇 없이)")
    parser.add_argument("--webcam", action="store_true", help="웹캠 사용")
    parser.add_argument("--model", type=str, default=None, help="YOLO 모델 경로")
    parser.add_argument("--calib-npy", type=str, default=None, help="캘리브레이션 NPY")
    parser.add_argument("--calib-json", type=str, default=None, help="캘리브레이션 JSON")
    
    # sys.argv에서 직접 파싱
    parsed = parser.parse_args()
    
    # 모델 경로 자동 탐색
    if not parsed.model:
        possible_model = os.path.expanduser("~/FiXit_ws/src/vision/models/result_4.pt")
        if os.path.exists(possible_model):
            parsed.model = possible_model
            print(f"✓ 모델 자동 발견: {possible_model}")
    
    if not parsed.model or not os.path.exists(parsed.model):
        print("Error: 모델 파일을 찾을 수 없습니다")
        print("사용법: ros2 run vision ros_vision_node --model <경로>")
        return
    
    # ROS2 초기화 (인자 없이)
    rclpy.init()
    
    # === 프로덕션 모드 ===
    print("\n" + "="*50)
    print("  프로덕션 모드: 실제 로봇 + RealSense")
    print("="*50 + "\n")
    
    # 캘리브레이션 파일 경로
    if not parsed.calib_npy:
        possible_npy = os.path.expanduser("~/FiXit_ws/src/vision/calibration/T_gripper2camera.npy")
        if os.path.exists(possible_npy):
            parsed.calib_npy = possible_npy
    
    if not parsed.calib_json:
        possible_json = os.path.expanduser("~/FiXit_ws/src/vision/calibration/calibrate_data.json")
        if os.path.exists(possible_json):
            parsed.calib_json = possible_json
    
    if not parsed.calib_npy or not os.path.exists(parsed.calib_npy):
        print("Error: 캘리브레이션 NPY 파일을 찾을 수 없습니다")
        print("사용법: --calib-npy <경로>")
        rclpy.shutdown()
        return
    
    if not parsed.calib_json or not os.path.exists(parsed.calib_json):
        print("Error: 캘리브레이션 JSON 파일을 찾을 수 없습니다")
        print("사용법: --calib-json <경로>")
        rclpy.shutdown()
        return
    
    print(f"✓ 모델: {parsed.model}")
    print(f"✓ 캘리브레이션 NPY: {parsed.calib_npy}")
    print(f"✓ 캘리브레이션 JSON: {parsed.calib_json}")
    
    # 로봇 초기화
    try:
        import DR_init
        ROBOT_ID = "dsr01"
        
        robot_node = rclpy.create_node("vision_interface_node", namespace=ROBOT_ID)
        DR_init.__dsr__node = robot_node
        
        from DSR_ROBOT2 import get_current_posx
        print("✓ 로봇 연결 성공")
        
        # 현재 로봇 포즈 확인
        current_pose = get_current_posx()[0]
        print(f"✓ 현재 로봇 포즈: {current_pose}")
        
    except Exception as e:
        print(f"Error: 로봇 연결 실패 - {e}")
        print("로봇이 연결되어 있는지 확인하세요")
        rclpy.shutdown()
        return
    
    # RealSense 카메라 초기화
    try:
        from vision.realsense import ImgNode
        img_node = ImgNode()
        print("✓ RealSense 카메라 토픽 구독 시작")
        
        # 카메라 준비 대기 (토픽이 발행될 때까지)
        print("카메라 토픽 대기 중...")
        import time
        
        # ImgNode가 토픽을 받을 수 있도록 spin
        for i in range(10):  # 최대 5초 대기
            rclpy.spin_once(img_node, timeout_sec=0.5)
            
            if img_node.get_color_frame() is not None:
                print("✓ 카메라 프레임 수신 확인")
                break
            
            if i == 9:
                print("Warning: 카메라 프레임을 받을 수 없습니다")
                print("RealSense 토픽 확인: ros2 topic list | grep camera")
        
        # intrinsics 대기
        for i in range(10):
            rclpy.spin_once(img_node, timeout_sec=0.5)
            
            if img_node.get_camera_intrinsic() is not None:
                print("✓ 카메라 intrinsics 수신 확인")
                break
            
            if i == 9:
                print("Warning: 카메라 intrinsics를 받을 수 없습니다")
        
    except Exception as e:
        print(f"Error: RealSense 연결 실패 - {e}")
        print("RealSense가 연결되어 있는지 확인하세요")
        print("토픽 확인: ros2 topic list | grep camera")
        rclpy.shutdown()
        return
    
    # Vision 노드 실행
    try:
        from rclpy.executors import MultiThreadedExecutor
        
        vision_node = VisionNode(
            parsed.model,
            parsed.calib_npy,
            img_node
        )
        
        # MultiThreadedExecutor로 ImgNode와 VisionNode 동시 실행
        executor = MultiThreadedExecutor()
        executor.add_node(vision_node)
        executor.add_node(img_node)
        
        print("\n=== Vision 노드 실행 중 ===")
        print("손가락으로 물체를 가리키세요!")
        print("Ctrl+C로 종료\n")
        
        executor.spin()
        
    except KeyboardInterrupt:
        print("\n종료 중...")
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()