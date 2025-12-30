import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from scipy.spatial.transform import Rotation
import numpy as np
import os
from collections import deque

from fixi_project.vision_detector import ObjectDetector, HandDetector, PointingAnalyzer

import DR_init

ROBOT_ID = "dsr01"


class CommandVisionNode(Node):
    def __init__(self, yolo_model_path: str, calibration_npy: str, img_node):
        super().__init__('command_vision_node')
        
        self.get_logger().info("=== Command Vision Node 초기화 ===")
        
        # 1. 카메라 설정
        self.img_node = img_node
        self.intrinsics = self.img_node.get_camera_intrinsic()
        self.depth_scale = 1.0
        
        # 2. CV 모듈 초기화
        self.object_detector = ObjectDetector(yolo_model_path, conf_threshold=0.6)
        self.hand_detector = HandDetector()
        self.pointing_analyzer = PointingAnalyzer(distance_threshold=50.0)
        
        # 3. 캘리브레이션
        self.gripper2cam = np.load(calibration_npy)
        self.get_logger().info(f"✓ 캘리브레이션 로드")
        
        # 구독: 명령 수신
        self.cmd_sub = self.create_subscription(
            String,
            '/vision/cmd',
            self.command_callback,
            10
        )
        
        # 발행: 검출 결과
        self.pose_pub = self.create_publisher(PoseStamped, '/vision/target_pose', 10)
        self.status_pub = self.create_publisher(String, '/vision/status', 10)
        
        # 5. 상태 관리
        self.current_mode = "IDLE"  # IDLE, OBJECT_DETECT, POINTING_DETECT
        self.target_object = None   # 찾을 객체 이름
        self.hit_buffer = deque(maxlen=5)
        self.last_published = None
        
        # 디버깅 카운터
        self.debug_counter = 0
        
        # 6. 메인 루프
        self.create_timer(1/15.0, self.main_loop)
        
        self.get_logger().info("✓ Command Vision Node 준비 완료")
        self.get_logger().info("  대기 중: /vision/cmd 토픽 구독")
    
    def command_callback(self, msg: String):
        command = msg.data.strip().lower()
        self.get_logger().info(f"명령 수신: '{command}'")
        
        # 명령 파싱
        self._parse_command(command)
    
    def _parse_command(self, command: str):        
        # 1. 특정 객체가 명시된 경우
        object_keywords = {
            'pcb': ['pcb', '기판'],
            'flux': ['flux', '플럭스'],
            'magnifier': ['magnifier', '돋보기'],
            'pump': ['pump', '흡입기']
        }
        
        for obj_name, keywords in object_keywords.items():
            if any(kw in command for kw in keywords):
                self.current_mode = "OBJECT_DETECT"
                self.target_object = obj_name
                self.get_logger().info(f"모드: 객체 검출 - '{self.target_object}' 찾기")
                self._publish_status(f"'{self.target_object}' 검색 중...")
                return
        
        # 2. 손가락 가리킴 모드
        if command == "track_hand":
            self.current_mode = "POINTING_DETECT"
            self.target_object = None
            self.hit_buffer.clear()
            self.debug_counter = 0
            self.get_logger().info("모드: 손가락 가리킴 검출")
            self.get_logger().info("🖐️ 손으로 물체를 가리켜주세요 (5회 연속 감지 필요)")
            self._publish_status("손가락으로 가리켜주세요")
            return
        
        # 3. 중지 명령
        if any(word in command for word in ['중지', '취소', '멈춰', 'stop', 'cancel']):
            self.current_mode = "IDLE"
            self.target_object = None
            self.hit_buffer.clear()
            self.get_logger().info("모드: IDLE")
            self._publish_status("대기 중")
            return
        
        # 인식 불가
        self.get_logger().warn(f"인식할 수 없는 명령: '{command}'")
        self._publish_status("명령을 인식할 수 없습니다")
    
    def main_loop(self):
        # 프레임 획득
        color_img = self.img_node.get_color_frame()
        depth_img = self.img_node.get_depth_frame()
        
        # Intrinsics 확인
        if self.intrinsics is None:
            self.intrinsics = self.img_node.get_camera_intrinsic()
            if self.intrinsics is None:
                return
        
        if color_img is None or depth_img is None:
            return
        
        # 모드별 처리
        if self.current_mode == "OBJECT_DETECT":
            self._process_object_detection(color_img, depth_img)
            
        elif self.current_mode == "POINTING_DETECT":
            self._process_pointing_detection(color_img, depth_img)
    
    def _process_object_detection(self, color_img, depth_img):
        # 객체 검출
        detections = self.object_detector.detect(color_img)
        
        # 타겟 객체 찾기
        target_detection = None
        for det in detections:
            if self.target_object.lower() in det.class_name.lower():
                target_detection = det
                break
        
        if target_detection is None:
            self.get_logger().debug(f"'{self.target_object}' 찾는 중...")
            return
        
        self.get_logger().info(f"✓ '{self.target_object}' 발견!")
        
        # 객체 중심의 3D 좌표 계산
        center_2d = target_detection.center_2d
        u, v = center_2d
        
        # Depth에서 3D 좌표 계산
        if 0 <= u < depth_img.shape[1] and 0 <= v < depth_img.shape[0]:
            z = float(depth_img[v, u]) * self.depth_scale
            
            if z > 0:
                x = (u - self.intrinsics['ppx']) * z / self.intrinsics['fx']
                y = (v - self.intrinsics['ppy']) * z / self.intrinsics['fy']
                camera_coords = np.array([x, y, z])
                
                # 베이스 좌표로 변환
                base_coords = self._camera_to_base(camera_coords)
                self._publish_pose(base_coords, target_detection.class_name)
                
                # 발행 후 IDLE로 전환
                self.current_mode = "IDLE"
                self._publish_status("대기 중")
    
    def _process_pointing_detection(self, color_img, depth_img):
        """손가락 가리킴 검출 모드 - 디버깅 강화"""
        
        # 디버깅: 3초마다 상태 출력
        self.debug_counter += 1
        if self.debug_counter % 45 == 0:  # 15fps * 3초
            self.get_logger().info(f"🔍 [Pointing Mode] 감지 대기 중... (버퍼: {len(self.hit_buffer)}/5)")
        
        # 객체 검출
        detections = self.object_detector.detect(color_img)
        
        if len(detections) == 0:
            if self.debug_counter % 45 == 0:
                self.get_logger().warn("⚠️ 물체가 감지되지 않습니다")
            self.hit_buffer.clear()
            return
        
        # 디버깅: 감지된 객체 출력
        if self.debug_counter % 45 == 0:
            obj_names = [d.class_name for d in detections]
            self.get_logger().info(f"✓ 감지된 물체: {obj_names}")
        
        # 손 검출
        hand_landmarks = self.hand_detector.detect(color_img)
        
        if hand_landmarks is None:
            if self.debug_counter % 45 == 0:
                self.get_logger().warn("⚠️ 손이 감지되지 않습니다")
            self.hit_buffer.clear()
            return
        
        # 디버깅: 손 감지됨
        if self.debug_counter % 45 == 0:
            self.get_logger().info("✓ 손 감지됨")
        
        # 3D 정보 업데이트
        try:
            self.pointing_analyzer.update_3d_info(
                hand_landmarks,
                detections,
                depth_img,
                self.intrinsics,
                self.depth_scale
            )
        except Exception as e:
            self.get_logger().error(f"3D 정보 업데이트 실패: {e}")
            return
        
        # 가리킨 객체 찾기
        pointed_object = self.pointing_analyzer.find_pointed_object(
            hand_landmarks,
            detections
        )
        
        if pointed_object:
            self.get_logger().info(f"👉 가리킴 감지: {pointed_object.class_name} (버퍼: {len(self.hit_buffer)+1}/5)")
            self.hit_buffer.append(pointed_object.class_name)
            
            # 버퍼 상태 표시
            if len(self.hit_buffer) >= 2:
                buffer_items = list(self.hit_buffer)
                self.get_logger().info(f"   버퍼 내용: {buffer_items}")
            
            # 5회 연속 같은 객체
            if (len(self.hit_buffer) >= 5 and
                len(set(self.hit_buffer)) == 1 and
                self.last_published != pointed_object.class_name):
                
                self.get_logger().info(f"✅ 5회 연속 감지 완료: {pointed_object.class_name}")
                
                # 베이스 좌표로 변환 및 발행
                base_coords = self._camera_to_base(pointed_object.center_3d)
                self._publish_pose(base_coords, pointed_object.class_name)
                
                self.last_published = pointed_object.class_name
                self.current_mode = "IDLE"
                self._publish_status("대기 중")
        else:
            # 가리킴 감지 안됨
            if len(self.hit_buffer) > 0:
                self.get_logger().info(f"❌ 가리킴 감지 실패 - 버퍼 초기화 (이전: {list(self.hit_buffer)})")
            self.hit_buffer.clear()
    
    def _camera_to_base(self, cam_coords):
        from DSR_ROBOT2 import get_current_posx
        
        coord = np.append(cam_coords, 1)
        curr_pose = get_current_posx()[0]
        
        # Base to Gripper 변환
        R = Rotation.from_euler("ZYZ", curr_pose[3:6], degrees=True).as_matrix()
        base2gripper = np.eye(4)
        base2gripper[:3, :3] = R
        base2gripper[:3, 3] = curr_pose[:3]
        
        base2cam = base2gripper @ self.gripper2cam
        base_coord = np.dot(base2cam, coord)[:3]
        
        return base_coord
    
    def _publish_pose(self, position, object_name):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base"
        msg.pose.position.x = float(position[0])
        msg.pose.position.y = float(position[1])
        msg.pose.position.z = float(position[2])
        
        # 자세는 기본값
        msg.pose.orientation.w = 1.0
        
        self.pose_pub.publish(msg)
        self.get_logger().info(f"✓ 발행: {object_name} at {position.round(1)}")
        self._publish_status(f"{object_name} 검출 완료")
    
    def _publish_status(self, message):
        msg = String()
        msg.data = message
        self.status_pub.publish(msg)


def main(args=None):
    import argparse
    
    # 인자 파싱
    parser = argparse.ArgumentParser(description="Command-based Vision Node")
    parser.add_argument("--test", action="store_true", help="테스트 모드 (로봇 없이)")
    parser.add_argument("--webcam", action="store_true", help="웹캠 사용")
    parser.add_argument("--model", type=str, default=None, help="YOLO 모델 경로")
    
    parsed = parser.parse_args()
    
    rclpy.init(args=args)
    
    print("\n" + "="*50)
    print("  Command-based Vision Node")
    print("="*50)
    
    # === 프로덕션 모드 ===
    print("모드: 프로덕션 (실제 로봇)")
    print("="*50 + "\n")
    
    # 로봇 초기화
    robot_node = rclpy.create_node("vision_interface_node", namespace=ROBOT_ID)
    DR_init.__dsr__node = robot_node
    
    try:
        from DSR_ROBOT2 import get_current_posx
        current = get_current_posx()[0]
        print(f"✓ 로봇: {current[:3]}")
    except Exception as e:
        print(f"Error: 로봇 실패 - {e}")
        rclpy.shutdown()
        return
    
    # 카메라 초기화
    try:
        from fixi_project.realsense import ImgNode
        img_node = ImgNode()
        print("✓ RealSense")
        
        for i in range(10):
            rclpy.spin_once(img_node, timeout_sec=0.5)
            if img_node.get_color_frame() is not None:
                print("✓ 카메라 OK")
                break
    except Exception as e:
        print(f"Error: 카메라 실패 - {e}")
        rclpy.shutdown()
        return
    
    # 파일 경로
    model_path = parsed.model if parsed.model else os.path.expanduser("~/FiXit_ws/src/fixi_project/models/result_4.pt")
    calib_path = os.path.expanduser("~/FiXit_ws/src/fixi_project/calibration/T_gripper2camera.npy")
    
    if not os.path.exists(model_path):
        print(f"Error: 모델 파일 없음 - {model_path}")
        rclpy.shutdown()
        return
    
    if not os.path.exists(calib_path):
        print(f"Error: 캘리브레이션 파일 없음 - {calib_path}")
        rclpy.shutdown()
        return
    
    print(f"✓ 모델: {model_path}")
    print(f"✓ 캘리브: {calib_path}")
    
    # Vision 노드
    vision_node = CommandVisionNode(model_path, calib_path, img_node)
    
    # Executor
    executor = MultiThreadedExecutor()
    executor.add_node(vision_node)
    executor.add_node(img_node)
    
    print("\n구독: /vision/cmd")
    print("발행: /vision/target_pose, /vision/status\n")
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        print("\n종료")
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()