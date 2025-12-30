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
        
        # 4. 물체-파트 매핑
        self.object_to_part_mapping = {
            'magnifier': 'mag_body',
            'pcb': 'pcb',
            'flux': 'flux',
            'pump': 'pump',
        }
        
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
        self.current_mode = "IDLE"  # IDLE, OBJECT_DETECT, POINTING_DETECT, HOLD_POINT
        self.target_object = None
        self.target_part = None
        self.hit_buffer = deque(maxlen=5)
        self.last_published = None
        
        # 6. 메인 루프
        self.create_timer(1/15.0, self.main_loop)
        
        self.get_logger().info("✓ Command Vision Node 준비 완료")
    
    def command_callback(self, msg: String):
        command = msg.data.strip().lower()
        self.get_logger().info(f"명령 수신: '{command}'")
        self._parse_command(command)
    
    def _parse_command(self, command: str):
        # ✅ 1. "여기 잡아줘" - 정확한 포인팅 위치
        if any(word in command for word in ['여기', '잡아', 'hold', 'grab here', 'here']):
            self.current_mode = "HOLD_POINT"
            self.target_object = None
            self.target_part = None
            self.hit_buffer.clear()
            self.get_logger().info("모드: 정확한 위치 잡기 (광선-물체 교점)")
            self._publish_status("손가락으로 정확한 위치를 가리켜주세요")
            return
        
        # 2. 특정 객체 검출
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
                self.target_part = self.object_to_part_mapping.get(obj_name, obj_name)
                
                if self.target_object != self.target_part:
                    self.get_logger().info(
                        f"모드: 객체 검출 - '{self.target_object}' 요청 → '{self.target_part}' 찾기"
                    )
                else:
                    self.get_logger().info(f"모드: 객체 검출 - '{self.target_object}' 찾기")
                
                self._publish_status(f"'{self.target_object}' 검색 중...")
                return
        
        # 3. 손가락 가리킴 (물체 중심)
        if command == "track_hand":
            self.current_mode = "POINTING_DETECT"
            self.target_object = None
            self.target_part = None
            self.hit_buffer.clear()
            self.get_logger().info("모드: 손가락 가리킴 검출 (물체 중심)")
            self._publish_status("손가락으로 가리켜주세요")
            return
        
        # 4. 중지
        if any(word in command for word in ['중지', '취소', '멈춰', 'stop', 'cancel']):
            self.current_mode = "IDLE"
            self.target_object = None
            self.target_part = None
            self.hit_buffer.clear()
            self.get_logger().info("모드: IDLE")
            self._publish_status("대기 중")
            return
        
        self.get_logger().warn(f"인식할 수 없는 명령: '{command}'")
        self._publish_status("명령을 인식할 수 없습니다")
    
    def main_loop(self):
        color_img = self.img_node.get_color_frame()
        depth_img = self.img_node.get_depth_frame()
        
        if self.intrinsics is None:
            self.intrinsics = self.img_node.get_camera_intrinsic()
            if self.intrinsics is None:
                return
        
        if color_img is None or depth_img is None:
            return
        
        if self.current_mode == "OBJECT_DETECT":
            self._process_object_detection(color_img, depth_img)
            
        elif self.current_mode == "POINTING_DETECT":
            self._process_pointing_detection(color_img, depth_img)
        
        elif self.current_mode == "HOLD_POINT":
            self._process_hold_point(color_img, depth_img)
    
    def _calculate_ray_object_intersection(self, hand_landmarks, detection, depth_img):
        """
        ✅ 손가락 광선과 물체의 교점 계산
        
        Args:
            hand_landmarks: MediaPipe 손 랜드마크
            detection: YOLO 검출 결과 (바운딩 박스)
            depth_img: Depth 이미지
        
        Returns:
            intersection_3d: 교점의 3D 좌표 (카메라 좌표계)
            intersection_2d: 교점의 2D 픽셀 좌표
        """
        
        # 1. 손가락 끝 (index finger tip) 위치
        h, w = depth_img.shape
        index_tip = hand_landmarks.landmark[8]  # INDEX_FINGER_TIP
        tip_x = int(index_tip.x * w)
        tip_y = int(index_tip.y * h)
        
        # 2. 손가락 방향 벡터 (MCP → TIP)
        index_mcp = hand_landmarks.landmark[5]  # INDEX_FINGER_MCP
        mcp_x = int(index_mcp.x * w)
        mcp_y = int(index_mcp.y * h)
        
        # 2D 방향 벡터
        direction_2d = np.array([tip_x - mcp_x, tip_y - mcp_y], dtype=float)
        direction_2d = direction_2d / np.linalg.norm(direction_2d)  # 정규화
        
        self.get_logger().info(f"  손가락 끝: ({tip_x}, {tip_y})")
        self.get_logger().info(f"  방향 벡터: {direction_2d}")
        
        # 3. 물체 바운딩 박스
        x1, y1, x2, y2 = detection.bbox
        
        # 4. 광선-박스 교점 찾기 (2D)
        # 손가락 끝에서 방향으로 광선을 쏘면서 바운딩 박스와의 교점 찾기
        max_distance = 1000  # 최대 검색 거리 (픽셀)
        
        for dist in range(0, max_distance, 2):  # 2픽셀씩 진행
            # 광선 위의 점
            ray_x = int(tip_x + direction_2d[0] * dist)
            ray_y = int(tip_y + direction_2d[1] * dist)
            
            # 이미지 범위 체크
            if ray_x < 0 or ray_x >= w or ray_y < 0 or ray_y >= h:
                break
            
            # 바운딩 박스 내부인지 확인
            if x1 <= ray_x <= x2 and y1 <= ray_y <= y2:
                # 교점 발견!
                self.get_logger().info(f"  ✓ 교점 발견: ({ray_x}, {ray_y}) at {dist}px")
                
                # 5. 교점의 Depth 값 추출
                z = float(depth_img[ray_y, ray_x]) * self.depth_scale
                
                if z > 0:
                    # 6. 3D 좌표 계산
                    x_3d = (ray_x - self.intrinsics['ppx']) * z / self.intrinsics['fx']
                    y_3d = (ray_y - self.intrinsics['ppy']) * z / self.intrinsics['fy']
                    intersection_3d = np.array([x_3d, y_3d, z])
                    
                    return intersection_3d, (ray_x, ray_y)
                else:
                    self.get_logger().warn(f"  유효하지 않은 Depth: {z}")
        
        # 교점을 찾지 못한 경우 - 물체 중심 사용 (fallback)
        self.get_logger().warn("  교점을 찾지 못함 - 물체 중심 사용")
        center_2d = detection.center_2d
        u, v = center_2d
        
        if 0 <= u < w and 0 <= v < h:
            z = float(depth_img[v, u]) * self.depth_scale
            if z > 0:
                x_3d = (u - self.intrinsics['ppx']) * z / self.intrinsics['fx']
                y_3d = (v - self.intrinsics['ppy']) * z / self.intrinsics['fy']
                return np.array([x_3d, y_3d, z]), (u, v)
        
        return None, None
    
    def _process_hold_point(self, color_img, depth_img):
        """✅ '여기 잡아줘' 모드 - 정확한 포인팅 위치"""
        
        # 1. 물체 검출
        detections = self.object_detector.detect(color_img)
        
        if len(detections) == 0:
            self.get_logger().debug("물체가 감지되지 않음")
            self.hit_buffer.clear()
            return
        
        # 2. 손 검출
        hand_landmarks = self.hand_detector.detect(color_img)
        
        if hand_landmarks is None:
            self.get_logger().debug("손이 감지되지 않음")
            self.hit_buffer.clear()
            return
        
        # 3. 3D 정보 업데이트
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
        
        # 4. 가리킨 물체 찾기 (기존 방식)
        pointed_object = self.pointing_analyzer.find_pointed_object(
            hand_landmarks,
            detections
        )
        
        if pointed_object:
            self.get_logger().info(f"👉 가리킨 물체: {pointed_object.class_name}")
            
            # ✅ 5. 광선-물체 교점 계산
            intersection_3d, intersection_2d = self._calculate_ray_object_intersection(
                hand_landmarks,
                pointed_object,
                depth_img
            )
            
            if intersection_3d is not None:
                self.get_logger().info(f"  교점 3D: {intersection_3d}")
                
                # 버퍼에 추가 (연속성 체크용)
                buffer_key = f"{pointed_object.class_name}:{intersection_2d[0]},{intersection_2d[1]}"
                self.hit_buffer.append(buffer_key)
                
                # 5회 연속 같은 영역을 가리킴
                if len(self.hit_buffer) >= 5:
                    # 같은 물체를 가리키고 있는지만 확인 (위치는 약간 달라도 됨)
                    pointed_classes = [key.split(':')[0] for key in self.hit_buffer]
                    
                    if len(set(pointed_classes)) == 1:  # 같은 물체
                        self.get_logger().info(f"✅ 5회 연속 감지: {pointed_object.class_name}")
                        
                        # 베이스 좌표로 변환 및 발행
                        base_coords = self._camera_to_base(intersection_3d)
                        self._publish_pose(base_coords, f"{pointed_object.class_name}_at_point")
                        
                        self.last_published = pointed_object.class_name
                        self.current_mode = "IDLE"
                        self._publish_status("대기 중")
                        self.hit_buffer.clear()
        else:
            self.hit_buffer.clear()
    
    def _find_associated_part(self, detections, main_object_detection, part_name):
        """메인 물체 근처에서 파트 찾기"""
        main_center = main_object_detection.center_2d
        main_x, main_y = main_center
        
        candidates = []
        for det in detections:
            if part_name.lower() in det.class_name.lower():
                part_center = det.center_2d
                part_x, part_y = part_center
                distance = np.sqrt((main_x - part_x)**2 + (main_y - part_y)**2)
                candidates.append((det, distance))
        
        if not candidates:
            return None
        
        candidates.sort(key=lambda x: x[1])
        closest_part, dist = candidates[0]
        
        self.get_logger().info(
            f"  연관 파트 발견: {closest_part.class_name} (거리: {dist:.1f}px)"
        )
        
        return closest_part
    
    def _process_object_detection(self, color_img, depth_img):
        """특정 객체 검출 모드"""
        
        detections = self.object_detector.detect(color_img)
        
        target_detection = None
        for det in detections:
            if self.target_part.lower() in det.class_name.lower():
                target_detection = det
                self.get_logger().info(f"✓ '{self.target_part}' 직접 발견!")
                break
        
        if target_detection is None and self.target_object != self.target_part:
            main_detection = None
            for det in detections:
                if self.target_object.lower() in det.class_name.lower():
                    main_detection = det
                    break
            
            if main_detection:
                self.get_logger().info(f"✓ '{self.target_object}' 발견! 연관 파트 찾는 중...")
                target_detection = self._find_associated_part(
                    detections, 
                    main_detection, 
                    self.target_part
                )
                
                if target_detection is None:
                    self.get_logger().warn(
                        f"'{self.target_object}'는 보이지만 '{self.target_part}'를 찾을 수 없습니다"
                    )
                    return
        
        if target_detection is None:
            self.get_logger().debug(f"'{self.target_part}' 찾는 중...")
            return
        
        self.get_logger().info(f"✓ 최종 목표: {target_detection.class_name}")
        
        center_2d = target_detection.center_2d
        u, v = center_2d
        
        if 0 <= u < depth_img.shape[1] and 0 <= v < depth_img.shape[0]:
            z = float(depth_img[v, u]) * self.depth_scale
            
            if z > 0:
                x = (u - self.intrinsics['ppx']) * z / self.intrinsics['fx']
                y = (v - self.intrinsics['ppy']) * z / self.intrinsics['fy']
                camera_coords = np.array([x, y, z])
                
                base_coords = self._camera_to_base(camera_coords)
                
                object_name = self.target_object if self.target_object else target_detection.class_name
                self._publish_pose(base_coords, object_name)
                
                self.current_mode = "IDLE"
                self._publish_status("대기 중")
    
    def _process_pointing_detection(self, color_img, depth_img):
        """손가락 가리킴 검출 모드 (물체 중심)"""
        
        detections = self.object_detector.detect(color_img)
        
        if len(detections) == 0:
            self.hit_buffer.clear()
            return
        
        hand_landmarks = self.hand_detector.detect(color_img)
        
        if hand_landmarks is None:
            self.hit_buffer.clear()
            return
        
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
        
        pointed_object = self.pointing_analyzer.find_pointed_object(
            hand_landmarks,
            detections
        )
        
        if pointed_object:
            self.get_logger().info(f"가리킴: {pointed_object.class_name}")
            self.hit_buffer.append(pointed_object.class_name)
            
            if (len(self.hit_buffer) >= 5 and
                len(set(self.hit_buffer)) == 1 and
                self.last_published != pointed_object.class_name):
                
                base_coords = self._camera_to_base(pointed_object.center_3d)
                self._publish_pose(base_coords, pointed_object.class_name)
                
                self.last_published = pointed_object.class_name
                self.current_mode = "IDLE"
                self._publish_status("대기 중")
        else:
            self.hit_buffer.clear()
    
    def _camera_to_base(self, cam_coords):
        from DSR_ROBOT2 import get_current_posx
        
        coord = np.append(cam_coords, 1)
        curr_pose = get_current_posx()[0]
        
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
    
    parser = argparse.ArgumentParser(description="Command-based Vision Node")
    parser.add_argument("--test", action="store_true", help="테스트 모드")
    parser.add_argument("--webcam", action="store_true", help="웹캠 사용")
    parser.add_argument("--model", type=str, default=None, help="YOLO 모델 경로")
    
    parsed = parser.parse_args()
    
    rclpy.init(args=args)
    
    print("\n" + "="*50)
    print("  Command-based Vision Node (Ray Intersection)")
    print("="*50)
    
    print("모드: 프로덕션 (실제 로봇)")
    print("="*50 + "\n")
    
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
    
    model_path = parsed.model if parsed.model else os.path.expanduser("~/FiXit_ws/src/fixi_project/models/result_5.pt")
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
    
    vision_node = CommandVisionNode(model_path, calib_path, img_node)
    
    executor = MultiThreadedExecutor()
    executor.add_node(vision_node)
    executor.add_node(img_node)
    
    print("\n구독: /vision/cmd")
    print("발행: /vision/target_pose, /vision/status")
    print("\n지원 명령:")
    print("  - '여기 잡아줘': 손가락 광선과 물체의 교점")
    print("  - 'track_hand': 물체 중심점")
    print("  - '돋보기 가져와': 특정 물체 검출\n")
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        print("\n종료")
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()