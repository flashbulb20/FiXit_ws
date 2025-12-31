import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation
import numpy as np
import os
from collections import deque
import cv2

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
        
        # ✅ 4. 물체별 Offset 설정 (픽셀 단위)
        # 돋보기는 이제 전체 실루엣이므로 손잡이 방향으로 Offset 적용
        self.object_offsets = {
            'magnifier': {
                'pixel': (0, 60),      # 아래쪽으로 60픽셀 (손잡이 방향)
                'description': '손잡이 위치',
                'enabled': True
            },
            'pcb': {
                'pixel': (0, 0),
                'description': '중앙',
                'enabled': False
            },
            'flux': {
                'pixel': (0, 0),
                'description': '중앙',
                'enabled': False
            },
            'pump': {
                'pixel': (0, 0),
                'description': '중앙',
                'enabled': False
            }
        }
        
        # 5. ROS2 Image Publisher (시각화용)
        self.bridge = CvBridge()
        self.debug_image_pub = self.create_publisher(Image, '/vision/debug_image', 10)
        
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
        
        # 6. 상태 관리
        self.current_mode = "IDLE"
        self.target_object = None
        self.hit_buffer = deque(maxlen=3)
        self.last_published = None
        
        # 7. 시각화 설정
        self.enable_visualization = True
        self.debug_frame = None
        
        # 8. 메인 루프
        self.create_timer(1/15.0, self.main_loop)
        
        self.get_logger().info("✓ Command Vision Node 준비 완료")
        self.get_logger().info("  시각화: /vision/debug_image 토픽")
        
        # Offset이 활성화된 물체 출력
        enabled_offsets = [obj for obj, cfg in self.object_offsets.items() if cfg['enabled']]
        if enabled_offsets:
            self.get_logger().info("  Offset 적용:")
            for obj in enabled_offsets:
                cfg = self.object_offsets[obj]
                self.get_logger().info(f"    - {obj}: {cfg['description']} {cfg['pixel']}")
    
    def _get_depth_robust(self, depth_img, center_2d, window_size=7):
        u, v = int(center_2d[0]), int(center_2d[1])
        h, w = depth_img.shape
        
        half_size = window_size // 2
        
        # 윈도우 범위
        u_min = max(0, u - half_size)
        u_max = min(w, u + half_size + 1)
        v_min = max(0, v - half_size)
        v_max = min(h, v + half_size + 1)
        
        # 윈도우 영역 추출
        window = depth_img[v_min:v_max, u_min:u_max]
        
        # 유효한 Depth만 추출 (0 제외)
        valid_depths = window[window > 0]
        
        if len(valid_depths) == 0:
            self.get_logger().warn(f"[DEPTH] 유효한 Depth 없음 at ({u}, {v})")
            return 0.0
        
        # 아웃라이어 제거 (IQR 방법)
        if len(valid_depths) > 3:
            q1 = np.percentile(valid_depths, 25)
            q3 = np.percentile(valid_depths, 75)
            iqr = q3 - q1
            
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # 범위 내 값만 사용
            filtered_depths = valid_depths[
                (valid_depths >= lower_bound) & (valid_depths <= upper_bound)
            ]
            
            if len(filtered_depths) > 0:
                valid_depths = filtered_depths
        
        # 평균 계산
        mean_depth = float(np.mean(valid_depths))
        
        # 디버그 정보
        single_pixel_depth = float(depth_img[v, u]) if depth_img[v, u] > 0 else 0.0
        diff = abs(mean_depth - single_pixel_depth) if single_pixel_depth > 0 else 0.0
        
        self.get_logger().debug(
            f"[DEPTH] 단일: {single_pixel_depth:.1f}mm, "
            f"평균({window_size}×{window_size}): {mean_depth:.1f}mm, "
            f"차이: {diff:.1f}mm"
        )
        
        return mean_depth
    
    def command_callback(self, msg: String):
        command = msg.data.strip().lower()
        self.get_logger().info(f"[CMD] 명령 수신: '{command}'")
        self._parse_command(command)
    
    def _parse_command(self, command: str):
        # 1. "여기 잡아줘"
        if any(word in command for word in ['여기', '잡아', 'hold', 'grab here', 'here']):
            self.current_mode = "HOLD_POINT"
            self.target_object = None
            self.hit_buffer.clear()
            self.last_published = None
            self.get_logger().info("[MODE] HOLD_POINT - 정확한 위치 잡기")
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
                self.last_published = None
                self.get_logger().info(f"[MODE] OBJECT_DETECT - '{self.target_object}' 찾기")
                self._publish_status(f"'{self.target_object}' 검색 중...")
                return
        
        # 3. 손가락 가리킴
        if command == "track_hand":
            self.current_mode = "POINTING_DETECT"
            self.target_object = None
            self.hit_buffer.clear()
            self.last_published = None
            self.get_logger().info("[MODE] POINTING_DETECT - 손가락 가리킴")
            self._publish_status("손가락으로 가리켜주세요")
            return
        
        # 4. 중지
        if any(word in command for word in ['중지', '취소', '멈춰', 'stop', 'cancel']):
            self.current_mode = "IDLE"
            self.target_object = None
            self.hit_buffer.clear()
            self.get_logger().info("[MODE] IDLE")
            self._publish_status("대기 중")
            return
        
        self.get_logger().warn(f"[CMD] 인식 불가: '{command}'")
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
        
        # 시각화용 프레임 복사
        if self.enable_visualization:
            self.debug_frame = color_img.copy()
        
        # 모드별 처리
        if self.current_mode == "OBJECT_DETECT":
            self._process_object_detection(color_img, depth_img)
            
        elif self.current_mode == "POINTING_DETECT":
            self._process_pointing_detection(color_img, depth_img)
        
        elif self.current_mode == "HOLD_POINT":
            self._process_hold_point(color_img, depth_img)
        
        # ROS2 이미지 발행
        if self.enable_visualization and self.debug_frame is not None:
            self._publish_debug_image()
    
    def _apply_offset(self, center_2d, obj_class_name):
        offset_config = self.object_offsets.get(obj_class_name.lower(), None)
        
        if offset_config is None or not offset_config['enabled']:
            return center_2d
        
        # Offset 적용
        x_offset, y_offset = offset_config['pixel']
        adjusted_u = int(center_2d[0] + x_offset)
        adjusted_v = int(center_2d[1] + y_offset)
        
        self.get_logger().info(
            f"  Offset 적용: {obj_class_name} - "
            f"원본({center_2d[0]}, {center_2d[1]}) → "
            f"조정({adjusted_u}, {adjusted_v}) "
            f"[{offset_config['description']}]"
        )
        
        return (adjusted_u, adjusted_v)
    
    def _draw_detections(self, detections):
        if self.debug_frame is None:
            return
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # 바운딩 박스
            cv2.rectangle(self.debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 클래스 이름
            label = f"{det.class_name} {det.confidence:.2f}"
            
            # 배경 박스
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(self.debug_frame, (x1, y1-label_h-10), (x1+label_w, y1), (0, 255, 0), -1)
            
            # 텍스트
            cv2.putText(self.debug_frame, label, (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # 원본 중심점
            cx, cy = det.center_2d
            cv2.circle(self.debug_frame, (int(cx), int(cy)), 5, (0, 255, 0), -1)
            
            # ✅ Offset 적용된 위치도 표시
            offset_config = self.object_offsets.get(det.class_name.lower(), None)
            if offset_config and offset_config['enabled']:
                adjusted_center = self._apply_offset(det.center_2d, det.class_name)
                cv2.circle(self.debug_frame, adjusted_center, 7, (255, 0, 255), 2)  # 마젠타 원
                cv2.line(self.debug_frame, (int(cx), int(cy)), adjusted_center, (255, 0, 255), 2)
                cv2.putText(self.debug_frame, "GRIP", (adjusted_center[0]+10, adjusted_center[1]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    
    def _draw_hand(self, hand_landmarks):
        if self.debug_frame is None or hand_landmarks is None:
            return
        
        h, w, _ = self.debug_frame.shape
        
        try:
            # ✅ fixi_project.vision_detector.HandLandmarks 구조
            # 속성: fingertip_2d, fingertip_3d, pointing_ray, wrist_2d, wrist_3d
            
            # 손가락 끝 (INDEX_FINGER_TIP)
            if hasattr(hand_landmarks, 'fingertip_2d'):
                fingertip = hand_landmarks.fingertip_2d
                
                # fingertip이 튜플/리스트인지 확인
                if isinstance(fingertip, (tuple, list)) and len(fingertip) >= 2:
                    tip_x, tip_y = int(fingertip[0]), int(fingertip[1])
                else:
                    self.get_logger().warn(f"fingertip_2d format unknown: {type(fingertip)}")
                    return
            else:
                self.get_logger().warn("HandLandmarks has no fingertip_2d attribute")
                return
            
            # 손목 (WRIST) - MCP 대신 사용
            if hasattr(hand_landmarks, 'wrist_2d'):
                wrist = hand_landmarks.wrist_2d
                
                if isinstance(wrist, (tuple, list)) and len(wrist) >= 2:
                    wrist_x, wrist_y = int(wrist[0]), int(wrist[1])
                else:
                    self.get_logger().warn(f"wrist_2d format unknown: {type(wrist)}")
                    return
            else:
                self.get_logger().warn("HandLandmarks has no wrist_2d attribute")
                return
            
            # 시각화
            cv2.circle(self.debug_frame, (tip_x, tip_y), 10, (255, 0, 0), -1)
            cv2.putText(self.debug_frame, "TIP", (tip_x+15, tip_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            cv2.circle(self.debug_frame, (wrist_x, wrist_y), 10, (0, 0, 255), -1)
            cv2.putText(self.debug_frame, "WRIST", (wrist_x+15, wrist_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            cv2.line(self.debug_frame, (wrist_x, wrist_y), (tip_x, tip_y), (255, 255, 0), 3)
            
            # ✅ 손 깊이 표시 (hand_depth)
            if hasattr(hand_landmarks, 'hand_depth') and hand_landmarks.hand_depth != 0.0:
                depth_text = f"depth: {hand_landmarks.hand_depth:.3f}"
                cv2.putText(self.debug_frame, depth_text, (wrist_x, wrist_y+30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(self.debug_frame, "(closer)", (wrist_x, wrist_y+50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # 광선 연장 - pointing_ray 속성 사용 가능하면 사용
            if hasattr(hand_landmarks, 'pointing_ray') and hand_landmarks.pointing_ray is not None:
                # pointing_ray를 직접 사용
                ray = hand_landmarks.pointing_ray
                if isinstance(ray, (tuple, list)) and len(ray) >= 2:
                    # 정규화된 방향 벡터
                    direction = np.array(ray[:2], dtype=float)
                    if np.linalg.norm(direction) > 0:
                        direction = direction / np.linalg.norm(direction)
                        end_x = int(tip_x + direction[0] * 300)
                        end_y = int(tip_y + direction[1] * 300)
                        cv2.arrowedLine(self.debug_frame, (tip_x, tip_y), (end_x, end_y), 
                                       (0, 255, 255), 3, tipLength=0.2)
            else:
                # pointing_ray가 없으면 fingertip - wrist로 계산
                direction = np.array([tip_x - wrist_x, tip_y - wrist_y], dtype=float)
                if np.linalg.norm(direction) > 0:
                    direction = direction / np.linalg.norm(direction)
                    end_x = int(tip_x + direction[0] * 300)
                    end_y = int(tip_y + direction[1] * 300)
                    cv2.arrowedLine(self.debug_frame, (tip_x, tip_y), (end_x, end_y), 
                                   (0, 255, 255), 3, tipLength=0.2)
                               
        except Exception as e:
            self.get_logger().error(f"손 시각화 실패: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
    
    def _draw_intersection(self, intersection_2d):
        if self.debug_frame is None or intersection_2d is None:
            return
        
        x, y = intersection_2d
        cv2.circle(self.debug_frame, (int(x), int(y)), 12, (0, 0, 255), 4)
        cv2.circle(self.debug_frame, (int(x), int(y)), 6, (255, 255, 255), -1)
        cv2.putText(self.debug_frame, "TARGET", (int(x)+20, int(y)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    def _draw_buffer_status(self):
        if self.debug_frame is None:
            return
        
        h, w, _ = self.debug_frame.shape
        
        # 반투명 배경
        overlay = self.debug_frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, self.debug_frame, 0.5, 0, self.debug_frame)
        
        # 모드 표시
        mode_color = {
            "IDLE": (128, 128, 128),
            "OBJECT_DETECT": (0, 255, 0),
            "POINTING_DETECT": (255, 255, 0),
            "HOLD_POINT": (255, 0, 255)
        }
        color = mode_color.get(self.current_mode, (255, 255, 255))
        
        cv2.putText(self.debug_frame, f"Mode: {self.current_mode}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # 타겟 표시
        if self.target_object:
            cv2.putText(self.debug_frame, f"Target: {self.target_object}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 버퍼 상태
        if len(self.hit_buffer) > 0:
            buffer_text = f"Buffer: {len(self.hit_buffer)}/3"
            cv2.putText(self.debug_frame, buffer_text, (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # 프로그레스 바
            bar_width = 200
            bar_height = 20
            filled = int(bar_width * len(self.hit_buffer) / 3)
            cv2.rectangle(self.debug_frame, (250, 75), (250+bar_width, 75+bar_height), 
                         (100, 100, 100), 2)
            cv2.rectangle(self.debug_frame, (250, 75), (250+filled, 75+bar_height), 
                         (0, 255, 0), -1)
    
    def _publish_debug_image(self):
        if self.debug_frame is None:
            return
        
        try:
            self._draw_buffer_status()
            img_msg = self.bridge.cv2_to_imgmsg(self.debug_frame, encoding="bgr8")
            img_msg.header.stamp = self.get_clock().now().to_msg()
            img_msg.header.frame_id = "camera"
            self.debug_image_pub.publish(img_msg)
        except Exception as e:
            self.get_logger().error(f"이미지 발행 실패: {e}")
    
    def _calculate_ray_object_intersection(self, hand_landmarks, detection, depth_img):
        h, w = depth_img.shape
        
        try:
            # 손가락 끝
            if not hasattr(hand_landmarks, 'fingertip_2d'):
                return None, None
            
            fingertip = hand_landmarks.fingertip_2d
            if not isinstance(fingertip, (tuple, list)) or len(fingertip) < 2:
                return None, None
            
            tip_x, tip_y = int(fingertip[0]), int(fingertip[1])
            
            # 손목 (방향 계산용)
            if not hasattr(hand_landmarks, 'wrist_2d'):
                return None, None
            
            wrist = hand_landmarks.wrist_2d
            if not isinstance(wrist, (tuple, list)) or len(wrist) < 2:
                return None, None
            
            wrist_x, wrist_y = int(wrist[0]), int(wrist[1])
            
            # 방향 벡터 계산
            direction_2d = np.array([tip_x - wrist_x, tip_y - wrist_y], dtype=float)
            if np.linalg.norm(direction_2d) == 0:
                return None, None
            direction_2d = direction_2d / np.linalg.norm(direction_2d)
            
            self.get_logger().info(f"  손가락: ({tip_x}, {tip_y}) → {direction_2d}")
            
            x1, y1, x2, y2 = detection.bbox
            max_distance = 1000
            
            for dist in range(0, max_distance, 2):
                ray_x = int(tip_x + direction_2d[0] * dist)
                ray_y = int(tip_y + direction_2d[1] * dist)
                
                if ray_x < 0 or ray_x >= w or ray_y < 0 or ray_y >= h:
                    break
                
                if x1 <= ray_x <= x2 and y1 <= ray_y <= y2:
                    self.get_logger().info(f"  ✓ 교점: ({ray_x}, {ray_y}) at {dist}px")
                    
                    z = float(depth_img[ray_y, ray_x]) * self.depth_scale
                    
                    if z > 0:
                        x_3d = (ray_x - self.intrinsics['ppx']) * z / self.intrinsics['fx']
                        y_3d = (ray_y - self.intrinsics['ppy']) * z / self.intrinsics['fy']
                        intersection_3d = np.array([x_3d, y_3d, z])
                        
                        self._draw_intersection((ray_x, ray_y))
                        
                        return intersection_3d, (ray_x, ray_y)
            
            self.get_logger().warn("  교점 못 찾음 - 중심 사용")
            center_2d = detection.center_2d
            u, v = center_2d
            
            if 0 <= u < w and 0 <= v < h:
                z = float(depth_img[v, u]) * self.depth_scale
                if z > 0:
                    x_3d = (u - self.intrinsics['ppx']) * z / self.intrinsics['fx']
                    y_3d = (v - self.intrinsics['ppy']) * z / self.intrinsics['fy']
                    return np.array([x_3d, y_3d, z]), (u, v)
            
            return None, None
            
        except Exception as e:
            self.get_logger().error(f"교점 계산 실패: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
            return None, None
    
    def _process_hold_point(self, color_img, depth_img):
        # ✅ 객체 검출 제거 - 손만 검출
        hand_landmarks = self.hand_detector.detect(color_img)
        
        # 손 깊이 정보 로그
        if hand_landmarks is not None and hand_landmarks.hand_depth != 0.0:
            self.get_logger().info(f"[HOLD] 손 선택: depth={hand_landmarks.hand_depth:.3f} (작을수록 앞)")
        
        self._draw_hand(hand_landmarks)
        
        if hand_landmarks is None:
            self.hit_buffer.clear()
            return
        
        # ✅ 손끝 2D 좌표
        fingertip_2d = hand_landmarks.fingertip_2d
        if not isinstance(fingertip_2d, (tuple, list)) or len(fingertip_2d) < 2:
            self.get_logger().warn("[HOLD] 손끝 좌표 형식 오류")
            self.hit_buffer.clear()
            return
        
        u, v = int(fingertip_2d[0]), int(fingertip_2d[1])
        h, w = depth_img.shape
        
        # 범위 체크
        if not (0 <= u < w and 0 <= v < h):
            self.get_logger().warn(f"[HOLD] 손끝 좌표 범위 초과: ({u}, {v})")
            self.hit_buffer.clear()
            return
        
        # ✅ 개선: 5×5 윈도우 평균 (손끝은 작은 영역)
        z = self._get_depth_robust(depth_img, (u, v), window_size=5) * self.depth_scale
        
        if z <= 0:
            self.get_logger().warn(f"[HOLD] 유효하지 않은 Depth: z={z}")
            self.hit_buffer.clear()
            return
        
        # 3D 좌표 계산
        x_3d = (u - self.intrinsics['ppx']) * z / self.intrinsics['fx']
        y_3d = (v - self.intrinsics['ppy']) * z / self.intrinsics['fy']
        fingertip_3d = np.array([x_3d, y_3d, z])
        
        self.get_logger().info(f"[HOLD] 손끝 위치: pixel=({u}, {v}), 3D={fingertip_3d.round(3)}")
        
        # ✅ 시각화: 손끝에 타겟 마커 표시
        self._draw_intersection((u, v))
        
        # ✅ 버퍼에 추가 (3D 좌표 저장)
        self.hit_buffer.append(fingertip_3d.copy())
        
        self.get_logger().info(f"[HOLD] 버퍼: {len(self.hit_buffer)}/5")
        
        # ✅ 5회 연속으로 안정적이면 발행
        if len(self.hit_buffer) >= 5:
            # 최근 5개 위치의 안정성 체크
            recent_positions = list(self.hit_buffer)[-5:]
            recent_array = np.array(recent_positions)  # shape: (5, 3)
            
            # 위치 변화량 계산 (표준편차)
            std_dev = np.std(recent_array, axis=0)  # [x_std, y_std, z_std]
            max_std = np.max(std_dev)  # 최대 표준편차
            
            # 안정성 기준: 최대 표준편차가 5mm 이하
            stability_threshold = 5.0  # mm
            
            if max_std < stability_threshold:
                # 안정적 → 평균 위치 사용
                avg_position = np.mean(recent_array, axis=0)
                
                self.get_logger().info(
                    f"[HOLD] ✅ 안정 (std: {max_std:.2f}mm < {stability_threshold}mm)"
                )
                self.get_logger().info(f"[HOLD] 평균 3D: {avg_position.round(3)}")
                
                # Base 좌표로 변환 (평균 위치 사용)
                base_coords = self._camera_to_base(avg_position)
                
                # 좌표 발행
                self._publish_pose(base_coords, "fingertip_position")
                
                self.current_mode = "IDLE"
                self._publish_status("손끝 위치로 이동")
                self.hit_buffer.clear()
                self.last_published = f"pos_{u}_{v}"
            else:
                # 불안정 → 계속 대기
                self.get_logger().info(
                    f"[HOLD] ⏳ 불안정 (std: {max_std:.2f}mm > {stability_threshold}mm)"
                )
    
    def _process_object_detection(self, color_img, depth_img):
        detections = self.object_detector.detect(color_img)
        self._draw_detections(detections)
        
        target_detection = None
        for det in detections:
            if self.target_object.lower() in det.class_name.lower():
                target_detection = det
                self.get_logger().info(f"[OBJ] ✓ '{self.target_object}' 발견")
                break
        
        if target_detection is None:
            self.get_logger().debug(f"[OBJ] '{self.target_object}' 찾는 중...")
            return
        
        self.get_logger().info(f"[OBJ] 최종: {target_detection.class_name}")
        
        # ✅ Offset 적용
        center_2d_original = target_detection.center_2d
        center_2d_adjusted = self._apply_offset(center_2d_original, target_detection.class_name)
        
        u, v = center_2d_adjusted
        
        if 0 <= u < depth_img.shape[1] and 0 <= v < depth_img.shape[0]:
            # ✅ 개선: 7×7 윈도우 평균 (아웃라이어 제거)
            z = self._get_depth_robust(depth_img, (u, v), window_size=7) * self.depth_scale
            
            if z > 0:
                x = (u - self.intrinsics['ppx']) * z / self.intrinsics['fx']
                y = (v - self.intrinsics['ppy']) * z / self.intrinsics['fy']
                camera_coords = np.array([x, y, z])
                
                self.get_logger().info(f"[OBJ] 3D 좌표: {camera_coords.round(3)}")
                
                base_coords = self._camera_to_base(camera_coords)
                self._publish_pose(base_coords, target_detection.class_name)
                
                self.current_mode = "IDLE"
                self._publish_status("대기 중")
            else:
                self.get_logger().warn(f"유효하지 않은 Depth: z={z}")
        else:
            self.get_logger().warn(f"좌표 범위 초과: u={u}, v={v}")
    
    def _process_pointing_detection(self, color_img, depth_img):
        detections = self.object_detector.detect(color_img)
        self._draw_detections(detections)
        
        if len(detections) == 0:
            self.hit_buffer.clear()
            return
        
        hand_landmarks = self.hand_detector.detect(color_img)
        
        # ✅ 손 깊이 정보 로그
        if hand_landmarks is not None and hand_landmarks.hand_depth != 0.0:
            self.get_logger().info(f"[POINT] 손 선택: depth={hand_landmarks.hand_depth:.3f} (작을수록 앞)")
        
        self._draw_hand(hand_landmarks)
        
        if hand_landmarks is None:
            self.hit_buffer.clear()
            return
        
        try:
            self.pointing_analyzer.update_3d_info(
                hand_landmarks, detections, depth_img,
                self.intrinsics, self.depth_scale
            )
        except Exception as e:
            self.get_logger().error(f"[POINT] 3D 업데이트 실패: {e}")
            return
        
        pointed_object = self.pointing_analyzer.find_pointed_object(
            hand_landmarks, detections
        )
        
        if pointed_object:
            self.get_logger().info(f"[POINT] 👉 {pointed_object.class_name}")
            
            # ✅ Offset 적용
            center_2d_adjusted = self._apply_offset(
                pointed_object.center_2d, 
                pointed_object.class_name
            )
            
            # 조정된 2D 좌표로 3D 재계산
            u, v = center_2d_adjusted
            if 0 <= u < depth_img.shape[1] and 0 <= v < depth_img.shape[0]:
                # ✅ 개선: 7×7 윈도우 평균 (객체 표면)
                z = self._get_depth_robust(depth_img, (u, v), window_size=7) * self.depth_scale
                if z > 0:
                    x = (u - self.intrinsics['ppx']) * z / self.intrinsics['fx']
                    y = (v - self.intrinsics['ppy']) * z / self.intrinsics['fy']
                    camera_coords_adjusted = np.array([x, y, z])
                    
                    self.get_logger().info(f"[POINT] 3D 좌표: {camera_coords_adjusted.round(3)}")
                    
                    self.hit_buffer.append(pointed_object.class_name)
                    
                    self.get_logger().info(f"[POINT] 버퍼: {len(self.hit_buffer)}/5")
                    
                    if (len(self.hit_buffer) >= 3 and
                        len(set(self.hit_buffer)) == 1):
                        
                        obj_name = pointed_object.class_name
                        
                        if self.last_published == obj_name:
                            self.get_logger().info(f"[POINT] 이미 발행: {obj_name}")
                            return
                        
                        self.get_logger().info(f"[POINT] ✅ 3회 연속: {obj_name}")
                        
                        base_coords = self._camera_to_base(camera_coords_adjusted)
                        self._publish_pose(base_coords, obj_name)
                        
                        self.last_published = obj_name
                        self.current_mode = "IDLE"
                        self._publish_status("대기 중")
                        self.hit_buffer.clear()
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
        
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"[PUBLISH] ✓ 좌표 발행!")
        self.get_logger().info(f"[PUBLISH]   Object: {object_name}")
        self.get_logger().info(f"[PUBLISH]   Position: {position.round(3)}")
        self.get_logger().info("=" * 60)
        
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
    print("  Command-based Vision Node (Simplified)")
    print("="*50)
    
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
    
    vision_node = CommandVisionNode(model_path, calib_path, img_node)
    
    executor = MultiThreadedExecutor()
    executor.add_node(vision_node)
    executor.add_node(img_node)
    
    print("\n구독: /vision/cmd")
    print("발행: /vision/target_pose, /vision/status, /vision/debug_image")
    print("\n시각화:")
    print("  rqt_image_view /vision/debug_image\n")
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        print("\n종료")
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()