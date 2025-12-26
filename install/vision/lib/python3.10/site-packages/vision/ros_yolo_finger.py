import cv2
import mediapipe as mp
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
from std_msgs.msg import String


class FingerPointingDetector(Node):
    def __init__(self, yolo_model):
        super().__init__('finger_pointing_detector')
        
        # ROS2 Publishers
        # 1. 가리킨 물체의 3D 좌표 발행 (PointStamped)
        self.point_pub = self.create_publisher(
            PointStamped, 
            '/vision/target_pose', 
            10
        )
        
        # 2. 가리킨 물체의 라벨 발행 (String)
        self.label_pub = self.create_publisher(
            String,
            '/vision/label',
            10
        )
        
        # 3. 전체 detection 정보 발행 (커스텀 메시지 형태로)
        self.detection_pub = self.create_publisher(
            String,
            '/vision/info',
            10
        )
        
        self.get_logger().info('Finger Pointing Detector Node Started')
        self.get_logger().info('Publishing to:')
        self.get_logger().info('  - /pointed_object/position (geometry_msgs/PointStamped)')
        self.get_logger().info('  - /pointed_object/label (std_msgs/String)')
        self.get_logger().info('  - /pointed_object/info (std_msgs/String)')
        
        # MediaPipe 초기화
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # YOLO 모델 초기화
        self.yolo = YOLO(yolo_model)
        self.get_logger().info(f'YOLO model loaded: {yolo_model}')
        
        # RealSense 카메라 초기화
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # RGB & Depth 스트림 설정
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        
        # 파이프라인 시작
        profile = self.pipeline.start(self.config)
        
        # Depth 센서 스케일 가져오기
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        
        # 카메라 intrinsics 가져오기
        color_stream = profile.get_stream(rs.stream.color)
        intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        self.fx = intrinsics.fx
        self.fy = intrinsics.fy
        self.cx = intrinsics.ppx
        self.cy = intrinsics.ppy
        
        # Align 객체 (depth를 color에 정렬)
        self.align = rs.align(rs.stream.color)
        
        self.get_logger().info('RealSense camera initialized')
        
    def pixel_to_3d(self, u, v, depth):
        """픽셀 좌표를 3D 좌표로 변환"""
        if depth == 0:
            return None
        
        # depth를 미터 단위로 변환
        Z = depth * self.depth_scale
        X = (u - self.cx) * Z / self.fx
        Y = (v - self.cy) * Z / self.fy
        
        return np.array([X, Y, Z])
    
    def get_stable_depth(self, depth_image, u, v, window_size=5):
        """주변 픽셀의 중앙값으로 안정적인 depth 값 얻기"""
        h, w = depth_image.shape
        half_w = window_size // 2
        
        u_min = max(0, u - half_w)
        u_max = min(w, u + half_w + 1)
        v_min = max(0, v - half_w)
        v_max = min(h, v + half_w + 1)
        
        window = depth_image[v_min:v_max, u_min:u_max]
        valid_depths = window[window > 0]
        
        if len(valid_depths) == 0:
            return 0
        
        return np.median(valid_depths)
    
    def get_pointing_ray(self, hand_landmarks, depth_image, img_w, img_h):
        """손목과 검지 끝으로 pointing ray 계산"""
        # 손목 (WRIST)
        wrist_lm = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
        uw = int(wrist_lm.x * img_w)
        vw = int(wrist_lm.y * img_h)
        
        # 검지 끝 (INDEX_FINGER_TIP)
        index_lm = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        ui = int(index_lm.x * img_w)
        vi = int(index_lm.y * img_h)
        
        # 안정적인 depth 값 가져오기
        Zw = self.get_stable_depth(depth_image, uw, vw)
        Zi = self.get_stable_depth(depth_image, ui, vi)
        
        if Zw == 0 or Zi == 0:
            return None, None, None, None
        
        # 3D 좌표 계산
        P_wrist = self.pixel_to_3d(uw, vw, Zw)
        P_index = self.pixel_to_3d(ui, vi, Zi)
        
        if P_wrist is None or P_index is None:
            return None, None, None, None
        
        # Ray 방향 벡터 (정규화)
        direction = P_index - P_wrist
        norm = np.linalg.norm(direction)
        
        if norm < 0.01:  # 너무 짧으면 무시
            return None, None, None, None
        
        direction = direction / norm
        
        return P_wrist, P_index, direction, (uw, vw, ui, vi)
    
    def bbox_to_3d(self, bbox, depth_image):
        """Bounding box의 중심점을 3D 좌표로 변환"""
        x1, y1, x2, y2 = bbox
        
        # Bounding box 중심점
        u = int((x1 + x2) / 2)
        v = int((y1 + y2) / 2)
        
        # 안정적인 depth 값
        depth = self.get_stable_depth(depth_image, u, v)
        
        if depth == 0:
            return None
        
        return self.pixel_to_3d(u, v, depth)
    
    def point_to_ray_distance(self, point_3d, ray_origin, ray_direction):
        """점과 ray 사이의 최단 거리 계산"""
        # point_3d: 물체의 3D 좌표
        # ray_origin: ray 시작점 (손목)
        # ray_direction: ray 방향 (정규화된 벡터)
        
        # 벡터: ray_origin에서 point까지
        v = point_3d - ray_origin
        
        # ray 방향으로의 투영 거리
        projection_length = np.dot(v, ray_direction)
        
        # ray 위의 가장 가까운 점
        closest_point_on_ray = ray_origin + projection_length * ray_direction
        
        # 최단 거리
        distance = np.linalg.norm(point_3d - closest_point_on_ray)
        
        return distance, projection_length
    
    def find_pointed_object(self, detections, depth_image, ray_origin, ray_direction):
        """
        YOLO detection 중에서 ray에 가장 가까운 물체 찾기
        
        Returns:
            dict: pointed object info or None
        """
        if ray_origin is None or ray_direction is None:
            return None
        
        min_distance = float('inf')
        pointed_object = None
        
        for detection in detections:
            bbox = detection['bbox']  # (x1, y1, x2, y2)
            label = detection['label']
            conf = detection['confidence']
            
            # Bounding box 중심의 3D 좌표
            obj_3d = self.bbox_to_3d(bbox, depth_image)
            
            if obj_3d is None:
                continue
            
            # Ray와의 거리 계산
            distance, projection = self.point_to_ray_distance(
                obj_3d, ray_origin, ray_direction
            )
            
            # ray 앞쪽에 있는 물체만 고려 (projection > 0)
            if projection > 0 and distance < min_distance:
                min_distance = distance
                pointed_object = {
                    'label': label,
                    'position_3d': obj_3d,
                    'bbox': bbox,
                    'confidence': conf,
                    'distance_to_ray': distance
                }
        
        return pointed_object
    
    def publish_pointed_object(self, pointed_object):
        """가리킨 물체 정보를 ROS2 topic으로 발행"""
        if pointed_object is None:
            return
        
        # 현재 시간
        current_time = self.get_clock().now().to_msg()
        
        # 1. 3D 좌표 발행 (PointStamped)
        point_msg = PointStamped()
        point_msg.header.stamp = current_time
        point_msg.header.frame_id = 'camera_color_optical_frame'  # RealSense 기본 프레임
        point_msg.point.x = float(pointed_object['position_3d'][0])
        point_msg.point.y = float(pointed_object['position_3d'][1])
        point_msg.point.z = float(pointed_object['position_3d'][2])
        self.point_pub.publish(point_msg)
        
        # 2. 라벨 발행
        label_msg = String()
        label_msg.data = pointed_object['label']
        self.label_pub.publish(label_msg)
        
        # 3. 전체 정보 발행 (JSON 형식)
        info_str = (
            f"label: {pointed_object['label']}, "
            f"x: {pointed_object['position_3d'][0]:.4f}, "
            f"y: {pointed_object['position_3d'][1]:.4f}, "
            f"z: {pointed_object['position_3d'][2]:.4f}, "
            f"confidence: {pointed_object['confidence']:.3f}, "
            f"ray_distance: {pointed_object['distance_to_ray']:.4f}"
        )
        info_msg = String()
        info_msg.data = info_str
        self.detection_pub.publish(info_msg)
        
        # 로그 출력
        self.get_logger().info(
            f"Published: {pointed_object['label']} at "
            f"({pointed_object['position_3d'][0]:.3f}, "
            f"{pointed_object['position_3d'][1]:.3f}, "
            f"{pointed_object['position_3d'][2]:.3f})"
        )
    
    def detect_objects(self, image, conf_threshold=0.7, iou_threshold=0.5):
        """YOLO로 객체 탐지"""
        results = self.yolo(
            image, 
            conf=conf_threshold,  # confidence threshold 높임
            classes=[0, 1, 2, 3], # others 제외
            iou=iou_threshold,    # NMS IOU threshold
            verbose=False,
            imgsz=640  # 이미지 크기 명시
        )
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = self.yolo.names[cls]

                area = (x2 - x1) * (y2 - y1)
                if area < 100:
                    continue
                
                detections.append({
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'label': label,
                    'confidence': conf
                })
        
        return detections
    
    def visualize(self, color_image, depth_colormap, hand_landmarks, 
                  pixel_coords, detections, pointed_object):
        """결과 시각화"""
        
        # 모든 detection 그리기
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            label = det['label']
            conf = det['confidence']
            
            # 일반 물체는 녹색으로
            color = (0, 255, 0)
            thickness = 2
            
            # 가리킨 물체는 노란색으로 강조
            if pointed_object and det['bbox'] == pointed_object['bbox']:
                color = (0, 255, 255)
                thickness = 3
            
            cv2.rectangle(color_image, (x1, y1), (x2, y2), color, thickness)
            
            # 라벨 표시
            text = f"{label}: {conf:.2f}"
            cv2.putText(color_image, text, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 손 랜드마크와 ray 그리기
        if pixel_coords is not None:
            uw, vw, ui, vi = pixel_coords
            
            # 손목과 검지 표시
            cv2.circle(color_image, (uw, vw), 8, (0, 255, 0), -1)
            cv2.circle(color_image, (ui, vi), 8, (0, 0, 255), -1)
            
            # Ray 표시
            direction_2d = np.array([ui - uw, vi - vw])
            norm = np.linalg.norm(direction_2d)
            if norm > 0:
                direction_2d = direction_2d / norm * 800
                ray_end = (int(uw + direction_2d[0]), int(vw + direction_2d[1]))
                cv2.arrowedLine(color_image, (uw, vw), ray_end, (255, 0, 0), 3)
            
            # 손 랜드마크 그리기
            self.mp_draw.draw_landmarks(
                color_image,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS
            )
        
        # 가리킨 물체 정보 표시
        if pointed_object is not None:
            X, Y, Z = pointed_object['position_3d']
            label = pointed_object['label']
            dist = pointed_object['distance_to_ray']
            
            # 물체 중심에 큰 원 표시
            x1, y1, x2, y2 = pointed_object['bbox']
            center_u = int((x1 + x2) / 2)
            center_v = int((y1 + y2) / 2)
            
            cv2.circle(color_image, (center_u, center_v), 15, (0, 255, 255), 3)
            cv2.circle(color_image, (center_u, center_v), 3, (0, 255, 255), -1)
            
            # 정보 텍스트
            info_text = [
                f"Pointed: {label}",
                f"3D: ({X:.3f}, {Y:.3f}, {Z:.3f})m",
                f"Ray dist: {dist:.3f}m",
                f"[PUBLISHED]"
            ]
            
            y_offset = 30
            for i, text in enumerate(info_text):
                cv2.putText(color_image, text, (10, y_offset + i * 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return color_image
    
    def run(self):
        """메인 루프"""
        try:
            while rclpy.ok():
                # 프레임 가져오기
                frames = self.pipeline.wait_for_frames()
                
                # Depth를 Color에 정렬
                aligned_frames = self.align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                
                if not color_frame or not depth_frame:
                    continue
                
                # NumPy 배열로 변환
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                
                h, w, _ = color_image.shape
                
                # YOLO 객체 탐지 (threshold 높임)
                detections = self.detect_objects(color_image, conf_threshold=0.7)
                
                # RGB로 변환 (MediaPipe용)
                rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                
                # 손 감지
                results = self.hands.process(rgb_image)
                
                pointed_object = None
                pixel_coords = None
                hand_lms = None
                
                if results.multi_hand_landmarks:
                    hand_lms = results.multi_hand_landmarks[0]
                    
                    # Pointing ray 계산
                    P_wrist, P_index, direction, pixel_coords = \
                        self.get_pointing_ray(hand_lms, depth_image, w, h)
                    
                    if direction is not None:
                        # 가리킨 물체 찾기
                        pointed_object = self.find_pointed_object(
                            detections, depth_image, P_wrist, direction
                        )
                        
                        # ROS2 topic으로 발행
                        if pointed_object is not None:
                            self.publish_pointed_object(pointed_object)
                
                # Depth 컬러맵 (시각화용)
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.03),
                    cv2.COLORMAP_JET
                )
                
                # 시각화
                color_image = self.visualize(
                    color_image, depth_colormap, hand_lms, 
                    pixel_coords, detections, pointed_object
                )
                
                # 화면 표시
                combined = np.hstack((color_image, depth_colormap))
                cv2.imshow('Finger Pointing Detection', combined)
                
                # ESC로 종료
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                
                # ROS2 spin once
                rclpy.spin_once(self, timeout_sec=0.001)
                    
        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()


def main(args=None):
    import os
    
    rclpy.init(args=args)
    
    # 모델 경로 설정
    HOME_DIR = os.path.expanduser("~")
    workspace_name = "FiXit_ws"
    pkg_name = "vision"
    pkg_path = os.path.join(HOME_DIR, workspace_name, f"src/{pkg_name}/{pkg_name}")
    pt_filename = "third_result.pt"
    
    model_path = os.path.join(pkg_path, pt_filename)
    
    try:
        detector = FingerPointingDetector(model_path)
        detector.run()
    except KeyboardInterrupt:
        pass
    finally:
        detector.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()