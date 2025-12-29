import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
from dataclasses import dataclass
from typing import Optional, List, Tuple


@dataclass
class Detection:
    """YOLO 검출 결과"""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    class_id: int
    class_name: str
    confidence: float
    center_2d: Tuple[int, int] = None  # 바운딩 박스 중심
    center_3d: Optional[np.ndarray] = None  # 3D 좌표 (나중에 추가)
    
    def __post_init__(self):
        if self.center_2d is None:
            x1, y1, x2, y2 = self.bbox
            self.center_2d = ((x1 + x2) // 2, (y1 + y2) // 2)


@dataclass
class HandLandmarks:
    """손 랜드마크 정보"""
    fingertip_2d: Tuple[int, int]  # 검지 끝 (픽셀)
    wrist_2d: Tuple[int, int]      # 손목 (픽셀)
    fingertip_3d: Optional[np.ndarray] = None  # 3D 좌표
    wrist_3d: Optional[np.ndarray] = None
    pointing_ray: Optional[np.ndarray] = None  # 가리키는 방향 벡터


class ObjectDetector:
    """YOLO 기반 객체 검출기"""
    
    def __init__(self, model_path: str, conf_threshold: float = 0.6):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
    
    def detect(self, image: np.ndarray) -> List[Detection]:
        results = self.model(image,
                             conf=self.conf_threshold,
                             classes=[0,1,2,3],
                             verbose=False)
        
        detections = []
        for box in results[0].boxes:
            bbox = tuple(box.xyxy[0].cpu().numpy().astype(int))
            cls_id = int(box.cls[0])
            cls_name = self.model.names[cls_id]
            confidence = float(box.conf[0])
            
            detections.append(Detection(
                bbox=bbox,
                class_id=cls_id,
                class_name=cls_name,
                confidence=confidence
            ))
        
        return detections
    
    def visualize(self, image: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """검출 결과를 이미지에 시각화"""
        img = image.copy()
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{det.class_name} {det.confidence:.2f}"
            cv2.putText(img, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return img


class HandDetector:
    def __init__(self, 
                 max_num_hands: int = 1,
                 min_detection_confidence: float = 0.7,
                 min_tracking_confidence: float = 0.5):

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
    
    def detect(self, image: np.ndarray) -> Optional[HandLandmarks]:
        # MediaPipe는 RGB 이미지 필요
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_image)
        
        if not results.multi_hand_landmarks:
            return None
        
        # 첫 번째 손만 사용
        landmarks = results.multi_hand_landmarks[0]
        h, w = image.shape[:2]
        
        # 검지 끝 (index 8)과 손목 (index 0)
        fingertip = landmarks.landmark[8]
        wrist = landmarks.landmark[0]
        
        return HandLandmarks(
            fingertip_2d=(int(fingertip.x * w), int(fingertip.y * h)),
            wrist_2d=(int(wrist.x * w), int(wrist.y * h))
        )
    
    def visualize(self, image: np.ndarray, hand_landmarks) -> np.ndarray:
        img = image.copy()
        if hand_landmarks:
            cv2.circle(img, hand_landmarks.fingertip_2d, 8, (255, 0, 0), -1)
            cv2.circle(img, hand_landmarks.wrist_2d, 8, (0, 0, 255), -1)
            cv2.line(img, hand_landmarks.wrist_2d, hand_landmarks.fingertip_2d, 
                    (255, 255, 0), 2)
        return img


class PointingAnalyzer:
    def __init__(self, distance_threshold: float = 30.0):
        self.distance_threshold = distance_threshold
    
    def compute_3d_coords(self, 
                          pixel_coords: Tuple[int, int],
                          depth_image: np.ndarray,
                          intrinsics: dict,
                          depth_scale: float) -> Optional[np.ndarray]:

        # Intrinsics 체크
        if intrinsics is None:
            return None
        
        u, v = pixel_coords
        h, w = depth_image.shape
        
        # 경계 체크
        if not (0 <= u < w and 0 <= v < h):
            return None
        
        # Depth 값 직접 사용 (이미 mm 단위)
        z = float(depth_image[v, u])
        if z <= 0:
            return None
        
        # 카메라 좌표 계산
        x = (u - intrinsics['ppx']) * z / intrinsics['fx']
        y = (v - intrinsics['ppy']) * z / intrinsics['fy']
        
        return np.array([x, y, z])
    
    def update_3d_info(self,
                       hand_landmarks: HandLandmarks,
                       detections: List[Detection],
                       depth_image: np.ndarray,
                       intrinsics: dict,
                       depth_scale: float) -> None:

        # 손가락과 손목의 3D 좌표 계산
        hand_landmarks.fingertip_3d = self.compute_3d_coords(
            hand_landmarks.fingertip_2d, depth_image, intrinsics, depth_scale
        )
        hand_landmarks.wrist_3d = self.compute_3d_coords(
            hand_landmarks.wrist_2d, depth_image, intrinsics, depth_scale
        )
        
        # 가리키는 방향 벡터 계산
        if hand_landmarks.fingertip_3d is not None and hand_landmarks.wrist_3d is not None:
            direction = hand_landmarks.fingertip_3d - hand_landmarks.wrist_3d
            hand_landmarks.pointing_ray = direction / np.linalg.norm(direction)
        
        # 각 객체의 3D 좌표 계산
        for det in detections:
            det.center_3d = self.compute_3d_coords(
                det.center_2d, depth_image, intrinsics, depth_scale
            )
    
    def find_pointed_object(self,
                           hand_landmarks: HandLandmarks,
                           detections: List[Detection]) -> Optional[Detection]:

        if (hand_landmarks.fingertip_3d is None or 
            hand_landmarks.pointing_ray is None):
            return None
        
        for det in detections:
            if det.center_3d is None:
                continue
            
            # 손가락 끝에서 물체로의 벡터
            vec_to_obj = det.center_3d - hand_landmarks.fingertip_3d
            
            # 손가락 방향으로의 투영
            projection = np.dot(vec_to_obj, hand_landmarks.pointing_ray)
            
            # 앞쪽을 가리키지 않으면 스킵
            if projection <= 0:
                continue
            
            # 손가락 방향 레이에서 물체까지의 수직 거리
            perpendicular_dist = np.linalg.norm(
                vec_to_obj - projection * hand_landmarks.pointing_ray
            )
            
            # 임계값 이내면 해당 객체 반환
            if perpendicular_dist < self.distance_threshold:
                return det
        
        return None


class VisionTester:
    def __init__(self, model_path: str = None, use_webcam: bool = True, camera_id: int = 0):
        self.use_webcam = use_webcam
        
        # CV 모듈 초기화
        self.object_detector = ObjectDetector(model_path) if model_path else None
        self.hand_detector = HandDetector()
        self.pointing_analyzer = PointingAnalyzer(distance_threshold=50.0)
        
        # 카메라 초기화
        if use_webcam:
            self.cap = cv2.VideoCapture(camera_id)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            print(f"✓ 웹캠 초기화 완료 (ID: {camera_id})")
        else:
            import pyrealsense2 as rs
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            profile = self.pipeline.start(config)
            
            # 카메라 파라미터 획득
            intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            self.intrinsics = {
                'fx': intr.fx, 'fy': intr.fy, 
                'ppx': intr.ppx, 'ppy': intr.ppy
            }
            self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
            self.align = rs.align(rs.stream.color)
            print("✓ RealSense 초기화 완료")
    
    def visualize_results(self, image: np.ndarray, 
                         detections: List[Detection],
                         hand_landmarks: HandLandmarks,
                         pointed_object: Detection = None) -> np.ndarray:

        vis_img = image.copy()
        
        # 1. 객체 검출 결과 그리기
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            
            # 가리킨 객체는 초록색, 나머지는 파란색
            if pointed_object and det == pointed_object:
                color = (0, 255, 0)  # 초록색
                thickness = 3
            else:
                color = (255, 0, 0)  # 파란색
                thickness = 2
            
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, thickness)
            
            # 라벨 (클래스명 + 신뢰도)
            label = f"{det.class_name} {det.confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            
            # 라벨 배경
            cv2.rectangle(vis_img, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # 라벨 텍스트
            cv2.putText(vis_img, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # 중심점 표시
            cv2.circle(vis_img, det.center_2d, 5, color, -1)
        
        # 2. 손 랜드마크 그리기
        if hand_landmarks:
            # 손목 (빨간색)
            cv2.circle(vis_img, hand_landmarks.wrist_2d, 8, (0, 0, 255), -1)
            cv2.putText(vis_img, "Wrist", 
                       (hand_landmarks.wrist_2d[0] + 10, hand_landmarks.wrist_2d[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # 검지 끝 (노란색)
            cv2.circle(vis_img, hand_landmarks.fingertip_2d, 8, (0, 255, 255), -1)
            cv2.putText(vis_img, "Fingertip",
                       (hand_landmarks.fingertip_2d[0] + 10, hand_landmarks.fingertip_2d[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # 가리키는 방향 선 (노란색)
            cv2.line(vis_img, hand_landmarks.wrist_2d, hand_landmarks.fingertip_2d,
                    (0, 255, 255), 2)
            
            # 가리키는 방향 연장선 그리기
            if hand_landmarks.fingertip_3d is not None:
                direction = (hand_landmarks.fingertip_2d[0] - hand_landmarks.wrist_2d[0],
                           hand_landmarks.fingertip_2d[1] - hand_landmarks.wrist_2d[1])
                length = 200  # 연장선 길이 (픽셀)
                end_point = (
                    int(hand_landmarks.fingertip_2d[0] + direction[0] * length / 
                        max(abs(direction[0]), abs(direction[1]))),
                    int(hand_landmarks.fingertip_2d[1] + direction[1] * length / 
                        max(abs(direction[0]), abs(direction[1])))
                )
                cv2.line(vis_img, hand_landmarks.fingertip_2d, end_point,
                        (0, 255, 255), 1, cv2.LINE_AA)
        
        # 3. 정보 패널
        info_y = 30
        if detections:
            cv2.putText(vis_img, f"Objects: {len(detections)}", (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            info_y += 30
        
        if hand_landmarks:
            cv2.putText(vis_img, "Hand: Detected", (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            info_y += 30
        
        if pointed_object:
            cv2.putText(vis_img, f"Pointing at: {pointed_object.class_name}", 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return vis_img
    
    def run(self):
        """메인 테스트 루프"""
        print("\n" + "="*50)
        print("Vision Detector 독립 테스트")
        print("="*50)
        print("사용법:")
        print("  - ESC: 종료")
        print("  - 's': 현재 프레임 저장")
        print("="*50 + "\n")
        
        frame_count = 0
        
        try:
            while True:
                # 프레임 획득
                if self.use_webcam:
                    ret, color_img = self.cap.read()
                    if not ret:
                        print("웹캠에서 프레임을 읽을 수 없습니다.")
                        break
                    depth_img = None
                else:
                    frames = self.pipeline.wait_for_frames()
                    aligned = self.align.process(frames)
                    color_img = np.asanyarray(aligned.get_color_frame().get_data())
                    depth_img = np.asanyarray(aligned.get_depth_frame().get_data())
                
                frame_count += 1
                
                # 객체 검출
                detections = []
                if self.object_detector:
                    detections = self.object_detector.detect(color_img)
                    if detections:
                        print(f"[Frame {frame_count}] 검출: {[d.class_name for d in detections]}")
                
                # 손 검출
                hand_landmarks = self.hand_detector.detect(color_img)
                
                # 3D 정보 업데이트 및 가리킴 분석 (RealSense인 경우만)
                pointed_object = None
                if not self.use_webcam and depth_img is not None and hand_landmarks:
                    self.pointing_analyzer.update_3d_info(
                        hand_landmarks, detections, depth_img,
                        self.intrinsics, self.depth_scale
                    )
                    pointed_object = self.pointing_analyzer.find_pointed_object(
                        hand_landmarks, detections
                    )
                    
                    if pointed_object:
                        print(f"[Frame {frame_count}] 👉 가리킴: {pointed_object.class_name}")
                
                # 시각화
                vis_img = self.visualize_results(
                    color_img, detections, hand_landmarks, pointed_object
                )
                
                # 화면 표시
                cv2.imshow("Vision Detector Test", vis_img)
                
                # 키 입력 처리
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break
                elif key == ord('s'):  # 스크린샷 저장
                    filename = f"vision_test_{frame_count}.jpg"
                    cv2.imwrite(filename, vis_img)
                    print(f"✓ 저장됨: {filename}")
        
        except KeyboardInterrupt:
            print("\n종료 중...")
        
        finally:
            if self.use_webcam:
                self.cap.release()
            else:
                self.pipeline.stop()
            cv2.destroyAllWindows()
            print("✓ 테스트 완료")


# 독립적 테스트를 위한 예제
if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Vision Detector 독립 테스트")
    parser.add_argument("--model", type=str, default=None,
                       help="YOLO 모델 경로 (예: third_result.pt)")
    parser.add_argument("--webcam", action="store_true",
                       help="웹캠 사용 (기본: RealSense)")
    parser.add_argument("--camera-id", type=int, default=0,
                       help="웹캠 ID (기본: 0)")
    
    args = parser.parse_args()
    
    # 모델 경로 자동 찾기
    if args.model is None:
        possible_model = os.path.expanduser("~/FiXit_ws/src/vision/models/result_4.pt")
        if os.path.exists(possible_model):
            args.model = possible_model
            print(f"✓ 모델 발견: {possible_model}")
    
    if args.model and not os.path.exists(args.model):
        print(f"⚠ 경고: 모델 파일을 찾을 수 없습니다: {args.model}")
        print("객체 검출 없이 손 검출만 실행됩니다.")
        args.model = None
    
    # 테스터 실행
    tester = VisionTester(
        model_path=args.model,
        use_webcam=args.webcam,
        camera_id=args.camera_id
    )
    
    tester.run()