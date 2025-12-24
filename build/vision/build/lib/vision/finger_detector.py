import cv2
import mediapipe as mp
import numpy as np
import pyrealsense2 as rs


class FingerPointingDetector:
    def __init__(self):
        # MediaPipe 초기화
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
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
    
    def find_pointed_object(self, depth_image, ray_origin, ray_direction, 
                           max_distance=2.0, threshold=0.05):
        """
        Ray를 따라 가장 가까운 물체 찾기
        max_distance: ray를 검사할 최대 거리 (미터)
        threshold: 물체로 판단할 depth 변화 임계값 (미터)
        """
        if ray_origin is None or ray_direction is None:
            return None
        
        # Ray를 따라 샘플링
        num_samples = 100
        distances = np.linspace(0.1, max_distance, num_samples)
        
        h, w = depth_image.shape
        min_distance = float('inf')
        best_point = None
        
        for dist in distances:
            # Ray 위의 3D 점
            point_3d = ray_origin + ray_direction * dist
            
            # 3D를 픽셀로 역변환
            X, Y, Z = point_3d
            if Z <= 0:
                continue
                
            u = int(self.fx * X / Z + self.cx)
            v = int(self.fy * Y / Z + self.cy)
            
            # 이미지 범위 체크
            if u < 0 or u >= w or v < 0 or v >= h:
                continue
            
            # 실제 depth 값
            actual_depth = self.get_stable_depth(depth_image, u, v) * self.depth_scale
            
            if actual_depth == 0:
                continue
            
            # Ray 상의 예상 depth와 실제 depth 비교
            expected_depth = Z
            depth_diff = abs(actual_depth - expected_depth)
            
            # 물체 감지 (실제 depth가 예상보다 가까움)
            if depth_diff < threshold and actual_depth < expected_depth:
                if actual_depth < min_distance:
                    min_distance = actual_depth
                    best_point = np.array([X, Y, actual_depth])
        
        return best_point
    
    def visualize(self, color_image, depth_colormap, hand_landmarks, 
                  pixel_coords, pointed_3d):
        """결과 시각화"""
        if pixel_coords is not None:
            uw, vw, ui, vi = pixel_coords
            
            # 손목과 검지 표시
            cv2.circle(color_image, (uw, vw), 8, (0, 255, 0), -1)
            cv2.circle(color_image, (ui, vi), 8, (0, 0, 255), -1)
            
            # Ray 표시
            direction_2d = np.array([ui - uw, vi - vw])
            norm = np.linalg.norm(direction_2d)
            if norm > 0:
                direction_2d = direction_2d / norm * 500
                ray_end = (int(uw + direction_2d[0]), int(vw + direction_2d[1]))
                cv2.arrowedLine(color_image, (uw, vw), ray_end, (255, 0, 0), 3)
            
            # 손 랜드마크 그리기
            self.mp_draw.draw_landmarks(
                color_image,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS
            )
        
        # 가리킨 물체 표시
        if pointed_3d is not None:
            X, Y, Z = pointed_3d
            
            # 3D를 픽셀로 변환
            u = int(self.fx * X / Z + self.cx)
            v = int(self.fy * Y / Z + self.cy)
            
            # 물체 위치에 표시
            cv2.circle(color_image, (u, v), 15, (0, 255, 255), 3)
            cv2.circle(color_image, (u, v), 3, (0, 255, 255), -1)
            
            # 3D 좌표 텍스트
            text = f"Target: ({X:.3f}, {Y:.3f}, {Z:.3f})m"
            cv2.putText(color_image, text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return color_image
    
    def run(self):
        """메인 루프"""
        try:
            while True:
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
                
                # RGB로 변환 (MediaPipe용)
                rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                
                # 손 감지
                results = self.hands.process(rgb_image)
                
                pointed_3d = None
                pixel_coords = None
                hand_lms = None
                
                if results.multi_hand_landmarks:
                    hand_lms = results.multi_hand_landmarks[0]
                    
                    # Pointing ray 계산
                    P_wrist, P_index, direction, pixel_coords = \
                        self.get_pointing_ray(hand_lms, depth_image, w, h)
                    
                    if direction is not None:
                        # 가리킨 물체 찾기
                        pointed_3d = self.find_pointed_object(
                            depth_image, P_wrist, direction
                        )
                
                # Depth 컬러맵 (시각화용)
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.03),
                    cv2.COLORMAP_JET
                )
                
                # 시각화
                color_image = self.visualize(
                    color_image, depth_colormap, hand_lms, 
                    pixel_coords, pointed_3d
                )
                
                # 화면 표시
                combined = np.hstack((color_image, depth_colormap))
                cv2.imshow('Finger Pointing Detection', combined)
                
                # ESC로 종료
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                    
        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = FingerPointingDetector()
    detector.run()