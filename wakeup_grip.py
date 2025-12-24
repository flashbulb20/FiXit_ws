import cv2
import rclpy
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from ultralytics import YOLO 
from pymodbus.client.sync import ModbusTcpClient as ModbusClient
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import time
import numpy as np
import os
from dotenv import load_dotenv

# 음성 인식 관련 모듈
import MicController
from wakeup_word import WakeupWord
from STT import STT

# 두산 로봇 관련
import DR_init

# --- 설정 ---
ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
VELOCITY, ACC = 60, 60
DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

GRIPPER_NAME = "rg2"
TOOLCHARGER_IP = "192.168.1.1"
TOOLCHARGER_PORT = "502"

TOOL_MAP = {
    "해머": "hammer", "망치": "hammer", "锤子": "hammer",'hammer':"hammer",
    "스패너": "monkey spanner", "몽키": "monkey spanner",
    "렌치": "ranch", "플라이어": "ranch",
}

class RG():
    def __init__(self, gripper, ip, port):
        self.client = ModbusClient(ip, port=port, stopbits=1, bytesize=8, parity='E', baudrate=115200, timeout=1)
        self.max_width = 1100 if gripper == 'rg2' else 1600
        self.client.connect()
    def open_gripper(self, force_val=400):
        self.client.write_registers(address=0, values=[force_val, self.max_width, 16], unit=65)
    def close_gripper(self, force_val=400):
        self.client.write_registers(address=0, values=[force_val, 0, 16], unit=65)

class ImgNode(Node):
    def __init__(self):
        super().__init__('img_node')
        self.bridge = CvBridge()
        self.color_frame, self.depth_frame, self.intrinsics = None, None, None
        self.create_subscription(Image, '/camera/camera/color/image_raw', self.color_callback, 10)
        self.create_subscription(Image, '/camera/camera/aligned_depth_to_color/image_raw', self.depth_callback, 10)
        self.create_subscription(CameraInfo, '/camera/camera/color/camera_info', self.camera_info_callback, 10)

    def camera_info_callback(self, msg): self.intrinsics = {"fx": msg.k[0], "fy": msg.k[4], "ppx": msg.k[2], "ppy": msg.k[5]}
    def color_callback(self, msg): self.color_frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
    def depth_callback(self, msg): self.depth_frame = self.bridge.imgmsg_to_cv2(msg, 'passthrough')

class RobotController:
    def __init__(self, img_node):
        self.img_node = img_node
        self.gripper2cam = np.load("T_gripper2camera.npy")
        self.yolo_model = YOLO("tools_result.pt") 
        self.gripper = RG(GRIPPER_NAME, TOOLCHARGER_IP, TOOLCHARGER_PORT)
        self.JReady = posj([0, 0, 100, 0, 90, 0]) 

    def visualize_and_get_pos(self, target_label=None):
        """화면 시각화 및 타겟 좌표/방향 반환"""
        rclpy.spin_once(self.img_node)
        img = self.img_node.color_frame
        depth = self.img_node.depth_frame
        intrinsics = self.img_node.intrinsics
        
        if img is None: return None, False

        results = self.yolo_model(img, verbose=False)
        annotated_img = results[0].plot() 
        target_pos = None
        is_horizontal = False # 가로 배치 여부

        for box in results[0].boxes:
            label_name = results[0].names[int(box.cls[0])].lower()
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            # 너비(w)와 높이(h) 계산
            w = x2 - x1
            h = y2 - y1

            cv2.circle(annotated_img, (cx, cy), 5, (0, 0, 255), -1)
            
            if target_label and label_name == target_label.lower():
                # 가로로 긴 경우(w > h) 90도 회전 필요
                if w > h:
                    is_horizontal = True
                    cv2.putText(annotated_img, "HORIZONTAL: ROTATE 90", (x1, y2+20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                z = depth[cy, cx] if depth is not None else 0
                if z > 0:
                    cv2.putText(annotated_img, "TARGET DETECTED", (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cam_x = (cx - intrinsics["ppx"]) * z / intrinsics["fx"]
                    cam_y = (cy - intrinsics["ppy"]) * z / intrinsics["fy"]
                    cam_pos = np.array([cam_x, cam_y, z, 1])
                    curr_pos = get_current_posx()[0]
                    R = Rotation.from_euler("ZYZ", curr_pos[3:], degrees=True).as_matrix()
                    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = curr_pos[:3]
                    target_pos = (T @ self.gripper2cam @ cam_pos)[:3]

        cv2.imshow("Robot Vision (YOLO)", annotated_img)
        cv2.waitKey(1)
        return target_pos, is_horizontal

    def execute_pick_and_place(self, coords, rotate_90=False):
        x, y, z = coords
        curr = get_current_posx()[0]
        rx, ry, rz = curr[3], curr[4], curr[5]

        # 가로일 경우 현재 각도에서 90도 회전
        if rotate_90:
            print("Rotating gripper 90 degrees for horizontal pick...")
            rz += 90.0

        # Approach (위에서 접근)
        movel(posx([x, y, z + 100, rx, ry, rz]), vel=VELOCITY, acc=ACC)
        
        # Pick (내려가서 잡기)
        movel(posx([x, y, z - 25, rx, ry, rz]), vel=VELOCITY/2, acc=ACC/2)
        self.gripper.close_gripper()
        wait(1.0)
        
        # Retract (다시 올라오기)
        movel(posx([x, y, z + 100, rx, ry, rz]), vel=VELOCITY, acc=ACC)
        
        # Place (준비 자세로 복귀 후 놓기)
        movej(self.JReady, vel=VELOCITY, acc=ACC)
        self.gripper.open_gripper()
        print("작업 완료.")

def main():
    load_dotenv(dotenv_path=".env")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    rclpy.init()
    node = rclpy.create_node("dsr_visual_pick_node", namespace=ROBOT_ID)
    DR_init.__dsr__node = node
    
    global get_current_posx, movej, movel, wait, posx, posj
    from DSR_ROBOT2 import get_current_posx, movej, movel, wait
    from DR_common2 import posx, posj

    img_node = ImgNode()
    robot = RobotController(img_node)
    mic = MicController.MicController()
    stt = STT(openai_api_key)
    wakeup = WakeupWord(mic.config.buffer_size)

    cv2.namedWindow("Robot Vision (YOLO)")

    try:
        while True:
            mic.open_stream()
            wakeup.set_stream(mic.stream)
            print("웨이크워드 대기 중...")
            
            while not wakeup.is_wakeup():
                robot.visualize_and_get_pos()
                if cv2.waitKey(1) & 0xFF == 27: break
            
            print(">>> 명령 대기 중...")
            mic.stream.stop_stream()
            mic.stream.close()
            
            text_result = stt.speech2text()
            print(f"인식: {text_result}")

            target_label = None
            for kr_word, en_label in TOOL_MAP.items():
                if kr_word in text_result:
                    target_label = en_label
                    break

            if target_label and any(cmd in text_result for cmd in ["잡아", "픽", "가져", "拿起", "pick"]):
                print(f"[{target_label}] 작업을 시작합니다.")
                movej(robot.JReady, vel=VELOCITY, acc=ACC)
                wait(1.5)
                
                # 물체 찾기 (좌표와 가로 여부 가져옴)
                coords, rotate_90 = robot.visualize_and_get_pos(target_label)
                
                if coords is not None:
                    robot.execute_pick_and_place(coords, rotate_90)
                else:
                    print("오류: 타겟을 찾을 수 없습니다.")
            
            elif "종료" in text_result: break

    except Exception as e: print(f"Error: {e}")
    finally:
        mic.audio.terminate()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == "__main__":
    main()