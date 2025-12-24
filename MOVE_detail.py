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
        self.gripper = RG(GRIPPER_NAME, TOOLCHARGER_IP, TOOLCHARGER_PORT)
        # JReady를 리스트 형태로 저장 (posj는 main에서 import 후 적용)
        self.JReady_pose = [0, 0, 90, 0, 90, 0] 

    def move_relative(self, direction_kr, distance=100.0):
        curr_pos = get_current_posx()[0]
        x, y, z, rx, ry, rz = curr_pos

        if "위" in direction_kr:         z += distance
        elif "아래" in direction_kr:     z -= distance
        elif "왼쪽" in direction_kr:     y += distance
        elif "오른쪽" in direction_kr:    y -= distance
        elif "앞" in direction_kr:       x += distance
        elif "뒤" in direction_kr:       x -= distance
        else: return False

        print(f">>> [이동] {direction_kr} 방향으로 {distance}mm 이동")
        movel(posx([x, y, z, rx, ry, rz]), vel=VELOCITY, acc=ACC)
        return True

def main():
    rclpy.init()
    node = rclpy.create_node("dsr_text_control_node", namespace=ROBOT_ID)
    DR_init.__dsr__node = node
    
    global get_current_posx, movej, movel, wait, posx, posj
    from DSR_ROBOT2 import get_current_posx, movej, movel, wait
    from DR_common2 import posx, posj

    img_node = ImgNode()
    robot = RobotController(img_node)

    print("\n" + "="*50)
    print("시스템 준비 완료 (텍스트 명령어 입력 모드)")
    print("사용 가능 명령어:")
    print("1. 이동: '앞', '뒤', '왼쪽', '오른쪽', '위', '아래'")
    print("2. 그리퍼: '잡아'/'닫아', '놓아'/'열어'")
    print("3. 기타: '홈'/'준비', '종료'")
    print("="*50)

    try:
        while rclpy.ok():
            # GUI/이미지 갱신을 위해 rclpy.spin_once 실행
            rclpy.spin_once(img_node, timeout_sec=0.1)
            
            # 사용자로부터 텍스트 명령어 입력 받음
            text_result = input("\n[명령어 입력]: ").strip()
            
            if not text_result:
                continue

            # 명령어 분석 및 실행
            if "종료" in text_result:
                print("프로그램을 종료합니다.")
                break

            # 그리퍼 동작
            if any(w in text_result for w in ["잡아", "닫아", "집어"]):
                print(">>> [그리퍼] 닫기")
                robot.gripper.close_gripper()
            elif any(w in text_result for w in ["놓아", "열어", "벌려", "펴"]):
                print(">>> [그리퍼] 열기")
                robot.gripper.open_gripper()
            
            # 홈 위치 이동
            elif any(w in text_result for w in ["준비", "홈", "원위치"]):
                print(">>> [이동] 홈 위치로 이동")
                movej(posj(robot.JReady_pose), vel=VELOCITY, acc=ACC)
            
            # 상대 이동 처리
            else:
                moved = False
                for direction in ["위", "아래", "왼쪽", "오른쪽", "앞", "뒤"]:
                    if direction in text_result:
                        moved = robot.move_relative(direction)
                        if moved: break
                
                if not moved:
                    print("! 알 수 없는 명령어입니다.")

    except Exception as e:
        print(f"오류 발생: {e}")
    finally:
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == "__main__":
    main()