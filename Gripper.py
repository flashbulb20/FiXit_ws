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

    def grip_adaptive(self, force_val=300):
        """물체 크기를 모를 때 0까지 닫으며 잡기"""
        print("물체 탐색 및 잡기 시작...")
        # 1. 0으로 닫기 명령 (설정한 힘으로)
        self.client.write_registers(address=0, values=[force_val, 0, 16], unit=65)
        
        # 2. 이동 시간 대기 (물체 크기에 따라 조절 필요)
        time.sleep(1.5)
        
        # 3. 현재 상태와 너비 읽기 (257번: 상태, 258번: 현재 너비)
        response = self.client.read_holding_registers(address=257, count=2, unit=65)
        if not response.isError():
            status = response.registers[0]
            current_width = response.registers[1]
            
            # OnRobot 상태 코드: 2nd byte가 1이면 잡기 성공(Grip detected)
            # 여기서는 단순하게 너비가 0보다 크면 무언가 잡은 것으로 판단
            if current_width > 10: 
                print(f"잡기 성공! 측정된 물체 너비: {current_width/10} mm")
            else:
                print("물체를 찾지 못하고 완전히 닫혔습니다.")
            return current_width
        return None   


def main():
    # 실험을 위해 최소한의 설정만 로드
    load_dotenv(dotenv_path=".env")
    rclpy.init()
    node = rclpy.create_node("gripper_test_node", namespace=ROBOT_ID)
    DR_init.__dsr__node = node
    
    # 그리퍼 객체 생성
    gripper = RG(GRIPPER_NAME, TOOLCHARGER_IP, TOOLCHARGER_PORT)

    print("=== 그리퍼 Adaptive Grip 실험 모드 ===")
    print("1. 'g' 입력: 물체 잡기 (Adaptive Grip)")
    print("2. 'o' 입력: 그리퍼 열기 (Open)")
    print("3. 'q' 입력: 종료")

    try:
        while True:
            user_input = input("\n명령을 입력하세요 (g/o/q): ").lower()

            if user_input == 'g':
                # 2번 Adaptive 방식 실행
                # 힘(force)을 200~400 사이에서 조절하며 테스트해보세요.
                width = gripper.grip_adaptive(force_val=300)
                if width is not None:
                    print(f"결과: {width/10}mm 위치에서 멈춤")
                
            elif user_input == 'o':
                print("그리퍼를 완전히 엽니다.")
                gripper.open_gripper()

            elif user_input == 'q':
                print("실험을 종료합니다.")
                break
            else:
                print("잘못된 입력입니다.")

    except Exception as e:
        print(f"실험 중 오류 발생: {e}")
    finally:
        rclpy.shutdown()

if __name__ == "__main__":
    main()

