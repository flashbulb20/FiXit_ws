import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from pymodbus.client.sync import ModbusTcpClient as ModbusClient
import time
import os
import threading

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

class Gripper():
    def __init__(self, gripper, ip, port):
        self.client = ModbusClient(ip, port=port, stopbits=1, bytesize=8, parity='E', baudrate=115200, timeout=1)
        self.max_width = 1100 if gripper == 'rg2' else 1600
        self.client.connect()

    def open_gripper(self, force_val=400):
        self.client.write_registers(address=0, values=[force_val, self.max_width, 16], unit=65)

    def close_gripper(self, force_val=300):
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
                print(f"잡기 성공!")
            else:
                print("물체를 찾지 못하고 완전히 닫혔습니다.")
            return current_width    
        return None  

class RobotController(Node): # Node를 상속받도록 변경
    def __init__(self):
        super().__init__('robot_control_node') # 노드 이름 초기화
        
        # 그리퍼 객체 생성
        self.gripper = Gripper(GRIPPER_NAME, TOOLCHARGER_IP, TOOLCHARGER_PORT)
        
        # --- Subscriber 생성 ---
        self.subscription = self.create_subscription(
            String, '/robot/gripper', self.gripper_callback, 10) 
        
        # 2. 각도 조정 (Jog)
        self.sub_jog = self.create_subscription(
            String, '/robot/jog', self.jog_callback, 10)
        
        # 3. 미세 조정 (Nudge)
        self.sub_nudge = self.create_subscription(
            String, '/robot/nudge_cmd', self.nudge_callback, 10)
        
        # 4. 장거리 이동 (Target Pose) - 추가된 부분
        self.sub_pose = self.create_subscription(PoseStamped, '/robot/target_pose', self.pose_callback, 10)
        
        self.get_logger().info("모든 토픽(/gripper, /jog, /nudge_cmd) 구독 시작")
        
        # Publisher 생성
        self.status_pub = self.create_publisher(String, '/robot/status', 10)

        # 기존 포즈 설정들
        self.JReady_pose = [0, 0, 90, 0, 90, 0]
        self.positions = {
            "1": [200.0, -200.0, 60.0, 0.0, 180.0, 0.0],
            "2": [200.0, 0.0, 60.0, 0.0, 180.0, 0.0],
            "3": [200.0, 200.0, 60.0, 0.0, 180.0, 0.0]
        }
        # 공통 목표 지점 (운반해서 놓을 위치)
        self.goal_pos = [600.0, 0.0, 60.0, 0.0, 180.0, 0.0]

    def move_relative(self, direction_kr, distance=50.0):
        try:
            # 1. 현재 위치 읽기
            curr_pos_res = get_current_posx()
            
            # 두산 라이브러리는 보통 (list, 0) 형태의 튜플을 반환하므로 첫 번째 요소 추출
            if isinstance(curr_pos_res, (list, tuple)):
                curr_pos = list(curr_pos_res[0])
            else:
                curr_pos = list(curr_pos_res)

            if curr_pos is None or len(curr_pos) < 6:
                self.get_logger().error("로봇 좌표 데이터가 올바르지 않습니다.")
                return False

            # 2. 새로운 좌표 계산 (값 복사 필수)
            new_pos = curr_pos[:] 

            if "위" in direction_kr: new_pos[2] += distance
            elif "아래" in direction_kr: new_pos[2] -= distance
            elif "왼쪽" in direction_kr: new_pos[0] -= distance
            elif "오른쪽" in direction_kr: new_pos[0] += distance
            elif "앞" in direction_kr: new_pos[1] += distance
            elif "뒤" in direction_kr: new_pos[1] -= distance
            else: return False

            self.get_logger().info(f"계산된 목표 좌표: {new_pos}")

            # 3. [핵심 수정] posx()로 반드시 감싸야 합니다.
            # posx 함수는 main에서 global로 선언했으므로 바로 사용 가능합니다.
            # 만약 에러가 나면 상단 임포트나 global 선언을 다시 확인하세요.
            movel(posx(new_pos), vel=VELOCITY, acc=ACC)
            
            return True

        except Exception as e:
            # 여기서 어떤 에러가 나는지 로그를 꼭 확인하세요!
            self.get_logger().error(f"move_relative 내 에러 발생: {e}")
            return False

    def execute_transport_task(self, pick_pos, place_pos, task_name="운반"):
        px, py, pz, prx, pry, prz = pick_pos
        lx, ly, lz, lrx, lry, lrz = place_pos

        print(f">>> [{task_name}] 시작: {pick_pos[:3]} -> {place_pos[:3]}")

        movel(posx([px, py, pz + 100, prx, pry, prz]), vel=VELOCITY, acc=ACC)
        movel(posx([px, py, pz, prx, pry, prz]), vel=VELOCITY/2, acc=ACC/2)
        self.gripper.close_gripper()
        wait(2.0)
        movel(posx([px, py, pz + 100, prx, pry, prz]), vel=VELOCITY, acc=ACC)

        movej(posj(self.JReady_pose), vel=VELOCITY, acc=ACC)

        movel(posx([lx, ly, lz + 100, lrx, lry, lrz]), vel=VELOCITY, acc=ACC)
        movel(posx([lx, ly, lz, lrx, lry, lrz]), vel=VELOCITY/2, acc=ACC/2)
        self.gripper.open_gripper()
        wait(2.0)
        movel(posx([lx, ly, lz + 100, lrx, lry, lrz]), vel=VELOCITY, acc=ACC)
        movej(posj(self.JReady_pose), vel=VELOCITY, acc=ACC)
        print(f">>> [{task_name}] 완료")
        
    
    def rotate_axis6(self, direction_kr, angle=20.0):
        """6번 축(손목 회전)만 정밀하게 회전"""
        # 현재 모든 관절의 각도(J1~J6)를 실시간으로 읽어옵니다.
        curr_j_obj = get_current_posj() 
        # 리스트 형태인지 확인 (일부 버전에서는 객체로 반환될 수 있으므로 첫 번째 요소 선택)
        curr_j = list(curr_j_obj[0]) if isinstance(curr_j_obj[0], (list, tuple)) else list(curr_j_obj)

        if "TURN_FRONT" in direction_kr:
            curr_j[5] += angle  # 6번 관절(Index 5) 각도 증가
        elif "TURN_BACK" in direction_kr:
            curr_j[5] -= angle  # 6번 관절(Index 5) 각도 감소
        else:
            return False

        print(f">>> [회전] 6번 축 타겟 각도: {curr_j[5]:.2f}")
        # posj로 감싸서 해당 각도로 관절 이동
        movej(posj(curr_j), vel=VELOCITY, acc=ACC)
        return True
    
    def publish_status(self, status_text):
            """상태를 /robot/status 토픽으로 발행하는 편의 함수"""
            msg = String()
            msg.data = status_text
            self.status_pub.publish(msg)
            self.get_logger().info(f"상태 발행: [{status_text}]")

    def pose_callback(self, msg):
        """목표 좌표(PoseStamped)를 받아 해당 위치로 이동"""
        # 메시지에서 x, y, z 좌표 추출 (단위 변환: ROS(m) -> DSR(mm))
        tx = msg.pose.position.x * 1000.0
        ty = msg.pose.position.y * 1000.0
        tz = msg.pose.position.z * 1000.0
        
        self.get_logger().info(f"목표 좌표 수신: x={tx:.2f}, y={ty:.2f}, z={tz:.2f}")

        # 방향(orientation)은 현재 자세를 유지하거나, 복잡한 계산이 필요하므로 
        # 우선 현재 로봇의 rx, ry, rz를 그대로 사용하도록 설정 예시
        curr_pos = get_current_posx()[0]
        rx, ry, rz = curr_pos[3], curr_pos[4], curr_pos[5]

        # 목표 위치로 이동 (MoveL 사용)
        movel(posx([tx, ty, tz, rx, ry, rz]), vel=VELOCITY, acc=ACC)

    def jog_callback(self, msg):
        """베이스(J6) 축 각도 조정 처리"""
        command = msg.data.upper().strip()
        self.get_logger().info(f"각도 조정(Jog) 수신: {command}")
        
        angle = 10.0  # 회전할 단위 각도
        if command == "TURN_FRONT":
            self.rotate_axis6("TURN_FRONT", angle=angle) # 기존 작성하신 rotate_axis6 활용
        elif command == "TURN_BACK":
            self.rotate_axis6("TURN_BACK", angle=angle)

    def nudge_callback(self, msg):
        """2~3cm 미세 직선 이동 처리"""
        command = msg.data.upper().strip()
        self.get_logger().info(f"미세 조정(Nudge) 수신: {command}")
        
        distance = 30.0  # 3cm (단위: mm)
        
        # 한글/영문 명령 매핑 및 이동 실행
        direction_map = {
            "UP": "위", "DOWN": "아래", 
            "LEFT": "왼쪽", "RIGHT": "오른쪽", 
            "FORWARD": "앞", "BACKWARD": "뒤"
        }
        
        if command in direction_map:
            target_direction = direction_map[command] # 변수명 명확히
            success = self.move_relative(target_direction, distance=distance)
                
            if success:
                self.publish_status("arrived")
        else:
            self.get_logger().warn(f"알 수 없는 미세 조정 명령: {command}")

    def gripper_callback(self, msg):
        """토픽 메시지를 받았을 때 실행되는 함수"""
        command = msg.data.lower().strip() # 수신된 문자열 소문자 변환
        
        if command == "open":
            self.get_logger().info("명령 수신: open (그리퍼 벌리기)")
            self.gripper.open_gripper()
        elif command == "close":
            self.get_logger().info("명령 수신: close (그리퍼 잡기)")
            self.gripper.close_gripper()
        else:
            self.get_logger().warn(f"알 수 없는 명령: {command}")

def main():
    rclpy.init()
    
    # 1. 노드 생성
    robot_node = RobotController()
    
    # 2. DSR 라이브러리 초기화 (순서 엄수)
    import DR_init
    DR_init.__dsr__node = robot_node
    DR_init.__dsr__id = ROBOT_ID
    DR_init.__dsr__model = ROBOT_MODEL
    
    # 전역 함수 임포트
    global get_current_posx, get_current_posj, movej, movel, wait, posx, posj
    from DSR_ROBOT2 import get_current_posx, get_current_posj, movej, movel, wait
    from DR_common2 import posx, posj

    # [핵심 수정] 멀티스레드 익스큐터 설정
    # 이렇게 해야 콜백 함수 내부에서 실행되는 movel()이 차단되지 않습니다.
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(robot_node)

    robot_node.get_logger().info("=== 로봇 제어 노드 가동 (Multi-Threaded) ===")

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        robot_node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()       