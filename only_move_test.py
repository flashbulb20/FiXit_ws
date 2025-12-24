import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from pymodbus.client.sync import ModbusTcpClient as ModbusClient
import time
import sys
from geometry_msgs.msg import PoseStamped

# 1. 기초 설정 라이브러리 임포트
import DR_init

# --- 로봇 및 그리퍼 설정 값 ---
ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
VELOCITY, ACC = 60, 60
GRIPPER_NAME = "rg2"
TOOLCHARGER_IP = "192.168.1.1"
TOOLCHARGER_PORT = "502"

# --- [중요] DSR 관련 함수 선언 (main에서 할당) ---
get_current_posx = None
get_current_posj = None
movej = None
movel = None
wait = None
posx = None
posj = None

class RG():
    def __init__(self, gripper, ip, port):
        self.client = ModbusClient(ip, port=port, stopbits=1, bytesize=8, parity='E', baudrate=115200, timeout=1)
        self.max_width = 1100 if gripper == 'rg2' else 1600
        self.client.connect()

    def open_gripper(self, force_val=400):
        self.client.write_registers(address=0, values=[force_val, self.max_width, 16], unit=65)

    def close_gripper(self, force_val=300):
        print("물체 탐색 및 잡기 시작...")
        self.client.write_registers(address=0, values=[force_val, 0, 16], unit=65)
        time.sleep(1.5)
        response = self.client.read_holding_registers(address=258, count=1, unit=65)
        if not response.isError():
            return response.registers[0]
        return None

class RobotControlNode(Node):
    def __init__(self):
        super().__init__('robot_control_node', namespace=ROBOT_ID)
        
        # 그리퍼 및 좌표 데이터 초기화
        self.gripper = RG(GRIPPER_NAME, TOOLCHARGER_IP, TOOLCHARGER_PORT)
        self.JReady_pose = [0, 0, 90, 0, 90, 0]
        self.table_pose = [585.23, 80.75, 153.14, 7.14, 129.52, 17.07]
        self.positions = {
            "1": [200.0, -200.0, 60.0, 0.0, 180.0, 0.0],
            "2": [200.0, 0.0, 60.0, 0.0, 180.0, 0.0],
            "3": [200.0, 200.0, 60.0, 0.0, 180.0, 0.0]
        }
        self.goal_pos = [600.0, 0.0, 60.0, 0.0, 180.0, 0.0]

        # 1. Sub: 먼 거리 이동 (MoveIt/경로 계획)
        self.target_pose_sub = self.create_subscription(
            PoseStamped, 
            '/robot/target_pose', 
            self.target_pose_callback, 
            10)

        # 2. Sub: 시선 맞추기용 회전 제어 (Velocity)
        self.jog_sub = self.create_subscription(
            String, 
            '/robot/jog', 
            self.jog_callback, 
            10)

        # 3. Sub: 현재 위치 기준 미세 이동 (MoveL, 2cm)
        self.nudge_sub = self.create_subscription(
            String, 
            '/robot/nudge_cmd', 
            self.nudge_callback, 
            10)

        # 4. Pub: 동작 완료 피드백
        self.status_pub = self.create_publisher(
            String, 
            '/robot/status', 
            10)

        self.get_logger().info("로봇 제어 노드 사양서 기준 통신 준비 완료")

    def command_callback(self, msg):
        user_input = msg.data.strip()
        self.get_logger().info(f"명령 수신: {user_input}")

        target_num = next((n for n in ["1", "2", "3"] if n in user_input), None)

        if "운반" in user_input and target_num:
            self.execute_transport(self.positions[target_num], self.goal_pos, f"물체{target_num} 운반")
        elif "복귀" in user_input and target_num:
            self.execute_transport(self.goal_pos, self.positions[target_num], f"물체{target_num} 복귀")
        elif any(w in user_input for w in ["잡아", "닫아"]):
            self.gripper.close_gripper()
        elif any(w in user_input for w in ["놔", "열어"]):
            self.gripper.open_gripper()
        elif any(w in user_input for w in ["준비", "홈"]):
            movej(posj(self.JReady_pose), vel=VELOCITY, acc=ACC)
        elif "앞으로 돌려" in user_input:
            self.rotate_axis6("앞", angle=-10.0)
        elif "뒤로 돌려" in user_input:
            self.rotate_axis6("뒤", angle=10.0)
        else:
            self.process_relative_move(user_input)

    def process_relative_move(self, cmd):
        directions = {"위": (2, 50), "아래": (2, -50), "왼쪽": (0, -50), "오른쪽": (0, 50), "앞": (1, 50), "뒤": (1, -50)}
        for key, (idx, dist) in directions.items():
            if key in cmd:
                curr_pos = list(get_current_posx()[0])
                curr_pos[idx] += dist
                movel(posx(curr_pos), vel=VELOCITY, acc=ACC)
                return

    def execute_transport(self, start, end, name):
        self.get_logger().info(f"{name} 시퀀스 시작")
        # 접근 -> 하강 -> 잡기 -> 상승 -> 홈 -> 목표접근 -> 하강 -> 놓기 -> 상승 -> 홈
        for p in [start, end]:
            movel(posx([p[0], p[1], p[2]+100, p[3], p[4], p[5]]), vel=VELOCITY, acc=ACC)
            movel(posx(p), vel=VELOCITY/2, acc=ACC/2)
            if p == start: self.gripper.close_gripper()
            else: self.gripper.open_gripper()
            wait(1.0)
            movel(posx([p[0], p[1], p[2]+100, p[3], p[4], p[5]]), vel=VELOCITY, acc=ACC)
            movej(posj(self.JReady_pose), vel=VELOCITY, acc=ACC)

    def rotate_axis6(self, direction, angle):
        curr_j = list(get_current_posj())
        curr_j[5] += angle if direction == "앞" else -angle
        movej(posj(curr_j), vel=VELOCITY, acc=ACC)

def main(args=None):
    global get_current_posx, get_current_posj, movej, movel, wait, posx, posj
    rclpy.init(args=args)
    
    # [핵심] 노드를 먼저 만들고 DR_init에 할당한 후 라이브러리 임포트
    node = rclpy.create_node("dsr_robot_node", namespace=ROBOT_ID)
    DR_init.__dsr__id = ROBOT_ID
    DR_init.__dsr__model = ROBOT_MODEL
    DR_init.__dsr__node = node

    from DSR_ROBOT2 import get_current_posx, get_current_posj, movej, movel, wait
    from DR_common2 import posx, posj

    robot_controller = RobotControlNode()
    
    try:
        rclpy.spin(robot_controller)
    except KeyboardInterrupt:
        pass
    finally:
        robot_controller.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()