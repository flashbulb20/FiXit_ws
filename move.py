import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from std_msgs.msg import String, Float64MultiArray
from geometry_msgs.msg import PoseStamped
from pymodbus.client.sync import ModbusTcpClient as ModbusClient
import threading
import time

# 두산 로봇 서비스 메시지 임포트
from dsr_msgs2.srv import MoveLine, MoveJoint, SetRobotMode

class RobotController(Node):
    def __init__(self):
        super().__init__('robot_control_node')
        self.group = ReentrantCallbackGroup()

        # --- [데이터 저장 변수] ---
        self.current_posx = None  # 현재 데카르트 좌표 [x, y, z, A, B, C]
        self.current_posj = None  # 현재 관절 각도 [j1, j2, j3, j4, j5, j6]

        # --- [그리퍼 설정] ---
        # 그리퍼 연결 (실제 환경에 맞춰 IP 수정 필요)
        try:
            self.client = ModbusClient('192.168.1.1', port=502, timeout=1)
            self.client.connect()
        except:
            self.get_logger().error("그리퍼 연결 실패")

        # --- [서비스 클라이언트 설정] ---
        # dsr_msgs2 직접 호출 방식으로 변경
        self.cli_movel = self.create_client(MoveLine, '/dsr01/motion/move_line')
        self.cli_movej = self.create_client(MoveJoint, '/dsr01/motion/move_joint')
        self.cli_mode = self.create_client(SetRobotMode, '/dsr01/system/set_robot_mode')

        # --- [Subscriber 설정] ---
        # 1. 로봇 상태 정보 (제공해주신 코드의 토픽 반영)
        self.create_subscription(Float64MultiArray, "/dsr01/msg/current_posx", self.posx_cb, 10)
        self.create_subscription(Float64MultiArray, "/dsr01/msg/joint_state", self.posj_cb, 10)

        # 2. 외부 제어 명령 (FiXiT 인터페이스)
        self.create_subscription(String, '/robot/gripper', self.gripper_cb, 10, callback_group=self.group)
        self.create_subscription(String, '/robot/jog', self.jog_cb, 10, callback_group=self.group)
        self.create_subscription(String, '/robot/nudge_cmd', self.nudge_cb, 10, callback_group=self.group)
        self.create_subscription(PoseStamped, '/robot/target_pose', self.pose_cb, 10, callback_group=self.group)

        # 상태 알림용 Publisher
        self.status_pub = self.create_publisher(String, '/robot/status', 10)
        
        self.get_logger().info("=== FiXiT 제어 노드 가동 (토픽 기반 위치 추적) ===")

    # --- [콜백 함수들] ---
    def posx_cb(self, msg): self.current_posx = msg.data
    def posj_cb(self, msg): self.current_posj = msg.data

    def set_mode(self, mode=0):
        req = SetRobotMode.Request()
        req.robot_mode = mode
        self.cli_mode.call_async(req)

    def send_movel(self, pos, vel=60.0, acc=60.0):
        """배열 규격에 맞춘 MoveLine 서비스 호출"""
        if not self.cli_movel.wait_for_service(timeout_sec=1.0):
            return
        req = MoveLine.Request()
        req.pos = pos
        req.vel = [vel, vel] # dsr_msgs2 규격 대응
        req.acc = [acc, acc]
        req.time = 0.0
        req.radius = 0.0
        self.cli_movel.call_async(req)

    def nudge_cb(self, msg):
        """미세 조정: 토픽으로 저장된 실시간 좌표 활용 (Hanging 없음)"""
        if self.current_posx is None:
            self.get_logger().warn("현재 좌표를 아직 수신하지 못했습니다.")
            return

        cmd = msg.data.upper().strip()
        dist = 30.0
        new_pos = list(self.current_posx)

        if cmd == "UP": new_pos[2] += dist
        elif cmd == "DOWN": new_pos[2] -= dist
        elif cmd == "LEFT": new_pos[1] += dist
        elif cmd == "RIGHT": new_pos[1] -= dist
        elif cmd == "FORWARD": new_pos[0] += dist
        elif cmd == "BACKWARD": new_pos[0] -= dist
        else: return

        self.get_logger().info(f"Nudge 이동: {cmd} -> {new_pos[:3]}")
        self.send_movel(new_pos)
        self.status_pub.publish(String(data="arrived"))

    def jog_cb(self, msg):
        """관절 회전: 토픽으로 저장된 j6 각도 활용"""
        if self.current_posj is None: return
        cmd = msg.data.upper().strip()
        new_j = list(self.current_posj)
        if "FRONT" in cmd: new_j[5] += 15.0
        elif "BACK" in cmd: new_j[5] -= 15.0
        
        req = MoveJoint.Request()
        req.pos = new_j
        req.vel = 60.0
        req.acc = 60.0
        self.cli_movej.call_async(req)

    def gripper_cb(self, msg):
        cmd = msg.data.lower().strip()
        force, width = (400, 1100) if cmd == "open" else (300, 0)
        self.client.write_registers(address=0, values=[force, width, 16], unit=65)
        if cmd == "close": self.status_pub.publish(String(data="gripped"))

    def pose_cb(self, msg):
        if self.current_posx is None: return
        tx, ty, tz = msg.pose.position.x * 1000.0, msg.pose.position.y * 1000.0, msg.pose.position.z * 1000.0
        curr = self.current_posx
        self.send_movel([tx, ty, tz, curr[3], curr[4], curr[5]])
        self.status_pub.publish(String(data="arrived"))

def main(args=None):
    rclpy.init(args=args)
    node = RobotController()
    
    # 시작 시 로봇 모드를 0(AUT)으로 설정 (제공해주신 코드 로직 반영)
    node.set_mode(0)
    
    # 멀티스레드 실행 (서비스와 구독 병렬 처리)
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
