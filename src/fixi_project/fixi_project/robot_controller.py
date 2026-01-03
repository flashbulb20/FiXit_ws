import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float64MultiArray
from geometry_msgs.msg import PoseStamped
from pymodbus.client.sync import ModbusTcpClient as ModbusClient
import time
import queue

# [1] 두산 로보틱스 라이브러리 설정
import DR_init
ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

# [2] 서비스 메시지 타입 (복구용)
try:
    from dsr_msgs2.srv import SetRobotControl
except ImportError:
    from dsr_msgs.srv import SetRobotControl

# 전역 함수
movej = None; movel = None
get_current_posx = None; get_current_posj = None
get_robot_state = None; set_robot_mode = None
set_safe_stop_reset_type = None

# --- 유틸리티 ---
def wait_and_spin(node, duration):
    end_time = time.time() + duration
    while rclpy.ok() and time.time() < end_time:
        rclpy.spin_once(node, timeout_sec=0.1)
        time.sleep(0.01)

# --- 그리퍼 ---
class GripperController:
    def __init__(self, ip, port):
        self.client = ModbusClient(ip, port=int(port), stopbits=1, bytesize=8, parity='E', baudrate=115200, timeout=1)
        try: self.client.connect()
        except: pass
    def open(self):
        try: self.client.write_registers(0, [400, 1100, 16], unit=65)
        except: pass
    def close(self):
        try: self.client.write_registers(0, [300, 0, 16], unit=65)
        except: pass

# --- 리스너 노드 ---
class RobotListener(Node):
    def __init__(self, cmd_queue):
        super().__init__("rokey_team_recovery_node", namespace=ROBOT_ID)
        self.cmd_queue = cmd_queue
        
        self.status_pub = self.create_publisher(String, '/robot/status', 10)
        self.create_subscription(String, '/robot/nudge_cmd', self.cmd_cb, 10)
        self.create_subscription(String, '/robot/gripper', self.grp_cb, 10)
        self.create_subscription(String, '/robot/jog', self.jog_cb, 10)
        self.create_subscription(PoseStamped, '/robot/target_pose', self.pose_cb, 10)
        self.create_subscription(Float64MultiArray, '/robot/target_joint', self.joint_cb, 10)

        # 복구용 서비스 클라이언트
        self.cli_control = self.create_client(SetRobotControl, f'/{ROBOT_ID}/system/set_robot_control')
        self.SetRobotControl = SetRobotControl

    def cmd_cb(self, msg): 
        cmd = msg.data.upper().strip()
        if cmd == "RECOVERY": self.cmd_queue.put(("RECOVERY", None))
        else: self.cmd_queue.put(("CMD", cmd))
    def grp_cb(self, msg): self.cmd_queue.put(("GRIPPER", msg.data.lower().strip()))
    def jog_cb(self, msg): self.cmd_queue.put(("JOG", msg.data.upper().strip()))
    def joint_cb(self, msg): self.cmd_queue.put(("JOINT", list(msg.data)))
    def pose_cb(self, msg): self.cmd_queue.put(("POSE", msg.pose))
    
    def publish_status(self, msg):
        self.status_pub.publish(String(data=msg))
        self.get_logger().info(f"📤 Status: {msg}")

    # 서비스 호출 헬퍼
    def call_control(self, val, name):
        if not self.cli_control.wait_for_service(timeout_sec=1.0): return False
        req = self.SetRobotControl.Request()
        req.robot_control = val
        future = self.cli_control.call_async(req)
        start = time.time()
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.01)
            if future.done(): return True
            if time.time() - start > 2.0: return False

    # 팀 고유 복구 (3 -> 2 -> 1)
    def recover_robot_team_style(self):
        print("\n🔄 [복구] 팀 고유 시퀀스 시작 (3 -> 2 -> 1)...")
        if set_safe_stop_reset_type: set_safe_stop_reset_type(1)
        self.call_control(3, "Safe Off Reset"); wait_and_spin(self, 1.0)
        self.call_control(2, "Safe Stop Reset"); wait_and_spin(self, 1.0)
        self.call_control(1, "Servo On")
        print("   -> 서보 켜지는 중 (3초)...")
        wait_and_spin(self, 3.0)
        self.publish_status("recovery_success")

# --- 메인 로직 ---
def main(args=None):
    rclpy.init(args=args)
    cmd_queue = queue.Queue()
    node = RobotListener(cmd_queue)
    DR_init.__dsr__node = node 

    global movej, movel, get_current_posx, get_current_posj, get_robot_state, set_robot_mode, set_safe_stop_reset_type
    
    try:
        from DSR_ROBOT2 import (
            movej, movel, get_current_posx, get_current_posj, 
            get_robot_state, set_robot_mode, set_safe_stop_reset_type
        )
        from DR_common2 import posx, posj
        print("✅ DSR 라이브러리 로드 성공")
    except ImportError as e:
        print(f"❌ 라이브러리 로드 실패: {e}")
        return

    gripper = GripperController("192.168.1.1", 502)
    print("=== 로봇 제어 (Hybrid + Updated Coords) ===")
    is_error_mode = False
    
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.01)

        # 에러 감지
        current_state = get_robot_state()
        if current_state in [3, 5, 6, 9, 10]:
            if not is_error_mode:
                is_error_mode = True
                print(f"🚨 에러 감지! (State: {current_state})")
                node.publish_status("fail: collision_detected")
        
        try: msg_type, msg_val = cmd_queue.get_nowait()
        except queue.Empty: continue

        # 복구 명령
        if msg_type == "RECOVERY":
            node.recover_robot_team_style()
            is_error_mode = False
            continue

        if is_error_mode:
            if get_robot_state() == 1: is_error_mode = False
            else: print("⛔ 에러 상태. RECOVERY 필요."); continue

        print(f"⚙️ [실행] {msg_type} -> {msg_val}")
        
        try:
            if msg_type == "GRIPPER":
                if msg_val == "open": gripper.open(); time.sleep(0.5); node.publish_status("opened")
                elif msg_val == "close": gripper.close(); time.sleep(1.0); node.publish_status("gripped")

            elif msg_type == "CMD":
                curr_pos, _ = get_current_posx()
                if curr_pos is None: continue
                target = list(curr_pos); d = 30.0
                
                # Nudge
                if msg_val == "UP": target[2] += d
                elif msg_val == "DOWN": target[2] -= d
                elif msg_val == "LEFT": target[0] -= d
                elif msg_val == "RIGHT": target[0] += d
                elif msg_val == "FORWARD": target[1] += d
                elif msg_val == "BACKWARD": target[1] -= d
                
                # [NEW] 탐색 위치 (SEARCH_TOOL)
                elif msg_val == "SEARCH_TOOL":
                    print("🔭 탐색 위치로 이동...")
                    movej([47.861, 31.503, 43.848, -0.071, 104.867, 47.664], vel=60, acc=60)
                    time.sleep(0.5); node.publish_status("arrived"); continue

                # [UPDATE] 전달 위치 (HANDOVER)
                elif msg_val == "HANDOVER":
                    print("🚚 전달 위치로 이동...")
                    movej([-7.522, -28.045, 113.411, -7.961, 53.899, 9.936], vel=60, acc=60)
                    time.sleep(1.0); node.publish_status("arrived"); continue

                # [UPDATE] 관측 위치 (SCAN) - 돋보기 탐색용
                elif msg_val == "SCAN":
                    print("👁️ 관측(Scan) 위치로 이동...")
                    movej([-51.291, -38.599, 129.907, 55.324, 68.009, 9.435], vel=60, acc=60)
                    time.sleep(1.0); node.publish_status("arrived"); continue

                # [NEW] 확대 준비 위치 (MAGNIFY_PRE)
                elif msg_val == "MAGNIFY_PRE":
                    print("🔍 확대 준비 자세...")
                    movej([-30.98, -1.46, 96.56, 24.10, 81.27, 76.75], vel=60, acc=60)
                    time.sleep(1.0); node.publish_status("arrived"); continue

                # [기타] 홈 위치
                elif msg_val == "START_TASK":
                    movej([0,0,90,0,90,0], vel=60, acc=60)
                    gripper.open(); node.publish_status("arrived"); continue

                movel(posx(target), vel=60, acc=60)
                node.publish_status("arrived")

            elif msg_type == "POSE":
                curr_pos, _ = get_current_posx()
                if curr_pos:
                    tx, ty, tz = msg_val.position.x, msg_val.position.y, msg_val.position.z
                    if abs(tx) < 10.0: tx*=1000; ty*=1000; tz*=1000
                    target = [tx, ty, tz, curr_pos[3], curr_pos[4], curr_pos[5]]
                    movel(posx(target), vel=100, acc=100)
                    time.sleep(0.5); node.publish_status("arrived_target")

            elif msg_type == "JOINT":
                movej(msg_val, vel=60, acc=60); time.sleep(0.5); node.publish_status("joint_arrived")

            elif msg_type == "JOG":
                curr_j = get_current_posj()
                if curr_j:
                    t = list(curr_j)
                    if msg_val == "TURN_FRONT": t[5] -= 15.0
                    elif msg_val == "TURN_BACK": t[5] += 15.0
                    movej(t, vel=60, acc=60); node.publish_status("jog_done")

        except Exception as e:
            print(f"💥 동작 실패: {e}")
            is_error_mode = True
            node.publish_status("fail: collision_detected")
            
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()