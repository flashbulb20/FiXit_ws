import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float64MultiArray
from geometry_msgs.msg import PoseStamped
from pymodbus.client.sync import ModbusTcpClient as ModbusClient
import time
import queue

# [1] 초기 설정
import DR_init
ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

# 전역 함수 선언
movej = None; movel = None; posx = None
get_robot_state = None; set_safe_stop_reset_type = None

def wait_motion(node, timeout=15.0):
    time.sleep(0.5) 
    start_time = time.time()
    last_pos = list(node.current_posx) if node.current_posx else []
    print("⏳ 이동 중...", end="", flush=True)
    
    while rclpy.ok() and (time.time() - start_time < timeout):
        rclpy.spin_once(node, timeout_sec=0.1)
        if node.current_posx is None: continue
        curr_pos = node.current_posx
        diff = sum(abs(curr_pos[i] - last_pos[i]) for i in range(3)) if last_pos else 100.0
        if diff < 1.0:
            print(" -> ✅ 도착 완료")
            return True
        last_pos = list(curr_pos)
    return True

class GripperController:
    def __init__(self, ip, port):
        self.client = ModbusClient(ip, port=int(port), stopbits=1, bytesize=8, parity='E', baudrate=115200, timeout=1)
        self.max_width = 1100
        try:
            if self.client.connect(): print(f"✅ 그리퍼({ip}) 연결 성공")
        except: pass
    def open(self, force=400):
        try: self.client.write_registers(address=0, values=[force, self.max_width, 16], unit=65)
        except: pass
    def close(self, force=300):
        try: self.client.write_registers(address=0, values=[force, 0, 16], unit=65)
        except: pass

class RobotListener(Node):
    def __init__(self, cmd_queue):
        # ✅ [수정] namespace 제거하여 토픽 경로를 깔끔하게 만듦
        super().__init__("rokey_listener",  namespace=ROBOT_ID)
        self.cmd_queue = cmd_queue
        self.ERROR_STATES = {3, 5, 6, 9, 10} 
        self.last_error_notified = False
        
        from dsr_msgs2.srv import SetRobotControl
        # ✅ [수정] 서비스 경로도 절대 경로로 지정
        self.srv_client = self.create_client(SetRobotControl, f'/{ROBOT_ID}/system/set_robot_control')
        
        # ✅ 모든 토픽 이름을 절대 경로('/')로 시작하도록 설정
        self.status_pub = self.create_publisher(String, '/robot/status', 10)
        self.create_subscription(String, '/robot/nudge_cmd', self.nudge_callback, 10)
        self.create_subscription(String, '/robot/gripper', self.gripper_callback, 10)
        self.create_subscription(String, '/robot/jog', self.jog_callback, 10)
        self.create_subscription(PoseStamped, '/robot/target_pose', self.pose_callback, 10)
        self.create_subscription(Float64MultiArray, '/robot/target_joint', self.joint_callback, 10)
        
        # 로봇 데이터 수신은 ROBOT_ID 네임스페이스 유지
        self.create_subscription(Float64MultiArray, f'/{ROBOT_ID}/msg/current_posx', self.posx_cb, 10)
        self.create_subscription(Float64MultiArray, f'/{ROBOT_ID}/msg/joint_state', self.posj_cb, 10)
        
        self.current_posx = None
        self.current_posj = None 
        

    def check_collision_and_notify(self):
        if get_robot_state is None: return
        state = get_robot_state()
        if state in self.ERROR_STATES:
            if not self.last_error_notified:
                self.publish_status("fail: collision_detected")
                self.last_error_notified = True
        else:
            self.last_error_notified = False

    def execute_recovery(self):
        from dsr_msgs2.srv import SetRobotControl
        srv_client = self.create_client(SetRobotControl, f'/{ROBOT_ID}/system/set_robot_control')
        
        def call_srv(val):
            if srv_client.wait_for_service(timeout_sec=1.0):
                req = SetRobotControl.Request()
                req.robot_control = val
                srv_client.call_async(req)

        self.get_logger().info("🛠️ 복구 시퀀스 시작...")
        set_safe_stop_reset_type(2)
        call_srv(3); time.sleep(1.0) # OFF_RESET
        call_srv(2); time.sleep(1.0) # STOP_RESET
        call_srv(1); time.sleep(2.0) # SERVO_ON
        self.publish_status("recovery_success")

    def _send_recovery_cmd(self, val):
        from dsr_msgs2.srv import SetRobotControl
        if self.srv_client.wait_for_service(timeout_sec=0.1):
            req = SetRobotControl.Request()
            req.robot_control = val
            self.srv_client.call_async(req)

    def posx_cb(self, msg): self.current_posx = [float(x) for x in msg.data]
    def posj_cb(self, msg): self.current_posj = [float(x) for x in msg.data]
    
    def nudge_callback(self, msg):
        cmd = msg.data.upper().strip()
        if cmd == "RECOVERY":
            self.cmd_queue.put(("RECOVERY", None)) # 큐에 복구 명령 넣기
        else:
            self.cmd_queue.put(("CMD", cmd))
        self.get_logger().info(f"📥 Nudge 명령: {cmd}")

    def gripper_callback(self, msg): self.cmd_queue.put(("GRIPPER", msg.data.lower().strip()))
    def jog_callback(self, msg): self.cmd_queue.put(("JOG", msg.data.upper().strip()))
    def pose_callback(self, msg): self.cmd_queue.put(("POSE", msg.pose))
    def joint_callback(self, msg): self.cmd_queue.put(("JOINT", list(msg.data)))

    def publish_status(self, msg):
        self.status_pub.publish(String(data=msg))
        self.get_logger().info(f"📤 상태 발행: {msg}")

def main(args=None):
    rclpy.init(args=args)
    cmd_queue = queue.Queue()

    node = RobotListener(cmd_queue)
    DR_init.__dsr__node = node

    global movej, movel, posx, get_robot_state, set_safe_stop_reset_type
    try:
        from DSR_ROBOT2 import movej, movel, get_robot_state, set_safe_stop_reset_type
        from DR_common2 import posx
        print("✅ 모든 시스템 준비 완료. 명령을 기다립니다...")
    except Exception as e:
        print(f"❌ 라이브러리 로드 실패: {e}"); return
        
    gripper = GripperController("192.168.1.1", 502)

    while rclpy.ok():
        try:
            rclpy.spin_once(node, timeout_sec=0.1)
            node.check_collision_and_notify()
            try:
                msg_type, msg_val = cmd_queue.get_nowait()
            except queue.Empty: continue

            print(f"⚙️ [실행 중] {msg_type}: {msg_val}")
            
            if msg_type == "RECOVERY":
                node.execute_recovery()

            if msg_type == "GRIPPER":
                if msg_val == "open": gripper.open(); node.publish_status("opened")
                elif msg_val == "close": gripper.close(); node.publish_status("gripped")

            elif msg_type == "JOG":
                if node.current_posj:
                    target_j = list(node.current_posj)
                    if msg_val == "TURN_FRONT": target_j[5] += 15.0
                    elif msg_val == "TURN_BACK": target_j[5] -= 15.0
                    movej(target_j, vel=60.0, acc=60.0); wait_motion(node); node.publish_status("jog_done")
            elif msg_type == "JOINT":
                movej([float(x) for x in msg_val], vel=60.0, acc=60.0); wait_motion(node); node.publish_status("joint_arrived")
            elif msg_type == "POSE":
                if node.current_posx:
                    curr = node.current_posx
                    target_pos = [float(msg_val.position.x), float(msg_val.position.y), float(msg_val.position.z), 
                                  float(curr[3]), float(curr[4]), float(curr[5])]
                    movel(posx(target_pos), vel=100.0, acc=100.0); wait_motion(node); node.publish_status("arrived_target")
            elif msg_type == "CMD":
                if msg_val == "START_TASK":
                    movej([0.0, 0.0, 90.0, 0.0, 90.0, 0.0], vel=60.0, acc=60.0); wait_motion(node); gripper.open(); node.publish_status("arrived")
                elif msg_val == "HANDOVER":
                    movej([-17.364, -0.953, 114.536, 26.699, 18.251, 59.157], vel=60.0, acc=60.0); wait_motion(node); node.publish_status("arrived")
                elif msg_val == "SCAN":
                    movej([-51.291, -38.599, 129.907, 55.324, 68.009, 9.435], vel=60.0, acc=60.0); wait_motion(node); node.publish_status("arrived")
                elif msg_val in ["UP", "DOWN", "LEFT", "RIGHT", "FORWARD", "BACKWARD", "DOWN_PICK", "UP_PICK"]:
                    if node.current_posx:
                        target = list(node.current_posx)
                        d = 30.0
                        if msg_val == "UP": target[2] += d
                        elif msg_val == "DOWN": target[2] -= d
                        elif msg_val == "LEFT": target[1] += d
                        elif msg_val == "RIGHT": target[1] -= d
                        elif msg_val == "FORWARD": target[0] += d
                        elif msg_val == "BACKWARD": target[0] -= d
                        elif msg_val == "DOWN_PICK": target[2] -= 60.0
                        elif msg_val == "UP_PICK": target[2] += 150.0
                        movel(posx([float(x) for x in target]), vel=60.0, acc=60.0); wait_motion(node); node.publish_status("arrived")
        except Exception as e:
            print(f"❌ 실행 에러: {e}")

    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()