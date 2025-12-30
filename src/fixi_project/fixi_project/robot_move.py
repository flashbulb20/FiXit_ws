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

# 전역 함수
movej = None
movel = None
posx = None

# ---------------------------------------------------------
# [Helper] 로봇 이동 대기 함수 (새로 추가됨!)
# ---------------------------------------------------------
def wait_motion(node, timeout=15.0):
    """
    로봇이 물리적으로 이동을 완료할 때까지 기다리는 함수
    (spin_once를 내부에서 호출하여 위치 업데이트를 계속 수신함)
    """
    # 1. 명령이 전달되고 로봇이 반응할 시간을 줌 (매우 중요)
    time.sleep(0.5) 
    
    start_time = time.time()
    last_pos = list(node.current_posx) if node.current_posx else []
    
    print("⏳ 이동 중...", end="", flush=True)
    
    while time.time() - start_time < timeout:
        # 대기 중에도 위치 업데이트를 받아야 하므로 spin_once 필수
        rclpy.spin_once(node, timeout_sec=0.1)
        
        if node.current_posx is None: continue
        
        curr_pos = node.current_posx
        diff = 0
        if len(last_pos) > 0:
            diff = sum(abs(curr_pos[i] - last_pos[i]) for i in range(3))
        
        # 변화량이 거의 없으면 도착으로 간주
        if diff < 1.0:
            print(" -> ✅ 도착 완료")
            return True
            
        last_pos = list(curr_pos)
        
    print(" -> ⚠️ 타임아웃 (이동 시간 초과)")
    return True

# --- 그리퍼 컨트롤러 ---
class GripperController:
    def __init__(self, ip, port):
        self.client = ModbusClient(ip, port=int(port), stopbits=1, bytesize=8, parity='E', baudrate=115200, timeout=1)
        self.max_width = 1100
        try:
            if self.client.connect():
                print(f"✅ 그리퍼({ip}) 연결 성공")
        except Exception: pass

    def open(self, force=400):
        try:
            self.client.write_registers(address=0, values=[force, self.max_width, 16], unit=65)
            time.sleep(1.0) 
        except: pass

    def close(self, force=300):
        try:
            self.client.write_registers(address=0, values=[force, 0, 16], unit=65)
            time.sleep(1.0)
        except: pass

# --- 리스너 노드 ---
class RobotListener(Node):
    def __init__(self, cmd_queue):
        super().__init__("rokey_listener", namespace=ROBOT_ID)
        self.cmd_queue = cmd_queue
        
        # Pub/Sub
        self.status_pub = self.create_publisher(String, '/robot/status', 10)
        self.create_subscription(String, '/robot/nudge_cmd', self.nudge_callback, 10)
        self.create_subscription(String, '/robot/gripper', self.gripper_callback, 10)
        self.create_subscription(String, '/robot/jog', self.jog_callback, 10)
        self.create_subscription(PoseStamped, '/robot/target_pose', self.pose_callback, 10)
        self.create_subscription(Float64MultiArray, '/robot/target_joint', self.joint_callback, 10)
        
        # 데이터 수신
        self.create_subscription(Float64MultiArray, f'/{ROBOT_ID}/msg/current_posx', self.posx_cb, 10)
        self.create_subscription(Float64MultiArray, f'/{ROBOT_ID}/msg/joint_state', self.posj_cb, 10)
        
        self.current_posx = None
        self.current_posj = None 

    def posx_cb(self, msg): self.current_posx = [float(x) for x in msg.data] # float 변환
    def posj_cb(self, msg): self.current_posj = [float(x) for x in msg.data] # float 변환

    def nudge_callback(self, msg):
        self.cmd_queue.put(("CMD", msg.data.upper().strip()))
        self.get_logger().info(f"📥 Nudge 명령: {msg.data}")

    def gripper_callback(self, msg):
        self.cmd_queue.put(("GRIPPER", msg.data.lower().strip()))
        self.get_logger().info(f"📥 Gripper 명령: {msg.data}")

    def jog_callback(self, msg):
        self.cmd_queue.put(("JOG", msg.data.upper().strip()))
        self.get_logger().info(f"📥 Jog 명령: {msg.data}")
    
    def pose_callback(self, msg):
        self.get_logger().info(f"📥 좌표 명령 수신")
        self.cmd_queue.put(("POSE", msg.pose))
    
    def joint_callback(self, msg):
        self.cmd_queue.put(("JOINT", list(msg.data)))

    def publish_status(self, msg):
        self.status_pub.publish(String(data=msg))
        self.get_logger().info(f"📤 상태 발행: {msg}")


# --- 메인 실행 로직 ---
def main(args=None):
    rclpy.init(args=args)
    cmd_queue = queue.Queue()

    node = RobotListener(cmd_queue)
    DR_init.__dsr__node = node

    global movej, movel, posx
    try:
        from DSR_ROBOT2 import movej, movel
        from DR_common2 import posx
        print("✅ DSR 라이브러리 로드 성공")
    except ImportError:
        print(f"⚠️ 라이브러리 로드 실패 (테스트 모드)")
        def movej(*args, **kwargs): time.sleep(1)
        def movel(*args, **kwargs): time.sleep(1)
        def posx(l): return l

    gripper = GripperController("192.168.1.1", 502)

    print("=== 모든 기능 준비 완료 (Safe Mode) ===")
    print("📌 ROS 메시지 처리: spin_once 방식")

    while rclpy.ok():
        try:
            # ROS 메시지 수신
            rclpy.spin_once(node, timeout_sec=0.1)
            
            # 대기열 확인
            try:
                msg_type, msg_val = cmd_queue.get_nowait()
            except queue.Empty:
                continue

            print(f"\n⚙️ [실행] {msg_type}")
            
            # --- CASE 1: 그리퍼 ---
            if msg_type == "GRIPPER":
                if msg_val == "open":
                    print("🖐️ 그리퍼 열기")
                    gripper.open()
                    node.publish_status("opened")
                elif msg_val == "close":
                    print("✊ 그리퍼 닫기")
                    gripper.close()
                    node.publish_status("gripped")
            
            # --- CASE 2: JOG ---
            elif msg_type == "JOG":
                if node.current_posj is None:
                    print("⚠️ 관절 정보 대기 중...")
                    time.sleep(0.5)
                    continue

                target_j = list(node.current_posj)
                angle = 15.0 
                
                if msg_val == "TURN_FRONT": target_j[5] += angle
                elif msg_val == "TURN_BACK": target_j[5] -= angle
                
                if movej:
                    movej(target_j, vel=60.0, acc=60.0) # float 속도
                    wait_motion(node) # [추가] 대기
                    node.publish_status("jog_done")

            # --- CASE 3: JOINT ---
            elif msg_type == "JOINT":
                # 모든 값을 float로 변환
                target_joints = [float(x) for x in msg_val]
                print(f"🤖 관절 이동: {target_joints[:3]}...")
                
                if movej:
                    movej(target_joints, vel=60.0, acc=60.0)
                    wait_motion(node) # [추가] 대기
                    node.publish_status("joint_arrived")

            # --- CASE 4: POSE (수정됨) ---
            elif msg_type == "POSE":
                if node.current_posx is None:
                    print("⚠️ 현재 위치 대기 중...")
                    time.sleep(0.5)
                    continue
                
                # [수정] 입력값 안전 변환 (Vision에서 mm로 준다고 가정)
                tx = float(msg_val.position.x)
                ty = float(msg_val.position.y)
                tz = float(msg_val.position.z)
                
                # [수정] 현재 자세(Rotation) 유지 + float 형변환
                curr = node.current_posx
                target_pos = [tx, ty, tz, float(curr[3]), float(curr[4]), float(curr[5])]
                
                print(f"📍 좌표 이동: {target_pos[:3]}")
                
                if movel:
                    movel(posx(target_pos), vel=100.0, acc=100.0)
                    wait_motion(node) # [추가] 실제 이동 대기
                    node.publish_status("arrived_target")

            # --- CASE 5: CMD (수정됨) ---
            elif msg_type == "CMD":
                raw_cmd = msg_val
                
                # 1. HOME 이동
                if raw_cmd == "START_TASK":
                    print("🏠 홈 위치로 이동")
                    home_joint = [0.0, 0.0, 90.0, 0.0, 90.0, 0.0]
                    movej(home_joint, vel=60.0, acc=60.0)
                    wait_motion(node) # 혹은 wait_motion
                    gripper.open()
                    node.publish_status("arrived")
                    continue

                # [추가] 2. HANDOVER 위치 (Joint 이동)
                elif raw_cmd == "HANDOVER":
                    print("🚚 전달(Handover) 위치로 이동")
                    # 전달해주신 posj 좌표
                    handover_joint = [5.784, 0.719, 86.65, -4.977, 51.269, 7.344]
                    movej(handover_joint, vel=60.0, acc=60.0)
                    wait_motion(node)
                    node.publish_status("arrived")
                    continue

                # [추가] 3. SCAN 위치 (Joint 이동)
                elif raw_cmd == "SCAN":
                    print("👁️ 관측(Scan) 위치로 이동")
                    # 전달해주신 posj 좌표
                    scan_joint = [-51.291, -38.599, 129.907, 55.324, 68.009, 9.435]
                    movej(scan_joint, vel=60.0, acc=60.0)
                    wait_motion(node)
                    node.publish_status("arrived")
                    continue

                if node.current_posx is None: continue

                target = list(node.current_posx)
                dist = 30.0
                
                if raw_cmd == "UP": target[2] += dist
                elif raw_cmd == "DOWN": target[2] -= dist
                elif raw_cmd == "LEFT": target[1] += dist
                elif raw_cmd == "RIGHT": target[1] -= dist
                elif raw_cmd == "FORWARD": target[0] += dist
                elif raw_cmd == "BACKWARD": target[0] -= dist
                elif raw_cmd == "DOWN_PICK": target[2] -= 60.0
                elif raw_cmd == "UP_PICK": target[2] += 150.0
                
                # [수정] 안전한 float 리스트로 세탁
                clean_target = [float(x) for x in target]
                
                if movel:
                    movel(posx(clean_target), vel=60.0, acc=60.0)
                    wait_motion(node) # [추가] 대기
                    node.publish_status("arrived")

        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(0.5)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()