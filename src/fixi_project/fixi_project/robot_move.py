import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float64MultiArray
from geometry_msgs.msg import PoseStamped # [추가] 좌표 메시지
from pymodbus.client.sync import ModbusTcpClient as ModbusClient
import threading
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

# --- 그리퍼 컨트롤러 ---
class GripperController:
    def __init__(self, ip, port):
        self.client = ModbusClient(ip, port=int(port), stopbits=1, bytesize=8, parity='E', baudrate=115200, timeout=1)
        self.max_width = 1100
        try:
            if self.client.connect():
                print(f"✅ 그리퍼({ip}) 연결 성공")
            else:
                print(f"⚠️ 그리퍼({ip}) 연결 실패")
        except Exception as e:
            print(f"⚠️ 그리퍼 에러: {e}")

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

# --- 리스너 노드 (모든 명령 수집) ---
class RobotListener(Node):
    def __init__(self, cmd_queue):
        super().__init__("rokey_listener", namespace=ROBOT_ID)
        self.cmd_queue = cmd_queue
        
        # Publisher
        self.status_pub = self.create_publisher(String, '/robot/status', 10)
        
        # [1] 이동 명령 (문자열)
        self.create_subscription(String, '/robot/nudge_cmd', self.nudge_callback, 10)
        # [2] 그리퍼 명령
        self.create_subscription(String, '/robot/gripper', self.gripper_callback, 10)
        # [3] Jog(회전) 명령
        self.create_subscription(String, '/robot/jog', self.jog_callback, 10)
        # [4] 좌표 이동 명령 (추가됨!)
        self.create_subscription(PoseStamped, '/robot/target_pose', self.pose_callback, 10)
        
        # 데이터 수신
        self.create_subscription(Float64MultiArray, f'/{ROBOT_ID}/msg/current_posx', self.posx_cb, 10)
        self.create_subscription(Float64MultiArray, f'/{ROBOT_ID}/msg/joint_state', self.posj_cb, 10)
        
        self.current_posx = None
        self.current_posj = None 

    def posx_cb(self, msg): self.current_posx = list(msg.data)
    def posj_cb(self, msg): self.current_posj = list(msg.data)

    def nudge_callback(self, msg):
        # 큐에 넣을 때 (타입, 데이터) 형태로 넣어서 구분합니다.
        self.cmd_queue.put(("CMD", msg.data.upper().strip()))

    def gripper_callback(self, msg):
        self.cmd_queue.put(("GRIPPER", msg.data.lower().strip()))

    def jog_callback(self, msg):
        self.cmd_queue.put(("JOG", msg.data.upper().strip()))
    
    def pose_callback(self, msg):
        # [추가] 좌표 메시지 수신
        self.get_logger().info(f"📥 [좌표] x={msg.pose.position.x}, y={msg.pose.position.y}")
        self.cmd_queue.put(("POSE", msg.pose))

    def publish_status(self, msg):
        self.status_pub.publish(String(data=msg))


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
    except ImportError:
        print("라이브러리 로드 실패")
        return

    gripper = GripperController("192.168.1.1", 502)

    # ROS 수신 스레드 (백그라운드)
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    print("=== 모든 기능 준비 완료 (Nudge / Gripper / Jog / Pose) ===")

    # 메인 루프 (순차 실행기)
    while rclpy.ok():
        try:
            # (1) 대기열 확인 (Type, Value 로 꺼냄)
            try:
                msg_type, msg_val = cmd_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            print(f"⚙️ [Start] 작업 수행: {msg_type} -> {msg_val}")
            
            # --- CASE 1: 그리퍼 ---
            if msg_type == "GRIPPER":
                if msg_val == "open":
                    gripper.open()
                    node.publish_status("gripper_opened")
                elif msg_val == "close":
                    gripper.close()
                    node.publish_status("gripper_closed")
            
            # --- CASE 2: JOG (관절 회전) ---
            elif msg_type == "JOG":
                if node.current_posj is None:
                    print("⚠️ 관절 정보 수신 대기 중...")
                    time.sleep(1)
                    continue

                target_j = node.current_posj[:] 
                angle = 15.0 
                if msg_val == "TURN_FRONT": target_j[5] += angle
                elif msg_val == "TURN_BACK": target_j[5] -= angle
                
                if movej:
                    movej(target_j, vel=60, acc=60)
                    node.publish_status(f"jog_done")

            # --- CASE 3: POSE (지정 좌표 이동) ---
            elif msg_type == "POSE":
                # msg_val은 pose 객체임
                if node.current_posx is None:
                    print("⚠️ 현재 위치 확인 불가 (대기 중)")
                    time.sleep(1)
                    continue
                
                # ROS(m) -> Robot(mm) 변환
                tx = msg_val.position.x * 1000.0
                ty = msg_val.position.y * 1000.0
                tz = msg_val.position.z * 1000.0
                
                # 방향(Orientation)은 현재 로봇 상태 유지 (rx, ry, rz)
                curr = node.current_posx
                target_pos = [tx, ty, tz, curr[3], curr[4], curr[5]]
                
                print(f"📍 목표 이동: {target_pos}")
                if movel:
                    movel(posx(target_pos), vel=60, acc=60)
                    node.publish_status("arrived_target")

            # --- CASE 4: CMD (NUDGE 및 기본동작) ---
            elif msg_type == "CMD":
                raw_cmd = msg_val
                
                if node.current_posx is None and raw_cmd != "START_TASK":
                    print("⚠️ 좌표 수신 대기 중...")
                    time.sleep(1)
                    continue
                
                if raw_cmd == "START_TASK":
                    movej([0, 0, 90, 0, 90, 0], vel=60, acc=60)
                    gripper.open()
                    node.publish_status("arrived")
                    continue

                target = node.current_posx[:]
                dist = 30.0
                
                if raw_cmd == "UP": target[2] += dist
                elif raw_cmd == "DOWN": target[2] -= dist
                elif raw_cmd == "LEFT": target[1] += dist
                elif raw_cmd == "RIGHT": target[1] -= dist
                elif raw_cmd == "FORWARD": target[0] += dist
                elif raw_cmd == "BACKWARD": target[0] -= dist
                
                if movel:
                    movel(posx(target), vel=60, acc=60)
                    node.publish_status("arrived")

            else:
                print(f"⚠️ 알 수 없는 타입: {msg_type}")

            print(f"✅ [End] 작업 완료")

        except Exception as e:
            print(f"Main Loop Error: {e}")
            time.sleep(1)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()