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
        # [4] 좌표 이동 명령
        self.create_subscription(PoseStamped, '/robot/target_pose', self.pose_callback, 10)
        # [5] 관절 이동 명령
        self.create_subscription(Float64MultiArray, '/robot/target_joint', self.joint_callback, 10)
        
        # 데이터 수신
        self.create_subscription(Float64MultiArray, f'/{ROBOT_ID}/msg/current_posx', self.posx_cb, 10)
        self.create_subscription(Float64MultiArray, f'/{ROBOT_ID}/msg/joint_state', self.posj_cb, 10)
        
        self.current_posx = None
        self.current_posj = None 

    def posx_cb(self, msg): 
        self.current_posx = list(msg.data)

    def posj_cb(self, msg): 
        self.current_posj = list(msg.data)

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
        self.get_logger().info(f"📥 좌표 명령: x={msg.pose.position.x:.3f}, y={msg.pose.position.y:.3f}, z={msg.pose.position.z:.3f}")
        self.cmd_queue.put(("POSE", msg.pose))
    
    def joint_callback(self, msg):
        self.get_logger().info(f"📥 관절 명령: {[round(x, 2) for x in msg.data]}")
        self.cmd_queue.put(("JOINT", list(msg.data)))

    def publish_status(self, msg):
        status = String(data=msg)
        self.status_pub.publish(status)
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
    except ImportError as e:
        print(f"❌ 라이브러리 로드 실패: {e}")
        return

    gripper = GripperController("192.168.1.1", 502)

    # ✅ 백그라운드 스레드 없이 메인 루프에서 spin_once 사용
    print("=== 모든 기능 준비 완료 (Nudge / Gripper / Jog / Pose / Joint) ===")
    print("📌 ROS 메시지 처리: spin_once 방식 (Thread-Safe)")

    # 메인 루프 (순차 실행기)
    while rclpy.ok():
        try:
            # ✅ ROS 메시지 수신 (non-blocking, 100ms timeout)
            rclpy.spin_once(node, timeout_sec=0.1)
            
            # (1) 대기열 확인 (non-blocking)
            try:
                msg_type, msg_val = cmd_queue.get_nowait()
            except queue.Empty:
                continue

            print(f"\n⚙️ [실행 시작] {msg_type} → {msg_val}")
            
            # --- CASE 1: 그리퍼 ---
            if msg_type == "GRIPPER":
                if msg_val == "open":
                    print("🖐️ 그리퍼 열기")
                    gripper.open()
                    node.publish_status("gripper_opened")
                elif msg_val == "close":
                    print("✊ 그리퍼 닫기")
                    gripper.close()
                    node.publish_status("gripper_closed")
            
            # --- CASE 2: JOG (관절 회전) ---
            elif msg_type == "JOG":
                if node.current_posj is None:
                    print("⚠️ 관절 정보 수신 대기 중...")
                    time.sleep(0.5)
                    continue

                target_j = node.current_posj[:] 
                angle = 15.0 
                
                if msg_val == "TURN_FRONT": 
                    target_j[5] += angle
                    print(f"🔄 J6 회전: +{angle}도")
                elif msg_val == "TURN_BACK": 
                    target_j[5] -= angle
                    print(f"🔄 J6 회전: -{angle}도")
                
                if movej:
                    movej(target_j, vel=60, acc=60)
                    node.publish_status("jog_done")

            # --- CASE 3: JOINT (관절 각도로 이동) ---
            elif msg_type == "JOINT":
                target_joints = msg_val
                print(f"🤖 관절 이동: {[round(x, 1) for x in target_joints]}")
                
                if movej:
                    movej(target_joints, vel=60, acc=60)
                    node.publish_status("joint_arrived")

            # --- CASE 4: POSE (지정 좌표 이동) ---
            elif msg_type == "POSE":
                if node.current_posx is None:
                    print("⚠️ 현재 위치 확인 불가 (대기 중)")
                    time.sleep(0.5)
                    continue
                
                # ROS(m) -> Robot(mm) 변환
                tx = msg_val.position.x
                ty = msg_val.position.y
                tz = msg_val.position.z
                
                # 방향(Orientation)은 현재 로봇 상태 유지
                curr = node.current_posx
                target_pos = [tx, ty, tz, curr[3], curr[4], curr[5]]
                
                print(f"📍 좌표 이동: X={tx:.1f}, Y={ty:.1f}, Z={tz:.1f}")
                
                if movel:
                    movel(posx(target_pos), vel=60, acc=60)
                    node.publish_status("arrived_target")

            # --- CASE 5: CMD (NUDGE 및 기본동작) ---
            elif msg_type == "CMD":
                raw_cmd = msg_val
                
                if node.current_posx is None and raw_cmd != "START_TASK":
                    print("⚠️ 좌표 수신 대기 중...")
                    time.sleep(0.5)
                    continue
                
                # HOME 위치로 이동
                if raw_cmd == "START_TASK":
                    print("🏠 홈 위치로 이동")
                    movej([0, 0, 90, 0, 90, 0], vel=60, acc=60)
                    gripper.open()
                    node.publish_status("arrived")
                    continue

                target = node.current_posx[:]
                dist = 30.0  # 기본 이동 거리
                
                # 기본 방향 명령
                if raw_cmd == "UP": 
                    target[2] += dist
                    print(f"⬆️ 상승: +{dist}mm")
                elif raw_cmd == "DOWN": 
                    target[2] -= dist
                    print(f"⬇️ 하강: -{dist}mm")
                elif raw_cmd == "LEFT": 
                    target[1] += dist
                    print(f"⬅️ 좌측: +{dist}mm")
                elif raw_cmd == "RIGHT": 
                    target[1] -= dist
                    print(f"➡️ 우측: -{dist}mm")
                elif raw_cmd == "FORWARD": 
                    target[0] += dist
                    print(f"⬆️ 전진: +{dist}mm")
                elif raw_cmd == "BACKWARD": 
                    target[0] -= dist
                    print(f"⬇️ 후진: -{dist}mm")
                
                # 픽업용 명령
                elif raw_cmd == "DOWN_PICK":
                    target[2] -= 60.0
                    print(f"⬇️ 픽업 하강: -150mm")
                elif raw_cmd == "UP_PICK":
                    target[2] += 150.0
                    print(f"⬆️ 픽업 상승: +150mm")
                else:
                    print(f"⚠️ 알 수 없는 명령: {raw_cmd}")
                    continue
                
                if movel:
                    movel(posx(target), vel=60, acc=60)
                    node.publish_status("arrived")

            else:
                print(f"⚠️ 알 수 없는 타입: {msg_type}")

            print(f"✅ [실행 완료]\n")
            
            # ✅ 명령 실행 후 즉시 ROS 메시지 처리
            for _ in range(5):
                rclpy.spin_once(node, timeout_sec=0.05)

        except Exception as e:
            print(f"❌ Main Loop Error: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(0.5)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()