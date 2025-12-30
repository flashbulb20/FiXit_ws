import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
import time
import threading
import math

class MainController(Node):
    def __init__(self):
        super().__init__('main_controller')
        self.get_logger().info("🚀 Fixit Main Controller Started (Debug Mode On)")

        # =========================================================
        # 1. 통신 인터페이스 설정
        # =========================================================
        self.create_subscription(String, '/voice/cmd', self.voice_callback, 10)
        self.create_subscription(String, '/robot/status', self.robot_status_callback, 10)
        self.create_subscription(PoseStamped, '/vision/target_pose', self.vision_pose_callback, 10)

        self.robot_pose_pub = self.create_publisher(PoseStamped, '/robot/target_pose', 10)
        self.robot_nudge_pub = self.create_publisher(String, '/robot/nudge_cmd', 10)
        self.robot_jog_pub = self.create_publisher(String, '/robot/jog', 10)
        self.robot_gripper_pub = self.create_publisher(String, '/robot/gripper', 10)
        self.vision_cmd_pub = self.create_publisher(String, '/vision/cmd', 10)
        self.tts_pub = self.create_publisher(String, '/voice/tts', 10)

        # =========================================================
        # 2. 상태 및 변수 초기화
        # =========================================================
        self.latest_robot_status = ""
        self.detected_pose = None
        self.state = "IDLE"
        self.target_type = "TOOL"

        # 좌표 설정 (로그로 확인)
        self.HANDOVER_POSE = self.set_doosan_pose(608.635, 38.081, 292.748, 179.445, -138.384, 179.867)
        self.SCAN_POSE = self.set_doosan_pose(369.391, -15.869, 255.78, 12.993, 122.119, 86.479)
        self.HOLD_READY_POSE = self.HANDOVER_POSE
        
        self.get_logger().info(f"📍 Initialized Handover Pose: {self.HANDOVER_POSE.pose.position}")

    def set_doosan_pose(self, x, y, z, rx, ry, rz):
        p = PoseStamped()
        p.header.frame_id = "base_0"
        p.pose.position.x = x / 1000.0
        p.pose.position.y = y / 1000.0
        p.pose.position.z = z / 1000.0
        q = self.euler_to_quaternion(math.radians(rx), math.radians(ry), math.radians(rz))
        p.pose.orientation.x, p.pose.orientation.y = q[0], q[1]
        p.pose.orientation.z, p.pose.orientation.w = q[2], q[3]
        return p

    def euler_to_quaternion(self, roll, pitch, yaw):
        qx = math.sin(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) - math.cos(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
        qy = math.cos(roll/2) * math.sin(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.cos(pitch/2) * math.sin(yaw/2)
        qz = math.cos(roll/2) * math.cos(pitch/2) * math.sin(yaw/2) - math.sin(roll/2) * math.sin(pitch/2) * math.cos(yaw/2)
        qw = math.cos(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
        return [qx, qy, qz, qw]

    # =========================================================
    # 3. 콜백 함수 (디버깅 로그 추가)
    # =========================================================
    def robot_status_callback(self, msg):
        self.latest_robot_status = msg.data
        # 너무 자주 들어오면 주석 처리 가능
        self.get_logger().debug(f"🤖 Robot Status: {self.latest_robot_status}")

    def vision_pose_callback(self, msg):
        self.detected_pose = msg
        # 시각적 확인을 위해 x, y 좌표만 로그 출력
        self.get_logger().debug(f"👁️ Vision Detected: x={msg.pose.position.x:.2f}, y={msg.pose.position.y:.2f}")

    def voice_callback(self, msg):
        command = msg.data.lower().strip()
        self.get_logger().info(f"\n📢 [VOICE CMD] Received: '{command}'")

        if "stop" in command or "멈춰" in command:
            self.stop_all()
            return
        
        if command == "go_home" or command == "home":
            self.get_logger().info("🏠 Action: Go Home")
            self.robot_nudge_pub.publish(String(data="HOME"))
            return

        if "_" in command:
            parts = command.split("_")
            action, target = parts[0], parts[1] if len(parts) > 1 else ""
        else:
            action, target = command, ""

        self.get_logger().info(f"🧩 Parsed -> Action: {action}, Target: {target}")

        # --- Logic Tree ---
        if action == "move" or action == "nudge":
            self.robot_nudge_pub.publish(String(data=target.upper()))
        
        elif action == "fetch" or action == "bring":
            self.get_logger().info("▶️ Starting Thread: sequence_fetch")
            threading.Thread(target=self.sequence_fetch, args=(target,)).start()

        elif action == "hold" or action == "grab":
            if "pcb" in target:
                self.target_type = "PCB"
                threading.Thread(target=self.sequence_hold_pcb).start()
            elif "here" in target:
                self.target_type = "TOOL"
                threading.Thread(target=self.sequence_hold_point).start()
            else: # catch trigger logic is handled below
                 pass

        elif (action == "catch" or action == "grab") and self.state == "WAITING_FOR_CATCH":
            self.get_logger().info("🔒 Catch Triggered! Executing Grip.")
            if self.target_type == "PCB":
                self.robot_gripper_pub.publish(String(data="close_pcb"))
                self.tts_pub.publish(String(data="기판을 잡았습니다."))
            else:
                self.robot_gripper_pub.publish(String(data="close_tool"))
                self.tts_pub.publish(String(data="잡았습니다."))
            self.state = "IDLE"
            self.get_logger().info("✅ Catch Sequence Complete. State -> IDLE")

    # =========================================================
    # 4. 시퀀스 로직 (Wait 로그 강화)
    # =========================================================
    def wait_for_robot(self, target_status, timeout=10.0):
        self.latest_robot_status = ""
        self.get_logger().info(f"⏳ Waiting for Robot status: '{target_status}' (Timeout: {timeout}s)")
        
        start = time.time()
        while time.time() - start < timeout:
            if self.latest_robot_status == target_status:
                self.get_logger().info(f"✅ Robot Reached: '{target_status}' ({time.time()-start:.2f}s)")
                return True
            time.sleep(0.1)
        
        self.get_logger().error(f"❌ Robot Timeout! Waited for '{target_status}' but got '{self.latest_robot_status}'")
        return False

    def wait_for_vision(self, timeout=5.0):
        self.detected_pose = None
        self.get_logger().info(f"⏳ Waiting for Vision Target... (Timeout: {timeout}s)")
        
        start = time.time()
        while time.time() - start < timeout:
            if self.detected_pose is not None:
                p = self.detected_pose.pose.position
                self.get_logger().info(f"✅ Vision Target Found: ({p.x:.2f}, {p.y:.2f}, {p.z:.2f})")
                return self.detected_pose
            time.sleep(0.1)
        
        self.get_logger().warn("❌ Vision Timeout! No target detected.")
        return None

    # --- [A. 도구 가져오기] ---
    def sequence_fetch(self, tool_name):
        self.state = "FETCHING"
        self.get_logger().info(f"--- [START] Sequence Fetch: {tool_name} ---")
        self.tts_pub.publish(String(data=f"{tool_name}를 찾고 있습니다."))
        
        # 1. Vision
        self.vision_cmd_pub.publish(String(data=f"find_{tool_name}"))
        target_pose = self.wait_for_vision()
        if not target_pose: 
            self.tts_pub.publish(String(data="물건을 찾을 수 없습니다."))
            self.state = "IDLE"
            return

        # 2. Move
        self.get_logger().info("🚀 Moving Robot to Target...")
        self.robot_pose_pub.publish(target_pose)
        if not self.wait_for_robot("arrived"): return

        # 3. Grip
        grip_cmd = "close_tool" if "flux" not in tool_name else "close_pcb"
        self.get_logger().info(f"✊ Gripping ({grip_cmd})...")
        self.robot_gripper_pub.publish(String(data=grip_cmd))
        if not self.wait_for_robot("gripped"): return

        # 4. Handover
        self.get_logger().info("🚚 Moving to Handover Position...")
        self.robot_pose_pub.publish(self.HANDOVER_POSE)
        if not self.wait_for_robot("arrived"): return

        self.robot_gripper_pub.publish(String(data="open"))
        self.tts_pub.publish(String(data="여기 있습니다."))
        self.get_logger().info("--- [END] Sequence Fetch Complete ---")
        
        # 복귀
        self.robot_nudge_pub.publish(String(data="HOME"))
        self.state = "IDLE"

    # --- [B. PCB 잡아주기] ---
    def sequence_hold_pcb(self):
        self.state = "PREPARING_HOLD"
        self.get_logger().info("--- [START] Sequence Hold PCB ---")
        self.tts_pub.publish(String(data="파지 준비 중입니다."))

        self.robot_pose_pub.publish(self.HOLD_READY_POSE)
        if not self.wait_for_robot("arrived"): return

        self.robot_gripper_pub.publish(String(data="open"))
        self.wait_for_robot("gripped_open")

        self.tts_pub.publish(String(data="기판을 넣고 잡아, 라고 말해주세요."))
        self.state = "WAITING_FOR_CATCH"
        self.get_logger().info("💤 State changed to WAITING_FOR_CATCH")

    # --- [D. 가리킨 곳 잡아주기] ---
    def sequence_hold_point(self):
        self.state = "SCANNING"
        self.get_logger().info("--- [START] Sequence Hold Point (Here) ---")
        self.tts_pub.publish(String(data="손을 보여주세요."))

        self.robot_pose_pub.publish(self.SCAN_POSE)
        if not self.wait_for_robot("arrived"): return

        self.vision_cmd_pub.publish(String(data="track_hand"))
        hand_pose = self.wait_for_vision(timeout=5.0)

        if not hand_pose:
            self.tts_pub.publish(String(data="손을 놓쳤습니다."))
            self.state = "IDLE"
            return

        self.get_logger().info("🚀 Moving to Hand Position...")
        self.robot_gripper_pub.publish(String(data="open"))
        self.robot_pose_pub.publish(hand_pose)
        if not self.wait_for_robot("arrived"): return

        self.tts_pub.publish(String(data="잡을까요? 잡아, 라고 말해주세요."))
        self.state = "WAITING_FOR_CATCH"
        self.get_logger().info("💤 State changed to WAITING_FOR_CATCH")

    def stop_all(self):
        self.get_logger().warn("🚨 EMERGENCY STOP TRIGGERED")
        self.robot_jog_pub.publish(String(data="STOP"))
        self.state = "IDLE"

def main(args=None):
    rclpy.init(args=args)
    node = MainController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
