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
        self.get_logger().info("🚀 Fixit Main Controller Started (Joint Mode)")

        # =========================================================
        # 1. 통신 인터페이스 설정
        # =========================================================
        self.create_subscription(String, '/voice/cmd', self.voice_callback, 10)
        self.create_subscription(String, '/robot/status', self.robot_status_callback, 10)
        self.create_subscription(PoseStamped, '/vision/target_pose', self.vision_pose_callback, 10)

        self.robot_pose_pub = self.create_publisher(PoseStamped, '/robot/target_pose', 10)
        self.robot_nudge_pub = self.create_publisher(String, '/robot/nudge_cmd', 10) # 이제 이걸로 명령
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

    # =========================================================
    # 3. 콜백 함수
    # =========================================================
    def robot_status_callback(self, msg):
        self.latest_robot_status = msg.data

    def vision_pose_callback(self, msg):
        self.detected_pose = msg
        self.get_logger().debug(f"👁️ Vision Detected: x={msg.pose.position.x:.2f}, y={msg.pose.position.y:.2f}")

    def voice_callback(self, msg):
        """음성 명령 파싱 및 스레드 분배"""
        command = msg.data.lower().strip()
        self.get_logger().info(f"\n📢 [VOICE CMD] Received: '{command}'")

        if "stop" in command or "멈춰" in command:
            self.stop_all()
            return
        
        if command == "go_home" or command == "home":
            self.get_logger().info("🏠 Action: Go Home")
            self.robot_nudge_pub.publish(String(data="START_TASK"))
            return

        if "_" in command:
            parts = command.split("_")
            action = parts[0]
            target = parts[1] if len(parts) > 1 else ""
        else:
            action = command
            target = ""

        self.get_logger().info(f"🧩 Parsed -> Action: {action}, Target: {target}")

        # --- [Logic Tree] ---

        # 1. J6 회전 (Jog)
        if action in ["turn", "rotate", "spin", "jog", "돌려", "회전"]:
            jog_cmd = "TURN_FRONT"
            if target in ["left", "counter", "반시계", "왼쪽", "back"]:
                jog_cmd = "TURN_BACK"
            elif target in ["right", "clock", "시계", "오른쪽", "front"]:
                jog_cmd = "TURN_FRONT"
            
            self.get_logger().info(f"🔄 J6 Rotating: {jog_cmd}")
            self.robot_jog_pub.publish(String(data=jog_cmd))

        # 2. 미세 조정 (Nudge)
        elif action == "move" or action == "nudge":
            self.robot_nudge_pub.publish(String(data=target.upper()))

        # 3. 도구 가져오기 (Fetch)
        elif action == "fetch" or action == "bring" or action == "get":
            threading.Thread(target=self.sequence_fetch, args=(target,)).start()

        # 4. 잡아주기 (Hold)
        elif action == "hold":
            if "pcb" in target:
                self.target_type = "PCB"
                threading.Thread(target=self.sequence_hold_pcb).start()
            elif "here" in target or "여기" in command: 
                self.target_type = "TOOL"
                threading.Thread(target=self.sequence_hold_point).start()
            else: 
                pass

        # 5. 파지 트리거 (Catch)
        elif (action == "catch" or action == "grab") and self.state == "WAITING_FOR_CATCH":
            self.get_logger().info("🔒 Catch Trigger 확인!")
            
            if self.target_type == "PCB":
                self.robot_gripper_pub.publish(String(data="close"))
                self.tts_pub.publish(String(data="기판을 잡았습니다."))
            else:
                self.robot_gripper_pub.publish(String(data="close"))
                self.tts_pub.publish(String(data="잡았습니다."))
            
            self.state = "IDLE"

        # 6. 확대하기 (Magnify)
        elif action == "magnify" or action == "zoom" or "확대" in command or "돋보기" in command:
             threading.Thread(target=self.sequence_magnify).start()

        # 7. 놓기 (Release/Drop)
        elif action in ["release", "drop", "open", "put", "let", "놔", "놓아", "풀어"]:
            self.get_logger().info("🔓 Action: Release (Open Gripper)")
            self.robot_gripper_pub.publish(String(data="open"))
            self.tts_pub.publish(String(data="물건을 놓습니다."))
            self.state = "IDLE"

    # =========================================================
    # 4. 시퀀스 로직 (Wait 함수 - 부분 일치 확인)
    # =========================================================
    def wait_for_robot(self, target_status, timeout=10.0):
        self.latest_robot_status = ""
        self.get_logger().info(f"⏳ Waiting for Robot status containing: '{target_status}' (Timeout: {timeout}s)")
        
        start = time.time()
        while time.time() - start < timeout:
            if target_status in self.latest_robot_status:
                self.get_logger().info(f"✅ Robot Reached: '{self.latest_robot_status}'")
                return True
            time.sleep(0.1)
        
        self.get_logger().error(f"❌ Robot Timeout! Waited for '{target_status}'")
        return False

    def wait_for_vision(self, timeout=10.0):
        self.detected_pose = None
        self.get_logger().info(f"⏳ Waiting for Vision Target... (Timeout: {timeout}s)")
        
        start = time.time()
        while time.time() - start < timeout:
            if self.detected_pose is not None:
                return self.detected_pose
            time.sleep(0.1)
        return None

    # --- [A. 도구 가져오기] ---
    def sequence_fetch(self, tool_name):
        self.state = "FETCHING"
        self.get_logger().info(f"--- [START] Sequence Fetch: {tool_name} ---")
        self.tts_pub.publish(String(data=f"{tool_name}를 찾고 있습니다."))
        
        # 1. Vision 탐색
        self.vision_cmd_pub.publish(String(data=f"find_{tool_name}"))
        target_pose = self.wait_for_vision()
        if not target_pose: 
            self.tts_pub.publish(String(data="물건을 찾을 수 없습니다."))
            self.state = "IDLE"
            return
        
        # Z축 오프셋 (실제 환경에 맞춰 수정)
        if tool_name == "flux": target_pose.pose.position.z -= 50
        elif tool_name == "pump": target_pose.pose.position.z -= 20
        elif tool_name == "magnifier": target_pose.pose.position.z -= 30
        elif tool_name == "pcb": target_pose.pose.position.z -= 25

        # 2. 로봇 이동 (Vision 좌표)
        self.get_logger().info("🚀 Moving Robot to Target...")
        self.robot_pose_pub.publish(target_pose)
        if not self.wait_for_robot("arrived"): return 

        # 3. 잡기
        self.get_logger().info("✊ Gripping...")
        self.robot_gripper_pub.publish(String(data="close"))
        if not self.wait_for_robot("gripped"): return

        time.sleep(0.5)

        # 4. Handover 이동 (명령어 사용)
        self.get_logger().info("🚚 Moving to Handover Position...")
        self.robot_nudge_pub.publish(String(data="HANDOVER")) 
        if not self.wait_for_robot("arrived"): return

        # 5. 전달
        self.robot_gripper_pub.publish(String(data="open"))
        self.tts_pub.publish(String(data="여기 있습니다."))
        self.get_logger().info("--- [END] Sequence Fetch Complete ---")
        

    # --- [B. PCB 잡아주기] ---
    def sequence_hold_pcb(self):
        self.state = "PREPARING_HOLD"
        self.get_logger().info("--- [START] Sequence Hold PCB ---")
        self.tts_pub.publish(String(data="파지 준비 중입니다."))

        # 1. 대기 위치 이동 (명령어 사용)
        self.robot_nudge_pub.publish(String(data="HANDOVER"))
        if not self.wait_for_robot("arrived"): return

        self.robot_gripper_pub.publish(String(data="open"))
        self.wait_for_robot("opened")

        self.tts_pub.publish(String(data="기판을 넣고 잡아, 라고 말해주세요."))
        self.state = "WAITING_FOR_CATCH"
        self.get_logger().info("💤 State changed to WAITING_FOR_CATCH")

    # --- [C. 돋보기 확대 (Magnify)] - [NEW!] ---
    def sequence_magnify(self):
        self.state = "MAGNIFYING"
        self.get_logger().info("--- [START] Sequence Magnify ---")
        
        # 1. 돋보기 가져오기 (Fetch 로직 일부 차용하지만 놓지 않음)
        self.tts_pub.publish(String(data="돋보기를 가져옵니다."))
        
        # [탐색]
        self.vision_cmd_pub.publish(String(data="find_magnifier"))
        target_pose = self.wait_for_vision()
        if not target_pose: 
            self.tts_pub.publish(String(data="돋보기를 못 찾았습니다."))
            self.state = "IDLE"
            return
        
        # [접근] (Offset 적용)
        target_pose.pose.position.z -= 30 
        self.robot_pose_pub.publish(target_pose)
        if not self.wait_for_robot("arrived"): return

        # [파지]
        self.robot_gripper_pub.publish(String(data="close"))
        if not self.wait_for_robot("gripped"): return

        # 2. 관측 위치(SCAN)로 이동
        self.get_logger().info("🔭 Moving to Scan Position...")
        self.robot_nudge_pub.publish(String(data="SCAN"))
        if not self.wait_for_robot("arrived"): return

        # 3. 손 추적
        self.tts_pub.publish(String(data="확대할 곳을 손으로 가리켜주세요."))
        self.vision_cmd_pub.publish(String(data="track_hand"))
        
        # 손 찾기 대기 (시간 넉넉히)
        hand_pose = self.wait_for_vision(timeout=15.0) 
        if not hand_pose:
            self.tts_pub.publish(String(data="손을 못 찾았습니다."))
            self.state = "IDLE"
            return
        
        # 4. 손 위치로 접근 (확대)
        # hand_pose.pose.position.z += 100
        
        self.get_logger().info("🔍 Zooming in (Approaching Hand)...")
        self.robot_pose_pub.publish(hand_pose)
        if not self.wait_for_robot("arrived"): return

        self.tts_pub.publish(String(data="확대했습니다."))
        self.get_logger().info("--- [END] Sequence Magnify Complete ---")
        self.state = "IDLE"


    # --- [D. 가리킨 곳 잡아주기] ---
    def sequence_hold_point(self):
        self.state = "SCANNING"
        self.get_logger().info("--- [START] Sequence Hold Point (Here) ---")
        self.tts_pub.publish(String(data="손을 보여주세요."))

        # 1. 관측 위치 이동 (명령어 사용)
        self.robot_nudge_pub.publish(String(data="SCAN"))
        if not self.wait_for_robot("arrived"): return

        self.vision_cmd_pub.publish(String(data="track_hand"))
        hand_pose = self.wait_for_vision(timeout=60.0)

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