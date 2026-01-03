import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
import time
import threading
import math

# [추가됨] 로봇 모드 변경 서비스
try:
    from dsr_msgs2.srv import SetRobotMode
except ImportError:
    from dsr_msgs.srv import SetRobotMode

class MainController(Node):
    def __init__(self):
        super().__init__('main_controller')
        self.get_logger().info("🚀 Fixit Main Controller Started (Env Updated + Magnify Pre-Pose)")

        # 1. 통신 인터페이스
        self.create_subscription(String, '/voice/cmd', self.voice_callback, 10)
        self.create_subscription(String, '/robot/status', self.robot_status_callback, 10)
        self.create_subscription(PoseStamped, '/vision/target_pose', self.vision_pose_callback, 10)

        self.robot_pose_pub = self.create_publisher(PoseStamped, '/robot/target_pose', 10)
        self.robot_nudge_pub = self.create_publisher(String, '/robot/nudge_cmd', 10) 
        self.robot_jog_pub = self.create_publisher(String, '/robot/jog', 10)
        self.robot_gripper_pub = self.create_publisher(String, '/robot/gripper', 10)
        self.vision_cmd_pub = self.create_publisher(String, '/vision/cmd', 10)
        
        # [TTS 주석 처리]
        # self.tts_pub = self.create_publisher(String, '/voice/tts', 10)

        self.mode_client = self.create_client(SetRobotMode, '/dsr01/system/set_robot_mode')

        # 2. 상태 초기화
        self.latest_robot_status = ""
        self.detected_pose = None
        self.state = "IDLE"
        self.target_type = "TOOL"

    # 3. 콜백 함수
    def robot_status_callback(self, msg):
        self.latest_robot_status = msg.data
        if "fail" in msg.data and "collision" in msg.data:
            if self.state != "COLLISION":
                self.state = "COLLISION"
                self.get_logger().error(f"💥 충돌 감지! - '복구'라고 말해주세요.")
                # [TTS 주석 처리]
                # self.tts_pub.publish(String(data="충돌이 감지되었습니다. 복구 명령을 내려주세요."))

    def vision_pose_callback(self, msg):
        self.detected_pose = msg
        self.get_logger().debug(f"👁️ Vision Detected")

    def voice_callback(self, msg):
        command = msg.data.lower().strip()
        self.get_logger().info(f"📢 CMD: {command}")

        if "stop" in command or "멈춰" in command:
            self.stop_all(); return
        
        if command == "recovery" or "복구" in command:
            self.get_logger().info("🚑 Recovery Mode")
            self.robot_nudge_pub.publish(String(data="recovery"))
            # [TTS 주석 처리]
            # self.tts_pub.publish(String(data="복구 모드를 시작합니다."))
            self.state = "IDLE"; return

        if "manual" in command or "수동" in command:
            threading.Thread(target=self.change_robot_mode, args=(0,)).start(); return
        elif "auto" in command or "자동" in command:
            threading.Thread(target=self.change_robot_mode, args=(1,)).start(); return

        if command in ["go_home", "home"]:
            self.robot_nudge_pub.publish(String(data="START_TASK")); return

        if "_" in command: parts = command.split("_"); action = parts[0]; target = parts[1] if len(parts) > 1 else ""
        else: action = command; target = ""

        # Logic Tree
        if action in ["turn", "rotate", "spin", "jog", "돌려", "회전"]:
            jog_cmd = "TURN_FRONT"
            if target in ["left", "counter", "반시계", "왼쪽", "back"]: jog_cmd = "TURN_BACK"
            self.robot_jog_pub.publish(String(data=jog_cmd))

        elif action in ["move", "nudge"]:
            self.robot_nudge_pub.publish(String(data=target.upper()))

        elif action in ["fetch", "bring", "get"]:
            threading.Thread(target=self.sequence_fetch, args=(target,)).start()

        elif action == "hold":
            if "pcb" in target: self.target_type = "PCB"; threading.Thread(target=self.sequence_hold_pcb).start()
            elif "here" in target or "여기" in command: self.target_type = "TOOL"; threading.Thread(target=self.sequence_hold_point).start()

        elif action in ["catch", "grab"]:
            if self.target_type == "PCB":
                self.robot_gripper_pub.publish(String(data="close"))
                # [TTS 주석 처리]
                # self.tts_pub.publish(String(data="기판을 잡았습니다."))
            else:
                self.robot_gripper_pub.publish(String(data="close"))
                # [TTS 주석 처리]
                # self.tts_pub.publish(String(data="잡았습니다."))
            self.state = "IDLE"

        elif action in ["magnify", "zoom", "확대", "돋보기"]:
             threading.Thread(target=self.sequence_magnify).start()

        elif action in ["release", "drop", "open", "put", "let", "놔", "놓아"]:
            self.robot_gripper_pub.publish(String(data="open"))
            # [TTS 주석 처리]
            # self.tts_pub.publish(String(data="놓습니다."))
            self.state = "IDLE"

    # 4. 서비스 호출
    def change_robot_mode(self, mode_int):
        mode_str = "수동" if mode_int == 0 else "자동"
        # [TTS 주석 처리]
        # self.tts_pub.publish(String(data=f"{mode_str} 모드로 변경합니다."))
        
        if not self.mode_client.wait_for_service(timeout_sec=2.0): return
        req = SetRobotMode.Request(); req.robot_mode = mode_int
        future = self.mode_client.call_async(req)
        start = time.time()
        while not future.done():
            time.sleep(0.1)
            if time.time() - start > 3.0: return
        
        # [TTS 주석 처리]
        # if future.result().success: self.tts_pub.publish(String(data="모드가 변경되었습니다."))

    # 5. 시퀀스 로직
    def wait_for_robot(self, target_status, timeout=15.0):
        self.latest_robot_status = ""
        start = time.time()
        while time.time() - start < timeout:
            if target_status in self.latest_robot_status: return True
            if "fail" in self.latest_robot_status: return False
            time.sleep(0.1)
        return False

    def wait_for_vision(self, timeout=10.0):
        self.detected_pose = None
        start = time.time()
        while time.time() - start < timeout:
            if self.detected_pose is not None: return self.detected_pose
            time.sleep(0.1)
        return None

    # --- [A. 도구 가져오기] ---
    def sequence_fetch(self, tool_name):
        self.state = "FETCHING"
        self.get_logger().info(f"--- Fetch Sequence: {tool_name} ---")
        # [TTS 주석 처리]
        # self.tts_pub.publish(String(data=f"{tool_name}를 찾겠습니다."))
        
        # [NEW] 0. 탐색 위치로 먼저 이동
        self.get_logger().info("🔭 Moving to Search Position...")
        self.robot_nudge_pub.publish(String(data="SEARCH_TOOL"))
        if not self.wait_for_robot("arrived"): return

        # 1. Vision 탐색
        time.sleep(0.5) 
        self.vision_cmd_pub.publish(String(data=f"find_{tool_name}"))
        target_pose = self.wait_for_vision()
        if not target_pose: 
            # [TTS 주석 처리]
            # self.tts_pub.publish(String(data="물건을 찾을 수 없습니다."))
            self.state = "IDLE"; return
        
        # Z축 오프셋
        if tool_name == "flux": target_pose.pose.position.z -= 50
        elif tool_name == "pump": target_pose.pose.position.z -= 20
        elif tool_name == "magnifier": target_pose.pose.position.z -= 30
        elif tool_name == "pcb": target_pose.pose.position.z -= 25

        # 2. 로봇 이동
        self.robot_pose_pub.publish(target_pose)
        if not self.wait_for_robot("arrived"): return

        # 3. 잡기
        self.robot_gripper_pub.publish(String(data="close"))
        if not self.wait_for_robot("gripped"): return

        # 4. Handover 이동
        self.get_logger().info("🚚 Delivering...")
        self.robot_nudge_pub.publish(String(data="HANDOVER")) 
        if not self.wait_for_robot("arrived"): return

        # 5. 전달
        self.robot_gripper_pub.publish(String(data="open"))
        # [TTS 주석 처리]
        # self.tts_pub.publish(String(data="여기 있습니다."))

    # --- [B. PCB] ---
    def sequence_hold_pcb(self):
        self.state = "PREPARING_HOLD"
        # [TTS 주석 처리]
        # self.tts_pub.publish(String(data="파지 준비 중입니다."))
        
        self.robot_nudge_pub.publish(String(data="HANDOVER"))
        if not self.wait_for_robot("arrived"): return
        self.robot_gripper_pub.publish(String(data="open"))
        self.wait_for_robot("opened")
        
        # [TTS 주석 처리]
        # self.tts_pub.publish(String(data="기판을 넣고 잡아, 라고 말해주세요."))
        self.state = "WAITING_FOR_CATCH"

    # --- [C. 돋보기] ---
    def sequence_magnify(self):
        self.state = "MAGNIFYING"
        # [TTS 주석 처리]
        # self.tts_pub.publish(String(data="돋보기를 가져옵니다."))
        
        # [NEW] 0. 탐색 위치 이동
        self.robot_nudge_pub.publish(String(data="SEARCH_TOOL"))
        if not self.wait_for_robot("arrived"): return
        
        self.vision_cmd_pub.publish(String(data="find_magnifier"))
        target_pose = self.wait_for_vision()
        if not target_pose: 
            # [TTS 주석 처리]
            # self.tts_pub.publish(String(data="돋보기를 못 찾았습니다."))
            self.state = "IDLE"; return
        
        target_pose.pose.position.z -= 30 
        self.robot_pose_pub.publish(target_pose)
        if not self.wait_for_robot("arrived"): return

        self.robot_gripper_pub.publish(String(data="close"))
        if not self.wait_for_robot("gripped"): return

        # 2. 관측 위치(SCAN)로 이동
        self.robot_nudge_pub.publish(String(data="SCAN"))
        if not self.wait_for_robot("arrived"): return


        # [NEW] 4. 확대 준비 자세 (MAGNIFY_PRE)
        self.get_logger().info("📐 Aligning for Magnification...")
        self.robot_nudge_pub.publish(String(data="MAGNIFY_PRE"))
        if not self.wait_for_robot("arrived"): return

        # 3. 손 추적
        # [TTS 주석 처리]
        # self.tts_pub.publish(String(data="확대할 곳을 손으로 가리켜주세요."))
        self.vision_cmd_pub.publish(String(data="track_hand"))
        hand_pose = self.wait_for_vision(timeout=15.0) 
        if not hand_pose:
            # [TTS 주석 처리]
            # self.tts_pub.publish(String(data="손을 못 찾았습니다."))
            self.state = "IDLE"; return
        
        # 5. 손 위치로 접근
        self.get_logger().info("🔍 Zooming in...")
        self.robot_pose_pub.publish(hand_pose)
        if not self.wait_for_robot("arrived"): return
        
        # [TTS 주석 처리]
        # self.tts_pub.publish(String(data="확대했습니다."))
        self.state = "IDLE"

    # --- [D. Here] ---
    def sequence_hold_point(self):
        self.state = "SCANNING"
        # [TTS 주석 처리]
        # self.tts_pub.publish(String(data="손을 보여주세요."))
        
        self.robot_nudge_pub.publish(String(data="SCAN"))
        if not self.wait_for_robot("arrived"): return

        self.vision_cmd_pub.publish(String(data="track_hand"))
        hand_pose = self.wait_for_vision(timeout=60.0)
        if not hand_pose:
            # [TTS 주석 처리]
            # self.tts_pub.publish(String(data="손을 놓쳤습니다."))
            self.state = "IDLE"; return

        self.robot_gripper_pub.publish(String(data="open"))
        self.robot_pose_pub.publish(hand_pose)
        if not self.wait_for_robot("arrived"): return

        # [TTS 주석 처리]
        # self.tts_pub.publish(String(data="잡을까요?"))
        self.state = "WAITING_FOR_CATCH"

    def stop_all(self):
        self.get_logger().warn("🚨 STOP")
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