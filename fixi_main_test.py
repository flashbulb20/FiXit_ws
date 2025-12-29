import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float64MultiArray
from geometry_msgs.msg import PoseStamped
from enum import Enum
import time

class State(Enum):
    IDLE = 0
    LISTEN = 1
    SENSING = 2
    ACTION = 3
    FEEDBACK = 4

class MainController(Node):
    def __init__(self):
        super().__init__('main_controller')
        self.state = State.IDLE
        self.get_logger().info('=== FiXiT Main Controller (Vertical Align & Delay) ===')

        # --- Publishers ---
        self.voice_tts_pub = self.create_publisher(String, '/voice/tts', 10)
        self.vision_cmd_pub = self.create_publisher(String, '/vision/cmd', 10)
        self.robot_pose_pub = self.create_publisher(PoseStamped, '/robot/target_pose', 10)
        self.robot_joint_pub = self.create_publisher(Float64MultiArray, '/robot/target_joint', 10)
        self.robot_nudge_pub = self.create_publisher(String, '/robot/nudge_cmd', 10)
        self.robot_jog_pub = self.create_publisher(String, '/robot/jog', 10)
        self.robot_gripper_pub = self.create_publisher(String, '/robot/gripper', 10)

        # --- Subscribers ---
        self.voice_cmd_sub = self.create_subscription(String, '/voice/cmd', self.voice_cmd_callback, 10)
        self.vision_pose_sub = self.create_subscription(PoseStamped, '/vision/target_pose', self.vision_pose_callback, 10)
        self.robot_status_sub = self.create_subscription(String, '/robot/status', self.robot_status_callback, 10)

        self.current_payload = ""
        self.process_step = 0 
        
        # 비전 좌표 임시 저장용
        self.target_vision_pose = None
        
        # [설정] 물체를 놓을 위치 (Drop Zone) - Joint Angle
        self.drop_zone_joint = Float64MultiArray()
        self.drop_zone_joint.data = [5.784, 0.719, 86.65, -4.977, 51.269, 7.344]

    def voice_cmd_callback(self, msg):
        if self.state == State.IDLE:
            self.current_payload = msg.data.lower().strip()
            self.get_logger().info(f"📥 Voice: {self.current_payload}")
            self.state = State.LISTEN
            
            if any(k in self.current_payload for k in ["fetch", "find", "get", "가져"]):
                self.start_sensing()
            elif "start" in self.current_payload or "home" in self.current_payload:
                self.state = State.ACTION
                self.robot_nudge_pub.publish(String(data="START_TASK"))
            else:
                self.state = State.IDLE

    def start_sensing(self):
        self.state = State.SENSING
        vision_msg = String()
        if "fetch_" in self.current_payload:
            vision_msg.data = self.current_payload.replace("fetch_", "find_")
        else:
            vision_msg.data = "find_object"
        self.vision_cmd_pub.publish(vision_msg)

    def vision_pose_callback(self, msg):
        """Step 0: 비전 좌표 수신 -> 수직 정렬(홈 이동) 먼저 수행"""
        if self.state == State.SENSING:
            self.get_logger().info("📍 [Step 0] 좌표 수신 완료. 수직 정렬을 위해 홈으로 이동합니다.")
            
            # 좌표 저장해두기
            self.target_vision_pose = msg
            
            self.state = State.ACTION
            self.process_step = 0 # 0단계: 수직 정렬(Home)
            
            # 로봇을 수직 상태(Home)로 리셋
            self.robot_nudge_pub.publish(String(data="START_TASK"))

    def robot_status_callback(self, msg):
        status = msg.data
        
        if self.state == State.ACTION:
            # [Step 0 -> 1] 수직 정렬(홈) 완료 -> 저장된 좌표로 이동
            if self.process_step == 0 and status == "arrived":
                self.get_logger().info("📍 [Step 1] 수직 정렬 완료. 물체 위치로 이동")
                self.process_step = 1
                if self.target_vision_pose:
                    self.robot_pose_pub.publish(self.target_vision_pose)

            # [Step 1 -> 2] 물체 위 도착 -> 하강
            elif self.process_step == 1 and status == "arrived_target":
                self.get_logger().info("⬇️ [Step 2] 픽업 하강 (250mm)")
                self.process_step = 2
                self.robot_nudge_pub.publish(String(data="DOWN_PICK"))

            # [Step 2 -> 3] 하강 완료 -> 잡기
            elif self.process_step == 2 and status == "arrived":
                self.get_logger().info("✊ [Step 3] 물체 잡기")
                self.process_step = 3
                self.robot_gripper_pub.publish(String(data="close"))

            # [Step 3 -> 4] 잡기 완료 -> 1초 대기 -> 상승
            elif self.process_step == 3 and status == "gripper_closed":
                self.get_logger().info("⏳ 잡았습니다. 1초 대기 중...")
                time.sleep(1.0) # [추가] 1초 대기
                
                self.get_logger().info("⬆️ [Step 4] 픽업 상승 (150mm)")
                self.process_step = 4
                self.robot_nudge_pub.publish(String(data="UP_PICK"))

            # [Step 4 -> 5] 상승 완료 -> Drop Zone 이동
            elif self.process_step == 4 and status == "arrived":
                self.get_logger().info("🚚 [Step 5] Drop Zone으로 이동 (Joint Move)")
                self.process_step = 5
                self.robot_joint_pub.publish(self.drop_zone_joint)

            # [Step 5 -> 6] 이동 완료 -> 놓기
            elif self.process_step == 5 and (status == "joint_arrived" or status == "jog_done"):
                self.get_logger().info("🖐️ [Step 6] 물체 놓기")
                self.process_step = 6
                self.robot_gripper_pub.publish(String(data="open"))

            # [Step 6 -> 7] 놓기 완료 -> 1초 대기 -> 홈 복귀
            elif self.process_step == 6 and status == "gripper_opened":
                self.get_logger().info("⏳ 놓았습니다. 1초 대기 중...")
                time.sleep(1.0) # [추가] 1초 대기

                self.get_logger().info("🏠 [Step 7] 작업 완료. 홈 위치로 복귀")
                self.process_step = 7
                self.robot_nudge_pub.publish(String(data="START_TASK"))

            # [Finish] 홈 복귀 완료
            elif self.process_step == 7 and status == "arrived":
                self.get_logger().info("✅ [Finish] 전체 시퀀스 완료")
                self.process_step = 0
                self.send_feedback("작업을 완료하고 복귀했습니다.")

            elif self.process_step == 0 and status in ["arrived", "done", "jog_done", "gripper_opened"]:
                self.send_feedback("명령 수행 완료.")

    def send_feedback(self, text):
        self.state = State.FEEDBACK
        self.voice_tts_pub.publish(String(data=text))
        self.state = State.IDLE

def main(args=None):
    rclpy.init(args=args)
    node = MainController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()