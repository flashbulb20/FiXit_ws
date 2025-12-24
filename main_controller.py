import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from enum import Enum

class State(Enum):
    IDLE = 0
    LISTEN = 1
    SENSING = 2
    PLANNING = 3
    ACTION = 4
    FEEDBACK = 5

class MainController(Node):
    def __init__(self):
        super().__init__('main_controller')

        # 상태 초기화
        self.state = State.IDLE
        self.get_logger().info('=== FiXiT Main Controller Initialized (State: IDLE) ===')

        # --- Publishers ---
        # 4.1 Voice Node로 TTS 출력 요청
        self.voice_tts_pub = self.create_publisher(String, '/voice/tts', 10)
        # 4.2 Vision Node로 비전 모드 전환 명령
        self.vision_cmd_pub = self.create_publisher(String, '/vision/cmd', 10)
        # 4.3 Robot Node로 이동 목표 및 그리퍼 제어 명령
        self.robot_pose_pub = self.create_publisher(PoseStamped, '/robot/target_pose', 10)
        self.robot_gripper_pub = self.create_publisher(String, '/robot/gripper', 10)

        # --- Subscribers ---
        # 4.1 Voice Node로부터 사용자 명령 수신
        self.voice_cmd_sub = self.create_subscription(String, '/voice/cmd', self.voice_cmd_callback, 10)
        # 4.2 Vision Node로부터 타겟 좌표 및 상태 수신
        self.vision_pose_sub = self.create_subscription(PoseStamped, '/vision/target_pose', self.vision_pose_callback, 10)
        self.vision_status_sub = self.create_subscription(String, '/vision/status', self.vision_status_callback, 10)
        # 4.3 Robot Node로부터 동작 상태 수신
        self.robot_status_sub = self.create_subscription(String, '/robot/status', self.robot_status_callback, 10)

        # 임시 데이터 저장 변수
        self.current_target_pose = None
        self.current_command = ""

    def voice_cmd_callback(self, msg):
        """LISTEN 단계: 음성 명령 수신 및 분석"""
        if self.state == State.IDLE:
            self.current_command = msg.data
            self.get_logger().info(f'Voice Command Received: "{self.current_command}"')
            self.state = State.LISTEN
            
            # SENSING 단계 진입: 비전 노드에 물체 탐색 명령 발행
            self.start_sensing()

    def start_sensing(self):
        """SENSING 단계: 비전 모드 전환"""
        self.get_logger().info('State: SENSING - Requesting Vision Detection...')
        vision_msg = String()
        
        # 명령 분석 로직 (예시)
        if "iron" in self.current_command:
            vision_msg.data = "find_iron"
        elif "hand" in self.current_command:
            vision_msg.data = "track_hand"
        else:
            vision_msg.data = "idle"
            
        self.vision_cmd_pub.publish(vision_msg)
        self.state = State.SENSING

    def vision_status_callback(self, msg):
        """비전 인식 상태 확인"""
        if self.state == State.SENSING:
            if msg.data == "fail":
                self.get_logger().error("Vision Detection Failed!")
                self.send_feedback("인식에 실패했습니다. 다시 시도해주세요.")
                self.state = State.IDLE

    def vision_pose_callback(self, msg):
        """PLANNING 단계: 좌표 수신 및 검증"""
        if self.state == State.SENSING:
            self.get_logger().info('Target Pose Received. State: PLANNING')
            self.current_target_pose = msg
            self.state = State.PLANNING
            
            # 좌표 유효성 검증 (안전 영역 체크 로직이 들어갈 자리)
            if self.validate_pose(msg):
                self.execute_action()
            else:
                self.send_feedback("안전 범위를 벗어난 좌표입니다.")
                self.state = State.IDLE

    def validate_pose(self, pose):
        """좌표 유효성 검증 로직 (여기서는 항상 True 반환)"""
        return True

    def execute_action(self):
        """ACTION 단계: 로봇 이동 명령 발행"""
        self.get_logger().info('State: ACTION - Moving Robot to Target...')
        self.robot_pose_pub.publish(self.current_target_pose)
        self.state = State.ACTION

    def robot_status_callback(self, msg):
        """로봇 동작 완료 및 그리퍼 제어"""
        if self.state == State.ACTION:
            if msg.data == "arrived":
                self.get_logger().info('Robot Arrived. Controlling Gripper...')
                gripper_msg = String()
                gripper_msg.data = "close" # 명령에 따라 open/close 결정
                self.robot_gripper_pub.publish(gripper_msg)
            
            elif msg.data == "gripped" or msg.data == "done":
                self.get_logger().info('Action Complete. State: FEEDBACK')
                self.state = State.FEEDBACK
                self.send_feedback("작업을 완료했습니다.")

    def send_feedback(self, text):
        """FEEDBACK 단계: TTS 출력 및 IDLE 복귀"""
        self.get_logger().info(f'Sending Feedback: {text}')
        tts_msg = String()
        tts_msg.data = text
        self.voice_tts_pub.publish(tts_msg)
        
        # 모든 공정 완료 후 대기 상태로 복귀
        self.state = State.IDLE
        self.get_logger().info('=== State: IDLE (Waiting for next command) ===')

def main(args=None):
    rclpy.init(args=args)
    node = MainController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()