import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from enum import Enum

class State(Enum):
    IDLE = 0        # 대기 상태
    LISTEN = 1      # 음성 명령 수신
    SENSING = 2     # 비전 탐색 및 모드 변경
    PLANNING = 3    # 이동 및 정밀 조정 계획
    ACTION = 4      # 로봇 동작 수행 (이동/그리퍼)
    FEEDBACK = 5    # 결과 음성 출력

class MainController(Node):
    def __init__(self):
        super().__init__('main_controller')

        # 상태 초기화
        self.state = State.IDLE
        self.get_logger().info('=== FiXiT Main Controller (Interface Spec v4.0) Initialized ===')

        # --- [Publishers] ---
        # 4.1 Voice Node: 음성 출력 요청
        self.voice_tts_pub = self.create_publisher(String, '/voice/tts', 10)
        
        # 4.2 Vision Node: 비전 모드 변경 명령 (find_tool, track_hand 등)
        self.vision_cmd_pub = self.create_publisher(String, '/vision/cmd', 10)
        
        # 4.3 Robot Node: 이동 및 제어 명령
        self.robot_target_pub = self.create_publisher(PoseStamped, '/robot/target_pose', 10) # 장거리 이동
        self.robot_jog_pub = self.create_publisher(String, '/robot/jog', 10)                 # 시선 맞추기
        self.robot_nudge_pub = self.create_publisher(String, '/robot/nudge_cmd', 10)        # 미세 조정
        self.robot_gripper_pub = self.create_publisher(String, '/robot/gripper', 10)        # 그리퍼 제어

        # --- [Subscribers] ---
        # 4.1 Voice Node: GPT가 판단한 최종 실행 의도 수신
        self.voice_cmd_sub = self.create_subscription(String, '/voice/cmd', self.voice_cmd_callback, 10)
        
        # 4.2 Vision Node: 비전 데이터 수신
        self.vision_dir_sub = self.create_subscription(String, '/vision/direction', self.vision_dir_callback, 10)
        self.vision_pose_sub = self.create_subscription(PoseStamped, '/vision/target_pose', self.vision_pose_callback, 10)
        
        # 4.3 Robot Node: 로봇 상태 수신 (arrived, gripped, stopped)
        self.robot_status_sub = self.create_subscription(String, '/robot/status', self.robot_status_callback, 10)

        # 데이터 저장 변수
        self.current_target_pose = None
        self.current_intent = ""

    # --- [Callback Functions] ---

    def voice_cmd_callback(self, msg):
        """[4.1 Voice] 음성 명령(의도) 수신 시 실행"""
        if self.state == State.IDLE:
            self.current_intent = msg.data
            self.get_logger().info(f'[VOICE] New Command: {self.current_intent}')
            self.state = State.LISTEN
            
            # 명령에 따른 비전 모드 설정 (예시 logic)
            self.request_vision_mode()

    def request_vision_mode(self):
        """[4.2 Vision] 명령 의도에 맞춰 비전 모드 변경 명령 발행"""
        cmd_msg = String()
        if "iron" in self.current_intent or "pcb" in self.current_intent:
            cmd_msg.data = "find_tool"
        else:
            cmd_msg.data = "track_hand"
            
        self.vision_cmd_pub.publish(cmd_msg)
        self.get_logger().info(f'[VISION] Mode Request Sent: {cmd_msg.data}')
        self.state = State.SENSING

    def vision_dir_callback(self, msg):
        """[4.2 Vision] 손의 대략적인 방향 수신 시 로봇 조그 제어"""
        if self.state == State.SENSING:
            jog_msg = String()
            if msg.data == "hand":
                # 손이 감지되면 시선을 맞추기 위해 조그 이동 명령 (예시)
                jog_msg.data = "TURN_LEFT" 
                self.robot_jog_pub.publish(jog_msg)
                self.get_logger().info(f'[ROBOT] Jogging to hand direction...')

    def vision_pose_callback(self, msg):
        """[4.2 Vision] 정밀 3D 좌표 수신 시 이동 계획 수립"""
        if self.state == State.SENSING:
            self.current_target_pose = msg
            self.get_logger().info('[PLANNING] Target Coordinate Received. Moving to Action...')
            self.state = State.PLANNING
            self.execute_robot_move()

    def execute_robot_move(self):
        """[4.3 Robot] 로봇에 목표 좌표 발행 (MoveIt)"""
        if self.current_target_pose:
            self.robot_target_pub.publish(self.current_target_pose)
            self.state = State.ACTION
            self.get_logger().info('[ROBOT] Sending target pose to MoveIt...')

    def robot_status_callback(self, msg):
        """[4.3 Robot] 로봇의 동작 상태 결과 처리"""
        if self.state == State.ACTION:
            if msg.data == "arrived":
                self.get_logger().info('[ROBOT] Arrived! Sending Gripper Close command.')
                # 도착 후 그리퍼 닫기 (물체 파지)
                gripper_msg = String()
                gripper_msg.data = "close"
                self.robot_gripper_pub.publish(gripper_msg)
            
            elif msg.data == "gripped":
                self.get_logger().info('[ROBOT] Gripped! Process Complete.')
                self.send_tts_feedback("작업을 완료했습니다.")
                
            elif msg.data == "stopped":
                self.get_logger().warn('[ROBOT] Movement Stopped unexpectedly.')
                self.state = State.IDLE

    def send_tts_feedback(self, text):
        """[4.1 Voice] 최종 결과를 음성으로 출력"""
        self.state = State.FEEDBACK
        tts_msg = String()
        tts_msg.data = text
        self.voice_tts_pub.publish(tts_msg)
        self.get_logger().info(f'[TTS] Feedback sent: {text}')
        
        # 완료 후 IDLE 복귀
        self.state = State.IDLE
        self.get_logger().info('=== System Ready (IDLE) ===')

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