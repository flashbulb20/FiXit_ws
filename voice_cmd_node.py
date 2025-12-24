import os
import io
import time
import sys
import pygame
from dotenv import load_dotenv

# ROS 2 관련 임포트
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# 사용자 제공 모듈 임포트
from MicController import MicController, MicConfig
from wakeup_word import WakeupWord
from STT import STT

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class VisionPointingVoiceNode(Node):
    def __init__(self):
        super().__init__('vision_pointing_voice_node')
        
        # 1. Pub: 메인 컨트롤러에 명령 전달 / Sub: 로봇 음성 출력 요청 수신
        self.publisher_ = self.create_publisher(String, '/voice/cmd', 10)
        self.subscription = self.create_subscription(String, '/voice/tts', self.tts_callback, 10)
        
        # 2. 하드웨어 설정 (호출어 인식률 최적화를 위해 48,000Hz 고정)
        config = MicConfig()
        config.rate = 48000  
        config.buffer_size = 3840 
        
        self.mic = MicController(config)
        self.wakeup = WakeupWord(self.mic.config.buffer_size)
        self.stt = STT(OPENAI_API_KEY)
        
        pygame.mixer.init()
        
        # 3. [완벽 분리] 시나리오 기반 명령어 맵
        self.command_logic = [
            # 사용자가 "이거"라고 하면 비전이 좌표를 인식하도록 유도함
            {"trigger": ["이거", "이것", "포인팅", "손가락", "좌표", "가리킨"], 
             "payload": "track_hand", 
             "msg": "포인팅하신 물체의 좌표를 비전으로 인식하여 가져오겠습니다."},
            
            # 돋보기 전달 (단순 물건 가져오기)
            {"trigger": ["돋보기", "magnifier"], 
             "payload": "fetch_magnifier", 
             "msg": "돋보기를 가져오겠습니다."},

            # 시나리오 1: 납 제거 -> 납흡입기
            {"trigger": ["납 흡입기", "떼", "흡입기"], "payload": "fetch_solder_sucker", "msg": "납 제거를 위해 납 흡입기를 가져올게요."},
            
            # 시나리오 2: 융해 촉진 -> 플럭스
            {"trigger": ["녹지", "안 녹네", "융해", "촉진", "플럭스"], "payload": "fetch_flux", "msg": "융해 촉진을 위해 플럭스를 준비하겠습니다."},
            
            # 시나리오 3: 기판 고정 -> hold_pcb
            {"trigger": ["잡아", "고정", "hold"], "payload": "hold_pcb", "msg": "기판을 고정하겠습니다."},
            
            # 시나리오 4: 세부 조작 -> move_up
            {"trigger": ["위로", "올려", "up"], "payload": "move_up", "msg": "조금 위로 이동할게요."},

            # 시나리오 5: 세부 조작 -> move_down
            {"trigger": ["밑으로", "내려", "down"], "payload":"move_down", "msg": "조금 밑으로 이동할게요"},
            
            # 기존 도구: PCB 가져오기
            {"trigger": ["피씨비", "pcb", "기판", "보드"], "payload": "fetch_PCB", "msg": "PCB를 가져오겠습니다."}
        ]
        
        self.hint = "납, 떼야겠어, 안 녹네, 플럭스, 잡아줘, 고정, 위로, 올려, 내려, 밑으로, 이거 가져와, 돋보기, 피씨비."
        self.get_logger().info("노드가 시작되었습니다.")
        self.display_and_speak("준비가 완료되었습니다. 무엇을 도와드릴까요?")

    def tts_callback(self, msg):
        """메인 노드에서 작업 완료 등의 상태를 음성으로 보고할 때 사용"""
        self.display_and_speak(msg.data)

    def display_and_speak(self, text):
        """로봇 음성 출력 및 로그 기록"""
        self.get_logger().info(f'로봇 응답: {text}')
        try:
            response = self.stt.client.audio.speech.create(
                model="tts-1", voice="nova", input=text
            )
            byte_stream = io.BytesIO(response.content)
            pygame.mixer.music.load(byte_stream)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                # 음성 재생 중에도 ROS 통신이 유지되도록 처리
                rclpy.spin_once(self, timeout_sec=0)
                time.sleep(0.1)
        except Exception as e:
            self.get_logger().error(f'TTS 에러: {e}')

    def publish_cmd(self, payload):
        """정해진 규격의 페이로드를 /voice/cmd로 발행"""
        msg = String()
        msg.data = payload
        self.publisher_.publish(msg)
        self.get_logger().info(f'[PUB] /voice/cmd >> {msg.data}')

    def handle_intelligence(self, text):
        """지능형 시나리오 분석 및 명령 발행 결정"""
        t = text.lower()
        
        if any(w in t for w in ["종료", "exit"]):
            self.display_and_speak("프로그램을 종료합니다.")
            return "EXIT"

        # 시나리오 매칭 로직
        for entry in self.command_logic:
            if any(trigger in t for trigger in entry["trigger"]):
                self.display_and_speak(entry["msg"])
                self.publish_cmd(entry["payload"])
                return "SUCCESS"

        return "RETRY"

    def run_loop(self):
        """메인 음성 인식 루프"""
        try:
            while rclpy.ok():
                # 1단계: 호출어 대기
                self.mic.open_stream()
                self.wakeup.set_stream(self.mic.stream)
                self.get_logger().info("'헬로 rokey' 대기 중...")
                
                while rclpy.ok():
                    rclpy.spin_once(self, timeout_sec=0.1)
                    if self.wakeup.is_wakeup(): 
                        self.display_and_speak("네, 말씀하세요.")
                        break
                
                self.mic.close_stream()
                if not rclpy.ok(): break

                # 2단계: 명령어 연속 인식 모드 (실패 시 재시도)
                while rclpy.ok():
                    self.get_logger().info("시나리오 명령 청취 중...")
                    audio = self.mic.record_audio()
                    
                    if audio and len(audio) > 500:
                        try:
                            # 400 에러 방지를 위한 파일 이름 부여
                            f = io.BytesIO(audio)
                            f.name = "pointing_mic.wav" 
                            res = self.stt.client.audio.transcriptions.create(
                                model="whisper-1", file=f, prompt=self.hint
                            )
                            
                            if res.text:
                                self.get_logger().info(f"사용자: {res.text}")
                                result = self.handle_intelligence(res.text)
                                if result == "EXIT": return
                                elif result == "SUCCESS": break 
                                else:
                                    self.display_and_speak("다시 말씀해 주세요.")
                        except Exception as e:
                            self.get_logger().error(f"STT 에러: {e}")
                            break
                    else:
                        self.display_and_speak("잘 듣지 못했습니다. 다시 말씀해 주세요.")
                    
                    rclpy.spin_once(self, timeout_sec=0)

        finally:
            self.mic.close_stream()

def main(args=None):
    rclpy.init(args=args)
    node = VisionPointingVoiceNode()
    try:
        node.run_loop()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()