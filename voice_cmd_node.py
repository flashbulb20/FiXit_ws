import os
import io
import time
import json
import pygame
from dotenv import load_dotenv

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from voice_control.MicController import MicController, MicConfig
from voice_control.wakeup_word import WakeupWord
from voice_control.STT import STT

load_dotenv(dotenv_path=os.path.join(".env"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class VisionPointingVoiceNode(Node):
    def __init__(self):
        super().__init__('vision_pointing_voice_node')
        
        # 1. ROS 2 통신 설정
        self.publisher_ = self.create_publisher(String, '/voice/cmd', 10)
        self.subscription = self.create_subscription(String, '/voice/tts', self.tts_callback, 10)
        
        # 2. 하드웨어 설정
        config = MicConfig()
        config.rate = 48000  
        self.mic = MicController(config)
        self.wakeup = WakeupWord(self.mic.config.buffer_size)
        self.stt = STT(OPENAI_API_KEY)
        
        pygame.mixer.init()
        self.system_prompt = """
        당신은 Doosan M0609 로봇 팔과 TurtleBot4의 지능형 통역사입니다. 
        사용자의 한국어 요청을 로봇이 이해할 수 있는 '영어 토픽 명칭'으로 변환하세요.

        [연속 대화 규칙]
        1. 당신은 현재 '연속 대화 모드'에 있습니다. 호출어 없이도 사용자의 명령을 계속 처리합니다.
        2. 사용자가 "대기"라는 단어가 인식되면 payload를 'standby'로 설정하세요.

        [토픽 생성 규칙]
        1. 손으로 가리키는 뉘앙스가 있다면 무조건 'track_hand'를 출력하세요.
        2. 이 모델은 전자기기 수리 보조기능을 담당합니다. 사용자의 모든 언어를 전자기기 수리와 관련된 언어로 인식하세요.
        3. 작업도구에는 pcb, 플럭스(flux), 전선, 돋보기, 납 흡입기, 인두기가 있습니다.
        4. 종료 요청: 사용자가 작업을 끝내려 하면 'program_finish'를 출력하세요.
        5. 기타: 로봇이 수행할 수 없는 일반 대화나 모호한 말은 'NONE'으로 처리하세요.
        6. 명확한 작업도구가 인식되지 않으면 문맥과 일치하는 명사를 제안하세요
        7. 작업 보조:
            - "잡아줘", "고정해" -> hold
            - "위로/아래로/왼쪽/오른쪽" -> nudge_up, nudge_down, nudge_left, nudge_right
        8. 물체 이름 매핑: 사용자가 부르는 용어가 달라도 표준 영어 단어를 사용하여 'fetch_단어' 형태로 만드세요.
        9. 사용자가 사용하는 언어를 자동으로 감지하세요.
        10. 응답 메시지('msg')는 반드시 사용자가 말한 것과 동일한 언어로 작성하세요.

        응답은 반드시 JSON 형식이어야 합니다: {"payload": "생성된_토픽", "msg": "로봇의 응답 멘트"}
        """
        
        self.get_logger().info("단어 분할 발행 방식의 지능형 노드가 시작되었습니다.")

    def publish_cmd(self, payload):
        """명령 토픽 발행 (발행 후 다시 인식 가능 상태로 전환 준비)"""
        if payload and payload not in ["NONE", "STANDBY"]:
            msg = String()
            msg.data = payload
            self.publisher_.publish(msg)
            self.get_logger().info(f'토픽: {msg.data}')

    def tts_callback(self, msg):
        """비전 노드 피드백 수신 (예: FOUND:iron, NOT_FOUND:magnifier)"""
        incoming = msg.data
        if "NOT_FOUND" in incoming:
            obj = incoming.split(":")[-1]
            self.handle_dynamic_feedback(f"사용자가 요청한 {obj}를 찾지 못했습니다.")
        else:
            self.display_and_speak(incoming)

    def handle_dynamic_feedback(self, context):
        """GPT가 상황에 맞는 유연한 대답 생성"""
        try:
            res = self.stt.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": "너는 작업 보조 로봇이야. 상황을 보고받고 한 문장으로 대답해."},
                          {"role": "user", "content": context}]
            )
            self.display_and_speak(res.choices[0].message.content)
        except Exception as e:
            self.get_logger().error(f"피드백 생성 오류: {e}")

    def display_and_speak(self, text):
        self.get_logger().info(f'로봇 응답: {text}')
        try:
            response = self.stt.client.audio.speech.create(model="tts-1", voice="nova", input=text)
            byte_stream = io.BytesIO(response.content)
            pygame.mixer.music.load(byte_stream)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                rclpy.spin_once(self, timeout_sec=0)
                time.sleep(0.1)
        except Exception as e:
            self.get_logger().error(f'TTS 오류: {e}')

    def handle_intelligence(self, text):
        """GPT가 언어를 분석하여 토픽 명칭을 동적으로 결정"""
        try:
            response = self.stt.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": text}
                ],
                response_format={ "type": "json_object" }
            )
            res_dict = json.loads(response.choices[0].message.content)
            payload = res_dict.get("payload", "NONE")
            answer_msg = res_dict.get("msg", "무엇을 도와드릴까요?")

            if payload == "program_finish":
                self.display_and_speak(answer_msg)
                self.publish_cmd(payload)
                return "program_finish"
            
            if payload == "standby":
                return "standby"

            self.display_and_speak(answer_msg)
            self.publish_cmd(payload)
            return "SUCCESS"

        except Exception as e:
            self.get_logger().error(f"GPT 시맨틱 분석 오류: {e}")
            return "RETRY"

    def run_loop(self):
        try:
            while rclpy.ok():
                self.mic.open_stream()
                self.wakeup.set_stream(self.mic.stream)
                while rclpy.ok():
                    rclpy.spin_once(self, timeout_sec=0.1)
                    if self.wakeup.is_wakeup(): 
                        self.display_and_speak("네, 말씀하세요.")
                        break
                self.mic.close_stream()
                
                # 연속 명령 모드
                while rclpy.ok():
                    audio = self.mic.record_audio()
                    if audio:
                        f = io.BytesIO(audio); f.name = "mic.wav"
                        res = self.stt.client.audio.transcriptions.create(model="whisper-1", file=f)
                        if res.text:
                            self.get_logger().info(f"사용자: {res.text}")
                            result = self.handle_intelligence(res.text)
                            
                            if result == "program_finish": 
                                return 
                            elif result == "standby":
                                break 
                    rclpy.spin_once(self, timeout_sec=0)
        finally:
            self.mic.close_stream()

def main(args=None):
    rclpy.init(args=args)
    node = VisionPointingVoiceNode()
    try:
        node.run_loop()
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__':
    main()
