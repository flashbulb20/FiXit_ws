import MicController
from wakeup_word import WakeupWord
from STT import STT
import os
from dotenv import load_dotenv

def main():
    # 1. 초기 설정
    load_dotenv(dotenv_path=".env")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    # 마이크 컨트롤러 및 STT 객체 생성
    mic = MicController.MicController()
    stt = STT(openai_api_key)
    
    # 웨이크워드 객체 생성
    wakeup = WakeupWord(mic.config.buffer_size)
    
    print("시스템이 준비되었습니다. '헬로 rokey'라고 불러보세요.")

    try:
        while True:
            # 2. 마이크 스트림 열기 (웨이크워드 대기 모드)
            mic.open_stream()
            wakeup.set_stream(mic.stream)
            
            print("웨이크워드 감시 중...")
            
            # 3. 웨이크워드가 감지될 때까지 무한 루프
            while True:
                if wakeup.is_wakeup():
                    break
            
            # 4. 웨이크워드 감지 후 처리
            print(">>> 웨이크워드 감지! 명령을 말씀해주세요.")
            
            # 마이크 스트림을 닫아야 STT 모듈(sounddevice)에서 마이크 권한을 사용할 수 있습니다.
            mic.stream.stop_stream()
            mic.stream.close()
            
            # 5. STT 실행 (5초 녹음 및 텍스트 변환)
            text_result = stt.speech2text()
            print(f"인식된 문장: {text_result}")
            
            # (선택 사항) 특정 단어가 포함되었을 때 종료하거나 추가 액션을 넣을 수 있습니다.
            if "종료" in text_result:
                print("프로그램을 종료합니다.")
                break
            
            print("\n다시 대기 상태로 돌아갑니다...")

    except KeyboardInterrupt:
        print("\n사용자에 의해 종료되었습니다.")
    finally:
        mic.audio.terminate()

if __name__ == "__main__":
    main()