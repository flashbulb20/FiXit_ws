import os
import cv2


DEVICE_NUMBER = 6

def main(args=None):
    # 데이터 저장 경로 설정
    HOME_DIR = os.path.expanduser("~")
    source_path = os.path.join(HOME_DIR, "data")
    os.makedirs(source_path, exist_ok=True)
    # 카메라 연결
    print(f"현재 선택된 device number는 {DEVICE_NUMBER}입니다.")
    cap = cv2.VideoCapture(DEVICE_NUMBER)

    # 실행하기 전에 무조건 확인!!!
    count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            print("카메라를 찾을 수 없습니다. DEVICE_NUMBER를 변경해주세요.")
            exit(True)
        cv2.imshow("camera", frame)

        # 사진 캡쳐
        if cv2.waitKey(1) & 0xFF == ord("c"):
            file_name = f"tools_{count}.jpg"
            count += 1
            # 현재 위치 기반 이미지 저장
            cv2.imwrite(f"{source_path}/{file_name}", frame)
            print(f"save img to {source_path}/{file_name}")
        # OpenCV 종료
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
