import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from geometry_msgs.msg import PoseStamped
import threading
import queue

import DR_init

# 로봇 설정
ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"

# 이동 파라미터
VELOCITY = 80
ACCELERATION = 80
SAFE_HEIGHT = 100.0


class RobotMoverNode(Node):   
    def __init__(self):
        super().__init__('robot_mover_node')
        
        # 로봇 함수 import
        from DSR_ROBOT2 import movej, movel, get_current_posx, wait, get_robot_mode, set_robot_mode
        from DR_common2 import posx
        
        self.movej = movej
        self.movel = movel
        self.get_current_posx = get_current_posx
        self.wait = wait
        self.posx = posx
        self.get_robot_mode = get_robot_mode
        self.set_robot_mode = set_robot_mode
        
        # 명령 큐 (토픽에서 받은 명령을 별도 스레드에서 처리)
        self.command_queue = queue.Queue(maxsize=1)  # 최대 1개만 저장
        
        # 구독자 설정
        qos = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        self.subscription = self.create_subscription(
            PoseStamped,
            '/vision/target_pose',
            self.pose_callback,
            qos
        )
        
        # 상태 관리
        self.last_target = None
        self.running = True
        
        # 홈 포지션으로 이동
        self.go_home()
        
        # 로봇 제어 스레드 시작
        self.robot_thread = threading.Thread(target=self._robot_control_loop, daemon=True)
        self.robot_thread.start()
        
        self.get_logger().info("✓ Robot Mover 노드 시작")
        self.get_logger().info(f"  - 속도: {VELOCITY}%, 가속도: {ACCELERATION}%")
        self.get_logger().info(f"  - 안전 높이: {SAFE_HEIGHT}mm")
    
    def go_home(self):
        try:
            home_joint = [0, 0, 90, 0, 90, 0]
            self.get_logger().info("홈 포지션으로 이동 중...")
            self.movej(home_joint, vel=VELOCITY, acc=ACCELERATION)
            self.wait(0.5)
            self.get_logger().info("✓ 홈 포지션 도착")
        except Exception as e:
            self.get_logger().error(f"홈 이동 실패: {e}")
    
    def pose_callback(self, msg: PoseStamped):
        target_x = msg.pose.position.x
        target_y = msg.pose.position.y
        target_z = msg.pose.position.z
        
        # 같은 목표는 무시
        current_target = (round(target_x, 1), round(target_y, 1), round(target_z, 1))
        if self.last_target == current_target:
            return
        
        self.last_target = current_target
        
        # 큐가 가득 차면 기존 명령 버리고 새 명령 추가
        if self.command_queue.full():
            try:
                self.command_queue.get_nowait()
            except queue.Empty:
                pass
        
        try:
            self.command_queue.put_nowait((target_x, target_y, target_z))
            self.get_logger().info(f"타겟 큐 추가: [{target_x:.1f}, {target_y:.1f}, {target_z:.1f}]")
        except queue.Full:
            self.get_logger().warn("명령 큐 가득참")
    
    def _robot_control_loop(self):
        while self.running:
            try:
                # 명령 대기 (timeout으로 종료 확인 가능)
                x, y, z = self.command_queue.get(timeout=0.5)
                
                self.get_logger().info(f"\n{'='*50}")
                self.get_logger().info(f"이동 시작: [{x:.1f}, {y:.1f}, {z:.1f}]")
                
                try:
                    self.move_to_target(x, y, z)
                    self.get_logger().info("✓ 이동 완료")
                except Exception as e:
                    self.get_logger().error(f"이동 실패: {e}")
                
                self.get_logger().info(f"{'='*50}\n")
                
            except queue.Empty:
                continue
    
    def move_to_target(self, x, y, z):
        current_pose = self.get_current_posx()[0]
        rx, ry, rz = current_pose[3], current_pose[4], current_pose[5]
        
        self.get_logger().info(f"현재: {[round(p, 1) for p in current_pose[:3]]}")
        self.get_logger().info(f"목표: [{x:.1f}, {y:.1f}, {z:.1f}]")
        
        # 로봇 상태 확인
        from DSR_ROBOT2 import get_robot_mode
        mode = get_robot_mode()
        self.get_logger().info(f"로봇 모드: {mode}")
        
        if mode != 1:  # 5 = 자동 모드
            self.get_logger().error(f"로봇이 자동 모드가 아닙니다! (현재: {mode})")
            self.get_logger().error("티치펜던트에서 자동 모드로 변경하세요")
            return
        
        try:
            # 1단계: 안전 높이
            safe_pose = self.posx([x, y, z + SAFE_HEIGHT, rx, ry, rz])
            self.get_logger().info(f"1단계: 안전 높이 (+{SAFE_HEIGHT}mm)")
            self.get_logger().info(f"  목표 포즈: {safe_pose}")
            
            result = self.movel(safe_pose, vel=VELOCITY, acc=ACCELERATION)
            self.get_logger().info(f"  movel 반환값: {result}")
            self.wait(0.5)
            
            # 실제 이동했는지 확인
            new_pose = self.get_current_posx()[0]
            self.get_logger().info(f"  이동 후: {[round(p, 1) for p in new_pose[:3]]}")
            
            # 2단계: 목표 위치
            target_pose = self.posx([x, y, z, rx, ry, rz])
            self.get_logger().info("2단계: 목표 위치로 하강")
            self.get_logger().info(f"  목표 포즈: {target_pose}")
            
            result = self.movel(target_pose, vel=VELOCITY//2, acc=ACCELERATION//2)
            self.get_logger().info(f"  movel 반환값: {result}")
            self.wait(1.5)
            
            new_pose = self.get_current_posx()[0]
            self.get_logger().info(f"  이동 후: {[round(p, 1) for p in new_pose[:3]]}")
            
            # 3단계: 복귀
            self.get_logger().info("3단계: 안전 높이로 복귀")
            result = self.movel(safe_pose, vel=VELOCITY, acc=ACCELERATION)
            self.get_logger().info(f"  movel 반환값: {result}")
            self.wait(0.5)
            
            new_pose = self.get_current_posx()[0]
            self.get_logger().info(f"  최종 위치: {[round(p, 1) for p in new_pose[:3]]}")
            
        except Exception as e:
            self.get_logger().error(f"이동 중 에러: {type(e).__name__}: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
    
    def shutdown(self):
        self.get_logger().info("종료 중...")
        self.running = False
        self.robot_thread.join(timeout=2.0)
        self.go_home()


def main(args=None):
    rclpy.init(args=args)
    
    # 로봇 노드 초기화
    robot_init_node = rclpy.create_node("robot_mover_init", namespace=ROBOT_ID)
    DR_init.__dsr__node = robot_init_node
    
    # DSR_ROBOT2 연결 확인
    try:
        from DSR_ROBOT2 import get_current_posx
        current_pose = get_current_posx()[0]
        print(f"✓ 로봇 연결 성공")
        print(f"✓ 현재 포즈: {current_pose}")
    except Exception as e:
        print(f"Error: 로봇 연결 실패 - {e}")
        rclpy.shutdown()
        return
    
    # Mover 노드 실행
    mover_node = RobotMoverNode()
    
    try:
        print("\n=== Robot Mover 실행 중 ===")
        print("Vision 좌표 수신 시 자동 이동")
        print("Ctrl+C로 종료\n")
        
        # spin_once를 반복 (generator 문제 회피)
        while rclpy.ok():
            rclpy.spin_once(mover_node, timeout_sec=0.1)
            
    except KeyboardInterrupt:
        print("\n종료 신호 수신")
    finally:
        mover_node.shutdown()
        rclpy.shutdown()


if __name__ == "__main__":
    main()