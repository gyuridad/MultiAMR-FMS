#!/usr/bin/env python3
import socket
import struct
import time
import cv2
from libcamera import Transform
from picamera2 import Picamera2

"""
TCP로 JPEG 프레임을 length-prefix 방식으로 전송
[packet]
  uint32: payload_len (network byte order)
  payload: jpeg_bytes

  Picamera2로 RGB 프레임을 캡처 → OpenCV로 JPEG 인코딩 → (길이 4바이트 + JPEG bytes) 형태로 TCP로 10Hz 전송
"""

HOST = "0.0.0.0"   # 모든 네트워크 인터페이스에서 접속 허용 (같은 공유기/망의 노트북이 접속 가능)
PORT = 9001        # 접속 포트
IMG_W = 640
IMG_H = 480
JPEG_QUALITY = 80  # JPEG 품질(높을수록 용량↑, 화질↑)
SEND_HZ = 10.0     # 초당 10장 목표(= 0.1초마다 1프레임) / 노트북 코드 tick_hz와 맞추면 예측 가능

# TCP의 sock.send()는 항상 요청한 바이트를 전부 보내준다는 보장이 없어.
# 예: 10000바이트 보내려 했는데 OS 상황에 따라 3000바이트만 보내고 반환할 수 있음.
# 그래서 send_all()은 남은 데이터를 끝까지 반복해서 보내는 함수야.
def send_all(sock: socket.socket, data: bytes):
    view = memoryview(data)
    while len(view):
        n = sock.send(view)
        view = view[n:]
        # 동작:
        #     memoryview(data)로 복사 없이 뷰 생성
        #     sock.send(view)로 일부 전송
        #     전송된 만큼 잘라서(view = view[n:]) 남은 부분 계속 전송
        #     남은 길이가 0이 될 때까지 반복

def main():
    # 카메라 준비
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"format": "RGB888", "size": (IMG_W, IMG_H)},
        # transform=Transform(vflip=1, hflip=1),   # 비전 판단 노드 실행때 사용
        transform=Transform(vflip=1),  # 아루코마커 찾을때 사용
    )
    picam2.configure(config)
    picam2.start()

    # TCP 서버 소켓 만들고 대기(listen)
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)    # SOCK_STREAM: TCP
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # SO_REUSEADDR: 재실행 시 “포트 아직 점유중” 문제를 줄여줌
    srv.bind((HOST, PORT))
    srv.listen(1)                                              # listen(1): 동시에 대기할 수 있는 연결 큐를 1로 설정(한 명만 받겠다 느낌)
    print(f"[PinkyCamServer] listen on {HOST}:{PORT}")

    period = 1.0 / max(1e-6, SEND_HZ)
    t_next = time.time()

    try:
        while True:
            # 클라이언트가 접속하면 accept
            conn, addr = srv.accept()    
                # accept()는 클라이언트가 연결될 때까지 블록(대기) 함
                # 연결되면 conn(통신용 소켓)과 addr(클라이언트 주소)를 받음
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # TCP_NODELAY: Nagle 알고리즘을 꺼서 작은 패킷도 빨리 보내도록(지연 줄이기)
            print(f"[PinkyCamServer] client connected: {addr}")

            try:
                while True:
                    # 주기 고정
                    now = time.time()
                    # “지금이 다음 전송 시간(t_next)보다 빠르면” 잠깐 자고 기다림
                    if now < t_next:
                        time.sleep(min(0.005, t_next - now))
                        continue
                    # 시간이 됐으면 프레임 처리하고, 다음 목표 시간은 t_next += period로 “앞으로 밀어둠”
                    t_next += period
                        # 이렇게 하면 루프가 어떤 이유로 잠깐 느려져도,
                        # 기준이 “현재시간 = now”가 아니라 “목표시간 = t_next”라서 전송 주기가 비교적 일정해짐

                    # 프레임 캡처(RGB) → JPEG 인코딩
                    frame_rgb = picam2.capture_array()

                    # JPEG 인코딩은 OpenCV가 BGR을 기대하는 경우가 많아서 변환
                    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                    ok, buf = cv2.imencode(
                        ".jpg", frame_bgr,
                        [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
                    )
                    if not ok:
                        continue

                    # length-prefix 패킷 전송(핵심)
                    jpg = buf.tobytes()
                    header = struct.pack("!I", len(jpg))
                    send_all(conn, header)   # len(jpg)를 먼저 4바이트로 보냄
                    send_all(conn, jpg)      # 그 다음에 JPEG 바이트 덩어리를 그대로 보냄
                        # 이 방식의 장점:
                        #     TCP는 “메시지 경계”가 없어서, 그냥 recv()하면 프레임이 잘릴 수도/합쳐질 수도 있음
                        #     그래서 “프레임 길이”를 먼저 보내면, 클라이언트는:
                        #     4바이트 먼저 정확히 읽고 길이 N을 얻고
                        #     그 다음 N바이트를 정확히 읽으면
                        #     → 프레임 단위로 복원이 가능해짐

            except (BrokenPipeError, ConnectionResetError):
                print("[PinkyCamServer] client disconnected")
            except Exception as e:
                print("[PinkyCamServer] error:", e)
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

    finally:
        try:
            picam2.stop()
            picam2.close()
        except Exception:
            pass
        srv.close()

if __name__ == "__main__":
    main()


### 핑키에서 실행
# ros2 run sensors image_socket
# python3 pinky_cam_server.py

# ## 소켓 찌꺼기 확인
# ps aux | grep follow_aruco | grep -v grep

# ## FollowAruco 먼저 종료 (클라이언트부터 끊기)
# kill 13797

# ## image_socket 종료 (서버 종료)
# kill 13793