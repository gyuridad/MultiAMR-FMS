#!/usr/bin/env python3
import socket
import struct
import threading
import time
import cv2
import numpy as np

def recv_exact(sock: socket.socket, n: int) -> bytes:
    data = bytearray()
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise ConnectionError("socket closed")
        data.extend(chunk)
    return bytes(data)

class TcpJpegFrameClient:
    """
    - 백그라운드 스레드로 계속 수신
    - 항상 '가장 최신 프레임'만 보관 (지연 누적 방지)
    - get_latest_rgb()로 RGB numpy(H,W,3) 리턴
    """
    def __init__(self, host: str, port: int, reconnect_sec=1.0):
        self.host = host
        self.port = port
        self.reconnect_sec = reconnect_sec

        self._lock = threading.Lock()
        self._latest_rgb = None
        self._running = False
        self._th = None

    def start(self):
        self._running = True
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()

    def stop(self):
        self._running = False
        if self._th:
            self._th.join(timeout=1.0)

    def get_latest_rgb(self):
        with self._lock:
            if self._latest_rgb is None:
                return None
            return self._latest_rgb.copy()

    def _set_latest(self, rgb):
        with self._lock:
            self._latest_rgb = rgb

    def _loop(self):
        while self._running:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5.0)
                sock.connect((self.host, self.port))
                sock.settimeout(None)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                print(f"[TcpJpegFrameClient] connected to {self.host}:{self.port}")

                while self._running:
                    hdr = recv_exact(sock, 4)
                    (length,) = struct.unpack("!I", hdr)
                    jpg = recv_exact(sock, length)

                    arr = np.frombuffer(jpg, dtype=np.uint8)
                    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if bgr is None:
                        continue
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    self._set_latest(rgb)

            except Exception as e:
                print("[TcpJpegFrameClient] reconnect:", e)
                time.sleep(self.reconnect_sec)
            finally:
                try:
                    sock.close()
                except Exception:
                    pass
