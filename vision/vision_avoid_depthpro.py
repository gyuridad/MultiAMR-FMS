#!/usr/bin/env python3
"""
DepthPro + YOLO (SYNC) + Wedge(부채꼴) 기반 회피 판단 노드

- Picamera2로 RGB 프레임 캡처
- DepthPro로 depth 맵 추정
- 화면 중심 기준 각도 구간(wedge)별 depth 통계 계산
- 전방이 가까우면 좌/우 wedge의 q10(가까운쪽 분위수) 비교로 회피 방향 suggest
- 결과를 JSON(String) 토픽으로 publish
- MJPEG 스트리밍(overlay 포함) 옵션 제공

✅ A 방식(토픽 기반 mover) 안정화를 위해 다음 2개 반영:
  1) payload에 seq(증가 번호) 추가  -> mover가 "움직임 이후 새 판단" 구분 쉬움
  2) 타이머 주기 고정 (기본 10Hz 목표) -> 루프 폭주 방지 + 예측 가능한 발행 흐름
"""

import time
import json
import threading
import math

import cv2
import numpy as np
import torch
from PIL import Image
from typing import Tuple, Dict, Any

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from flask import Flask, Response, render_template_string
# from libcamera import Transform
# from picamera2 import Picamera2

from ultralytics import YOLO
import depth_pro
# git clone https://github.com/apple/ml-depth-pro.git
# cd ml-depth-pro
# pip install -e .
# ml-depth-pro 안에 get_pretrained_models.sh 내용 확인하여 내용 그대로 설치

from .tcp_frame_client import TcpJpegFrameClient


# ---------------- FPS Meter ----------------
class FPSMeter:
    def __init__(self, avg_window=30):
        # avg_window는 FPS를 계산할 때 사용할 최근 프레임 수(윈도우 크기)
        # 최근 avg_window개의 프레임 간 시간(dt)만 유지
        # 그 평균으로 FPS를 계산
        self.avg_window = int(avg_window)
        self.dt = []
        self.t_last = None
        self.fps = 0.0

    def tick(self):
        t = time.time()
        if self.t_last is not None:
            d = t - self.t_last
            self.dt.append(d)
            # 이 두 줄은 **self.dt 리스트를 “최근 avg_window개만 유지”**하려는 코드
            # 그래서 최대 avg_window개까지만 남기고, 그보다 많아지면 가장 오래된 값(맨 앞)을 하나 제거함
            if len(self.dt) > self.avg_window:
                self.dt.pop(0)

            # 지금까지 모아둔 dt(프레임 간 시간 간격)들의 평균값(mean Δt) 를 구하는 코드
            mean_dt = sum(self.dt) / max(1, len(self.dt))
            self.fps = (1.0 / mean_dt) if mean_dt > 1e-9 else 0.0
        self.t_last = t
        return self.fps
    
def resize_depth_to_frame(depth_m: np.ndarray, H0: int, W0: int) -> np.ndarray:
    if depth_m is None:
        return None
    Hd, Wd = depth_m.shape[:2]
    if (Hd, Wd) == (H0, W0):
        return depth_m
    return cv2.resize(depth_m, (W0, H0), interpolation=cv2.INTER_LINEAR)

def fmt2(v):
    return f"{v:.2f}" if (v is not None and np.isfinite(v)) else "N/A"
    

# ---------------- Flask Stream Server ----------------
class SimpleStreamServer:
    def __init__(self, host="0.0.0.0", port=5001):
        self.host = host
        self.port = port
        self._frame_lock = threading.Lock()
        self._latest_bgr = None
        self._app = Flask(__name__)
        self._register_routes()

    def _register_routes(self):
        @self._app.route("/")
        def index():
            return render_template_string(
                """<!doctype html>
<html><head><meta charset="utf-8"><title>Wedge Avoid Stream</title></head>
<body style="background:#111;color:#eee;text-align:center;font-family:Arial">
<h2>DepthPro Wedge Avoid (SYNC)</h2>
<img src="/stream" style="max-width:95vw; border:2px solid #444; border-radius:10px"/>
</body></html>"""
            )

        @self._app.route("/stream")
        def stream():
            return Response(
                self._stream_frames(),
                mimetype="multipart/x-mixed-replace; boundary=frame",
            )

    def set_latest_bgr(self, frame_bgr):
        with self._frame_lock:
            self._latest_bgr = frame_bgr

    def _get_latest_bgr(self):
        with self._frame_lock:
            if self._latest_bgr is None:
                return None
            return self._latest_bgr.copy()

    def _stream_frames(self):
        while True:
            frame = self._get_latest_bgr()
            if frame is None:
                time.sleep(0.05)
                continue
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ok:
                time.sleep(0.05)
                continue
            jpg = buf.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            )
            time.sleep(0.03)

    def start(self):
        th = threading.Thread(
            target=self._app.run,
            kwargs=dict(
                host=self.host,
                port=self.port,
                debug=False,
                threaded=True,
                use_reloader=False,
            ),
            daemon=True,
        )
        th.start()
        return th
    
# ---------------- Wedge / Polar ROI Utils ----------------
# 이 함수는 이미지에서 픽셀의 x좌표(u)가 카메라 중심에서 얼마나 왼쪽/오른쪽에 있는지를 보고,
# 그게 **카메라 기준으로 몇 라디안(rad) 방향인지(수평 각도)**로 바꿔주는 거야
# 예를 들어:
#     fx = 600, cx = 320
#     픽셀 u = 320 → arctan(0/600)=0 rad (정면)
#     픽셀 u = 620 → arctan(300/600)=arctan(0.5)≈0.464 rad ≈ 26.6° 오른쪽
#     픽셀 u = 20 → arctan(-300/600)≈-0.464 rad ≈ 26.6° 왼쪽
def theta_from_x(u: np.ndarray, fx: float, cx: float) -> np.ndarray:
    """픽셀 x좌표 -> 수평각(rad)"""
    return np.arctan((u - cx) / fx)

# 이 함수 wedge_mask()는 이미지에서 “부채꼴(각도 구간) + 바닥쪽(y구간)” 영역만 True로 표시하는 2D 마스크(H×W) 를 만들어줘요.
# “카메라 기준 각도 ROI” 를 만들 때 쓰는 핵심 함수
# 출력: mask (shape: (H, W), dtype: bool)
#     True: “관심영역(부채꼴+바닥쪽)”
#     False: 그 외 영역
def wedge_mask(
    H: int,
    W: int,
    fx: float,
    cx: float,
    deg_min: float,
    deg_max: float,
    y_min_ratio: float = 0.55,   # 이미지 높이의 아래쪽 55% 지점부터
    y_max_ratio: float = 0.98,   # 거의 바닥(98%)까지
) -> np.ndarray:
    """
    deg_min~deg_max: 수평각 범위(도)
      - 왼쪽: 음수, 오른쪽: 양수
    y_min_ratio~y_max_ratio: 바닥쪽(y가 큰 영역)만 보려는 비율
    """
    # 1) x좌표(픽셀 열 인덱스) 배열 만들기
    x = np.arange(W, dtype=np.float32)   # x = [0, 1, 2, ..., W-1]
    # 2) 각 열(x픽셀)이 의미하는 “수평각(theta)”로 변환
    theta = theta_from_x(x, fx, cx)      
        # 결과 theta는 길이 W짜리 배열:
        #     theta[j] = “j번째 열이 카메라 정면에서 몇 rad 방향인지”
        #     중심 cx는 0 rad 근처
        #     왼쪽은 음수, 오른쪽은 양수

    # 3) 입력된 각도 범위(deg)를 라디안으로 바꾸고 정렬
    th0 = math.radians(deg_min)
    th1 = math.radians(deg_max)
    th_lo, th_hi = (th0, th1) if th0 <= th1 else (th1, th0)
        # 혹시 사용자가 거꾸로 넣어도(예: deg_min=10, deg_max=-30) 문제 없게
        # 작은 값을 th_lo, 큰 값을 th_hi로 정렬

    # 4) “각도 범위에 해당하는 열(column)들”만 True인 1D 마스크 생성
    xmask = (theta >= th_lo) & (theta <= th_hi)  # (W,)
        # xmask[j] == True면 j번째 열이 deg_min~deg_max 각도 사이에 들어감

    # 5) y 범위를 비율로 받아서 실제 픽셀 범위로 변환 + 클램프
    y0 = int(H * y_min_ratio)
    y1 = int(H * y_max_ratio)
    y0 = max(0, min(H - 1, y0))
    y1 = max(0, min(H, y1))

    # 6) “y 범위에 해당하는 행(row)들”만 True인 1D 마스크 생성
    ymask = np.zeros((H,), dtype=bool)
    ymask[y0:y1] = True

    # 7) xmask와 ymask를 2D로 브로드캐스팅해서 “부채꼴+바닥쪽” 최종 마스크 만들기
    return ymask[:, None] & xmask[None, :]

def depth_metrics_in_mask(depth_m: np.ndarray, mask: np.ndarray, free_z_thr: float) -> dict:
    vals = depth_m[mask]
    vals = vals[np.isfinite(vals)]
    vals = vals[(vals > 0) & (vals < 50.0)]   # 이건 “제거”라기보다 센서/모델 오류 방지용 최소 클리핑
    if vals.size == 0:
        return {"valid": 0, "z_mean": None, "z_wmean": None, "z_min": None, "free_ratio": 0.0}

    z_mean = float(np.mean(vals))  # 그냥 평균 (참고용)
    z_min  = float(np.min(vals))   # 최단거리 (노이즈에 민감)

    # ✅ 가까운 픽셀에 더 큰 가중치: w = 1/(z^p)
    p = 2.0
    w = 1.0 / np.maximum(vals, 1e-3) ** p
    z_wmean = float(np.sum(vals * w) / np.sum(w))

    free_ratio = float(np.mean(vals > free_z_thr))
    return {"valid": int(vals.size), "z_mean": z_mean, "z_q10": z_wmean, "z_min": z_min, "free_ratio": free_ratio}
    

# 정면/왼쪽/오른쪽 부채꼴(wedge)에서 뽑은 depth 지표를 보고, “정지/좌회피/우회피/그냥 진행” 같은 결정을 내리는 함수
def decide_avoid_from_wedges(
    center_m: dict,
    left_m: dict,
    right_m: dict,
    avoid_trigger_z: float,   # **“회피를 시작할지 말지”**를 정하는 기준 거리(미터)
    hard_stop: float,         # **“너무 가까우면 무조건 정지”**하는 더 강한 기준 거리(미터)
    diff_margin: float,       # 좌/우 중 어느 쪽이 ‘더 낫다’고 확실히 말할 수 있는 최소 차이
        # 차이가 이 값보다 작으면 → “둘 다 비슷” → 애매하면 직진/느리게/기본 방향 유지 같은 정책을 할 수 있음
    min_free_ratio: float,    # 어떤 방향을 “갈 수 있는 방향”으로 인정하기 위한 최소 뚫림 비율
    basic_dir: str = "left",   # ✅ 동점/비슷하면 기본 방향 (기본=left)
) -> Tuple[str, Dict[str, Any]]:
    """
    front(q10)이 가까우면 회피 판단 시작.
    - 양쪽 q10이 hard_stop보다 작으면 stop
    - free_ratio가 너무 낮은 쪽은 통로로 보기 어려움
    - 기본은 q10이 더 큰 쪽(=가까운 장애물이 더 멀리 있음)으로 회피

    return: (suggest, info)
      suggest: "none" | "left" | "right" | "stop"
    """
    # --- 대표값(여기서는 q10를 기준으로 비교) ---
    fn = center_m.get("z_q10", None)
    ln = left_m.get("z_q10", None)
    rn = right_m.get("z_q10", None)

    # valid/free_ratio
    lf = float(left_m.get("free_ratio", 0.0))
    rf = float(right_m.get("free_ratio", 0.0))
    lv = int(left_m.get("valid", 0))
    rv = int(right_m.get("valid", 0))

    # ------------------ 0) 정면 데이터 없으면 보수적으로 stop ------------------
    if fn is None:
        return "stop", {"reason": "no_front_data"}

    # ------------------ 1) 하드 스톱 ------------------
    if fn < hard_stop:
        return "stop", {"reason": "hard_stop", "front_q10": fn, "hard_stop": hard_stop}

    # ------------------ 2) 정면이 위험하지 않으면 그냥 진행 ------------------
    if fn >= avoid_trigger_z:
        return "none", {"reason": "front_ok", "front_q10": fn, "avoid_trigger_z": avoid_trigger_z}

    # 여기부터는 "정면 위험"이므로 좌/우 중 선택해야 함

    # ------------------ 3) free_ratio 기반: 한쪽이 너무 좁으면 반대 선택 ------------------
    lf_bad = (lv > 0 and lf < min_free_ratio)
    rf_bad = (rv > 0 and rf < min_free_ratio)

    if lf_bad and (not rf_bad) and (rn is not None):
        return "right", {
            "reason": "left_narrow",
            "front_q10": fn,
            "L_free": lf, "R_free": rf,
            "L": ln, "R": rn
        }

    if rf_bad and (not lf_bad) and (ln is not None):
        return "left", {
            "reason": "right_narrow",
            "front_q10": fn,
            "L_free": lf, "R_free": rf,
            "L": ln, "R": rn
        }

    # ------------------ 4) 좌/우 모두 후보: 비슷하면 기본 방향(기본=left) ------------------
    both_wide = (lf >= min_free_ratio) and (rf >= min_free_ratio)

    if (ln is not None) and (rn is not None):
        diff = float(abs(ln - rn))

        # ✅ 좌/우가 비슷 + 둘 다 넓으면 → 기본 방향으로 간다(기본 left)
        if diff < diff_margin and both_wide:
            pick = "left" if basic_dir != "right" else "right"
            return pick, {
                "reason": "similar_sides_basic_dir",
                "front_q10": fn,
                "diff": diff,
                "basic_dir": pick,
                "L": ln, "R": rn,
                "L_free": lf, "R_free": rf
            }

    # ------------------ 5) 비슷하지 않으면 더 좋은 쪽(q10 큰 쪽) ------------------
        if ln > rn + diff_margin:
            return "left", {
                "reason": "left_more_clear",
                "front_q10": fn,
                "diff": diff,
                "L": ln, "R": rn,
                "L_free": lf, "R_free": rf
            }

        if rn > ln + diff_margin:
            return "right", {
                "reason": "right_more_clear",
                "front_q10": fn,
                "diff": diff,
                "L": ln, "R": rn,
                "L_free": lf, "R_free": rf
            }

        # 여기까지 왔는데도 애매하면(둘 다 넓지 않거나 diff가 애매한 구간)
        return "stop", {
            "reason": "ambiguous_or_not_wide",
            "front_q10": fn,
            "diff": diff,
            "L": ln, "R": rn,
            "L_free": lf, "R_free": rf
        }

    # ------------------ 6) 한쪽만 있으면 있는 쪽 선택 ------------------
    if ln is not None and rn is None:
        return "left", {"reason": "only_left_available", "front_q10": fn, "L": ln, "L_free": lf}
    if rn is not None and ln is None:
        return "right", {"reason": "only_right_available", "front_q10": fn, "R": rn, "R_free": rf}

    # ------------------ 7) 둘 다 없으면 stop ------------------
    return "stop", {"reason": "no_side_data", "front_q10": fn}


# ---------------- Main ROS2 Node ----------------
class WedgeAvoid(Node):
    def __init__(self):
        super().__init__("wedge_avoid")

        # ---------- Params ----------
        self.declare_parameter("img_w", 640)
        self.declare_parameter("img_h", 480)

        self.declare_parameter("json_topic", "/perception/avoid_wedge_json")

        # wedge angle ranges (deg)
        self.declare_parameter("center_deg_min", -8.0)
        self.declare_parameter("center_deg_max", +8.0)
        self.declare_parameter("left_deg_min", -45.0)
        self.declare_parameter("left_deg_max", -15.0)
        self.declare_parameter("right_deg_min", +15.0)
        self.declare_parameter("right_deg_max", +45.0)

        # wedge y좌표-range (bottom focus)
        self.declare_parameter("y_min_ratio", 0.30)
        self.declare_parameter("y_max_ratio", 0.50)

        # metrics thresholds
        self.declare_parameter("avoid_trigger_z", 3.0)
        self.declare_parameter("hard_stop", 0.2)
        self.declare_parameter("diff_margin", 0.15)
        self.declare_parameter("min_free_ratio", 0.20)
        self.declare_parameter("free_z_thr", 1.2) 
            # “이 거리 임계값보다 멀면(깊이가 크면) 그 픽셀은 ‘장애물 없음/비어있음(free)’로 치자”
        
        # publish rate (timer period)
        self.declare_parameter("tick_hz", 10.0)   # ✅ 10Hz 목표

        # Flask stream
        self.declare_parameter("flask_enable", True)
        self.declare_parameter("flask_host", "0.0.0.0")
        self.declare_parameter("flask_port", 5001)

        # ---------- Camera intrinsics ----------
        self.K = np.array(
            [
                [598.5252422042978, 0.0, 321.35841069961424],
                [0.0, 596.80057563617913, 249.25531392912907],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        self.fx = float(self.K[0, 0])
        self.cx = float(self.K[0, 2])

        # ---------- Read params ----------
        self.img_w = int(self.get_parameter("img_w").value)
        self.img_h = int(self.get_parameter("img_h").value)
        self.json_topic = str(self.get_parameter("json_topic").value)

        self.center_deg_min = float(self.get_parameter("center_deg_min").value)
        self.center_deg_max = float(self.get_parameter("center_deg_max").value)
        self.left_deg_min = float(self.get_parameter("left_deg_min").value)
        self.left_deg_max = float(self.get_parameter("left_deg_max").value)
        self.right_deg_min = float(self.get_parameter("right_deg_min").value)
        self.right_deg_max = float(self.get_parameter("right_deg_max").value)

        self.y_min_ratio = float(self.get_parameter("y_min_ratio").value)
        self.y_max_ratio = float(self.get_parameter("y_max_ratio").value)

        self.avoid_trigger_z = float(self.get_parameter("avoid_trigger_z").value)
        self.hard_stop = float(self.get_parameter("hard_stop").value)
        self.diff_margin = float(self.get_parameter("diff_margin").value)
        self.min_free_ratio = float(self.get_parameter("min_free_ratio").value)
        self.free_z_thr = float(self.get_parameter("free_z_thr").value)

        self.tick_hz = float(self.get_parameter("tick_hz").value)
        if self.tick_hz <= 0:
            self.tick_hz = 10.0
        self.tick_period = 1.0 / self.tick_hz  # ✅ 타이머 주기 고정

        self.flask_enable = bool(self.get_parameter("flask_enable").value)
        self.flask_host = str(self.get_parameter("flask_host").value)
        self.flask_port = int(self.get_parameter("flask_port").value)

        # ---------- ROS pub ----------
        self.pub_json = self.create_publisher(String, self.json_topic, 10)

        # 각 요청을 구분하는 ID(번호표)
        self.seq = 0

        # ---------- Device ----------
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"device={self.device}")

        # ---------- DepthPro model ----------
        self.depth_model, self.transform = depth_pro.create_model_and_transforms(
            device=self.device, precision=torch.float16
        )
        self.depth_model.eval()
        self.f_px = torch.tensor([float(self.fx)], device=self.device, dtype=torch.float32)

        # ---------- Picamera2 이 방식은 아래 방식으로 대체함 ----------
        # self.picam2 = Picamera2()
        # config = self.picam2.create_preview_configuration(
        #     main={"format": "RGB888", "size": (self.img_w, self.img_h)},
        #     transform=Transform(vflip=1),
        # )
        # self.picam2.configure(config)
        # self.picam2.start()

        # ---- TCP camera client params ----
        self.declare_parameter("cam_host", "192.168.0.45")  # 핑키 IP
        self.declare_parameter("cam_port", 9001)

        self.cam_host = str(self.get_parameter("cam_host").value)
        self.cam_port = int(self.get_parameter("cam_port").value)

        self.cam = TcpJpegFrameClient(self.cam_host, self.cam_port)
        self.cam.start()
        self.get_logger().info(f"[TCP CAM] connect to {self.cam_host}:{self.cam_port}")

        # ---------- Stream ----------
        self.stream = SimpleStreamServer(self.flask_host, self.flask_port) if self.flask_enable else None
        if self.stream:
            self.stream.start()
            self.get_logger().info(f"[Flask] http://<raspi_ip>:{self.flask_port}/  (stream=/stream)")

        # ---------- FPS ----------
        self.main_fps = FPSMeter(avg_window=30)

        # ---------- Precompute wedge masks ----------
        self._build_wedge_masks(self.img_h, self.img_w)

        self.get_logger().info(
            f"Wedges(deg): center[{self.center_deg_min},{self.center_deg_max}] "
            f"left[{self.left_deg_min},{self.left_deg_max}] "
            f"right[{self.right_deg_min},{self.right_deg_max}] "
            f"y_ratio[{self.y_min_ratio},{self.y_max_ratio}]  "
            f"tick_hz={self.tick_hz}"
        )

        self._running = True

        # ✅ 타이머 주기 고정
        self.timer = self.create_timer(self.tick_period, self._tick)
            # tick 함수 흐름:
            #     카메라 프레임 캡처 → Depth 추론 → 좌/정면/우 분석 → 회피 방향 결정 → 화면 오버레이 → JSON 발행 → 스트리밍 갱신

    # 이 함수의 “결과”는 클래스 멤버 변수에 저장되는 3개의 마스크 배열
    # True인 곳 = 해당 wedge(부채꼴 + 바닥 y구간) 영역
    # False인 곳 = 그 외
    def _build_wedge_masks(self, H: int, W: int):
        self.center_mask = wedge_mask(
            H, W, self.fx, self.cx,
            self.center_deg_min, self.center_deg_max,
            self.y_min_ratio, self.y_max_ratio
        )
        self.left_mask = wedge_mask(
            H, W, self.fx, self.cx,
            self.left_deg_min, self.left_deg_max,
            self.y_min_ratio, self.y_max_ratio
        )
        self.right_mask = wedge_mask(
            H, W, self.fx, self.cx,
            self.right_deg_min, self.right_deg_max,
            self.y_min_ratio, self.y_max_ratio
        )
            # 예를 들어 H=6, W=10처럼 아주 작은 이미지라고 치고(설명용)
            # self.center_mask.shape
            #     # (6, 10)
            # self.center_mask
            #     # array([
            #     #  [False False False False False False False False False False],
            #     #  [False False False False False False False False False False],
            #     #  [False False False  True  True  True False False False False],
            #     #  [False False False  True  True  True False False False False],
            #     #  [False False False  True  True  True False False False False],
            #     #  [False False False  True  True  True False False False False],
            #     # ], dtype=bool)
            # 위처럼 **아래쪽 행(y가 큰 영역)**에만 True가 있고
            # 그 중에서도 **센터 각도 범위에 해당하는 열(x)**만 True가 켜진 모습

        over_lr = float(np.mean(self.left_mask & self.right_mask))
        over_lc = float(np.mean(self.left_mask & self.center_mask))
        over_rc = float(np.mean(self.right_mask & self.center_mask))
        self.get_logger().info(f"[mask overlap] lr={over_lr:.6f} lc={over_lc:.6f} rc={over_rc:.6f}")

    def destroy_node(self):
        self._running = False
        try:
            # self.picam2.stop()    # 아래 방식으로 대체
            # self.picam2.close()
            self.cam.stop()
        except Exception:
            pass
        super().destroy_node()

    def _tick(self):
        if not self._running:
            return

        # 1) Frame RGB
        # frame_rgb = self.picam2.capture_array()  # 아래 방식으로 대체
        frame_rgb = self.cam.get_latest_rgb()
        if frame_rgb is None:
            # 아직 수신 전이면 이번 tick은 스킵
            return
        
        H0, W0 = frame_rgb.shape[:2]

        # (혹시 해상도 변하면 wedge mask 재생성)
        if (H0, W0) != (self.img_h, self.img_w):
            self.img_h, self.img_w = H0, W0
            self._build_wedge_masks(H0, W0)

        # 2) DepthPro inference
        rgb_pil = Image.fromarray(frame_rgb)
        rgb_tensor = self.transform(rgb_pil)

        with torch.no_grad():
            result = self.depth_model.infer(rgb_tensor, f_px=self.f_px)

        depth_m = result["depth"].detach().cpu().numpy().astype(np.float32)
        depth_frame = resize_depth_to_frame(depth_m, H0, W0)

        # 3) Wedge metrics
        center_m = depth_metrics_in_mask(depth_frame, self.center_mask, free_z_thr=self.free_z_thr)
        left_m = depth_metrics_in_mask(depth_frame, self.left_mask, free_z_thr=self.free_z_thr)
        right_m = depth_metrics_in_mask(depth_frame, self.right_mask, free_z_thr=self.free_z_thr)

        avoid_dir, avoid_meta = decide_avoid_from_wedges(
            center_m, left_m, right_m,
            avoid_trigger_z=self.avoid_trigger_z,
            hard_stop=self.hard_stop,
            diff_margin=self.diff_margin,
            min_free_ratio=self.min_free_ratio,
            basic_dir="left",
        )

        # 4) Overlay (선택)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        overlay = frame_bgr.copy()
        mfps = self.main_fps.tick()

        # ---------- DEBUG: show wedge masks ----------
        # left_mask: magenta, right_mask: yellow
        overlay[self.left_mask]  = (overlay[self.left_mask]  * 0.5 + np.array([255, 0, 255], dtype=np.float32) * 0.5).astype(np.uint8)
        overlay[self.right_mask] = (overlay[self.right_mask] * 0.5 + np.array([0, 255, 255], dtype=np.float32) * 0.5).astype(np.uint8)

        # “카메라 기준 어떤 각도(deg)가 이미지에서 몇 번째 x픽셀(가로 위치)인지”
        # 각도 → 픽셀 x좌표 (theta → u) 역변환
        def x_from_deg(deg):
            th = math.radians(deg)
            return int(self.cx + self.fx * math.tan(th))
                # 핀홀 카메라 모델에서 수평 방향은 이렇게 연결돼:
                #     theta = arctan((u - cx) / fx) (픽셀 → 각도)
                # 가정:
                #     이미지 폭 W=640 → 중심 cx = 320
                #     fx = 600
                #     1) 정면(0도)
                #         deg = 0
                #         th = 0 rad
                #         tan(0) = 0
                #         x = 320 + 600*0 = 320
                #         ✅ 화면 정중앙
                #     2) 오른쪽 10도
                #         deg = +10
                #         th ≈ 0.1745 rad
                #         tan(th) ≈ 0.1763
                #         x = 320 + 600*0.1763 ≈ 320 + 105.8 ≈ 425
                #         ✅ 중앙(320)에서 오른쪽으로 약 106픽셀 이동 → x≈425
                
        for deg in [
            self.left_deg_min, self.left_deg_max,
            self.center_deg_min, self.center_deg_max,
            self.right_deg_min, self.right_deg_max
        ]:
            x = x_from_deg(deg)
            x = max(0, min(W0 - 1, x))
            cv2.line(
                overlay,
                (x, int(H0 * self.y_min_ratio)),
                (x, int(H0 * self.y_max_ratio)),
                (255, 255, 0),
                1,
            )

        txt1 = f"seq={self.seq}  SYNC FPS={mfps:.1f} | avoid={avoid_dir}"
        txt2 = f"front(wmean)={fmt2(center_m.get('z_q10'))}  L(wmean)={fmt2(left_m.get('z_q10'))}  R(wmean)={fmt2(right_m.get('z_q10'))}"
        txt3 = f"L_free={left_m.get('free_ratio',0):.2f}  R_free={right_m.get('free_ratio',0):.2f}  thr={self.free_z_thr:.1f}"
        cv2.putText(overlay, txt1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(overlay, txt2, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(overlay, txt3, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

        L_raw_mean = float(np.mean(depth_frame[self.left_mask]))
        R_raw_mean = float(np.mean(depth_frame[self.right_mask]))
        cv2.putText(overlay, f"L_raw_mean={L_raw_mean:.2f} R_raw_mean={R_raw_mean:.2f}",
                    (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

        # 5) JSON publish (✅ seq 포함)
        self.seq += 1

        payload = {
            "seq": int(self.seq),
            "ts": float(time.time()),
            "fps": float(mfps),
            "frame": {"w": int(W0), "h": int(H0)},
            "wedges": {
                "center_deg": [self.center_deg_min, self.center_deg_max],
                "left_deg": [self.left_deg_min, self.left_deg_max],
                "right_deg": [self.right_deg_min, self.right_deg_max],
                "y_ratio": [self.y_min_ratio, self.y_max_ratio],
                "metrics": {
                    "front": center_m,
                    "left": left_m,
                    "right": right_m,
                },
            },
            "avoidance": {"suggest": avoid_dir, "meta": avoid_meta},
        }

        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.pub_json.publish(msg)

        # 6) Stream
        if self.stream:
            self.stream.set_latest_bgr(overlay)


def main():
    rclpy.init()
    node = WedgeAvoid()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

        


### 실행예시:
# ros2 run pinky_vision vision_avoid --ros-args \
#   -p tick_hz:=8.0 \
#   -p avoid_trigger_z:=1.2 \
#   -p hard_stop:=0.35 \

### 가상환경 실행방법
# $VIRTUAL_ENV/bin/python3   ~/pinky_pro/install/pinky_vision/lib/pinky_vision/vision_avoid

### flask 스트리밍 서버 
# http://127.0.0.1:5001


