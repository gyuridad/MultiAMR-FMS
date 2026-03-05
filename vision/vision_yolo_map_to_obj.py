#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yolo_map_projector_once.py

✅ 너의 기존 YOLO+TCP 코드 + (아루코 참조 코드 스타일) TF 행렬 합성(T_map_base @ T_base_cam @ T_cam_obj)
로 "YOLO로 얻은 xyz_cam_m"을 map 좌표로 변환해서 JSON 토픽으로 발행하는 **단일 파일**.

입력
- TCP 카메라: TcpJpegFrameClient(host, port) 로 RGB 프레임 수신
- YOLO: teddy det -> xyz_cam_m 계산 (estimate_center_xyz_from_bbox)
- TF:
    map -> base_frame  (예: base_footprint)
    base_frame -> camera_frame (예: front_camera_link)

출력
- /yolo_target_map (std_msgs/String JSON)
    {
      "seq": 12,
      "class_name": "teddybear",
      "conf": 0.82,
      "xyz_cam_m": [X,Y,Z],
      "xyz_map_m": [mx,my,mz],
      "robot_xy_map_m": [rx, ry],
      "dist_m": 0.53,
      "frames": {"map":"map","base":"base_footprint","camera":"front_camera_link"},
      "meta": {...}
    }

⚠️ 매우 중요
- 이 코드는 xyz_cam_m이 "camera_frame(=front_camera_link)" 기준이라고 가정한다.
- 만약 optical 규약(+Z forward) 기준인데 TF가 camera_link 규약(+X forward)면 결과가 뒤집힐 수 있음.
  그 경우 (A) optical frame을 TF에 추가하거나, (B) xyz축 변환을 넣어야 함.
"""

import math
import time
import json

import cv2
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

import rclpy
from rclpy.node import Node

import tf2_ros
from tf2_ros import TransformException

from std_msgs.msg import String
from ultralytics import YOLO

from .tcp_frame_client import TcpJpegFrameClient

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def _get_class_name(names: Any, cls_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(cls_id, cls_id))
    if isinstance(names, (list, tuple)):
        return str(names[cls_id]) if 0 <= cls_id < len(names) else str(cls_id)
    return str(cls_id)

def quat_to_rot(qx, qy, qz, qw) -> np.ndarray:
    n = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n
    xx, yy, zz = qx*qx, qy*qy, qz*qz
    xy, xz, yz = qx*qy, qx*qz, qy*qz
    wx, wy, wz = qw*qx, qw*qy, qw*qz
    return np.array([
        [1 - 2*(yy + zz),     2*(xy - wz),       2*(xz + wy)],
        [2*(xy + wz),         1 - 2*(xx + zz),   2*(yz - wx)],
        [2*(xz - wy),         2*(yz + wx),       1 - 2*(xx + yy)],
    ], dtype=np.float64)

def rot_to_quat(R: np.ndarray) -> Tuple[float, float, float, float]:
    tr = float(R[0, 0] + R[1, 1] + R[2, 2])
    if tr > 0.0:
        S = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S
    return (float(qx), float(qy), float(qz), float(qw))

def T_from_tfmsg(tfmsg) -> np.ndarray:
    t = tfmsg.transform.translation
    q = tfmsg.transform.rotation
    R = quat_to_rot(q.x, q.y, q.z, q.w)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.array([t.x, t.y, t.z], dtype=np.float64)
    return T

def yaw_from_R(R: np.ndarray) -> float:
    return math.atan2(R[1, 0], R[0, 0])

def wrap_pi(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a

def T_from_xyz(x: float, y: float, z: float) -> np.ndarray:
    """YOLO xyz_cam_m을 camera_frame 기준 translation으로만 해석(회전 I)"""
    T = np.eye(4, dtype=np.float64)
    T[0, 3] = float(x)
    T[1, 3] = float(y)
    T[2, 3] = float(z)
    return T

# ------------------------- YOLO xyz from bbox -------------------------

def estimate_center_xyz_from_bbox(
    x1: float, y1: float, x2: float, y2: float,
    K: np.ndarray,
    real_height_m: float,
    min_bbox_h_px: int = 18,
    scale_s: float = 1.0, 
) -> Tuple[Optional[Tuple[float, float, float]], Dict[str, Any]]:
    fx = float(K[0, 0]); fy = float(K[1, 1])
    cx = float(K[0, 2]); cy = float(K[1, 2])

    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1

    h_px = float(max(1.0, (y2 - y1)))
    if h_px < float(min_bbox_h_px):
        return None, {"ok": False, "reason": "bbox_too_small", "h_px": h_px}

    u = 0.5 * (x1 + x2)
    v = 0.5 * (y1 + y2)

    # “실제 높이(real_height_m)로 Z를 추정하는 핵심”
    # (참고) 테디베어가 서있지 않은 경우를 걸러내는 간단한 조건(예: bbox 종횡비)도 향후 추가 필요 !!
    Z_est = (fy * float(real_height_m)) / h_px
        # 왜 이렇게 계산되냐?
        #     핀홀 카메라 모델에서
        #     이미지에서 보이는 높이(픽셀) h_px는
        #     실제 높이 H에 비례하고
        #     거리 Z에 반비례해.

    s = float(scale_s) if (scale_s is not None) else 1.0
    Z = Z_est * s                      # ✅ 보정

    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

    return (float(X), float(Y), float(Z)), {
        "ok": True, "u": float(u), "v": float(v),
        "h_px": float(h_px), "H_m": float(real_height_m),
        "fx": fx, "fy": fy, "cx": cx, "cy": cy,
        "Z_est": float(Z_est),
        "scale_s": float(s),
    }

def extract_teddy_xyz_from_yolo(
    yolo_model,
    frame_bgr,
    K: np.ndarray,
    teddy_height_m: float = 0.075,     # ✅ 네 테디베어 실측 높이로 바꿔 (m)
    min_bbox_h_px: int = 25,          # ✅ 테디는 작게 잡히기 쉬워서 약간 올리는 걸 추천
    conf: float = 0.4,
    iou: float = 0.45,
    max_dets: int = 10,
    scale_s: float = 1.0, 
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    returns:
      teddy_dets: 테디베어 det들 (xyz_cam 포함)
      best_target: 가장 가까운(Z 최소) 테디 det
    """
    H, W = frame_bgr.shape[:2]
    names = yolo_model.names

    res = yolo_model.predict(
        source=frame_bgr,
        conf=conf,
        iou=iou,
        max_det=max_dets,
        verbose=False
    )[0]

    teddy_dets: List[Dict[str, Any]] = []
    best_target: Optional[Dict[str, Any]] = None

    if res.boxes is None or res.boxes.data is None:
        return teddy_dets, best_target

    for row in res.boxes.data.tolist():
        x1, y1, x2, y2, c, cls = row[:6]
        cls = int(cls); c = float(c)

        # class name 얻기
        cls_name = names.get(cls, str(cls)) if isinstance(names, dict) else names[cls]
        cls_name_norm = str(cls_name).strip().lower().replace(" ", "").replace("_", "")

        # ✅ "teddybear"만 필터
        if "teddybear" not in cls_name_norm:
            continue

        # bbox clamp
        x1 = clamp(float(x1), 0, W - 1)
        x2 = clamp(float(x2), 0, W - 1)
        y1 = clamp(float(y1), 0, H - 1)
        y2 = clamp(float(y2), 0, H - 1)

        xyz_cam, meta = estimate_center_xyz_from_bbox(
            x1, y1, x2, y2,
            K=K,
            real_height_m=float(teddy_height_m),
            min_bbox_h_px=int(min_bbox_h_px),
            scale_s=float(scale_s),
        )

        det = {
            "class_id": cls,
            "class_name": cls_name,
            "conf": c,
            "bbox": [x1, y1, x2, y2],
            "real_height_m": float(teddy_height_m),
            "xyz_cam_m": list(xyz_cam) if xyz_cam is not None else None,
            "xyz_cam_meta": meta,
        }
        # print(f"h_px={meta.get('h_px')} Z={(det['xyz_cam_m'][2] if det['xyz_cam_m'] else None)}")

        teddy_dets.append(det)

        # best target: 가장 가까운 Z
        if xyz_cam is not None:
            if best_target is None or xyz_cam[2] < best_target["xyz_cam_m"][2]:
                best_target = det

    return teddy_dets, best_target


# ------------------------- Main Node -------------------------
class YoloMapProjector(Node):
    def __init__(self):
        super().__init__("yolo_map_projector")

        # ---- Params (너가 쓰는 값들을 기본값으로) ----
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_footprint")
        self.declare_parameter("camera_frame", "front_camera_link")

        self.declare_parameter("robot_name", "")  # e.g., "pinky1"
        self.declare_parameter("out_topic", "")   # 비워두면 자동 생성 / "yolo_target_map"

        self.declare_parameter("tcp_host", "192.168.0.45")
        self.declare_parameter("tcp_port", 9001)

        self.declare_parameter("yolo_model_path", "yolo_model/yolo11m.pt")
        self.declare_parameter("tick_hz", 10.0)

        self.declare_parameter("teddy_height_m", 0.35)
        self.declare_parameter("min_bbox_h_px", 28)
        self.declare_parameter("conf", 0.4)
        self.declare_parameter("iou", 0.45)
        self.declare_parameter("max_dets", 10)
        self.declare_parameter("scale_s", 1.0)

        # map-projection gating (ArUco 스타일)
        self.declare_parameter("need_stable_ticks", 10)
        self.declare_parameter("stable_xy_m", 0.02)
        self.declare_parameter("stable_yaw_deg", 2.0)
        self.declare_parameter("target_fresh_sec", 0.5)

        # output
        self.declare_parameter("show_debug_window", True)

        # ---- Camera intrinsics (네가 쓰던 값) ----
        self.K = np.array(
            [
                [598.5252422042978, 0.0, 321.35841069961424],
                [0.0, 596.8005756361791, 249.25531392912907],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        # ---- Read params ----
        self.map_frame = str(self.get_parameter("map_frame").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.camera_frame = str(self.get_parameter("camera_frame").value)

        self.robot_name = str(self.get_parameter("robot_name").value).strip()
        self.out_topic  = str(self.get_parameter("out_topic").value).strip()

        # ✅ out_topic 자동 생성 (빈 문자열이면)
        if not self.out_topic:
            if self.robot_name:
                self.out_topic = f"/{self.robot_name}/yolo_target_map"
            else:
                self.out_topic = "/yolo_target_map"   # fallback

        self.tcp_host = str(self.get_parameter("tcp_host").value)
        self.tcp_port = int(self.get_parameter("tcp_port").value)

        self.yolo_model_path = str(self.get_parameter("yolo_model_path").value)
        self.tick_hz = float(self.get_parameter("tick_hz").value)
        self.tick_period = 1.0 / max(1e-6, self.tick_hz)

        self.teddy_height_m = float(self.get_parameter("teddy_height_m").value)
        self.min_bbox_h_px = int(self.get_parameter("min_bbox_h_px").value)
        self.conf = float(self.get_parameter("conf").value)
        self.iou = float(self.get_parameter("iou").value)
        self.max_dets = int(self.get_parameter("max_dets").value)
        self.scale_s = float(self.get_parameter("scale_s").value)

        self.need_stable_ticks = int(self.get_parameter("need_stable_ticks").value)
        self.stable_xy_m = float(self.get_parameter("stable_xy_m").value)
        self.stable_yaw_deg = float(self.get_parameter("stable_yaw_deg").value)
        self.target_fresh_sec = float(self.get_parameter("target_fresh_sec").value)

        self.show_debug_window = bool(self.get_parameter("show_debug_window").value)

        # ---- TF ----
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ---- Publisher ----
        self.pub = self.create_publisher(String, self.out_topic, 10)

        # ---- YOLO ----
        self.model = YOLO(self.yolo_model_path)
        self.get_logger().info(f"[YOLO] model loaded: {self.yolo_model_path}")

        # ---- TCP cam ----
        self.cam = TcpJpegFrameClient(self.tcp_host, self.tcp_port)
        self.cam.start()
        self.get_logger().info(f"[TCP CAM] connect to {self.tcp_host}:{self.tcp_port}")

        # ---- Gating / state ----
        self._seq = 0
        self._prev_map_base: Optional[np.ndarray] = None
        self._stable_count = 0
        self._last_best_time = 0.0
        self._last_best_det: Optional[Dict[str, Any]] = None

        # ---- Timer ----
        self.timer = self.create_timer(self.tick_period, self._tick)

        tag = self.robot_name if self.robot_name else self.get_name()

        self.get_logger().info(
            "[YoloMapProjector] start\n"
            f" [{tag}] frames: map={self.map_frame}, base={self.base_frame}, camera={self.camera_frame}\n"
            f" [{tag}] out: {self.out_topic}\n"
            f" [{tag}] tcp={self.tcp_host}:{self.tcp_port}\n"
            f" [{tag}] stable: need={self.need_stable_ticks} ticks, xy<{self.stable_xy_m}m, yaw<{self.stable_yaw_deg}deg\n"
            f" [{tag}] target_fresh<{self.target_fresh_sec}s, tick_hz={self.tick_hz}"
        )

    def destroy_node(self):
        try:
            self.cam.stop()
        except Exception:
            pass
        try:
            if self.show_debug_window:
                cv2.destroyAllWindows()
        except Exception:
            pass
        super().destroy_node()

    def _lookup_T_map_base(self) -> np.ndarray:
        tf_map_base = self.tf_buffer.lookup_transform(self.map_frame, self.base_frame, rclpy.time.Time())
        return T_from_tfmsg(tf_map_base)

    def _lookup_T_base_cam(self) -> np.ndarray:
        tf_base_cam = self.tf_buffer.lookup_transform(self.base_frame, self.camera_frame, rclpy.time.Time())
        return T_from_tfmsg(tf_base_cam)

    def _update_stability(self, T_map_base: np.ndarray) -> bool:
        """ArUco 코드의 stability gating과 동일한 로직"""
        if self._prev_map_base is None:
            self._prev_map_base = T_map_base
            self._stable_count = 0
            return False

        dx = float(T_map_base[0, 3] - self._prev_map_base[0, 3])
        dy = float(T_map_base[1, 3] - self._prev_map_base[1, 3])
        dxy = math.hypot(dx, dy)

        yaw_now = yaw_from_R(T_map_base[:3, :3])
        yaw_prev = yaw_from_R(self._prev_map_base[:3, :3])
        dyaw = abs(wrap_pi(yaw_now - yaw_prev))

        if (dxy <= self.stable_xy_m) and (math.degrees(dyaw) <= self.stable_yaw_deg):
            self._stable_count += 1
        else:
            self._stable_count = 0

        self._prev_map_base = T_map_base
        return self._stable_count >= self.need_stable_ticks

    def _is_best_fresh(self) -> bool:
        return (time.time() - self._last_best_time) <= self.target_fresh_sec

    def _tick(self):
        t0 = time.time()

        # 1) 프레임 수신
        frame_rgb = self.cam.get_latest_rgb()
        if frame_rgb is None:
            return

        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # 2) YOLO -> best teddy
        teddy_dets, best = extract_teddy_xyz_from_yolo(
            yolo_model=self.model,
            frame_bgr=frame_bgr,
            K=self.K,
            teddy_height_m=self.teddy_height_m,
            min_bbox_h_px=self.min_bbox_h_px,
            conf=self.conf,
            iou=self.iou,
            max_dets=self.max_dets,
            scale_s=self.scale_s,
        )

        if best is not None and best.get("xyz_cam_m") is not None:
            self._last_best_det = best
            self._last_best_time = time.time()

        # 3) 디버그 표시 (선택)
        if self.show_debug_window:
            if best and best.get("bbox"):
                x1, y1, x2, y2 = map(int, best["bbox"])
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if best.get("xyz_cam_m"):
                    X, Y, Z = best["xyz_cam_m"]
                    cv2.putText(
                        frame_bgr,
                        f"Z={Z:.2f}m X={X:.2f} Y={Y:.2f}",
                        (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
            cv2.putText(
                frame_bgr,
                f"teddy={len(teddy_dets)} stable={self._stable_count}/{self.need_stable_ticks}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("tcp_yolo_teddy", frame_bgr)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                rclpy.shutdown()
                return

        # 4) 최신 best가 없으면 종료(이번 tick은 변환 불가)
        if not self._is_best_fresh():
            self._stable_count = 0
            return
        if self._last_best_det is None or self._last_best_det.get("xyz_cam_m") is None:
            self._stable_count = 0
            return

        # 5) TF 준비 + stability gating (ArUco 동일)
        try:
            T_map_base = self._lookup_T_map_base()
            T_base_cam = self._lookup_T_base_cam()
        except TransformException as e:
            self._stable_count = 0
            self.get_logger().warn(f"TF not ready: {e}")
            return

        if not self._update_stability(T_map_base):
            return

        # 6) ✅ 핵심: map 좌표로 변환 (ArUco 스타일 T 합성)
        X, Y, Z = self._last_best_det["xyz_cam_m"]
        T_cam_obj = T_from_xyz(X, Y, Z)
        T_map_obj = T_map_base @ T_base_cam @ T_cam_obj

        mx, my, mz = float(T_map_obj[0, 3]), float(T_map_obj[1, 3]), float(T_map_obj[2, 3])
        rx, ry = float(T_map_base[0, 3]), float(T_map_base[1, 3])
        dist = math.hypot(mx - rx, my - ry)

        # 7) publish JSON
        self._seq += 1
        out = {
            "seq": self._seq,
            "class_name": self._last_best_det.get("class_name"),
            "conf": float(self._last_best_det.get("conf", 0.0)),
            "bbox": self._last_best_det.get("bbox"),
            "xyz_cam_m": [float(X), float(Y), float(Z)],
            "xyz_map_m": [mx, my, mz],
            "robot_xy_map_m": [rx, ry],
            "dist_m": float(dist),
            "frames": {"map": self.map_frame, "base": self.base_frame, "camera": self.camera_frame},
            "meta": {
                "teddy_height_m": float(self.teddy_height_m),
                "min_bbox_h_px": int(self.min_bbox_h_px),
                "scale_s": float(self.scale_s),
                "stable_count": int(self._stable_count),
                "need_stable_ticks": int(self.need_stable_ticks),
            },
            "ts": time.time(),
        }
        self.pub.publish(String(data=json.dumps(out, ensure_ascii=False)))

        # 8) tick period 유지(너의 기존 스타일)
        dt = time.time() - t0
        sleep_s = self.tick_period - dt
        if sleep_s > 0:
            time.sleep(sleep_s)


def main():
    rclpy.init()
    node = YoloMapProjector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()


### 실행 예시
# 핑키1
# $VIRTUAL_ENV/bin/python3 ~/pinky_pro/install/pinky_vision/lib/pinky_vision/map_to_obj \
#   --ros-args \
#   -p scale_s:=0.21 \

# 로봇명 넣은 버전
# $VIRTUAL_ENV/bin/python3 ~/pinky_pro/install/pinky_vision/lib/pinky_vision/map_to_obj \
#    --ros-args   -p scale_s:=0.21   -p robot_name:=pinky1   -p tcp_host:=192.168.0.45   -p tcp_port:=9001


# 핑키2
# $VIRTUAL_ENV/bin/python3 ~/pinky_pro/install/pinky_vision/lib/pinky_vision/map_to_obj \
#   --ros-args \
#   -p scale_s:=0.21 \
#   -p robot_name:=pinky2 \
#   -p tcp_host:=192.168.0.42 \
#   -p tcp_port:=9001


# ros2 run pinky_vision vision_avoid_yolo --ros-args -p scale_s:=0.21

# ros2 run pinky_vision vision_avoid_yolo --ros-args \
#   -p base_frame:=base_footprint \
#   -p camera_frame:=front_camera_link \
#   -p tcp_host:=192.168.0.45 \
#   -p tcp_port:=9001 \
#   -p yolo_model_path:=yolo_model/yolo11m.pt \
#   -p out_topic:=/yolo_target_map \
#   -p tick_hz:=10.0 \
#   -p teddy_height_m:=0.35 \
#   -p scale_s:=0.21

### (보기)
# ros2 topic echo /yolo_target_map --once --full-length \
# | sed -n 's/^data: //p' \
# | sed "s/^'//; s/'$//" \
# | jq '{seq, class_name, conf, xyz_cam_m, xyz_map_m, robot_xy_map_m, dist_m}'


# 'teddybear' class_id(인덱스)는 모델의 names를 출력해서 확인
#     from ultralytics import YOLO

#     model = YOLO("yolo_model/best.pt")
#     print(model.names)  # dict 또는 list로 나옴

# (참고코드) static_transform_publisher.py
#     아루코 코드 스타일로 “YOLO xyz_cam_m → map 좌표” 변환하기
#     아루코 코드에서 이 부분이었지:
#         T_cam_aruco = T_from_posemsg(self._latest_pose)
#         T_map_aruco = T_map_base @ T_base_cam @ T_cam_aruco
#     이번엔 /aruco_pose 대신 YOLO xyz를 쓰니까:
#         T_cam_obj = I (회전 없음) + translation(x,y,z)
#         T_map_obj = T_map_base @ T_base_cam @ T_cam_obj