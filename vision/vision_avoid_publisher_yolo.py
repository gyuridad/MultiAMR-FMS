#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
import time
from typing import Any, Dict, List, Optional, Tuple

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import tf2_ros


def yaw_from_quat(x, y, z, w) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)

def dist2(ax: float, ay: float, bx: float, by: float) -> float:
    dx = ax - bx
    dy = ay - by
    return dx * dx + dy * dy

def min_dist_to_walls(x: float, y: float, walls_xy: List[List[float]]) -> float:
    """walls_xy: [[wx,wy], ...] 에서 (x,y)까지 최소거리(m)."""
    if not walls_xy:
        return float("inf")
    best = float("inf")
    for w in walls_xy:
        wx, wy = float(w[0]), float(w[1])
        d = math.hypot(x - wx, y - wy)
        if d < best:
            best = d
    return best

# 이 함수의 역할 : “yaw 방향으로 길이 1짜리 화살표를 만들어라”
def unit_from_yaw(yaw: float) -> Tuple[float, float]:
    return (math.cos(yaw), math.sin(yaw))
        # 핵심 개념: “각도 → 원 위의 점”
        # 반지름이 1인 원을 생각해보자.
        # 어떤 방향으로 길이 1만큼 나간다고 생각해봐.
        # 그 방향이 yaw = θ 라면:
        #     x방향으로는 얼마나 가야 할까?
        #     y방향으로는 얼마나 가야 할까?
        # 숫자 예시:
        #     yaw = 0 rad (동쪽, +x 방향)
        #         cos(0) = 1
        #         sin(0) = 0
        #         → (1, 0)
        #             ✔ x방향으로만 1
        #             ✔ y는 0
        #             ✔ 오른쪽을 본다는 뜻
        #     yaw = 90도 = π/2 rad (북쪽, +y 방향)
        #         cos(π/2) = 0
        #         sin(π/2) = 1
        #         → (0, 1)
        #             ✔ 위쪽을 보는 방향

        # 벡터 (vx, vy)는 “화살표”라고 생각하면 돼.
        #     오른쪽을 가리키는 화살표: (1, 0)
        #     위를 가리키는 화살표: (0, 1)
        #     왼쪽을 가리키는 화살표: (-1, 0)
        #     아래를 가리키는 화살표: (0, -1)


def rot90_left(vx: float, vy: float) -> Tuple[float, float]:
    return (-vy, vx)

def rot90_right(vx: float, vy: float) -> Tuple[float, float]:
    return (vy, -vx)


class AvoidDecision(Node):
    """
    입력:
      - /yolo_target_map (String JSON): xyz_map_m, robot_xy_map_m, dist_m 등
      - /debug/walls_xy_json (String JSON): walls_xy, local.robot_pose_map 등

    출력:
      - /vision/avoid_dir_json (String JSON):
          {
            "ok": true,
            "decision": "LEFT|RIGHT|KEEP|STOP",
            "robot": {"x":..,"y":..,"yaw":..},
            "object": {"x":..,"y":..,"dist2d":..},
            "scores": {...},
            "chosen": {...},
            "ts": ...
          }
    """

    def __init__(self):
        super().__init__("avoid_decision")

        # ---- params ----
        self.declare_parameter("yolo_topic", "yolo_target_map")     # "yolo_target_map" / 런치실행안하고 핑키1 테스트시 "/pinky1/yolo_target_map"
        self.declare_parameter("walls_topic", "debug/walls_xy_json")    # "debug/walls_xy_json" / 런치실행안하고 핑키1 테스트시 "/pinky1/debug/walls_xy_json"
        self.declare_parameter("out_topic", "vision/avoid_dir_json")    # "vision/avoid_dir_json" / 런치실행안하고 핑키1 테스트시 "/pinky1/vision/avoid_dir_json"

        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_link")

        # decision parameters
        self.declare_parameter("trigger_dist_m", 0.8)
        self.declare_parameter("step_m", 0.20)
        self.declare_parameter("wall_clear_min_m", 0.25)
        self.declare_parameter("fresh_sec", 0.6)

        # (선택) 좌/우 중 “객체 반대쪽”을 아주 약하게 선호
        #  - 0.0이면 미사용
        #  - 0.1~0.4 추천
        self.declare_parameter("prefer_away_from_object_gain", 0.15)

        self.yolo_topic = str(self.get_parameter("yolo_topic").value)
        self.walls_topic = str(self.get_parameter("walls_topic").value)
        self.out_topic = str(self.get_parameter("out_topic").value)

        self.map_frame = str(self.get_parameter("map_frame").value)
        self.base_frame = str(self.get_parameter("base_frame").value)

        self.trigger_dist_m = float(self.get_parameter("trigger_dist_m").value)
        self.step_m = float(self.get_parameter("step_m").value)
        self.wall_clear_min_m = float(self.get_parameter("wall_clear_min_m").value)
        self.fresh_sec = float(self.get_parameter("fresh_sec").value)
        self.prefer_away_gain = float(self.get_parameter("prefer_away_from_object_gain").value)

        # ---- TF ----
        self.tf_buffer = tf2_ros.Buffer(cache_time=rclpy.duration.Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ---- state cache ----
        self._last_yolo: Optional[Dict[str, Any]] = None
        self._last_yolo_t = 0.0
        self._last_walls: Optional[Dict[str, Any]] = None
        self._last_walls_t = 0.0

        # ---- subs/pubs ----
        self.sub_yolo = self.create_subscription(String, self.yolo_topic, self._on_yolo, 10)
        self.sub_walls = self.create_subscription(String, self.walls_topic, self._on_walls, 10)
        self.pub = self.create_publisher(String, self.out_topic, 10)

        self.timer = self.create_timer(0.1, self._tick)

        self.get_logger().info(
            f"[AvoidDecision-LR] yolo={self.yolo_topic}, walls={self.walls_topic}, out={self.out_topic}\n"
            f"  trigger_dist_m={self.trigger_dist_m}, step_m={self.step_m}, wall_clear_min_m={self.wall_clear_min_m}, prefer_away_gain={self.prefer_away_gain}\n"
            f"  frames: map={self.map_frame}, base={self.base_frame}"
        )

    def _on_yolo(self, msg: String):
        raw = msg.data.strip()
        if not raw:
            return
        try:
            obj = json.loads(raw)
        except Exception as e:
            self.get_logger().warn(f"[yolo] invalid JSON: {e}")
            return
        self._last_yolo = obj
        self._last_yolo_t = time.time()

    def _on_walls(self, msg: String):
        raw = msg.data.strip()
        if not raw:
            return
        try:
            obj = json.loads(raw)
        except Exception as e:
            self.get_logger().warn(f"[walls] invalid JSON: {e}")
            return
        self._last_walls = obj
        self._last_walls_t = time.time()

    # “마지막 메시지 받은 뒤 몇 초 지났는가?” 아직 유효한지(신선한지) 검사하는 타임아웃 체크
    def _fresh(self, t: float) -> bool:
        return (time.time() - t) <= self.fresh_sec

    def _lookup_robot_yaw_map(self) -> Optional[float]:
        try:
            tf = self.tf_buffer.lookup_transform(self.map_frame, self.base_frame, rclpy.time.Time())
            q = tf.transform.rotation
            return yaw_from_quat(q.x, q.y, q.z, q.w)
        except Exception:
            return None

    def _tick(self):
        """
        객체가 멀면: walls 없어도 KEEP
        객체가 가까우면: walls ok 아니면 STOP
        robot_xy는 가능하면 yolo에서, 없으면 walls에서, 그마저 없으면 return
        yaw는 walls에 있으면 쓰고, 없으면 TF로 보완
        """
        # 1) yolo만 먼저 확인 (객체 유무/거리 판단을 위해)
        if self._last_yolo is None or not self._fresh(self._last_yolo_t):
            return
        
        yolo = self._last_yolo

        # --- object map ---
        xyz_map = yolo.get("xyz_map_m")
        if not (isinstance(xyz_map, list) and len(xyz_map) >= 2):
            return
        ox, oy = float(xyz_map[0]), float(xyz_map[1])

        # --- robot map xy (우선 yolo에서) ---
        rx = ry = None
        robot_xy = yolo.get("robot_xy_map_m")
        if isinstance(robot_xy, list) and len(robot_xy) >= 2:
            rx, ry = float(robot_xy[0]), float(robot_xy[1])

        # yolo에 robot_xy가 없으면, walls에서 보충해야 함
        walls_msg = None
        if rx is None or ry is None:
            if self._last_walls is None or not self._fresh(self._last_walls_t):
                return
            walls_msg = self._last_walls

            rp = (walls_msg.get("local") or {}).get("robot_pose_map") or {}
            rx = float(rp.get("x", 0.0))
            ry = float(rp.get("y", 0.0))

        # 2) 객체 거리 계산
        d_obj = math.hypot(ox - rx, oy - ry)

        # 3) 객체가 멀면: 벽정보 상관없이 KEEP
        if d_obj > self.trigger_dist_m:
            # yaw는 walls 없어도 TF로 구할 수 있으면 넣고, 아니면 생략/0 처리 가능
            ryaw = self._lookup_robot_yaw_map()
            payload = {
                "ok": True,
                "decision": "KEEP",
                "robot": {"x": rx, "y": ry, "yaw": float(ryaw) if ryaw is not None else None},
                "object": {"x": ox, "y": oy, "dist2d": d_obj},
                "ts": time.time(),
                "reason": "object_far",
            }
            self.pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))
            return

        # 4) 여기부터는 “회피 판단” 구간 -> walls가 반드시 필요
        if walls_msg is None:
            if self._last_walls is None or not self._fresh(self._last_walls_t):
                payload = {
                    "ok": False,
                    "decision": "STOP",
                    "reason": "walls_not_fresh",      # “벽 메시지가 아예 없거나 너무 오래됨(타임아웃)”
                    "ts": time.time(),
                }
                self.pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))
                return
            walls_msg = self._last_walls

        if not walls_msg.get("ok", False):
            # 정상일 때: "ok": true
            # TF가 안 잡혔을 때 같은 오류: "ok": false, "reason":"tf_not_ready"
            payload = {
                "ok": False,
                "decision": "STOP",
                "reason": "walls_not_ok",         # “벽 메시지는 최신으로 받았지만, 내용이 실패 상태(ok=false)”
                "walls_reason": walls_msg.get("reason"),
                "ts": time.time(),
            }
            self.pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))
            return

        # --- robot yaw ---
        ryaw = None
        rp = (walls_msg.get("local") or {}).get("robot_pose_map") or {}
        if "yaw" in rp:
            try:
                ryaw = float(rp["yaw"])
            except Exception:
                ryaw = None
        if ryaw is None:
            ryaw = self._lookup_robot_yaw_map()
        if ryaw is None:
            payload = {
                "ok": False,
                "decision": "STOP",
                "reason": "robot_yaw_not_ready",
                "ts": time.time(),
            }
            self.pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))
            return

        # --- walls list ---
        walls_xy = walls_msg.get("walls_xy") or []
        if not isinstance(walls_xy, list):
            walls_xy = []

        # ✅ LEFT/RIGHT만 후보로 사용
        fx, fy = unit_from_yaw(ryaw)
        lx, ly = rot90_left(fx, fy)
        rxv, ryv = rot90_right(fx, fy)

        candidates = {
            "LEFT":  (lx, ly),
            "RIGHT": (rxv, ryv),
        }

        scores: Dict[str, Dict[str, float]] = {}
        best_dir = "STOP"
        best_score = -1e18
        chosen_detail: Dict[str, float] = {}

        # 로봇->객체 벡터(좌/우 판단 보조)
        v_ox = ox - rx
        v_oy = oy - ry

        # (선택) 객체가 왼쪽에 있으면 오른쪽이 약간 유리, 오른쪽에 있으면 왼쪽이 약간 유리
        # cross = forward x object (사실 forward가 필요하지만 여기서는 좌/우 벡터로 간단히 판단)
        # 더 간단히: 객체가 LEFT 벡터 쪽(+면 왼쪽)에 있으면 RIGHT 보너스
        # dot_left = (v·left)
        dot_left = v_ox * lx + v_oy * ly
            # v_ox, v_oy : 로봇 → 객체 벡터  (객체가 있는 방향)
            # lx, ly : 로봇의 왼쪽 방향 단위벡터  (로봇의 왼쪽 방향)
            # dot_left는 내적의 의미를 내포  (“객체 방향이 왼쪽 방향과 얼마나 같은가?”)
            #     두 벡터 A와 B의 내적 A·B는:
            #         B 방향으로 A가 얼마나 “같은 방향 성분”을 가지는지를 숫자로 나타낸 것

        for name, (vx, vy) in candidates.items():
            cx = rx + vx * self.step_m    # step_m : 옆으로 피할 거리(예: 0.2m)
            cy = ry + vy * self.step_m
                # LEFT로 0.2m 옮겼으면 어디? → (cx,cy)
                # RIGHT로 0.2m 옮겼으면 어디? → (cx,cy)
                #     를 각각 계산하는 거야.
                # 로봇이 (2.0, 2.0)에 있고 step_m=0.2라고 하자.
                #     LEFT 방향 벡터가 (0, +1)이면
                #     → LEFT 후보 위치: (2.0, 2.2)
                #     RIGHT 방향 벡터가 (0, -1)이면
                #     → RIGHT 후보 위치: (2.0, 1.8)

            # 후보 위치가 벽에 얼마나 가까운지 계산
            # “그쪽으로 피했을 때 벽과의 여유거리(안전거리)가 몇 m냐?”
            clearance = min_dist_to_walls(cx, cy, walls_xy)

            # 후보 위치에서 객체와의 거리를 계산
            # “그쪽으로 피한 뒤, 로봇과 객체 사이 거리가 얼마나 되나?”
            obj_after = math.hypot(ox - cx, oy - cy)
            gain = obj_after - d_obj  # +면 객체에서 멀어짐
                # d_obj : 현재 로봇-객체 거리
                # obj_after : 피한 뒤 로봇-객체 거리

            # 벽 너무 가까우면 강한 패널티
            penalty = 0.0
            if clearance < self.wall_clear_min_m:
                penalty -= 1000.0 * (self.wall_clear_min_m - clearance)

            away_bonus = 0.0
            if self.prefer_away_gain > 0.0:
                if dot_left > 0.0 and name == "RIGHT":
                    away_bonus += self.prefer_away_gain
                elif dot_left < 0.0 and name == "LEFT":
                    away_bonus += self.prefer_away_gain

            # 점수: 좌우만이면 단순화해도 충분
            # (벽 clearance가 제일 중요 + 객체에서 멀어지는 gain 조금 + away_bonus)
            score = (3.0 * clearance) + (1.5 * gain) + away_bonus + penalty

            scores[name] = {
                "cx": cx, "cy": cy,
                "clearance_m": clearance,
                "obj_gain_m": gain,
                "away_bonus": away_bonus,
                "score": score,
            }

            if score > best_score:
                best_score = score
                best_dir = name
                chosen_detail = scores[name]

        # 둘 다 위험하면 STOP (둘다 clearance가 너무 낮거나 패널티가 큰 경우)
        cL = scores.get("LEFT", {}).get("clearance_m", 0.0)
        cR = scores.get("RIGHT", {}).get("clearance_m", 0.0)
        if max(cL, cR) < max(0.05, self.wall_clear_min_m * 0.5):
            best_dir = "STOP"

        out = {
            "ok": True,
            "decision": best_dir,  # ✅ LEFT/RIGHT/STOP/KEEP 만
            "robot": {"x": rx, "y": ry, "yaw": ryaw},
            "object": {"x": ox, "y": oy, "dist2d": d_obj},
            "params": {
                "trigger_dist_m": self.trigger_dist_m,
                "step_m": self.step_m,
                "wall_clear_min_m": self.wall_clear_min_m,
                "prefer_away_from_object_gain": self.prefer_away_gain,
            },
            "scores": scores,
            "chosen": {"dir": best_dir, **chosen_detail},
            "ts": time.time(),
        }
        self.pub.publish(String(data=json.dumps(out, ensure_ascii=False)))


def main():
    rclpy.init()
    node = AvoidDecision()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()



### 실행 (런치 실행 안하고 트래픽매니저와 오케스트레이터 사용시 로봇명 넣어야함)
# $VIRTUAL_ENV/bin/python3 ~/pinky_pro/install/pinky_vision/lib/pinky_vision/avoid_publisher \
#   --ros-args \
#   -p trigger_dist_m:=0.8 \
#   -p step_m:=0.35 \
#   -p wall_clear_min_m:=0.25 \
#   -p prefer_away_from_object_gain:=0.15

### 로봇명 추가 버전
# $VIRTUAL_ENV/bin/python3 ~/pinky_pro/install/pinky_vision/lib/pinky_vision/avoid_publisher \
#   --ros-args \
#   -r __ns:=/pinky1 \
#   -p trigger_dist_m:=0.8 \
#   -p step_m:=0.35 \
#   -p wall_clear_min_m:=0.25 \
#   -p prefer_away_from_object_gain:=0.15

### 실행 조건
# (0) 핑키로봇 터미널에서 이미지 보내기
# ros2 run sensors image_socket
# (1) 로봇 반경 맵상에서의 벽체 거리 발행
# ros2 run pinky_vision map_wall_xy_near_robot
# (2) 맵상에서 객체 위치 발행
# $VIRTUAL_ENV/bin/python3 ~/pinky_pro/install/pinky_vision/lib/pinky_vision/map_to_obj   --ros-args -p scale_s:=0.21

### 보기
# ros2 topic echo /vision/avoid_dir_json --once --full-length \
# | sed -n 's/^data: //p' \
# | sed "s/^'//; s/'$//" \
# | jq '{decision, robot, object, chosen}'

# (근거 보기)
# ros2 topic echo /vision/avoid_dir_json --once --full-length | sed -n 's/^data: //p' | sed "s/^'//; s/'$//" | jq '.scores'
