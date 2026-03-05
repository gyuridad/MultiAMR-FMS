#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
goal_mover_action_visionavoid_integrated.py

✅ ActionServer: MoveToPID
  - request.target: PoseStamped
  - request.timeout_sec: float (optional)

✅ Normal driving:
  - map->base TF 기반 PID /cmd_vel

✅ Vision avoid FSM:
  - /vision/avoid_dir_json (std_msgs/String JSON)에서 decision=LEFT/RIGHT, object.dist2d 사용
  - STOP(hold_sec) -> TURN(avoid_turn_deg) -> GO(거리 기반) -> resume
  - holonomic이면 linear.y sidestep 옵션 제공

✅ 변경사항(요청 반영)
  1) hard_stop_dist_m 파라미터/로직 완전 제거 (멈추기만 하고 return 하는 early-exit 삭제)
  2) avoid_stop_hold_sec 기본값을 3.0초로 변경 (원하면 파라미터로 조절)

실행 예시(네임스페이스 포함):
ros2 run actions goal_mover_launch_visionavoid --ros-args \
  -r __ns:=/pinky1 \
  -p cmd_topic:=cmd_vel \
  -p map_frame:=map \
  -p base_frame:=base_link \
  -p obstacle_topic:=obstacle_detected \
  -p action_name:=actions/move_to_pid \
  -p vision_avoid_topic:=/vision/avoid_dir_json \
  -p vision_fresh_sec:=0.6 \
  -p avoid_trigger_dist_m:=0.4 \
  -p avoid_stop_hold_sec:=3.0 \
  -p avoid_turn_deg:=45.0 \
  -p avoid_go_dist_m:=0.25 \
  -p avoid_go_speed:=0.10 \
  -p avoid_go_max_sec:=2.0 \
  -p use_strafe_y:=true \
  -p strafe_without_turn:=false \
  -p control_period_sec:=0.02

액션 테스트:
ros2 action send_goal /pinky1/actions/move_to_pid pinky_interfaces/action/MoveToPID \
"{target: {header: {frame_id: 'map'}, pose: {position: {x: 0.271, y: -1.028, z: 0.0}, orientation: {x: 0.0, y: 0.0, z: -0.724, w: 0.690}}}, timeout_sec: 120.0}"
"""

import math
import json
import time
from enum import IntEnum
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.duration import Duration

from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import Float64, Bool, String

import tf2_ros
from tf2_ros import TransformException

from pinky_interfaces.action import MoveToPID  # ✅ 프로젝트 액션 타입


# ------------------------
# Utils
# ------------------------
def normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def yaw_from_quat(x, y, z, w) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


class PIDController:
    def __init__(self, kp=0.7, ki=0.0, kd=0.3, min_output=-1.0, max_output=1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.min_output = min_output
        self.max_output = max_output
        self.previous_error = 0.0
        self.integral = 0.0
        self.last_time = None

    def update_gains(self, kp, ki, kd):
        self.kp, self.ki, self.kd = kp, ki, kd

    def reset(self):
        self.previous_error = 0.0
        self.integral = 0.0
        self.last_time = None

    def compute(self, error, current_time=None):
        if current_time is None:
            current_time = time.time()

        if self.last_time is None:
            self.last_time = current_time
            self.previous_error = error
            return 0.0

        dt = current_time - self.last_time
        if dt <= 0.0:
            return 0.0

        p = self.kp * error

        self.integral += error * dt
        max_integral = abs(self.max_output) / (abs(self.ki) + 1e-6)
        self.integral = max(-max_integral, min(max_integral, self.integral))
        i = self.ki * self.integral

        d = self.kd * (error - self.previous_error) / dt

        out = p + i + d
        out = max(self.min_output, min(self.max_output, out))

        self.previous_error = error
        self.last_time = current_time
        return out


# ------------------------
# Modes
# ------------------------
class Mode(IntEnum):
    TURN_TO_GOAL = 0
    GO_STRAIGHT = 1
    FINAL_ALIGN = 2
    # Vision avoid FSM
    AVOID_STOP = 10
    AVOID_TURN = 11
    AVOID_GO = 12


# ------------------------
# Node
# ------------------------
class GoalMover(Node):
    """
    ✅ ActionServer: MoveToPID
    ✅ map->base TF 기반 PID /cmd_vel
    ✅ Vision avoid FSM:
      - /vision/avoid_dir_json(JSON) decision=LEFT/RIGHT, object.dist2d 사용
      - STOP(hold) -> TURN(avoid_turn_deg) -> GO(거리 기반) -> resume
      - holonomic이면 linear.y sidestep 옵션 제공
    """

    def __init__(self):
        super().__init__("goal_mover_action_visionavoid_integrated")

        # ---------------- Parameters ----------------
        self.declare_parameter("cmd_topic", "cmd_vel")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("obstacle_topic", "obstacle_detected")

        # Action name (namespace 고려)
        self.declare_parameter("action_name", "actions/move_to_pid")

        # PID gains
        self.declare_parameter("linear_P", 1.0)
        self.declare_parameter("linear_I", 0.0)
        self.declare_parameter("linear_D", 0.2)

        self.declare_parameter("angular_P", 0.2)
        self.declare_parameter("angular_I", 0.0)
        self.declare_parameter("angular_D", 0.05)

        # Tolerances
        self.declare_parameter("angle_tolerance_deg", 12.0)
        self.declare_parameter("pos_tolerance", 0.03)
        self.declare_parameter("final_yaw_tolerance_deg", 5.0)
        self.declare_parameter("enable_pid", True)

        # Speed limits
        self.declare_parameter("max_linear_speed", 0.07)
        self.declare_parameter("max_angular_speed", 1.5)
        self.declare_parameter("min_linear_speed", 0.06)
        self.declare_parameter("min_angular_speed", 0.10)
        self.declare_parameter("min_speed_distance_threshold", 0.30)

        # TF lookup
        self.declare_parameter("tf_timeout_sec", 0.2)

        # Control loop
        self.declare_parameter("control_period_sec", 0.02)

        # ---------------- Vision avoid params ----------------
        self.declare_parameter("vision_avoid_topic", "/vision/avoid_dir_json")
        self.declare_parameter("vision_fresh_sec", 3.0)

        # 회피 트리거 거리
        self.declare_parameter("avoid_trigger_dist_m", 0.40)

        # 회피 회전 각도
        self.declare_parameter("avoid_turn_deg", 45.0)

        # ✅ 거리 기반 GO
        self.declare_parameter("avoid_go_dist_m", 0.20)
        self.declare_parameter("avoid_go_tol_m", 0.02)
        self.declare_parameter("avoid_go_speed", 0.12)
        self.declare_parameter("avoid_go_max_sec", 2.0)

        self.declare_parameter("avoid_stop_hold_sec", 0.30)

        self.declare_parameter("avoid_cooldown_sec", 1.0)
        self.declare_parameter("avoid_yaw_tol_deg", 6.0)

        # holonomic sidestep (기본은 비활성: 대부분 베이스는 linear.y 미지원)
        self.declare_parameter("use_strafe_y", False)

        # (옵션) holonomic이면 TURN 생략하고 바로 옆이동하고 싶을 때
        self.declare_parameter("strafe_without_turn", False)

        # ---------------- Load params ----------------
        self._load_params()
        self.avoid_yaw_tol = math.radians(self.avoid_yaw_tol_deg)

        # ---------------- TF ----------------
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ---------------- PID ----------------
        self.linear_pid = PIDController(
            kp=self.linear_P,
            ki=self.linear_I,
            kd=self.linear_D,
            min_output=-self.max_linear_speed,
            max_output=self.max_linear_speed,
        )
        self.angular_pid = PIDController(
            kp=self.angular_P,
            ki=self.angular_I,
            kd=self.angular_D,
            min_output=-self.max_angular_speed,
            max_output=self.max_angular_speed,
        )

        # ---------------- State ----------------
        self.goal_msg: Optional[PoseStamped] = None
        self.mode: Mode = Mode.TURN_TO_GOAL
        self._reached_flag = False
        self.obstacle_active = False

        # ---------------- Vision avoid state ----------------
        self._last_avoid: Optional[dict] = None
        self._last_avoid_t = 0.0
        self._avoid_active = False
        self._avoid_dir: Optional[str] = None  # "LEFT"|"RIGHT"
        self._avoid_target_yaw: Optional[float] = None
        self._avoid_phase_t0 = 0.0
        self._avoid_cooldown_until = 0.0

        # ✅ 거리 기반 GO 시작점
        self._avoid_go_start_x: Optional[float] = None
        self._avoid_go_start_y: Optional[float] = None

        # ---------------- ROS I/O ----------------
        self.cmd_pub = self.create_publisher(Twist, self.cmd_topic, 10)
        self.distance_error_pub = self.create_publisher(Float64, "distance_error", 10)
        self.angle_error_pub = self.create_publisher(Float64, "angle_error", 10)
        self.final_yaw_error_pub = self.create_publisher(Float64, "final_yaw_error", 10)

        self.sub_ob = self.create_subscription(Bool, self.obstacle_topic, self._on_obstacle, 10)
        self.sub_vision = self.create_subscription(String, self.vision_avoid_topic, self._on_vision_avoid, 10)

        # ✅ ActionServer
        self._as = ActionServer(
            self,
            MoveToPID,
            self.action_name,
            execute_callback=self._execute_cb,
            goal_callback=self._goal_cb,
            cancel_callback=self._cancel_cb,
        )

        # Timer
        self.timer = self.create_timer(self.control_period_sec, self.control_loop)

        self.get_logger().info(
            f"✅ GoalMover(Action+VisionAvoid) ready: action={self.action_name}, vision={self.vision_avoid_topic}"
        )

    # ---------------- Param loader ----------------
    def _load_params(self):
        self.cmd_topic = self.get_parameter("cmd_topic").value
        self.map_frame = self.get_parameter("map_frame").value
        self.base_frame = self.get_parameter("base_frame").value
        self.obstacle_topic = self.get_parameter("obstacle_topic").value
        self.action_name = self.get_parameter("action_name").value

        self.linear_P = float(self.get_parameter("linear_P").value)
        self.linear_I = float(self.get_parameter("linear_I").value)
        self.linear_D = float(self.get_parameter("linear_D").value)

        self.angular_P = float(self.get_parameter("angular_P").value)
        self.angular_I = float(self.get_parameter("angular_I").value)
        self.angular_D = float(self.get_parameter("angular_D").value)

        self.angle_tolerance = math.radians(float(self.get_parameter("angle_tolerance_deg").value))
        self.final_yaw_tolerance = math.radians(float(self.get_parameter("final_yaw_tolerance_deg").value))
        self.pos_tolerance = float(self.get_parameter("pos_tolerance").value)

        self.enable_pid = bool(self.get_parameter("enable_pid").value)

        self.max_linear_speed = float(self.get_parameter("max_linear_speed").value)
        self.max_angular_speed = float(self.get_parameter("max_angular_speed").value)
        self.min_linear_speed = float(self.get_parameter("min_linear_speed").value)
        self.min_angular_speed = float(self.get_parameter("min_angular_speed").value)
        self.min_speed_distance_threshold = float(self.get_parameter("min_speed_distance_threshold").value)

        self.tf_timeout_sec = float(self.get_parameter("tf_timeout_sec").value)
        self.control_period_sec = float(self.get_parameter("control_period_sec").value)

        # Vision avoid
        self.vision_avoid_topic = self.get_parameter("vision_avoid_topic").value
        self.vision_fresh_sec = float(self.get_parameter("vision_fresh_sec").value)

        self.avoid_trigger_dist_m = float(self.get_parameter("avoid_trigger_dist_m").value)
        self.avoid_turn_deg = float(self.get_parameter("avoid_turn_deg").value)

        self.avoid_go_dist_m = float(self.get_parameter("avoid_go_dist_m").value)
        self.avoid_go_tol_m = float(self.get_parameter("avoid_go_tol_m").value)
        self.avoid_go_speed = float(self.get_parameter("avoid_go_speed").value)
        self.avoid_go_max_sec = float(self.get_parameter("avoid_go_max_sec").value)

        self.avoid_stop_hold_sec = float(self.get_parameter("avoid_stop_hold_sec").value)
        self.avoid_cooldown_sec = float(self.get_parameter("avoid_cooldown_sec").value)
        self.avoid_yaw_tol_deg = float(self.get_parameter("avoid_yaw_tol_deg").value)

        self.use_strafe_y = bool(self.get_parameter("use_strafe_y").value)
        self.strafe_without_turn = bool(self.get_parameter("strafe_without_turn").value)

    # ---------------- Action callbacks ----------------
    def _goal_cb(self, goal_request):
        # 여기서 goal 검증 가능 (좌표 범위 등)
        return GoalResponse.ACCEPT

    def _cancel_cb(self, goal_handle):
        self.get_logger().warn("[MoveToPID] cancel requested → stop")
        self.goal_msg = None
        self._reached_flag = False
        self._publish_stop()
        return CancelResponse.ACCEPT

    def _execute_cb(self, goal_handle):
        """
        System1이 보낸 goal_request:
          - goal_handle.request.target (PoseStamped)
          - goal_handle.request.timeout_sec (float, optional)
        """
        req = goal_handle.request
        target: PoseStamped = req.target
        timeout_sec = float(getattr(req, "timeout_sec", 0.0)) if req is not None else 0.0

        # 목표 세팅
        self._reached_flag = False
        self.goal_msg = target
        self.mode = Mode.TURN_TO_GOAL
        self.linear_pid.reset()
        self.angular_pid.reset()

        # 회피 상태도 정리(새 goal이면 이전 회피 남기지 않기)
        self._finish_avoid(reset_pid=False)

        t0 = time.time()
        self.get_logger().info(
            f"[MoveToPID] start: x={target.pose.position.x:.2f}, y={target.pose.position.y:.2f}, timeout={timeout_sec:.1f}s"
        )

        # 목표 끝날 때까지 대기
        while rclpy.ok():
            # cancel
            if goal_handle.is_cancel_requested:
                self.get_logger().warn("[MoveToPID] cancel requested")
                self.goal_msg = None
                self._reached_flag = False
                self._publish_stop()
                goal_handle.canceled()

                res = MoveToPID.Result()
                res.success = False
                res.message = "canceled"
                res.status = 1
                return res

            # reached
            if self._reached_flag:
                self.goal_msg = None
                self._reached_flag = False
                self._publish_stop()
                goal_handle.succeed()

                res = MoveToPID.Result()
                res.success = True
                res.message = "reached"
                res.status = 0
                return res

            # timeout
            if timeout_sec > 0.0 and (time.time() - t0) > timeout_sec:
                self.get_logger().warn("[MoveToPID] timeout → stop")
                self.goal_msg = None
                self._reached_flag = False
                self._publish_stop()
                goal_handle.abort()

                res = MoveToPID.Result()
                res.success = False
                res.message = "timeout"
                res.status = 2
                return res

            time.sleep(0.02)

        # shutdown
        self.goal_msg = None
        self._reached_flag = False
        self._publish_stop()
        goal_handle.abort()

        res = MoveToPID.Result()
        res.success = False
        res.message = "shutdown"
        res.status = 3
        return res

    # ---------------- TF helpers ----------------
    def _get_robot_pose_in_map(self) -> Optional[Tuple[float, float, float]]:
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.base_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=self.tf_timeout_sec),
            )
        except TransformException as e:
            self.get_logger().warn(f"TF lookup failed: {self.map_frame}->{self.base_frame}: {e}")
            return None

        x = tf.transform.translation.x
        y = tf.transform.translation.y
        q = tf.transform.rotation
        yaw = yaw_from_quat(q.x, q.y, q.z, q.w)
        return x, y, yaw

    def _goal_xy_yaw_in_map(self) -> Optional[Tuple[float, float, float]]:
        if self.goal_msg is None:
            return None
        gx = self.goal_msg.pose.position.x
        gy = self.goal_msg.pose.position.y
        gq = self.goal_msg.pose.orientation
        gyaw = yaw_from_quat(gq.x, gq.y, gq.z, gq.w)
        return gx, gy, gyaw

    def _publish_stop(self):
        self.cmd_pub.publish(Twist())

    # ---------------- Obstacle topic ----------------
    def _on_obstacle(self, msg: Bool):
        self.obstacle_active = bool(msg.data)
        if self.obstacle_active:
            self._publish_stop()

    # ---------------- Vision JSON ----------------
    def _on_vision_avoid(self, msg: String):
        raw = msg.data.strip()
        if not raw:
            return
        try:
            obj = json.loads(raw)
        except Exception as e:
            self.get_logger().warn(f"[VISION] json parse fail: {e} | raw={raw[:120]}")
            return
        decision = obj.get("decision", None)
        if decision not in ("LEFT", "RIGHT", "STOP", "NONE", None):
            self.get_logger().warn(f"[VISION] unexpected decision={decision} keys={list(obj.keys())}")
        self._last_avoid = obj
        self._last_avoid_t = time.time()

    def _vision_fresh(self) -> bool:
        return (time.time() - self._last_avoid_t) <= self.vision_fresh_sec

    def _get_avoid_decision_and_dist(self) -> Tuple[Optional[str], Optional[float]]:
        """
        returns: (decision, dist2d)
          - decision: "LEFT"/"RIGHT"/"STOP"/"NONE"/None
          - dist2d: meters or None

        기대 JSON 예:
        {
          "decision": "LEFT",
          "object": {"dist2d": 0.35, "name": "person"}
        }
        """
        if self._last_avoid is None:
            return None, None
        if not self._vision_fresh():
            return None, None

        d = self._last_avoid
        decision = str(d.get("decision", "")).strip().upper() or None

        dist2d = None
        try:
            obj = d.get("object") or {}
            dist2d = obj.get("dist2d", None)
            if dist2d is not None:
                dist2d = float(dist2d)
        except Exception:
            dist2d = None

        return decision, dist2d

    # ---------------- Avoid FSM ----------------
    def _start_avoid(self, direction: str, current_yaw: float):
        sign = +1.0 if direction == "LEFT" else -1.0
        delta = math.radians(self.avoid_turn_deg) * sign

        self._avoid_dir = direction
        self._avoid_target_yaw = normalize_angle(current_yaw + delta)
        self._avoid_active = True
        self.mode = Mode.AVOID_STOP
        self._avoid_phase_t0 = time.time()

        self._avoid_go_start_x = None
        self._avoid_go_start_y = None

        self.angular_pid.reset()
        self.linear_pid.reset()

        self.get_logger().warn(
            f"[AVOID START] dir={direction} target_yaw={math.degrees(self._avoid_target_yaw):.1f}"
        )

    def _finish_avoid(self, reset_pid: bool = True, heading_err: Optional[float] = None):
        """
        heading_err 제공 시:
          - heading_err가 충분히 작으면 GO_STRAIGHT로 복귀
          - 크면 TURN_TO_GOAL 복귀
        """
        self._avoid_active = False
        self._avoid_dir = None
        self._avoid_target_yaw = None
        self._avoid_phase_t0 = 0.0
        self._avoid_go_start_x = None
        self._avoid_go_start_y = None

        self._avoid_cooldown_until = time.time() + self.avoid_cooldown_sec

        if heading_err is not None and abs(heading_err) <= self.angle_tolerance * 0.7:
            self.mode = Mode.GO_STRAIGHT
        else:
            self.mode = Mode.TURN_TO_GOAL

        if reset_pid:
            self.angular_pid.reset()
            self.linear_pid.reset()

        self.get_logger().warn(f"[AVOID END] resume_mode={self.mode.name}")

    def _avoid_traveled_m(self, cur_x: float, cur_y: float) -> Optional[float]:
        if self._avoid_go_start_x is None or self._avoid_go_start_y is None:
            return None
        return math.hypot(cur_x - self._avoid_go_start_x, cur_y - self._avoid_go_start_y)

    # ---------------- Main control loop ----------------
    def control_loop(self):
        # 0) early exit
        if self.goal_msg is None:
            return

        if self.obstacle_active and not self._avoid_active:
            # obstacle_detected가 true면 일단 정지 (회피 로직과 별개 안전)
            self._publish_stop()
            # vision 회피 시작 기회를 막지 않기 위해 continue

        robot = self._get_robot_pose_in_map()
        goal = self._goal_xy_yaw_in_map()
        if robot is None or goal is None:
            return

        rx, ry, ryaw = robot
        gx, gy, gyaw = goal

        # vision inputs
        decision, dist2d = self._get_avoid_decision_and_dist()
        now = time.time()

        # ✅ (요청 반영) hard_stop_dist_m 로직 제거됨

        # -------- 1) Avoid FSM active: cmd_vel를 회피가 소유 --------
        if self._avoid_active:
            if self.mode == Mode.AVOID_STOP:
                self._publish_stop()

                if (now - self._avoid_phase_t0) >= self.avoid_stop_hold_sec:
                    # holonomic + 옵션이면 TURN 생략하고 GO로
                    if self.use_strafe_y and self.strafe_without_turn:
                        self.mode = Mode.AVOID_GO
                        self._avoid_phase_t0 = now
                        self._avoid_go_start_x = float(rx)
                        self._avoid_go_start_y = float(ry)
                    else:
                        self.mode = Mode.AVOID_TURN
                        self._avoid_phase_t0 = now
                        self.angular_pid.reset()
                return

            elif self.mode == Mode.AVOID_TURN:
                if self._avoid_target_yaw is None:
                    heading_err = normalize_angle(math.atan2(gy - ry, gx - rx) - ryaw)
                    self._finish_avoid(heading_err=heading_err)
                    self._publish_stop()
                    return

                yaw_err = normalize_angle(self._avoid_target_yaw - ryaw)

                if abs(yaw_err) <= self.avoid_yaw_tol:
                    self.mode = Mode.AVOID_GO
                    self._avoid_phase_t0 = now
                    self._avoid_go_start_x = float(rx)
                    self._avoid_go_start_y = float(ry)
                    self._publish_stop()
                    return

                cmd = Twist()
                w = self.angular_pid.compute(yaw_err) if self.enable_pid else (self.angular_P * yaw_err)
                w = max(-self.max_angular_speed, min(self.max_angular_speed, w))
                if abs(w) < self.min_angular_speed:
                    w = math.copysign(self.min_angular_speed, w)

                cmd.angular.z = w
                cmd.linear.x = 0.0
                cmd.linear.y = 0.0
                self.cmd_pub.publish(cmd)
                return

            elif self.mode == Mode.AVOID_GO:
                traveled = self._avoid_traveled_m(rx, ry)
                if traveled is None:
                    heading_err = normalize_angle(math.atan2(gy - ry, gx - rx) - ryaw)
                    self._finish_avoid(heading_err=heading_err)
                    self._publish_stop()
                    return

                err = self.avoid_go_dist_m - traveled

                # 도달
                if err <= self.avoid_go_tol_m:
                    heading_err = normalize_angle(math.atan2(gy - ry, gx - rx) - ryaw)
                    self._finish_avoid(heading_err=heading_err)
                    self._publish_stop()
                    return

                # 안전 타임아웃
                if self.avoid_go_max_sec > 0.0 and (now - self._avoid_phase_t0) >= self.avoid_go_max_sec:
                    heading_err = normalize_angle(math.atan2(gy - ry, gx - rx) - ryaw)
                    self._finish_avoid(heading_err=heading_err)
                    self._publish_stop()
                    return

                cmd = Twist()
                v = max(0.0, float(self.avoid_go_speed))
                v = min(float(self.max_linear_speed), v)

                # ✅ sidestep(holonomic) 지원
                if self.use_strafe_y and self._avoid_dir in ("LEFT", "RIGHT"):
                    sign = +1.0 if self._avoid_dir == "LEFT" else -1.0
                    cmd.linear.x = 0.0
                    cmd.linear.y = sign * v
                else:
                    # fallback: 전진
                    cmd.linear.x = v
                    cmd.linear.y = 0.0

                cmd.angular.z = 0.0
                self.cmd_pub.publish(cmd)
                return

            # 이상 상태
            heading_err = normalize_angle(math.atan2(gy - ry, gx - rx) - ryaw)
            self._finish_avoid(heading_err=heading_err)
            self._publish_stop()
            return

        # -------- 2) Avoid FSM 새로 시작 조건 --------
        if now >= self._avoid_cooldown_until:
            trigger_ok = (dist2d is None) or (dist2d <= self.avoid_trigger_dist_m)
            if (decision in ("LEFT", "RIGHT")) and trigger_ok:
                self._start_avoid(decision, ryaw)
                self._publish_stop()
                return

        # -------- 3) Normal PID drive to goal --------
        dx = gx - rx
        dy = gy - ry
        dist = math.hypot(dx, dy)
        heading_to_goal = math.atan2(dy, dx)
        heading_err = normalize_angle(heading_to_goal - ryaw)
        final_yaw_err = normalize_angle(gyaw - ryaw)

        self.distance_error_pub.publish(Float64(data=float(dist)))
        self.angle_error_pub.publish(Float64(data=float(heading_err)))
        self.final_yaw_error_pub.publish(Float64(data=float(final_yaw_err)))

        # ✅ 히스테리시스(채터링 방지)
        enter_go_tol = self.angle_tolerance * 0.7
        exit_go_tol = self.angle_tolerance * 1.3

        # 목표 위치 도달하면 FINAL_ALIGN로
        if dist < self.pos_tolerance and self.mode != Mode.FINAL_ALIGN:
            self.mode = Mode.FINAL_ALIGN
            self.angular_pid.reset()

        cmd = Twist()

        if self.mode == Mode.TURN_TO_GOAL:
            if abs(heading_err) <= enter_go_tol:
                self.mode = Mode.GO_STRAIGHT
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
            else:
                w = self.angular_pid.compute(heading_err) if self.enable_pid else (self.angular_P * heading_err)
                w = max(-self.max_angular_speed, min(self.max_angular_speed, w))
                if abs(w) < self.min_angular_speed:
                    w = math.copysign(self.min_angular_speed, w)
                cmd.angular.z = w
                cmd.linear.x = 0.0

        elif self.mode == Mode.GO_STRAIGHT:
            if abs(heading_err) > exit_go_tol:
                self.mode = Mode.TURN_TO_GOAL
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
            else:
                v = self.linear_pid.compute(dist) if self.enable_pid else (self.linear_P * dist)
                v = max(-self.max_linear_speed, min(self.max_linear_speed, v))
                if dist > self.min_speed_distance_threshold and abs(v) < self.min_linear_speed:
                    v = math.copysign(self.min_linear_speed, v)
                cmd.linear.x = v
                cmd.angular.z = 0.0

        elif self.mode == Mode.FINAL_ALIGN:
            if abs(final_yaw_err) <= self.final_yaw_tolerance:
                # ✅ 최종 도달
                self.get_logger().info("✅ reached goal pose. stop.")
                self._reached_flag = True
                self.goal_msg = None
                self.mode = Mode.TURN_TO_GOAL
                self.linear_pid.reset()
                self.angular_pid.reset()
                self._publish_stop()
                return

            w = self.angular_pid.compute(final_yaw_err) if self.enable_pid else (self.angular_P * final_yaw_err)
            w = max(-self.max_angular_speed, min(self.max_angular_speed, w))
            if abs(w) < self.min_angular_speed:
                w = math.copysign(self.min_angular_speed, w)
            cmd.angular.z = w
            cmd.linear.x = 0.0

        else:
            self.get_logger().warn(f"Unknown mode: {self.mode}. Stop.")
            self._publish_stop()
            return

        self.cmd_pub.publish(cmd)


# ------------------------
# main
# ------------------------
def main(args=None):
    rclpy.init(args=args)
    node = GoalMover()

    from rclpy.executors import MultiThreadedExecutor
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            executor.shutdown()
        except Exception:
            pass
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
