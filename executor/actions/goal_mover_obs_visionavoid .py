#!/usr/bin/env python3
import sys
import math
import json
import time
from enum import IntEnum
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import Float64, String
import tf2_ros
from tf2_ros import TransformException


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


def quat_from_yaw(yaw: float):
    half = yaw * 0.5
    return (0.0, 0.0, math.sin(half), math.cos(half))


class PIDController:
    def __init__(self, kp=1.0, ki=0.0, kd=0.0, min_output=-1.0, max_output=1.0):
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


class Mode(IntEnum):
    TURN_TO_GOAL = 0
    GO_STRAIGHT = 1
    FINAL_ALIGN = 2

    # ✅ Vision-based avoid FSM
    AVOID_STOP = 10
    AVOID_TURN = 11
    AVOID_GO = 12


class GoalMover(Node):
    """
    - TF(map->base_link)로 현재 pose를 읽어서 /cmd_vel PID 제어
    - goal_callback(x, y, yaw_deg)로 목표를 직접 세팅
    - ✅ Vision JSON(/vision/avoid_dir_json) 기반 회피 FSM 포함
      ✅ LEFT/RIGHT -> STOP(0.3s) -> SIDE STEP(0.2m) -> resume
    """

    def __init__(self):
        super().__init__("goal_mover_vision_avoid")

        # ---- Parameters ----
        self.declare_parameter("cmd_topic", "cmd_vel")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_link")

        # PID gains
        self.declare_parameter("linear_P", 0.5)
        self.declare_parameter("linear_I", 0.0)
        self.declare_parameter("linear_D", 0.0)
        self.declare_parameter("angular_P", 0.2)
        self.declare_parameter("angular_I", 0.0)
        self.declare_parameter("angular_D", 0.05)

        # Tolerances
        self.declare_parameter("angle_tolerance_deg", 12.0)
        self.declare_parameter("pos_tolerance", 0.03)
        self.declare_parameter("final_yaw_tolerance_deg", 5.0)
        self.declare_parameter("enable_pid", 1.0)

        # Speed limits
        self.declare_parameter("max_linear_speed", 0.10)
        self.declare_parameter("max_angular_speed", 1.5)
        self.declare_parameter("min_linear_speed", 0.06)
        self.declare_parameter("min_angular_speed", 0.10)
        self.declare_parameter("min_speed_distance_threshold", 0.30)

        # TF lookup
        self.declare_parameter("tf_timeout_sec", 0.05)

        # Control rate
        self.declare_parameter("control_period_sec", 0.01)

        # ------------ Vision avoid ---------------
        self.declare_parameter("vision_avoid_topic", "/vision/avoid_dir_json")
        self.declare_parameter("vision_fresh_sec", 3.0)

        self.declare_parameter("avoid_trigger_dist_m", 0.30)
        self.declare_parameter("avoid_turn_deg", 45.0)    # (남겨둠: 이 버전에서는 사용 안함)

        # ✅ GO: 시간 기반 제거 -> 거리 기반 추가
        self.declare_parameter("avoid_go_dist_m", 0.20)   # ✅ 옆이동 거리 (sidestep distance)
        self.declare_parameter("avoid_go_tol_m", 0.02)    # ✅ 도달 허용 오차
        self.declare_parameter("avoid_go_speed", 0.12)    # ✅ 옆이동 속도(|m/s|)
        self.declare_parameter("avoid_go_max_sec", 2.0)   # 안전 탈출 타임아웃

        self.declare_parameter("avoid_stop_hold_sec", 0.30)
        self.declare_parameter("avoid_cooldown_sec", 1.0)
        self.declare_parameter("avoid_yaw_tol_deg", 6.0)  # (남겨둠: 이 버전에서는 yaw정렬 안함)

        # ✅ sidestep이 cmd_vel.linear.y를 사용하도록 (베이스가 지원해야 함)
        self.declare_parameter("use_strafe_y", True)
        # ------------ Vision avoid ---------------

        self._load_params()

        # ---- TF ----
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ---- PID ----
        self.linear_pid = PIDController(
            kp=self.linear_P, ki=self.linear_I, kd=self.linear_D,
            min_output=-self.max_linear_speed, max_output=self.max_linear_speed,
        )
        self.angular_pid = PIDController(
            kp=self.angular_P, ki=self.angular_I, kd=self.angular_D,
            min_output=-self.max_angular_speed, max_output=self.max_angular_speed,
        )

        # ---- State ----
        self.goal_msg: Optional[PoseStamped] = None
        self.mode: Mode = Mode.TURN_TO_GOAL

        # ---- Vision avoid state ----
        self._last_avoid: Optional[dict] = None
        self._last_avoid_t = 0.0
        self._avoid_active = False
        self._avoid_dir: Optional[str] = None           # "LEFT"|"RIGHT"
        self._avoid_target_yaw: Optional[float] = None  # (남겨둠: 이 버전에서는 사용 안함)
        self._avoid_phase_t0 = 0.0
        self._avoid_cooldown_until = 0.0
        self.avoid_yaw_tol = math.radians(self.avoid_yaw_tol_deg)

        # ✅ sidestep 거리 측정용 시작 pose/yaw + 진행방향(좌/우)
        self._avoid_go_start_x: Optional[float] = None
        self._avoid_go_start_y: Optional[float] = None
        self._avoid_go_start_yaw: Optional[float] = None
        self._avoid_side_sign: float = 0.0  # LEFT:+1, RIGHT:-1

        # ---- ROS I/O ----
        self.cmd_pub = self.create_publisher(Twist, self.cmd_topic, 10)
        self.distance_error_pub = self.create_publisher(Float64, "distance_error", 10)
        self.angle_error_pub = self.create_publisher(Float64, "angle_error", 10)
        self.final_yaw_error_pub = self.create_publisher(Float64, "final_yaw_error", 10)

        self.sub_vision_avoid = self.create_subscription(
            String, self.vision_avoid_topic, self._on_vision_avoid, 10
        )

        self.timer = self.create_timer(self.control_period_sec, self.control_loop)

        self.get_logger().info(
            f"✅ GoalMover(VisionAvoid) ready. vision_topic={self.vision_avoid_topic}"
        )

    def _load_params(self):
        self.cmd_topic = self.get_parameter("cmd_topic").value
        self.map_frame = self.get_parameter("map_frame").value
        self.base_frame = self.get_parameter("base_frame").value

        self.linear_P = float(self.get_parameter("linear_P").value)
        self.linear_I = float(self.get_parameter("linear_I").value)
        self.linear_D = float(self.get_parameter("linear_D").value)
        self.angular_P = float(self.get_parameter("angular_P").value)
        self.angular_I = float(self.get_parameter("angular_I").value)
        self.angular_D = float(self.get_parameter("angular_D").value)

        self.angle_tolerance = math.radians(float(self.get_parameter("angle_tolerance_deg").value))
        self.final_yaw_tolerance = math.radians(float(self.get_parameter("final_yaw_tolerance_deg").value))
        self.pos_tolerance = float(self.get_parameter("pos_tolerance").value)

        self.enable_pid = float(self.get_parameter("enable_pid").value) > 0.5

        self.max_linear_speed = float(self.get_parameter("max_linear_speed").value)
        self.max_angular_speed = float(self.get_parameter("max_angular_speed").value)
        self.min_linear_speed = float(self.get_parameter("min_linear_speed").value)
        self.min_angular_speed = float(self.get_parameter("min_angular_speed").value)
        self.min_speed_distance_threshold = float(self.get_parameter("min_speed_distance_threshold").value)

        self.tf_timeout_sec = float(self.get_parameter("tf_timeout_sec").value)
        self.control_period_sec = float(self.get_parameter("control_period_sec").value)

        # ---- Vision avoid ----
        self.vision_avoid_topic = self.get_parameter("vision_avoid_topic").value
        self.vision_fresh_sec = float(self.get_parameter("vision_fresh_sec").value)

        self.avoid_trigger_dist_m = float(self.get_parameter("avoid_trigger_dist_m").value)
        self.avoid_turn_deg = float(self.get_parameter("avoid_turn_deg").value)

        # ✅ 의미: sidestep distance/tol/speed
        self.avoid_go_dist_m = float(self.get_parameter("avoid_go_dist_m").value)
        self.avoid_go_tol_m = float(self.get_parameter("avoid_go_tol_m").value)
        self.avoid_go_speed = float(self.get_parameter("avoid_go_speed").value)
        self.avoid_go_max_sec = float(self.get_parameter("avoid_go_max_sec").value)

        self.avoid_stop_hold_sec = float(self.get_parameter("avoid_stop_hold_sec").value)
        self.avoid_cooldown_sec = float(self.get_parameter("avoid_cooldown_sec").value)
        self.avoid_yaw_tol_deg = float(self.get_parameter("avoid_yaw_tol_deg").value)

        self.use_strafe_y = bool(self.get_parameter("use_strafe_y").value)

    # ---------------- Goal set ----------------
    def goal_callback(self, x: float, y: float, yaw_deg: float, frame_id: str = "map"):
        yaw = math.radians(yaw_deg)
        qx, qy, qz, qw = quat_from_yaw(yaw)

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = 0.0
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw

        self.goal_msg = msg
        self.mode = Mode.TURN_TO_GOAL
        self.linear_pid.reset()
        self.angular_pid.reset()

        # 회피 상태 초기화
        self._finish_avoid(reset_pid=False)

        self.get_logger().info(
            f"🚀 Auto goal set: ({x:.2f}, {y:.2f}, yaw={yaw_deg:.1f}deg) frame={frame_id}"
        )

    # ---------------- TF helpers ----------------
    def _get_robot_pose_in_map(self) -> Optional[Tuple[float, float, float]]:
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.base_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=self.tf_timeout_sec),
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

    # ---------------- Vision avoid I/O ----------------
    def _on_vision_avoid(self, msg: String):
        try:
            data = json.loads(msg.data)
        except Exception as e:
            self.get_logger().warn(f"[VISION] json parse fail: {e} | raw={msg.data[:120]}")
            return

        # decision 키 확인
        decision = data.get("decision", None)
        if decision not in ("LEFT", "RIGHT", "STOP", "NONE", None):
            self.get_logger().warn(f"[VISION] unexpected decision={decision} keys={list(data.keys())}")

        # ✅ 수신 시각 기준으로 fresh 판단
        self._last_avoid = data
        self._last_avoid_t = time.time()

        # 디버그 로그(처음엔 켜두는 게 좋아)
        self.get_logger().debug(f"[VISION RX] decision={decision} t={self._last_avoid_t:.3f}")

    def _vision_fresh(self) -> bool:
        return (time.time() - self._last_avoid_t) <= self.vision_fresh_sec

    def _get_avoid_decision_and_dist(self) -> Tuple[Optional[str], Optional[float]]:
        """
        returns: (decision, dist2d)
          - decision: "LEFT"/"RIGHT"/"KEEP"/None
          - dist2d: meters or None

        기대 JSON 예:
        {
          "decision": "LEFT",
          "object": { "dist2d": 0.35, "name": "person" }
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

    def _start_avoid(self, direction: str, cur_yaw: float):
        """
        direction: "LEFT" or "RIGHT"
        - STOP(hold) -> TURN -> GO(distance) -> resume
        """
        sign = +1.0 if direction == "LEFT" else -1.0
        delta = math.radians(self.avoid_turn_deg) * sign
        self._avoid_dir = direction
        self._avoid_target_yaw = normalize_angle(cur_yaw + delta)
        self._avoid_active = True

        self.mode = Mode.AVOID_STOP
        self._avoid_phase_t0 = time.time()

        # GO 시작 시점(턴 완료 후)에 세팅
        self._avoid_go_start_x = None
        self._avoid_go_start_y = None
        self._avoid_go_start_yaw = None
        self._avoid_side_sign = sign

        self.angular_pid.reset()
        self.linear_pid.reset()

        self.get_logger().warn(
            f"[AVOID START] dir={direction} yaw={math.degrees(cur_yaw):.1f} target={math.degrees(self._avoid_target_yaw):.1f}"
        )

    def _finish_avoid(self, reset_pid: bool = True):
        self._avoid_active = False
        self._avoid_dir = None
        self._avoid_target_yaw = None
        self._avoid_phase_t0 = 0.0

        self._avoid_go_start_x = None
        self._avoid_go_start_y = None
        self._avoid_go_start_yaw = None
        self._avoid_side_sign = 0.0

        self._avoid_cooldown_until = time.time() + self.avoid_cooldown_sec

        # 원래 goal 주행으로 복귀
        self.mode = Mode.TURN_TO_GOAL
        if reset_pid:
            self.angular_pid.reset()
            self.linear_pid.reset()

    def _avoid_traveled_m(self, cur_x: float, cur_y: float) -> Optional[float]:
        """
        GO 단계에서 시작점으로부터 누적 이동거리(미터).
        """
        if (self._avoid_go_start_x is None or
            self._avoid_go_start_y is None):
            return None

        dxw = float(cur_x) - float(self._avoid_go_start_x)
        dyw = float(cur_y) - float(self._avoid_go_start_y)
        return math.hypot(dxw, dyw)

    # ---------------- Main control loop ----------------
    def control_loop(self):
        if self.goal_msg is None:
            return

        robot = self._get_robot_pose_in_map()
        goal = self._goal_xy_yaw_in_map()
        if robot is None or goal is None:
            self.get_logger().warn("[EARLY RETURN] robot/goal not ready (TF fail) -> avoid not running")
            return

        rx, ry, ryaw = robot
        gx, gy, gyaw = goal

        # -------- Vision avoid input --------
        decision, dist2d = self._get_avoid_decision_and_dist()
        now = time.time()
        fresh = (self._last_avoid is not None) and ((now - self._last_avoid_t) <= self.vision_fresh_sec)
        self.get_logger().debug(f"[VISION] decision={decision} dist2d={dist2d} fresh={fresh}")

        # -------- If avoid FSM active, it owns cmd_vel --------
        if self._avoid_active:
            if self.mode == Mode.AVOID_STOP:
                self._publish_stop()
                if (now - self._avoid_phase_t0) >= self.avoid_stop_hold_sec:
                    self.mode = Mode.AVOID_TURN
                    self._avoid_phase_t0 = now
                    self.angular_pid.reset()
                return

            elif self.mode == Mode.AVOID_TURN:
                if self._avoid_target_yaw is None:
                    self._finish_avoid()
                    self._publish_stop()
                    return

                yaw_err = normalize_angle(self._avoid_target_yaw - ryaw)
                if abs(yaw_err) <= self.avoid_yaw_tol:
                    self.mode = Mode.AVOID_GO
                    self._avoid_phase_t0 = now
                    self._avoid_go_start_x = float(rx)
                    self._avoid_go_start_y = float(ry)
                    self._avoid_go_start_yaw = float(ryaw)
                    self._publish_stop()
                    return

                cmd = Twist()
                w = self.angular_pid.compute(yaw_err) if self.enable_pid else (self.angular_P * yaw_err)
                w = max(-self.max_angular_speed, min(self.max_angular_speed, w))
                if abs(w) < self.min_angular_speed:
                    w = math.copysign(self.min_angular_speed, w)
                cmd.angular.z = w
                cmd.linear.x = 0.0
                self.cmd_pub.publish(cmd)
                return

            elif self.mode == Mode.AVOID_GO:
                traveled = self._avoid_traveled_m(rx, ry)
                if traveled is None:
                    self._finish_avoid()
                    self._publish_stop()
                    return

                err = float(self.avoid_go_dist_m) - float(traveled)
                if err <= float(self.avoid_go_tol_m):
                    self._finish_avoid()
                    self._publish_stop()
                    return

                if self.avoid_go_max_sec > 0.0 and (now - self._avoid_phase_t0) >= self.avoid_go_max_sec:
                    self._finish_avoid()
                    self._publish_stop()
                    return

                cmd = Twist()
                v = max(0.0, float(self.avoid_go_speed))
                v = min(float(self.max_linear_speed), v)
                cmd.linear.x = float(v)
                cmd.angular.z = 0.0

                self.cmd_pub.publish(cmd)
                return

            else:
                self._finish_avoid()
                self._publish_stop()
                return
            
        # -------- Start avoid if trigger met --------
        if now >= self._avoid_cooldown_until:
            trigger_ok = (dist2d is None) or (dist2d <= self.avoid_trigger_dist_m)
            if (decision in ("LEFT", "RIGHT")) and trigger_ok:
                self._start_avoid(decision, ryaw)
                self._publish_stop()

                return

        # -------- Normal PID drive to goal --------
        dx = gx - rx
        dy = gy - ry
        dist = math.hypot(dx, dy)

        heading_to_goal = math.atan2(dy, dx)
        heading_err = normalize_angle(heading_to_goal - ryaw)
        final_yaw_err = normalize_angle(gyaw - ryaw)

        dm = Float64(); dm.data = float(dist)
        am = Float64(); am.data = float(heading_err)
        fm = Float64(); fm.data = float(final_yaw_err)
        self.distance_error_pub.publish(dm)
        self.angle_error_pub.publish(am)
        self.final_yaw_error_pub.publish(fm)

        if dist < self.pos_tolerance:
            self.mode = Mode.FINAL_ALIGN

        cmd = Twist()

        if self.mode == Mode.TURN_TO_GOAL:
            if abs(heading_err) <= self.angle_tolerance:
                self.mode = Mode.GO_STRAIGHT
                self.angular_pid.reset()
                cmd.angular.z = 0.0
                cmd.linear.x = 0.0
            else:
                w = self.angular_pid.compute(heading_err) if self.enable_pid else (self.angular_P * heading_err)
                w = max(-self.max_angular_speed, min(self.max_angular_speed, w))
                if abs(w) < self.min_angular_speed:
                    w = math.copysign(self.min_angular_speed, w)
                cmd.angular.z = w
                cmd.linear.x = 0.0

        elif self.mode == Mode.GO_STRAIGHT:
            if abs(heading_err) > self.angle_tolerance:
                self.mode = Mode.TURN_TO_GOAL
                self.linear_pid.reset()
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
            else:
                v = self.linear_pid.compute(dist) if self.enable_pid else (self.linear_P * dist)
                v = max(-self.max_linear_speed, min(self.max_linear_speed, v))

                if dist > self.min_speed_distance_threshold:
                    if abs(v) < self.min_linear_speed:
                        v = math.copysign(self.min_linear_speed, v)

                cmd.linear.x = v
                cmd.angular.z = 0.0

        elif self.mode == Mode.FINAL_ALIGN:
            if abs(final_yaw_err) <= self.final_yaw_tolerance:
                self.get_logger().info("✅ reached goal pose. stop.")
                self.goal_msg = None
                self.mode = Mode.TURN_TO_GOAL
                self.linear_pid.reset()
                self.angular_pid.reset()
                self._publish_stop()
                return
            else:
                w = self.angular_pid.compute(final_yaw_err) if self.enable_pid else (self.angular_P * final_yaw_err)
                w = max(-self.max_angular_speed, min(self.max_angular_speed, w))
                if abs(w) < self.min_angular_speed:
                    w = math.copysign(self.min_angular_speed, w)
                cmd.angular.z = w
                cmd.linear.x = 0.0

        self.cmd_pub.publish(cmd)


def _parse_goal_from_argv(argv):
    """
    Usage:
      ros2 run actions goal_mover_simple_visionavoid X Y YAW [--yaw-rad] [--frame map]
    """
    if len(argv) < 4:
        print("Usage: goal_mover X Y YAW [--yaw-rad] [--frame map]")
        raise SystemExit(2)
    x = float(argv[1])
    y = float(argv[2])
    yaw_in = float(argv[3])

    yaw_deg = yaw_in
    if "--yaw-rad" in argv:
        yaw_deg = math.degrees(yaw_in)

    frame = "map"
    if "--frame" in argv:
        i = argv.index("--frame")
        if i + 1 < len(argv):
            frame = argv[i + 1]

    return x, y, yaw_deg, frame


def main(args=None):
    x, y, yaw_deg, frame = _parse_goal_from_argv(sys.argv)

    rclpy.init(args=args)
    node = GoalMover()
    node.goal_callback(x, y, yaw_deg, frame_id=frame)

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

# 실행 예시:
# ros2 run actions goal_mover_obs_visionavoid 0.322 -0.684 -1.574 --yaw-rad  (F)
# ros2 run actions goal_mover_obs_visionavoid 0.271 -1.028 -1.615 --yaw-rad  (G)
# ros2 run actions goal_mover_obs_visionavoid 0.069 -0.972 1.542 --yaw-rad  (H)

# 파라미터 예시(요청 값):
# -p avoid_stop_hold_sec:=0.30  (정지 0.3s)
# -p avoid_go_dist_m:=0.20      (옆이동 0.2m)
# -p avoid_go_speed:=0.12       (옆이동 속도)
# -p use_strafe_y:=true         (linear.y 사용)

### 회피 운행 조건
## 핑키 on 
# ros2 run sensors image_socket
# ros2 run sensors map_wall_xy_near_robot

## vision 서버 on
# (가상환경) $VIRTUAL_ENV/bin/python3 ~/pinky_pro/install/pinky_vision/lib/pinky_vision/map_to_obj   --ros-args -p scale_s:=0.21
# (가상환경) $VIRTUAL_ENV/bin/python3 ~/pinky_pro/install/pinky_vision/lib/pinky_vision/avoid_publisher   --ros-args   -p trigger_dist_m:=0.8   -p step_m:=0.35   -p wall_clear_min_m:=0.25   -p prefer_away_from_object_gain:=0.15

## vision 판단 정보 확인
# ros2 topic echo /vision/avoid_dir_json

## 맵상에서 벽체정보가 잘들어오는지 
# ros2 topic echo /debug/walls_xy_json

