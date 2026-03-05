#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import String

import tf2_ros
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy


def yaw_from_quat(x, y, z, w) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)

class MapWallXYPublisher(Node):
    """
    OccupancyGrid(/map)에서 벽(점유 셀) 좌표를 world(map) 좌표 (x,y)로 변환해 JSON으로 발행.
    ✅ 개선: 로봇(base_frame) 주변 radius_m 이내의 벽만 추출 (방법 A: 로컬 맵만 사용)
    """

    def __init__(self):
        super().__init__("map_wall_xy_publisher")

        # ---- params ----
        self.declare_parameter("map_topic", "/map")
        self.declare_parameter("occ_thresh", 50)        # 50~65 많이 씀
        self.declare_parameter("stride", 2)             # 샘플링 간격 (클수록 포인트 수 감소)
        self.declare_parameter("max_points", 5000)      # JSON 크기 제한
        self.declare_parameter("json_topic", "/debug/walls_xy_json")
        self.declare_parameter("publish_period_sec", 0.5)

        # TF frames
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_link")

        # Local radius (✅ 1.5 ~ 2.0m 권장)
        self.declare_parameter("radius_m", 2.0)

        # ---- read params ----
        self.map_topic = str(self.get_parameter("map_topic").value)
        self.occ_thresh = int(self.get_parameter("occ_thresh").value)
        self.stride = max(1, int(self.get_parameter("stride").value))
        self.max_points = max(100, int(self.get_parameter("max_points").value))
        self.json_topic = str(self.get_parameter("json_topic").value)
        self.publish_period = float(self.get_parameter("publish_period_sec").value)

        self.map_frame = str(self.get_parameter("map_frame").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.radius_m = float(self.get_parameter("radius_m").value)
        if self.radius_m <= 0:
            self.radius_m = 2.0
        self.radius2 = self.radius_m * self.radius_m

        # ---- QoS: /map publisher가 TRANSIENT_LOCAL 이므로 구독도 맞춰야 함 ----
        qos_map = QoSProfile(depth=1)
        qos_map.reliability = ReliabilityPolicy.RELIABLE
        qos_map.durability  = DurabilityPolicy.TRANSIENT_LOCAL

        self.sub = self.create_subscription(OccupancyGrid, self.map_topic, self._on_map, qos_map)
        self.pub = self.create_publisher(String, self.json_topic, 10)

        # ---- TF ----
        self.tf_buffer = tf2_ros.Buffer(cache_time=rclpy.duration.Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self._last_map: OccupancyGrid | None = None
        self.timer = self.create_timer(self.publish_period, self._tick)

        self.get_logger().info(
            f"sub={self.map_topic}, pub={self.json_topic}, occ_thresh={self.occ_thresh}, "
            f"stride={self.stride}, max_points={self.max_points}, radius_m={self.radius_m:.2f}"
        )
        self.get_logger().info(f"frames: map={self.map_frame}, base={self.base_frame}")

    def _on_map(self, msg: OccupancyGrid):
        self._last_map = msg
        self.get_logger().info(
            f"got /map: {msg.info.width}x{msg.info.height}, res={msg.info.resolution:.3f}, "
            f"origin=({msg.info.origin.position.x:.3f},{msg.info.origin.position.y:.3f})"
        )

    def _lookup_robot_pose_map(self) -> Optional[Tuple[float, float, float]]:
        """
        map -> base_link (x,y,yaw)
        """
        try:
            tf = self.tf_buffer.lookup_transform(self.map_frame, self.base_frame, rclpy.time.Time())
        except Exception:
            return None

        tx = float(tf.transform.translation.x)
        ty = float(tf.transform.translation.y)
        q = tf.transform.rotation
        yaw = yaw_from_quat(q.x, q.y, q.z, q.w)
        return tx, ty, yaw

    def _tick(self):
        msg = self._last_map
        if msg is None:
            return
        
        robot_pose = self._lookup_robot_pose_map()
        if robot_pose is None:
            # TF가 아직 준비 안 됐으면 로컬 추출이 불가
            payload = {
                "ok": False,
                "reason": "tf_not_ready",
                "frames": {"map": self.map_frame, "base": self.base_frame},
            }
            out = String()
            out.data = json.dumps(payload, ensure_ascii=False)
            self.pub.publish(out)
            return

        rx, ry, ryaw = robot_pose

        info = msg.info
        res = float(info.resolution)
        ox = float(info.origin.position.x)
        oy = float(info.origin.position.y)
        w = int(info.width)
        h = int(info.height)

        data = msg.data  # length = w*h, row-major, index = x + y*w
        points = []

        # 로봇 주변 radius_m 이내만 추출
        # world = origin + (gx+0.5)*res, (gy+0.5)*res (셀 중심)
        for gy in range(0, h, self.stride):
            base = gy * w
            wy = oy + (gy + 0.5) * res

            # y만으로도 radius 밖이면 스킵 가능(가벼운 pruning)
            dy = wy - ry
            if dy * dy > self.radius2:
                continue

            for gx in range(0, w, self.stride):
                v = int(data[base + gx])
                if v < 0:
                    continue  # unknown
                if v < self.occ_thresh:
                    continue

                wx = ox + (gx + 0.5) * res
                dx = wx - rx
                if dx * dx + dy * dy <= self.radius2:
                    points.append([float(wx), float(wy)])
                    if len(points) >= self.max_points:
                        break
            if len(points) >= self.max_points:
                break

        # (선택) bbox 요약도 같이 넣으면 디버깅 쉬움
        xs = [p[0] for p in points] if points else []
        ys = [p[1] for p in points] if points else []

        payload = {
            "ok": True,
            "frame": msg.header.frame_id,  # 보통 "map"
            "stamp": {"sec": int(msg.header.stamp.sec), "nanosec": int(msg.header.stamp.nanosec)},
            "map": {
                "resolution": res,
                "origin": [ox, oy],
                "width": w,
                "height": h,
                "occ_thresh": self.occ_thresh,
                "stride": self.stride,
            },
            "local": {
                "radius_m": float(self.radius_m),
                "robot_pose_map": {"x": float(rx), "y": float(ry), "yaw": float(ryaw)},
                "bbox": {
                    "x_min": float(min(xs)) if xs else None,
                    "x_max": float(max(xs)) if xs else None,
                    "y_min": float(min(ys)) if ys else None,
                    "y_max": float(max(ys)) if ys else None,
                },
            },
            "count": len(points),
            "walls_xy": points,  # 로봇 주변 R미터 이내 벽 좌표만!
        }

        out = String()
        out.data = json.dumps(payload, ensure_ascii=False)
        self.pub.publish(out)


def main():
    rclpy.init()
    node = MapWallXYPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()


## 실행
# ros2 run pinky_vision map_wall_xy_near_robot

# ros2 run pinky_vision map_wall_xy --ros-args -p stride:=1

## 보기
# addinedu@addinedu-GF65-Thin-10UE:~$ ros2 topic echo /debug/walls_xy_json --once --full-length
# data: '{"frame": "map", "stamp": {"sec": 1770860715, "nanosec": 835399851}, "map": {"resolution": 0.05000000074505806, "origin": [-0.374, -1.406], "width": 26, "height": 42, "occ_thresh": 50, "stride": 4}, "count": 9, "walls_xy": [[-0.34899999962747097, -1.3809999996274709], [0.8510000182539225, -1.1809999966472386], [-0.14899999664723873, -0.7809999906867742], [-0.14899999664723873, 0.6190000301748515], [0.05100000633299351, 0.6190000301748515], [0.25100000931322575, 0.6190000301748515], [0.451000012293458, 0.6190000301748515], [0.6510000152736902, 0.6190000301748515], [0.8510000182539225, 0.6190000301748515]]}'
# ---

# ros2 topic echo /debug/walls_xy_json --once --full-length \
# | sed -n 's/^data: //p' \
# | sed "s/^'//; s/'$//" \
# | jq '.count, .map, .walls_xy'
