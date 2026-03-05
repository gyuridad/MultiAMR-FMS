#!/usr/bin/env python3
import json
import time
import math
import heapq
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
    # ROS 2(rclpy)에서 “콜백을 동시에(재진입 가능하게) 실행해도 된다” 는 성격의 Callback Group 타입을 가져오는 임포트
from rclpy.duration import Duration

from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import Float32, String

import tf2_ros
from tf2_ros import TransformListener

# 인터페이스에 파일 생성해야함 !!!
from pinky_interfaces.msg import RobotState
from pinky_interfaces.action import MoveToPID, FollowAruco


def yaw_from_quat(x, y, z, w):
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def quat_from_yaw(yaw: float):
    half = yaw * 0.5
    return (0.0, 0.0, math.sin(half), math.cos(half))


@dataclass
class WP:
    x: float
    y: float
    yaw: float = 0.0


class PinkySystem1(Node):
    def __init__(self):
        super().__init__("pinky1_system1")

        self.cb_group = ReentrantCallbackGroup()

        # params
        self.declare_parameter("robot_name", "pinky1")
        self.declare_parameter("state_topic", "robot_state")   # namespaced
        self.declare_parameter("result_topic", "mission_result")     # namespaced

        # domain_bridge에서 보통 GLOBAL이 편함(권장)
        self.declare_parameter("mission_request_topic", "mission_request")
        self.declare_parameter("mission_cancel_topic", "mission_cancel")

        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_link")

        self.declare_parameter("move_to_pid_action_name", "actions/move_to_pid")
        self.declare_parameter("follow_aruco_action_name", "actions/follow_aruco")

        ##### rtb #####
        self.declare_parameter("battery_voltage_topic", "battery_voltage")
        self.declare_parameter("battery_low_voltage_threshold", 7.5)
        self.declare_parameter("battery_watch_period_sec", 1.0)
        self.declare_parameter("battery_low_hold_sec", 2.0)  # 2초 연속 low면 발동(추천)

        # ✅ mission_id dedup cache
        self.declare_parameter("completed_cache_size", 50)         # 최근 몇 개 보관
        self.declare_parameter("completed_ttl_sec", 600.0)         # 완료 기록 유지시간(초). 0이면 무제한(캐시 사이즈만 적용)

        # ---- Waypoint params ----
        # empty_map2 수정전
        # default_wps = {
        #     "A": {"x": 0.07991521616038465, "y": 0.0208531509311552, "yaw": 1.569092840518374},    
        #     "B": {"x": 0.0982715949828098, "y": 0.4152183361819458, "yaw": 1.5588626874735612},  
        #     "C": {"x": 0.3273481339916138, "y": 0.3808828616634004, "yaw": -1.65683453511985},  
        #     "D": {"x": 0.31136767117452996, "y": -0.03364462326598874, "yaw": -1.6353521361867218},  
        #     "E": {"x": 0.2311741862494205, "y": -0.3661416362053365, "yaw": -1.6249211092392029},    
        #     "F": {"x": 0.3225842275905132, "y": -0.6840307808082434 , "yaw": -1.5742126342716196}, 
        #     "G": {"x": 0.27105983124661476, "y": -1.0283068390452972, "yaw": -1.6156553100392443}, 
        #     "H": {"x": 0.06957616919930265, "y": -0.9729490000804174, "yaw": 1.5421559861678504}, 
        #     "I": {"x": 0.0933211439620267, "y": -0.6331708255890548, "yaw": 1.5373546316370412}, 
        #     "J": {"x": 0.09680786078312531, "y": -0.3154627564498954, "yaw": 1.5511221530554202}, 
        # }
        # empty_map2 수정후
        default_wps = {
            "A": {"x": 0.15891485402430086, "y": 0.006662365163616427, "yaw": 1.5989778609767642},    
            "B": {"x": 0.0982715949828098, "y": 0.4152183361819458, "yaw": 1.5588626874735612},  
            "C": {"x": 0.3273481339916138, "y": 0.3808828616634004, "yaw": -1.65683453511985},  
            "D": {"x": 0.31136767117452996, "y": -0.03364462326598874, "yaw": -1.6353521361867218},  
            "E": {"x": 0.32810263277098783, "y": -0.37014285859706564, "yaw": -1.6203311273233227},    
            "F": {"x": 0.3225842275905132, "y": -0.6840307808082434 , "yaw": -1.5742126342716196}, 
            "G": {"x": 0.27105983124661476, "y": -1.0283068390452972, "yaw": -1.6156553100392443}, 
            "H": {"x": 0.06957616919930265, "y": -0.9729490000804174, "yaw": 1.5421559861678504}, 
            "I": {"x": 0.153143961865185, "y": -0.6665573416040188, "yaw": 1.5772812791268003}, 
            "J": {"x": 0.15761047209156467, "y": -0.3254817351067633, "yaw": 1.5633740768497586}, 
        }
        default_edges = {
            "A": ["B", "D", "J"],
            "B": ["A", "C"],
            "C": ["B", "D"],
            "D": ["A", "C", "E"],
            "E": ["D", "F", "J"],
            "F": ["E", "G", "I"],
            "G": ["F", "H"],
            "H": ["G", "I"],
            "I": ["F", "H", "J"],
            "J": ["A", "E", "I"],
        }
        self.declare_parameter("waypoints_json", json.dumps(default_wps))
        self.declare_parameter("waypoint_edges_json", json.dumps(default_edges))
        self.declare_parameter("waypoint_snap_max_dist", 0.5)
        self.declare_parameter("wp_timeout_sec", 60.0)       # waypoint 1개당 timeout
        self.declare_parameter("final_timeout_sec", 120.0)    # 최종 목표 timeout

        # ---------------- read params ----------------
        self.robot_name = self.get_parameter("robot_name").value
        self.state_topic = self.get_parameter("state_topic").value
        self.result_topic = str(self.get_parameter("result_topic").value)
        self.mission_request_topic = str(self.get_parameter("mission_request_topic").value)
        self.mission_cancel_topic = str(self.get_parameter("mission_cancel_topic").value)

        self.map_frame = self.get_parameter("map_frame").value
        self.base_frame = self.get_parameter("base_frame").value
        self.move_to_pid_action_name = self.get_parameter("move_to_pid_action_name").value
        self.follow_aruco_action_name = self.get_parameter("follow_aruco_action_name").value

        self.battery_voltage_topic = self.get_parameter("battery_voltage_topic").value
        self.battery_watch_period = float(self.get_parameter("battery_watch_period_sec").value)
        self.battery_low_hold_sec = float(self.get_parameter("battery_low_hold_sec").value)
        self.battery_low_voltage_threshold = float(self.get_parameter("battery_low_voltage_threshold").value)

        self.max_completed_cache = int(self.get_parameter("completed_cache_size").value)
        self.completed_ttl_sec = float(self.get_parameter("completed_ttl_sec").value)

        # ---------------- Internal states ----------------
        self.battery_voltage = 8.82
        self.battery_soc = 100.0 

        self._low_v_latched = False   # ✅ 저전압 RTB 재발동 방지 래치
        self._rtb_in_progress = False          # RTB 중복 방지
        self._low_v_since = None               # low 전압 지속 시간(선택)

        self._follow_timeout_timer = None
        self._follow_deadline = 0.0
        #######################
        
        # ✅ completed mission cache: mission_id -> {"t":..., "success":..., "message":...}
        self.completed_missions: Dict[str, Dict[str, Any]] = {}

        self._load_waypoints()

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ---- robot state internal ----
        self.system_state = "IDLE"     # 예: "IDLE", "RUNNING", "ERROR"
        self.queue_status = "idle"     # 예: "idle", "running", "done", "error", "canceled"
        self.current_index = -1        # 미션 plan의 steps 리스트에서 지금 몇 번째 작업 중인지 표시
        self.mission_id = ""
        self.plan_json = ""            # 받은 미션 계획 전체 JSON 문자열 원본
        self.history = []              # 각 step(또는 move_to_pid 호출) 결과를 쌓아두는 실행 기록
        self.last_violation = ""       # 가장 최근 규칙 위반/안전 위반 같은 내용을 한 줄로 저장하는 칸
        self.events = []               # 중요 이벤트 로그(INFO/WARN/ERROR 같은 이벤트를 시간과 함께 기록)
        self.latest_twist = Twist()
        self.primary_id = -1
        self.lost_sec = 999.0
        self.vision_json = "{}"

        # ---------------- mission runtime (topic-based) ----------------
        self._mission_active = False
        self._steps: List[Dict[str, Any]] = []
        self._step_i = -1
        self._subgoal_queue: List[Tuple[PoseStamped, float, str]] = []
        self._move_goal_handle = None
        self._follow_goal_handle = None
        self._active_action = ""  # "move"/"follow"/""

        # ---- RTB Future-chain runtime state ----  !!수정 필요!!
        self._rtb_goal_queue: List[Tuple[float, float, float, float, str]] = []
            # RTB를 할 때 System1이 “한 번에 한 goal만” 보내는 게 아니라,
            # (A까지) 웨이포인트 경유 경로를 만들고
            # 그 경로에 포함된 목표들을 순서대로 MoveToPID 액션으로 보내야 하잖아?
            # 그 “순서대로 보낼 목표 목록(큐)”가 _rtb_goal_queue야.
            # (x, y, yaw, timeout_sec, 좌표명) 형태로 저장

        # pub/sub
        self.state_pub = self.create_publisher(RobotState, self.state_topic, 10)
        self.result_pub = self.create_publisher(String, self.result_topic, 10)

        self.mission_sub = self.create_subscription(
            String, self.mission_request_topic, self._on_mission_request, 10,
            callback_group=self.cb_group
        )
        self.cancel_sub = self.create_subscription(
            String, self.mission_cancel_topic, self._on_mission_cancel, 10,
            callback_group=self.cb_group
        )

        self.battery_sub = self.create_subscription(
            Float32,  # 토픽 타입이 Float64면 Float64로 바꿔
            self.battery_voltage_topic,
            self._on_battery_voltage,
            10,
            callback_group=self.cb_group,
        )

        # ---------------- action clients ----------------
        self.move_client = ActionClient(self, MoveToPID, self.move_to_pid_action_name, callback_group=self.cb_group)
        self.follow_client = ActionClient(
            self, FollowAruco, self.follow_aruco_action_name, callback_group=self.cb_group
        )

        # ---------------- Timers ----------------
        self.create_timer(0.2, self.publish_state, callback_group=self.cb_group)
        self.create_timer(self.battery_watch_period, self._battery_watchdog, callback_group=self.cb_group)

        self.get_logger().info(
            f"[System1Topic] ready robot={self.robot_name} "
            f"req={self.mission_request_topic} cancel={self.mission_cancel_topic} "
            f"move_action={self.move_to_pid_action_name} follow_action={self.follow_aruco_action_name} "
            f"dedup_cache={self.max_completed_cache} ttl={self.completed_ttl_sec}s"
        )

    # ======================================================================
    # Dedup helpers
    # ======================================================================
    def _prune_completed(self):
        """TTL/size 기반으로 completed_missions를 정리."""
        now = time.time()

        # TTL prune
        if self.completed_ttl_sec > 0:
            dead = []
            for mid, info in self.completed_missions.items():
                if (now - float(info.get("t", 0.0))) > self.completed_ttl_sec:
                    dead.append(mid)
            for mid in dead:
                self.completed_missions.pop(mid, None)

        # Size prune (oldest-first)
        if self.max_completed_cache > 0 and len(self.completed_missions) > self.max_completed_cache:
            items = sorted(self.completed_missions.items(), key=lambda kv: float(kv[1].get("t", 0.0)))
            overflow = len(items) - self.max_completed_cache
            for i in range(overflow):
                self.completed_missions.pop(items[i][0], None)

    def _is_completed(self, mission_id: str) -> bool:
        self._prune_completed()
        return mission_id in self.completed_missions

    # ======================================================================
    # Battery / RTB
    # ======================================================================
    def _on_battery_voltage(self, msg):
        self.battery_voltage = float(msg.data)
    
    def _battery_is_low(self) -> bool:
        return float(self.battery_voltage) <= float(self.battery_low_voltage_threshold)

    def _battery_is_recovered(self) -> bool:
        # ✅ 같은 threshold를 쓰되, 회복은 더 높은 값에서만 인정(히스테리시스)
        margin = 0.5  # 예: 0.5V
        return float(self.battery_voltage) >= float(self.battery_low_voltage_threshold + margin)

    def _battery_watchdog(self):
        # IDLE에서만 자동 RTB (미션 중이면 execute 쪽이 처리)
        if self.system_state != "IDLE":
            self._low_v_since = None
                # 배터리 저전압 상태가 “얼마나 오래 지속됐는지”를 측정하기 위한 타임스탬프 변수야.
                # 즉, 순간적인 전압 드롭(노이즈) 때문에
                # RTB(Return-To-Base)가 바로 트리거되지 않도록 하는 디바운스 / 홀드 타이머 역할을 해
            return

        if self._rtb_in_progress:
            return

        # ✅ 저전압 래치가 걸려있으면, 회복될 때까지 RTB 재발동 금지
        if self._low_v_latched:
            if self._battery_is_recovered():      # ✅ recovered일 때만 unlatch!
                self._low_v_latched = False
                self._low_v_since = None
                self.events.append({"t": time.time(), "type": "INFO",
                                    "msg": f"battery recovered -> unlatch (V={self.battery_voltage:.2f}V)"})
            return

        
        if self._battery_is_low():
            now = time.time()
            # 전압이 낮아졌지만, 처음 감지됐을 때
            if self._low_v_since is None:
                self._low_v_since = now   # 저전압 처음 감지 now를 기록
                return
            # 전압이 계속 낮은 상태로 유지될 때, 지속 시간이 아직 부족
            if (now - self._low_v_since) < self.battery_low_hold_sec:
                return

            # 저전압이 충분히 오래 유지되었을 때
            # ✅ RTB 발동 + 래치 ON
            self._rtb_in_progress = True
            self._low_v_latched = True
            # “저전압 때문에 RTB가 발동됐다”는 원인 로그
            self.events.append({"t": now, "type": "WARN", "msg": f"IDLE low voltage -> RTB start (V={self.battery_voltage:.2f}V)"})

            self._start_rtb_future_chain()
        else:
            self._low_v_since = None

    def _finish_rtb(self, ok: bool):
        self.events.append({"t": time.time(),
                            "type": "INFO" if ok else "ERROR",
                            "msg": f"IDLE RTB {'done' if ok else 'failed'}"})
        self._rtb_goal_queue = []
        self._rtb_in_progress = False
        self._low_v_since = None

    # ---- send → accept 확인 → result 확인 → 다음 send → … 반복 ----
    def _start_rtb_future_chain(self):
        wp = self.waypoints.get("A")
        if wp is None:
            self.events.append({
                "t": time.time(),
                "type": "ERROR",
                "msg": "RTB failed: Home waypoint 'A' not found in waypoints"
            })
            self._finish_rtb(False)
            return

        ok, rx, ry, _ = self._tf_pose()  # 현재 로봇 위치 좌표 추출
        if not ok:
            self.events.append({"t": time.time(), "type": "ERROR", "msg": "TF pose unavailable (RTB)"})
            self._finish_rtb(False)
            return
        
        s_wp, s_d = self._nearest_wp(rx, ry)   # 시작 waypoint명 / 현재 위치에서 시작 포인트까지 거리
        g_wp, g_d = self._nearest_wp(wp.x, wp.y)   # 초기 대기장소 A 좌표

        wp_timeout = float(self.get_parameter("wp_timeout_sec").value)
        final_timeout = float(self.get_parameter("final_timeout_sec").value)

        queue: List[Tuple[float, float, float, float, str]] = []

        if (s_wp is None or g_wp is None or
            s_d > self.wp_snap_max_dist or g_d > self.wp_snap_max_dist):

            self.events.append({"t": time.time(), "type": "WARN",
                                "msg": f"RTB WP snap failed: start_d={s_d:.2f}, goal_d={g_d:.2f} -> direct"})
            queue.append((wp.x, wp.y, wp.yaw, final_timeout, "home_direct"))
        else:
            chain = self._dijkstra_wp_path(s_wp, g_wp)
            if not chain:
                self.events.append({"t": time.time(), "type": "ERROR", "msg": f"RTB No WP path {s_wp}->{g_wp}"})
                self._finish_rtb(False)
                return

            self.events.append({"t": time.time(), "type": "INFO", "msg": f"RTB WP path {s_wp}->{g_wp}: {chain}"})

            for name in chain:
                w = self.waypoints[name]
                queue.append((w.x, w.y, w.yaw, wp_timeout, f"wp:{name}"))

            queue.append((wp.x, wp.y, wp.yaw, final_timeout, "home_final"))
                # 체인이 chain = ["M", "A"] 일때:
                # 결과 queue는
                #     queue = [
                #     (0.2692192294524333,  -1.1414312884666327,  3.0821446616985146, 60.0,  "wp:M"),
                #     (0.0,                 0.0,                 0.0,                60.0,  "wp:A"),
                #     (0.0,                 0.0,                 0.0,                120.0, "home_final"),
                #     ]

        self._rtb_goal_queue = queue
        self._rtb_send_next_goal()

    def _rtb_send_next_goal(self):
        if not self._rtb_goal_queue:
            self._finish_rtb(True)
            return

        if not self.move_client.wait_for_server(timeout_sec=2.0):
            self.events.append({"t": time.time(), "type": "ERROR", "msg": "RTB MoveToPID server not available"})
            self._finish_rtb(False)
            return
        
        x, y, yaw, timeout_sec, label = self._rtb_goal_queue.pop(0)
        ps = self._pose_stamped(x, y, yaw)   # ros2 형식으로 포즈를 변경

        goal = MoveToPID.Goal()
        goal.target = ps
        goal.timeout_sec = float(timeout_sec)

        self.events.append({"t": time.time(), "type": "INFO",
                            "msg": f"RTB send MoveToPID ({label}) x={x:.2f},y={y:.2f},t={timeout_sec:.1f}s"})
        
        send_fut = self.move_client.send_goal_async(goal)
        send_fut.add_done_callback(lambda fut: self._rtb_on_goal_response(fut, label, x, y))

    def _rtb_on_goal_response(self, fut, label: str, x: float, y: float):
        try:
            goal_handle = fut.result()
        except Exception as e:
            self.events.append({"t": time.time(), "type": "ERROR", "msg": f"RTB goal response error: {e}"})
            self._finish_rtb(False)
            return

        if not goal_handle.accepted:
            self.events.append({"t": time.time(), "type": "ERROR", "msg": f"RTB goal rejected ({label})"})
            self._finish_rtb(False)
            return

        res_fut = goal_handle.get_result_async()
        res_fut.add_done_callback(lambda rfut: self._rtb_on_result(rfut, label, x, y))

    def _rtb_on_result(self, fut, label: str, x: float, y: float):
        try:
            res = fut.result().result
            ok = bool(res.success)
            status = int(getattr(res, "status", 0))
            msg = str(getattr(res, "message", ""))
        except Exception as e:
            self.events.append({"t": time.time(), "type": "ERROR", "msg": f"RTB result exception: {e}"})
            self._finish_rtb(False)
            return

        self.history.append({
            "task": "rtb_move_to_pid",
            "label": label,
            "success": ok,
            "status": status,
            "message": msg,
            "x": x,
            "y": y,
        })

        if not ok:
            self.events.append({"t": time.time(), "type": "ERROR", "msg": f"RTB failed at {label}: {msg}"})
            self._finish_rtb(False)
            return

        self._rtb_send_next_goal()

    # ======================================================================
    # Waypoints
    # ======================================================================
    def _load_waypoints(self):
        wps_raw = json.loads(self.get_parameter("waypoints_json").value)
        edges_raw = json.loads(self.get_parameter("waypoint_edges_json").value)
        self.waypoints: Dict[str, WP] = {
            k: WP(float(v["x"]), float(v["y"]), float(v.get("yaw", 0.0)))
            for k, v in wps_raw.items()
        }
        self.wp_edges: Dict[str, List[str]] = {k: list(v) for k, v in edges_raw.items()}
        self.wp_snap_max_dist = float(self.get_parameter("waypoint_snap_max_dist").value)

    def _dist2(self, ax, ay, bx, by):
        dx = ax - bx
        dy = ay - by
        return dx*dx + dy*dy
    
    # 현재 위치 (x, y)에서 가장 가까운 웨이포인트(waypoint)를 찾음 
    def _nearest_wp(self, x: float, y: float) -> Tuple[Optional[str], float]:
        best_name = None  # 현재까지 "가장 가까운 웨이포인트 이름"
        best_d2 = 1e18    # 현재까지 "가장 가까운 거리의 제곱" (아주 큰 값으로 시작)
        for name, wp in self.waypoints.items():
            d2 = self._dist2(x, y, wp.x, wp.y)   # 저장된 모든 waypoint들을 하나씩 확인
            if d2 < best_d2:
                best_d2 = d2
                best_name = name
        return best_name, math.sqrt(best_d2) if best_name is not None else 1e18

    def _dijkstra_wp_path(self, start: str, goal: str) -> List[str]:
        # 1) 준비물 세팅
        pq = [(0.0, start)]    # “지금까지 비용이 가장 작은 후보”를 빨리 꺼내기 위한 우선순위 큐
        dist = {start: 0.0}    # “start에서 각 노드까지 알고 있는 최소 비용”
        prev = {start: None}   # “최단경로로 올 때, 바로 이전 노드가 뭐였는지” (경로 복원용)

        while pq:
            # 2) 가장 싼 후보부터 하나씩 꺼내서 확장
            cost, u = heapq.heappop(pq)
            if u == goal:
                break
            if cost != dist.get(u, 1e18):
                continue
            # 3) u에서 갈 수 있는 이웃 v들을 검사
            for v in self.wp_edges.get(u, []):
                if v not in self.waypoints:
                    continue
                wu = self.waypoints[u]
                wv = self.waypoints[v]
                # 4) u→v로 가는 간선 비용(거리) 계산
                w = math.hypot(wv.x - wu.x, wv.y - wu.y)
                nd = cost + w   # start→u까지 비용(cost) + u→v 거리(w) = start→v까지 새 비용(nd)
                # 5) 더 짧게 갱신 가능하면 업데이트
                if nd < dist.get(v, 1e18):
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))

        if goal not in prev and goal != start:
            return []

        path = []
        cur = goal
        path.append(cur)
        while cur != start:
            cur = prev.get(cur)
            if cur is None:
                return []
            path.append(cur)
        path.reverse()
        return path

    def _pose_stamped(self, x: float, y: float, yaw: float) -> PoseStamped:
        ps = PoseStamped()
        ps.header.frame_id = self.map_frame
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x = float(x)
        ps.pose.position.y = float(y)
        qx, qy, qz, qw = quat_from_yaw(float(yaw))
        ps.pose.orientation.x = qx
        ps.pose.orientation.y = qy
        ps.pose.orientation.z = qz
        ps.pose.orientation.w = qw
        return ps

    # ======================================================================
    # TF / State publish
    # ======================================================================
    def _tf_pose(self):
        """map->base pose"""
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.base_frame,
                rclpy.time.Time(),                 # 최신(가능한) 변환
                timeout=Duration(seconds=0.2),      # ✅ 기다려줌
            )
            t = tf.transform.translation
            q = tf.transform.rotation
            yaw = yaw_from_quat(q.x, q.y, q.z, q.w)
            return True, float(t.x), float(t.y), float(yaw)
        except Exception as e:
            self.get_logger().warn(f"[TF] {self.map_frame}->{self.base_frame} lookup failed: {e}")
            return False, 0.0, 0.0, 0.0
        
    # 로봇의 현재 상태 스냅샷(자세/속도/배터리/미션/비전/안전/이벤트 등)을 /robot_state 같은 토픽에 계속 발행
    def publish_state(self):
        ok, x, y, yaw = self._tf_pose()

        msg = RobotState()
        msg.stamp = self.get_clock().now().to_msg()
        msg.robot_name = self.robot_name
        msg.system_state = self.system_state
        msg.pose_frame = self.map_frame
        msg.pose_x = x
        msg.pose_y = y
        msg.pose_yaw = yaw
        msg.vel_vx = float(self.latest_twist.linear.x)
        msg.vel_vy = float(self.latest_twist.linear.y)
        msg.vel_wz = float(self.latest_twist.angular.z)
        msg.battery_voltage = float(self.battery_voltage)
        msg.primary_id = int(self.primary_id)
        msg.lost_sec = float(self.lost_sec)
        msg.vision_json = self.vision_json
        msg.mission_id = self.mission_id
        msg.plan_json = self.plan_json
        msg.current_index = int(self.current_index)
        msg.queue_status = self.queue_status
        msg.history_json = json.dumps(self.history, ensure_ascii=False)
        msg.roe_ok = True
        msg.safe_backstop = True
        msg.max_speed = 0.6
        msg.last_violation = self.last_violation
        msg.events_json = json.dumps(self.events, ensure_ascii=False)

        self.state_pub.publish(msg)

    # ======================================================================
    # Topic: mission request/cancel
    # ======================================================================
    def _on_mission_request(self, msg: String):
        raw = msg.data.strip()
        if not raw:
            return

        try:
            plan = json.loads(raw)
        except Exception as e:
            self.get_logger().error(f"[mission_request] invalid JSON: {e}")
            return

        # (선택) plan["robot"]가 있으면 타겟 로봇만 수행
        tgt = str(plan.get("robot", "")).strip()
        if tgt and tgt != self.robot_name:
            return

        mid = str(plan.get("mission_id", "")).strip()
        if not mid:
            self._publish_result(False, "missing mission_id", mission_id="")
            return

        # ✅ 0) completed dedup: 이미 완료된 mission_id면 즉시 응답 + 무시
        if self._is_completed(mid):
            info = self.completed_missions.get(mid, {})
            msg_done = str(info.get("message", "already done"))
            self.events.append({"t": time.time(), "type": "INFO",
                                "msg": f"mission ignored (already done): id={mid}"})
            # 이미 완료된 건 success=True로 응답하는 편이 재전송을 멈추게 함
            self._publish_result(True, f"already done: {msg_done}", mission_id=mid)
            return

        # ✅ 1) 같은 mission_id가 “현재 실행 중”이면 즉시 응답 + 무시
        if self._mission_active and self.mission_id == mid:
            self.events.append({"t": time.time(), "type": "INFO",
                                "msg": f"mission ignored (already running): id={mid}"})
            self._publish_result(True, "already running", mission_id=mid)
            return

        # RTB 중이면 거절
        if self._rtb_in_progress:
            self._publish_result(False, "RTB already in progress", mission_id=str(plan.get("mission_id", "")))
            return

        # busy면 거절
        if self._mission_active or self.system_state != "IDLE":
            self._publish_result(False, "system busy", mission_id=str(plan.get("mission_id", "")))
            return

        # 배터리 저전압이면 미션 거절 + RTB 시작
        if self._battery_is_low():
            self._rtb_in_progress = True
            self._low_v_latched = True
            self.events.append({"t": time.time(), "type": "WARN",
                                "msg": f"mission blocked: battery low -> RTB (V={self.battery_voltage:.2f}V)"})
            self._start_rtb_future_chain()
            self._publish_result(False, f"battery low -> RTB started (V={self.battery_voltage:.2f}V)",
                                 mission_id=str(plan.get("mission_id", "")))
            return

        # mission init
        self._mission_active = True
        self.system_state = "RUNNING"
        self.queue_status = "running"
        self.current_index = 0
        self.mission_id = mid
        self.plan_json = raw
        self.history = []
        self._steps = list(plan.get("steps", []))
        self._step_i = -1
        self.events.append({"t": time.time(), "type": "INFO",
                            "msg": f"mission start: id={self.mission_id} steps={len(self._steps)}"})
        self._dispatch_next_step()

    def _on_mission_cancel(self, msg: String):
        raw = msg.data.strip()
        if not raw:
            return
        try:
            req = json.loads(raw)
        except Exception:
            req = {"mission_id": raw}

        tgt = str(req.get("robot", "")).strip()
        mid = str(req.get("mission_id", "")).strip()

        if tgt and tgt != self.robot_name:
            return
        if mid and self.mission_id and mid != self.mission_id:
            return
        if not self._mission_active:
            return

        self.events.append({"t": time.time(), "type": "WARN",
                            "msg": f"mission cancel requested: id={self.mission_id} action={self._active_action}"})

        # cancel currently active action if possible
        if self._active_action == "move" and self._move_goal_handle is not None:
            try:
                fut = self._move_goal_handle.cancel_goal_async()
                fut.add_done_callback(lambda f: self._finish_mission(False, "canceled"))
            except Exception:
                self._finish_mission(False, "canceled")
            return

        if self._active_action == "follow" and self._follow_goal_handle is not None:
            try:
                fut = self._follow_goal_handle.cancel_goal_async()
                fut.add_done_callback(lambda f: self._finish_mission(False, "canceled"))
            except Exception:
                self._finish_mission(False, "canceled")
            return

        self._finish_mission(False, "canceled")

    # ======================================================================
    # Mission dispatcher (callback-based state machine)
    # ======================================================================
    def _dispatch_next_step(self):
        if not self._mission_active:
            return

        self._step_i += 1
        self.current_index = self._step_i

        if self._step_i >= len(self._steps):
            self._finish_mission(True, "mission done")
            return

        step = self._steps[self._step_i]
        task = str(step.get("task", "")).strip()

        if task == "move_to":
            gx = float(step.get("x", 0.0))
            gy = float(step.get("y", 0.0))
            gyaw = float(step.get("yaw", 0.0))
            use_wp = bool(step.get("use_waypoints", True))
            timeout_sec = float(step.get("timeout_sec", self.get_parameter("final_timeout_sec").value))
            self._start_move_to(gx, gy, gyaw, use_waypoints=use_wp, timeout_override=timeout_sec)
            return

        if task == "follow_aruco":
            marker_id = int(step.get("marker_id", 0))
            timeout_sec = float(step.get("timeout_sec", 120.0))
            if marker_id <= 0:
                self._finish_mission(False, f"follow_aruco invalid marker_id: {marker_id}")
                return
            self._start_follow_aruco(marker_id, timeout_sec=timeout_sec)
            return

        self._finish_mission(False, f"unknown task: {task}")


    # ---------------- move_to: via waypoints -> subgoal queue -> callbacks ----------------
    def _start_move_to(self, goal_x: float, goal_y: float, goal_yaw: float,
                       use_waypoints: bool, timeout_override: float):
        if not self.move_client.wait_for_server(timeout_sec=0.5):
            self._finish_mission(False, "MoveToPID server not available")
            return

        self._subgoal_queue = []

        if not use_waypoints:
            ps = self._pose_stamped(goal_x, goal_y, goal_yaw)
            self._subgoal_queue.append((ps, float(timeout_override), "direct"))
            self._send_next_subgoal()
            return

        ok, rx, ry, _ = self._tf_pose()
        if not ok:
            self._finish_mission(False, "TF pose unavailable")
            return

        s_wp, s_d = self._nearest_wp(rx, ry)
        g_wp, g_d = self._nearest_wp(goal_x, goal_y)

        wp_timeout = float(self.get_parameter("wp_timeout_sec").value)
        final_timeout = float(self.get_parameter("final_timeout_sec").value)

        if s_wp is None or g_wp is None or s_d > self.wp_snap_max_dist or g_d > self.wp_snap_max_dist:
            self.events.append({"t": time.time(), "type": "WARN",
                                "msg": f"WP snap failed: start_d={s_d:.2f}, goal_d={g_d:.2f} -> direct"})
            ps = self._pose_stamped(goal_x, goal_y, goal_yaw)
            self._subgoal_queue.append((ps, float(timeout_override), "direct_due_to_snap_fail"))
            self._send_next_subgoal()
            return

        chain = self._dijkstra_wp_path(s_wp, g_wp)
        if not chain:
            self._finish_mission(False, f"No WP path {s_wp}->{g_wp}")
            return

        self.events.append({"t": time.time(), "type": "INFO", "msg": f"WP path {s_wp}->{g_wp}: {chain}"})

        for name in chain:
            wp = self.waypoints[name]
            ps = self._pose_stamped(wp.x, wp.y, wp.yaw)
            self._subgoal_queue.append((ps, wp_timeout, f"wp:{name}"))

        ps = self._pose_stamped(goal_x, goal_y, goal_yaw)
        self._subgoal_queue.append((ps, final_timeout, "final"))

        self._send_next_subgoal()

    def _send_next_subgoal(self):
        if not self._mission_active:
            return

        if not self._subgoal_queue:
            self._active_action = ""
            self._move_goal_handle = None
            self._dispatch_next_step()
            return

        ps, timeout_sec, label = self._subgoal_queue.pop(0)

        goal = MoveToPID.Goal()
        goal.target = ps
        goal.timeout_sec = float(timeout_sec)

        self._active_action = "move"
        self.events.append({"t": time.time(), "type": "INFO",
                            "msg": f"send MoveToPID ({label}) x={ps.pose.position.x:.2f},y={ps.pose.position.y:.2f},t={timeout_sec:.1f}s"})

        send_fut = self.move_client.send_goal_async(goal)
        send_fut.add_done_callback(lambda fut: self._on_move_goal_response(fut, label, ps, timeout_sec))

    def _on_move_goal_response(self, fut, label: str, ps: PoseStamped, timeout_sec: float):
        if not self._mission_active:
            return
        try:
            gh = fut.result()
        except Exception as e:
            self._finish_mission(False, f"MoveToPID goal response error: {e}")
            return

        if not gh.accepted:
            self._finish_mission(False, f"MoveToPID goal rejected ({label})")
            return

        self._move_goal_handle = gh
        res_fut = gh.get_result_async()
        res_fut.add_done_callback(lambda rfut: self._on_move_result(rfut, label, ps, timeout_sec))

    def _on_move_result(self, fut, label: str, ps: PoseStamped, timeout_sec: float):
        if not self._mission_active:
            return
        try:
            res = fut.result().result
            ok = bool(res.success)
            status = int(getattr(res, "status", 0))
            msg = str(getattr(res, "message", ""))
        except Exception as e:
            self._finish_mission(False, f"MoveToPID result exception: {e}")
            return

        self.history.append({
            "task": "move_to_pid",
            "label": label,
            "success": ok,
            "status": status,
            "message": msg,
            "timeout_sec": float(timeout_sec),
            "x": float(ps.pose.position.x),
            "y": float(ps.pose.position.y),
        })

        if not ok:
            self._finish_mission(False, f"move_to failed at {label}: {msg}")
            return

        self._send_next_subgoal()

    # ---------------- follow_aruco callbacks ----------------
    def _start_follow_aruco(self, marker_id: int, timeout_sec: float):
        if not self.follow_client.wait_for_server(timeout_sec=0.5):
            self._finish_mission(False, "FollowAruco server not available")
            return

        goal = FollowAruco.Goal()
        goal.marker_id = int(marker_id)
        if hasattr(goal, "timeout_sec"):
            goal.timeout_sec = float(timeout_sec)

        # ✅ 클라이언트 가드 타임아웃(무조건 동작)
        self._follow_deadline = time.time() + float(timeout_sec)
        if self._follow_timeout_timer is not None:
            self._follow_timeout_timer.cancel()
            self._follow_timeout_timer = None
        self._follow_timeout_timer = self.create_timer(
            0.2,  # 5Hz 체크
            self._follow_timeout_watchdog,
            callback_group=self.cb_group,
        )

        self._active_action = "follow"
        self.events.append({"t": time.time(), "type": "INFO",
                            "msg": f"send FollowAruco marker_id={marker_id} timeout={timeout_sec:.1f}s"})

        send_fut = self.follow_client.send_goal_async(goal)
        send_fut.add_done_callback(lambda fut: self._on_follow_goal_response(fut, marker_id, timeout_sec))

    def _on_follow_goal_response(self, fut, marker_id: int, timeout_sec: float):
        if not self._mission_active:
            return
        try:
            gh = fut.result()
        except Exception as e:
            self._finish_mission(False, f"FollowAruco goal response error: {e}")
            return

        if not gh.accepted:
            self._finish_mission(False, "FollowAruco goal rejected")
            return

        self._follow_goal_handle = gh
        res_fut = gh.get_result_async()
        res_fut.add_done_callback(lambda rfut: self._on_follow_result(rfut, marker_id, timeout_sec))

    def _on_follow_result(self, fut, marker_id: int, timeout_sec: float):
        if self._follow_timeout_timer is not None:
            self._follow_timeout_timer.cancel()
            self._follow_timeout_timer = None

        if not self._mission_active:
            return
        try:
            res = fut.result().result
            ok = bool(res.success)
            msg = str(getattr(res, "message", ""))
        except Exception as e:
            self._finish_mission(False, f"FollowAruco result exception: {e}")
            return

        self.history.append({
            "task": "follow_aruco",
            "success": ok,
            "marker_id": int(marker_id),
            "timeout_sec": float(timeout_sec),
            "message": msg,
            "cause": "ok" if ok else "server_failed",
        })

        if not ok:
            self._finish_mission(False, f"follow_aruco failed: {msg}")
            return

        self._active_action = ""
        self._follow_goal_handle = None
        self._dispatch_next_step()

    def _follow_timeout_watchdog(self):
        # follow가 아닐 때는 타이머 정리
        if self._active_action != "follow":
            if self._follow_timeout_timer is not None:
                self._follow_timeout_timer.cancel()
                self._follow_timeout_timer = None
            return

        if time.time() < self._follow_deadline:
            return

        # ✅ 타임아웃 발생: cancel 시도
        self.events.append({"t": time.time(), "type": "WARN",
                            "msg": "FollowAruco timeout -> cancel goal"})

        if self._follow_goal_handle is not None:
            try:
                fut = self._follow_goal_handle.cancel_goal_async()
                fut.add_done_callback(lambda f: self._finish_mission(False, "follow_aruco timeout"))
            except Exception as e:
                self._finish_mission(False, f"follow_aruco timeout (cancel failed): {e}")
        else:
            # 아직 goal_handle을 못 받은 상태면 그냥 실패 처리
            self._finish_mission(False, "follow_aruco timeout (no goal_handle yet)")

    # ======================================================================
    # Mission finish/result
    # ======================================================================
    def _finish_mission(self, success: bool, message: str):
        if self._follow_timeout_timer is not None:
            self._follow_timeout_timer.cancel()
            self._follow_timeout_timer = None

        mid = str(self.mission_id)

        # ✅ 완료 캐시에 기록 (중복 실행 방지 핵심)
        if mid:
            self.completed_missions[mid] = {
                "t": float(time.time()),
                "success": bool(success),
                "message": str(message),
            }
            self._prune_completed()

        self._mission_active = False
        self._steps = []
        self._step_i = -1
        self._subgoal_queue = []
        self._active_action = ""
        self._move_goal_handle = None
        self._follow_goal_handle = None

        self.queue_status = "done" if success else "error"
        self.system_state = "IDLE" if success else "ERROR"

        self.events.append({"t": time.time(),
                            "type": "INFO" if success else "ERROR",
                            "msg": f"mission end: id={mid} success={success} msg={message}"})

        self._publish_result(success, message, mission_id=mid)

        # 운영 편의상 실패해도 IDLE 복귀(원하면 ERROR 유지 가능)
        if not success:
            self.system_state = "IDLE"
            self.queue_status = "idle"
            self.current_index = -1

        # 현재 미션 정보 초기화(선택)
        self.mission_id = ""
        self.plan_json = ""

    def _publish_result(self, success: bool, message: str, mission_id: str):
        payload = {
            "robot": self.robot_name,
            "mission_id": str(mission_id),
            "success": bool(success),
            "message": str(message),
            "history": self.history,
            "t": float(time.time()),
        }
        out = String()
        out.data = json.dumps(payload, ensure_ascii=False)
        self.result_pub.publish(out)


def main():
    rclpy.init()
    node = PinkySystem1()

    # 콜백+액션 done_callback 많아서 MultiThreaded 권장
    from rclpy.executors import MultiThreadedExecutor
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()



### 실행 예시
# ros2 launch controller controller.launch.py 

# ros2 run controller controller_domainbridge \
#   --ros-args -p battery_low_voltage_threshold:=7.8


# ✅ 사용 예시
# 0) 도메인 브릿지 연결
# ros2 run domain_bridge domain_bridge ~/jazzy_ws/src/pinky_structure/config/domain_bridge.yaml --ros-args --log-level debug

# 1) 미션 요청 (System2 → System1)
# ros2 topic pub /traffic/mission_request std_msgs/String "
# data: '{
#   \"robot\":\"pinky1\",
#   \"mission_id\":\"m_test_001\",
#   \"steps\":[
#     {\"task\":\"move_to\",\"x\":0.269,\"y\":-1.141,\"yaw\":3.082,\"use_waypoints\":true},
#     {\"task\":\"follow_aruco\",\"marker_id\":600,\"timeout_sec\":120.0}
#   ]
# }'
# "

# 2) 미션 취소 (선택)
# ros2 topic pub /traffic/mission_cancel std_msgs/String "data: '{\"robot\":\"pinky1\",\"mission_id\":\"m_test_001\"}'"

# 3) 결과 수신
# ros2 topic echo /pinky1/mission_result