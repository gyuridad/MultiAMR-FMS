#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
import heapq
import time
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple, Set, Callable

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from pinky_interfaces.msg import RobotState


# --------------------------- Data Models ---------------------------

@dataclass
class Pose2D:
    x: float
    y: float
    yaw: float = 0.0


# --------------------------- Lock Manager (Option A + C) ---------------------------

class NodeLockManager:
    """
    Option A + C:
      - 모든 노드를 critical로 간주
      - "다음 노드 1개"만 lock 잡는다 (next node lock)
      - lock은 node 단위만 관리
    """
    def __init__(self, all_nodes: Set[str]):
        self.all_nodes = set(all_nodes)
        self.locks: Dict[Tuple, Dict[str, Any]] = {}  # key=("node", node) -> {"robot","mission_id","ts"}

    def _k(self, node: str) -> Tuple:
        return ("node", node)

    def is_locked_by_other(self, node: str, robot: str) -> bool:
        k = self._k(node)
        if k not in self.locks:
            return False
        return str(self.locks[k].get("robot", "")) != str(robot)

    def try_lock_node(self, robot: str, mission_id: str, node: str) -> bool:
        k = self._k(node)
        v = self.locks.get(k)
        if v and str(v.get("robot", "")) != str(robot):
            return False
        self.locks[k] = {"robot": robot, "mission_id": mission_id, "ts": time.time()}
        return True

    def release_node_of_owned(self, robot: str, node: str) -> bool:
        k = self._k(node)
        v = self.locks.get(k)
        if not v:
            return False
        if str(v.get("robot", "")) != str(robot):
            return False
        self.locks.pop(k, None)
        return True

    def release_all_for_mission(self, robot: str, mission_id: str) -> int:
        to_del = []
        for k, v in self.locks.items():
            if str(v.get("robot", "")) == str(robot) and str(v.get("mission_id", "")) == str(mission_id):
                to_del.append(k)
        for k in to_del:
            self.locks.pop(k, None)
        return len(to_del)

    def release_occupy_if_owned(self, robot: str, node: str) -> bool:
        k = self._k(node)
        v = self.locks.get(k)
        if not v:
            return False
        if str(v.get("robot", "")) != str(robot):
            return False
        # occupy 락만 해제
        if not str(v.get("mission_id", "")).startswith("occupy__"):
            return False
        self.locks.pop(k, None)
        return True

    def dump(self) -> Dict[str, Any]:
        out = []
        for k, v in self.locks.items():
            out.append({"type": "node", "node": k[1], **v})
        return {"locks": out}


# --------------------------- Waypoint Graph ---------------------------

class WaypointGraph:
    def __init__(self, waypoints: Dict[str, Pose2D], edges: Dict[str, List[str]]):
        self.waypoints = waypoints
        self.edges = edges

    def dist(self, a: str, b: str) -> float:
        pa, pb = self.waypoints[a], self.waypoints[b]
        return math.hypot(pa.x - pb.x, pa.y - pb.y)

    def dijkstra(self, start: str, goal: str) -> List[str]:
        """기본 최단경로(락 회피 없음 버전)"""
        return self.dijkstra_avoiding_locked(
            start=start,
            goal=goal,
            is_blocked_fn=lambda _n: False,
            allow_nodes={start, goal},
        )

    def dijkstra_avoiding_locked(
        self,
        start: str,
        goal: str,
        is_blocked_fn: Callable[[str], bool],
        allow_nodes: Optional[Set[str]] = None,
    ) -> List[str]:
        """
        ✅ 회피 다익스트라:
          - is_blocked_fn(node)==True 인 노드는 통과 금지
          - allow_nodes에 포함된 노드는 blocked여도 예외 허용(보통 start/goal)
        """
        if start == goal:
            return [start]
        if start not in self.waypoints or goal not in self.waypoints:
            return []

        if allow_nodes is None:
            allow_nodes = set()

        def blocked(n: str) -> bool:
            if n in allow_nodes:
                return False
            return bool(is_blocked_fn(n))

        if blocked(start):
            return []

        pq = [(0.0, start)]
        dist = {start: 0.0}
        prev = {start: None}

        while pq:
            cost, u = heapq.heappop(pq)
            if u == goal:
                break
            if cost != dist.get(u, 1e18):
                continue

            for v in self.edges.get(u, []):
                if v not in self.waypoints:
                    continue
                if blocked(v):
                    continue
                nd = cost + self.dist(u, v)
                if nd < dist.get(v, 1e18):
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))

        if goal not in prev:
            return []

        path = []
        cur = goal
        while cur is not None:
            path.append(cur)
            cur = prev.get(cur)
        path.reverse()
        return path


# --------------------------- Traffic Manager Node ---------------------------

class TrafficManagerNode(Node):
    """
    ✅ System1은 그대로, TM만 "옛날 방식" 단일 토픽 출력
    (입력) Orchestrator -> TM:
      - /traffic/tm_request (std_msgs/String JSON)
        {"type":"GOAL","robot":"pinky1","mission_id":"...","goal_wp":"A",...}

    (출력) TM -> System1: (단일 토픽)
      - /traffic/mission_request (std_msgs/String JSON)
        {"robot":"pinky1","mission_id":"...","steps":[...]}

    (취소 출력) TM -> System1: (단일 토픽)
      - /traffic/mission_cancel (std_msgs/String JSON)
        {"robot":"pinky1","mission_id":"..."}
    """

    def __init__(self):
        super().__init__("traffic_manager")

        # ---------------- Params ----------------
        self.declare_parameter("robots", ["pinky1", "pinky2", "pinky3"])
        self.declare_parameter("tm_request_topic", "/traffic/tm_request")
        self.declare_parameter("tm_cancel_topic", "/traffic/tm_cancel")
        self.declare_parameter("mission_request_topic", "/traffic/mission_request")
        self.declare_parameter("mission_cancel_topic", "/traffic/mission_cancel")

        # ✅ NEW: TM 결과 토픽
        self.declare_parameter("tm_result_topic", "/traffic/mission_result")

        self.declare_parameter("robot_state_suffix", "/robot_state")
        self.declare_parameter("robot_result_suffix", "/mission_result")

        self.declare_parameter("snap_accept_dist_m", 0.10)
        self.declare_parameter("keep_current_node_locked", False)
        self.declare_parameter("retry_wait_sec", 0.5)
        self.declare_parameter("tick_period_sec", 0.1)

        # ✅ P노드(대기/충전) 좌표를 파라미터로도 바꿀 수 있게(기본값은 네 Orchestrator station_to_pose 쪽 값)
        self.declare_parameter("p1_pose", json.dumps({"x": 0.0, "y": 0.0, "yaw": 0.0}))
        self.declare_parameter("p2_pose", json.dumps({"x": -0.0096, "y": -0.3430, "yaw": -0.0580}))
        self.declare_parameter("p3_pose", json.dumps({"x": 0.0130, "y": -0.6830, "yaw": 0.0354}))

        self.robots: List[str] = list(self.get_parameter("robots").value)
        self.snap_accept_dist_m = float(self.get_parameter("snap_accept_dist_m").value)
        self.keep_current_node_locked = bool(self.get_parameter("keep_current_node_locked").value)
        self.retry_wait_sec = float(self.get_parameter("retry_wait_sec").value)
        tick_period = float(self.get_parameter("tick_period_sec").value)

        self.req_fmt = str(self.get_parameter("mission_request_topic").value)
        self.cancel_fmt = str(self.get_parameter("mission_cancel_topic").value)
        self.tm_result_topic = str(self.get_parameter("tm_result_topic").value)

        # ---------------- Graph ----------------
        waypoints = {
            "A": Pose2D(0.15891485402430086,  0.006662365163616427,  1.5989778609767642),
            "B": Pose2D(0.0982715949828098,   0.4152183361819458,    1.5588626874735612),
            "C": Pose2D(0.3273481339916138,   0.3808828616634004,   -1.65683453511985),
            "D": Pose2D(0.31136767117452996, -0.03364462326598874,  -1.6353521361867218),
            "E": Pose2D(0.32810263277098783, -0.37014285859706564,  -1.6203311273233227),
            "F": Pose2D(0.3225842275905132,  -0.6840307808082434,   -1.5742126342716196),
            "G": Pose2D(0.27105983124661476, -1.0283068390452972,   -1.6156553100392443),
            "H": Pose2D(0.06957616919930265, -0.9729490000804174,    1.5421559861678504),
            "I": Pose2D(0.153143961865185,   -0.6665573416040188,    1.5772812791268003),
            "J": Pose2D(0.15761047209156467, -0.3254817351067633,    1.5633740768497586),
        }

        # ✅ P노드 로드
        def _load_pose(param_name: str) -> Pose2D:
            try:
                d = json.loads(str(self.get_parameter(param_name).value))
                return Pose2D(float(d.get("x", 0.0)), float(d.get("y", 0.0)), float(d.get("yaw", 0.0)))
            except Exception:
                return Pose2D(0.0, 0.0, 0.0)

        p1 = _load_pose("p1_pose")
        p2 = _load_pose("p2_pose")
        p3 = _load_pose("p3_pose")

        waypoints["P1"] = Pose2D(p1.x, p1.y, p1.yaw)
        waypoints["P2"] = Pose2D(p2.x, p2.y, p2.yaw)
        waypoints["P3"] = Pose2D(p3.x, p3.y, p3.yaw)

        # 엣지: 기존 + P노드 연결
        # 연결 원칙(현장/좌표 기반):
        # - P1은 A 근처 대기라고 가정 -> P1<->A
        # - P2는 J 근처 대기라고 가정 -> P2<->J
        # - P3는 I 근처 대기라고 가정 -> P3<->I
        edges = {
            "A": ["B", "D", "J", "P1"],
            "B": ["A", "C"],
            "C": ["B", "D"],
            "D": ["A", "C", "E"],
            "E": ["D", "F", "J"],
            "F": ["E", "G", "I"],
            "G": ["F", "H"],
            "H": ["G", "I"],
            "I": ["F", "H", "J", "P3"],
            "J": ["A", "E", "I", "P2"],

            # 신규 P노드
            "P1": ["A"],
            "P2": ["J"],
            "P3": ["I"],
        }

        self.graph = WaypointGraph(waypoints, edges)
        self.lock_mgr = NodeLockManager(all_nodes=set(waypoints.keys()))

        # ---------------- Per-robot internal states ----------------
        self.robot_current_wp: Dict[str, str] = {r: "A" for r in self.robots}
        self.robot_wp_quality: Dict[str, float] = {r: 1e9 for r in self.robots}

        # ✅ 추가: 로봇 pose 캐시
        self.robot_pose_xy: Dict[str, Tuple[float, float]] = {r: (0.0, 0.0) for r in self.robots}
        self.robot_pose_ok: Dict[str, bool] = {r: False for r in self.robots}

        self.ctx: Dict[str, Dict[str, Any]] = {}
        self.robot_busy: Dict[str, bool] = {r: False for r in self.robots}

        # pubs
        self.mission_req_pub = self.create_publisher(String, self.req_fmt, 10)
        self.mission_cancel_pub = self.create_publisher(String, self.cancel_fmt, 10)

        # ✅ NEW: big mission 결과 pub
        self.tm_result_pub = self.create_publisher(String, self.tm_result_topic, 10)

        # subs
        self.tm_req_sub = self.create_subscription(
            String, str(self.get_parameter("tm_request_topic").value),
            self._on_tm_request, 10
        )
        self.tm_cancel_sub = self.create_subscription(
            String, str(self.get_parameter("tm_cancel_topic").value),
            self._on_tm_cancel, 10
        )

        state_suffix = str(self.get_parameter("robot_state_suffix").value)
        result_suffix = str(self.get_parameter("robot_result_suffix").value)

        for r in self.robots:
            st_topic = f"/{r}{state_suffix}"
            rs_topic = f"/{r}{result_suffix}"
            self.create_subscription(
                RobotState, st_topic,
                lambda msg, robot=r: self._on_robot_state(robot, msg),
                10
            )
            self.create_subscription(
                String, rs_topic,
                lambda msg, robot=r: self._on_robot_result(robot, msg),
                10
            )
            self.get_logger().info(f"Subscribed: {st_topic} , {rs_topic}")

        self.create_timer(tick_period, self._tick)

        self.get_logger().info(
            f"[TrafficManagerTopic] ready. tm_req={self.get_parameter('tm_request_topic').value} "
            f"-> out(mission_request)={self.req_fmt}"
        )
        self.get_logger().info(
            f"[P-NODES] P1=({waypoints['P1'].x:.3f},{waypoints['P1'].y:.3f}) "
            f"P2=({waypoints['P2'].x:.3f},{waypoints['P2'].y:.3f}) "
            f"P3=({waypoints['P3'].x:.3f},{waypoints['P3'].y:.3f}) "
            f"edges: P1<->A, P2<->J, P3<->I"
        )
    
    # ----------------------------------------------------------------------
    # ✅ publish TM big mission result (ONLY big_id)
    # ----------------------------------------------------------------------
    def _publish_tm_result(self, robot: str, big_id: str, ok: bool, goal_wp: str, reason: str = ""):
        payload = {
            "robot": str(robot),
            "mission_id": str(big_id),   # ✅ big_id 그대로
            "ok": bool(ok),
            "goal_wp": str(goal_wp),
            "reason": str(reason),
            "ts": float(time.time()),
        }
        self.tm_result_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))


    # ----------------------------------------------------------------------
    # ✅ NEW: nearest wp from pose
    # ----------------------------------------------------------------------
    # “현재 로봇 위치 (x,y)가 주어졌을 때, 웨이포인트들 중에서 가장 가까운 WP 이름과 그 거리(m)를 찾아주는 함수”
    def _nearest_wp_from_pose(self, x: float, y: float) -> Tuple[Optional[str], float]:
        best = None
        best_d2 = 1e18
        for name, p in self.graph.waypoints.items():  # 그래프에 등록된 모든 웨이포인트(A,B,C,...)를 하나씩 확인
            d2 = (p.x - x) ** 2 + (p.y - y) ** 2      # 현재 위치 (x,y)와 WP 좌표 (p.x,p.y) 사이의 거리의 제곱을 계산
            if d2 < best_d2:
                best_d2 = d2
                best = name
        return best, math.sqrt(best_d2) if best else 1e18   # 가장 가까운 WP 이름과 **진짜 거리(m)**를 돌려줌

    # ---------------- RobotState: current_wp 자동 갱신 (pose 기반) ----------------
    def _on_robot_state(self, robot: str, msg: RobotState):
        """
        이 함수는 **로봇이 보내는 RobotState(현재 위치 정보)**를 받을 때마다,
        로봇의 현재 (x,y)를 저장하고
        그 위치가 어떤 웨이포인트(WP)에 가장 가까운지 계산한 뒤
        충분히 가까우면(snap_accept_dist_m 이내면)
        “로봇이 지금 WP에 도착했다”고 보고 TM 내부의 current_wp를 업데이트하는 역할
        """
        # 1) pose 저장
        try:
            x = float(getattr(msg, "pose_x", 0.0))
            y = float(getattr(msg, "pose_y", 0.0))
        except Exception:
            return

        self.robot_pose_xy[robot] = (x, y)
        self.robot_pose_ok[robot] = True

        # 2) TM 내부에서 nearest 계산
        wp, d = self._nearest_wp_from_pose(x, y)
        if wp and (wp in self.graph.waypoints) and d <= self.snap_accept_dist_m:
            prev = self.robot_current_wp.get(robot, "A")   # robot의 current_wp 가져와서 이전 포인트로 저장
            self.robot_current_wp[robot] = wp              # robot의 current_wp 갱신
            self.robot_wp_quality[robot] = d
            if prev != wp:                   # 이전 wp와 달라질 때만 로그 찍어서 “A -> B로 바뀌었네” 같은 변화만 보여줌
                self.get_logger().info(f"[{robot}] current_wp update(by pose): {prev} -> {wp} (d={d:.2f}m)")

    # ======================================================================
    # 상위 입력: tm_request
    # ======================================================================
    def _on_tm_request(self, msg: String):
        """
        /traffic/tm_request로 “미션 요청”이 들어왔을 때, TM이 그 요청을 받아서
        요청이 맞는지 검사하고
        지금 로봇이 바쁜지 확인하고
        PLAN이면 그대로 System1으로 전달하고
        GOAL이면 “큰 미션(big mission)” 상태(ctx)를 만들고 첫 hop을 보내는
        “미션 접수 창구” 역할
        """
        raw = msg.data.strip()
        if not raw:
            return
        try:
            req = json.loads(raw)
        except Exception as e:
            self.get_logger().error(f"[tm_request] invalid JSON: {e}")
            return

        r = str(req.get("robot", "")).strip()
        mid = str(req.get("mission_id", "")).strip()
        rtype = str(req.get("type", "")).strip().upper()

        if not r or r not in self.robots:
            self.get_logger().error(f"[tm_request] invalid robot: {r}")
            return
        if not mid:
            self.get_logger().error("[tm_request] missing mission_id")
            return

        if self.robot_busy.get(r, False):
            self.get_logger().warn(f"[{r}] busy. reject tm_request big_id={mid}")
            return

        # ✅ 새 미션 시작 전: 현재 노드에 걸려있는 occupy 락이 있으면 해제
        cur_wp = self.robot_current_wp.get(r, "A")
        if cur_wp in self.graph.waypoints:
            if self.lock_mgr.release_occupy_if_owned(r, cur_wp):
                self.get_logger().info(f"[{r}] released occupy lock at {cur_wp} (before new mission)")

        # (B) PLAN 포워딩
        # PLAN은 “LLM용으로도 쓸 수 있게 열어둔 것”에 가깝고, 현재 Orchestrator 흐름에서는 아직 미사용(확장 슬롯)
        if rtype == "PLAN" or ("steps" in req and "goal_wp" not in req):
            plan = {
                "robot": r,
                "mission_id": mid,
                "steps": list(req.get("steps", [])),
            }
            self._publish_mission_request(r, plan)
            self.robot_busy[r] = True
            self.ctx[r] = {
                "mode": "PLAN",
                "big_id": mid,
                "inflight": True,
                "current_hop_id": mid,
            }
            self.get_logger().info(f"[{r}] forwarded PLAN mission_id={mid} steps={len(plan['steps'])}")
            return

        # (A) GOAL 기반
        goal_wp = str(req.get("goal_wp", "")).strip()
        if goal_wp not in self.graph.waypoints:
            self.get_logger().error(f"[{r}] invalid goal_wp={goal_wp}")
            return

        # ✅ NEW: GOAL 수락 직전 start snap(by pose)
        if self.robot_pose_ok.get(r, False):
            x, y = self.robot_pose_xy[r]
            wp, d = self._nearest_wp_from_pose(x, y)
            if wp and d <= self.snap_accept_dist_m:
                prev = self.robot_current_wp.get(r, "A")
                self.robot_current_wp[r] = wp
                self.robot_wp_quality[r] = d
                if prev != wp:
                    self.get_logger().info(f"[{r}] start snap(by pose) {prev}->{wp} d={d:.2f}m")

        ctx = {
            "mode": "GOAL",
            "big_id": mid,
            "goal_wp": goal_wp,
            "final_yaw": float(req.get("final_yaw", 0.0)),
            "do_follow": bool(req.get("do_follow_aruco", False)),
            "marker_id": int(req.get("marker_id", 0)),
            "timeout_sec": float(req.get("timeout_sec", 35.0)),
            "inflight": False,
            "current_hop_id": "",
            "last_hop": None,
            "seq": 0,
            "next_retry_time": 0.0,
            "final_pose": req.get("final_pose", None),
            "final_use_waypoints": bool(req.get("final_use_waypoints", False)),
        }
        self.ctx[r] = ctx
        self.robot_busy[r] = True
        self.get_logger().info(f"[{r}] GOAL accepted: big_id={mid} goal={goal_wp}")
        self._try_send_next_step(r)

        # 상위 제어(Orchestrator/운영자/MES/LLM Supervisor 등) 입장에서 TM에 미션을 넣는 방식이 2가지라고 보면 돼요.
        #         GOAL 요청: “어디로 가”만 던지고, 경로 쪼개기/락 회피/다익스트라는 TM이 알아서 함
        #         PLAN 요청: “이 순서대로 이렇게 움직여”를 상위가 이미 결정했고, TM은 그걸 그대로 System1에 전달만 함
        #     1) GOAL 요청이 들어오는 상황 (상위는 목표만 안다)
        #         상황 A: 오케스트레이터의 일반 작업 지시 (가장 흔함)
        #         작업 흐름: “pinky1이 LOADING_ZONE(B)로 가서 작업하고, 다음 QC_ZONE(C)로 가라”
        #         상위(Orchestrator)는 ‘목표 스테이션’만 결정하고,
        #         경로는 TM이 lock 상태 보면서 안전하게 고르길 원함.
        #         예)
        #             {"type":"GOAL","robot":"pinky1","mission_id":"J100__work","goal_wp":"B"}
        #             TM이 알아서
        #                 현재 WP가 A면: A→B
        #                 만약 B가 잠겨있으면: 다른 우회 경로 시도(가능하면)
        #                 ✅ 핵심: 상위는 “B로 가”만 말한다. “A→D→…” 같은 경로는 TM이 결정.
        #     2) PLAN 요청이 들어오는 상황 (상위가 경로/스텝을 ‘이미 확정’했다)
        #             PLAN은 쉽게 말해 “TM 너는 교통정리/락이 아니라 전달자 역할만 해” 모드예요.
        #         상황 D: 상위가 “정해진 순서대로” 가야 하는 공정 동선(고정 루트)
        #         예를 들어 현장에서 “안전상/품질상” 이유로
        #             “A에서 C로 갈 때 무조건 A→B→C로 가라”
        #             “D로 새는 길은 위험 구역이라 금지”
        #             같은 규칙이 있으면 상위가 경로를 고정해버리는 경우가 있어요.
        #         예)
        #             {
        #             "type":"PLAN",
        #             "robot":"pinky1",
        #             "mission_id":"J200__fixedroute",
        #             "steps":[
        #                 {"task":"move_to","x":...,"y":...,"yaw":...,"use_waypoints":false,"timeout_sec":120},
        #                 {"task":"move_to","x":...,"y":...,"yaw":...,"use_waypoints":false,"timeout_sec":120}
        #             ]
        #             }
        #         TM은 이 steps를 그대로 /traffic/mission_request로 내보냄.
        #         ✅ 핵심: 상위가 “스텝 시퀀스”를 이미 만들었다.

    # ======================================================================
    # 상위 취소: tm_cancel
    # ======================================================================
    def _on_tm_cancel(self, msg: String):
        raw = msg.data.strip()
        if not raw:
            return
        try:
            req = json.loads(raw)
        except Exception:
            req = {"mission_id": raw}

        r = str(req.get("robot", "")).strip()
        big_id = str(req.get("mission_id", "")).strip()
        if not r or r not in self.robots:
            return
        if not big_id:
            return

        ctx = self.ctx.get(r)
        if not ctx:
            return
        if str(ctx.get("big_id", "")) != big_id:
            return

        hop_id = str(ctx.get("current_hop_id", "")).strip()
        if hop_id:
            self._publish_mission_cancel(robot=r, mission_id=hop_id)
            self.get_logger().warn(f"[{r}] tm_cancel -> published mission_cancel hop_id={hop_id}")
        else:
            self.get_logger().warn(f"[{r}] tm_cancel requested but no inflight hop_id")

        released = self.lock_mgr.release_all_for_mission(r, big_id)
        goal_wp = str(ctx.get("goal_wp", ""))
        self.ctx.pop(r, None)
        self.robot_busy[r] = False

        # ✅ NEW: CANCEL도 결과 publish (ok=false, reason="canceled")
        self._publish_tm_result(robot=r, big_id=big_id, ok=False, goal_wp=goal_wp, reason="canceled")

        self.get_logger().warn(f"[{r}] canceled big_id={big_id}, released_locks={released}")

    # ======================================================================
    # Tick: inflight 아니면 다음 hop 재시도
    # ======================================================================
    def _tick(self):
        now = time.time()
        for r in self.robots:
            if not self.robot_busy.get(r, False):
                continue
            ctx = self.ctx.get(r)
            if not ctx:
                continue
            if ctx.get("mode") != "GOAL":
                continue
            if bool(ctx.get("inflight", False)):
                continue
            if now < float(ctx.get("next_retry_time", 0.0)):
                continue
            self._try_send_next_step(r)

    # ======================================================================
    # 로봇 결과 수신: /{robot}/mission_result
    # ======================================================================
    def _on_robot_result(self, robot: str, msg: String):
        raw = msg.data.strip()
        if not raw:
            return
        try:
            res = json.loads(raw)
        except Exception:
            return

        mid = str(res.get("mission_id", "")).strip()

        # ⚠️ System1이 success 키를 다르게 낼 수 있으니 방탄 처리
        # - success / ok / result / done 등
        if "success" in res:
            ok = bool(res.get("success", False))
        elif "ok" in res:
            ok = bool(res.get("ok", False))
        else:
            ok = bool(res.get("result", False))

        message = str(res.get("message", res.get("reason", "")))

        ctx = self.ctx.get(robot)
        if not ctx:
            return

        # PLAN 모드
        if ctx.get("mode") == "PLAN":
            if mid == str(ctx.get("current_hop_id", "")):
                self.robot_busy[robot] = False
                self.ctx.pop(robot, None)
                # ✅ PLAN 완료도 TM result로 쏴줄지 여부는 정책인데,
                # 요청사항은 "big_id 기준 완료/실패"라서 PLAN도 publish 해줌(원하면 제거 가능)
                self._publish_tm_result(
                    robot=robot,
                    big_id=str(ctx.get("big_id", mid)),
                    ok=ok,
                    goal_wp=str(ctx.get("goal_wp", "")),
                    reason=message if not ok else "",
                )
                self.get_logger().info(f"[{robot}] PLAN done ok={ok} msg={message}")
            return

        # GOAL 모드: 현재 hop id 결과만 처리
        cur_hop_id = str(ctx.get("current_hop_id", "")).strip()
        if not cur_hop_id:
            return
        if mid != cur_hop_id:
            return

        ctx["inflight"] = False

        if not ok:
            big_id = str(ctx.get("big_id"))
            goal_wp = str(ctx.get("goal_wp", ""))
            released = self.lock_mgr.release_all_for_mission(robot, big_id)
            self.robot_busy[robot] = False
            self.ctx.pop(robot, None)

            # ✅ NEW: FAIL publish
            self._publish_tm_result(robot=robot, big_id=big_id, ok=False, goal_wp=goal_wp, reason=message)

            self.get_logger().error(
                f"[{robot}] HOP FAILED hop_id={mid} msg={message} -> big_mission stop (released={released})"
            )
            return

        # hop 성공이면 TM 내부 current_wp를 "to"로 강제 전진 (RobotState 의존 최소화)
        last_hop = ctx.get("last_hop", None)
        if last_hop:
            try:
                _from_wp, to_wp = last_hop
                if to_wp in self.graph.waypoints:
                    prev = self.robot_current_wp.get(robot, "A")
                    self.robot_current_wp[robot] = to_wp
                    self.robot_wp_quality[robot] = 0.0
                    if prev != to_wp:
                        self.get_logger().info(f"[{robot}] force current_wp: {prev} -> {to_wp} (by hop success)")
            except Exception:
                pass

        # hop 성공 시 previous node release
        self._release_previous_node_on_hop_success(robot, ctx)

        self.get_logger().info(f"[{robot}] HOP OK hop_id={mid} -> next")
        self._try_send_next_step(robot)

    def _release_previous_node_on_hop_success(self, robot: str, ctx: Dict[str, Any]):
        last_hop = ctx.get("last_hop", None)
        if not last_hop:
            return
        try:
            prev_wp, _next_wp = last_hop
        except Exception:
            return

        if self.keep_current_node_locked:
            return

        released = self.lock_mgr.release_node_of_owned(robot, str(prev_wp))
        if released:
            self.get_logger().info(f"[{robot}] lock release (prev node): {prev_wp}")

    # ======================================================================
    # 핵심: hop 계획 + lock + publish
    # ======================================================================
    def _is_blocked_for(self, node: str, robot: str) -> bool:
        return self.lock_mgr.is_locked_by_other(node, robot)

    def _try_send_next_step(self, robot: str):
        ctx = self.ctx.get(robot)
        if not ctx:
            return

        big_id = str(ctx["big_id"])
        goal_wp = str(ctx["goal_wp"])

        cur_wp = self.robot_current_wp.get(robot, "A")
        if cur_wp not in self.graph.waypoints:
            cur_wp = "A"

        # 목표 도착 처리
        if cur_wp == goal_wp:
            # final_pose 1회
            fp = ctx.get("final_pose", None)
            if isinstance(fp, dict) and ("x" in fp) and ("y" in fp):
                hop_id = f"{big_id}__finalpose__{int(ctx['seq'])}"
                ctx["seq"] += 1
                ctx["current_hop_id"] = hop_id
                ctx["inflight"] = True

                yaw = float(fp.get("yaw", 0.0))
                plan = {
                    "robot": robot,
                    "mission_id": hop_id,
                    "steps": [{
                        "task": "move_to",
                        "x": float(fp["x"]),
                        "y": float(fp["y"]),
                        "yaw": float(yaw),
                        "use_waypoints": bool(ctx.get("final_use_waypoints", False)),
                        "timeout_sec": float(ctx.get("timeout_sec", 35.0)),
                    }]
                }
                ctx["final_pose"] = None
                self._publish_mission_request(plan)
                self.get_logger().info(f"[{robot}] publish FINAL_POSE hop_id={hop_id}")
                return

            # follow
            if bool(ctx.get("do_follow", False)):
                hop_id = f"{big_id}__follow__{int(ctx['seq'])}"
                ctx["seq"] += 1
                ctx["current_hop_id"] = hop_id
                ctx["inflight"] = True

                # ✅ 핵심: FOLLOW는 1회만 하도록 latch
                ctx["do_follow"] = False

                plan = {
                    "robot": robot,
                    "mission_id": hop_id,
                    "steps": [{
                        "task": "follow_aruco",
                        "marker_id": int(ctx.get("marker_id", 0)),
                        "timeout_sec": float(ctx.get("timeout_sec", 120.0)),
                    }]
                }
                self._publish_mission_request(plan)
                self.get_logger().info(f"[{robot}] publish FOLLOW hop_id={hop_id}")
                return

            # 종료
            released = self.lock_mgr.release_all_for_mission(robot, big_id)
            self.lock_mgr.try_lock_node(robot, f"occupy__{robot}", goal_wp)
            self.robot_busy[robot] = False
            self.ctx.pop(robot, None)

            # ✅ NEW: DONE publish (big_id 기준)
            self._publish_tm_result(robot=robot, big_id=big_id, ok=True, goal_wp=goal_wp, reason="")

            self.get_logger().info(f"[{robot}] BIG mission done at goal={goal_wp} released={released}")
            return

        # 회피 다익스트라로 next_wp 선택
        path = self.graph.dijkstra_avoiding_locked(
            start=cur_wp,
            goal=goal_wp,
            is_blocked_fn=lambda n: self._is_blocked_for(n, robot),
            allow_nodes={cur_wp, goal_wp},
        )
        if not path or len(path) < 2:
            self.get_logger().warn(f"[{robot}] no available path cur={cur_wp} goal={goal_wp} -> wait")
            ctx["next_retry_time"] = time.time() + self.retry_wait_sec
            return

        next_wp = path[1]

        # current 노드 락 옵션
        if self.keep_current_node_locked:
            self.lock_mgr.try_lock_node(robot, big_id, cur_wp)

        hop_id = f"{big_id}__hop_{cur_wp}_to_{next_wp}__{int(ctx['seq'])}"
        if not self.lock_mgr.try_lock_node(robot, big_id, next_wp):
            self.get_logger().info(f"[{robot}] wait: next node locked -> {next_wp}")
            ctx["next_retry_time"] = time.time() + self.retry_wait_sec
            return

        # 진행 방향 yaw
        p_next = self.graph.waypoints[next_wp]
        p_cur = self.graph.waypoints[cur_wp]
        yaw = math.atan2(p_next.y - p_cur.y, p_next.x - p_cur.x)

        # goal_wp 마지막 hop에서만 final_yaw 적용
        if next_wp == goal_wp and float(ctx.get("final_yaw", 0.0)) != 0.0:
            yaw = float(ctx["final_yaw"])

        ctx["seq"] += 1
        ctx["current_hop_id"] = hop_id
        ctx["inflight"] = True
        ctx["last_hop"] = (cur_wp, next_wp)

        p = self.graph.waypoints[next_wp]
        plan = {
            "robot": robot,
            "mission_id": hop_id,
            "steps": [{
                "task": "move_to",
                "x": float(p.x),
                "y": float(p.y),
                "yaw": float(yaw),
                "use_waypoints": False,  # System1에서 다익스트라 사용 안하게
                "timeout_sec": float(ctx.get("timeout_sec", 35.0)),
            }]
        }
        self._publish_mission_request(plan)
        self.get_logger().info(f"[{robot}] publish HOP {cur_wp}->{next_wp} locked={next_wp} hop_id={hop_id}")

    # ======================================================================
    # Topic output helpers
    # ======================================================================
    def _publish_mission_request(self, plan: Dict[str, Any]):
        self.mission_req_pub.publish(String(data=json.dumps(plan, ensure_ascii=False)))

    def _publish_mission_cancel(self, robot: str, mission_id: str):
        payload = {"robot": robot, "mission_id": mission_id}
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.mission_cancel_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = TrafficManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()



"""
==================== 실행 예시 ====================
# 우선 도메인 브릿지 실행
ros2 run domain_bridge domain_bridge ~/jazzy_ws/src/pinky_structure/config/domain_bridge_b44f.yaml \
 --ros-args --log-level debug

# 1) TrafficManager 실행
ros2 run traffic_manager traffic_manager


상하차(B)
ros2 topic pub --once /traffic/tm_request std_msgs/String "{data: '{\"type\":\"GOAL\",\"robot\":\"pinky1\",\"mission_id\":\"m_loading_B_001\",\"goal_wp\":\"B\",\"timeout_sec\":120.0,\"final_pose\":{\"x\":0.1341,\"y\":0.3854,\"yaw\":-0.0451},\"final_use_waypoints\":false}'}"

검수대(C)
ros2 topic pub --once /traffic/tm_request std_msgs/String "{data: '{\"type\":\"GOAL\",\"robot\":\"pinky1\",\"mission_id\":\"m_qc_C_001\",\"goal_wp\":\"C\",\"timeout_sec\":120.0,\"final_pose\":{\"x\":0.3315,\"y\":0.2787,\"yaw\":-1.6064},\"final_use_waypoints\":false}'}"

조립대(E)
ros2 topic pub --once /traffic/tm_request std_msgs/String "{data: '{\"type\":\"GOAL\",\"robot\":\"pinky1\",\"mission_id\":\"m_assembly_E_001\",\"goal_wp\":\"E\",\"timeout_sec\":240.0,\"final_pose\":{\"x\":0.2328,\"y\":-0.4291,\"yaw\":3.0524},\"final_use_waypoints\":false}'}"

모듈창고(G)
ros2 topic pub --once /traffic/tm_request std_msgs/String "{data: '{\"type\":\"GOAL\",\"robot\":\"pinky1\",\"mission_id\":\"m_module_G_001\",\"goal_wp\":\"G\",\"timeout_sec\":240.0,\"final_pose\":{\"x\":0.3441,\"y\":-0.9863,\"yaw\":-3.1315},\"final_use_waypoints\":false}'}"

출고창고(H)
ros2 topic pub --once /traffic/tm_request std_msgs/String "{data: '{\"type\":\"GOAL\",\"robot\":\"pinky1\",\"mission_id\":\"m_ship_H_001\",\"goal_wp\":\"H\",\"timeout_sec\":240.0,\"final_pose\":{\"x\":0.0714,\"y\":-1.0637,\"yaw\":1.5159},\"final_use_waypoints\":false}'}"

ros2 topic pub --once /traffic/tm_request std_msgs/String "{data: '{\
\"type\":\"GOAL\",\
\"robot\":\"pinky1\",\
\"mission_id\":\"m_D_001\",\
\"goal_wp\":\"D\",\
\"timeout_sec\":120.0\
}'}"

ros2 topic pub --once /traffic/tm_request std_msgs/String "{data: '{\
\"type\":\"GOAL\",\
\"robot\":\"pinky1\",\
\"mission_id\":\"m_F_001\",\
\"goal_wp\":\"F\",\
\"timeout_sec\":240.0\
}'}"

"""
