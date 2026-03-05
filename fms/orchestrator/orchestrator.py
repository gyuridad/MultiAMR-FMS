#!/usr/bin/env python3
"""
Orchestrator v4 (FULL FILE) - MissionResult 기반 (Domain 5)  +  Return-to-Wait (mission_result sync)  +  do_follow_aruco

✅ 핵심 변경점 (중요)
- "pose 도착(_amr_at_station)" 기반 상태전환을 완전히 제거
- 오직 /{robot}/mission_result (std_msgs/String JSON) 의 OK/FAIL 로만
  GO_WORK / GO_NEXT / WAIT 복귀 완료를 판정한다.
- 그래서 TrafficManager의 busy/reject 문제를 구조적으로 제거한다.

✅ Topic payloads (std_msgs/String JSON)
- /traffic/tm_request (Orchestrator → TrafficManager):
  {"type":"GOAL","robot":"pinky1","mission_id":"J100__work","goal_wp":"B","timeout_sec":120.0,
   "do_follow_aruco":false,"marker_id":600(optional),"final_pose":{...},"final_use_waypoints":false}

- /{robot}/mission_result (TrafficManager → Orchestrator):
  {"robot":"pinky1","mission_id":"J100__work","ok":true,"reason":"...", "ts": 123456.7}

- /armX/command (Orchestrator → Arm Adapter):
  {"cmd":"START_LOAD|START_UNLOAD|CANCEL|RESET","job_id":"J001","assigned_amr":"pinky1",
   "job_type":"...","work_station":"...","next_station":"..."}

- /armX/state (Arm Adapter → Orchestrator):
  {"arm":"arm3","state":"DONE|LOADING|LOADED|...","job_id":"J001","assigned_amr":"pinky1","action_mode":"LOAD|UNLOAD"}

✅ Job Phase (상태머신)
NEW -> GO_WORK -> WAIT_ARM -> GO_NEXT -> DONE -> WAIT_RETURN -> FINISHED
(arm 없는 이동-only job은 GO_WORK 성공 후 GO_NEXT 또는 DONE으로)
(return_to_wait_enabled=False면 DONE에서 즉시 FINISHED/AMR FREE)

✅ 포함된 Job types 예시
- AMR_TO_QC (work=LOADING_ZONE, next=QC_ZONE, arm 없음/또는 필요시 매핑 가능)
- ASSEMBLY_TO_MODULE_STORAGE (work=ASSEMBLY_ZONE, next=MODULE_STORAGE, arm=arm3)
- MODULE_STORAGE_TO_SHIP (move-only, work=MODULE_STORAGE, next=SHIPPING_ZONE, arm 없음)
"""

import json
import time
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from pinky_interfaces.msg import RobotState


# ---------------------------- Data Models ----------------------------
@dataclass
class Job:
    job_id: str
    job_type: str

    # "arm 작업이 실제로 수행되는 스테이션"
    work_station: str
    next_station: str = ""   

    work_arm: str = ""
    assigned_amr: str = ""
    requires_amr: bool = True
    phase: str = "NEW"    # NEW -> GO_WORK -> WAIT_ARM -> GO_NEXT -> DONE/FAIL
    ts: float = 0.0

    # ✅ arm3 DONE 트리거로 만들어진 ship job의 "원본(base) id"를 기억 (방탄 매칭용)
    base_job_id: str = ""

    # 일반 FAIL: 바로 FREE (지금처럼)
    # RTB FAIL: WAIT_RETURN로 보내고, wait 미션 결과를 받은 뒤 FREE
    fail_reason: str = ""

# ---------------------------- Orchestrator ----------------------------
class OrchestratorNode(Node):
    def __init__(self):
        super().__init__("orchestrator")

        # ---------------- Params ----------------
        self.declare_parameter("robots", ["pinky1", "pinky2", "pinky3"])
        self.declare_parameter("arms", ["arm1", "arm2", "arm3"])

        self.declare_parameter("mission_request_topic", "/traffic/tm_request")  # AMR(핑키)에게 “미션 요청”
        self.declare_parameter("job_request_topic", "/orchestrator/job_request")  # 누군가(운영자/MES)가 orchestrator에게 job 요청

        self.declare_parameter("robot_state_suffix", "/robot_state")

        # ✅ NEW (최소 수정 핵심): TM result topic 단일 구독
        self.declare_parameter("tm_result_topic", "/traffic/mission_result")

        # ---------------- Station -> WP ----------------
        self.declare_parameter(
            "station_to_pose_json",
            json.dumps({
                "Pinky1_ZONE": {"x": 0.0, "y": 0.0, "yaw": 0.0},
                "Pinky2_ZONE": {"x": -0.0096, "y": -0.3430, "yaw": -0.0580},
                "Pinky3_ZONE": {"x": 0.0130, "y": -0.6830, "yaw": 0.0354},

                "LOADING_ZONE": {"x": 0.1341, "y": 0.3854, "yaw": -0.0451},
                "QC_ZONE": {"x": 0.3315, "y": 0.2787, "yaw": -1.6064},
                "ASSEMBLY_ZONE": {"x": 0.2328, "y": -0.4291, "yaw": 3.0524},
                "MODULE_STORAGE": {"x": 0.3441, "y": -0.9863, "yaw": -3.1315},
                "SHIPPING_ZONE": {"x": 0.0714, "y": -1.0637, "yaw": 1.5159},
            })
        )

        self.declare_parameter(
            "station_to_wp_json",
            # 아래 부분은 교통정리매니저 파일과 이름을 맞춰야함 
            json.dumps(
                {
                    "Pinky1_ZONE": "P1",     # 충전구역
                    "Pinky2_ZONE": "P2",     # 충전구역
                    "Pinky3_ZONE": "P3",     # 충전구역

                    "LOADING_ZONE": "B",      # 상하차구역
                    "QC_ZONE": "C",           # 입고검수구역
                    "ASSEMBLY_ZONE": "E",     # 조립대구역
                    "MODULE_STORAGE": "G",    # 모듈창고구역
                    "SHIPPING_ZONE": "H",     # 출고창고구역
                }
            )
        )
        # ---------------- Station -> Arm ----------------
        # 이 매핑은 이후에 **Job을 만들 때 “work_station에 맞는 work_arm을 결정”**하는 데 쓰입니다.
        self.declare_parameter(
            "station_to_arm_json",
            json.dumps(
                {
                    "QC_ZONE": "arm1",
                    "PARTS_STORAGE": "arm2",
                    "ASSEMBLY_ZONE": "arm3",
                }
            )
        )
        # ---------------- Arm Roles ----------------
        self.declare_parameter(
            "arm_action_json",
            json.dumps(
                {
                    "arm1": "UNLOAD",
                    "arm2": "UNLOAD",
                    "arm3": "LOAD",
                }
            )
        )
            # 로봇팔은 “물류 의미”를 알아야만 움직일 수 있음
            #     로봇팔에게 필요한 정보는 단순 위치가 아니라:
            #         어디서 작업하나? (station)
            #         무엇을 하나? (적재냐 / 하역이냐)
            #         누구를 대상으로 하나? (AMR인지, 컨베이어인지)
            #     그래서 오케스트레이터가
            #         QC_ZONE + arm1 + UNLOAD
            #     같은 의미 패킷을 만들어 주는 거예요.


        # ---------------- Job Type Defaults ----------------
        # “job_type(작업 종류)을 입력하면,
        # 그 작업이 어디서 시작하고(Work), 어디로 가며(Next), AMR이 필요한지까지
        # 기본값을 자동으로 채워주는 ‘작업 레시피(템플릿) 사전’
        self.declare_parameter(
            "job_type_defaults_json",
            json.dumps(
                {
                    "AMR_TO_QC": {
                        "work_station": "QC_ZONE",
                        "next_station": "",
                        "requires_amr": True,
                    },
                    "QC_TO_PARTS_STORAGE": {
                        "work_station": "PARTS_STORAGE",
                        "next_station": "",
                        "requires_amr": False,
                    },
                    "PARTS_TO_ASSEMBLY": {
                        "work_station": "ASSEMBLY_ZONE",
                        "next_station": "",
                        "requires_amr": False,
                    },
                    "ASSEMBLY_TO_MODULE_STORAGE": {
                        "work_station": "ASSEMBLY_ZONE",
                        "next_station": "MODULE_STORAGE",
                        "requires_amr": True,
                    },
                    "MODULE_STORAGE_TO_SHIP": {
                        "work_station": "MODULE_STORAGE",
                        "next_station": "SHIPPING_ZONE",
                        "requires_amr": True,
                    },
                }
            )
        )
            # 잡 타입 늘어나면: 일단 job_type_defaults_json에 추가하면 됨 ✅
            # 새로운 스테이션을 쓰면: station_to_wp_json, station_to_arm_json도 같이 추가해야 함 ✅

        # ---------------- Assembly DONE Trigger ----------------
        self.declare_parameter("assembly_done_creates_job_type", "ASSEMBLY_TO_MODULE_STORAGE")
            # “조립이 끝났으니 다음 공정(예: 출고로 보내기)을 오케스트레이터가 자동으로 작업지시로 만들어준다”
            # 그 새 Job은 보통 “AMR이 필요”한 작업이라 AMR을 배정해서 출고로 보냄
        
        # ---------------- Return-to-Wait (simple) ----------------
        self.declare_parameter("return_to_wait_enabled", True)
        # ✅ 여기! 작업 종료 후 기본 복귀 장소를 충전소로
        self.declare_parameter("wait_station", "Pinky1_ZONE")
        self.declare_parameter(
            "wait_station_by_robot_json",
            json.dumps({
                "pinky1": "Pinky1_ZONE",
                "pinky2": "Pinky2_ZONE",
                "pinky3": "Pinky3_ZONE",
            })
        )
        self.declare_parameter("wait_mission_suffix", "__wait")
            # 원래 작업 미션과 구분하기 위해 사용
            #     작업 이동: J100__work
            #     다음 이동: J100__next
            #     대기 복귀: J100__wait

        # ✅ 도킹 스테이션(복수) - 이 목록에 있으면 do_follow_aruco=True
        self.declare_parameter("docking_stations_json", json.dumps([]))
        
        # ✅ Station -> ArUco Marker ID (FOLLOW_ARUCO에 쓸 marker_id)
        self.declare_parameter(
            "station_to_aruco_id_json",
            json.dumps({
                "QC_ZONE": 600,
                "ASSEMBLY_ZONE": 600,
            })
        )

        # ---------------- Timeout ----------------
        # 오케스트레이터의 “상태머신이 멈추지 않게 하는 안전장치(타임아웃)” + **“루프 주기(틱)”**를 정하는 값들
        self.declare_parameter("go_work_timeout_sec", 300.0)
        self.declare_parameter("wait_arm_timeout_sec", 600.0)
        self.declare_parameter("go_next_timeout_sec", 300.0)
        self.declare_parameter("wait_return_timeout_sec", 300.0)
        self.declare_parameter("tick_period_sec", 0.2)

        # ---------------- Read params ----------------
        self.robots = list(self.get_parameter("robots").value)
        self.arms = list(self.get_parameter("arms").value)

        self.station_to_pose = json.loads(self.get_parameter("station_to_pose_json").value)
        self.station_to_wp = json.loads(self.get_parameter("station_to_wp_json").value)
        self.station_to_arm = json.loads(self.get_parameter("station_to_arm_json").value)
        self.arm_action = json.loads(self.get_parameter("arm_action_json").value)
        self.job_type_defaults = json.loads(self.get_parameter("job_type_defaults_json").value)

        self.assembly_done_creates_job_type = self.get_parameter("assembly_done_creates_job_type").value

        self.return_to_wait_enabled = self.get_parameter("return_to_wait_enabled").value
        self.wait_station = self.get_parameter("wait_station").value
        self.wait_mission_suffix = self.get_parameter("wait_mission_suffix").value

        # ✅ docking stations list
        try:
            ds = json.loads(str(self.get_parameter("docking_stations_json").value))
            if not isinstance(ds, list):
                ds = []
        except Exception:
            ds = []
        self.docking_stations: List[str] = [str(x).strip() for x in ds if str(x).strip()]

        # ✅ station_to_aruco_id map
        try:
            sta = json.loads(str(self.get_parameter("station_to_aruco_id_json").value))
            if not isinstance(sta, dict):
                sta = {}
        except Exception:
            sta = {}
        self.station_to_aruco_id: Dict[str, int] = {}
        for k, v in sta.items():
            ks = str(k).strip()
            if not ks:
                continue
            try:
                self.station_to_aruco_id[ks] = int(v)
            except Exception:
                # 숫자 변환 실패하면 무시
                pass
        self.get_logger().info(f"Station->ArucoID: {self.station_to_aruco_id}")

        # ✅ wait_station_by_robot map
        try:
            wbr = json.loads(str(self.get_parameter("wait_station_by_robot_json").value))
            if not isinstance(wbr, dict):
                wbr = {}
        except Exception:
            wbr = {}

        self.wait_station_by_robot: Dict[str, str] = {
            str(k).strip(): str(v).strip()
            for k, v in wbr.items()
            if str(k).strip() and str(v).strip()
        }

        self.go_work_timeout = self.get_parameter("go_work_timeout_sec").value
        self.wait_arm_timeout = self.get_parameter("wait_arm_timeout_sec").value
        self.go_next_timeout = self.get_parameter("go_next_timeout_sec").value
        self.wait_return_timeout = float(self.get_parameter("wait_return_timeout_sec").value)
        tick_period = self.get_parameter("tick_period_sec").value

        # ---------------- Publishers ----------------
        # AMR(핑키)에게 “미션 요청” 보내는 발행자
        self.mission_pub = self.create_publisher(String, self.get_parameter("mission_request_topic").value, 10)
            # 예시 미션 발행:
            #     {
            #     "mission_id": "J100__work",
            #     "robot_name": "pinky1",
            #     "goal_wp": "B",
            #     "do_follow_aruco": false
            #     }

        self.arm_cmd_pub = {
            arm: self.create_publisher(String, f"/{arm}/command", 10)
            for arm in self.arms
        }
            # 예시 미션 발행
            #     {
            #     "cmd": "START_UNLOAD",
            #     "job_id": "J100",
            #     "assigned_amr": "pinky1",
            #     "job_type": "AMR_TO_QC",
            #     "work_station": "QC_ZONE",
            #     "next_station": "LOADING_ZONE"
            #     }

        # ---------------- Subscriptions ----------------
        self.arm_state: Dict[str, Dict[str, Any]] = {
            a: {"state": "IDLE", "job_id": "", "assigned_amr": "", "action_mode": ""} for a in self.arms
        }
        for arm in self.arms:
            self.create_subscription(String, f"/{arm}/state", lambda msg, a=arm: self._on_arm_state(a, msg), 10)
            self.get_logger().info(f"Subscribed arm state: /{arm}/state")
        
        # 누군가(운영자/MES/테스트)가 job 요청을 토픽에 발행
        job_req_topic = str(self.get_parameter("job_request_topic").value)
        self.create_subscription(String, job_req_topic, self._on_job_request, 10)
        self.get_logger().info(f"Subscribed job request: {job_req_topic}")
            # 예를 들어 “QC 작업을 시작해줘” 요청을 보내면:
            #     ros2 topic pub --once /orchestrator/job_request std_msgs/msg/String \
            #     "{data: '{\"job_id\":\"J100\",\"job_type\":\"AMR_TO_QC\"}'}"

        # self.robot_current_wp = {r: "" for r in self.robots}
        # self.robot_wp_dist = {r: 1e9 for r in self.robots}

        state_suffix = str(self.get_parameter("robot_state_suffix").value)
        for r in self.robots:
            topic = f"/{r}{state_suffix}"
            self.create_subscription(RobotState, topic, lambda msg, rr=r: self._on_robot_state(rr, msg), 10)
            self.get_logger().info(f"Subscribed AMR RobotState: {topic}")

        # ✅ NEW: only TM result topic
        tm_result_topic = str(self.get_parameter("tm_result_topic").value)
        self.create_subscription(String, tm_result_topic, self._on_tm_result, 10)
        self.get_logger().info(f"Subscribed TM MissionResult: {tm_result_topic}")

        # ---------------- Internal ----------------
        self.jobs: List[Job] = []
        self.amr_busy = {r: False for r in self.robots}
        self.amr_job: Dict[str, str] = {r: "" for r in self.robots}

        # arm3가 내보내는 job_id가 “가끔 base로 오기도 하고 ship으로 오기도 하는” 흔들림을 흡수하기 위한 ‘매핑표(번역표)’
        # arm3가 "DONE"을 보내면: 
        # **“조립 완료”**로 보고 오케스트레이터가 배송용 job(= ship job) 을 자동 생성해.
        # 예: base=J999 → ship=J999__ship
        # 그런데 arm3가 다음에 "LOADED"를 보낼 때,
        # job_id를 ship(J999__ship)로 보내지 않고 base(J999)로 보내는 버그/관습/실수가 생길 수 있음
        self.arm3_base_to_ship: Dict[str, str] = {}    # "J999" → "J999__ship"
        self.arm3_ship_to_base: Dict[str, str] = {}    # "J999__ship" → "J999"
            # 왜 굳이 두 개나 만들었냐?
            # 한쪽만 있어도 어느 정도는 처리 가능하지만, 실무에서 둘 다 있으면 편해:
            #     base로 들어온 신호를 ship으로 바꾸기 쉬움 (base_to_ship)
            #     ship을 기준으로 원본 base를 추적하기 쉬움 (ship_to_base)
            # 이 두 딕셔너리는 “주문번호(원본) ↔ 송장번호(파생)” 연결표 같은 거야.
            # 현장에서 송장번호로 들어오든 주문번호로 들어오든 같은 건으로 처리하려는 목적.

        self.robot_pose = {r: {"x": 0.0, "y": 0.0, "yaw": 0.0, "frame": ""} for r in self.robots}

        # mission result cache: mission_id -> (ok:bool, ts:float, robot:str, reason:str)
        self.mission_cache: Dict[str, Dict[str, Any]] = {}

        self.create_timer(tick_period, self._tick)

        self.get_logger().info("✅ Orchestrator v4 started (MISSION_RESULT based).")
        self.get_logger().info(f"Station->Arm: {self.station_to_arm}")
        self.get_logger().info(f"Arm->Action : {self.arm_action}")
        self.get_logger().info(f"Station->WP : {self.station_to_wp}")
        self.get_logger().info(f"Return-to-wait: enabled={self.return_to_wait_enabled} wait_station={self.wait_station}")
        self.get_logger().info(f"Docking stations (do_follow_aruco=True): {self.docking_stations}")
        self.get_logger().info(f"Assembly DONE creates: {self.assembly_done_creates_job_type}")
        self.get_logger().info(f"Wait station by robot: {self.wait_station_by_robot}")

    # ---------------- Callbacks ----------------
    def _on_job_request(self, msg: String):
        """
        기본:
          {"job_id":"J100","job_type":"AMR_TO_QC"}
          {"job_id":"J200","job_type":"QC_TO_PARTS_STORAGE"}
        override 가능:
          {"job_id":"J901","job_type":"AMR_TO_QC","work_station":"QC_ZONE","next_station":"LOADING_ZONE","requires_amr":true}
        """
        try:
            d = json.loads(msg.data)
            job_id = str(d["job_id"]).strip()
            job_type = str(d.get("job_type", "")).strip().upper()
        except Exception as e:
            self.get_logger().warn(f"Bad job_request: {e}")
            return
        
        if not job_id:
            return

        if any(j.job_id == job_id for j in self.jobs):
            return
        
        if job_type not in self.job_type_defaults:
            self.get_logger().warn(f"[JOB] unknown job_type={job_type} (ignored)")
            return
        
        job_base = self.job_type_defaults[job_type]
        work_station = d.get("work_station", job_base["work_station"])
        next_station = d.get("next_station", job_base["next_station"])
        requires_amr = d.get("requires_amr", job_base["requires_amr"])

        if not work_station:
            self.get_logger().warn(f"[JOB] missing work_station for {job_type}")
            return

        work_arm = str(self.station_to_arm.get(work_station, "")).strip()

        # ✅ 이동-only job 지원:
        # requires_amr=True 인 경우 work_arm 없어도 허용 (MODULE_STORAGE_TO_SHIP 같은 케이스)
        if not work_arm:
            if requires_amr:
                self.get_logger().info(f"[JOB {job_id}] move-only OK (no arm) work_station={work_station}")
            else:
                self.get_logger().warn(f"[JOB {job_id}] work_station={work_station} has no arm mapping")
                return
        
        job = Job(
            job_id=job_id,
            job_type=job_type,
            work_station=work_station,
            next_station=next_station,
            work_arm=work_arm,
            assigned_amr="",
            requires_amr=requires_amr,
            phase="NEW",
            ts=time.time(),
            base_job_id="",
        )
        self.jobs.append(job)
        self.get_logger().info(
            f"[JOB] created: {job_id} type={job_type} requires_amr={requires_amr} work={work_station}(arm={work_arm}) next={next_station or 'none'}"
        )

    # “arm3이 조립 작업을 끝냈다(DONE) 라는 상태 이벤트가 들어올 때”만 필요한 로직
    # DONE은 “Job 생성 트리거” / LOADED는 “WAIT_ARM 완료 트리거”
    def _on_arm_state(self, arm: str, msg: String):
        try:
            d = json.loads(msg.data)
        except Exception as e:
            self.get_logger().warn(f"Bad arm_state from {arm}: {e}")
            return
        
        # ✅ arm이 보내준 state를 “비교하기 쉬운 형태”로 정리해서 캐시에 저장
        # {"arm":"arm3","state":" loaded  ","job_id":" J999 ","assigned_amr":"pinky1"} 이렇게 대소문자 혼용시 정규화 필요
        state_norm = str(d.get("state", "")).strip().upper()
        job_id_norm = str(d.get("job_id", "")).strip()
        assigned_amr_norm = str(d.get("assigned_amr", "")).strip()

        # 원래 dict d 안에 정규화 결과를 “추가 필드”로 저장
        d["_state_norm"] = state_norm
        d["_job_id_norm"] = job_id_norm
        d["_assigned_amr_norm"] = assigned_amr_norm
        
        # ✅ arm별 최신 상태를 캐시에 업데이트
        self.arm_state[arm] = d

        # ---- 여기서부터는 arm3 전용 특수 규칙 ----
        # arm3가 아니면 여기서 끝
        if arm != "arm3":
            return

        # arm3라도 DONE이 아니면 끝
        if state_norm != "DONE":
            return
        
        # arm3 DONE이 의미하는 것
        #     arm3 "DONE" = 조립 완료
        #     조립 완료되면 AMR을 불러서 SHIPPING_ZONE으로 보내는 job(= ship job) 을 자동 생성

        base = job_id_norm
        if not base:
            return
        
        ship_job_id = f"{base}__ship"

        # 이미 존재하면 생성하지 않음
        if any(j.job_id == ship_job_id for j in self.jobs):
            return
        
        # ✅ 매핑 저장 (나중에 LOADED가 base로 와도 ship job을 완료시킬 수 있게)
        self.arm3_base_to_ship[base] = ship_job_id
        self.arm3_ship_to_base[ship_job_id] = base

        job_base = self.job_type_defaults[self.assembly_done_creates_job_type]  # ASSEMBLY_TO_MODULE_STORAGE
        job = Job(
            job_id=ship_job_id,
            job_type=self.assembly_done_creates_job_type,
            work_station="ASSEMBLY_ZONE",
            next_station=str(job_base.get("next_station", "")).strip(),
            work_arm="arm3",
            requires_amr=True,
            phase="NEW",
            ts=time.time(),
            base_job_id=base,
        )
        self.jobs.append(job)
        self.get_logger().info(f"[AUTOJOB] arm3 DONE(base={base}) -> created {ship_job_id} ({self.assembly_done_creates_job_type})")

    def _on_robot_state(self, robot: str, msg: RobotState):
        # ✅ 항상 pose 업데이트
        self.robot_pose[robot] = {
            "x": float(msg.pose_x),
            "y": float(msg.pose_y),
            "yaw": float(msg.pose_yaw),
            "frame": str(msg.pose_frame),
        }

    # ---------------- Core Loop ----------------
    # “잡 리스트를 계속 훑어보면서, 지금 할 일이 생기면 AMR 보내고 / 팔 시키고 / 다음 목적지 보내고 / 끝내는 루프”
    #     NEW → GO_WORK → WAIT_ARM → GO_NEXT → DONE
    #     (상황에 따라 GO_NEXT 없이 DONE)
    #     그리고 requires_amr=False면 이동 없이:
    #     NEW → WAIT_ARM → DONE

    # “스테이션 도착 판정” 함수 거리기반 도착 판정 함수 사용안함 !!! 아래의 미션 결과 받아서 도착 판정으로 사용 !!!
    # 왜냐? 핑키가 도착하지도 않았는데 거리기반이라 도착이라고 판단하고 다시 해당 핑키에게 작업 지시를 내림
    # def _amr_at_station(self, amr: str, station: str) -> bool:
    #     if not amr or not station:
    #         return False
    #     pose = self.robot_pose.get(amr, None)
    #     target = self.station_to_pose.get(station, None)  # {"x":..,"y":..,"yaw":..}
    #     if not pose or not isinstance(target, dict):
    #         return False
    #     dx = float(pose["x"]) - float(target["x"])
    #     dy = float(pose["y"]) - float(target["y"])
    #     dist = (dx*dx + dy*dy) ** 0.5
    #     return dist <= self.snap_accept_dist_m
    
    # ✅ NEW: TM 단일 결과 구독
    def _on_tm_result(self, msg: String):
        """
        기대 JSON:
          {"robot":"pinky1","mission_id":"J100__work","ok":true,"reason":"...", "ts": 123.4}
        """
        raw = msg.data.strip()
        if not raw:
            return
        try:
            d = json.loads(raw)
        except Exception as e:
            self.get_logger().warn(f"Bad /traffic/mission_result: {e}")
            return

        mission_id = str(d.get("mission_id", "")).strip()
        if not mission_id:
            return

        ok = bool(d.get("ok", False))
        reason = str(d.get("reason", "")).strip()
        ts = float(d.get("ts", time.time()))
        robot = str(d.get("robot", "")).strip()

        self.mission_cache[mission_id] = {
            "ok": ok,
            "robot": robot,
            "reason": reason,
            "ts": ts,
        }
    
    def _tick(self):
        now = time.time()

        # ---------------- NEW -> GO_WORK / WAIT_ARM ----------------
        for job in list(self.jobs):

            # ---------------- NEW ----------------
            if job.phase == "NEW":
                requires_amr = bool(job.requires_amr)

                # (A) AMR 포함 Job: idle AMR 할당 -> GO_WORK
                if requires_amr:
                    # ✅ (편향 최소화는 별도 정책에서) 여기서는 기존 방식 유지
                    amr = next((r for r in self.robots if not self.amr_busy.get(r, False)), None)
                    if not amr:
                        continue

                    if not self._wp_of_station(job.work_station):
                        self._fail_job(job, f"unknown station mapping work_station={job.work_station}")
                        continue

                    self.amr_busy[amr] = True
                    self.amr_job[amr] = job.job_id
                    job.assigned_amr = amr
                    job.phase = "GO_WORK"
                    job.ts = now

                    self._send_mission_request(
                        robot=amr,
                        mission_id=f"{job.job_id}__work",
                        station=job.work_station,
                    )
                    self.get_logger().info(
                        f"[JOB {job.job_id}] assign {amr} -> GO_WORK {job.work_station} (arm={job.work_arm or 'none'})"
                    )
                    continue

                # (B) ARM-only Job: 바로 arm 작업 시작 -> WAIT_ARM
                if not job.work_arm:
                    self._fail_job(job, "arm_only_but_no_work_arm")
                    continue

                job.phase = "WAIT_ARM"
                job.ts = now
                self._start_arm_job(job, assigned_amr="")
                self.get_logger().info(
                    f"[JOB {job.job_id}] ARM-only -> WAIT_ARM arm={job.work_arm} station={job.work_station}"
                )
                continue

            # ---------------- GO_WORK ----------------
            if job.phase == "GO_WORK":
                if now - job.ts > self.go_work_timeout:
                    self._fail_job(job, "go_work_timeout")
                    continue

                mid = f"{job.job_id}__work"
                mr = self._consume_mission_result(mid)
                if mr is None:
                    continue

                if not mr.get("ok", False):
                    self._fail_job(job, f"work_mission_failed:{mr.get('reason','')}")
                    continue

                # ✅ IMPORTANT: GO_WORK 이후에는 amr 변수를 쓰지 말고 job.assigned_amr 사용
                amr = str(job.assigned_amr).strip()
                if not amr:
                    self._fail_job(job, "go_work_ok_but_no_assigned_amr")
                    continue

                # ✅ 도착 후 분기:
                # - arm 있으면 WAIT_ARM
                # - arm 없으면 move-only: GO_NEXT(or DONE)
                if job.work_arm:
                    job.phase = "WAIT_ARM"
                    job.ts = now
                    self._start_arm_job(job, assigned_amr=amr)  # ✅ job.assigned_amr
                    self.get_logger().info(
                        f"[JOB {job.job_id}] {amr} WORK_OK -> WAIT_ARM arm={job.work_arm}"
                    )
                    continue

                # move-only
                if job.next_station:
                    if not self._wp_of_station(job.next_station):
                        self._fail_job(job, f"unknown station mapping next_station={job.next_station}")
                        continue
                    job.phase = "GO_NEXT"
                    job.ts = now
                    self._send_mission_request(
                        robot=amr,  # ✅ job.assigned_amr
                        mission_id=f"{job.job_id}__next",
                        station=job.next_station,
                    )
                    self.get_logger().info(
                        f"[JOB {job.job_id}] {amr} WORK_OK -> GO_NEXT {job.next_station} (move-only)"
                    )
                else:
                    job.phase = "DONE"
                    job.ts = now
                    self.get_logger().info(
                        f"[JOB {job.job_id}] {amr} WORK_OK -> DONE (move-only, no next)"
                    )
                continue

            # ---------------- WAIT_ARM ----------------
            if job.phase == "WAIT_ARM":
                if now - job.ts > self.wait_arm_timeout:
                    self._fail_job(job, "wait_arm_timeout")
                    continue

                arm = str(job.work_arm).strip()
                if not arm:
                    self._fail_job(job, "wait_arm_no_work_arm")
                    continue

                st = self.arm_state.get(arm, {}) or {}

                # ✅ state normalize
                st_state = str(st.get("_state_norm", "")).strip().upper()
                if not st_state:
                    st_state = str(st.get("state", "")).strip().upper()

                # ✅ Adapter와 맞추기: LOADED 또는 DONE 둘 다 완료로 인정 (필요시 여기만 수정)
                if st_state not in ("LOADED", "DONE"):
                    continue

                # ✅ job_id normalize
                st_job_id = str(st.get("_job_id_norm", "")).strip()
                if not st_job_id:
                    st_job_id = str(st.get("job_id", "")).strip()

                # (옵션) arm3 base_job_id 허용
                if arm == "arm3" and job.base_job_id:
                    if st_job_id not in (job.job_id, job.base_job_id):
                        continue
                else:
                    if st_job_id != job.job_id:
                        continue

                # ✅ assigned_amr normalize
                st_assigned = str(st.get("_assigned_amr_norm", "")).strip()
                if not st_assigned:
                    st_assigned = str(st.get("assigned_amr", "")).strip()

                # A) AMR 포함 job
                if job.requires_amr:
                    amr = str(job.assigned_amr).strip()
                    if not amr:
                        self._fail_job(job, "wait_arm_requires_amr_but_no_assigned_amr")
                        continue

                    # assigned_amr mismatch 방지 (arm이 값을 주는 경우만)
                    if st_assigned and st_assigned != amr:
                        continue

                    # next 있으면 GO_NEXT, 없으면 DONE
                    if job.next_station:
                        if not self._wp_of_station(job.next_station):
                            self._fail_job(job, f"unknown station mapping next_station={job.next_station}")
                            continue

                        job.phase = "GO_NEXT"
                        job.ts = now
                        self._send_mission_request(
                            robot=amr,  # ✅ job.assigned_amr
                            mission_id=f"{job.job_id}__next",
                            station=job.next_station,
                        )
                        self.get_logger().info(
                            f"[JOB {job.job_id}] Arm({arm}) {st_state} -> {amr} GO_NEXT {job.next_station}"
                        )
                    else:
                        job.phase = "DONE"
                        job.ts = now
                        self.get_logger().info(
                            f"[JOB {job.job_id}] Arm({arm}) {st_state} -> DONE (no next)"
                        )
                    continue

                # B) ARM-only job
                job.phase = "DONE"
                job.ts = now
                self.get_logger().info(f"[JOB {job.job_id}] Arm({arm}) {st_state} -> DONE (ARM-only)")
                continue

            # ---------------- GO_NEXT ----------------
            if job.phase == "GO_NEXT":
                if now - job.ts > self.go_next_timeout:
                    self._fail_job(job, "go_next_timeout")
                    continue

                mid = f"{job.job_id}__next"
                mr = self._consume_mission_result(mid)
                if mr is None:
                    continue

                if not mr.get("ok", False):
                    self._fail_job(job, f"next_mission_failed:{mr.get('reason','')}")
                    continue

                # ✅ IMPORTANT: GO_NEXT 이후에도 job.assigned_amr 사용
                amr = str(job.assigned_amr).strip()
                if not amr:
                    self._fail_job(job, "go_next_ok_but_no_assigned_amr")
                    continue

                job.phase = "DONE"
                job.ts = now
                self.get_logger().info(f"[JOB {job.job_id}] {amr} NEXT_OK -> DONE")
                continue

            # ---------------- DONE -> WAIT_RETURN or FINISHED ----------------
            if job.phase == "DONE":
                if (not self.return_to_wait_enabled) or (not job.requires_amr):
                    if job.requires_amr:
                        self._free_amr(job.assigned_amr)
                    job.phase = "FINISHED"
                    job.ts = now
                    continue

                # wait 미션 발행 + WAIT_RETURN
                amr = str(job.assigned_amr).strip()
                if not amr:
                    self._fail_job(job, "done_requires_wait_return_but_no_assigned_amr")
                    continue

                # wait mission id 규칙: suffix 사용
                wait_mid = f"{job.job_id}{self.wait_mission_suffix}"
                target_station = self.wait_station_by_robot.get(amr, self.wait_station)

                self._send_mission_request(
                    robot=amr,
                    mission_id=wait_mid,
                    station=target_station,
                )
                job.phase = "WAIT_RETURN"
                job.ts = now
                self.get_logger().info(f"[WAIT] {amr} -> WAIT_RETURN (mission_id={wait_mid})")
                continue

            # ---------------- WAIT_RETURN (synchronize wait arrival) ----------------
            if job.phase == "WAIT_RETURN":
                amr = str(job.assigned_amr).strip()
                if not amr:
                    self._fail_job(job, "wait_return_but_no_assigned_amr")
                    continue

                if now - job.ts > self.wait_return_timeout:
                    # ✅ 정책: wait 실패/timeout이어도 AMR은 풀어준다 (운영 편의)
                    self.get_logger().warn(f"[JOB {job.job_id}] WAIT_RETURN timeout -> free amr anyway")
                    self._free_amr(amr)
                    job.phase = "FINISHED"
                    job.ts = now
                    continue

                wait_mid = f"{job.job_id}{self.wait_mission_suffix}"
                mr = self._consume_mission_result(wait_mid)
                if mr is None:
                    continue

                if not mr.get("ok", False):
                    self.get_logger().warn(
                        f"[JOB {job.job_id}] WAIT mission failed: {mr.get('reason','')} (free anyway)"
                    )

                self._free_amr(amr)
                job.phase = "FINISHED"
                job.ts = now
                continue

        # ✅ FINISHED/FAIL 정리
        self.jobs = [j for j in self.jobs if j.phase not in ("FINISHED", "FAIL")]

    # ---------------- Helpers ----------------
    def _wp_of_station(self, station: str) -> str:
        return str(self.station_to_wp.get(station, "")).strip()
    
    def _consume_mission_result(self, mission_id: str) -> Optional[Dict[str, Any]]:
        """
        mission_result를 '한 번만' 소비(pop)해서
        같은 결과로 상태가 중복 진행되는 걸 방지.
        """
        if mission_id not in self.mission_cache:
            return None
        return self.mission_cache.pop(mission_id, None)
    
    # 핑키에게 보내는 미션
    def _send_mission_request(self, robot: str, mission_id: str, station: str):
        wp = self._wp_of_station(station)
        if not wp:
            self.get_logger().warn(f"[MISSION] unknown station mapping: {station}")
            return
        
        # ✅ docking stations(복수)에 포함되면 ArUco follow ON
        do_follow_aruco = (station in self.docking_stations)

        # ✅ do_follow_aruco=True 이면 marker_id를 station 매핑에서 넣어준다
        marker_id = 0
        if do_follow_aruco:
            marker_id = int(self.station_to_aruco_id.get(station, 0))
            if marker_id <= 0:
                # marker_id가 없으면 FOLLOW가 marker_id=0으로 실패하므로 방어
                self.get_logger().warn(
                    f"[MISSION] docking requested but no valid marker_id for station={station} "
                    f"(station_to_aruco_id_json에 추가 필요). do_follow_aruco forced False"
                )
                do_follow_aruco = False

        pose = self.station_to_pose.get(station)  # {"x":..,"y":..,"yaw":..} or None

        payload = {
        "type": "GOAL",
        "robot": robot,
        "mission_id": mission_id,
        "goal_wp": wp,
        "do_follow_aruco": bool(do_follow_aruco),
        "timeout_sec": 120.0,
        }

        # ✅ follow_aruco 사용이면 marker_id 포함
        if do_follow_aruco:
            payload["marker_id"] = int(marker_id)

        if isinstance(pose, dict):
            payload["final_pose"] = pose
            payload["final_use_waypoints"] = False

        self.mission_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))

    # “AMR이 스테이션에 도착했을 때, 해당 스테이션 담당 로봇팔에게 ‘적재/하역 작업 시작’ 명령을 보내는 함수”
    # 현재는 각 로봇팔별로 할 일이 정해져 있음.
    def _start_arm_job(self, job: Job, assigned_amr: str):
        role = self.arm_action.get(job.work_arm, "LOAD").strip().upper() or "LOAD"
        cmd = "START_LOAD" if role == "LOAD" else "START_UNLOAD"

        self._send_arm_command(job.work_arm, {
            "cmd": cmd,
            "job_id": job.job_id,
            "assigned_amr": assigned_amr,
            "job_type": job.job_type,
            "work_station": job.work_station,
            "next_station": job.next_station,
        })
            # 예를들어:
            #     {
            #     "cmd": "START_UNLOAD",
            #     "job_id": "J100",
            #     "assigned_amr": "pinky1",
            #     "job_type": "AMR_TO_QC",
            #     "work_station": "QC_ZONE",
            #     "next_station": "LOADING_ZONE"
            #     }
            # ✅ 의미:
            #     “arm1아, 지금 pinky1이 QC_ZONE에 도착했으니 J100 작업으로 하역 시작해!”

    def _send_arm_command(self, arm: str, cmd: Dict[str, Any]):
        pub = self.arm_cmd_pub.get(arm)
        if not pub:
            return
        pub.publish(String(data=json.dumps(cmd)))

    def _free_amr(self, amr: str):
        if not amr:
            return
        self.amr_busy[amr] = False
        self.amr_job[amr] = ""

    def _is_rtb_reason(self, reason: str) -> bool:
        s = (reason or "").lower()
        # TM이 넣는 reason 문자열에 맞춰 키워드 조정
        return ("battery" in s) or ("low" in s and "v" in s) or ("rtb" in s) or ("return" in s)

    def _fail_job(self, job: Job, reason: str):
        self.get_logger().warn(f"[JOB {job.job_id}] FAIL reason={reason}")
        job.fail_reason = reason
        job.ts = time.time()
    
        # ✅ RTB/배터리 실패면: AMR을 즉시 FREE로 풀지 않는다.
        if job.assigned_amr and self.return_to_wait_enabled and self._is_rtb_reason(reason):
            amr = str(job.assigned_amr).strip()
            # wait 미션 발행 + WAIT_RETURN로 전환 (현재 DONE에서 하던 로직 재사용)
            wait_mid = f"{job.job_id}{self.wait_mission_suffix}"
            target_station = self.wait_station_by_robot.get(amr, self.wait_station)

            self._send_mission_request(
                robot=amr,
                mission_id=wait_mid,
                station=target_station,
            )
            job.phase = "WAIT_RETURN"
            self.get_logger().warn(
                f"[JOB {job.job_id}] FAIL but RTB detected -> keep {amr} busy until WAIT_RETURN done "
                f"(mission_id={wait_mid}, target={target_station})"
            )
            return

        # ✅ 일반 FAIL은 기존처럼 즉시 FREE
        job.phase = "FAIL"
        if job.assigned_amr:
            self._free_amr(job.assigned_amr)


def main(args=None):
    rclpy.init(args=args)
    node = OrchestratorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

"""
========================
✅ 테스트 실행문 정리
========================

[1] Orchestrator 실행
ros2 run orchestrator orchestrator

(특정 로봇만 쓰고 싶으면)
ros2 run orchestrator orchestrator --ros-args -p robots:="['pinky3']"


------------------------------------------------------------
[2] AMR_TO_QC  (work=LOADING_ZONE -> next=QC_ZONE)
------------------------------------------------------------
# job 생성
ros2 topic pub --once /orchestrator/job_request std_msgs/msg/String \
"{data: '{\"job_id\":\"J100\",\"job_type\":\"AMR_TO_QC\"}'}"

# (수동 테스트) TrafficManager 대신 mission_result 주입
ros2 topic pub --once /pinky1/mission_result std_msgs/msg/String \
"{data: '{\"mission_id\":\"J100__work\",\"ok\":true}'}"

# (arm이 붙는 구조로 쓰는 경우에만) arm1 완료 주입
ros2 topic pub --once /arm1/state std_msgs/msg/String \
"{data: '{\"arm\":\"arm1\",\"state\":\"LOADED\",\"job_id\":\"J100\",\"assigned_amr\":\"pinky1\"}'}"

# next 완료 주입 (move-only거나 arm 완료 후 next로 갔을 때)
ros2 topic pub --once /pinky1/mission_result std_msgs/msg/String \
"{data: '{\"mission_id\":\"J100__next\",\"ok\":true}'}"

# wait 복귀 완료 주입(RETURN_TO_WAIT enabled일 때 AMR FREE는 wait OK 후에 됨)
ros2 topic pub --once /pinky1/mission_result std_msgs/msg/String \
"{data: '{\"mission_id\":\"J100__wait\",\"ok\":true}'}"


------------------------------------------------------------
[3] ASSEMBLY_TO_MODULE_STORAGE (work=ASSEMBLY_ZONE, arm3, next=MODULE_STORAGE)
------------------------------------------------------------
# job 생성
ros2 topic pub --once /orchestrator/job_request std_msgs/msg/String \
"{data: '{\"job_id\":\"J200\",\"job_type\":\"ASSEMBLY_TO_MODULE_STORAGE\"}'}"

# work 미션 완료(도착) 주입
ros2 topic pub --once /pinky2/mission_result std_msgs/msg/String \
"{data: '{\"mission_id\":\"J200__work\",\"ok\":true}'}"

# arm3 완료(여기 assigned_amr 반드시 pinky2로!)
ros2 topic pub --once /arm3/state std_msgs/msg/String \
"{data: '{\"arm\":\"arm3\",\"state\":\"LOADED\",\"job_id\":\"J200\",\"assigned_amr\":\"pinky2\",\"action_mode\":\"LOAD\"}'}"

# next 완료 주입
ros2 topic pub --once /pinky2/mission_result std_msgs/msg/String \
"{data: '{\"mission_id\":\"J200__next\",\"ok\":true}'}"

# wait 완료 주입
ros2 topic pub --once /pinky2/mission_result std_msgs/msg/String \
"{data: '{\"mission_id\":\"J200__wait\",\"ok\":true}'}"


------------------------------------------------------------
[4] MODULE_STORAGE_TO_SHIP (move-only, arm 없음)
------------------------------------------------------------
ros2 topic pub --once /orchestrator/job_request std_msgs/msg/String \
"{data: '{\"job_id\":\"J500\",\"job_type\":\"MODULE_STORAGE_TO_SHIP\"}'}"

# work 완료
ros2 topic pub --once /pinky3/mission_result std_msgs/msg/String \
"{data: '{\"mission_id\":\"J500__work\",\"ok\":true}'}"

# next 완료
ros2 topic pub --once /pinky3/mission_result std_msgs/msg/String \
"{data: '{\"mission_id\":\"J500__next\",\"ok\":true}'}"

# wait 완료 (AMR FREE)
ros2 topic pub --once /pinky3/mission_result std_msgs/msg/String \
"{data: '{\"mission_id\":\"J500__wait\",\"ok\":true}'}"
"""

