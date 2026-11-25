#!/usr/bin/env python3
import math
import json
import time
import collections
import numpy as np
from dataclasses import dataclass
import os
import yaml
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Point, PoseStamped
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray


# 주행 모드 (로그/속도 결정용)
FOLLOW = "FOLLOW"   # 레이스라인 추종 (기본)
AVOID = "AVOID"     # 장애물 회피/복귀 경로 추종 (DP 계획 경로)
STOP = "STOP"       # 통과 가능한 경로가 없어 정지


def smoothstep(u):
    """0~1 구간 3차 보간 (양 끝 기울기 0)."""
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


@dataclass
class MPCSettings:
    """ Dataclass to store MPC settings and hyperparameters """
    node: Node

    def __post_init__(self):
        gp = lambda name: self.node.get_parameter(name).value
        # Fallback cruising speed (used when the raceline speed is invalid)
        self.candidate_speed = gp("candidate_speed")

        # 후보 경로 시뮬레이션
        self.horizon = gp("horizon")
        self.dt = gp("dt")

        # Vehicle Parameters
        self.veh_length = gp("veh_length")
        self.veh_width = gp("veh_width")
        self.wheelbase = gp("wheelbase")
        self.min_steer = gp("min_steer")
        self.max_steer = gp("max_steer")
        self.max_steer_rate = gp("max_steer_rate")
        self.max_speed = gp("max_speed")
        self.min_speed = gp("min_speed")
        self.max_accel = gp("max_accel")


@dataclass
class VehicleState:
    x: float = 0.0   # x position [m]
    y: float = 0.0   # y position [m]
    v: float = 0.0   # velocity [m/s]
    yaw: float = 0.0  # heading angle [rad]


class MPCNode(Node):
    """
    레이스라인 추종을 기본으로 하고, 앞쪽 계획 구간 안에 장애물이 있으면
    레이스라인 기준 Frenet 격자 위에서 동적계획법(DP)으로 횡오프셋 d(s) 를 계획한다.
    계획 구간 안의 장애물을 한꺼번에 고려하므로 가까이 붙은 두 장애물(시케인, 같은 쪽 연속,
    두 장애물 사이 통과)도 하나의 경로로 처리한다. 장애물을 지나면 같은 DP 가 레이스라인으로
    부드럽게 복귀시킨다. 통과 가능한 경로가 없으면 STOP.

    경로는 Pure Pursuit 로 추종하고, 상수 조향 후보(fan)는 충돌 안전 점검 용도로만 쓴다.

    좌표 규약: 레이스라인 기준 Frenet. s = 진행 방향 거리(ego 기준 상대값),
    d = 레이스라인에서 왼쪽이 +인 횡방향 오프셋.
    """

    # 장애물 메모리 격자 크기 [m] (s, d 모두)
    MEM_CELL = 0.10
    # DP 횡방향 격자 간격 [m]
    DP_DD = 0.04
    # DP 가 고려하는 레이스라인 좌우 최대 오프셋 [m]
    DP_DMAX = 1.6

    def __init__(self):
        super().__init__('mpc_node')

        self.setup_parameters()
        self.settings = MPCSettings(self)
        self.load_avoidance_parameters()
        self.prev_steer = 0.0
        self.last_steer_update = time.monotonic()
        self.last_scan = None
        self.use_real = self.get_parameter("real_environment").value
        self.map_filename = self.get_parameter("map_name").value
        self.map_dir = self.get_parameter("map_dir").value
        self.trajectory_key = self.get_parameter("trajectory_key").value

        # Ego pose at current step (global)
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.current_speed = 0.0
        self.last_pose_for_speed = None   # 실차(PoseStamped) 속도 추정용 (t, x, y)

        self.path_steer = 0.0   # 현재 활성 경로의 Pure Pursuit 조향
        self.max_sight = self.get_parameter("max_sight").value
        self.all_candidates_blocked = False

        # 계획 상태
        self.mode = FOLLOW
        self.scan_generation = 0
        self.last_mem_scan_generation = 0
        self.prev_idx = None
        self.lidar_local = np.zeros((0, 2))
        self.mem = {}                      # (si, di) -> [s_g, d, hits, last_seen]
        self.plan_s_g = None               # 직전 계획 (트랙 전역 s) / 오프셋
        self.plan_d = None
        self.plan_min_clear = math.inf     # 직전 계획의 앞쪽 최소 여유 [m]
        self.plan_relaxed = False          # 완화 조건으로 구한 계획인지
        self.plan_infeasible = False       # 이번 주기에 경로를 찾지 못했는지
        self.infeasible_count = 0
        self.last_log_time = 0.0
        self.prev_speed_cmd = 0.0
        self.pose_hist = collections.deque(maxlen=60)   # (stamp, x, y, yaw): scan 시각의 자세를 찾는 데 사용
        self.recovering = False                          # 저속 복구 모드 (공칭 경로가 없을 때)
        self.fan_checks = 0                # 안전층 점검 횟수 / PP 조향을 다른 후보로 바꾼 횟수 (진단용)
        self.fan_overrides = 0

        # Topics
        # The simulator's fused car state is republished by frenet_odom_republisher.
        pose_topic = "/pf/viz/inferred_pose" if self.use_real else "/car_state/odom"
        drive_topic = "/drive"

        # Subscribers (속도는 같은 메시지에서 읽는다)
        self.pose_sub = self.create_subscription(
            PoseStamped if self.use_real else Odometry, pose_topic, self.pose_update, 1)
        self.lidar_sub = self.create_subscription(LaserScan, "/scan", self.lidar_callback, 10)

        # Publishers for drive commands and visualization markers
        self.drive_pub = self.create_publisher(AckermannDriveStamped, drive_topic, 1)
        self.drive_msg = AckermannDriveStamped()
        self.ref_traj_pub = self.create_publisher(Marker, "/ref_traj_marker", 1)
        self.ref_traj_marker = Marker()
        self.wpts_pub = self.create_publisher(Marker, "/waypoints_marker", 1)
        self.wpts_marker = Marker()
        self.pred_path_pub = self.create_publisher(MarkerArray, "/pred_path_marker", 1)
        self.obs_pub = self.create_publisher(Marker, "/tspc_obstacles_marker", 1)

        # Load waypoints from JSON file
        waypoints_file = os.path.join(self.map_dir, 'global_waypoints.json')
        try:
            with open(waypoints_file, 'r', encoding='utf-8') as f:
                waypoint_data = json.load(f)[self.trajectory_key]['wpnts']
            self.waypoints = np.asarray([
                [wp['id'], wp['x_m'], wp['y_m'], wp['psi_rad'],
                 wp['kappa_radpm'], wp['vx_mps']]
                for wp in waypoint_data
            ], dtype=float)

            # 레이스라인에서 좌/우 트랙 경계까지의 거리. 없으면 기본 반폭으로 대체한다.
            default_hw = float(self.get_parameter("track_half_width_default").value)
            has_bounds = all(('d_left' in wp and 'd_right' in wp) for wp in waypoint_data)
            if not has_bounds:
                self.get_logger().warn(
                    "Raceline has no d_left/d_right; using track_half_width_default="
                    f"{default_hw:.2f} m for both sides.")
            self.d_left = np.asarray(
                [float(wp['d_left']) if has_bounds else default_hw for wp in waypoint_data],
                dtype=float)
            self.d_right = np.asarray(
                [float(wp['d_right']) if has_bounds else default_hw for wp in waypoint_data],
                dtype=float)
        except (OSError, KeyError, ValueError, TypeError) as exc:
            raise RuntimeError(
                f"Could not load TSPC raceline from {waypoints_file} "
                f"(trajectory_key={self.trajectory_key!r}): {exc}"
            ) from exc

        # 시작점과 끝점이 같은 폐곡선이면 중복점 제거 (길이 0 구간 방지)
        if (len(self.waypoints) > 2 and
                np.hypot(*(self.waypoints[-1, 1:3] - self.waypoints[0, 1:3])) < 1e-3):
            self.waypoints = self.waypoints[:-1]
            self.d_left = self.d_left[:-1]
            self.d_right = self.d_right[:-1]
        if self.waypoints.ndim != 2 or self.waypoints.shape[0] < 3:
            raise RuntimeError(f"Raceline in {waypoints_file} must have at least three waypoints")
        self.d_left = np.maximum(self.d_left, 0.0)
        self.d_right = np.maximum(self.d_right, 0.0)

        self.get_logger().info(
            f"Loaded {len(self.waypoints)} waypoints from {waypoints_file} ({self.trajectory_key})")
        self.speed_scale = self.get_parameter("speed_scale").value
        self.speed_profile = self.waypoints[:, 5] * self.speed_scale

        self.build_raceline_geometry()
        self.speed_profile = self.limit_speed_profile(self.speed_profile, np.abs(self.waypoints[:, 4]))
        self.get_logger().info(
            f"Speed profile: max {self.speed_profile.max():.2f} m/s, min {self.speed_profile.min():.2f} m/s, "
            f"est. lap {np.sum(self.seg_len / np.maximum(self.speed_profile, 0.1)):.2f} s")

        self.load_wall_map()
        self.show_waypoints()

        self.get_logger().info("Raceline-follow / DP obstacle-avoidance node initialized.")
        self.get_logger().info(
            f"Listening to state on {pose_topic}, LiDAR on /scan; publishing commands on {drive_topic}")

    # -----------------------------------------------------------
    # Parameters cache / raceline geometry
    # -----------------------------------------------------------
    def load_avoidance_parameters(self):
        gp = lambda name: self.get_parameter(name).value
        self.veh_half = 0.5 * self.settings.veh_width

        self.lookahead = float(gp("lookahead_distance"))
        self.lookahead_gain = float(gp("lookahead_speed_gain"))
        self.lookahead_max = float(gp("lookahead_max"))
        self.lookahead_min = float(gp("lookahead_min"))
        self.tracker = str(gp("tracker")).lower()
        # 회피(AVOID) 중에만 쓸 추종기: "same" = tracker 와 동일, "pp" = 저속 회피용 Pure Pursuit
        self.avoid_tracker = str(gp("avoid_tracker")).lower()
        self.avoid_tracker_max_v = float(gp("avoid_tracker_max_speed"))
        self.stanley_k = float(gp("stanley_k"))
        self.stanley_k_heading = float(gp("stanley_k_heading"))
        self.stanley_v_soft = float(gp("stanley_v_soft"))
        self.front_axle_offset = float(gp("front_axle_offset"))
        self.understeer_gain = float(gp("understeer_gain"))
        self.kappa_preview_t = float(gp("kappa_preview_time"))
        self.steer_lookahead_t = float(gp("steer_lookahead_time"))
        self.speed_lookahead_t = float(gp("speed_lookahead_time"))
        self.avoid_lookahead = float(gp("avoid_lookahead_distance"))
        self.avoid_lookahead_gain = float(gp("avoid_lookahead_speed_gain"))
        self.rear_axle_offset = float(gp("pp_rear_axle_offset"))
        self.steer_lpf = float(np.clip(gp("steer_filter_alpha"), 0.0, 1.0))

        self.detect_distance = float(gp("obstacle_detect_distance"))
        self.obstacle_fov = math.radians(float(gp("obstacle_fov_deg")))
        self.confirm_scans = max(int(gp("obstacle_confirm_scans")), 1)
        self.lidar_stride = max(int(gp("lidar_stride")), 1)
        # LiDAR 원점이 차량 기준점(pose)보다 앞에 있는 거리 [m] (시뮬/실차 laser 프레임 0.27 m)
        self.lidar_x_offset = float(gp("lidar_x_offset"))
        self.scan_range_limit = self.detect_distance + 1.0
        self.wall_margin = float(gp("track_wall_margin"))
        self.mem_time = float(gp("obstacle_memory_time"))
        self.use_map_filter = bool(gp("use_map_wall_filter"))
        self.map_wall_margin = float(gp("map_wall_margin"))
        self.obs_depth = float(gp("obstacle_assumed_depth"))

        # 차체 가장자리 ~ 장애물 점 사이 간격: 최소(hard) / 원하는(soft) 값
        self.min_clear = float(gp("avoid_min_clearance"))
        self.clear = max(float(gp("avoid_clearance")), self.min_clear)
        # 차체 가장자리 ~ 트랙 경계 사이 간격: 최소 / 원하는 값
        self.edge_min = float(gp("track_boundary_min_margin"))
        self.edge_soft = max(float(gp("track_boundary_margin")), self.edge_min)
        self.s_margin = float(gp("avoid_longitudinal_margin"))
        self.collision_radius = self.veh_half + float(gp("collision_margin"))
        self.safety_min_speed = float(gp("safety_min_rollout_speed"))
        # "path": 계획 경로를 먼저 검사하고 막혔을 때만 fan 후보 사용 / "fan": 항상 상수 조향 후보 검사
        self.safety_mode = str(gp("safety_mode")).lower()
        self.safety_check_t = float(gp("safety_check_time"))
        self.smooth_passes = max(int(gp("plan_smooth_passes")), 0)
        self.replan_reset = float(gp("replan_reset_distance"))
        self.planner = str(gp("planner")).lower()
        self.lattice_len_times = list(gp("lattice_transition_times"))
        self.lattice_min_len = float(gp("lattice_min_length"))
        self.lattice_d_stride = max(int(gp("lattice_d_stride")), 1)
        self.w_latacc = float(gp("plan_w_lat_accel"))
        self.lattice_compare_dp = bool(gp("lattice_compare_dp"))

        self.plan_ds = float(gp("plan_ds"))
        self.max_slope = float(gp("plan_max_lateral_slope"))
        # 공칭 조건으로 경로가 없을 때 재시도: 최소 여유 x relax_factor, 최대 기울기 x relax_slope
        self.relax_factor = float(gp("plan_relax_clearance_factor"))
        self.relax_slope = float(gp("plan_relax_slope_factor"))
        self.w_off = float(gp("plan_w_offset"))
        self.w_slope = float(gp("plan_w_slope"))
        self.w_clear = float(gp("plan_w_clearance"))
        self.w_prev = float(gp("plan_w_consistency"))
        self.prev_decay = max(float(gp("plan_consistency_decay")), 0.1)
        self.clear_speed_gain = float(gp("clearance_speed_gain"))
        self.follow_clearance = float(gp("follow_ignore_clearance"))
        self.avoid_curve_ay = float(gp("avoid_curve_lat_accel"))
        hg = float(gp("clearance_speed_gain_hard"))
        self.clear_speed_gain_hard = self.clear_speed_gain if hg < 0.0 else hg
        self.stop_after = max(int(gp("plan_stop_after_infeasible")), 1)
        self.recover_enable = bool(gp("recovery_enable"))
        self.recover_speed = float(gp("recovery_speed"))
        self.recover_slope = float(gp("recovery_max_lateral_slope"))
        self.recover_clear_factor = float(gp("recovery_clearance_factor"))
        self.recover_footprint = float(gp("recovery_footprint_half"))
        self.scan_pose_sync = bool(gp("scan_pose_sync"))

        self.ay_max = float(gp("speed_ay_max"))
        self.ax_max = float(gp("speed_ax_max"))
        self.dec_max = float(gp("speed_dec_max"))
        self.avoid_ay_max = float(gp("avoid_ay_max"))
        self.avoid_speed = float(gp("avoid_speed"))
        self.tight_speed = float(gp("avoid_tight_speed"))
        self.rejoin_distance = max(float(gp("rejoin_distance")), 0.1)
        self.rejoin_tol = float(gp("rejoin_done_tolerance"))
        self.rejoin_heading_tol = float(gp("rejoin_heading_tolerance"))
        self.kappa_smooth = max(int(gp("path_kappa_smooth_points")) // 2 * 2 + 1, 1)
        self.cmd_accel = float(gp("speed_cmd_accel"))
        self.cmd_decel = float(gp("speed_cmd_decel"))
        self.avoid_dec = float(gp("avoid_approach_decel"))

    def build_raceline_geometry(self):
        """레이스라인 누적거리, 접선/좌측 법선, 윈도우 크기를 미리 계산한다 (폐곡선 가정)."""
        x = self.waypoints[:, 1]
        y = self.waypoints[:, 2]
        n = len(x)

        seg = np.hypot(np.roll(x, -1) - x, np.roll(y, -1) - y)
        seg = np.maximum(seg, 1e-6)
        self.seg_len = seg                                  # i -> i+1 구간 길이
        self.s = np.concatenate(([0.0], np.cumsum(seg[:-1])))
        self.track_length = float(np.sum(seg))
        self.ds_mean = self.track_length / n

        tx = np.roll(x, -1) - np.roll(x, 1)
        ty = np.roll(y, -1) - np.roll(y, 1)
        norm = np.maximum(np.hypot(tx, ty), 1e-9)
        self.tan = np.stack([tx / norm, ty / norm], axis=1)
        self.nrm = np.stack([-self.tan[:, 1], self.tan[:, 0]], axis=1)  # 왼쪽이 +

        # 차량 뒤쪽 일부도 포함 (옆/뒤에 있는 장애물 점을 올바른 s 로 투영하기 위해)
        self.window_back = int(min(n // 4, math.ceil(1.0 / self.ds_mean)))
        window_len = self.detect_distance + self.rejoin_distance + 3.0
        self.window_points = int(min(n - self.window_back,
                                     math.ceil(window_len / self.ds_mean) + 2))

    def limit_speed_profile(self, v, kappa):
        """
        레이스라인 속도를 차량 한계로 다시 제한한다 (폐곡선 가정).
          v <= sqrt(speed_ay_max / |kappa|)  (횡가속도 한계)
          전/후방 패스로 가속 speed_ax_max, 감속 speed_dec_max 를 지키게 한다.
        speed_ay_max <= 0 이면 제한하지 않는다.
        """
        v = np.asarray(v, dtype=float).copy()
        v = np.where(np.isfinite(v), v, self.settings.candidate_speed)
        v = np.clip(v, self.settings.min_speed, self.settings.max_speed)
        if self.ay_max <= 0.0:
            return v
        v = np.minimum(v, np.sqrt(self.ay_max / np.maximum(kappa, 1e-4)))
        n = len(v)
        ds = self.seg_len
        for _ in range(2):
            for i in range(1, 2 * n):                    # 가속 한계 (앞으로)
                a, b = i % n, (i - 1) % n
                v[a] = min(v[a], math.sqrt(v[b] ** 2 + 2.0 * self.ax_max * ds[b]))
            for i in range(2 * n, 0, -1):                # 감속 한계 (뒤로)
                a, b = (i - 1) % n, i % n
                v[a] = min(v[a], math.sqrt(v[b] ** 2 + 2.0 * self.dec_max * ds[a]))
        return np.clip(v, self.settings.min_speed, self.settings.max_speed)

    # -----------------------------------------------------------
    # Callbacks
    # -----------------------------------------------------------
    def lidar_callback(self, scan_msg):
        """ Store the latest LaserScan. """
        self.last_scan = scan_msg
        self.scan_generation += 1

    def odometry_update(self, msg):
        """(호환용) 속도 갱신. pose_update 에서도 같은 메시지로 갱신한다."""
        if not self.use_real:
            lin = msg.twist.twist.linear
            self.current_speed = math.hypot(lin.x, lin.y)

    def estimate_speed_from_pose(self, x, y):
        """실차 PoseStamped 에는 속도가 없으므로 위치 차분 + 저역통과로 추정한다."""
        now = time.monotonic()
        if self.last_pose_for_speed is not None:
            t0, x0, y0 = self.last_pose_for_speed
            dt = now - t0
            if 1e-3 < dt < 0.5:
                # 로컬라이제이션 점프로 속도가 튀지 않도록 상한을 둔다
                v = min(math.hypot(x - x0, y - y0) / dt, 1.5 * self.settings.max_speed)
                self.current_speed = 0.7 * self.current_speed + 0.3 * v
        self.last_pose_for_speed = (now, x, y)

    def pose_update(self, pose_msg):
        """
        Pose 메시지마다:
        1) 레이스라인 윈도우 / 2) 장애물 메모리 갱신 / 3) DP 경로 계획
        4) Pure Pursuit / 5) 후보 경로 안전 점검 후 주행 명령 발행
        """
        if self.use_real:
            p = pose_msg.pose
        else:
            p = pose_msg.pose.pose
            self.odometry_update(pose_msg)
        q = p.orientation
        state = VehicleState(
            x=p.position.x, y=p.position.y,
            yaw=math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y ** 2 + q.z ** 2)))
        if self.use_real:
            self.estimate_speed_from_pose(state.x, state.y)
        state.v = self.current_speed
        self.current_x, self.current_y, self.current_yaw = state.x, state.y, state.yaw
        st_ = pose_msg.header.stamp
        t_stamp = st_.sec + st_.nanosec * 1e-9
        if t_stamp > 0.0:
            self.pose_hist.append((t_stamp, state.x, state.y, state.yaw))

        win = self.build_window(state)
        if self.scan_generation != self.last_mem_scan_generation:
            self.last_mem_scan_generation = self.scan_generation
            self.update_lidar_points()
            self.update_obstacle_memory(win)
        obs = self.memory_obstacles(win)

        off, planned = self.plan_offsets(win, obs)
        path = self.build_active_path(win, off)
        self.show_reference_traj(np.vstack((path['x'], path['y'])))

        v = max(self.current_speed, 0.0)
        if self.mode == AVOID:
            ld = self.avoid_lookahead + self.avoid_lookahead_gain * v
        else:
            ld = self.lookahead + self.lookahead_gain * v
        ld = float(np.clip(ld, self.lookahead_min, self.lookahead_max))
        # 저속 회피용 PP 는 충분히 감속한 뒤에만 쓴다 (고속 PP 는 코너 안쪽을 가로질러 벽에 닿는다)
        use_pp = (self.recovering or (self.mode == AVOID and self.avoid_tracker == "pp")) \
            and self.current_speed <= self.avoid_tracker_max_v
        if self.tracker == "stanley" and not use_pp:
            self.path_steer = self.stanley_steer(path)
        else:
            self.path_steer = self.pure_pursuit_steer(path, ld)

        # ---------- 목표 속도 ----------
        v_race = win['v_race']
        v_lim = self.path_speed_limit(path)
        if self.mode == FOLLOW:
            v_target = min(v_race, v_lim)
        elif self.mode == AVOID:
            # 회피 속도 상한은 가장 가까운 앞쪽 장애물까지 거리에 맞춘 감속 곡선으로 적용 (급제동 방지)
            # 앞쪽(차 옆 포함) 장애물까지 거리. 이미 지나쳤으면 감속하지 않는다 (복귀 구간은 곡률 한계만 적용)
            ahead = obs['s'][obs['s'] > -0.5 * self.settings.veh_length] if obs['s'].size else obs['s']
            if ahead.size:
                gap = max(float(np.min(ahead)) - 0.5 * self.settings.veh_length, 0.0)
            else:
                gap = math.inf
            v_avoid = self.avoid_speed
            if self.avoid_curve_ay > 0.0 and ahead.size:
                # 장애물이 급코너에 있으면 회피 속도를 그 곡률에 맞게 낮춘다 (헤어핀 ~1.7 m/s, 직선은 avoid_speed)
                s_o = float(np.min(ahead))
                m = (win['s'] >= s_o - 1.0) & (win['s'] <= s_o + 1.5)
                k_o = float(np.max(np.abs(win['kappa'][m]))) if np.any(m) else 0.0
                v_avoid = min(v_avoid, max(math.sqrt(self.avoid_curve_ay / max(k_o, 1e-3)), self.tight_speed))
            v_av = math.sqrt(v_avoid ** 2 + 2.0 * self.avoid_dec * gap) if math.isfinite(gap) else math.inf
            v_target = min(v_race, v_av, v_lim)
            if self.plan_min_clear < self.clear or self.plan_relaxed:
                v_target = min(v_target, self.tight_speed)
        else:
            v_target = 0.0
        if self.recovering and self.mode != STOP:
            v_target = min(v_target, self.recover_speed)
        if self.plan_infeasible and self.mode != STOP:
            # 경로가 잠시 없을 때: 직전 계획을 유지하며 최대 감속 (바로 정지 명령을 내리면 급제동으로 불안정)
            v_target = min(v_target, max(self.current_speed - self.dec_max * 0.1, 0.0))
        if self.mode != STOP:
            v_target = float(np.clip(v_target, self.settings.min_speed, self.settings.max_speed))

        # ---------- 후보 경로 안전 점검 ----------
        steer_cmd, global_paths, best_idx, path_clear = self.run_candidate_mpc(state, path, v_target)
        # 속도 명령 증가율 제한 (모드 전환 등으로 목표 속도가 갑자기 뛰는 것 방지). 감속은 즉시 허용.
        dt_cmd = min(max(time.monotonic() - self.last_steer_update, 0.0), 0.1)
        # 가속 제한은 (직전 명령, 현재 속도) 중 큰 값 기준, 감속 제한은 직전 명령 기준.
        # (현재 속도 기준으로 감속을 제한하면 명령이 실제 속도 아래로 내려가지 못해 브레이크가 거의 안 걸린다)
        v_target = min(v_target, max(self.prev_speed_cmd, self.current_speed) + self.cmd_accel * dt_cmd)
        if self.mode != STOP:
            v_target = max(v_target, self.prev_speed_cmd - self.cmd_decel * dt_cmd)
        speed_cmd = v_target
        if self.mode == STOP:
            steer_cmd = self.path_steer
            speed_cmd = 0.0
        elif not path_clear:
            # 모든 후보가 충돌이면 진행하지 않고 정지 (조향은 경로 쪽으로)
            steer_cmd = self.path_steer
            speed_cmd = 0.0
            if not self.all_candidates_blocked:
                self.get_logger().error(
                    "All sampled trajectories are in collision; stopping until a clear path is available.")
            self.all_candidates_blocked = True
        else:
            self.all_candidates_blocked = False

        # 조향 저역통과 + 변화율 제한
        now = time.monotonic()
        elapsed = max(now - self.last_steer_update, 0.0)
        steer_cmd = self.prev_steer + self.steer_lpf * (steer_cmd - self.prev_steer)
        max_steer_delta = self.settings.max_steer_rate * elapsed
        steer_cmd = float(np.clip(steer_cmd, self.prev_steer - max_steer_delta,
                                  self.prev_steer + max_steer_delta))
        self.prev_steer = steer_cmd
        self.last_steer_update = now

        self.drive_msg.drive.steering_angle = steer_cmd
        self.prev_speed_cmd = speed_cmd
        self.drive_msg.drive.speed = (-1.0 if self.use_real else 1.0) * speed_cmd
        self.drive_pub.publish(self.drive_msg)

        if now - self.last_log_time > 0.5:
            self.last_log_time = now
            self.get_logger().info(
                f"mode={self.mode} v_race={v_race:.2f} v_odom={self.current_speed:.2f} "
                f"v_cmd={speed_cmd:.2f} steer={steer_cmd:.3f} d_ego={win['d_ego']:.2f} "
                f"obs_pts={obs['s'].size}")

        self.show_predicted_paths(global_paths, best_idx)
        self.show_obstacles(win, obs)

    def normalize_angle(self, angle):
        """Wrap angle to [-pi, pi]."""
        return math.atan2(math.sin(angle), math.cos(angle))

    # -----------------------------------------------------------
    # Raceline window (Frenet helpers)
    # -----------------------------------------------------------
    def find_nearest_index(self, p):
        """직전 인덱스 주변을 먼저 찾고, 너무 멀면 전체에서 다시 찾는다."""
        n = len(self.waypoints)
        xy = self.waypoints[:, 1:3]
        if self.prev_idx is not None:
            half = max(int(2.0 / self.ds_mean), 20)
            if 2 * half + 1 < n:
                cand = (self.prev_idx + np.arange(-half, half + 1)) % n
                d2 = np.sum((xy[cand] - p) ** 2, axis=1)
                j = int(np.argmin(d2))
                if d2[j] < 3.0 ** 2:
                    self.prev_idx = int(cand[j])
                    return self.prev_idx
        d2 = np.sum((xy - p) ** 2, axis=1)
        self.prev_idx = int(np.argmin(d2))
        return self.prev_idx

    def ahead_of(self, s_global, s_ego_global):
        """트랙 전역 s 가 ego 앞(+)/뒤(-)로 얼마나 떨어져 있는지."""
        L = self.track_length
        return (np.asarray(s_global) - s_ego_global + 0.5 * L) % L - 0.5 * L

    def build_window(self, state):
        """ego 주변(뒤 window_back ~ 앞쪽) 레이스라인 구간을 잘라 ego 기준 s, 경계 거리 등을 담는다."""
        n = len(self.waypoints)
        p = np.array([state.x, state.y])
        idx = self.find_nearest_index(p)
        nb = self.window_back
        ks = (idx + np.arange(-nb, self.window_points)) % n

        xy = self.waypoints[ks, 1:3]
        tan = self.tan[ks]
        nrm = self.nrm[ks]
        seg = self.seg_len[ks]
        s_rel = np.concatenate(([0.0], np.cumsum(seg[:-1])))
        s_rel = s_rel - s_rel[nb]                               # waypoint[idx] 기준 거리

        rel0 = p - xy[nb]
        along_ego = float(np.dot(rel0, tan[nb]))
        d_ego = float(np.dot(rel0, nrm[nb]))
        psi_err = self.normalize_angle(state.yaw - math.atan2(tan[nb, 1], tan[nb, 0]))

        # 속도 명령 지연을 고려해 v * speed_lookahead_time 만큼 앞의 레이스라인 속도를 쓴다
        ahead_pts = int(round(max(state.v, 0.0) * self.speed_lookahead_t / self.ds_mean))
        v_race = float(self.speed_profile[(idx + ahead_pts) % n])
        if not np.isfinite(v_race):
            v_race = float(self.settings.candidate_speed)
        v_race = float(np.clip(v_race, self.settings.min_speed, self.settings.max_speed))

        return {
            'idx': idx, 'xy': xy, 'tan': tan, 'nrm': nrm,
            'dl': self.d_left[ks], 'dr': self.d_right[ks],
            'v_prof': self.speed_profile[ks],
            'kappa': self.waypoints[ks, 4],
            's_rel': s_rel, 'along_ego': along_ego,
            's': s_rel - along_ego,                       # ego 위치 = 0
            's_ego_g': (self.s[idx] + along_ego) % self.track_length,
            'd_ego': d_ego, 'psi_err': psi_err, 'v_race': v_race,
        }

    # -----------------------------------------------------------
    # LiDAR -> track obstacles (with short memory)
    # -----------------------------------------------------------
    def load_wall_map(self):
        """
        정적 맵(occupancy grid)을 읽어 '가장 가까운 벽까지 거리' 지도를 만든다.
        d_left/d_right 경계가 실제 벽과 어긋난 구간에서 벽 LiDAR 점이 장애물로 잡히는 것을 막는다.
        (스폰 장애물은 맵에 없으므로 그대로 남는다.) 시뮬레이터처럼 <map>_orig.yaml 이 있으면 우선 사용.
        """
        self.wall_dist = None
        if not self.use_map_filter:
            return
        try:
            from PIL import Image
            from scipy.ndimage import distance_transform_edt
            yaml_path = os.path.join(self.map_dir, f"{self.map_filename}_orig.yaml")
            if not os.path.exists(yaml_path):
                yaml_path = os.path.join(self.map_dir, f"{self.map_filename}.yaml")
            with open(yaml_path, 'r', encoding='utf-8') as f:
                info = yaml.safe_load(f)
            img_path = info['image']
            if not os.path.isabs(img_path):
                img_path = os.path.join(os.path.dirname(yaml_path), img_path)
            img = np.asarray(Image.open(img_path).convert('L'), dtype=float) / 255.0
            occ_p = img if int(info.get('negate', 0)) else 1.0 - img
            occupied = occ_p >= float(info.get('occupied_thresh', 0.65))
            res = float(info['resolution'])
            self.wall_dist = distance_transform_edt(~occupied) * res
            self.map_res = res
            self.map_origin = (float(info['origin'][0]), float(info['origin'][1]))
            self.get_logger().info(
                f"Map wall filter enabled ({yaml_path}, margin {self.map_wall_margin:.2f} m)")
        except Exception as exc:  # noqa: BLE001 - 맵이 없어도 노드는 동작해야 한다
            self.wall_dist = None
            self.get_logger().warn(f"Map wall filter disabled: {exc}")

    def map_wall_distance(self, P):
        """전역 좌표 점들에서 가장 가까운 맵 벽까지 거리 [m]. 맵 밖은 0."""
        h, w = self.wall_dist.shape
        col = np.floor((P[:, 0] - self.map_origin[0]) / self.map_res).astype(int)
        row = h - 1 - np.floor((P[:, 1] - self.map_origin[1]) / self.map_res).astype(int)
        inside = (col >= 0) & (col < w) & (row >= 0) & (row < h)
        dist = np.zeros(P.shape[0])
        dist[inside] = self.wall_dist[row[inside], col[inside]]
        return dist

    def near_map_wall(self, P):
        """전역 좌표 점들이 맵 벽에서 map_wall_margin 이내(또는 맵 밖)인지."""
        if self.wall_dist is None:
            return np.zeros(P.shape[0], dtype=bool)
        return self.map_wall_distance(P) < self.map_wall_margin

    def update_lidar_points(self):
        """최신 스캔을 ego 좌표계 (x 전방, y 좌측) 점들로 변환한다."""
        scan = self.last_scan
        if scan is None:
            self.lidar_local = np.zeros((0, 2))
            return
        ranges = np.asarray(scan.ranges, dtype=float)
        if ranges.size == 0:
            self.lidar_local = np.zeros((0, 2))
            return
        angles = scan.angle_min + np.arange(ranges.size) * scan.angle_increment
        wrapped = np.arctan2(np.sin(angles), np.cos(angles))

        max_r = self.scan_range_limit
        if scan.range_max > 0.0:
            max_r = min(max_r, float(scan.range_max))
        min_r = max(float(scan.range_min), 0.05)

        valid = (np.isfinite(ranges) & (ranges > min_r) & (ranges < max_r)
                 & (np.abs(wrapped) <= 0.5 * self.obstacle_fov))
        sel = np.nonzero(valid)[0][::self.lidar_stride]
        r = ranges[sel]
        a = angles[sel]
        self.lidar_local = np.stack([r * np.cos(a) + self.lidar_x_offset, r * np.sin(a)], axis=1)

    def pose_at(self, t):
        """
        타임스탬프 t 시점의 자세 (x, y, yaw). pose 이력을 보간하고, 최신보다 늦으면 최대 50 ms 까지 외삽.
        scan(95 Hz)과 pose(49 Hz)는 시점이 달라서, 최신 pose 로 scan 을 변환하면 회전 중 장애물이 번진다.
        """
        cur = (self.current_x, self.current_y, self.current_yaw)
        if not self.scan_pose_sync or t <= 0.0 or len(self.pose_hist) < 2:
            return cur
        h = self.pose_hist
        if t >= h[-1][0]:
            a, b = h[-2], h[-1]
            t = min(t, b[0] + 0.05)
        elif t <= h[0][0]:
            return h[0][1:]
        else:
            j = next(i for i in range(len(h) - 1, 0, -1) if h[i - 1][0] <= t)
            a, b = h[j - 1], h[j]
        dt = b[0] - a[0]
        if dt <= 1e-6:
            return b[1:]
        u = (t - a[0]) / dt
        dyaw = self.normalize_angle(b[3] - a[3])
        return (a[1] + u * (b[1] - a[1]), a[2] + u * (b[2] - a[2]), a[3] + u * dyaw)

    def frenet_of_points(self, win, pts, pose=None):
        """ego 좌표계 점들을 레이스라인 Frenet (s: ego 기준, d) 로 변환하고 트랙 안 여부를 돌려준다."""
        px, py, pyaw = pose if pose is not None else (self.current_x, self.current_y, self.current_yaw)
        cy = math.cos(pyaw)
        sy = math.sin(pyaw)
        P = np.stack([
            px + cy * pts[:, 0] - sy * pts[:, 1],
            py + sy * pts[:, 0] + cy * pts[:, 1],
        ], axis=1)
        W = win['xy']
        diff = P[:, None, :] - W[None, :, :]               # (M, K, 2)
        k = np.argmin(np.sum(diff * diff, axis=2), axis=1)
        rel = diff[np.arange(P.shape[0]), k]
        along = np.sum(rel * win['tan'][k], axis=1)
        lat = np.sum(rel * win['nrm'][k], axis=1)
        s_p = win['s_rel'][k] + along - win['along_ego']
        in_track = ((lat < win['dl'][k] - self.wall_margin)
                    & (lat > -win['dr'][k] + self.wall_margin))
        in_track &= ~self.near_map_wall(P)
        return s_p, lat, in_track

    def update_obstacle_memory(self, win):
        """
        트랙 안쪽 LiDAR 점(벽 제외)을 트랙 전역 (s, d) 격자 메모리에 누적한다.
        - 같은 칸이 confirm_scans 번 이상 보여야 장애물로 인정 (노이즈 방지)
        - obstacle_memory_time 동안 다시 보이지 않거나 차량 뒤로 지나가면 삭제
        메모리 덕분에 차 옆으로 지나가는 중이거나 앞 장애물에 가려진 장애물도 유지된다.
        """
        now = time.monotonic()
        L = self.track_length
        s_ego_g = win['s_ego_g']
        pts = self.lidar_local
        if pts.shape[0]:
            sc = self.last_scan
            t_scan = (sc.header.stamp.sec + sc.header.stamp.nanosec * 1e-9) if sc is not None else 0.0
            s_p, lat, in_track = self.frenet_of_points(win, pts, self.pose_at(t_scan))
            keep = in_track & (s_p > -1.0) & (s_p < self.detect_distance)
            c = self.MEM_CELL
            seen = set()
            for sp, dp in zip(s_p[keep], lat[keep]):
                sg = (s_ego_g + sp) % L
                key = (int(sg // c), int(math.floor(dp / c)))
                cell = self.mem.get(key)
                if cell is None:
                    self.mem[key] = [sg, dp, 1, now]
                else:
                    cell[0], cell[1], cell[3] = sg, dp, now
                    if key not in seen:
                        cell[2] += 1
                seen.add(key)

        # 오래되었거나 이미 지나친 칸 제거
        dead = [key for key, cell in self.mem.items()
                if now - cell[3] > self.mem_time
                or self.ahead_of(cell[0], s_ego_g) < -1.5]
        for key in dead:
            del self.mem[key]

    def memory_obstacles(self, win):
        """확정된 메모리 점들을 ego 기준 (s, d) 로 돌려준다."""
        if not self.mem:
            return {'s': np.zeros(0), 'd': np.zeros(0)}
        arr = np.array([(c[0], c[1]) for c in self.mem.values() if c[2] >= self.confirm_scans])
        if arr.size == 0:
            return {'s': np.zeros(0), 'd': np.zeros(0)}
        s = self.ahead_of(arr[:, 0], win['s_ego_g'])
        return {'s': s, 'd': arr[:, 1]}

    # -----------------------------------------------------------
    # DP lateral planner
    # -----------------------------------------------------------
    def plan_offsets(self, win, obs):
        """
        레이스라인 위 각 waypoint 의 횡오프셋을 돌려준다. 모드(FOLLOW/AVOID/STOP)도 갱신.

        Frenet 격자 (s: plan_ds 간격, d: DP_DD 간격) 에서
          비용 = w_off*d^2 + w_slope*(Δd/Δs)^2 + w_clear*(여유 부족)^2 + w_prev*(d - 직전 계획)^2
        을 최소화하는 d(s) 를 DP 로 구한다. 장애물/트랙 경계의 최소 여유는 hard 제약.
        """
        s_win = win['s']
        d_ego = win['d_ego']
        in_range = (obs['s'] > -self.settings.veh_length) & (obs['s'] < self.detect_distance)
        relevant = obs['s'].size > 0 and np.any(in_range)
        if relevant and self.mode == FOLLOW and self.follow_clearance > 0.0:
            # 레이스라인에서 충분히 떨어진 장애물은 무시하고 레이스라인 속도 그대로 지나간다
            relevant = bool(np.any(in_range & (np.abs(obs['d']) < self.veh_half + self.follow_clearance)))
        if not relevant and self.mode != STOP and (self.mode == FOLLOW or (abs(d_ego) <= self.rejoin_tol
                                     and abs(win['psi_err']) <= self.rejoin_heading_tol)):
            # 장애물이 없으면 레이스라인 추종. (FOLLOW 중의 작은 추종 오차는 Pure Pursuit 가 처리하고,
            #  회피 직후에만 DP 경로로 레이스라인에 복귀한다)
            if self.mode != FOLLOW:
                self.set_mode(FOLLOW, "레이스라인 복귀 완료")
            self.plan_s_g = None
            self.plan_d = None
            self.plan_min_clear = math.inf
            self.plan_relaxed = False
            self.plan_infeasible = False
            self.infeasible_count = 0
            return np.zeros(len(s_win)), False

        ds = self.plan_ds
        H = max(self.detect_distance, ds * 4)
        st = np.arange(0.0, H + 1e-6, ds)                    # 계획 station (ego 기준 s)
        N = st.size
        dg = np.arange(-self.DP_DMAX, self.DP_DMAX + 1e-9, self.DP_DD)
        D = dg.size
        vh = self.veh_half

        # --- 트랙 경계 (station 별) ---
        dl = np.interp(st, s_win, win['dl'])
        dr = np.interp(st, s_win, win['dr'])
        room_l = dl[:, None] - vh - dg[None, :]               # 차 왼쪽 가장자리 ~ 왼쪽 경계
        room_r = dr[:, None] - vh + dg[None, :]
        room = np.minimum(room_l, room_r)
        if self.wall_dist is not None:
            # 맵의 실제 벽까지 거리로도 제한 (d_left/d_right 가 실제 벽과 어긋난 구간 대비)
            gx = np.interp(st, s_win, win['xy'][:, 0])
            gy = np.interp(st, s_win, win['xy'][:, 1])
            nx = np.interp(st, s_win, win['nrm'][:, 0])
            ny = np.interp(st, s_win, win['nrm'][:, 1])
            P = np.stack([(gx[:, None] + dg[None, :] * nx[:, None]).ravel(),
                          (gy[:, None] + dg[None, :] * ny[:, None]).ravel()], axis=1)
            room = np.minimum(room, self.map_wall_distance(P).reshape(N, D) - vh)
        # 속도가 빠를수록 추종 오차가 커지므로 요구 여유를 늘린다
        extra = self.clear_speed_gain * max(self.current_speed, 0.0)            # soft 여유 증가분
        hard_extra = self.clear_speed_gain_hard * max(self.current_speed, 0.0)  # hard 여유 증가분
        edge_soft = self.edge_soft + extra
        clear_soft = self.clear + extra
        node_cost = self.w_off * dg[None, :] ** 2 + self.w_clear * np.maximum(edge_soft - room, 0.0) ** 2

        # --- 장애물 (station 별 횡방향 여유) ---
        def make_clear_map(s_half, depth):
            """station 별, 횡오프셋 별 '차체 가장자리 ~ 가장 가까운 장애물 점' 거리."""
            cm = np.full((N, D), np.inf)
            if obs['s'].size:
                # 장애물 점은 보이는 면뿐이므로 진행 방향 뒤쪽으로 depth 만큼 이어져 있다고 본다
                lo = obs['s'] - s_half
                hi = obs['s'] + s_half + depth
                occ = (st[:, None] >= lo[None, :]) & (st[:, None] <= hi[None, :])   # (N, M)
                gap = np.abs(dg[:, None] - obs['d'][None, :]) - vh                    # (D, M)
                for k in range(N):
                    m = occ[k]
                    if np.any(m):
                        cm[k] = np.min(gap[:, m], axis=1)
            return cm

        clear_map = make_clear_map(0.5 * self.settings.veh_length + self.s_margin, self.obs_depth)
        if obs['s'].size:
            node_cost = node_cost + self.w_clear * np.maximum(clear_soft - clear_map, 0.0) ** 2

        # --- 직전 계획과의 일관성 (좌/우 선택이 매 주기 바뀌지 않도록) ---
        if self.plan_s_g is not None:
            prev_rel = self.ahead_of(self.plan_s_g, win['s_ego_g'])
            order = np.argsort(prev_rel)
            prev_rel = prev_rel[order]
            prev_d = self.plan_d[order]
            valid = (st >= prev_rel[0]) & (st <= prev_rel[-1])
            prev_k = np.interp(st, prev_rel, prev_d)
            # 가까운 구간일수록 직전 계획을 유지 (먼 구간은 새로 보인 정보로 좌/우를 바꿀 수 있게)
            w = self.w_prev * valid * np.exp(-st / self.prev_decay)
            node_cost = node_cost + w[:, None] * (dg[None, :] - prev_k[:, None]) ** 2

        slope0 = math.tan(float(np.clip(win['psi_err'], -0.8, 0.8)))
        d0 = float(np.clip(d_ego, dg[0], dg[-1]))
        # 차량이 직전 계획에서 크게 벗어나지 않았으면 직전 계획 위(현재 s)에서 이어서 계획한다.
        # 매번 실제 위치에서 다시 시작하면 언더스티어 등으로 밀린 만큼 경로도 따라 밀려 추종기가 보정하지 못한다.
        if self.plan_s_g is not None and self.replan_reset > 0.0:
            pr = self.ahead_of(self.plan_s_g, win['s_ego_g'])
            o = np.argsort(pr)
            pr, pd = pr[o], self.plan_d[o]
            if pr[0] <= 0.0 <= pr[-1]:
                dp0 = float(np.interp(0.0, pr, pd))
                if abs(dp0 - d_ego) < self.replan_reset:
                    d0 = float(np.clip(dp0, dg[0], dg[-1]))
                    slope0 = float(np.interp(0.0, pr, np.gradient(pd, pr)))

        def solve(min_clear, edge_min, max_slope, cm=None):
            """hard 제약(최소 여유, 최대 횡기울기)을 주고 DP 를 푼다. 해가 없으면 None."""
            cm = clear_map if cm is None else cm
            blocked = (room < edge_min) | (cm < min_clear)
            blocked[0] = False          # 현재 위치 station 은 제약하지 않는다 (이미 그 자리에 있음)
            nc = np.where(blocked, np.inf, node_cost)
            step = dg[None, :] - dg[:, None]                    # i -> j
            trans = self.w_slope * (step / ds) ** 2
            trans = np.where(np.abs(step) <= max_slope * ds + 1e-9, trans, np.inf)
            # 첫 구간은 현재 헤딩 방향을 반영 (d'(0) ≈ tan(psi_err))
            cost = self.w_slope * ((dg - d0) / ds - slope0) ** 2
            cost = np.where(np.abs(dg - d0) <= max_slope * ds + self.DP_DD, cost, np.inf)
            cost = cost + nc[1]
            back = np.zeros((N, D), dtype=np.int32)
            for k in range(2, N):
                tot = cost[:, None] + trans
                back[k] = np.argmin(tot, axis=0)
                cost = tot[back[k], np.arange(D)] + nc[k]
            j = int(np.argmin(cost))
            if not np.isfinite(cost[j]):
                return None
            path_idx = np.zeros(N, dtype=np.int64)
            path_idx[-1] = j
            for k in range(N - 1, 1, -1):
                path_idx[k - 1] = back[k][path_idx[k]]
            return dg[path_idx], blocked

        sol = None
        used_lattice = False
        if self.planner == "lattice":
            blocked = (room < self.edge_min + hard_extra) | (clear_map < self.min_clear + hard_extra)
            blocked[0] = False
            kref = np.interp(st, s_win, win['kappa'])
            nc_b = np.where(blocked, np.inf, node_cost)
            d_lat = self.lattice_solve(st, dg, nc_b, d0, slope0, kref)
            cands = []
            if d_lat is not None:
                cands.append(d_lat)
            # DP 해도 같은 비용(횡가속도 포함)으로 평가해 더 나은 쪽을 쓴다.
            # (격자 후보에 없는 '기다렸다가 짧게 옮기는' 경로도 DP 는 표현할 수 있다)
            dp_sol = (solve(self.min_clear + hard_extra, self.edge_min + hard_extra, self.max_slope)
                      if self.lattice_compare_dp else None)
            if dp_sol is not None:
                raw = dp_sol[0].copy()
                raw[0] = d0
                sm = raw.copy()
                for _ in range(self.smooth_passes):
                    sm[1:-1] = 0.25 * sm[:-2] + 0.5 * sm[1:-1] + 0.25 * sm[2:]
                cands.extend([sm, raw])
            if cands:
                costs = [self.lattice_path_cost(st, dg, nc_b, c, kref) for c in cands]
                b = int(np.argmin(costs))
                if np.isfinite(costs[b]):
                    sol = (cands[b], blocked)
                    used_lattice = True
        if sol is None:
            sol = solve(self.min_clear + hard_extra, self.edge_min + hard_extra, self.max_slope)
        self.plan_relaxed = False
        if sol is None:
            # 좁은 시케인 등: 최소 여유를 줄이고 더 급한 횡이동을 허용해 다시 시도 (저속 통과)
            sol = solve(self.min_clear * self.relax_factor + hard_extra, self.edge_min * self.relax_factor + hard_extra,
                        self.max_slope * self.relax_slope)
            self.plan_relaxed = sol is not None
        recovering = False
        if sol is None and self.recover_enable:
            # 저속 복구: 속도 비례 여유 없이 크게 완화한 조건으로 다시 풀고 recover_speed 로 빠져나간다
            # 저속에서는 차를 짧은 점유(진행 방향 ±recovery_footprint_half)로 근사한다. 1 m 간격 시케인처럼
            # 차를 비스듬히 틀어야만 빠져나가는 배치도 통과할 수 있게. 실제 충돌은 안전층(fan)이 따로 점검한다.
            cm_r = make_clear_map(self.recover_footprint, 0.0)
            sol = solve(self.min_clear * self.recover_clear_factor, self.edge_min * self.recover_clear_factor,
                        self.recover_slope, cm_r)
            recovering = sol is not None
        if sol is not None and recovering != self.recovering:
            self.get_logger().info("Recovery mode " + ("ON (low-speed)" if recovering else "OFF"))
        self.recovering = recovering

        if sol is None:
            self.infeasible_count += 1
            self.plan_infeasible = True
            if self.infeasible_count >= self.stop_after and self.mode != STOP:
                self.set_mode(STOP, "통과 가능한 경로 없음")
                self.get_logger().warn(
                    f"No feasible path: {obs['s'].size} obstacle points within {H:.1f} m")
            if self.plan_d is not None and self.mode != STOP:
                # 한 주기 정도는 직전 계획 유지
                prev_rel = self.ahead_of(self.plan_s_g, win['s_ego_g'])
                order = np.argsort(prev_rel)
                return self.offsets_from_plan(win, prev_rel[order], self.plan_d[order]), True
            return np.full(len(s_win), d_ego), True
        self.infeasible_count = 0
        self.plan_infeasible = False
        d_plan, blocked = sol
        d_plan = d_plan.copy()
        d_plan[0] = d0

        # 가벼운 평활화 (제약을 깨면 원래 해 사용). lattice 해는 이미 매끄럽다.
        sm = d_plan.copy()
        for _ in range(0 if used_lattice else self.smooth_passes):
            sm[1:-1] = 0.25 * sm[:-2] + 0.5 * sm[1:-1] + 0.25 * sm[2:]
        sm_idx = np.clip(np.rint((sm - dg[0]) / self.DP_DD).astype(int), 0, D - 1)
        if not np.any(blocked[np.arange(1, N), sm_idx[1:]]):
            d_plan = sm

        # 계획 경로 앞쪽(약 1.5 m) 최소 여유 -> 좁은 통로 감속에 사용
        near = st <= 1.5
        cidx = np.clip(np.rint((d_plan - dg[0]) / self.DP_DD).astype(int), 0, D - 1)
        cvals = clear_map[np.arange(N), cidx][near]
        self.plan_min_clear = float(np.min(cvals)) if cvals.size else math.inf

        self.plan_s_g = (win['s_ego_g'] + st) % self.track_length
        self.plan_d = d_plan.copy()
        if self.mode != AVOID:
            reason = "장애물 회피 경로" if relevant else "레이스라인 복귀 중"
            self.set_mode(AVOID, f"{reason}: max |d|={np.max(np.abs(d_plan)):.2f} m")
        return self.offsets_from_plan(win, st, d_plan), True

    @staticmethod
    def _quintic(u):
        """0->1 5차 전이 (양 끝 기울기/곡률 0)와 그 1, 2차 미분 (u 기준)."""
        u = np.clip(u, 0.0, 1.0)
        q = u ** 3 * (10 - 15 * u + 6 * u * u)
        q1 = 30 * u ** 2 * (1 - u) ** 2
        q2 = 60 * u * (1 - u) * (1 - 2 * u)
        return q, q1, q2

    def lattice_solve(self, st, dg, nc, d0, slope0, kref):
        """
        Frenet lattice: 현재 (d0, 기울기 slope0) 에서 d1 로 L1 동안, 이어서 d2 로 L2 동안
        5차 다항식으로 옮겨가는 매끄러운 경로 후보를 모두 만들고 가장 싼 것을 고른다.
        비용 = DP 와 같은 station 비용표(nc, 금지 칸은 inf) 조회 + 횡가속도 v^2*(kappa_ref + d'') 비용.
        전이 길이는 속도에 비례하므로 고속일수록 완만하게 옮겨간다. 해가 없으면 None.
        """
        v = max(self.current_speed, 1.0)
        H = float(st[-1])
        Ls = np.unique(np.clip(v * np.asarray(self.lattice_len_times), self.lattice_min_len, H))
        dgl = dg[::self.lattice_d_stride]
        N = st.size
        # stage 1: (d0, slope0, 0) -> (d1, 0, 0) over L1.  d = d0 + (d1-d0) q(u) + slope0 * h(u) L1
        u1 = st[None, :] / Ls[:, None]                                   # (NL, N)
        q, _, q2 = self._quintic(u1)
        uc = np.clip(u1, 0.0, 1.0)
        h = uc - 6 * uc ** 3 + 8 * uc ** 4 - 3 * uc ** 5                 # h(0)=0,h'(0)=1, h(1)=h'(1)=h''(1)=0
        h2 = (-36 * uc + 96 * uc ** 2 - 60 * uc ** 3) * (u1 <= 1.0)
        base = d0 + slope0 * h * Ls[:, None]                             # (NL, N)
        base2 = slope0 * h2 / Ls[:, None]
        S1 = base[None] + (dgl[:, None, None] - d0) * q[None]            # (A, NL, N)
        S1dd = base2[None] + (dgl[:, None, None] - d0) * q2[None] / Ls[None, :, None] ** 2
        # stage 2: start at L1, (d1,0,0) -> (d2,0,0) over L2
        u2 = (st[None, None, :] - Ls[:, None, None]) / Ls[None, :, None]  # (NL1, NL2, N)
        Q, _, Q2 = self._quintic(u2)
        Q2 = Q2 * ((u2 >= 0) & (u2 <= 1)) / Ls[None, :, None] ** 2
        delta = dgl[None, :] - dgl[:, None]                              # (A, B): d2 - d1
        d = (S1[:, :, None, None, :]
             + delta[:, None, :, None, None] * Q[None, :, None, :, :])  # (A, NL1, B, NL2, N)
        dd = (S1dd[:, :, None, None, :]
              + delta[:, None, :, None, None] * Q2[None, :, None, :, :])
        idx = np.clip(np.rint((d - dg[0]) / self.DP_DD).astype(np.int64), 0, dg.size - 1)
        kk = np.arange(N)
        c = nc[kk, idx][..., 1:].sum(axis=-1)                            # station 0 제외
        lat_acc = v * v * (kref + dd)
        c = c + self.w_latacc * np.mean(lat_acc ** 2, axis=-1)
        flat = int(np.argmin(c))
        if not np.isfinite(c.flat[flat]):
            return None
        a, l1, b, l2 = np.unravel_index(flat, c.shape)
        return d[a, l1, b, l2].copy()

    def lattice_path_cost(self, st, dg, nc, d, kref):
        """임의의 오프셋 경로 d(st) 를 lattice 와 같은 비용으로 평가 (금지 칸을 지나면 inf)."""
        v = max(self.current_speed, 1.0)
        idx = np.clip(np.rint((d - dg[0]) / self.DP_DD).astype(np.int64), 0, dg.size - 1)
        c = float(np.sum(nc[np.arange(st.size), idx][1:]))
        dd = np.gradient(np.gradient(d, st), st)
        return c + self.w_latacc * float(np.mean((v * v * (kref + dd)) ** 2))

    def offsets_from_plan(self, win, st, d_plan):
        """계획 station 오프셋을 윈도우 waypoint 오프셋으로 보간. 계획 끝 이후에는 0 으로 복귀."""
        s = win['s']
        off = np.interp(s, st, d_plan)
        beyond = s > st[-1]
        if np.any(beyond):
            u = (s[beyond] - st[-1]) / self.rejoin_distance
            off[beyond] = d_plan[-1] * (1.0 - smoothstep(u))
        return self.clamp_to_track(off, win)

    def clamp_to_track(self, off, win):
        """각 waypoint에서 오프셋을 트랙 경계 안쪽으로 제한한다."""
        hi = win['dl'] - self.veh_half - self.edge_min
        lo = -(win['dr'] - self.veh_half - self.edge_min)
        bad = lo > hi
        mid = 0.5 * (hi + lo)
        hi = np.where(bad, mid, hi)
        lo = np.where(bad, mid, lo)
        return np.clip(off, lo, hi)

    def set_mode(self, new_mode, reason=""):
        if new_mode != self.mode:
            self.get_logger().info(f"Mode {self.mode} -> {new_mode}  {reason}")
            self.mode = new_mode

    # -----------------------------------------------------------
    # Active path + Pure Pursuit
    # -----------------------------------------------------------
    def build_active_path(self, win, off):
        """레이스라인 + 횡오프셋 경로. 차량에 가장 가까운 점부터 시작한다."""
        xy = win['xy'] + off[:, None] * win['nrm']
        d2 = (xy[:, 0] - self.current_x) ** 2 + (xy[:, 1] - self.current_y) ** 2
        i0 = int(np.argmin(d2))
        x = xy[i0:, 0]
        y = xy[i0:, 1]
        seg = np.hypot(np.diff(x), np.diff(y))
        cum = np.concatenate(([0.0], np.cumsum(seg)))
        return {'x': x, 'y': y, 'cum': cum, 'off': off[i0:], 'v_prof': win['v_prof'][i0:]}

    def path_heading_curvature(self, path):
        """경로 점들의 헤딩과 (평활화한) 곡률. 한 주기에 한 번만 계산한다."""
        if 'psi' not in path:
            path['psi'], path['kappa'] = self._heading_curvature(path)
        return path['psi'], path['kappa']

    def _heading_curvature(self, path):
        x, y = path['x'], path['y']
        n = len(x)
        if n < 5:
            psi = np.full(n, self.current_yaw)
            return psi, np.zeros(n)
        psi = np.unwrap(np.arctan2(np.gradient(y), np.gradient(x)))
        ds = np.maximum(np.gradient(path['cum']), 1e-3)
        kappa = np.gradient(psi) / ds
        w = self.kappa_smooth
        kappa = np.convolve(np.pad(kappa, w // 2, mode='edge'), np.ones(w) / w, mode='valid')
        return psi, kappa

    def path_speed_limit(self, path):
        """
        활성 경로(회피 경로 포함)의 곡률로 정한 속도 한계를 앞쪽 제동거리까지 고려해 지금 낼 수 있는 속도.
          v_i = min(경로 속도 프로파일, sqrt(ay_max/|kappa_i|)),  v_now <= min_i sqrt(v_i^2 + 2*dec*s_i)
        """
        if self.ay_max <= 0.0:
            return math.inf
        _, kappa = self.path_heading_curvature(path)
        ay = self.ay_max
        if self.mode == AVOID and self.avoid_ay_max > 0.0:
            ay = min(ay, self.avoid_ay_max)
        v_i = np.minimum(path['v_prof'], np.sqrt(ay / np.maximum(np.abs(kappa), 1e-4)))
        return float(np.min(np.sqrt(v_i ** 2 + 2.0 * self.dec_max * path['cum'])))

    def path_collides(self, path):
        """활성 경로의 앞쪽 (v*safety_check_time + 0.5 m) 이 LiDAR 점과 collision_radius 이내로 지나가는지."""
        pts = self.lidar_local
        if pts.shape[0] == 0:
            return False
        dist = max(self.current_speed, 0.0) * self.safety_check_t + 0.5
        m = path['cum'] <= dist
        m[0] = False
        if not np.any(m):
            return False
        cy, sy = math.cos(self.current_yaw), math.sin(self.current_yaw)
        dx = path['x'][m] - self.current_x
        dy = path['y'][m] - self.current_y
        lx = cy * dx + sy * dy
        ly = -sy * dx + cy * dy
        d2 = (lx[:, None] - pts[None, :, 0]) ** 2 + (ly[:, None] - pts[None, :, 1]) ** 2
        return float(np.sqrt(np.min(d2))) < self.collision_radius

    def stanley_steer(self, path):
        """
        곡률 피드포워드 + Stanley 피드백 (앞축 기준).
          delta = atan(L*kappa) + K_us*v^2*kappa - k_h*heading_err - atan(k_e*e / (v + v_soft))
        Pure Pursuit 과 달리 lookahead 로 코너 안쪽을 가로지르지 않아 고속에서 레이스라인을 정확히 따른다.
        steer_lookahead_time 만큼 앞의 자세를 예측해 조향 지연을 보상한다.
        """
        v = max(self.current_speed, 0.0)
        L = self.settings.wheelbase
        t = self.steer_lookahead_t
        r = v * math.tan(self.prev_steer) / L
        yaw_p = self.current_yaw + r * t
        xm = self.current_x + v * t * math.cos(self.current_yaw + 0.5 * r * t)
        ym = self.current_y + v * t * math.sin(self.current_yaw + 0.5 * r * t)
        fx = xm + self.front_axle_offset * math.cos(yaw_p)
        fy = ym + self.front_axle_offset * math.sin(yaw_p)

        psi, kappa = self.path_heading_curvature(path)
        j = int(np.argmin((path['x'] - fx) ** 2 + (path['y'] - fy) ** 2))
        e = (-(fx - path['x'][j]) * math.sin(psi[j]) + (fy - path['y'][j]) * math.cos(psi[j]))  # 왼쪽 +
        head_err = self.normalize_angle(yaw_p - psi[j])
        # 곡률은 앞축 위치에서 kappa_preview_time 만큼 앞을 본다
        jp = min(int(np.searchsorted(path['cum'], path['cum'][j] + v * self.kappa_preview_t)), len(kappa) - 1)
        k_ff = float(kappa[jp])
        steer = (math.atan(L * k_ff) + self.understeer_gain * v * v * k_ff
                 - self.stanley_k_heading * head_err
                 - math.atan2(self.stanley_k * e, v + self.stanley_v_soft))
        return float(np.clip(steer, self.settings.min_steer, self.settings.max_steer))

    def pure_pursuit_steer(self, path, lookahead):
        """경로 위 lookahead 거리 앞의 점을 향하는 Pure Pursuit 조향각 (기준점: 뒤축)."""
        cy, sy = math.cos(self.current_yaw), math.sin(self.current_yaw)
        # 조향 지연 보상: steer_lookahead_time 뒤의 위치(현재 헤딩 기준 직진 가정)에서 계산
        adv = max(self.current_speed, 0.0) * self.steer_lookahead_t
        rx = self.current_x + (adv - self.rear_axle_offset) * cy
        ry = self.current_y + (adv - self.rear_axle_offset) * sy
        # 뒤축 기준 거리로 목표점을 찾는다 (경로 시작점은 차량 근처)
        d = np.hypot(path['x'] - rx, path['y'] - ry)
        ahead = (path['x'] - rx) * cy + (path['y'] - ry) * sy > 0.0
        cand = np.nonzero(ahead & (d >= lookahead))[0]
        j = int(cand[0]) if cand.size else len(path['x']) - 1
        dx = path['x'][j] - rx
        dy = path['y'][j] - ry
        ld = max(math.hypot(dx, dy), 0.1)
        alpha = self.normalize_angle(math.atan2(dy, dx) - self.current_yaw)
        steer = math.atan2(2.0 * self.settings.wheelbase * math.sin(alpha), ld)
        return float(np.clip(steer, self.settings.min_steer, self.settings.max_steer))

    # -----------------------------------------------------------
    # Candidate-trajectory safety layer
    # -----------------------------------------------------------
    def generate_candidate_controls(self):
        """Pure Pursuit 조향을 첫 후보로, 그 주변에 일정 조향 후보(fan)를 만든다."""
        if self.mode == AVOID:
            num_candidates = 17
            fan_half_angle = np.deg2rad(self.get_parameter("obstacle_fan_half_angle_deg").value)
        else:
            num_candidates = 9
            fan_half_angle = np.deg2rad(self.get_parameter("track_fan_half_angle_deg").value)
        lo, hi = self.settings.min_steer, self.settings.max_steer
        base = float(np.clip(self.path_steer, lo, hi))
        fan = np.linspace(np.clip(base - fan_half_angle, lo, hi),
                          np.clip(base + fan_half_angle, lo, hi), num_candidates)
        return np.concatenate(([base], fan))

    def simulate_candidate(self, steer_const, v_target):
        """ego 좌표계에서 자전거 모델로 상수 조향 후보를 굴린다."""
        N = self.settings.horizon
        dt = self.settings.dt
        x = np.zeros(N + 1)
        y = np.zeros(N + 1)
        yaw = np.zeros(N + 1)
        v = np.zeros(N + 1)
        # 롤아웃 길이가 현재 속도(≈ 정지거리)에 비례하도록: 정지/저속에서는 가까운 구간만 검사해
        # '모든 후보 충돌 -> 정지 -> 계속 충돌' 데드락을 피한다
        v_floor = self.safety_min_speed
        v[0] = max(self.current_speed, v_floor)
        a = float(np.clip(2.0 * (v_target - self.current_speed),
                          -self.settings.max_accel, self.settings.max_accel))
        k = math.tan(steer_const) / self.settings.wheelbase
        for t in range(N):
            v[t + 1] = np.clip(v[t] + a * dt, v_floor, self.settings.max_speed)
            x[t + 1] = x[t] + v[t + 1] * math.cos(yaw[t]) * dt
            y[t + 1] = y[t] + v[t + 1] * math.sin(yaw[t]) * dt
            yaw[t + 1] = yaw[t] + v[t + 1] * k * dt
        return x, y, yaw, v

    def check_collision_and_cost(self, x_local, y_local, steer_const, path):
        """
        - 충돌: 후보 궤적을 촘촘히 나눈 점과 LiDAR 점 사이 최소 거리 < collision_radius
          (collision_radius 는 DP 최소 여유 veh_half + avoid_min_clearance 보다 작아야 한다)
        - 비용: 활성 경로 추종 오차 + Pure Pursuit 조향과의 차이 + 직전 조향과의 차이
        """
        cos_y = math.cos(self.current_yaw)
        sin_y = math.sin(self.current_yaw)
        x_global = self.current_x + cos_y * x_local - sin_y * y_local
        y_global = self.current_y + sin_y * x_local + cos_y * y_local

        collision = False
        pts = self.lidar_local
        if pts.shape[0] > 0:
            front = pts[:, 0] > -0.1
            pts = pts[front]
        if pts.shape[0] > 0:
            substeps = 4
            n_pts = len(x_local)
            tt = np.linspace(0.0, n_pts - 1, (n_pts - 1) * substeps + 1)
            qx = np.interp(tt, np.arange(n_pts), x_local)
            qy = np.interp(tt, np.arange(n_pts), y_local)
            far = np.hypot(qx, qy) >= 0.10   # 출발점 근처는 제외
            qx, qy = qx[far], qy[far]
            if qx.size:
                dx = qx[:, None] - pts[None, :, 0]
                dy = qy[:, None] - pts[None, :, 1]
                dmin = float(np.sqrt(np.min(dx * dx + dy * dy)))
                collision = dmin < self.collision_radius

        arc = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(x_local), np.diff(y_local)))))
        rx = np.interp(arc, path['cum'], path['x'])
        ry = np.interp(arc, path['cum'], path['y'])
        track_err = float(np.mean((x_global - rx) ** 2 + (y_global - ry) ** 2))
        cost = (10.0 * track_err
                + 2.0 * (steer_const - self.path_steer) ** 2
                + 1.0 * (steer_const - self.prev_steer) ** 2)
        if collision:
            cost += 1e6
        return collision, cost

    def run_candidate_mpc(self, vehicle_state, path, v_target):
        """
        Pure Pursuit 조향이 충돌하지 않으면 그대로 쓴다 (이산 후보 간 튐 방지).
        충돌하면 충돌하지 않는 후보 중 비용이 가장 작은 것을 고른다.
        """
        self.fan_checks += 1
        if self.safety_mode == "path" and not self.path_collides(path):
            # 계획 경로가 정지거리 안에서 안전하면 경로 추종 조향을 그대로 쓴다
            return float(self.path_steer), [], -1, True
        steering_candidates = self.generate_candidate_controls()
        global_paths = []
        costs = []
        collisions = []
        cos_y = math.cos(self.current_yaw)
        sin_y = math.sin(self.current_yaw)
        for steer_const in steering_candidates:
            x_l, y_l, yaw_l, v_l = self.simulate_candidate(steer_const, v_target)
            collision, cost = self.check_collision_and_cost(x_l, y_l, steer_const, path)
            path_global = np.zeros((4, self.settings.horizon + 1))
            path_global[0, :] = self.current_x + cos_y * x_l - sin_y * y_l
            path_global[1, :] = self.current_y + sin_y * x_l + cos_y * y_l
            path_global[2, :] = v_l
            path_global[3, :] = yaw_l + self.current_yaw
            global_paths.append(path_global)
            costs.append(cost)
            collisions.append(collision)

        if not collisions[0]:
            return float(steering_candidates[0]), global_paths, 0, True
        self.fan_overrides += 1
        best_idx = int(np.argmin(costs))
        return float(steering_candidates[best_idx]), global_paths, best_idx, not collisions[best_idx]

    # -----------------------------------------------------------
    # Parameters and Visualization
    # -----------------------------------------------------------
    def setup_parameters(self):
        dp = self.declare_parameter
        dp("real_environment", False)
        dp("map_name", "hyhall12")
        dp("map_dir", "")
        dp("trajectory_key", "global_traj_wpnts_iqp")
        dp("candidate_speed", 3.0)
        dp("speed_scale", 1.0)
        dp("use_csv_speed", True)

        # 후보 경로(fan) 시뮬레이션
        dp("horizon", 8)
        dp("dt", 0.1)

        # Vehicle dimensions and limits
        dp("veh_length", 0.58)
        dp("veh_width", 0.2032)
        dp("wheelbase", 0.33)
        dp("min_steer", -0.4189)
        dp("max_steer", 0.4189)
        dp("max_steer_rate", 3.0)
        dp("max_speed", 6.0)
        dp("min_speed", 1.0)
        dp("max_accel", 2.5)

        # Pure Pursuit
        dp("lookahead_distance", 0.6)
        dp("lookahead_speed_gain", 0.25)
        dp("lookahead_max", 2.0)
        dp("lookahead_min", 0.0)
        dp("tracker", "pp")
        dp("avoid_tracker", "same")
        dp("avoid_tracker_max_speed", 100.0)
        dp("stanley_k", 2.0)
        dp("stanley_k_heading", 1.0)
        dp("stanley_v_soft", 1.0)
        dp("front_axle_offset", 0.16)
        dp("understeer_gain", 0.0)
        dp("kappa_preview_time", 0.0)
        dp("steer_lookahead_time", 0.0)
        dp("speed_lookahead_time", 0.0)
        dp("avoid_lookahead_distance", 0.5)
        dp("avoid_lookahead_speed_gain", 0.15)
        dp("pp_rear_axle_offset", 0.0)
        dp("steer_filter_alpha", 1.0)

        # LiDAR / fan
        dp("max_sight", 5.0)
        dp("lidar_stride", 2)
        dp("lidar_x_offset", 0.27)
        dp("track_fan_half_angle_deg", 6.0)
        dp("obstacle_fan_half_angle_deg", 20.0)
        dp("collision_margin", 0.03)
        dp("safety_min_rollout_speed", 0.3)
        dp("safety_mode", "fan")
        dp("safety_check_time", 0.4)
        dp("plan_smooth_passes", 3)
        dp("replan_reset_distance", 0.0)
        dp("planner", "dp")
        dp("lattice_transition_times", [0.4, 0.6, 0.85, 1.2])
        dp("lattice_min_length", 0.8)
        dp("lattice_d_stride", 2)
        dp("plan_w_lat_accel", 0.05)
        dp("lattice_compare_dp", False)

        # 장애물 감지 / 메모리
        dp("obstacle_detect_distance", 4.0)
        dp("obstacle_fov_deg", 270.0)
        dp("obstacle_confirm_scans", 2)
        dp("obstacle_memory_time", 1.5)
        dp("obstacle_assumed_depth", 0.15)
        dp("track_half_width_default", 1.0)
        dp("track_wall_margin", 0.15)
        dp("use_map_wall_filter", True)
        dp("map_wall_margin", 0.12)

        # DP 경로 계획
        dp("avoid_clearance", 0.20)
        dp("avoid_min_clearance", 0.06)
        dp("track_boundary_margin", 0.20)
        dp("track_boundary_min_margin", 0.05)
        dp("avoid_longitudinal_margin", 0.05)
        dp("plan_ds", 0.2)
        dp("plan_max_lateral_slope", 0.6)
        dp("plan_relax_clearance_factor", 0.5)
        dp("plan_relax_slope_factor", 1.6)
        dp("plan_w_offset", 1.0)
        dp("plan_w_slope", 2.0)
        dp("plan_w_clearance", 150.0)
        dp("plan_w_consistency", 3.0)
        dp("plan_consistency_decay", 100.0)
        dp("clearance_speed_gain", 0.0)
        dp("follow_ignore_clearance", 0.0)
        dp("avoid_curve_lat_accel", 0.0)
        dp("clearance_speed_gain_hard", -1.0)
        dp("plan_stop_after_infeasible", 2)
        dp("recovery_enable", False)
        dp("recovery_speed", 1.0)
        dp("recovery_max_lateral_slope", 1.5)
        dp("recovery_clearance_factor", 0.5)
        dp("recovery_footprint_half", 0.12)
        dp("scan_pose_sync", True)

        # 속도 / 복귀
        dp("speed_ay_max", 0.0)
        dp("speed_ax_max", 7.0)
        dp("speed_dec_max", 8.0)
        dp("avoid_ay_max", 0.0)
        dp("avoid_speed", 1.5)
        dp("avoid_tight_speed", 1.0)
        dp("rejoin_distance", 1.5)
        dp("rejoin_done_tolerance", 0.10)
        dp("rejoin_heading_tolerance", 3.2)
        dp("path_kappa_smooth_points", 5)
        dp("speed_cmd_accel", 100.0)
        dp("speed_cmd_decel", 1000.0)
        dp("avoid_approach_decel", 1000.0)

    def show_waypoints(self):
        self.wpts_marker.points = []
        self.wpts_marker.header.frame_id = '/map'
        self.wpts_marker.type = Marker.POINTS
        self.wpts_marker.color.g = 0.75
        self.wpts_marker.color.a = 1.0
        self.wpts_marker.scale.x = 0.05
        self.wpts_marker.scale.y = 0.05
        self.wpts_marker.id = 0
        for i in range(self.waypoints.shape[0]):
            self.wpts_marker.points.append(
                Point(x=self.waypoints[i, 1], y=self.waypoints[i, 2], z=0.1))
        self.wpts_pub.publish(self.wpts_marker)

    def show_reference_traj(self, ref_traj):
        """활성 경로(레이스라인/회피)를 파란 선으로 표시한다. ref_traj: (2+, M)."""
        self.ref_traj_marker.points = []
        self.ref_traj_marker.header.frame_id = '/map'
        self.ref_traj_marker.type = Marker.LINE_STRIP
        self.ref_traj_marker.color.b = 0.75
        self.ref_traj_marker.color.a = 1.0
        self.ref_traj_marker.scale.x = 0.08
        self.ref_traj_marker.scale.y = 0.08
        self.ref_traj_marker.id = 0
        for i in range(ref_traj.shape[1]):
            self.ref_traj_marker.points.append(
                Point(x=float(ref_traj[0, i]), y=float(ref_traj[1, i]), z=0.2))
        self.ref_traj_pub.publish(self.ref_traj_marker)

    def show_obstacles(self, win, obs):
        """DP 가 사용하는 장애물 점(메모리 포함)을 주황 점으로 표시한다."""
        m = Marker()
        m.header.frame_id = '/map'
        m.type = Marker.POINTS
        m.id = 0
        m.scale.x = 0.06
        m.scale.y = 0.06
        m.color.r = 1.0
        m.color.g = 0.5
        m.color.a = 1.0
        if obs['s'].size:
            s_g = (win['s_ego_g'] + obs['s']) % self.track_length
            idx = np.searchsorted(self.s, s_g, side='right') - 1
            idx = np.clip(idx, 0, len(self.s) - 1)
            along = s_g - self.s[idx]
            xy = (self.waypoints[idx, 1:3] + along[:, None] * self.tan[idx]
                  + obs['d'][:, None] * self.nrm[idx])
            m.points = [Point(x=float(px), y=float(py), z=0.15) for px, py in xy]
        self.obs_pub.publish(m)

    def show_predicted_paths(self, global_paths, best_idx):
        """후보 궤적 fan 을 빨간 선으로, 선택된 후보를 굵은 흰 선으로 표시한다."""
        marker_array = MarkerArray()
        now = self.get_clock().now().to_msg()
        for k, path in enumerate(global_paths):
            marker = Marker()
            marker.header.frame_id = '/map'
            marker.header.stamp = now
            marker.ns = "mpc_candidates"
            marker.id = k
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            if k == best_idx:
                marker.scale.x = 0.08
                marker.color.r = marker.color.g = marker.color.b = 1.0
                marker.color.a = 1.0
            else:
                marker.scale.x = 0.03
                marker.color.r = 1.0
                marker.color.a = 0.7
            marker.points = [Point(x=float(path[0, i]), y=float(path[1, i]), z=0.2)
                             for i in range(path.shape[1])]
            marker_array.markers.append(marker)
        self.pred_path_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = MPCNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
