import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rcl_interfaces.msg import FloatingPointRange, ParameterDescriptor, SetParametersResult, ParameterType
from rclpy.parameter import Parameter

from f110_msgs.msg import Obstacle, ObstacleArray, OTWpntArray, Wpnt, WpntArray
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32
from visualization_msgs.msg import Marker, MarkerArray

import copy
import itertools
import numpy as np
import time
from dataclasses import dataclass, field
from typing import List, Any, Tuple, Optional
from scipy.interpolate import PchipInterpolator
from frenet_conversion.frenet_converter import FrenetConverter


# ---------------------------------------------------------------------------
# 다중 장애물 처리용 내부 자료구조
# ---------------------------------------------------------------------------
@dataclass(eq=False)
class _ObsInfo:
    """예측(propagation)까지 끝난 장애물 1개. s는 cur_s 기준으로 언랩(unwrap)된 값."""
    obs: Any
    s0: float          # unwrapped s_start
    s1: float          # unwrapped s_end
    d_r: float         # 오른쪽 모서리 (작은 d)
    d_l: float         # 왼쪽 모서리 (큰 d)
    d_c: float
    blocking: bool     # 레이스라인 위(|d_center| < obs_traj_tresh)에 있는지
    cluster: Any = None

    @property
    def s_mid(self) -> float:
        return 0.5 * (self.s0 + self.s1)


@dataclass(eq=False)
class _Cluster:
    """s 방향으로 겹치거나 아주 가까워서 '한 번에' 지나가야 하는 장애물 묶음."""
    members: List[_ObsInfo]
    s0: float
    s1: float
    blocking: bool
    d_ref: float = 0.0                                  # left/right 판정 기준 d
    outside: str = "right"                              # 이 구간에서 코너 바깥쪽
    options: List[Tuple[float, str]] = field(default_factory=list)  # [(d_apex, side)]

    @property
    def s_mid(self) -> float:
        return 0.5 * (self.s0 + self.s1)


class ObstacleSpliner(Node):
    """
    This class implements a ROS node that performs splining around (multiple) obstacles.

    It subscribes to the following topics:
        - `/perception/obstacles`: Subscribes to the obstacle array.
        - `/car_state/odom_frenet`: Subscribes to the car state in Frenet coordinates.
        - `/global_waypoints`: Subscribes to global waypoints.
        - `/global_waypoints_scaled`: Subscribes to the scaled global waypoints.

    The node publishes the following topics:
        - `/planner/avoidance/markers`: Publishes spline markers.
        - `/planner/avoidance/otwpnts`: Publishes splined waypoints.
        - `/planner/avoidance/considered_OBS`: Publishes markers for the considered obstacle clusters.
        - `/planner/avoidance/propagated_obs`: Publishes markers for the propagated obstacles.
        - `/planner/avoidance/latency`: Publishes the latency of the spliner node. (only if measuring is enabled)

    Multi-obstacle extension:
        1. lookahead 안에서 레이스라인을 막는 장애물(blocking)을 모두 찾는다.
        2. s 방향으로 겹치거나 merge_gap 이내로 붙은 장애물은 하나의 클러스터로 묶는다.
        3. 클러스터마다 통과 가능한 d 구간(왼쪽/오른쪽/장애물 사이 틈)을 계산한다.
        4. 클러스터별 선택지의 모든 조합을 평가(횡편차 + 좌우 전환 비용 + 히스테리시스)하고,
           비용이 낮은 순서로 경로를 만들어 트랙 경계·전 장애물 충돌 검사를 통과하는 첫 경로를 쓴다.
        5. 레이스라인 밖 장애물은 평소엔 무시하지만, 경로가 그 장애물과 충돌하면 그 클러스터를
           계획 대상에 추가(promotion)해서 다시 계획한다.
        6. 전체 조합이 불가능하면 먼 클러스터부터 빼면서(prefix) 다시 시도하고,
           그래도 안 되면 기존 코드처럼 회피를 포기한다(danger).
    """

    # 조합 탐색 관련 상수 (필요하면 파라미터화)
    MAX_CLUSTERS = 4          # 한 번에 계획하는 blocking 클러스터 최대 개수
    MAX_OPTIONS = 3           # 클러스터당 통과 선택지 최대 개수
    MAX_PROMOTIONS = 3        # 레이스라인 밖 장애물 추가 계획 최대 횟수
    MAX_PATH_TRIALS = 15      # 비용 순으로 실제 경로를 만들어 검사해볼 최대 조합 수
    TRANSITION_COST = 1.0     # 레이스라인 복귀 없이 d가 바뀔 때 |Δd|에 곱하는 가중치
    SIDE_HYST_COST = 0.3      # 직전 추월 방향과 다를 때 붙는 비용 [m 등가]
    SPLINE_RESOLUTION = 0.1   # state_machine의 waypoints_dist(0.1)와 같아야 함
    # --- 좁은 길목 / 연속성 관련 ---
    RELAX_SCALES = (1.0, 0.75, 0.55)  # 통로가 없을 때 장애물 간격·트랙 경계 여유를 단계적으로 줄이는 배율
    MIN_BOUND_MARGIN = 0.20   # 완화해도 트랙 경계와 차량 중심 사이에 최소한 남길 거리 [m] (차폭 절반 + 여유)
    MIN_RETURN_GAP = 0.3      # 클러스터 사이에서 '완전히 raceline으로 복귀'가 성립하기 위한 최소 knot 간격 [m]
    REAR_KEEP_M = 1.0         # 방금 지나친 장애물을 이 거리(≈차 길이)만큼은 계속 계획에 포함 [m]
    MATCH_TOL_M = 1.5         # 직전 계획의 클러스터와 같은 클러스터로 보는 s 거리 [m]
    PREV_MAX_AGE_S = 1.0      # 직전 경로/방향 정보를 신뢰하는 최대 시간 [s]
    BLEND_HOLD_M = 0.5        # 차량 앞 이 거리까지는 직전 경로를 그대로 유지 [m]
    BLEND_LEN_M = 3.0         # 그 뒤 이 거리에 걸쳐 새 경로로 전환 [m]
    BLEND_BACK_M = 0.5        # 차량 뒤 이 거리까지 경로를 이어 붙여 차체 아래 waypoint가 끊기지 않게 함 [m]

    def __init__(self):
        super().__init__('spliner_node')

        # initialize the instance variable
        self.obs = ObstacleArray()
        self.gb_wpnts = None
        self.gb_scaled_wpnts = None
        self.gb_vmax = None
        self.gb_max_idx = None
        self.gb_max_s = None
        self.cur_s = None
        self.cur_d = None
        self.cur_vs = None
        # 장애물이 spline 전체 길이(pre+post, 최대 ±10m)보다 가까워지기 전에 계획에 넣어야 경로가 갑자기 바뀌지 않는다
        self.lookahead = 20  # in meters [m]
        self.last_switch_time = self.get_clock().now().to_msg()
        self.last_ot_side = ""
        self._wpnt_dist = 0.1
        self._side_rejected = False
        self._relax = 1.0              # 현재 간격 완화 단계 (RELAX_SCALES 중 하나)
        self._prev_path = None         # 직전에 publish한 경로 (절대 s, d)
        self._prev_sides = []          # 직전 계획의 [(클러스터 절대 s, side)]
        self._prev_stamp = -1e9

        # Initialize instance variables
        self.declare_parameters(
            namespace='',
            parameters=[
                ('from_bag', False),
                ('measure', False),
            ])
        self.from_bag = self.get_parameter(
            'from_bag').get_parameter_value().bool_value
        self.measuring = self.get_parameter(
            'measure').get_parameter_value().bool_value

        # Setup subscriptions
        self.create_subscription(
            ObstacleArray, '/perception/obstacles', self.obs_cb, QoSProfile(depth=10))
        self.create_subscription(
            Odometry, '/car_state/frenet/odom', self.state_cb, QoSProfile(depth=10))
        self.create_subscription(
            WpntArray, '/global_waypoints', self.gb_cb, QoSProfile(depth=10))
        self.create_subscription(
            WpntArray, '/global_waypoints_scaled', self.gb_scaled_cb, QoSProfile(depth=10))

        # Setup publishers
        self.mrks_pub = self.create_publisher(
            MarkerArray, '/planner/avoidance/markers', QoSProfile(depth=10))
        self.evasion_pub = self.create_publisher(
            OTWpntArray, '/planner/avoidance/otwpnts', QoSProfile(depth=10))
        self.closest_obs_pub = self.create_publisher(
            Marker, '/planner/avoidance/considered_OBS', QoSProfile(depth=10))
        self.pub_propagated = self.create_publisher(
            Marker, '/planner/avoidance/propagated_obs', QoSProfile(depth=10))

        # dyn params sub
        self.pre_apex_0 = -4.0
        self.pre_apex_1 = -3.0
        self.pre_apex_2 = -1.5
        self.post_apex_0 = 2.0
        self.post_apex_1 = 3.0
        self.post_apex_2 = 4.0
        self.evasion_dist = 0.55
        self.obs_traj_tresh = 0.3
        self.spline_bound_mindist = 0.25
        self.fixed_pred_time = 0.15
        self.kd_obs_pred = 1.0
        # --- multi-obstacle params ---
        self.merge_gap = 0.5            # 이 거리[m]보다 가까운 장애물은 한 클러스터로 묶음
        self.collision_clearance = 0.25  # 충돌 검사 시 경로 중심~장애물 모서리 최소 거리[m]
        self.max_lateral_slope = 0.5     # 레이스라인 복귀 없이 d를 바꿀 때 허용 평균 |Δd/Δs| (PCHIP 최대 기울기는 이의 ~1.5배)

        pd = ParameterDescriptor(
            type=ParameterType.PARAMETER_DOUBLE,
            floating_point_range=[FloatingPointRange(from_value=0.1, to_value=8.0, step=0.001)]
        )
        param_dicts = [
            {'name': 'pre_apex_0', 'default': abs(self.pre_apex_0), 'descriptor': pd},
            {'name': 'pre_apex_1', 'default': abs(self.pre_apex_1), 'descriptor': pd},
            {'name': 'pre_apex_2', 'default': abs(self.pre_apex_2), 'descriptor': pd},
            {'name': 'post_apex_0', 'default': self.post_apex_0, 'descriptor': pd},
            {'name': 'post_apex_1', 'default': self.post_apex_1, 'descriptor': pd},
            {'name': 'post_apex_2', 'default': self.post_apex_2, 'descriptor': pd},
            {'name': 'evasion_dist', 'default': self.evasion_dist, 'descriptor': pd},
            {'name': 'obs_traj_tresh', 'default': self.obs_traj_tresh, 'descriptor': pd},
            {'name': 'spline_bound_mindist', 'default': self.spline_bound_mindist, 'descriptor': pd},
            {'name': 'fixed_pred_time', 'default': self.fixed_pred_time, 'descriptor': pd},
            {'name': 'kd_obs_pred', 'default': self.kd_obs_pred, 'descriptor': pd},
            {'name': 'merge_gap', 'default': self.merge_gap, 'descriptor': pd},
            {'name': 'collision_clearance', 'default': self.collision_clearance, 'descriptor': pd},
            {'name': 'max_lateral_slope', 'default': self.max_lateral_slope, 'descriptor': pd},
        ]

        self.declare_all_parameters(param_dicts=param_dicts)
        self.add_on_set_parameters_callback(self.dyn_param_cb)

        if self.measuring:
            self.latency_pub = self.create_publisher(Float32, '/planner/avoidance/latency', QoSProfile(depth=10))

        # Wait for messages
        self.wait_for_messages()

        # Initialize converter, if required
        self.converter = self.initialize_converter()

        # Create timer
        self.create_timer(1/20, self.spliner_loop)

    #################### DYNAMIC PARAMS####################
    def declare_all_parameters(self, param_dicts: List[dict]):
        params = []
        for param_dict in param_dicts:
            param = self.declare_parameter(
                param_dict['name'], param_dict['default'], param_dict['descriptor'])
            params.append(param)
        return params

    def dyn_param_cb(self, params: List[Parameter]):
        """
        Notices the change in the parameters and alters the spline params accordingly
        """
        for param in params:
            param_name = param.name
            if param_name == 'pre_apex_0':
                self.pre_apex_0 = -1 * param.value
            elif param_name == 'pre_apex_1':
                self.pre_apex_1 = -1 * param.value
            elif param_name == 'pre_apex_2':
                self.pre_apex_2 = -1 * param.value
            elif param_name == 'post_apex_0':
                self.post_apex_0 = param.value
            elif param_name == 'post_apex_1':
                self.post_apex_1 = param.value
            elif param_name == 'post_apex_2':
                self.post_apex_2 = param.value
            elif param_name == 'evasion_dist':
                self.evasion_dist = param.value
            elif param_name == 'obs_traj_tresh':
                self.obs_traj_tresh = param.value
            elif param_name == 'spline_bound_mindist':
                self.spline_bound_mindist = param.value
            elif param_name == 'fixed_pred_time':
                self.fixed_pred_time = param.value
            elif param_name == 'kd_obs_pred':
                self.kd_obs_pred = param.value
            elif param_name == 'merge_gap':
                self.merge_gap = param.value
            elif param_name == 'collision_clearance':
                self.collision_clearance = param.value
            elif param_name == 'max_lateral_slope':
                self.max_lateral_slope = param.value

        # Ensure ascending order for spline parameters
        if self.pre_apex_1 < self.pre_apex_0:
            self.get_logger().info("Adjusting pre_apex_1 to ensure ascending order.")
            self.pre_apex_1 = self.pre_apex_0

        if self.pre_apex_2 < self.pre_apex_1:
            self.get_logger().info("Adjusting pre_apex_2 to ensure ascending order.")
            self.pre_apex_2 = self.pre_apex_1

        if self.post_apex_0 < self.pre_apex_2:
            self.get_logger().info("Adjusting post_apex_0 to ensure ascending order.")
            self.post_apex_0 = self.pre_apex_2

        if self.post_apex_1 < self.post_apex_0:
            self.get_logger().info("Adjusting post_apex_1 to ensure ascending order.")
            self.post_apex_1 = self.post_apex_0

        if self.post_apex_2 < self.post_apex_1:
            self.get_logger().info("Adjusting post_apex_2 to ensure ascending order.")
            self.post_apex_2 = self.post_apex_1

        spline_params = [
            self.pre_apex_0,
            self.pre_apex_1,
            self.pre_apex_2,
            0,
            self.post_apex_0,
            self.post_apex_1,
            self.post_apex_2,
        ]

        self.get_logger().info(
            f" Dynamic reconf triggered new spline params: {spline_params} [m],\n"
            f" evasion apex distance: {self.evasion_dist} [m],\n"
            f" obstacle trajectory treshold: {self.obs_traj_tresh} [m]\n"
            f" obstacle prediciton k_d: {self.kd_obs_pred},    obstacle prediciton constant time: {self.fixed_pred_time} [s]\n"
            f" multi-obs: merge_gap={self.merge_gap} [m], collision_clearance={self.collision_clearance} [m], "
            f"max_lateral_slope={self.max_lateral_slope}"
        )
        return SetParametersResult(successful=True)

    #################### CALLBACKS####################
    def obs_cb(self, msg: ObstacleArray):
        # Callback function for obstacle data
        self.obs = msg

    def state_cb(self, msg: Odometry):
        # Callback function for car state data
        self.cur_s = msg.pose.pose.position.x
        self.cur_d = msg.pose.pose.position.y
        self.cur_vs = msg.twist.twist.linear.x

    def gb_cb(self, msg: WpntArray):
        # Callback function for global waypoints
        self.gb_wpnts = msg
        if self.gb_vmax is None:
            self.gb_vmax = np.max(
                np.array([wpnt.vx_mps for wpnt in msg.wpnts]))
            self.gb_max_idx = msg.wpnts[-1].id
            self.gb_max_s = msg.wpnts[-1].s_m

    def gb_scaled_cb(self, msg: WpntArray):
        # Callback function for scaled global waypoints
        self.gb_scaled_wpnts = msg
        # 경계/곡률/속도 검사를 벡터화하기 위해 배열로 캐싱
        self._gb_kappa = np.array([w.kappa_radpm for w in msg.wpnts])
        self._gb_tb_left = np.abs(np.array([w.d_left for w in msg.wpnts]))
        self._gb_tb_right = np.abs(np.array([w.d_right for w in msg.wpnts]))
        self._gb_vx = np.array([w.vx_mps for w in msg.wpnts])

    #################### HELPER FUNCTIONS####################
    def wait_for_messages(self):
        self.get_logger().info('Carstate Node waiting for Odometry messages...')
        state_print = False
        gb_print = False
        scaled_gb_print = False
        waitlist = [self.cur_s, self.gb_wpnts, self.gb_scaled_wpnts]
        while None in waitlist:
            rclpy.spin_once(self)
            if self.cur_s is not None and not state_print:
                self.get_logger().info('Received State message.')
                state_print = True
            if self.gb_wpnts is not None and not gb_print:
                self.get_logger().info('Received Global Waypoints message.')
                gb_print = True
            if self.gb_scaled_wpnts is not None and not scaled_gb_print:
                self.get_logger().info('Received Scaled Global Waypoints message.')
                scaled_gb_print = True
            waitlist = [self.cur_s, self.gb_wpnts, self.gb_scaled_wpnts]
        self.get_logger().info('All required messages received. Continuing...')

    def initialize_converter(self):
        # Initialize any converters or additional components
        waypoint_array = self.gb_wpnts.wpnts
        waypoints_x = [waypoint.x_m for waypoint in waypoint_array]
        waypoints_y = [waypoint.y_m for waypoint in waypoint_array]
        waypoints_psi = [waypoint.psi_rad for waypoint in waypoint_array]
        return FrenetConverter(np.array(waypoints_x), np.array(waypoints_y), np.array(waypoints_psi))

    def _signed_ds(self, s: float) -> float:
        """cur_s 기준 부호 있는 s 거리 (-L/2, L/2], wraparound 처리."""
        L = self.gb_max_s
        return ((s - self.cur_s + L / 2) % L) - L / 2

    def _gb_idx(self, s):
        """(언랩된) s -> 글로벌 웨이포인트 인덱스. 스칼라/배열 모두 지원."""
        idx = np.floor((np.asarray(s) % self.gb_max_s) / self._wpnt_dist).astype(int) % self.gb_max_idx
        return int(idx) if np.ndim(idx) == 0 else idx

    #################### UTILS FUNCTIONS####################
    def _predict_obs_movement(self, obs: Obstacle, mode: str = "constant") -> Obstacle:
        """
        Predicts the movement of an obstacle based on the current state and mode.

        TODO: opponent prediction should be completely isolated for added modularity

        Args:
            obs (Obstacle): The obstacle to predict the movement for.
            mode (str, optional): The mode for predicting the movement. Defaults to "constant".

        Returns:
            Obstacle: The updated obstacle with the predicted movement.
        """
        # propagate opponent by time dependent on distance
        if (obs.s_center - self.cur_s) % self.gb_max_s < self.lookahead:
            if mode == "adaptive":
                # distance in s coordinate
                cur_s = self.cur_s
                ot_distance = (obs.s_center - cur_s) % self.gb_max_s
                rel_speed = np.clip(self.gb_scaled_wpnts.wpnts[int(
                    cur_s * 10)].vx_mps - obs.vs, 0.1, 10)
                ot_time_distance = np.clip(ot_distance / rel_speed, 0, 5) * 0.5

                delta_s = ot_time_distance * obs.vs
                delta_d = ot_time_distance * obs.vd
                delta_d = -(obs.d_center + delta_d) * \
                    np.exp(-np.abs(self.kd_obs_pred * obs.d_center))

            elif mode == "adaptive_velheuristic":
                opponent_scaler = 0.7
                cur_s = self.cur_s
                ego_speed = self.gb_scaled_wpnts.wpnts[int(cur_s * 10)].vx_mps

                # distance in s coordinate
                ot_distance = (obs.s_center - cur_s) % self.gb_max_s
                rel_speed = (1 - opponent_scaler) * ego_speed
                ot_time_distance = np.clip(ot_distance / rel_speed, 0, 5)

                delta_s = ot_time_distance * opponent_scaler * ego_speed
                delta_d = -(obs.d_center) * \
                    np.exp(-np.abs(self.kd_obs_pred * obs.d_center))

            # propagate opponent by constant time
            elif mode == "constant":
                delta_s = self.fixed_pred_time * obs.vs
                delta_d = self.fixed_pred_time * obs.vd
                # delta_d = -(obs.d_center+delta_d)*np.exp(-np.abs(self.kd_obs_pred*obs.d_center))

            elif mode == "heuristic":
                # distance in s coordinate
                ot_distance = (obs.s_center - self.cur_s) % self.gb_max_s
                rel_speed = 3
                ot_time_distance = ot_distance / rel_speed

                delta_s = 0.0
                delta_d = ot_time_distance * obs.vd
                delta_d = -(obs.d_center + delta_d) * \
                    np.exp(-np.abs(self.kd_obs_pred * obs.d_center))

            # update
            obs.s_start += delta_s
            obs.s_center += delta_s
            obs.s_end += delta_s
            obs.s_start %= self.gb_max_s
            obs.s_center %= self.gb_max_s
            obs.s_end %= self.gb_max_s

            obs.d_left += delta_d
            obs.d_center += delta_d
            obs.d_right += delta_d

            resp = self.converter.get_cartesian([obs.s_center], [obs.d_center])

            marker = self.xy_to_point(resp[0], resp[1], opponent=True)
            # 장애물마다 다른 id를 써야 RViz에서 서로 덮어쓰지 않음
            marker.id = int(getattr(obs, 'id', 0))
            self.pub_propagated.publish(marker)

        return obs

    def _check_ot_side_possible(self, switched: bool, log: bool = True) -> bool:
        """raceline에서 벗어난 상태에서 '진짜' 방향 전환(switched)은 허용하지 않는다."""
        # TODO make rosparam for cur_d threshold
        if abs(self.cur_d) > 0.5 and switched:
            if log:
                self.get_logger().info("Can't switch sides, because we are not on the raceline")
            return False
        return True

    def _prev_valid(self) -> bool:
        return (time.perf_counter() - self._prev_stamp) < self.PREV_MAX_AGE_S

    def _prev_side_of(self, c: "_Cluster") -> Optional[str]:
        """직전 계획에서 같은 위치에 있던 클러스터의 side. 없으면 None."""
        if not self._prev_valid():
            return None
        L = self.gb_max_s
        for s_prev, side_prev in self._prev_sides:
            if abs((c.s_mid - s_prev + L / 2) % L - L / 2) < self.MATCH_TOL_M:
                return side_prev
        return None

    def _margins(self, blocking: bool) -> Tuple[float, float]:
        """현재 완화 단계(self._relax)에서의 (장애물 간격, 트랙 경계 여유)."""
        base_clr = self.evasion_dist if blocking else self.collision_clearance
        if self._relax >= 1.0:
            return base_clr, self.spline_bound_mindist
        clr = min(base_clr, max(base_clr * self._relax, self.collision_clearance))
        bound = min(self.spline_bound_mindist, max(self.spline_bound_mindist * self._relax, self.MIN_BOUND_MARGIN))
        return clr, bound

    #################### MULTI-OBSTACLE FUNCTIONS####################
    def _collect_obstacles(self, obstacles: ObstacleArray) -> List[_ObsInfo]:
        """
        모든 장애물을 예측·언랩해서 반환한다.
        (기존 _obs_filtering과 달리 레이스라인 밖 장애물도 '충돌 검사용'으로 남겨둔다.)
        """
        L = self.gb_max_s
        infos = []
        for raw in obstacles.obstacles:
            # 원본 메시지를 직접 수정하면, 새 메시지가 안 들어온 루프에서 예측이 누적됨 -> 복사본 사용
            obs = self._predict_obs_movement(copy.deepcopy(raw))

            length = (obs.s_end - obs.s_start) % L
            if length > L / 2:          # 비정상 값 방어
                length = 0.0
            s_end_rel = self._signed_ds(obs.s_end)
            if s_end_rel <= -self.REAR_KEEP_M:  # 차체가 완전히 벗어난 장애물만 제외 (뒤쪽 차체가 아직 옆에 있을 수 있음)
                continue
            s1 = self.cur_s + s_end_rel
            s0 = s1 - length
            d_r, d_l = sorted((float(obs.d_right), float(obs.d_left)))
            infos.append(_ObsInfo(
                obs=obs, s0=s0, s1=s1, d_r=d_r, d_l=d_l, d_c=float(obs.d_center),
                blocking=abs(obs.d_center) < self.obs_traj_tresh))
        infos.sort(key=lambda o: o.s0)
        return infos

    def _cluster_obstacles(self, infos: List[_ObsInfo]) -> List[_Cluster]:
        """s 방향으로 겹치거나 merge_gap 이내로 붙은 장애물을 하나의 클러스터로 묶는다."""
        clusters: List[_Cluster] = []
        for info in infos:
            if clusters and info.s0 - clusters[-1].s1 < self.merge_gap:
                c = clusters[-1]
                c.members.append(info)
                c.s1 = max(c.s1, info.s1)
                c.blocking = c.blocking or info.blocking
            else:
                clusters.append(_Cluster(members=[info], s0=info.s0, s1=info.s1, blocking=info.blocking))
            info.cluster = clusters[-1]

        for c in clusters:
            ref = [m.d_c for m in c.members if m.blocking] or [m.d_c for m in c.members]
            c.d_ref = float(np.mean(ref))
            c.outside = self._outside(c)
        return clusters

    def _outside(self, c: _Cluster) -> str:
        # Approximate next 20 indexes of global wpnts with wrapping => 2m and compute which side is the outside of the raceline
        idxs = self._gb_idx(c.s_mid + np.arange(20) * self._wpnt_dist)
        return "left" if np.sum(self._gb_kappa[idxs]) < 0 else "right"

    def _cluster_options(self, c: _Cluster, hint: float = 0.0) -> List[Tuple[float, str]]:
        """
        클러스터를 지나갈 수 있는 d 선택지들을 계산한다.
        각 장애물의 [d_right - clr, d_left + clr]을 금지 구간으로 놓고, 트랙 경계 안의 빈 구간마다
        hint(기본 0 = 레이스라인)에 가장 가까운 d를 apex로 고른다.
        -> 장애물 1개일 땐 기존 _more_space의 apex 계산과 동일한 결과.
        좁은 길목에서는 self._relax < 1 로 clr/경계 여유를 줄여 호출되며, 이때는 통로 안쪽으로 한 걸음 들어간 위치를 고른다.
        """
        idxs = self._gb_idx(np.arange(c.s0, c.s1 + self._wpnt_dist, self._wpnt_dist))
        _, bound = self._margins(True)
        left_lim = float(np.min(self._gb_tb_left[idxs])) - bound
        right_lim = -(float(np.min(self._gb_tb_right[idxs])) - bound)

        forbidden = []
        for m in c.members:
            clr, _ = self._margins(m.blocking)
            forbidden.append([m.d_r - clr, m.d_l + clr])
        forbidden.sort()
        merged = []
        for lo, hi in forbidden:
            if merged and lo <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], hi)
            else:
                merged.append([lo, hi])

        free = []
        cursor = right_lim
        for lo, hi in merged:
            if lo > cursor:
                free.append((cursor, lo))
            cursor = max(cursor, hi)
        if left_lim > cursor:
            free.append((cursor, left_lim))

        options = []
        for lo, hi in free:
            if self._relax < 1.0:
                pad = min(0.1, 0.5 * (hi - lo))   # 좁으면 한가운데, 넓으면 양쪽 벽에서 0.1m 떨어진 곳
                d = float(np.clip(hint, lo + pad, hi - pad))
            else:
                d = float(np.clip(hint, lo, hi))
            side = "left" if d >= c.d_ref else "right"
            options.append((d, side))
        options.sort(key=lambda o: abs(o[0] - hint))
        return options[:self.MAX_OPTIONS]

    def _returns_to_raceline(self, c0: _Cluster, k0: float, c1: _Cluster, k1: float) -> bool:
        """
        앞 클러스터의 post knot가 전부 끝난 뒤에야 뒤 클러스터의 pre knot가 시작할 때만 '완전 복귀'로 본다.
        (예전에는 post_apex_0 vs pre_apex_2 만 비교해서, 같은 쪽 장애물 사이에서 d=0 근처까지 잠깐 갔다 오는 지그재그가 생겼다)
        """
        return c0.s1 + self.post_apex_2 * k0 + self.MIN_RETURN_GAP <= c1.s0 + self.pre_apex_0 * k1

    def _evaluate_combo(self, planned: List[_Cluster], combo) -> Optional[Tuple[float, list]]:
        """클러스터별 선택 조합 하나의 비용 계산. 불가능하면 None."""
        base_k = float(np.clip(1.0 + self.cur_vs / self.gb_vmax, 1, 1.5))
        plan = []
        for c, (d, side) in zip(planned, combo):
            # If we overtake on the outside, we smoothen the spline
            k = base_k * (1.75 if c.outside == side else 1.0)  # TODO make parameter
            plan.append((c, d, side, k))

        cost = sum(abs(d) for _, d, _, _ in plan)
        slope_limit = self.max_lateral_slope / base_k   # 속도가 높을수록 완만하게
        for (c0, d0, _, k0), (c1, d1, _, k1) in zip(plan, plan[1:]):
            if not self._returns_to_raceline(c0, k0, c1, k1):
                gap = max(c1.s0 - c0.s1, 1e-3)
                if abs(d1 - d0) / gap > slope_limit:
                    return None
                cost += self.TRANSITION_COST * abs(d1 - d0)

        first = next(((c, side) for c, _, side, _ in plan if c.blocking), None)
        # 직전 계획의 같은 클러스터와 방향이 달라질수록 비용 (좌우 떨림 방지)
        for c, _, side, _ in plan:
            prev = self._prev_side_of(c)
            if prev is None and first is not None and c is first[0]:
                prev = self.last_ot_side or None
            if prev is not None and prev != side:
                cost += self.SIDE_HYST_COST

        if first is not None:
            c, side = first
            # '진짜' side 전환만 거부한다. 지그재그에서 첫 장애물을 지나 다음 클러스터가 첫 번째가 되는 것은 전환이 아니다.
            ref = self._prev_side_of(c) or self.last_ot_side
            if not self._check_ot_side_possible(side != ref, log=False):
                self._side_rejected = True
                return None
        return cost, plan

    def _build_path(self, plan) -> Tuple[np.ndarray, np.ndarray]:
        """
        클러스터별 apex 정보를 knot 열로 바꾸고 PCHIP으로 보간한다.
        - 클러스터 구간 [s0, s1]은 apex d를 유지(plateau) -> 장애물 길이 전체에서 간격 보장
        - 클러스터 사이가 충분히 멀면 post/pre knot(d=0)로 레이스라인 복귀, 아니면 apex끼리 직접 연결
        - PCHIP은 knot 사이에서 오버슈트가 없어서 기존처럼 clip할 필요가 없음
        """
        res = self.SPLINE_RESOLUTION
        pre = (self.pre_apex_0, self.pre_apex_1, self.pre_apex_2)
        post = (self.post_apex_0, self.post_apex_1, self.post_apex_2)

        knots = []  # (s, d, priority) ; apex=1, 레이스라인 복귀점=0
        for i, (c, d, _, k) in enumerate(plan):
            pre_s = [c.s0 + p * k for p in pre]
            if i == 0:
                knots += [(s, 0.0, 0) for s in pre_s]
            else:
                pc, _, _, pk = plan[i - 1]
                if self._returns_to_raceline(pc, pk, c, k):   # _evaluate_combo와 같은 기준 (완전 복귀)
                    knots += [(s, 0.0, 0) for s in [pc.s1 + p * pk for p in post]]
                    knots += [(s, 0.0, 0) for s in pre_s]
            if c.s1 - c.s0 < 2 * res:
                knots.append((c.s_mid, d, 1))
            else:
                knots += [(c.s0, d, 1), (c.s1, d, 1)]
        c, _, _, k = plan[-1]
        knots += [(c.s1 + p * k, 0.0, 0) for p in post]

        knots.sort(key=lambda kn: kn[0])
        clean = []
        for kn in knots:
            if clean and kn[0] - clean[-1][0] < 0.05:
                if kn[2] > clean[-1][2]:
                    clean[-1] = kn
                continue
            clean.append(kn)

        xs = np.array([kn[0] for kn in clean])
        ys = np.array([kn[1] for kn in clean])
        interp = PchipInterpolator(xs, ys)
        s = np.arange(xs[0], xs[-1], res)
        return s, interp(s)

    def _validate_path(self, s: np.ndarray, d: np.ndarray,
                       infos: List[_ObsInfo]) -> Tuple[bool, Optional[_ObsInfo]]:
        """트랙 경계 + 모든 장애물(레이스라인 밖 포함)과의 충돌 검사. (ok, 충돌한 장애물)"""
        res = self.SPLINE_RESOLUTION
        # Check if wpnt is too close to the trackbounds but only if spline is actually off the raceline
        idx = self._gb_idx(s)
        _, bound = self._margins(True)
        lim = np.where(d > 0, self._gb_tb_left[idx], self._gb_tb_right[idx]) - bound
        if np.any((np.abs(d) > res) & (np.abs(d) > lim)):
            return False, None

        for ob in infos:
            mask = (s >= ob.s0 - res / 2) & (s <= ob.s1 + res / 2)
            if not mask.any():
                continue
            clr = min(self.collision_clearance,
                      self.evasion_dist if ob.blocking else self.collision_clearance)
            dd = d[mask]
            if np.any((dd > ob.d_r - clr) & (dd < ob.d_l + clr)):
                return False, ob
        return True, None

    def _blend_with_prev(self, s: np.ndarray, d: np.ndarray, infos: List[_ObsInfo]):
        """
        새 경로를 직전에 publish한 경로와 이어 붙인다. 차량 바로 앞은 직전 경로를 유지하고,
        멀어질수록 새 경로로 전환(smoothstep)한다. 클러스터가 계획에 들어오고 나가거나 선택지가 바뀌어도
        차량 위치에서 경로가 순간이동하지 않는다. 이어 붙인 경로가 충돌/경계 검사를 못 통과하면 새 경로를 그대로 쓴다.
        """
        if self._prev_path is None or not self._prev_valid():
            return s, d
        L = self.gb_max_s
        ps_abs, pd = self._prev_path
        ps = self.cur_s + (((ps_abs - self.cur_s + L / 2) % L) - L / 2)
        if not (ps[0] <= self.cur_s <= ps[-1]):   # 직전 경로가 차량 위치를 덮지 않으면 이을 필요 없음
            return s, d

        res = self.SPLINE_RESOLUTION
        s_full = np.arange(min(s[0], self.cur_s - self.BLEND_BACK_M), s[-1], res)
        d_new = np.interp(s_full, s, d, left=0.0, right=0.0)   # 새 경로 시작 전은 레이스라인(d=0)
        d_prev = np.interp(s_full, ps, pd, left=np.nan, right=np.nan)
        t = np.clip((s_full - self.cur_s - self.BLEND_HOLD_M) / self.BLEND_LEN_M, 0.0, 1.0)
        w = t * t * (3.0 - 2.0 * t)
        d_blend = np.where(np.isnan(d_prev), d_new, w * d_new + (1.0 - w) * d_prev)

        ok, _ = self._validate_path(s_full, d_blend, infos)
        if not ok:
            return s, d
        return s_full, d_blend

    def _best_plan(self, planned: List[_Cluster], infos: List[_ObsInfo]):
        """planned 클러스터들에 대해 최적 조합을 찾는다. (결과 or None, promotion 후보 or None)"""
        if any(not c.options for c in planned):
            return None, None

        candidates = []
        for combo in itertools.product(*[c.options for c in planned]):
            r = self._evaluate_combo(planned, combo)
            if r is not None:
                candidates.append(r)
        candidates.sort(key=lambda r: r[0])

        promote = None
        for _, plan in candidates[:self.MAX_PATH_TRIALS]:
            s, d = self._build_path(plan)
            ok, hit = self._validate_path(s, d, infos)
            if ok:
                return (plan, s, d), None
            if hit is not None and promote is None and all(hit.cluster is not c for c in planned):
                hint = float(np.interp(hit.s_mid, s, d))
                promote = (hit.cluster, hint)
        return None, promote

    def _plan_with_promotion(self, base: List[_Cluster], infos: List[_ObsInfo], excluded: set):
        planned = list(base)
        for _ in range(self.MAX_PROMOTIONS + 1):
            result, promote = self._best_plan(planned, infos)
            if result is not None:
                return result
            if promote is None:
                return None
            cl, hint = promote
            if id(cl) in excluded:
                return None
            # 경로가 원래 지나가려던 d(hint)에 가장 가까운 통과 위치로 선택지 생성
            cl.options = self._cluster_options(cl, hint)
            if not cl.options:
                return None
            planned = sorted(planned + [cl], key=lambda c: c.s0)
        return None

    def _plan_evasion(self, clusters: List[_Cluster], infos: List[_ObsInfo]):
        """
        blocking 클러스터 전부 -> 안 되면 먼 것부터 하나씩 빼면서 계획.
        각 단계에서 먼저 정상 간격으로 시도하고, 통로가 없으면 간격을 단계적으로 줄여(좁은 길목) 다시 시도한다.
        """
        targets = [c for c in clusters
                   if any(m.blocking and (m.s_mid - self.cur_s) < self.lookahead for m in c.members)]
        targets = targets[:self.MAX_CLUSTERS]
        if not targets:
            return None

        for n in range(len(targets), 0, -1):
            excluded = {id(c) for c in targets[n:]}
            for relax in self.RELAX_SCALES:
                self._relax = relax
                for c in clusters:
                    c.options = self._cluster_options(c, 0.0)
                result = self._plan_with_promotion(targets[:n], infos, excluded)
                if result is not None:
                    if relax < 1.0:
                        self.get_logger().info(
                            f"Narrow passage: evasion clearance relaxed to x{relax:.2f}, slowing down")
                    if n < len(targets):
                        self.get_logger().info(
                            f"Multi-obstacle evasion: only first {n}/{len(targets)} clusters are passable for now")
                    return result
        self._relax = 1.0
        return None

    #################### MAIN FUNCTIONS####################
    def spliner_loop(self):
        # Main loop for the spliner node
        if self.measuring:
            start = time.perf_counter()

        # Sample data
        obs = self.obs
        gb_scaled_wpnts = self.gb_scaled_wpnts.wpnts
        wpnts = OTWpntArray()
        mrks = MarkerArray()

        # If obs then do splining around it
        if len(obs.obstacles) > 0:
            wpnts, mrks = self.do_spline(
                obstacles=obs, gb_wpnts=gb_scaled_wpnts)
        # Else delete spline markers
        else:
            del_mrk = Marker()
            del_mrk.header.stamp = self.get_clock().now().to_msg()
            del_mrk.action = Marker.DELETEALL
            mrks.markers.append(del_mrk)

        # Publish wpnts and markers
        if self.measuring:
            end = time.perf_counter()
            self.latency_pub.publish(Float32(data=float(end - start)))
        self.evasion_pub.publish(wpnts)
        self.mrks_pub.publish(mrks)

    def do_spline(self, obstacles: ObstacleArray, gb_wpnts: WpntArray) -> Tuple[WpntArray, MarkerArray]:
        """
        Creates ONE evasion trajectory that passes ALL relevant obstacles ahead.

        Args:
        - obstacles (ObstacleArray): An array of obstacle objects to be evaded.
        - gb_wpnts (WpntArray): A list of global waypoints that describe a reference raceline.

        Returns:
        - wpnts (OTWpntArray): Contiguous waypoints (0.1 m spacing in s) of the evasion trajectory.
          ot_side / ot_line refer to the first (closest) blocking obstacle cluster.
        - mrks (MarkerArray): Markers of the waypoints in a visualization format.
        """
        # Return wpnts and markers
        mrks = MarkerArray()
        wpnts = OTWpntArray()

        # 경로 길이가 매번 달라지므로 이전 마커를 먼저 지움
        del_mrk = Marker()
        del_mrk.header.stamp = self.get_clock().now().to_msg()
        del_mrk.action = Marker.DELETEALL
        mrks.markers.append(del_mrk)

        # Get spacing between wpnts for rough approximations
        self._wpnt_dist = gb_wpnts[1].s_m - gb_wpnts[0].s_m

        infos = self._collect_obstacles(obstacles)
        clusters = self._cluster_obstacles(infos)

        # 레이스라인을 막는 장애물이 lookahead 안에 없으면 회피 불필요 (기존 동작과 동일)
        if not any(m.blocking and (m.s_mid - self.cur_s) < self.lookahead
                   for c in clusters for m in c.members):
            return wpnts, mrks

        self._side_rejected = False
        result = self._plan_evasion(clusters, infos)
        danger_flag = result is None

        if danger_flag:
            if self._side_rejected:
                self.get_logger().info("Can't switch sides, because we are not on the raceline")
            self.get_logger().info(
                "No collision-free (multi-)obstacle evasion within TRACKBOUNDS, aborting evasion")
        else:
            plan, evasion_s, evasion_d = result
            first = next((p for p in plan if p[0].blocking), plan[0])
            more_space, outside = first[2], first[0].outside
            relax_speed = 0.5 + 0.5 * self._relax   # 좁은 길목을 통과할 때는 감속 (1.0 / 0.875 / 0.775)

            # 직전 경로와 이어 붙여서 차량 앞 경로가 갑자기 바뀌지 않게 한다
            evasion_s, evasion_d = self._blend_with_prev(evasion_s, evasion_d, infos)

            # Publish the points around which we are splining
            for i, (c, _, _, _) in enumerate(plan):
                gi = self._gb_idx(c.s_mid)
                mrk = self.xy_to_point(x=gb_wpnts[gi].x_m, y=gb_wpnts[gi].y_m, opponent=False)
                mrk.id = i
                self.closest_obs_pub.publish(mrk)

            # 각 점에서 가장 가까운 클러스터 기준으로 안쪽 추월이면 감속
            dist_to_cl = np.stack([np.maximum.reduce([c.s0 - evasion_s, np.zeros_like(evasion_s), evasion_s - c.s1])
                                   for c, _, _, _ in plan])
            nearest = np.argmin(dist_to_cl, axis=0)
            on_outside = np.array([c.outside == side for c, _, side, _ in plan])

            # Handle Wrapping of s
            evasion_s = evasion_s % self.gb_max_s
            resp = self.converter.get_cartesian(evasion_s, evasion_d)
            gb_idx = self._gb_idx(evasion_s)

            for i in range(evasion_s.shape[0]):
                # Get V from gb wpnts and go slower if we are going through the inside
                # TODO make speed scaling ros param
                v_gb = gb_wpnts[gb_idx[i]].vx_mps
                vi = (v_gb if on_outside[nearest[i]] else v_gb * 0.9) * relax_speed
                wpnts.wpnts.append(
                    self.xyv_to_wpnts(
                        x=resp[0, i], y=resp[1, i], s=evasion_s[i], d=evasion_d[i], v=vi, wpnts=wpnts)
                )
                mrks.markers.append(self.xyv_to_markers(
                    x=resp[0, i], y=resp[1, i], v=vi, mrks=mrks))

        # Fill the rest of OTWpnts
        wpnts.header.stamp = self.get_clock().now().to_msg()
        wpnts.header.frame_id = "map"
        if not danger_flag:
            wpnts.ot_side = more_space
            wpnts.ot_line = outside
            # 직전 계획에 같은 클러스터가 있었으면 그 클러스터의 이전 side와 비교한다. (지그재그에서 가짜 side_switch 방지)
            ref_side = self._prev_side_of(first[0]) or self.last_ot_side
            switched = more_space != ref_side
            wpnts.side_switch = bool(switched)
            wpnts.last_switch_time = self.last_switch_time

            # Update the last switch time and the last side
            if switched:
                self.last_switch_time = self.get_clock().now().to_msg()
            self.last_ot_side = more_space

            # 다음 프레임의 연속성/히스테리시스용 상태 저장
            self._prev_sides = [(c.s_mid % self.gb_max_s, side) for c, _, side, _ in plan]
            self._prev_path = (evasion_s % self.gb_max_s, np.array(evasion_d))
            self._prev_stamp = time.perf_counter()
        else:
            wpnts.wpnts = []
            mrks.markers = [del_mrk]
            # This fools the statemachine to cool down
            wpnts.side_switch = True
            self.last_switch_time = self.get_clock().now().to_msg()
        return wpnts, mrks

    #################### VISUALIZATION FUNCTIONS####################
    def xyv_to_markers(self, x: float, y: float, v: float, mrks: MarkerArray) -> Marker:
        mrk = Marker()
        mrk.header.frame_id = "map"
        mrk.header.stamp = self.get_clock().now().to_msg()
        mrk.type = mrk.CYLINDER
        mrk.scale.x = 0.1
        mrk.scale.y = 0.1
        mrk.scale.z = float(v / self.gb_vmax)
        mrk.color.a = 1.0
        mrk.color.b = 0.75
        mrk.color.r = 0.75
        if self.from_bag:
            mrk.color.g = 0.75

        mrk.id = len(mrks.markers)
        mrk.pose.position.x = float(x)
        mrk.pose.position.y = float(y)
        mrk.pose.position.z = float(v / float(self.gb_vmax / 2))
        mrk.pose.orientation.w = 1.0

        return mrk

    def xy_to_point(self, x: float, y: float, opponent=True) -> Marker:
        mrk = Marker()
        mrk.header.frame_id = "map"
        mrk.header.stamp = self.get_clock().now().to_msg()
        mrk.type = mrk.SPHERE
        mrk.scale.x = 0.5
        mrk.scale.y = 0.5
        mrk.scale.z = 0.5
        mrk.color.a = 0.8
        mrk.color.b = 0.65
        mrk.color.r = 1.0 if opponent else 0.0
        mrk.color.g = 0.65

        mrk.pose.position.x = float(x)
        mrk.pose.position.y = float(y)
        mrk.pose.position.z = 0.01
        mrk.pose.orientation.w = 1.0

        return mrk

    def xyv_to_wpnts(self, s: float, d: float, x: float, y: float, v: float, wpnts: OTWpntArray) -> Wpnt:
        wpnt = Wpnt()
        wpnt.id = len(wpnts.wpnts)
        wpnt.x_m = float(x)
        wpnt.y_m = float(y)
        wpnt.s_m = float(s)
        wpnt.d_m = float(d)
        wpnt.vx_mps = float(v)
        return wpnt


def main(args=None):
    rclpy.init(args=args)
    spline_planner = ObstacleSpliner()
    rclpy.spin(spline_planner)
    spline_planner.destroy_node()
    rclpy.shutdown()
