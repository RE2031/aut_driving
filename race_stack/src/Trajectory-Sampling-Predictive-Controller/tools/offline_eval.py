#!/usr/bin/env python3
"""Offline closed-loop test: f110_gym (berlin) + race_manager obstacle draw + TSPC node logic.

ROS 없이 f110_gym(berlin) + race_manager 장애물 배치(시드별 동일) + TSPC 노드 코드를 폐루프로 돌려
완주/충돌/흔들림(wobble)을 측정합니다. ROS humble 과 워크스페이스 install 을 source 한 뒤 실행.

Usage: python3 offline_eval.py <tspc_node.py> <tspc_params.yaml> [--seeds 0-19] [--no-obs] [--laps 1]
                              [--set param:=value ...] [--tmax 60]
"""
import argparse
import importlib.util
import math
import os
import random
import sys
import types

import numpy as np
import yaml

WS = '/home/misys/forza_ws/race_stack'
MAP_DIR = f'{WS}/stack_master/maps/berlin'
SIM_PARAMS = f'{WS}/stack_master/config/SIM/sim_params.yaml'
RM_PARAMS = f'{WS}/utilities/nodes/race_manager/config/race_manager_params.yaml'

import gymnasium as gym  # noqa: E402
import f110_gym  # noqa: E402,F401
import rclpy  # noqa: E402
from nav_msgs.msg import Odometry  # noqa: E402
from sensor_msgs.msg import LaserScan  # noqa: E402
from f110_gym.envs.base_classes import RaceCar  # noqa: E402
from race_manager.track import Track  # noqa: E402
from race_manager.placement import generate_candidates, draw, parse_recipe  # noqa: E402


class Sink:
    def __init__(self):
        self.last = None

    def publish(self, msg):
        self.last = msg


def load_node_module(path):
    spec = importlib.util.spec_from_file_location('tspc_under_test', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_obstacles(track, seed, rm):
    p = rm['obstacle_spawner']['ros__parameters']
    cands = generate_candidates(
        track, size=p['obstacle_size'], wall_margin=p['wall_margin'], car_width=p['car_width'],
        squeeze_clearance=p['squeeze_clearance'], spacing=p['candidate_spacing'],
        d_step=p['candidate_d_step'], thetas=list(p['candidate_thetas']),
        exclusion_ahead=p['exclusion_ahead'], exclusion_behind=p['exclusion_behind'],
        min_severity=p['min_severity'])
    rng = random.Random(seed)
    recipes = [(r, *parse_recipe(r)) for r in p['recipes']]
    recipe, spread, slots = rng.choice(recipes)
    chosen, _ = draw(cands, p['num_obstacles'], rng, track, wall_margin=p['wall_margin'],
                     size=p['obstacle_size'], min_obstacle_gap=p['min_obstacle_gap'],
                     slots=slots, chicane_span=p['chicane_span'], car_width=p['car_width'],
                     squeeze_clearance=p['squeeze_clearance'], spread=spread)
    obs = [(c.x, c.y, c.theta, p['obstacle_size'], p['obstacle_size']) for c in chosen]
    return obs, recipe, chosen


def run(mod, params_file, obstacles, laps=1, t_max=90.0, verbose=False):
    sim_params = {k: float(v) for k, v in yaml.safe_load(open(SIM_PARAMS)).items()}
    env = gym.make('f110_gym:f110-v1', map=f'{MAP_DIR}/berlin', map_ext='.png',
                   params=sim_params, num_agents=1, disable_env_checker=True).unwrapped
    RaceCar.static_obstacles = list(obstacles)

    clock = {'t': 0.0}
    mod.time = types.SimpleNamespace(monotonic=lambda: clock['t'], time=lambda: clock['t'])
    node = mod.MPCNode()
    node.drive_pub = Sink()
    node.ref_traj_pub = Sink()
    node.wpts_pub = Sink()
    node.pred_path_pub = Sink()
    if not verbose:
        node.get_logger().set_level(rclpy.logging.LoggingSeverity.ERROR)

    wp = node.waypoints
    x0, y0, psi0 = wp[0, 1], wp[0, 2], wp[0, 3]
    obs, _, _, _ = env.reset(poses=np.array([[x0, y0, psi0]]))

    fov, beams = 4.7, 1080
    scan = LaserScan()
    scan.angle_min = -fov / 2
    scan.angle_max = fov / 2
    scan.angle_increment = fov / (beams - 1)
    scan.range_min = 0.0
    scan.range_max = 30.0

    dt = 0.01
    steer, speed = 0.0, 0.0
    L = node.track_length
    s_prev = None
    progress = 0.0
    log = {'t': [], 'steer': [], 'd': [], 'v': [], 'mode': []}
    result = 'timeout'
    k = 0
    while clock['t'] < t_max:
        if k % 2 == 0:  # controller @ 50 Hz, scan @ 50 Hz
            scan.ranges = [float(r) for r in obs['scans'][0]]
            node.lidar_callback(scan)
            od = Odometry()
            od.pose.pose.position.x = float(obs['poses_x'][0])
            od.pose.pose.position.y = float(obs['poses_y'][0])
            th = float(obs['poses_theta'][0])
            od.pose.pose.orientation.z = math.sin(th / 2)
            od.pose.pose.orientation.w = math.cos(th / 2)
            od.twist.twist.linear.x = float(obs['linear_vels_x'][0])
            od.twist.twist.linear.y = float(obs['linear_vels_y'][0])
            node.odometry_update(od)
            node.pose_update(od)
            m = node.drive_pub.last
            steer, speed = m.drive.steering_angle, m.drive.speed

            # progress along the raceline
            px, py = od.pose.pose.position.x, od.pose.pose.position.y
            i = int(np.argmin((wp[:, 1] - px) ** 2 + (wp[:, 2] - py) ** 2))
            s = node.s[i]
            d = float(np.dot([px - wp[i, 1], py - wp[i, 2]], node.nrm[i]))
            if s_prev is not None:
                ds = (s - s_prev + 0.5 * L) % L - 0.5 * L
                progress += ds
            s_prev = s
            log['t'].append(clock['t'])
            log['steer'].append(steer)
            log['d'].append(d)
            log['v'].append(float(obs['linear_vels_x'][0]))
            log['mode'].append(node.mode)
            if progress >= laps * L:
                result = 'done'
                break
        obs, _, _, _ = env.step(np.array([[steer, speed]]))
        clock['t'] += dt
        k += 1
        if obs['collisions'][0]:
            kind = env.sim.agents[0].collision_kind
            result = f'crash:{kind}'
            break
    node.destroy_node()
    env.close()
    st = np.array(log['steer'])
    v = np.array(log['v'])
    modes = np.array(log['mode'])
    follow = modes == 'FOLLOW'
    d = np.array(log['d'])
    stats = {
        'result': result,
        't': round(clock['t'], 2),
        'progress': round(progress, 2),
        'steer_rate_rms': round(float(np.sqrt(np.mean((np.diff(st) / 0.02) ** 2))), 3) if st.size > 2 else 0,
        'd_rms_follow': round(float(np.sqrt(np.mean(d[follow] ** 2))), 3) if follow.any() else None,
        'wobble': round(float(np.sqrt(np.mean(
            (st - np.convolve(st, np.ones(25) / 25, mode='same'))[25:-25] ** 2))), 4) if st.size > 60 else None,
        'stop_frac': round(float(np.mean(modes == 'STOP')), 3),
        'avoid_frac': round(float(np.mean(modes == 'AVOID')), 3),
        'v_mean': round(float(v.mean()), 2) if v.size else 0,
    }
    return stats, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('node_py')
    ap.add_argument('params')
    ap.add_argument('--seeds', default='0-9')
    ap.add_argument('--no-obs', action='store_true')
    ap.add_argument('--laps', type=int, default=1)
    ap.add_argument('--verbose', action='store_true')
    ap.add_argument('--tmax', type=float, default=90.0)
    ap.add_argument('--set', action='append', default=[])
    a = ap.parse_args()

    rclpy.init(args=['--ros-args', '-r', '__node:=tspc_node', '--params-file', a.params,
                     '-p', f'map_dir:={MAP_DIR}', '-p', 'map_name:=berlin'] + sum((['-p', kv] for kv in a.set), []))
    mod = load_node_module(a.node_py)
    track = Track(f'{MAP_DIR}/global_waypoints.json')
    rm = yaml.safe_load(open(RM_PARAMS))

    if a.no_obs:
        seeds = [None]
    else:
        lo, hi = (int(x) for x in a.seeds.split('-')) if '-' in a.seeds else (int(a.seeds),) * 2
        seeds = range(lo, hi + 1)
    ok = 0
    for seed in seeds:
        if seed is None:
            obstacles, recipe = [], 'none'
        else:
            obstacles, recipe, chosen = make_obstacles(track, seed, rm)
        stats, _ = run(mod, a.params, obstacles, laps=a.laps, t_max=a.tmax, verbose=a.verbose)
        ok += stats['result'] == 'done'
        print(f'seed={seed} [{recipe}] {stats}', flush=True)
    print(f'== {ok}/{len(list(seeds))} completed')
    rclpy.shutdown()


if __name__ == '__main__':
    main()
