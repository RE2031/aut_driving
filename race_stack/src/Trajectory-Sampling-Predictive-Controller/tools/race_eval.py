#!/usr/bin/env python3
"""Multi-lap race evaluation like race_manager: obstacles re-drawn every lap with one seeded RNG.

Lap 1 starts from standstill on the line; laps 2.. are flying laps (what best_lap counts).
A collision ends the run (the sim stops the car), recorded as crash on that lap.

Usage: python3 race_eval.py <tspc_node.py> <params.yaml> [--seeds 0-9] [--laps 3] [--nobs 5]
                            [--no-obs] [--set k:=v ...]
"""
import argparse
import math
import random
import types

import numpy as np
import yaml

import offline_eval as h
import rclpy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from f110_gym.envs.base_classes import RaceCar
from race_manager.placement import generate_candidates, draw, parse_recipe


class Spawner:
    def __init__(self, track, seed, rm, nobs):
        p = rm['obstacle_spawner']['ros__parameters']
        self.p = p
        self.track = track
        self.nobs = nobs
        self.cands = generate_candidates(
            track, size=p['obstacle_size'], wall_margin=p['wall_margin'], car_width=p['car_width'],
            squeeze_clearance=p['squeeze_clearance'], spacing=p['candidate_spacing'],
            d_step=p['candidate_d_step'], thetas=list(p['candidate_thetas']),
            exclusion_ahead=p['exclusion_ahead'], exclusion_behind=p['exclusion_behind'],
            min_severity=p['min_severity'])
        self.rng = random.Random(seed)
        self.recipes = [(r, *parse_recipe(r)) for r in p['recipes']]

    def place(self):
        p = self.p
        recipe, spread, slots = self.rng.choice(self.recipes)
        chosen, _ = draw(self.cands, self.nobs, self.rng, self.track, wall_margin=p['wall_margin'],
                         size=p['obstacle_size'], min_obstacle_gap=p['min_obstacle_gap'],
                         slots=slots, chicane_span=p['chicane_span'], car_width=p['car_width'],
                         squeeze_clearance=p['squeeze_clearance'], spread=spread)
        RaceCar.static_obstacles = [(c.x, c.y, c.theta, p['obstacle_size'], p['obstacle_size'])
                                    for c in chosen]
        return recipe, chosen


def run(mod, spawner, laps, t_max, verbose=False, obs_delay=0, cmd_delay=0, live=False):
    import collections
    sim_params = {k: float(v) for k, v in yaml.safe_load(open(h.SIM_PARAMS)).items()}
    env = h.gym.make('f110_gym:f110-v1', map=f'{h.MAP_DIR}/berlin', map_ext='.png',
                     params=sim_params, num_agents=1, disable_env_checker=True).unwrapped
    RaceCar.static_obstacles = []
    recipes = []
    if spawner:
        recipes.append(spawner.place()[0])

    clock = {'t': 0.0}
    mod.time = types.SimpleNamespace(monotonic=lambda: clock['t'], time=lambda: clock['t'])
    node = mod.MPCNode()
    for name in ('drive_pub', 'ref_traj_pub', 'wpts_pub', 'pred_path_pub', 'obs_pub'):
        setattr(node, name, h.Sink())
    if not verbose:
        node.get_logger().set_level(rclpy.logging.LoggingSeverity.ERROR)

    wp = node.waypoints
    obs, _, _, _ = env.reset(poses=np.array([[wp[0, 1], wp[0, 2], wp[0, 3]]]))
    scan = LaserScan()
    scan.angle_min = -4.7 / 2
    scan.angle_increment = 4.7 / 1079
    scan.range_max = 30.0

    L = node.track_length
    s_prev, progress = None, 0.0
    lap_t0, lap_times = 0.0, []
    steer = speed = 0.0
    result = 'timeout'
    k = 0
    obs_hist = collections.deque(maxlen=obs_delay + 1)
    cmd_q = collections.deque([(0.0, 0.0)] * cmd_delay)
    vmax = 0.0
    dd = []
    jumps = []
    stop_t = [0.0, []]
    def stamp(msg, t):
        msg.header.stamp.sec = int(t)
        msg.header.stamp.nanosec = int(round((t - int(t)) * 1e9)) % 1000000000
    while clock['t'] < t_max:
        obs_hist.append((clock['t'] + 1000.0, obs))
        if live:
            # 실제 스택처럼 scan 은 매 스텝(100 Hz), 타임스탬프 포함
            t_o, o = obs_hist[min(1, len(obs_hist) - 1)]   # scan 이 pose 보다 10 ms 새것 (실제 스택과 같은 비동기)
            scan.ranges = [float(r) for r in o['scans'][0]]
            stamp(scan, t_o)
            node.lidar_callback(scan)
        if k % 2 == 0:
            obs_now = obs
            t_o, obs = obs_hist[0]          # 지연된 관측 (obs_delay 스텝 전)
            if not live:
                scan.ranges = [float(r) for r in obs['scans'][0]]
                node.lidar_callback(scan)
            od = Odometry()
            if live:
                stamp(od, t_o)
            od.pose.pose.position.x = float(obs['poses_x'][0])
            od.pose.pose.position.y = float(obs['poses_y'][0])
            th = float(obs['poses_theta'][0])
            od.pose.pose.orientation.z = math.sin(th / 2)
            od.pose.pose.orientation.w = math.cos(th / 2)
            od.twist.twist.linear.x = float(obs['linear_vels_x'][0])
            od.twist.twist.linear.y = float(obs['linear_vels_y'][0])
            node.pose_update(od)
            m = node.drive_pub.last
            cmd_q.append((m.drive.steering_angle, m.drive.speed))
            vmax = max(vmax, float(obs['linear_vels_x'][0]))
            if node.mode == 'STOP':
                stop_t[0] += 0.02
                if not stop_t[1] or stop_t[1][-1][1] < clock['t'] - 0.05:
                    stop_t[1].append([round(clock['t'], 2), clock['t'], round(float(node.s[node.prev_idx]), 1)])
                stop_t[1][-1][1] = clock['t']

            px, py = od.pose.pose.position.x, od.pose.pose.position.y
            i = int(np.argmin((wp[:, 1] - px) ** 2 + (wp[:, 2] - py) ** 2))
            s = node.s[i]
            if node.mode == 'FOLLOW':
                dd.append(float(np.dot([px - wp[i, 1], py - wp[i, 2]], node.nrm[i])))
            if s_prev is not None:
                dsp = (s - s_prev + 0.5 * L) % L - 0.5 * L
                if abs(dsp) > 0.6:
                    jumps.append((round(s_prev, 1), round(s, 1)))
                progress += dsp
            s_prev = s
            if progress >= (len(lap_times) + 1) * L:
                lap_times.append(round(clock['t'] - lap_t0, 2))
                lap_t0 = clock['t']
                if len(lap_times) >= laps:
                    result = 'done'
                    break
                if spawner:
                    recipes.append(spawner.place()[0])
            obs = obs_now
        if cmd_q:
            steer, speed = cmd_q.popleft() if len(cmd_q) > cmd_delay else (steer, speed)
        obs, _, _, _ = env.step(np.array([[steer, speed]]))
        clock['t'] += 0.01
        k += 1
        if obs['collisions'][0]:
            result = (f'crash:{env.sim.agents[0].collision_kind}@lap{len(lap_times) + 1}'
                      f'(s={s_prev:.1f},v={float(obs["linear_vels_x"][0]):.1f},mode={node.mode})')
            break
    fan = round(node.fan_overrides / max(node.fan_checks, 1), 3)
    node.destroy_node()
    env.close()
    drms = round(float(np.sqrt(np.mean(np.square(dd)))), 3) if dd else None
    return {'result': result, 'laps': lap_times, 'vmax': round(vmax, 2), 'fan': fan, 'drms': drms, 'jumps': jumps[:5], 'stop': round(stop_t[0], 2),
            'stops': [(a, round(b - a + 0.02, 2), c) for a, b, c in stop_t[1]][:8],
            'recipes': recipes}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('node_py')
    ap.add_argument('params')
    ap.add_argument('--seeds', default='0-4')
    ap.add_argument('--laps', type=int, default=3)
    ap.add_argument('--nobs', type=int, default=5)
    ap.add_argument('--no-obs', action='store_true')
    ap.add_argument('--tmax', type=float, default=60.0)
    ap.add_argument('--verbose', action='store_true')
    ap.add_argument('--set', action='append', default=[])
    ap.add_argument('--obs-delay', type=int, default=0, help='관측 지연 (10 ms 스텝)')
    ap.add_argument('--cmd-delay', type=int, default=0, help='명령 지연 (20 ms 제어 주기)')
    ap.add_argument('--live', action='store_true', help='scan 100 Hz + 타임스탬프 (실제 스택과 같은 비동기 조건)')
    a = ap.parse_args()
    rclpy.init(args=['--ros-args', '-r', '__node:=tspc_node', '--params-file', a.params,
                     '-p', f'map_dir:={h.MAP_DIR}', '-p', 'map_name:=berlin']
               + sum((['-p', kv] for kv in a.set), []))
    mod = h.load_node_module(a.node_py)
    track = h.Track(f'{h.MAP_DIR}/global_waypoints.json')
    rm = yaml.safe_load(open(h.RM_PARAMS))
    if a.no_obs:
        seeds = [None]
    else:
        lo, hi = (int(x) for x in a.seeds.split('-')) if '-' in a.seeds else (int(a.seeds),) * 2
        seeds = list(range(lo, hi + 1))
    for seed in seeds:
        sp = None if seed is None else Spawner(track, seed, rm, a.nobs)
        r = run(mod, sp, a.laps, a.tmax, a.verbose, a.obs_delay, a.cmd_delay, a.live)
        flying = r['laps'][1:]
        print(f"seed={seed} {r['result']} laps={r['laps']} best_flying="
              f"{min(flying) if flying else None} vmax={r['vmax']} fan_override={r['fan']} d_rms={r['drms']} s_jumps={r['jumps']} stop_time={r['stop']} stops(t,dur,s)={r['stops']}", flush=True)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
