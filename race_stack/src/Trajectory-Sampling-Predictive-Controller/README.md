# Trajectory-Sampling Predictive Controller

**Author:** Mohamed Elgouhary  
**Project:** TSPC Controller for F1TENTH-style Autonomous Driving

A ROS 2 (Humble) Python controller for F1TENTH-style racing: it follows a precomputed raceline and avoids static obstacles detected with LiDAR.

This version was extended for the ForzaETH race stack (`berlin` map, randomly spawned obstacles). Compared with the upstream controller, steering now comes from a path tracker that follows a planned avoidance path. The sampled candidate trajectories are kept as a collision-safety layer. See [Controller overview](#controller-overview).

## Repository structure

```text
Trajectory-Sampling-Predictive-Controller/
├── tspc_controller/                 # ROS 2 ament_python package
│   ├── tspc_controller/
│   │   ├── __init__.py
│   │   ├── tspc_node.py             # the controller node
│   │   └── utils.py                 # upstream helper (not used by the node)
│   ├── package.xml
│   ├── setup.py
│   ├── setup.cfg
│   └── resource/tspc_controller
├── config/
│   ├── tspc_params.yaml             # default: low speed, stable
│   └── tspc_params_fast.yaml        # experimental high-speed setup
├── launch/
│   └── tspc.launch.py
├── tools/                           # offline evaluation scripts (no ROS graph needed)
│   ├── offline_eval.py
│   └── race_eval.py
├── csv_data/README.md               # kept from upstream; no CSV is needed
├── .gitignore
└── README.md
```

## Dependencies

- ROS 2 Humble with `rclpy`, `nav_msgs`, `sensor_msgs`, `geometry_msgs`, `ackermann_msgs`, `visualization_msgs`, `launch`, `launch_ros`, `ament_index_python`
- Python: `numpy`, `scipy`, `PyYAML`, `Pillow`. All are declared in `package.xml`, so `rosdep` installs them.
- `stack_master` (ForzaETH race stack). The launch file uses it to find the map folder.

## Build

Inside a ForzaETH `race_stack` workspace:

```bash
cd ~/forza_ws/race_stack/src
git clone <this repository>
cd ..
rosdep install --from-paths src --ignore-src -r -y
source /opt/ros/humble/setup.bash
colcon build --packages-select tspc_controller   # build the full workspace once first
source install/setup.bash
```

## Running with the ForzaETH simulator

Terminal 1, simulator and base system:

```bash
ros2 launch stack_master base_system_launch.xml map_name:=berlin pose_source:=gt
```

Terminal 2, TSPC:

```bash
ros2 launch tspc_controller tspc.launch.py                                   # default config
ros2 launch tspc_controller tspc.launch.py params:=tspc_params_fast.yaml     # experimental high-speed config
```

Terminal 3, optional obstacles and lap timing:

```bash
ros2 launch race_manager race_manager_launch.xml map_name:=berlin seed:=2026 num_obstacles:=2 plot:=true
```

Launch arguments of `tspc.launch.py`:

| Argument | Default | Meaning |
| --- | --- | --- |
| `map_name` | `berlin` | Map folder under `stack_master/maps` (must contain `global_waypoints.json`) |
| `trajectory_key` | `global_traj_wpnts_iqp` | Trajectory in `global_waypoints.json` (`global_traj_wpnts_sp` for shortest path) |
| `params` | `tspc_params.yaml` | Parameter file in `tspc_controller/config` |

> **Note:** if `params` names a file that does not exist, `ros2 launch` only prints a warning and starts the node with the built-in defaults (`speed_scale: 1.0`, no tuned settings). Check the first lines of the log.

You can also run the node directly: `ros2 run tspc_controller tspc_node --ros-args --params-file <yaml> -p map_dir:=<map folder>`.

## ROS 2 interface

| Direction | Topic | Type | Notes |
| --- | --- | --- | --- |
| Sub | `/car_state/odom` | `nav_msgs/Odometry` | Simulation (`real_environment: false`) |
| Sub | `/pf/viz/inferred_pose` | `geometry_msgs/PoseStamped` | Real car (`real_environment: true`). Speed is estimated from pose differences |
| Sub | `/scan` | `sensor_msgs/LaserScan` | Obstacle detection uses LiDAR only |
| Pub | `/drive` | `ackermann_msgs/AckermannDriveStamped` | Speed sign is inverted when `real_environment: true` |
| Pub | `/ref_traj_marker` | `visualization_msgs/Marker` | Active path (raceline or avoidance path) |
| Pub | `/waypoints_marker` | `visualization_msgs/Marker` | Raceline waypoints |
| Pub | `/pred_path_marker` | `visualization_msgs/MarkerArray` | Candidate trajectories of the safety layer |
| Pub | `/tspc_obstacles_marker` | `visualization_msgs/Marker` | Obstacle points used by the planner (incl. memory) |

The node reads the raceline and track bounds (`d_left`/`d_right`) from `stack_master/maps/<map_name>/global_waypoints.json`, and the static map (`<map_name>.yaml`, or `<map_name>_orig.yaml` if present) for wall filtering. Obstacle positions are never read from the simulator.

## Controller overview

Each pose message runs this pipeline:

1. **Raceline window.** Find the nearest waypoint and build a local Frenet frame (s, d) around the car.
2. **Obstacle points.** LiDAR points are shifted by the laser mounting offset (`lidar_x_offset`) and transformed with the pose at scan time. They are then projected to (s, d). Points outside the track bounds or near a wall of the static map are discarded. The rest go into a short-term grid memory, which keeps obstacles that are occluded or beside the car.
3. **Avoidance path.** If obstacles are within `obstacle_detect_distance`, a dynamic-programming search over a Frenet grid picks a lateral offset profile d(s). It handles all obstacles in the horizon at once (chicanes, pairs, gaps between obstacles), with hard and soft clearance to obstacles and walls. A consistency term keeps the left/right choice stable. Optional alternatives: a quintic lattice planner (`planner: lattice`), a relaxed retry, and a low-speed recovery mode for very tight layouts.
4. **Steering.** Pure Pursuit with speed-dependent lookahead (default), or curvature feed-forward plus Stanley feedback (`tracker: stanley`). Stanley has understeer and actuation-delay compensation.
5. **Speed.** The raceline speed (`vx_mps * speed_scale`) is limited by lateral acceleration and acceleration/braking passes (`speed_ay_max`), then by the path curvature and the distance-based slow-down near obstacles.
6. **Safety layer.** Constant-steering candidate trajectories (or the planned path itself, `safety_mode: path`) are rolled out and checked against LiDAR points. The tracker command is used unless it collides. If every candidate collides, the car stops.
7. **Publish** the Ackermann command and the RViz markers.

## Configurations

| | `tspc_params.yaml` (default) | `tspc_params_fast.yaml` (experimental) |
| --- | --- | --- |
| Speed | `speed_scale 0.25` (max ~2.6 m/s) | full raceline speed, lateral accel ≤ 8 m/s² |
| Steering | Pure Pursuit | Stanley + delay compensation (0.04 s); Pure Pursuit while avoiding |
| Avoidance speed | 1.5 m/s | 1.5 m/s, approached with a 5 m/s² slow-down curve |
| Recovery mode | off | on |

Every parameter is documented in Korean comments inside the YAML files. `tspc_params_fast.yaml` overrides values at the bottom of the file.

### Evaluation status (simulation only)

These results come from the offline harness (`tools/`) on `berlin` with the race_manager obstacle spawner, using ground-truth pose:

- **Default config:** 40/40 seeds (4 random obstacles, one lap each) completed without collision. Lap ≈ 30–36 s.
- **Fast config, no obstacles:** flying lap ≈ 9.0 s. This includes an emulated 40 ms sensor/actuation delay and asynchronous scan/pose.
- **Fast config, 2 obstacles, 10 laps per seed:** best flying lap ≈ 11.8–11.9 s. 1 of 4 seeds still had a collision, so treat this config as experimental.

### Known limitations

- With the default config (`recovery_enable: false`), a chicane of two obstacles about 1 m apart that forces opposite sides can leave the car stopped (`STOP`).
- Testing used `pose_source:=gt`. With the particle filter (`pose_source:=pf`), poses arrive about 30 ms late and are stamped at publish time. Wall points can then be smeared into phantom obstacles during fast turns. This mode is not tuned yet.
- Not tested on a physical car. Check `lidar_x_offset` (base_link → laser), vehicle dimensions, speed limits and the drive speed sign before running on hardware.

## Offline evaluation tools

`tools/offline_eval.py` and `tools/race_eval.py` run the node code in closed loop with `f110_gym` and the race_manager obstacle placement, without a ROS graph. `race_eval.py` re-draws obstacles every lap like the race manager and reports flying laps. It can also emulate delay and asynchronous sensors (`--obs-delay`, `--cmd-delay`, `--live`).

```bash
source /opt/ros/humble/setup.bash && source ~/forza_ws/race_stack/install/setup.bash
cd tools
python3 race_eval.py ../tspc_controller/tspc_controller/tspc_node.py ../config/tspc_params_fast.yaml \
    --seeds 0-3 --nobs 2 --laps 5 --obs-delay 2 --cmd-delay 1 --live
python3 offline_eval.py ../tspc_controller/tspc_controller/tspc_node.py ../config/tspc_params.yaml --seeds 0-9
```

The scripts use workspace paths that are hard-coded to `/home/misys/forza_ws/race_stack` (`WS` in `offline_eval.py`). Adjust them for your machine.

## Author

The original TSPC controller was developed by:

**Mohamed Elgouhary**  
PhD Student and Graduate Research Assistant  
Lane Department of Computer Science and Electrical Engineering  
West Virginia University

## License and attribution

The main TSPC controller implementation was developed by **Mohamed Elgouhary**.

`utils.py` includes an MIT License header and credits the original authors listed in that file. Keep that header if you modify or redistribute the file.

Add your preferred license for the full repository after confirming how you want to release the controller code.

## Notes

This repository is intended for research and development in F1TENTH-style autonomous racing. Before using the controller on a physical vehicle, test it carefully in simulation and verify all topic names, vehicle parameters, speed limits, steering limits and safety behavior.
