from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from ament_index_python.packages import get_package_prefix, get_package_share_directory
import os


def generate_launch_description():
    maps_dir = os.path.abspath(os.path.join(
        get_package_prefix('stack_master'), '..', '..', 'stack_master', 'maps'))
    return LaunchDescription([
        DeclareLaunchArgument('map_name', default_value='berlin',
                              description='Map folder under stack_master/maps'),
        DeclareLaunchArgument('trajectory_key', default_value='global_traj_wpnts_iqp',
                              description='Trajectory entry in global_waypoints.json'),
        DeclareLaunchArgument('params', default_value='tspc_params.yaml',
                              description='Parameter file in tspc_controller/config '
                                          '(tspc_params_fast.yaml = experimental high-speed setup)'),
        Node(
            package='tspc_controller',
            executable='tspc_node',
            name='tspc_node',
            output='screen',
            parameters=[
                PathJoinSubstitution([get_package_share_directory('tspc_controller'),
                                      'config', LaunchConfiguration('params')]),
                {
                    'map_name': LaunchConfiguration('map_name'),
                    'map_dir': [maps_dir, '/', LaunchConfiguration('map_name')],
                    'trajectory_key': LaunchConfiguration('trajectory_key'),
                },
            ]
        )
    ])
