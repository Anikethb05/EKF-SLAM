#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess, TimerAction
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Get package directory
    pkg_dir = get_package_share_directory('ekf_slam_gazebo')
    
    # File paths
    world_file = os.path.join(pkg_dir, 'worlds', 'room.world')
    urdf_file = os.path.join(pkg_dir, 'models', 'robot.urdf')
    rviz_config = os.path.join(pkg_dir, 'rviz', 'slam_config.rviz')
    # Load URDF
    with open(urdf_file, 'r') as f:
        robot_desc = f.read()
    
    return LaunchDescription([
        
        # 1. Robot State Publisher
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': robot_desc}]
        ),
        
        # 2. Start Gazebo with ROS factory plugin
        ExecuteProcess(
            cmd=['gazebo', '--verbose', '-s', 'libgazebo_ros_factory.so', world_file],
            output='screen'
        ),
        
        # 3. Spawn Robot (delayed to let Gazebo start)
        TimerAction(
            period=3.0,
            actions=[
                Node(
                    package='gazebo_ros',
                    executable='spawn_entity.py',
                    arguments=[
                        '-topic', 'robot_description',
                        '-entity', 'robot',
                        '-x', '5.0',
                        '-y', '4.0',
                        '-z', '0.1'
                    ],
                    output='screen'
                )
            ]
        ),
        
        # 4. EKF-SLAM Node
        TimerAction(
            period=5.0,
            actions=[
                Node(
                    package='ekf_slam_gazebo',
                    executable='slam_node',
                    name='slam_node',
                    output='screen'
                )
            ]
        ),
        
        # 5. Controller Node  
        TimerAction(
            period=6.0,
            actions=[
                Node(
                    package='ekf_slam_gazebo',
                    executable='controller_node',
                    name='controller_node',
                    output='screen'
                )
            ]
        ),
        
        # 6. RViz2
         # 6. RViz2 with AUTO-CONFIG
        TimerAction(
            period=4.0,
            actions=[
                Node(
                    package='rviz2',
                    executable='rviz2',
                    name='rviz2',
                    arguments=['-d', rviz_config],  # ← AUTO-LOAD CONFIG
                    output='screen'
                )
            ]
        ),
        
    ])
