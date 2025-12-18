import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/aniketh05/ros2_slam_ws/install/ekf_slam_gazebo'
