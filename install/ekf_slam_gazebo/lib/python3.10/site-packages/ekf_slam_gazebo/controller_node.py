#!/usr/bin/env python3
"""
P-Control Waypoint Tracking Controller
PROPERLY FIXED: Smooth navigation without jittering
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import math
import tf_transformations
import numpy as np


class ControllerNode(Node):
    def __init__(self):
        super().__init__('controller_node')

        # Waypoints
        self.waypoints = [(2.0, 2.0), (8.0, 4.0), (5.0, 7.0)]
        self.wp_idx = 0

        # P-Control gains (TUNED for smooth operation)
        self.Kd = 0.6       # Distance gain (reduced for stability)
        self.Ktheta = 2.0   # Heading gain (moderate)

        # Velocity limits
        self.max_v = 0.35   # Max linear velocity
        self.max_w = 1.0    # Max angular velocity
        self.min_v = 0.05   # Minimum velocity to overcome friction
        
        # Waypoint tolerance
        self.goal_tol = 0.3
        
        # Deadband for heading (prevents jitter)
        self.heading_deadband = 0.08  # ~4.5 degrees

        # Obstacle avoidance parameters
        self.obstacle_range = 0.5
        self.critical_range = 0.25
        self.avoidance_gain = 2.5

        self.pose = None
        self.scan = None
        self.initialized = False

        # Publishers/Subscribers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(Odometry, '/ekf_odom', self.odom_cb, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)

        # Control loop at 20 Hz
        self.create_timer(0.05, self.control_loop)

        self.get_logger().info('Controller node initialized')
        self.get_logger().info(f'Waypoints: {self.waypoints}')
        self.get_logger().info(f'Gains: Kd={self.Kd}, Ktheta={self.Ktheta}')

    def odom_cb(self, msg):
        """Update pose from EKF odometry"""
        self.pose = msg.pose.pose
        if not self.initialized:
            self.initialized = True
            x = self.pose.position.x
            y = self.pose.position.y
            self.get_logger().info(f'🤖 ROBOT STARTING POSITION: ({x:.2f}, {y:.2f})')

    def scan_cb(self, msg):
        """Update LiDAR scan"""
        self.scan = msg

    def normalize(self, a):
        """Normalize angle to [-pi, pi]"""
        return math.atan2(math.sin(a), math.cos(a))

    def compute_obstacle_avoidance(self):
        """Compute obstacle avoidance vector"""
        if self.scan is None:
            return 0.0, 0.0
        
        ranges = np.array(self.scan.ranges)
        angle_min = self.scan.angle_min
        angle_inc = self.scan.angle_increment
        
        repulsive_x = 0.0
        repulsive_y = 0.0
        max_urgency = 0.0
        
        for i in range(len(ranges)):
            r = ranges[i]
            angle = angle_min + i * angle_inc
            
            if abs(angle) > math.pi / 2:
                continue
            
            if self.scan.range_min < r < self.obstacle_range:
                if r < self.critical_range:
                    force_magnitude = 1.0
                else:
                    force_magnitude = (self.obstacle_range - r) / (self.obstacle_range - self.critical_range)
                
                repulsive_x -= force_magnitude * math.cos(angle)
                repulsive_y -= force_magnitude * math.sin(angle)
                
                max_urgency = max(max_urgency, force_magnitude)
        
        if abs(repulsive_x) > 0.01 or abs(repulsive_y) > 0.01:
            avoidance_angle = math.atan2(repulsive_y, repulsive_x)
            return avoidance_angle, max_urgency
        
        return 0.0, 0.0

    def control_loop(self):
        """P-Control loop for waypoint tracking"""
        
        # Wait for pose
        if not self.initialized or self.pose is None:
            return

        # Check if all waypoints reached
        if self.wp_idx >= len(self.waypoints):
            self.cmd_pub.publish(Twist())
            return

        # Extract current state
        x = self.pose.position.x
        y = self.pose.position.y
        q = self.pose.orientation
        _, _, theta = tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])

        # Get target waypoint
        gx, gy = self.waypoints[self.wp_idx]
        dx, dy = gx - x, gy - y
        dist = math.hypot(dx, dy)

        # Check if waypoint reached
        if dist < self.goal_tol:
            self.get_logger().info(
                f'✓✓✓ Waypoint G{self.wp_idx + 1} at ({gx:.1f}, {gy:.1f}) REACHED!\n'
                f'    Robot EKF position: ({x:.2f}, {y:.2f}), distance={dist:.3f}m'
            )
            self.wp_idx += 1
            
            # Stop at waypoint
            self.cmd_pub.publish(Twist())
            return

        # Compute heading error
        desired_heading = math.atan2(dy, dx)
        heading_err = self.normalize(desired_heading - theta)
        
        # Obstacle avoidance - ONLY when heading error is small (not already turning)
        avoidance_angle, urgency = self.compute_obstacle_avoidance()
        speed_factor = 1.0
        
        if urgency > 0.1 and abs(heading_err) < math.pi / 4:  # Only avoid if not already turning
            goal_weight = 1.0 - urgency
            avoid_weight = urgency * self.avoidance_gain
            desired_heading = self.normalize(goal_weight * desired_heading + avoid_weight * avoidance_angle)
            speed_factor = 0.5 + 0.5 * (1.0 - urgency)
            heading_err = self.normalize(desired_heading - theta)

        # Create velocity command
        cmd = Twist()

        # === ANGULAR CONTROL (with deadband to prevent jitter) ===
        if abs(heading_err) < self.heading_deadband:
            cmd.angular.z = 0.0
        else:
            # P-control for angular velocity
            w_raw = self.Ktheta * heading_err
            cmd.angular.z = max(-self.max_w, min(w_raw, self.max_w))

        # === LINEAR CONTROL ===
        # Only move forward if reasonably aligned
        if abs(heading_err) > math.pi / 2:
            # Turn in place if far off
            cmd.linear.x = 0.0
        elif abs(heading_err) > math.pi / 4:
            # Slow down for large heading errors
            v_raw = self.Kd * dist * 0.3 * speed_factor
            if v_raw < self.min_v:
                cmd.linear.x = 0.0
            else:
                cmd.linear.x = min(v_raw, self.max_v)
        else:
            # Normal operation
            v_raw = self.Kd * dist * speed_factor
            if v_raw < self.min_v:
                cmd.linear.x = 0.0
            else:
                cmd.linear.x = min(v_raw, self.max_v)

        # Publish command
        self.cmd_pub.publish(cmd)

        # DETAILED LOGGING (every 0.5 seconds)
        if self.get_clock().now().nanoseconds % int(5e8) < int(5e7):
            self.get_logger().info(
                f'Target: G{self.wp_idx+1}({gx:.1f}, {gy:.1f}) | '
                f'Robot: ({x:.2f}, {y:.2f}) | '
                f'Dist: {dist:.2f}m | '
                f'Heading_err: {math.degrees(heading_err):.1f}° | '
                f'v={cmd.linear.x:.2f}, ω={cmd.angular.z:.2f}'
            )


def main():
    rclpy.init()
    node = ControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()