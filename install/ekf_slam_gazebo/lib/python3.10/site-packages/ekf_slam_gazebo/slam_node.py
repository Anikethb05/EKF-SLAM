#!/usr/bin/env python3
"""
EKF-SLAM Node with Known Correspondences (Thrun Table 10.1)
EXACT MATCH to Pygame implementation
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import TransformStamped
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import TransformBroadcaster
import numpy as np
import math
from scipy.stats import chi2
import tf_transformations
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose


class SLAMNode(Node):
    def __init__(self):
        super().__init__('slam_node')
        self.map_pub = self.create_publisher(OccupancyGrid, '/map', 10)

        # Add occupancy grid
        self.grid_resolution = 0.05  # 5cm
        self.grid_width = 250  # 12.5m
        self.grid_height = 200  # 10m
        self.grid_data = np.full((self.grid_height, self.grid_width), 50, dtype=np.int8)

        # Add timer for map publishing
        self.create_timer(1.0, self.publish_map)
        
        # Known landmark positions (ground truth for correspondence)
        self.TRUE_LANDMARKS = [
            (0.0, 0.0),   # A: Wall corner
            (10.0, 0.0),  # B: Opposite wall corner
            (4.0, 3.0),   # C: Pillar
            (8.0, 5.0),   # D: Table corner
            (2.0, 7.0)    # E: Cabinet edge
        ]
        
        self.N_LANDMARKS = len(self.TRUE_LANDMARKS)
        self.N_STATE = 3  # [x, y, theta]
        self.state_size = self.N_STATE + 2 * self.N_LANDMARKS
        
        # EXACT PYGAME PARAMETERS
        self.ROBOT_FOV = 5.0  # Sensor range
        self.CORRESPONDENCE_THRESHOLD = 0.8  # For landmark association
        
        # Initialize state - UNKNOWN start position (with noise)
        self.mu = np.zeros((self.state_size, 1))
        self.sigma = np.eye(self.state_size) * 1000.0
        
        # Initial robot pose 
        self.mu[0, 0] = 5.0  # Exact X
        self.mu[1, 0] = 4.0  # Exact Y  
        self.mu[2, 0] = 0.0  # Exact theta

        # Very low initial uncertainty (we know exact starting position)
        self.sigma[0, 0] = 0.001  # Almost zero X uncertainty
        self.sigma[1, 1] = 0.001  # Almost zero Y uncertainty
        self.sigma[2, 2] = 0.001  # Almost zero theta uncertainty
        
        # Landmarks start unknown
        self.mu[3:] = np.nan
        
        # Noise matrices (EXACT from Pygame)
        self.R = np.diag([0.01, 0.01, 0.01])  # Process noise
        self.Q = np.diag([0.02, 0.02])        # Measurement noise
        
        self.last_time = None
        
        # TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        
        # Publishers
        self.ekf_pub = self.create_publisher(Odometry, '/ekf_odom', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/landmark_markers', 10)
        
        # Timer for periodic publishing
        self.create_timer(0.1, self.publish_state)
        
        self.get_logger().info('EKF-SLAM Node initialized')
        self.get_logger().info(f'Known landmarks: {self.N_LANDMARKS}')
        self.get_logger().info(f'Sensor range: {self.ROBOT_FOV}m')
    
    def normalize_angle(self, angle):
        return np.arctan2(np.sin(angle), np.cos(angle))
    
    def prediction_step(self, v, w, dt):
        """
        EXACT Pygame prediction_update function
        """
        if dt <= 0 or dt > 0.5:
            return self.mu, self.sigma
        
        theta = self.mu[2, 0]
        mu_bar = self.mu.copy()
        
        # Differential drive kinematics (EXACT)
        if abs(w) > 1e-4:  # Circular motion
            dx = -(v/w) * np.sin(theta) + (v/w) * np.sin(theta + w*dt)
            dy = (v/w) * np.cos(theta) - (v/w) * np.cos(theta + w*dt)
            dtheta = w * dt
        else:  # Straight line
            dx = v * np.cos(theta) * dt
            dy = v * np.sin(theta) * dt
            dtheta = 0.0
        
        mu_bar[0, 0] += dx
        mu_bar[1, 0] += dy
        mu_bar[2, 0] = self.normalize_angle(mu_bar[2, 0] + dtheta)
        
        # Jacobian G_t (EXACT)
        G_t = np.eye(self.state_size)
        
        if abs(w) > 1e-4:
            G_t[0, 2] = (v/w) * (-np.cos(theta) + np.cos(theta + w*dt))
            G_t[1, 2] = (v/w) * (-np.sin(theta) + np.sin(theta + w*dt))
        else:
            G_t[0, 2] = -v * np.sin(theta) * dt
            G_t[1, 2] = v * np.cos(theta) * dt
        
        # Process noise (EXACT)
        R_t = np.zeros((self.state_size, self.state_size))
        R_t[0:3, 0:3] = self.R
        
        # Predicted covariance
        sigma_bar = G_t @ self.sigma @ G_t.T + R_t
        
        return mu_bar, sigma_bar
    
    def extract_landmark_detections(self, scan):
        """
        Extract landmark detections (SIMPLIFIED for known correspondences)
        """
        detections = []
        ranges = np.array(scan.ranges)
        angle_min = scan.angle_min
        angle_inc = scan.angle_increment
        
        # Robot pose
        rx, ry, theta = self.mu[0, 0], self.mu[1, 0], self.mu[2, 0]
        
        # FOV limits (120 degrees = ±60°)
        fov_half = math.radians(60)
        
        # Simple clustering
        in_cluster = False
        cluster = []
        
        for i in range(len(ranges)):
            r = ranges[i]
            
            if scan.range_min < r < self.ROBOT_FOV:
                angle_local = angle_min + i * angle_inc
                
                # Check FOV (Pygame uses 120° FOV)
                if abs(angle_local) > fov_half:
                    continue
                
                cluster.append((r, angle_local))
                in_cluster = True
            else:
                # End of cluster
                if in_cluster and len(cluster) >= 3:
                    avg_r = np.mean([r for r, a in cluster])
                    avg_angle = np.mean([a for r, a in cluster])
                    
                    # Convert to global
                    global_x = rx + avg_r * math.cos(theta + avg_angle)
                    global_y = ry + avg_r * math.sin(theta + avg_angle)
                    
                    # Room bounds check
                    if -0.5 < global_x < 10.5 and -0.5 < global_y < 8.5:
                        detections.append((avg_r, avg_angle, global_x, global_y))
                
                cluster = []
                in_cluster = False
        
        # Last cluster
        if in_cluster and len(cluster) >= 3:
            avg_r = np.mean([r for r, a in cluster])
            avg_angle = np.mean([a for r, a in cluster])
            global_x = rx + avg_r * math.cos(theta + avg_angle)
            global_y = ry + avg_r * math.sin(theta + avg_angle)
            
            if -0.5 < global_x < 10.5 and -0.5 < global_y < 8.5:
                detections.append((avg_r, avg_angle, global_x, global_y))
        
        return detections
    
    def associate_to_landmark(self, global_x, global_y):
        """Associate detection to closest true landmark (KNOWN CORRESPONDENCES)"""
        min_dist = self.CORRESPONDENCE_THRESHOLD
        best_idx = None
        
        for i, (true_x, true_y) in enumerate(self.TRUE_LANDMARKS):
            dist = math.hypot(global_x - true_x, global_y - true_y)
            if dist < min_dist:
                min_dist = dist
                best_idx = i
        
        return best_idx
    
    def measurement_update(self, detections):
        """
        EXACT Pygame measurement_update function
        EKF-SLAM with Known Correspondences (Thrun Table 10.1)
        """
        if len(detections) == 0:
            return
        
        mu = self.mu.copy()
        sigma = self.sigma.copy()
        
        rx, ry, theta = mu[0, 0], mu[1, 0], mu[2, 0]
        
        # Observability guard (EXACT from Pygame)
        # Don't update if robot is stationary
        if hasattr(self, 'last_v') and hasattr(self, 'last_w'):
            if abs(self.last_v) < 0.05 and abs(self.last_w) < 0.05:
                return
        
        for r_meas, phi_meas, global_x, global_y in detections:
            # KNOWN CORRESPONDENCES - associate to true landmark
            lidx = self.associate_to_landmark(global_x, global_y)
            
            if lidx is None:
                continue
            
            lm_idx_start = self.N_STATE + lidx * 2
            
            # LANDMARK INITIALIZATION (EXACT from Pygame)
            if np.isnan(mu[lm_idx_start, 0]):
                lx_init = rx + r_meas * np.cos(theta + phi_meas)
                ly_init = ry + r_meas * np.sin(theta + phi_meas)
                
                mu[lm_idx_start, 0] = lx_init
                mu[lm_idx_start+1, 0] = ly_init
                
                # Initialize covariance (EXACT)
                angle_total = theta + phi_meas
                G_z = np.array([
                    [np.cos(angle_total), -r_meas * np.sin(angle_total)],
                    [np.sin(angle_total),  r_meas * np.cos(angle_total)]
                ])
                
                sigma[lm_idx_start:lm_idx_start+2, lm_idx_start:lm_idx_start+2] = \
                    G_z @ self.Q @ G_z.T + np.diag([0.5, 0.5])
                
                self.get_logger().info(f'Init landmark {chr(65+lidx)} at '
                                      f'({mu[lm_idx_start,0]:.2f}, {mu[lm_idx_start+1,0]:.2f})')
                continue  # initialization only, no update yet
            
            # EKF UPDATE (EXACT from Pygame)
            mx = mu[lm_idx_start, 0]
            my = mu[lm_idx_start + 1, 0]
            
            delta_x = mx - rx
            delta_y = my - ry
            q = delta_x**2 + delta_y**2
            
            if q < 1e-6:
                continue
            
            r_expected = np.sqrt(q)
            phi_expected = self.normalize_angle(np.arctan2(delta_y, delta_x) - theta)
            
            # Expected measurement
            z_expected = np.array([[r_expected], [phi_expected]])
            z_actual = np.array([[r_meas], [phi_meas]])
            
            # Innovation
            innovation = z_actual - z_expected
            innovation[1, 0] = self.normalize_angle(innovation[1, 0])
            
            # Measurement Jacobian H_t (EXACT)
            H_low = np.array([
                [-delta_x/r_expected, -delta_y/r_expected, 0, delta_x/r_expected, delta_y/r_expected],
                [delta_y/q, -delta_x/q, -1, -delta_y/q, delta_x/q]
            ])
            
            H_t = np.zeros((2, mu.shape[0]))
            H_t[:, 0:3] = H_low[:, 0:3]
            H_t[:, lm_idx_start:lm_idx_start+2] = H_low[:, 3:5]
            
            # Innovation covariance
            S = H_t @ sigma @ H_t.T + self.Q
            
            # Kalman gain
            try:
                K = sigma @ H_t.T @ np.linalg.inv(S)
            except np.linalg.LinAlgError:
                continue
            
            # State update
            mu = mu + K @ innovation
            mu[2, 0] = self.normalize_angle(mu[2, 0])
            
            # Covariance update using Joseph form (EXACT)
            I_KH = np.eye(mu.shape[0]) - K @ H_t
            sigma = I_KH @ sigma @ I_KH.T + K @ self.Q @ K.T
            
            # Ensure symmetry
            sigma = (sigma + sigma.T) / 2
        
        self.mu = mu
        self.sigma = sigma
    
    def odom_callback(self, msg):
        """Process odometry (EXACT from Pygame)"""
        current_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        
        if self.last_time is None:
            self.last_time = current_time
            return
        
        dt = current_time - self.last_time
        
        if dt > 0 and dt < 0.5:
            # Extract velocities
            v = msg.twist.twist.linear.x
            w = msg.twist.twist.angular.z
            
            # Store for observability check
            self.last_v = v
            self.last_w = w
            
            # EKF Prediction
            self.mu, self.sigma = self.prediction_step(v, w, dt)
        
        self.last_time = current_time
    
    def scan_callback(self, msg):
        """Process LiDAR scan (EXACT from Pygame)"""
        self.update_occupancy_grid(msg) 
        detections = self.extract_landmark_detections(msg)
        
        if detections:
            self.measurement_update(detections)
    
    def publish_state(self):
        """Publish EKF state and markers"""
        now = self.get_clock().now().to_msg()
        
        # Publish odometry
        odom = Odometry()
        odom.header.stamp = now
        odom.header.frame_id = 'map'
        odom.child_frame_id = 'base_link_ekf'
        
        odom.pose.pose.position.x = self.mu[0, 0]
        odom.pose.pose.position.y = self.mu[1, 0]
        odom.pose.pose.position.z = 0.0
        
        q = tf_transformations.quaternion_from_euler(0, 0, self.mu[2, 0])
        odom.pose.pose.orientation.x = q[0]
        odom.pose.pose.orientation.y = q[1]
        odom.pose.pose.orientation.z = q[2]
        odom.pose.pose.orientation.w = q[3]
        
        self.ekf_pub.publish(odom)
        
        # Publish TF
        t = TransformStamped()
        t.header.stamp = now
        t.header.frame_id = 'map'
        t.child_frame_id = 'base_link_ekf'
        t.transform.translation.x = self.mu[0, 0]
        t.transform.translation.y = self.mu[1, 0]
        t.transform.translation.z = 0.0
        t.transform.rotation = odom.pose.pose.orientation
        self.tf_broadcaster.sendTransform(t)
        
        # Publish markers
        self.publish_markers(now)
    
    def publish_markers(self, stamp):
        """Publish landmark markers with uncertainty"""
        markers = MarkerArray()
        
        for i in range(self.N_LANDMARKS):
            idx = self.N_STATE + 2*i
            
            if np.isnan(self.mu[idx, 0]):
                continue
            
            lm_x = self.mu[idx, 0]
            lm_y = self.mu[idx+1, 0]
            
            # Bounds check
            if lm_x < -0.5 or lm_x > 10.5 or lm_y < -0.5 or lm_y > 8.5:
                continue
            
            # Get covariance
            lm_cov = self.sigma[idx:idx+2, idx:idx+2]
            
            try:
                eigvals = np.linalg.eigvalsh(lm_cov)
                eigvals = np.maximum(eigvals, 1e-9)
                max_eigval = max(eigvals)
                
                # Only show if converged (relaxed threshold)
                if max_eigval > 1.0 or max_eigval < 1e-6:
                    continue
                
                eigvecs = np.linalg.eigh(lm_cov)[1]
                order = np.argsort(eigvals)[::-1]
                eigvals = eigvals[order]
                eigvecs = eigvecs[:, order]
                
            except:
                continue
            
            # Landmark sphere
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = stamp
            m.ns = 'landmarks'
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = lm_x
            m.pose.position.y = lm_y
            m.pose.position.z = 0.3
            m.scale.x = 0.3
            m.scale.y = 0.3
            m.scale.z = 0.3
            m.color.a = 1.0
            m.color.r = 1.0
            m.color.g = 0.5
            m.color.b = 0.0
            markers.markers.append(m)
            
            # Uncertainty ellipse (3-sigma like Pygame)
            scale = 3.0  # 3-sigma
            eigvals_scaled = np.sqrt(eigvals) * scale
            angle = math.atan2(eigvecs[1,0], eigvecs[0,0])
            
            e = Marker()
            e.header.frame_id = 'map'
            e.header.stamp = stamp
            e.ns = 'ellipses'
            e.id = i + 100
            e.type = Marker.CYLINDER
            e.action = Marker.ADD
            e.pose.position.x = lm_x
            e.pose.position.y = lm_y
            e.pose.position.z = 0.01
            
            qe = tf_transformations.quaternion_from_euler(0, 0, angle)
            e.pose.orientation.x = qe[0]
            e.pose.orientation.y = qe[1]
            e.pose.orientation.z = qe[2]
            e.pose.orientation.w = qe[3]
            
            e.scale.x = max(eigvals_scaled[0], 0.05)
            e.scale.y = max(eigvals_scaled[1], 0.05)
            e.scale.z = 0.02
            e.color.a = 0.3
            e.color.r = 1.0
            e.color.g = 0.0
            e.color.b = 0.0
            markers.markers.append(e)
            
            # Label
            txt = Marker()
            txt.header.frame_id = 'map'
            txt.header.stamp = stamp
            txt.ns = 'labels'
            txt.id = i + 200
            txt.type = Marker.TEXT_VIEW_FACING
            txt.action = Marker.ADD
            txt.pose.position.x = lm_x
            txt.pose.position.y = lm_y
            txt.pose.position.z = 0.7
            txt.scale.z = 0.3
            txt.color.a = 1.0
            txt.color.r = 1.0
            txt.color.g = 1.0
            txt.color.b = 1.0
            txt.text = chr(65 + i)
            markers.markers.append(txt)
        
        self.marker_pub.publish(markers)
    
    def update_occupancy_grid(self, scan):
        """Update occupancy grid from LiDAR"""
        rx, ry = self.mu[0, 0], self.mu[1, 0]
        
        ranges = np.array(scan.ranges)
        angle_min = scan.angle_min
        angle_inc = scan.angle_increment
        
        for i in range(0, len(ranges), 5):
            r = ranges[i]
            if scan.range_min < r < scan.range_max:
                angle = angle_min + i * angle_inc
                global_x = rx + r * math.cos(self.mu[2,0] + angle)
                global_y = ry + r * math.sin(self.mu[2,0] + angle)
                
                gx = int(global_x / self.grid_resolution)
                gy = int(global_y / self.grid_resolution)
                
                if 0 <= gx < self.grid_width and 0 <= gy < self.grid_height:
                    self.grid_data[gy, gx] = min(100, self.grid_data[gy, gx] + 5)

    def publish_map(self):
        """Publish occupancy grid"""
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.info.resolution = self.grid_resolution
        msg.info.width = self.grid_width
        msg.info.height = self.grid_height
        msg.info.origin = Pose()
        msg.info.origin.position.x = 0.0
        msg.info.origin.position.y = 0.0
        msg.info.origin.orientation.w = 1.0
        msg.data = self.grid_data.flatten().tolist()
        self.map_pub.publish(msg)




def main(args=None):
    rclpy.init(args=args)
    node = SLAMNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
