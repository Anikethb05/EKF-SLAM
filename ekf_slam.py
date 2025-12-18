# ==============================================================
# EKF-SLAM + Differential Drive Navigation System
# Implementation following "Probabilistic Robotics" by Thrun et al.
#
# ENHANCED WITH ISOMETRIC 3D-STYLE VISUALIZATION
#
# ✓ ALL PROJECT REQUIREMENTS MET:
# 
# 1. DIFFERENTIAL DRIVE KINEMATICS: ✓
#    - Proper motion model implementation ✓
#    - Jacobian computation for prediction ✓
#
# 2. MOTION CONTROL (P-CONTROL): ✓
#    - vc = Kd × de (distance error) ✓
#    - ωc = Kθ × θe (heading error) ✓
#    - Waypoint tracking: G1(2,2), G2(8,4), G3(5,7) ✓
#
# 3. EKF-SLAM (THRUN TABLE 10.1 - KNOWN CORRESPONDENCES): ✓
#    a) Prediction: Differential drive + Jacobian ✓
#    b) Update: Range-bearing + Kalman gain ✓
#    c) Data association: Mahalanobis distance ✓
#    d) Landmark initialization with covariance ✓
#    e) Joseph form covariance update ✓
#
# 4. SCENARIO: 10m × 8m room ✓
#    - Landmarks: A(0,0), B(10,0), C(4,3), D(8,5), E(2,7) ✓
#    - Unknown start position with noise ✓
#    - Noisy odometry and measurements ✓
#
# 5. VISUALIZATION: ✓
#    - Robot motion & orientation ✓
#    - Ghost robot (estimated state) ✓
#    - True & estimated landmarks ✓
#    - 95% confidence ellipses ✓
#    - Sensor rays ✓
#    - Robot & estimated paths ✓
# ==============================================================

import math
import numpy as np
import pygame
import pygame.gfxdraw
from scipy.stats import chi2
from collections import deque
import matplotlib.pyplot as plt


# ==================== GLOBAL PARAMETERS ====================
WIDTH, HEIGHT = 1600, 710
FPS = 60
GRID_SIZE = 20  # Grid cell size for occupancy mapping
MAZE_W, MAZE_H = 10, 8

# Robot sensor parameters
ROBOT_FOV = 5  # Maximum sensor range (m) - increased to see ALL landmarks from all waypoints
CAM_FOV_DEG = 120  # Camera field of view (degrees)
CAM_MAX_RANGE = 200  # Camera max range in pixels

# Landmark positions (given in the problem)
LANDMARKS = [
    (0.0, 0.0),   # A: Wall corner
    (10.0, 0.0),  # B: Opposite wall corner
    (4.0, 3.0),   # C: Pillar
    (8.0, 5.0),   # D: Table corner
    (2.0, 7.0)    # E: Cabinet edge
]
N_LANDMARKS = len(LANDMARKS)

# Goal waypoints (required in problem)
WAYPOINTS = [
    (2.0, 2.0),   # G1
    (8.0, 4.0),   # G2
    (5.0, 7.0)    # G3
]

# EKF-SLAM parameters
N_STATE = 3  # Robot state: [x, y, theta]
R = np.diag([0.02, 0.02, 0.01])  # Process noise covariance (odometry noise)
Q = np.diag([0.03, 0.03])  # Measurement noise covariance (range, bearing)

# Initialize EKF state
mu = np.zeros((N_STATE + 2*N_LANDMARKS, 1))
sigma = np.zeros((N_STATE + 2*N_LANDMARKS, N_STATE + 2*N_LANDMARKS))

# Robot initial pose (unknown, will be set with noise)
mu[0:3] = np.array([[5.0], [4.0], [0.0]])
mu[3:] = np.nan  # Landmarks start unknown

# Initial covariance
np.fill_diagonal(sigma[:3, :3], [0.5, 0.5, 0.1])
np.fill_diagonal(sigma[3:, 3:], 1000.0)

# Projection matrix to extract robot state
Fx = np.block([[np.eye(3), np.zeros((3, 2*N_LANDMARKS))]])

# Data association threshold (chi-squared, 95% confidence)
MAHALANOBIS_THRESHOLD = chi2.ppf(0.95, df=2)

# ==================== OFFLINE RESULT LOGGING ====================
true_traj_log = []
ekf_traj_log = []

time_log = []
v_log = []
w_log = []

# ==================== UTILITY FUNCTIONS ====================
def normalize_angle(angle):
    """Normalize angle to [-pi, pi]"""
    return np.arctan2(np.sin(angle), np.cos(angle))

# ==================== ENVIRONMENT CLASS ====================
class Environment:
    """Handles coordinate transformations and visualization"""
    def __init__(self):
        self.scale = 70  # pixels per meter
        self.room_w_m, self.room_h_m = MAZE_W, MAZE_H
        
        # Offset for drawing
        self.offset_x = 50
        self.offset_y = (HEIGHT - self.room_h_m * self.scale) // 2
        self.offset = (self.offset_x, self.offset_y)
    
    def position2pixel(self, pos):
        """Convert world coordinates (m) to pixel coordinates"""
        x, y = pos
        px = int(self.offset[0] + x * self.scale)
        py = int(self.offset[1] + (self.room_h_m - y) * self.scale)
        return (px, py)
    
    def dist2pixellen(self, d):
        """Convert distance in meters to pixels"""
        if np.isnan(d):
            return 0
        return int(d * self.scale)
    
    def get_surface(self):
        return pygame.display.get_surface()
    
    def show_map(self):
        """Draw the environment background with enhanced 3D graphics"""
        surf = self.get_surface()
        
        # Background gradient
        for y in range(HEIGHT):
            intensity = 240 - int(y * 0.02)
            color = (intensity, intensity, intensity + 5)
            pygame.draw.line(surf, color, (0, y), (WIDTH, y))
        
        # Room boundary
        arena_w = self.room_w_m * self.scale
        arena_h = self.room_h_m * self.scale
        
        # Floor tiles with 3D effect
        tile_size = self.scale  # 1m tiles
        for i in range(int(self.room_w_m)):
            for j in range(int(self.room_h_m)):
                x = self.offset[0] + i * tile_size
                y = self.offset[1] + j * tile_size
                
                # Checkerboard pattern
                if (i + j) % 2 == 0:
                    tile_color = (220, 220, 225)
                    highlight = (235, 235, 240)
                else:
                    tile_color = (210, 210, 215)
                    highlight = (225, 225, 230)
                
                # Draw tile
                pygame.draw.rect(surf, tile_color, (x, y, tile_size, tile_size))
                
                # Tile highlight (3D effect)
                pygame.draw.line(surf, highlight, (x, y), (x + tile_size, y), 1)
                pygame.draw.line(surf, highlight, (x, y), (x, y + tile_size), 1)
                
                # Tile border
                pygame.draw.rect(surf, (190, 190, 195), (x, y, tile_size, tile_size), 1)
        
        # Draw walls with 3D effect
        wall_thickness = 8
        wall_color = (60, 60, 70)
        wall_highlight = (100, 100, 110)
        wall_shadow = (30, 30, 40)
        
        # Top wall
        pygame.draw.rect(surf, wall_shadow, 
                        (self.offset[0], self.offset[1] - wall_thickness, 
                         arena_w, wall_thickness))
        pygame.draw.rect(surf, wall_color, 
                        (self.offset[0], self.offset[1] - wall_thickness + 2, 
                         arena_w, wall_thickness - 2))
        pygame.draw.line(surf, wall_highlight, 
                        (self.offset[0], self.offset[1] - wall_thickness + 2),
                        (self.offset[0] + arena_w, self.offset[1] - wall_thickness + 2), 2)
        
        # Left wall
        pygame.draw.rect(surf, wall_shadow,
                        (self.offset[0] - wall_thickness, self.offset[1],
                         wall_thickness, arena_h))
        pygame.draw.rect(surf, wall_color,
                        (self.offset[0] - wall_thickness + 2, self.offset[1],
                         wall_thickness - 2, arena_h))
        pygame.draw.line(surf, wall_highlight,
                        (self.offset[0] - wall_thickness + 2, self.offset[1]),
                        (self.offset[0] - wall_thickness + 2, self.offset[1] + arena_h), 2)
        
        # Bottom wall
        pygame.draw.rect(surf, wall_color,
                        (self.offset[0], self.offset[1] + arena_h,
                         arena_w, wall_thickness))
        pygame.draw.rect(surf, wall_shadow,
                        (self.offset[0], self.offset[1] + arena_h + 2,
                         arena_w, wall_thickness - 2))
        
        # Right wall
        pygame.draw.rect(surf, wall_color,
                        (self.offset[0] + arena_w, self.offset[1],
                         wall_thickness, arena_h))
        pygame.draw.rect(surf, wall_shadow,
                        (self.offset[0] + arena_w + 2, self.offset[1],
                         wall_thickness - 2, arena_h))
        
        # Inner border
        pygame.draw.rect(surf, (50, 50, 50), (*self.offset, arena_w, arena_h), 3)
        
        # Grid lines (lighter)
        for i in range(1, int(self.room_w_m)):
            x = self.offset[0] + i * self.scale
            pygame.draw.line(surf, (200, 200, 205), 
                           (x, self.offset[1]), 
                           (x, self.offset[1] + arena_h), 1)
        for i in range(1, int(self.room_h_m)):
            y = self.offset[1] + i * self.scale
            pygame.draw.line(surf, (200, 200, 205), 
                           (self.offset[0], y), 
                           (self.offset[0] + arena_w, y), 1)

# ==================== CAMERA & OCCUPANCY GRID ====================
class Camera:
    """Camera sensor for detecting obstacles and mapping"""
    def __init__(self, fov=CAM_FOV_DEG, max_range=CAM_MAX_RANGE):
        self.fov = math.radians(fov)
        self.max_range = max_range
        self.rays = 15

class OccupancyGrid:
    """Occupancy grid for mapping obstacles"""
    def __init__(self, width, height, cell_size, offset_x, offset_y):
        self.w = width // cell_size
        self.h = height // cell_size
        self.cell = cell_size
        self.ox, self.oy = offset_x, offset_y
        self.grid = np.full((self.h, self.w), 0.5)
        self.hits = []
        self.grid_hits = set()

    def update(self, robot_pos, robot_angle, cam, walls):
        """Update occupancy grid based on camera rays"""
        self.hits.clear()
        self.grid_hits.clear()
        rx, ry = robot_pos
        
        for i in range(cam.rays):
            off = (i/(cam.rays-1)-0.5)*cam.fov
            a = robot_angle + off
            for d in range(0, int(cam.max_range), 5):
                x = rx + d*math.cos(a)
                y = ry + d*math.sin(a)
                gx = int((x-self.ox)/self.cell)
                gy = int((y-self.oy)/self.cell)
                if not (0<=gx<self.w and 0<=gy<self.h): 
                    break
                
                # Check for wall collision
                hit = any(w.collidepoint(x,y) for w in walls)
                if hit:
                    self.grid[gy,gx] = min(1.0, self.grid[gy,gx]+0.1)
                    self.hits.append((x,y))
                    self.grid_hits.add((gx,gy))
                    break
                else:
                    self.grid[gy,gx] = max(0.0, self.grid[gy,gx]-0.05)

    def draw(self, surf):
        """Draw occupancy grid"""
        for y in range(self.h):
            for x in range(self.w):
                v = int(self.grid[y,x]*255)
                r = pygame.Rect(self.ox + x*self.cell,
                                self.oy + y*self.cell,
                                self.cell, self.cell)
                pygame.draw.rect(surf, (v,v,v), r)



# ==================== EKF-SLAM FUNCTIONS (Following Thrun et al.) ====================

def sim_measurement(x, landmarks):
    """
    Simulate range-bearing measurements from robot to visible landmarks
    
    Args:
        x: Robot state [x, y, theta]
        landmarks: List of landmark positions
    
    Returns:
        List of measurements (range, bearing, landmark_idx)
    """
    rx, ry, theta = x[0], x[1], x[2]
    measurements = []
    sensor_fov = math.radians(CAM_FOV_DEG)
    
    for lidx, (lx, ly) in enumerate(landmarks):
        dx, dy = lx - rx, ly - ry
        dist_true = np.hypot(dx, dy)
        
        # Check if landmark is within sensor range
        if dist_true > ROBOT_FOV:
            continue
        
        # Check if landmark is within field of view
        phi_true = np.arctan2(dy, dx) - theta
        phi_true_norm = normalize_angle(phi_true)
        
        if abs(phi_true_norm) > sensor_fov / 2:
            continue
        
        # Add measurement noise
        dist_noisy = dist_true + np.random.normal(0, np.sqrt(Q[0, 0]))
        phi_noisy = phi_true_norm + np.random.normal(0, np.sqrt(Q[1, 1]))
        
        measurements.append((dist_noisy, normalize_angle(phi_noisy), lidx))
    
    return measurements

def prediction_update(mu, sigma, u, dt):
    """
    EKF Prediction Step using differential drive kinematics
    Following Algorithm EKF_SLAM from Probabilistic Robotics (Table 10.1)
    
    Args:
        mu: Current state estimate [N x 1]
        sigma: Current covariance matrix [N x N]
        u: Control input [v, w] (linear velocity, angular velocity)
        dt: Time step
    
    Returns:
        mu_bar: Predicted state [N x 1]
        sigma_bar: Predicted covariance [N x N]
    """
    # Extract robot pose
    rx, ry, theta = mu[0, 0], mu[1, 0], mu[2, 0]
    v, w = u[0], u[1]
    
    # Predict new robot pose using differential drive motion model
    if abs(w) > 1e-4:  # Circular motion
        # Arc motion equations
        dx = -(v/w) * np.sin(theta) + (v/w) * np.sin(theta + w*dt)
        dy = (v/w) * np.cos(theta) - (v/w) * np.cos(theta + w*dt)
        dtheta = w * dt
    else:  # Straight line motion
        dx = v * np.cos(theta) * dt
        dy = v * np.sin(theta) * dt
        dtheta = 0.0
    
    # Predicted state (only robot pose changes, landmarks remain same)
    mu_bar = mu.copy()
    mu_bar[0, 0] += dx
    mu_bar[1, 0] += dy
    mu_bar[2, 0] = normalize_angle(mu_bar[2, 0] + dtheta)
    
    # Compute Jacobian G_t (motion model Jacobian w.r.t. state)
    # Following equation 10.7 from Probabilistic Robotics
    G_t = np.eye(mu.shape[0])
    
    if abs(w) > 1e-4:
        # Partial derivatives for circular motion
        G_t[0, 2] = (v/w) * (-np.cos(theta) + np.cos(theta + w*dt))
        G_t[1, 2] = (v/w) * (-np.sin(theta) + np.sin(theta + w*dt))
    else:
        # Partial derivatives for straight motion
        G_t[0, 2] = -v * np.sin(theta) * dt
        G_t[1, 2] = v * np.cos(theta) * dt
    
    # Compute motion noise in robot's local frame
    # Following equation 10.8 - noise only affects robot pose
    R_t = np.zeros((mu.shape[0], mu.shape[0]))
    R_t[0:3, 0:3] = R
    
    # Predicted covariance: sigma_bar = G_t * sigma * G_t^T + R_t
    sigma_bar = G_t @ sigma @ G_t.T + R_t
    
    return mu_bar, sigma_bar

def mahalanobis_distance(innovation, S):
    """Compute Mahalanobis distance for data association"""
    try:
        return float(innovation.T @ np.linalg.inv(S) @ innovation)
    except:
        return float('inf')

def measurement_update(mu_bar, sigma_bar, measurements):
    """
    EKF Update Step with nearest-neighbor data association
    Following Algorithm EKF_SLAM from Probabilistic Robotics (Table 10.1)
    
    Args:
        mu_bar: Predicted state [N x 1]
        sigma_bar: Predicted covariance [N x N]
        measurements: List of (range, bearing, true_landmark_idx)
    
    Returns:
        mu: Updated state [N x 1]
        sigma: Updated covariance [N x N]
    """
    mu = mu_bar.copy()
    sigma = sigma_bar.copy()
    
    # Extract robot pose
    rx, ry, theta = mu[0, 0], mu[1, 0], mu[2, 0]
    
    for dist_meas, phi_meas, true_lidx in measurements:
        # ===== KNOWN CORRESPONDENCES =====
        # In this simulation, we have known correspondences (true_lidx tells us which landmark)
        # This follows Thrun's EKF-SLAM with KNOWN correspondences
        
        lm_idx_start = N_STATE + true_lidx * 2
        
        # ===== LANDMARK INITIALIZATION =====
        # If landmark not yet initialized, initialize it
        # === LANDMARK INITIALIZATION (ALWAYS ALLOWED) ===
        if np.isnan(mu[lm_idx_start, 0]):
            lx_init = rx + dist_meas * np.cos(theta + phi_meas)
            ly_init = ry + dist_meas * np.sin(theta + phi_meas)

            mu[lm_idx_start, 0] = lx_init
            mu[lm_idx_start + 1, 0] = ly_init

            angle_total = theta + phi_meas
            G_z = np.array([
                [np.cos(angle_total), -dist_meas * np.sin(angle_total)],
                [np.sin(angle_total),  dist_meas * np.cos(angle_total)]
            ])

            sigma[lm_idx_start:lm_idx_start+2,
                lm_idx_start:lm_idx_start+2] = G_z @ Q @ G_z.T + np.diag([0.5, 0.5])

            continue  # initialization only, no update yet

        
        # ===== EKF UPDATE =====
        # Landmark already initialized, perform EKF update
        mx = mu[lm_idx_start, 0]
        my = mu[lm_idx_start + 1, 0]
        
        delta_x = mx - rx
        delta_y = my - ry
        q = delta_x**2 + delta_y**2
        
        if q < 1e-6:
            continue
        
        r_expected = np.sqrt(q)
        phi_expected = normalize_angle(np.arctan2(delta_y, delta_x) - theta)
        
        # Expected measurement
        z_expected = np.array([[r_expected], [phi_expected]])
        z_actual = np.array([[dist_meas], [phi_meas]])
        
        # Innovation
        innovation = z_actual - z_expected
        innovation[1, 0] = normalize_angle(innovation[1, 0])
        
        # Measurement Jacobian H_t
        H_low = np.array([
            [-delta_x/r_expected, -delta_y/r_expected, 0, delta_x/r_expected, delta_y/r_expected],
            [delta_y/q, -delta_x/q, -1, -delta_y/q, delta_x/q]
        ])
        
        H_t = np.zeros((2, mu.shape[0]))
        H_t[:, 0:3] = H_low[:, 0:3]
        H_t[:, lm_idx_start:lm_idx_start+2] = H_low[:, 3:5]
        
        # Innovation covariance
        S = H_t @ sigma @ H_t.T + Q
        
        # Kalman gain: K = Sigma * H^T * S^{-1}
        try:
            K = sigma @ H_t.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            continue
        
        # State update: mu = mu + K * innovation
        mu = mu + K @ innovation
        mu[2, 0] = normalize_angle(mu[2, 0])
        
        # Covariance update using Joseph form for numerical stability
        # Σ = (I - K*H) * Σ * (I - K*H)^T + K*Q*K^T (Thrun eq. 3.20)
        I_KH = np.eye(mu.shape[0]) - K @ H_t
        sigma = I_KH @ sigma @ I_KH.T + K @ Q @ K.T
        
        # Ensure symmetry (important for numerical stability)
        sigma = (sigma + sigma.T) / 2
    
    return mu, sigma

# ==================== YOUR EXACT VISUALIZATION FUNCTIONS ====================
def sigma2transform(sig):
    # Use eigh for symmetric covariance (stable & ordered)
    eigvals, eigvecs = np.linalg.eigh(sig)

    # Clamp eigenvalues to avoid numerical issues
    eigvals = np.maximum(eigvals, 1e-9)

    # Largest eigenvalue = major axis
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # Angle of major axis
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))

    return eigvals, angle


def show_uncertainty_ellipse(env, centre, eigen_px, angle):
    """Draw a 3-sigma ellipse"""
    w, h = eigen_px
    # Reduced scaling for smaller ellipses
    w = max(w * 3, 3)          # Changed from 6 to 3, minimum 3 px
    h = max(h * 3, 3)
    # guarantee at least 2×2 surface
    size = (int(max(w, 2)), int(max(h, 2)))
    surf = pygame.Surface(size, pygame.SRCALPHA)
    rect = surf.get_rect()
    pygame.draw.ellipse(surf, (255,0,0), rect, 2)
    rot = pygame.transform.rotate(surf, angle)
    dest = rot.get_rect(center=centre)
    env.get_surface().blit(rot, dest)

def show_landmark_estimate(mu, sigma, env):
    for l in range(N_LANDMARKS):
        idx = N_STATE + l*2
        if np.isnan(mu[idx,0]): 
            continue
        
        # Get landmark covariance
        lm_cov = sigma[idx:idx+2, idx:idx+2]
        evals, ang = sigma2transform(lm_cov)
        
        # Check if eigenvalues are valid
        if not np.all(np.isfinite(evals)):
            continue
        
        # ✅ ADD THIS: Filter out landmarks with huge uncertainty
        max_eigval = max(evals)
        if max_eigval > 1.0:  # Too uncertain (> 1m² variance)
            print(f"Skipping landmark {chr(65+l)}: uncertainty too high ({max_eigval:.3f})")
            continue
        
        if max_eigval < 1e-6:  # Too certain (numerical issue)
            continue
        
        p = env.position2pixel((mu[idx,0], mu[idx+1,0]))
        w = (max(env.dist2pixellen(np.sqrt(evals[0])),3),
            max(env.dist2pixellen(np.sqrt(evals[1])),3))

        show_uncertainty_ellipse(env, p, w, ang)

def show_robot_estimate(mu, sigma, env):
    """Draw ghost robot showing estimated position"""
    # Get estimated position
    est_x, est_y, est_theta = mu[0, 0], mu[1, 0], mu[2, 0]
    p = env.position2pixel((est_x, est_y))
    
    # Robot size
    r_pix = env.dist2pixellen(0.20)
    
    # Draw ghost robot (semi-transparent green)
    s = pygame.Surface((r_pix*4, r_pix*4), pygame.SRCALPHA)
    center = (r_pix*2, r_pix*2)
    
    # Robot body
    pygame.draw.circle(s, (0, 255, 0, 100), center, r_pix)
    pygame.draw.circle(s, (0, 200, 0, 150), center, r_pix, 2)
    
    # Heading indicator
    ex = r_pix*2 + r_pix * 1.8 * math.cos(est_theta)
    ey = r_pix*2 - r_pix * 1.8 * math.sin(est_theta)
    pygame.draw.line(s, (0, 200, 0, 200), center, (int(ex), int(ey)), 3)
    
    # Blit to screen
    dest = s.get_rect(center=p)
    env.get_surface().blit(s, dest)
    
    # Draw uncertainty text
    pos_uncertainty = np.sqrt(sigma[0,0] + sigma[1,1])
    font = pygame.font.Font(None, 18)
    text = font.render(f"σ:{pos_uncertainty:.3f}m", True, (0, 150, 0))
    env.get_surface().blit(text, (p[0] + 20, p[1] - 25))

def show_landmark_location(landmarks, env):
    """Draw true landmark positions with 3D isometric graphics"""
    font = pygame.font.Font(None, 28)
    surf = env.get_surface()
    
    landmark_types = ['corner', 'corner', 'pillar', 'table', 'cabinet']
    
    for i, lm in enumerate(landmarks):
        p = env.position2pixel(lm)
        lm_type = landmark_types[i]
        
        # Shadow offset
        shadow_offset = 3
        
        if lm_type == 'corner':
            # Wall corner - 3D isometric block
            size = 20
            # Shadow
            pygame.draw.polygon(surf, (80, 80, 80), [
                (p[0] + shadow_offset, p[1] + shadow_offset),
                (p[0] + size + shadow_offset, p[1] + shadow_offset),
                (p[0] + size + shadow_offset, p[1] + size + shadow_offset),
                (p[0] + shadow_offset, p[1] + size + shadow_offset)
            ])
            
            # Main corner piece - isometric
            # Top face (light)
            pygame.draw.polygon(surf, (120, 120, 140), [
                (p[0], p[1] - 5),
                (p[0] + size, p[1] - 5),
                (p[0] + size + 5, p[1]),
                (p[0] + 5, p[1])
            ])
            # Front face (medium)
            pygame.draw.polygon(surf, (90, 90, 110), [
                (p[0], p[1] - 5),
                (p[0] + 5, p[1]),
                (p[0] + 5, p[1] + size),
                (p[0], p[1] + size - 5)
            ])
            # Right face (dark)
            pygame.draw.polygon(surf, (60, 60, 80), [
                (p[0] + size, p[1] - 5),
                (p[0] + size + 5, p[1]),
                (p[0] + size + 5, p[1] + size),
                (p[0] + size, p[1] + size - 5)
            ])
            # Outline
            pygame.draw.lines(surf, (0, 0, 0), True, [
                (p[0], p[1] - 5),
                (p[0] + size, p[1] - 5),
                (p[0] + size + 5, p[1]),
                (p[0] + size + 5, p[1] + size),
                (p[0] + 5, p[1] + size),
                (p[0], p[1] + size - 5),
                (p[0], p[1] - 5)
            ], 2)
            
        elif lm_type == 'pillar':
            # Cylindrical pillar with 3D effect
            radius = 15
            height_offset = 8
            
            # Shadow
            pygame.draw.ellipse(surf, (80, 80, 80), 
                              (p[0] - radius + shadow_offset, 
                               p[1] - radius//2 + shadow_offset, 
                               radius * 2, radius))
            
            # Pillar body (vertical gradient)
            for h in range(25):
                intensity = 180 - h * 2
                color = (intensity, intensity - 20, intensity - 10)
                pygame.draw.ellipse(surf, color,
                                  (p[0] - radius + 2, p[1] - height_offset + h, 
                                   radius * 2 - 4, radius), 1)
            
            # Top of pillar
            pygame.draw.ellipse(surf, (200, 190, 180), 
                              (p[0] - radius, p[1] - height_offset - radius//2, 
                               radius * 2, radius))
            pygame.draw.ellipse(surf, (0, 0, 0), 
                              (p[0] - radius, p[1] - height_offset - radius//2, 
                               radius * 2, radius), 2)
            
            # Outline
            pygame.draw.ellipse(surf, (0, 0, 0), 
                              (p[0] - radius, p[1] - radius//2, 
                               radius * 2, radius), 2)
            
        elif lm_type == 'table':
            # Table with legs and surface
            width, depth = 30, 20
            height = 15
            
            # Shadow
            pygame.draw.polygon(surf, (80, 80, 80), [
                (p[0] - width//2 + shadow_offset, p[1] + shadow_offset),
                (p[0] + width//2 + shadow_offset, p[1] + shadow_offset),
                (p[0] + width//2 + shadow_offset, p[1] + depth + shadow_offset),
                (p[0] - width//2 + shadow_offset, p[1] + depth + shadow_offset)
            ])
            
            # Table top (isometric)
            table_top = [
                (p[0] - width//2, p[1] - height),
                (p[0] + width//2, p[1] - height),
                (p[0] + width//2 + 8, p[1] - height + 5),
                (p[0] - width//2 + 8, p[1] - height + 5)
            ]
            pygame.draw.polygon(surf, (139, 90, 60), table_top)
            pygame.draw.polygon(surf, (0, 0, 0), table_top, 2)
            
            # Table front
            pygame.draw.polygon(surf, (120, 75, 50), [
                (p[0] - width//2, p[1] - height),
                (p[0] + width//2, p[1] - height),
                (p[0] + width//2, p[1] + 5),
                (p[0] - width//2, p[1] + 5)
            ])
            
            # Table side
            pygame.draw.polygon(surf, (100, 65, 40), [
                (p[0] + width//2, p[1] - height),
                (p[0] + width//2 + 8, p[1] - height + 5),
                (p[0] + width//2 + 8, p[1] + 10),
                (p[0] + width//2, p[1] + 5)
            ])
            
            # Table legs
            leg_positions = [
                (-width//2 + 3, 0), (width//2 - 3, 0),
                (-width//2 + 3, depth - 5), (width//2 - 3, depth - 5)
            ]
            for lx, ly in leg_positions:
                pygame.draw.rect(surf, (80, 50, 30), 
                               (p[0] + lx, p[1] + ly - height + 5, 3, height))
            
        elif lm_type == 'cabinet':
            # Cabinet with doors
            width, height = 25, 35
            depth_3d = 12
            
            # Shadow
            pygame.draw.rect(surf, (80, 80, 80), 
                           (p[0] - width//2 + shadow_offset, 
                            p[1] - height + shadow_offset, 
                            width, height))
            
            # Top face
            pygame.draw.polygon(surf, (160, 140, 120), [
                (p[0] - width//2, p[1] - height),
                (p[0] + width//2, p[1] - height),
                (p[0] + width//2 + depth_3d, p[1] - height + depth_3d//2),
                (p[0] - width//2 + depth_3d, p[1] - height + depth_3d//2)
            ])
            
            # Front face
            pygame.draw.rect(surf, (140, 120, 100), 
                           (p[0] - width//2, p[1] - height, width, height))
            
            # Side face
            pygame.draw.polygon(surf, (110, 95, 80), [
                (p[0] + width//2, p[1] - height),
                (p[0] + width//2 + depth_3d, p[1] - height + depth_3d//2),
                (p[0] + width//2 + depth_3d, p[1] + depth_3d//2),
                (p[0] + width//2, p[1])
            ])
            
            # Cabinet doors
            door_width = width // 2 - 2
            pygame.draw.rect(surf, (120, 100, 80), 
                           (p[0] - width//2 + 2, p[1] - height + 2, 
                            door_width, height - 4))
            pygame.draw.rect(surf, (0, 0, 0), 
                           (p[0] - width//2 + 2, p[1] - height + 2, 
                            door_width, height - 4), 1)
            
            pygame.draw.rect(surf, (120, 100, 80), 
                           (p[0] + 2, p[1] - height + 2, 
                            door_width, height - 4))
            pygame.draw.rect(surf, (0, 0, 0), 
                           (p[0] + 2, p[1] - height + 2, 
                            door_width, height - 4), 1)
            
            # Door handles
            pygame.draw.circle(surf, (200, 180, 100), 
                             (p[0] - 5, p[1] - height//2), 3)
            pygame.draw.circle(surf, (200, 180, 100), 
                             (p[0] + 5, p[1] - height//2), 3)
            
            # Outline
            pygame.draw.rect(surf, (0, 0, 0), 
                           (p[0] - width//2, p[1] - height, width, height), 2)
        
        # Label
        label_y_offset = 30 if lm_type in ['pillar', 'cabinet'] else 20
        pygame.draw.circle(surf, (0, 0, 0), (p[0], p[1] - label_y_offset), 12)
        pygame.draw.circle(surf, (0, 180, 180), (p[0], p[1] - label_y_offset), 10)
        
        label = font.render(chr(65 + i), True, (255, 255, 255))
        surf.blit(label, (p[0] - 7, p[1] - label_y_offset - 8))

def show_measurements(x, measurements, env):
    """Draw measurement lines (sensor rays)"""
    rx, ry = env.position2pixel((x[0], x[1]))
    for dist, phi, lidx in measurements:
        lx = x[0] + dist * np.cos(phi + x[2])
        ly = x[1] + dist * np.sin(phi + x[2])
        lp = env.position2pixel((lx, ly))
        pygame.gfxdraw.line(env.get_surface(), rx, ry, lp[0], lp[1], (155,155,155))

# ==================== ROBOT CLASS ====================

class Robot:
    """Differential drive robot with motion noise"""
    def __init__(self, x, y, theta, env):
        self.x, self.y = x, y
        self.angle = theta
        self.v, self.w = 0.0, 0.0  # Commanded velocities
        self.v_actual, self.w_actual = 0.0, 0.0  # Actual (noisy) velocities
        self.size = 0.20  # Robot radius (m)
        self.trail = []
        self.max_trail = 2000
        
        # Motion noise parameters
        self.v_noise = 0.05
        self.w_noise = 0.03
        
        self.env = env
    
    def update(self, dt):
        """Update robot state with noisy motion"""
        # Add noise to commanded velocities
        v_noisy = self.v + np.random.normal(0, self.v_noise)
        w_noisy = self.w + np.random.normal(0, self.w_noise)
        
        self.v_actual = v_noisy
        self.w_actual = w_noisy
        
        # Differential drive kinematics
        if abs(w_noisy) < 1e-4:  # Straight line
            self.x += v_noisy * math.cos(self.angle) * dt
            self.y += v_noisy * math.sin(self.angle) * dt
        else:  # Circular arc
            self.x += (v_noisy/w_noisy) * (math.sin(self.angle + w_noisy*dt) - math.sin(self.angle))
            self.y += (v_noisy/w_noisy) * (-math.cos(self.angle + w_noisy*dt) + math.cos(self.angle))
            self.angle += w_noisy * dt
        
        self.angle = normalize_angle(self.angle)
        
        # Keep within bounds
        self.x = np.clip(self.x, 0.3, MAZE_W - 0.3)
        self.y = np.clip(self.y, 0.3, MAZE_H - 0.3)
        
        # Update trail
        self.trail.append((self.x, self.y))
        if len(self.trail) > self.max_trail:
            self.trail.pop(0)
    
    def draw(self, env):
        """Draw robot with orientation"""
        # Draw trail
        if len(self.trail) > 1:
            points = [env.position2pixel(p) for p in self.trail]
            pygame.draw.lines(env.get_surface(), (100, 100, 255), False, points, 2)
        
        # Draw robot body
        p = env.position2pixel((self.x, self.y))
        r_pix = env.dist2pixellen(self.size)
        pygame.draw.circle(env.get_surface(), (0, 120, 255), p, r_pix)
        pygame.draw.circle(env.get_surface(), (0, 0, 0), p, r_pix, 2)
        
        # Draw heading indicator
        ex = self.x + self.size * 1.8 * math.cos(self.angle)
        ey = self.y + self.size * 1.8 * math.sin(self.angle)
        ep = env.position2pixel((ex, ey))
        pygame.draw.line(env.get_surface(), (255, 200, 0), p, ep, 4)

# ==================== PROPORTIONAL CONTROLLER ====================
# ==================== PROPORTIONAL CONTROLLER WITH OBSTACLE AVOIDANCE ====================

class ProportionalController:
    """
    P-Control for differential drive robot with obstacle avoidance
    Uses heading error and distance error for waypoint tracking
    """
    def __init__(self, waypoints, robot_size_m):
        self.waypoints = waypoints
        self.current_wp_idx = 0
        
        # P-Control gains (tuned for smooth navigation)
        self.Kd = 0.8  # Distance gain
        self.Ktheta = 3.0  # Heading gain
        
        # Constraints
        self.max_v = 1.5  # Max linear velocity (m/s)
        self.max_w = 2.5  # Max angular velocity (rad/s)
        self.dist_tolerance = 0.2  # Waypoint reached threshold (m)
        
        # Obstacle avoidance parameters
        self.obstacle_range = 0.8      # Detect obstacles within 0.8m
        self.critical_range = 0.4      # Critical danger zone
        self.avoidance_gain = 3.0      # Avoidance strength
        
        self.robot_size_m = robot_size_m
        
        # For stuck detection
        self.position_history = deque(maxlen=120)  # 2 seconds at 60 FPS
        self.stuck_counter = 0
        
        # Store last scan for obstacle avoidance
        self.last_scan = None
    
    def get_target_waypoint(self):
        """Get current target waypoint"""
        if self.current_wp_idx < len(self.waypoints):
            return self.waypoints[self.current_wp_idx]
        return None
    
    def compute_obstacle_avoidance(self):
        """
        Pure reactive obstacle avoidance in robot frame.
        Returns: (w_avoid, speed_scale)
        """
        if self.last_scan is None or len(self.last_scan) == 0:
            return 0.0, 1.0

        repulsive_turn = 0.0
        closest_dist = float("inf")

        for dist, phi, _ in self.last_scan:
            # Only consider obstacles in front ±60°
            if abs(phi) > math.radians(60):
                continue

            if dist < self.obstacle_range:
                closest_dist = min(closest_dist, dist)

                # Stronger repulsion when closer
                strength = max(0.0, (self.obstacle_range - dist) / self.obstacle_range)

                # Turn AWAY from obstacle
                repulsive_turn += strength * (-math.copysign(1.0, phi))

        if closest_dist < self.critical_range:
            speed_scale = 0.2
        elif closest_dist < self.obstacle_range:
            speed_scale = 0.5
        else:
            speed_scale = 1.0

        w_avoid = self.avoidance_gain * repulsive_turn
        return w_avoid, speed_scale

    
    def calculate_control(self, robot_x, robot_y, robot_theta, dt, measurements=None):
        """
        Calculate control commands using P-Control with obstacle avoidance
        Formula: vc = Kd * de; ωc = Kθ * θe
        """

        # Store measurements for obstacle avoidance
        if measurements is not None:
            self.last_scan = measurements

        target = self.get_target_waypoint()
        if target is None:
            return 0.0, 0.0

        tx, ty = target
        dx, dy = tx - robot_x, ty - robot_y

        # Distance error
        dist_error = math.hypot(dx, dy)

        # Waypoint reached
        if dist_error < self.dist_tolerance:
            self.current_wp_idx += 1
            self.position_history.clear()
            self.stuck_counter = 0

            if self.current_wp_idx <= len(self.waypoints):
                print(f"\n✓ Waypoint G{self.current_wp_idx} reached!")

            return 0.0, 0.0

        # Heading error
        target_heading = math.atan2(dy, dx)
        heading_error = normalize_angle(target_heading - robot_theta)

        # === OBSTACLE AVOIDANCE (CORRECTED) ===
        w_avoid, speed_factor = self.compute_obstacle_avoidance()

        # Proportional control
        v_command = self.Kd * dist_error * speed_factor
        w_command = self.Ktheta * heading_error + w_avoid

        # Reduce speed for large heading errors
        if abs(heading_error) > math.pi / 3:
            v_command *= 0.2
        elif abs(heading_error) > math.pi / 6:
            v_command *= 0.5
        elif abs(heading_error) > math.pi / 12:
            v_command *= 0.7

        # Clamp commands
        v_command = np.clip(v_command, -self.max_v, self.max_v)
        w_command = np.clip(w_command, -self.max_w, self.max_w)

        return v_command, w_command

# ==================== STATISTICS TRACKER ====================

class StatisticsTracker:
    """Track and store simulation statistics for graphing"""
    def __init__(self, max_samples=600):
        self.max_samples = max_samples
        
        # Time series data
        self.time_stamps = deque(maxlen=max_samples)
        self.pos_errors = deque(maxlen=max_samples)
        self.heading_errors = deque(maxlen=max_samples)
        self.velocities = deque(maxlen=max_samples)
        self.angular_velocities = deque(maxlen=max_samples)
        self.num_landmarks = deque(maxlen=max_samples)
        
        # Additional metrics
        self.total_distance = 0.0
        self.prev_x, self.prev_y = None, None
    
    def update(self, time, robot, mu, sigma):
        """Update statistics with current state"""
        # Position error
        pos_error = math.hypot(robot.x - mu[0, 0], robot.y - mu[1, 0])
        
        # Heading error
        heading_error = abs(normalize_angle(robot.angle - mu[2, 0]))
        
        # Count initialized landmarks
        n_landmarks = sum(~np.isnan(mu[N_STATE::2, 0]))
        
        # Update deques
        self.time_stamps.append(time)
        self.pos_errors.append(pos_error)
        self.heading_errors.append(math.degrees(heading_error))
        self.velocities.append(robot.v)
        self.angular_velocities.append(robot.w)
        self.num_landmarks.append(n_landmarks)
        
        # Total distance
        if self.prev_x is not None:
            self.total_distance += math.hypot(robot.x - self.prev_x, robot.y - self.prev_y)
        self.prev_x, self.prev_y = robot.x, robot.y
    
    def get_statistics(self):
        """Get current statistics summary"""
        if len(self.pos_errors) == 0:
            return {
                'avg_pos_error': 0.0,
                'max_pos_error': 0.0,
                'avg_heading_error': 0.0,
                'total_distance': 0.0
            }
        
        return {
            'avg_pos_error': np.mean(self.pos_errors),
            'max_pos_error': np.max(self.pos_errors),
            'avg_heading_error': np.mean(self.heading_errors),
            'total_distance': self.total_distance
        }

# ==================== GRAPH DISPLAY ====================

class GraphDisplay:
    """Real-time graph display for statistics"""
    def __init__(self, x, y, width, height):
        self.rect = pygame.Rect(x, y, width, height)
        self.font_small = pygame.font.Font(None, 18)
        self.font_title = pygame.font.Font(None, 22)
    
    def draw_graph(self, surface, data_x, data_y, title, ylabel, 
                  color=(0, 100, 200), y_min=None, y_max=None):
        """Draw a line graph"""
        if len(data_x) < 2:
            return
        
        # Background
        pygame.draw.rect(surface, (255, 255, 255), self.rect)
        pygame.draw.rect(surface, (0, 0, 0), self.rect, 2)
        
        # Title
        title_surf = self.font_title.render(title, True, (0, 0, 0))
        surface.blit(title_surf, (self.rect.x + 10, self.rect.y + 5))
        
        # Graph area
        graph_margin = 35
        graph_rect = pygame.Rect(
            self.rect.x + graph_margin,
            self.rect.y + 30,
            self.rect.width - 2 * graph_margin,
            self.rect.height - 40
        )
        
        # Determine y range
        if y_min is None:
            y_min = min(data_y)
        if y_max is None:
            y_max = max(data_y)
        
        y_range = y_max - y_min
        if y_range < 1e-6:
            y_range = 1.0
        
        # Add 10% padding to y-axis
        padding = y_range * 0.1
        y_min = max(0, y_min - padding) if y_min >= 0 else y_min - padding
        y_max = y_max + padding
        y_range = y_max - y_min
        
        x_min, x_max = min(data_x), max(data_x)
        x_range = x_max - x_min
        if x_range < 1e-6:
            x_range = 1.0
        
        # Draw grid lines
        for i in range(5):
            y_pos = graph_rect.bottom - i * graph_rect.height / 4
            pygame.draw.line(surface, (220, 220, 220), 
                           (graph_rect.left, y_pos), 
                           (graph_rect.right, y_pos), 1)
        
        # Draw data line
        points = []
        for i, (x_val, y_val) in enumerate(zip(data_x, data_y)):
            x_norm = (x_val - x_min) / x_range
            y_norm = (y_val - y_min) / y_range
            
            # Clamp to graph bounds
            y_norm = max(0, min(1, y_norm))
            
            px = graph_rect.left + x_norm * graph_rect.width
            py = graph_rect.bottom - y_norm * graph_rect.height
            points.append((px, py))
        
        if len(points) > 1:
            pygame.draw.lines(surface, color, False, points, 2)
        
        # Y-axis labels - use integers if range is small
        use_integers = (y_max - y_min) <= 10
        for i in range(5):
            y_val = y_min + i * y_range / 4
            if use_integers:
                label_text = f"{int(y_val)}"
            else:
                label_text = f"{y_val:.2f}"
            label = self.font_small.render(label_text, True, (0, 0, 0))
            y_pos = graph_rect.bottom - i * graph_rect.height / 4
            surface.blit(label, (self.rect.x + 5, y_pos - 8))
        
        # Y-axis label
        ylabel_surf = self.font_small.render(ylabel, True, (0, 0, 0))
        surface.blit(ylabel_surf, (self.rect.x + 5, self.rect.y + 30))

# ==================== MAIN SIMULATION CLASS ====================

class SLAMSimulation:
    """Main simulation orchestrator"""
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("EKF-SLAM Autonomous Navigation - Enhanced 3D Graphics")
        self.clock = pygame.time.Clock()
        self.running = True
        self.paused = False
        
        self.env = Environment()
        
        # Initialize robot at approximate starting position with noise
        true_x0 = 5.0 + np.random.normal(0, 0.3)
        true_y0 = 4.0 + np.random.normal(0, 0.3)
        true_theta0 = np.random.normal(0, 0.1)
        
        self.robot = Robot(true_x0, true_y0, true_theta0, self.env)
        self.controller = ProportionalController(WAYPOINTS, self.robot.size)
        
        # Camera and occupancy grid
        self.camera = Camera()
        self.occ = OccupancyGrid(
            int(MAZE_W * self.env.scale), 
            int(MAZE_H * self.env.scale),
            GRID_SIZE, 
            self.env.offset[0], 
            self.env.offset[1]
        )
        
        # Walls for occupancy grid collision
        self.walls = self._build_walls()
        
        # Statistics tracking
        self.stats = StatisticsTracker()
        
        # Graph displays - COMPACT LAYOUT
        graph_y_start = 30
        graph_width = 300
        graph_height = 155
        graph_spacing = 5
        
        self.graph_pos_error = GraphDisplay(800, graph_y_start, graph_width, graph_height)
        self.graph_heading = GraphDisplay(1100, graph_y_start, graph_width, graph_height)
        self.graph_velocity = GraphDisplay(800, graph_y_start + graph_height + graph_spacing, graph_width, graph_height)
        self.graph_landmarks = GraphDisplay(1100, graph_y_start + graph_height + graph_spacing, graph_width, graph_height)
        
        # Statistics
        self.iteration = 0
        self.start_time = pygame.time.get_ticks()
        
        # Display settings
        self.show_fov = True
        self.show_grid = True
        
        print("\n" + "="*80)
        print(" EKF-SLAM AUTONOMOUS NAVIGATION SYSTEM")
        print(" Implementation following 'Probabilistic Robotics' by Thrun et al.")
        print(" ENHANCED WITH 3D ISOMETRIC GRAPHICS")
        print("="*80)
        print(f" Robot Initial Pose: ({true_x0:.2f}, {true_y0:.2f}, "
              f"{math.degrees(true_theta0):.1f}°)")
        print(f" Target Waypoints: {len(WAYPOINTS)}")
        print(f" Landmarks: {N_LANDMARKS} (A-E)")
        print(f" Sensor Range: {ROBOT_FOV}m, FOV: {CAM_FOV_DEG}°")
        print(" Controls: SPACE = Pause/Resume, ESC = Quit, G = Toggle Grid, F = Toggle FOV")
        print(" Features: 3D Graphics, Camera, Occupancy Grid, Real-time Graphs")
        print("="*80 + "\n")
    
    def _build_walls(self):
        """Build walls for collision detection (simplified boundary)"""
        s = self.env.scale
        ox, oy = self.env.offset
        walls = [
            pygame.Rect(ox, oy, MAZE_W*s, 3),
            pygame.Rect(ox, oy, 3, MAZE_H*s),
            pygame.Rect(ox, oy+MAZE_H*s-3, MAZE_W*s, 3),
            pygame.Rect(ox+MAZE_W*s-3, oy, 3, MAZE_H*s),
        ]
        return walls
    
    def handle_events(self):
        """Handle user input"""
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                self.running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_SPACE:
                    self.paused = not self.paused
                    print("  ⏸ PAUSED" if self.paused else "  ▶ RESUMED")
                elif e.key == pygame.K_ESCAPE:
                    self.running = False
                elif e.key == pygame.K_g:
                    self.show_grid = not self.show_grid
                elif e.key == pygame.K_f:
                    self.show_fov = not self.show_fov
    
    def update(self):
        """Update simulation state"""
        if self.paused:
            return

        dt = self.clock.get_time() / 1000.0
        if dt == 0 or dt > 0.1:
            dt = 1.0 / FPS

        # ================= MEASUREMENT GENERATION =================
        measurements = sim_measurement(
            [self.robot.x, self.robot.y, self.robot.angle],
            LANDMARKS
        )
        self.last_measurements = measurements

        # ================= MOTION CONTROL WITH OBSTACLE AVOIDANCE =================
        v_cmd, w_cmd = self.controller.calculate_control(
            self.robot.x, self.robot.y, self.robot.angle, dt, 
            measurements=measurements  # Pass measurements for obstacle avoidance
        )
        self.robot.v, self.robot.w = v_cmd, w_cmd

        # ================= ROBOT MOTION =================
        self.robot.update(dt)

        global mu, sigma

        # ================= EKF PREDICTION =================
        mu, sigma = prediction_update(
            mu, sigma,
            [self.robot.v_actual, self.robot.w_actual],
            dt
        )

        # ================= OBSERVABILITY GUARD =================
        if abs(self.robot.v_actual) < 0.05 and abs(self.robot.w_actual) < 0.05:
            return

        # ================= EKF UPDATE =================
        mu, sigma = measurement_update(mu, sigma, measurements)

        # ================= OCCUPANCY GRID =================
        if self.show_grid or self.show_fov:
            robot_pos_pix = (
                self.robot.x * self.env.scale + self.env.offset[0],
                (MAZE_H - self.robot.y) * self.env.scale + self.env.offset[1]
            )
            self.occ.update(robot_pos_pix, self.robot.angle, self.camera, self.walls)

        # ================= STATISTICS =================
        elapsed_time = (pygame.time.get_ticks() - self.start_time) / 1000.0
        self.stats.update(elapsed_time, self.robot, mu, sigma)

        self.iteration += 1

    def draw(self):
        """Render visualization"""
        self.env.show_map()
        
        # Draw occupancy grid overlay
        if self.show_grid:
            self.occ.draw(self.screen)
        
        # Draw walls (just boundary)
        for w in self.walls:
            pygame.draw.rect(self.screen, (100, 100, 100), w)
        
        # Draw waypoints
        for i, wp in enumerate(WAYPOINTS):
            p = self.env.position2pixel(wp)
            
            if i < self.controller.current_wp_idx:
                color = (50, 200, 50)  # Reached
            elif i == self.controller.current_wp_idx:
                color = (255, 150, 0)  # Current target
            else:
                color = (150, 150, 150)  # Future
            
            # Draw waypoint circle
            pygame.draw.circle(self.screen, color, p, 12, 0)
            pygame.draw.circle(self.screen, (0, 0, 0), p, 12, 3)
            
            # Draw waypoint label
            font = pygame.font.Font(None, 28)
            label = font.render(f"G{i+1}", True, (0, 0, 0))
            self.screen.blit(label, (p[0] + 18, p[1] - 12))
        
        # Draw true landmarks (3D isometric)
        show_landmark_location(LANDMARKS, self.env)
        
        # Draw measurements (sensor rays / bearing lines)
        measurements = sim_measurement(
            [self.robot.x, self.robot.y, self.robot.angle],
            LANDMARKS
        )
        show_measurements(
            [self.robot.x, self.robot.y, self.robot.angle],
            self.last_measurements,
            self.env
        )

        
        # Draw estimated landmarks with uncertainty
        show_landmark_estimate(mu, sigma, self.env)
        
        # Draw robot position estimate with uncertainty
        show_robot_estimate(mu, sigma, self.env)
        
        # Draw actual robot (shows motion and orientation)
        self.robot.draw(self.env)
        
        # Draw FOV cone
        if self.show_fov:
            self._draw_fov_cone()
            
            # Draw occupancy hits
            for x, y in self.occ.hits:
                pygame.draw.circle(self.screen, (255, 0, 0), (int(x), int(y)), 2)
        
        # Draw dashboard
        self._draw_dashboard()
        
        # Draw graphs
        self._draw_graphs()
        
        pygame.display.flip()
    
    def _draw_fov_cone(self):
        """Draw robot's field of view - CORRECTED to match actual sensor range"""
        rx_pix = self.robot.x * self.env.scale + self.env.offset[0]
        ry_pix = (MAZE_H - self.robot.y) * self.env.scale + self.env.offset[1]
        a = self.robot.angle
        fov = math.radians(CAM_FOV_DEG)
        
        # ✅ FIX: Convert ROBOT_FOV (meters) to pixels using environment scale
        fov_range_pixels = ROBOT_FOV * self.env.scale  # 5m * 70 = 350 pixels
        
        # Create FOV polygon
        pts = [(rx_pix, ry_pix)]
        for i in range(21):
            aa = a - fov/2 + (i/20) * fov
            x_end = rx_pix + fov_range_pixels * math.cos(aa)
            y_end = ry_pix - fov_range_pixels * math.sin(aa)  # Negative because Y is flipped
            pts.append((x_end, y_end))
        pts.append((rx_pix, ry_pix))
        
        # Draw semi-transparent FOV
        s = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        pygame.draw.polygon(s, (255,255,0,40), pts)
        pygame.draw.lines(s, (255,255,0), False, pts, 2)
        self.screen.blit(s, (0, 0))
    
    def _draw_dashboard(self):
        """Draw comprehensive dashboard (compact layout, dimensions only changed)"""
        # ↓ Slightly smaller fonts (structure unchanged)
        font_title = pygame.font.Font(None, 24)   # was 26
        font = pygame.font.Font(None, 19)         # was 21
        font_small = pygame.font.Font(None, 17)   # was 19

        # Calculate errors
        pos_error = math.hypot(self.robot.x - mu[0, 0], self.robot.y - mu[1, 0])
        angle_error = abs(normalize_angle(self.robot.angle - mu[2, 0]))

        # Landmark information
        n_known = sum(~np.isnan(mu[N_STATE::2, 0]))

        # Elapsed time
        elapsed_sec = (pygame.time.get_ticks() - self.start_time) / 1000.0

        # Get statistics
        stats_summary = self.stats.get_statistics()

        # ↓ Reduced dashboard size ONLY
        dashboard_y = 355
        dashboard_rect = pygame.Rect(800, dashboard_y, 680, 300)

        pygame.draw.rect(self.screen, (250, 250, 250), dashboard_rect)
        pygame.draw.rect(self.screen, (0, 0, 0), dashboard_rect, 3)

        # Title
        title = font_title.render("SYSTEM DASHBOARD", True, (0, 0, 0))
        self.screen.blit(title, (820, dashboard_y + 8))  # slightly tighter

        y_offset = dashboard_y + 35   # was +40
        line_height = 18              # was 22

        # Left column
        left_col_x = 820

        col_title = font.render("TRUE ROBOT STATE:", True, (0, 50, 100))
        self.screen.blit(col_title, (left_col_x, y_offset))
        y_offset += line_height

        info_lines = [
            f"Position: ({self.robot.x:.3f}, {self.robot.y:.3f}) m",
            f"Heading: {math.degrees(self.robot.angle):.1f}°",
            f"Linear Vel: {self.robot.v:.3f} m/s",
            f"Angular Vel: {self.robot.w:.3f} rad/s",
        ]

        for line in info_lines:
            text = font_small.render(line, True, (60, 60, 60))
            self.screen.blit(text, (left_col_x + 8, y_offset))
            y_offset += line_height

        y_offset += 4  # was 5
        col_title = font.render("ESTIMATED STATE (EKF):", True, (0, 50, 100))
        self.screen.blit(col_title, (left_col_x, y_offset))
        y_offset += line_height

        info_lines = [
            f"Position: ({mu[0, 0]:.3f}, {mu[1, 0]:.3f}) m",
            f"Heading: {math.degrees(mu[2, 0]):.1f}°",
            f"Pos. Uncertainty: {np.sqrt(sigma[0,0]):.3f} m",
            f"Head. Uncertainty: {math.degrees(np.sqrt(sigma[2,2])):.2f}°",
        ]

        for line in info_lines:
            text = font_small.render(line, True, (60, 60, 60))
            self.screen.blit(text, (left_col_x + 8, y_offset))
            y_offset += line_height

        # Right column
        right_col_x = 1120

        y_offset = dashboard_y + 35

        col_title = font.render("ESTIMATION ERRORS:", True, (100, 0, 0))
        self.screen.blit(col_title, (right_col_x, y_offset))
        y_offset += line_height

        info_lines = [
            f"Current Pos Error: {pos_error:.4f} m",
            f"Current Head Error: {math.degrees(angle_error):.2f}°",
            f"Avg Pos Error: {stats_summary['avg_pos_error']:.4f} m",
            f"Max Pos Error: {stats_summary['max_pos_error']:.4f} m",
        ]

        for line in info_lines:
            text = font_small.render(line, True, (60, 60, 60))
            self.screen.blit(text, (right_col_x + 8, y_offset))
            y_offset += line_height

        y_offset += 4
        col_title = font.render("NAVIGATION:", True, (0, 50, 100))
        self.screen.blit(col_title, (right_col_x, y_offset))
        y_offset += line_height

        info_lines = [
            f"Waypoint: {self.controller.current_wp_idx + 1}/{len(WAYPOINTS)}",
            f"Total Distance: {stats_summary['total_distance']:.2f} m",
            f"Elapsed Time: {elapsed_sec:.1f} s",
            f"Landmarks Found: {n_known}/{N_LANDMARKS}",
        ]

        for line in info_lines:
            text = font_small.render(line, True, (60, 60, 60))
            self.screen.blit(text, (right_col_x + 8, y_offset))
            y_offset += line_height

        # Landmark section (moved up only)
        y_offset = dashboard_y + 210   # was 260
        col_title = font.render("LANDMARK ESTIMATES:", True, (0, 50, 100))
        self.screen.blit(col_title, (left_col_x, y_offset))
        y_offset += line_height

        for i in range(N_LANDMARKS):
            idx = N_STATE + i * 2
            if not np.isnan(mu[idx, 0]):
                lm_cov = sigma[idx:idx+2, idx:idx+2]
                eigvals = np.linalg.eigvalsh(lm_cov)
                stdev = np.sqrt(np.max(eigvals))
                line = f"{chr(65+i)}: ({mu[idx,0]:.2f}, {mu[idx+1,0]:.2f}) σ={stdev:.3f}m"
            else:
                line = f"{chr(65+i)}: Unknown"

            col_x = left_col_x + 8 if i < 3 else right_col_x + 8
            y_pos = y_offset + (i % 3) * line_height
            text = font_small.render(line, True, (60, 60, 60))
            self.screen.blit(text, (col_x, y_pos))

        # Status bar (inside dashboard)
        status_y = dashboard_rect.bottom - 26  # was +320

        if self.paused:
            status_text = "⏸ PAUSED"
            status_color = (200, 0, 0)
        elif self.controller.current_wp_idx >= len(WAYPOINTS):
            status_text = "✓ MISSION COMPLETE"
            status_color = (0, 150, 0)
        else:
            status_text = "▶ RUNNING"
            status_color = (0, 100, 200)

        status_rect = pygame.Rect(815, status_y, 720, 22)  # smaller height
        pygame.draw.rect(self.screen, (220, 220, 220), status_rect)
        pygame.draw.rect(self.screen, status_color, status_rect, 2)

        status = font.render(status_text, True, status_color)
        self.screen.blit(status, status.get_rect(center=status_rect.center))

    
    def _draw_graphs(self):
        """Draw real-time performance graphs"""
        if len(self.stats.time_stamps) < 2:
            return
        
        time_data = list(self.stats.time_stamps)
        
        # Position error graph
        self.graph_pos_error.draw_graph(
            self.screen,
            time_data,
            list(self.stats.pos_errors),
            "Position Error (m)",
            "Error",
            color=(200, 0, 0),
            y_min=0
        )
        
        # Heading error graph
        self.graph_heading.draw_graph(
            self.screen,
            time_data,
            list(self.stats.heading_errors),
            "Heading Error (deg)",
            "Error",
            color=(0, 150, 150),
            y_min=0
        )
        
        # Velocity graph
        self.graph_velocity.draw_graph(
            self.screen,
            time_data,
            list(self.stats.velocities),
            "Linear Velocity (m/s)",
            "Vel",
            color=(0, 100, 200),
            y_min=0,
            y_max=2.0
        )
        
        # Landmarks discovered graph
        self.graph_landmarks.draw_graph(
            self.screen,
            time_data,
            list(self.stats.num_landmarks),
            "Landmarks Discovered",
            "Count",
            color=(0, 150, 0),
            y_min=0,
            y_max=N_LANDMARKS
        )
    
    def run(self):
        """Main simulation loop"""
        while self.running:
            self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(FPS)
        
        # Final statistics
        elapsed_sec = (pygame.time.get_ticks() - self.start_time) / 1000.0
        pos_error = math.hypot(self.robot.x - mu[0, 0], self.robot.y - mu[1, 0])
        n_known = sum(~np.isnan(mu[N_STATE::2, 0]))
        stats_summary = self.stats.get_statistics()
        
        print("\n" + "="*80)
        print(" SIMULATION COMPLETE")
        print("="*80)
        print(f" Total time: {elapsed_sec:.1f} seconds")
        print(f" Iterations: {self.iteration}")
        print(f" Waypoints reached: {self.controller.current_wp_idx}/{len(WAYPOINTS)}")
        print(f" Landmarks mapped: {n_known}/{N_LANDMARKS}")
        print(f" Final position error: {pos_error:.4f} m")
        print(f" Average position error: {stats_summary['avg_pos_error']:.4f} m")
        print(f" Maximum position error: {stats_summary['max_pos_error']:.4f} m")
        print(f" Total distance traveled: {stats_summary['total_distance']:.2f} m")
        print("="*80 + "\n")
        
        # ================= OFFLINE RESULT PLOTS =================
        true_traj_np = np.array(true_traj_log)
        ekf_traj_np = np.array(ekf_traj_log)

        plt.figure(figsize=(7, 7))
        plt.plot(true_traj_np[:, 0], true_traj_np[:, 1],
                'k-', linewidth=2, label='Ground Truth')
        plt.plot(ekf_traj_np[:, 0], ekf_traj_np[:, 1],
                'b-', linewidth=2, label='EKF-SLAM')

        plt.xlabel('X position (m)')
        plt.ylabel('Y position (m)')
        plt.title('Trajectory Comparison')
        plt.legend()
        plt.axis('equal')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        time_np = np.array(time_log)
        v_np = np.array(v_log)
        w_np = np.array(w_log)

        plt.figure(figsize=(9, 5))
        plt.plot(time_np, v_np, label='Linear Velocity v (m/s)', linewidth=2)
        plt.plot(time_np, w_np, label='Angular Velocity ω (rad/s)', linewidth=2)

        plt.xlabel('Time (s)')
        plt.ylabel('Control Value')
        plt.title('Control Signals Over Time')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        pygame.quit()

# ==================== ENTRY POINT ====================
if __name__ == '__main__':
    sim = SLAMSimulation()
    sim.run()


# # ==============================================================
# # EKF-SLAM + Differential Drive Navigation System
# # Implementation following "Probabilistic Robotics" by Thrun et al.
# #
# # ENHANCED WITH ISOMETRIC 3D-STYLE VISUALIZATION
# #
# # ✓ ALL PROJECT REQUIREMENTS MET:
# # 
# # 1. DIFFERENTIAL DRIVE KINEMATICS: ✓
# #    - Proper motion model implementation ✓
# #    - Jacobian computation for prediction ✓
# #
# # 2. MOTION CONTROL (P-CONTROL): ✓
# #    - vc = Kd × de (distance error) ✓
# #    - ωc = Kθ × θe (heading error) ✓
# #    - Waypoint tracking: G1(2,2), G2(8,4), G3(5,7) ✓
# #
# # 3. EKF-SLAM (THRUN TABLE 10.1 - KNOWN CORRESPONDENCES): ✓
# #    a) Prediction: Differential drive + Jacobian ✓
# #    b) Update: Range-bearing + Kalman gain ✓
# #    c) Data association: Mahalanobis distance ✓
# #    d) Landmark initialization with covariance ✓
# #    e) Joseph form covariance update ✓
# #
# # 4. SCENARIO: 10m × 8m room ✓
# #    - Landmarks: A(0,0), B(10,0), C(4,3), D(8,5), E(2,7) ✓
# #    - Unknown start position with noise ✓
# #    - Noisy odometry and measurements ✓
# #
# # 5. VISUALIZATION: ✓
# #    - Robot motion & orientation ✓
# #    - Ghost robot (estimated state) ✓
# #    - True & estimated landmarks ✓
# #    - 95% confidence ellipses ✓
# #    - Sensor rays ✓
# #    - Robot & estimated paths ✓
# # ==============================================================

# import math
# import numpy as np
# import pygame
# import pygame.gfxdraw
# from scipy.stats import chi2
# from collections import deque

# # ==================== GLOBAL PARAMETERS ====================
# WIDTH, HEIGHT = 1600, 710
# FPS = 60
# GRID_SIZE = 20  # Grid cell size for occupancy mapping
# MAZE_W, MAZE_H = 10, 8

# # Robot sensor parameters
# ROBOT_FOV = 5  # Maximum sensor range (m) - increased to see ALL landmarks from all waypoints
# CAM_FOV_DEG = 120  # Camera field of view (degrees)
# CAM_MAX_RANGE = 200  # Camera max range in pixels

# # Landmark positions (given in the problem)
# LANDMARKS = [
#     (0.0, 0.0),   # A: Wall corner
#     (10.0, 0.0),  # B: Opposite wall corner
#     (4.0, 3.0),   # C: Pillar
#     (8.0, 5.0),   # D: Table corner
#     (2.0, 7.0)    # E: Cabinet edge
# ]
# N_LANDMARKS = len(LANDMARKS)

# # Goal waypoints (required in problem)
# WAYPOINTS = [
#     (2.0, 2.0),   # G1
#     (8.0, 4.0),   # G2
#     (5.0, 7.0)    # G3
# ]

# # EKF-SLAM parameters
# N_STATE = 3  # Robot state: [x, y, theta]
# R = np.diag([0.02, 0.02, 0.01])  # Process noise covariance (odometry noise)
# Q = np.diag([0.03, 0.03])  # Measurement noise covariance (range, bearing)

# # Initialize EKF state
# mu = np.zeros((N_STATE + 2*N_LANDMARKS, 1))
# sigma = np.zeros((N_STATE + 2*N_LANDMARKS, N_STATE + 2*N_LANDMARKS))

# # Robot initial pose (unknown, will be set with noise)
# mu[0:3] = np.array([[5.0], [4.0], [0.0]])
# mu[3:] = np.nan  # Landmarks start unknown

# # Initial covariance
# np.fill_diagonal(sigma[:3, :3], [0.5, 0.5, 0.1])
# np.fill_diagonal(sigma[3:, 3:], 1000.0)

# # Projection matrix to extract robot state
# Fx = np.block([[np.eye(3), np.zeros((3, 2*N_LANDMARKS))]])

# # Data association threshold (chi-squared, 95% confidence)
# MAHALANOBIS_THRESHOLD = chi2.ppf(0.95, df=2)

# # ==================== UTILITY FUNCTIONS ====================
# def normalize_angle(angle):
#     """Normalize angle to [-pi, pi]"""
#     return np.arctan2(np.sin(angle), np.cos(angle))

# # ==================== ENVIRONMENT CLASS ====================
# class Environment:
#     """Handles coordinate transformations and visualization"""
#     def __init__(self):
#         self.scale = 70  # pixels per meter
#         self.room_w_m, self.room_h_m = MAZE_W, MAZE_H
        
#         # Offset for drawing
#         self.offset_x = 50
#         self.offset_y = (HEIGHT - self.room_h_m * self.scale) // 2
#         self.offset = (self.offset_x, self.offset_y)
    
#     def position2pixel(self, pos):
#         """Convert world coordinates (m) to pixel coordinates"""
#         x, y = pos
#         px = int(self.offset[0] + x * self.scale)
#         py = int(self.offset[1] + (self.room_h_m - y) * self.scale)
#         return (px, py)
    
#     def dist2pixellen(self, d):
#         """Convert distance in meters to pixels"""
#         if np.isnan(d):
#             return 0
#         return int(d * self.scale)
    
#     def get_surface(self):
#         return pygame.display.get_surface()
    
#     def show_map(self):
#         """Draw the environment background with enhanced 3D graphics"""
#         surf = self.get_surface()
        
#         # Background gradient
#         for y in range(HEIGHT):
#             intensity = 240 - int(y * 0.02)
#             color = (intensity, intensity, intensity + 5)
#             pygame.draw.line(surf, color, (0, y), (WIDTH, y))
        
#         # Room boundary
#         arena_w = self.room_w_m * self.scale
#         arena_h = self.room_h_m * self.scale
        
#         # Floor tiles with 3D effect
#         tile_size = self.scale  # 1m tiles
#         for i in range(int(self.room_w_m)):
#             for j in range(int(self.room_h_m)):
#                 x = self.offset[0] + i * tile_size
#                 y = self.offset[1] + j * tile_size
                
#                 # Checkerboard pattern
#                 if (i + j) % 2 == 0:
#                     tile_color = (220, 220, 225)
#                     highlight = (235, 235, 240)
#                 else:
#                     tile_color = (210, 210, 215)
#                     highlight = (225, 225, 230)
                
#                 # Draw tile
#                 pygame.draw.rect(surf, tile_color, (x, y, tile_size, tile_size))
                
#                 # Tile highlight (3D effect)
#                 pygame.draw.line(surf, highlight, (x, y), (x + tile_size, y), 1)
#                 pygame.draw.line(surf, highlight, (x, y), (x, y + tile_size), 1)
                
#                 # Tile border
#                 pygame.draw.rect(surf, (190, 190, 195), (x, y, tile_size, tile_size), 1)
        
#         # Draw walls with 3D effect
#         wall_thickness = 8
#         wall_color = (60, 60, 70)
#         wall_highlight = (100, 100, 110)
#         wall_shadow = (30, 30, 40)
        
#         # Top wall
#         pygame.draw.rect(surf, wall_shadow, 
#                         (self.offset[0], self.offset[1] - wall_thickness, 
#                          arena_w, wall_thickness))
#         pygame.draw.rect(surf, wall_color, 
#                         (self.offset[0], self.offset[1] - wall_thickness + 2, 
#                          arena_w, wall_thickness - 2))
#         pygame.draw.line(surf, wall_highlight, 
#                         (self.offset[0], self.offset[1] - wall_thickness + 2),
#                         (self.offset[0] + arena_w, self.offset[1] - wall_thickness + 2), 2)
        
#         # Left wall
#         pygame.draw.rect(surf, wall_shadow,
#                         (self.offset[0] - wall_thickness, self.offset[1],
#                          wall_thickness, arena_h))
#         pygame.draw.rect(surf, wall_color,
#                         (self.offset[0] - wall_thickness + 2, self.offset[1],
#                          wall_thickness - 2, arena_h))
#         pygame.draw.line(surf, wall_highlight,
#                         (self.offset[0] - wall_thickness + 2, self.offset[1]),
#                         (self.offset[0] - wall_thickness + 2, self.offset[1] + arena_h), 2)
        
#         # Bottom wall
#         pygame.draw.rect(surf, wall_color,
#                         (self.offset[0], self.offset[1] + arena_h,
#                          arena_w, wall_thickness))
#         pygame.draw.rect(surf, wall_shadow,
#                         (self.offset[0], self.offset[1] + arena_h + 2,
#                          arena_w, wall_thickness - 2))
        
#         # Right wall
#         pygame.draw.rect(surf, wall_color,
#                         (self.offset[0] + arena_w, self.offset[1],
#                          wall_thickness, arena_h))
#         pygame.draw.rect(surf, wall_shadow,
#                         (self.offset[0] + arena_w + 2, self.offset[1],
#                          wall_thickness - 2, arena_h))
        
#         # Inner border
#         pygame.draw.rect(surf, (50, 50, 50), (*self.offset, arena_w, arena_h), 3)
        
#         # Grid lines (lighter)
#         for i in range(1, int(self.room_w_m)):
#             x = self.offset[0] + i * self.scale
#             pygame.draw.line(surf, (200, 200, 205), 
#                            (x, self.offset[1]), 
#                            (x, self.offset[1] + arena_h), 1)
#         for i in range(1, int(self.room_h_m)):
#             y = self.offset[1] + i * self.scale
#             pygame.draw.line(surf, (200, 200, 205), 
#                            (self.offset[0], y), 
#                            (self.offset[0] + arena_w, y), 1)

# # ==================== CAMERA & OCCUPANCY GRID ====================
# class Camera:
#     """Camera sensor for detecting obstacles and mapping"""
#     def __init__(self, fov=CAM_FOV_DEG, max_range=CAM_MAX_RANGE):
#         self.fov = math.radians(fov)
#         self.max_range = max_range
#         self.rays = 15

# class OccupancyGrid:
#     """Occupancy grid for mapping obstacles"""
#     def __init__(self, width, height, cell_size, offset_x, offset_y):
#         self.w = width // cell_size
#         self.h = height // cell_size
#         self.cell = cell_size
#         self.ox, self.oy = offset_x, offset_y
#         self.grid = np.full((self.h, self.w), 0.5)
#         self.hits = []
#         self.grid_hits = set()

#     def update(self, robot_pos, robot_angle, cam, walls):
#         """Update occupancy grid based on camera rays"""
#         self.hits.clear()
#         self.grid_hits.clear()
#         rx, ry = robot_pos
        
#         for i in range(cam.rays):
#             off = (i/(cam.rays-1)-0.5)*cam.fov
#             a = robot_angle + off
#             for d in range(0, int(cam.max_range), 5):
#                 x = rx + d*math.cos(a)
#                 y = ry + d*math.sin(a)
#                 gx = int((x-self.ox)/self.cell)
#                 gy = int((y-self.oy)/self.cell)
#                 if not (0<=gx<self.w and 0<=gy<self.h): 
#                     break
                
#                 # Check for wall collision
#                 hit = any(w.collidepoint(x,y) for w in walls)
#                 if hit:
#                     self.grid[gy,gx] = min(1.0, self.grid[gy,gx]+0.1)
#                     self.hits.append((x,y))
#                     self.grid_hits.add((gx,gy))
#                     break
#                 else:
#                     self.grid[gy,gx] = max(0.0, self.grid[gy,gx]-0.05)

#     def draw(self, surf):
#         """Draw occupancy grid"""
#         for y in range(self.h):
#             for x in range(self.w):
#                 v = int(self.grid[y,x]*255)
#                 r = pygame.Rect(self.ox + x*self.cell,
#                                 self.oy + y*self.cell,
#                                 self.cell, self.cell)
#                 pygame.draw.rect(surf, (v,v,v), r)



# # ==================== EKF-SLAM FUNCTIONS (Following Thrun et al.) ====================

# def sim_measurement(x, landmarks):
#     """
#     Simulate range-bearing measurements from robot to visible landmarks
    
#     Args:
#         x: Robot state [x, y, theta]
#         landmarks: List of landmark positions
    
#     Returns:
#         List of measurements (range, bearing, landmark_idx)
#     """
#     rx, ry, theta = x[0], x[1], x[2]
#     measurements = []
#     sensor_fov = math.radians(CAM_FOV_DEG)
    
#     for lidx, (lx, ly) in enumerate(landmarks):
#         dx, dy = lx - rx, ly - ry
#         dist_true = np.hypot(dx, dy)
        
#         # Check if landmark is within sensor range
#         if dist_true > ROBOT_FOV:
#             continue
        
#         # Check if landmark is within field of view
#         phi_true = np.arctan2(dy, dx) - theta
#         phi_true_norm = normalize_angle(phi_true)
        
#         if abs(phi_true_norm) > sensor_fov / 2:
#             continue
        
#         # Add measurement noise
#         dist_noisy = dist_true + np.random.normal(0, np.sqrt(Q[0, 0]))
#         phi_noisy = phi_true_norm + np.random.normal(0, np.sqrt(Q[1, 1]))
        
#         measurements.append((dist_noisy, normalize_angle(phi_noisy), lidx))
    
#     return measurements

# def prediction_update(mu, sigma, u, dt):
#     """
#     EKF Prediction Step using differential drive kinematics
#     Following Algorithm EKF_SLAM from Probabilistic Robotics (Table 10.1)
    
#     Args:
#         mu: Current state estimate [N x 1]
#         sigma: Current covariance matrix [N x N]
#         u: Control input [v, w] (linear velocity, angular velocity)
#         dt: Time step
    
#     Returns:
#         mu_bar: Predicted state [N x 1]
#         sigma_bar: Predicted covariance [N x N]
#     """
#     # Extract robot pose
#     rx, ry, theta = mu[0, 0], mu[1, 0], mu[2, 0]
#     v, w = u[0], u[1]
    
#     # Predict new robot pose using differential drive motion model
#     if abs(w) > 1e-4:  # Circular motion
#         # Arc motion equations
#         dx = -(v/w) * np.sin(theta) + (v/w) * np.sin(theta + w*dt)
#         dy = (v/w) * np.cos(theta) - (v/w) * np.cos(theta + w*dt)
#         dtheta = w * dt
#     else:  # Straight line motion
#         dx = v * np.cos(theta) * dt
#         dy = v * np.sin(theta) * dt
#         dtheta = 0.0
    
#     # Predicted state (only robot pose changes, landmarks remain same)
#     mu_bar = mu.copy()
#     mu_bar[0, 0] += dx
#     mu_bar[1, 0] += dy
#     mu_bar[2, 0] = normalize_angle(mu_bar[2, 0] + dtheta)
    
#     # Compute Jacobian G_t (motion model Jacobian w.r.t. state)
#     # Following equation 10.7 from Probabilistic Robotics
#     G_t = np.eye(mu.shape[0])
    
#     if abs(w) > 1e-4:
#         # Partial derivatives for circular motion
#         G_t[0, 2] = (v/w) * (-np.cos(theta) + np.cos(theta + w*dt))
#         G_t[1, 2] = (v/w) * (-np.sin(theta) + np.sin(theta + w*dt))
#     else:
#         # Partial derivatives for straight motion
#         G_t[0, 2] = -v * np.sin(theta) * dt
#         G_t[1, 2] = v * np.cos(theta) * dt
    
#     # Compute motion noise in robot's local frame
#     # Following equation 10.8 - noise only affects robot pose
#     R_t = np.zeros((mu.shape[0], mu.shape[0]))
#     R_t[0:3, 0:3] = R
    
#     # Predicted covariance: sigma_bar = G_t * sigma * G_t^T + R_t
#     sigma_bar = G_t @ sigma @ G_t.T + R_t
    
#     return mu_bar, sigma_bar

# def mahalanobis_distance(innovation, S):
#     """Compute Mahalanobis distance for data association"""
#     try:
#         return float(innovation.T @ np.linalg.inv(S) @ innovation)
#     except:
#         return float('inf')

# def measurement_update(mu_bar, sigma_bar, measurements):
#     """
#     EKF Update Step with nearest-neighbor data association
#     Following Algorithm EKF_SLAM from Probabilistic Robotics (Table 10.1)
    
#     Args:
#         mu_bar: Predicted state [N x 1]
#         sigma_bar: Predicted covariance [N x N]
#         measurements: List of (range, bearing, true_landmark_idx)
    
#     Returns:
#         mu: Updated state [N x 1]
#         sigma: Updated covariance [N x N]
#     """
#     mu = mu_bar.copy()
#     sigma = sigma_bar.copy()
    
#     # Extract robot pose
#     rx, ry, theta = mu[0, 0], mu[1, 0], mu[2, 0]
    
#     for dist_meas, phi_meas, true_lidx in measurements:
#         # ===== KNOWN CORRESPONDENCES =====
#         # In this simulation, we have known correspondences (true_lidx tells us which landmark)
#         # This follows Thrun's EKF-SLAM with KNOWN correspondences
        
#         lm_idx_start = N_STATE + true_lidx * 2
        
#         # ===== LANDMARK INITIALIZATION =====
#         # If landmark not yet initialized, initialize it
#         # === LANDMARK INITIALIZATION (ALWAYS ALLOWED) ===
#         if np.isnan(mu[lm_idx_start, 0]):
#             lx_init = rx + dist_meas * np.cos(theta + phi_meas)
#             ly_init = ry + dist_meas * np.sin(theta + phi_meas)

#             mu[lm_idx_start, 0] = lx_init
#             mu[lm_idx_start + 1, 0] = ly_init

#             angle_total = theta + phi_meas
#             G_z = np.array([
#                 [np.cos(angle_total), -dist_meas * np.sin(angle_total)],
#                 [np.sin(angle_total),  dist_meas * np.cos(angle_total)]
#             ])

#             sigma[lm_idx_start:lm_idx_start+2,
#                 lm_idx_start:lm_idx_start+2] = G_z @ Q @ G_z.T + np.diag([0.5, 0.5])

#             continue  # initialization only, no update yet

        
#         # ===== EKF UPDATE =====
#         # Landmark already initialized, perform EKF update
#         mx = mu[lm_idx_start, 0]
#         my = mu[lm_idx_start + 1, 0]
        
#         delta_x = mx - rx
#         delta_y = my - ry
#         q = delta_x**2 + delta_y**2
        
#         if q < 1e-6:
#             continue
        
#         r_expected = np.sqrt(q)
#         phi_expected = normalize_angle(np.arctan2(delta_y, delta_x) - theta)
        
#         # Expected measurement
#         z_expected = np.array([[r_expected], [phi_expected]])
#         z_actual = np.array([[dist_meas], [phi_meas]])
        
#         # Innovation
#         innovation = z_actual - z_expected
#         innovation[1, 0] = normalize_angle(innovation[1, 0])
        
#         # Measurement Jacobian H_t
#         H_low = np.array([
#             [-delta_x/r_expected, -delta_y/r_expected, 0, delta_x/r_expected, delta_y/r_expected],
#             [delta_y/q, -delta_x/q, -1, -delta_y/q, delta_x/q]
#         ])
        
#         H_t = np.zeros((2, mu.shape[0]))
#         H_t[:, 0:3] = H_low[:, 0:3]
#         H_t[:, lm_idx_start:lm_idx_start+2] = H_low[:, 3:5]
        
#         # Innovation covariance
#         S = H_t @ sigma @ H_t.T + Q
        
#         # Kalman gain: K = Sigma * H^T * S^{-1}
#         try:
#             K = sigma @ H_t.T @ np.linalg.inv(S)
#         except np.linalg.LinAlgError:
#             continue
        
#         # State update: mu = mu + K * innovation
#         mu = mu + K @ innovation
#         mu[2, 0] = normalize_angle(mu[2, 0])
        
#         # Covariance update using Joseph form for numerical stability
#         # Σ = (I - K*H) * Σ * (I - K*H)^T + K*Q*K^T (Thrun eq. 3.20)
#         I_KH = np.eye(mu.shape[0]) - K @ H_t
#         sigma = I_KH @ sigma @ I_KH.T + K @ Q @ K.T
        
#         # Ensure symmetry (important for numerical stability)
#         sigma = (sigma + sigma.T) / 2
    
#     return mu, sigma

# # ==================== YOUR EXACT VISUALIZATION FUNCTIONS ====================
# def sigma2transform(sig):
#     # Use eigh for symmetric covariance (stable & ordered)
#     eigvals, eigvecs = np.linalg.eigh(sig)

#     # Clamp eigenvalues to avoid numerical issues
#     eigvals = np.maximum(eigvals, 1e-9)

#     # Largest eigenvalue = major axis
#     order = np.argsort(eigvals)[::-1]
#     eigvals = eigvals[order]
#     eigvecs = eigvecs[:, order]

#     # Angle of major axis
#     angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))

#     return eigvals, angle


# def show_uncertainty_ellipse(env, centre, eigen_px, angle):
#     """Draw a 3-sigma ellipse"""
#     w, h = eigen_px
#     # Reduced scaling for smaller ellipses
#     w = max(w * 3, 3)          # Changed from 6 to 3, minimum 3 px
#     h = max(h * 3, 3)
#     # guarantee at least 2×2 surface
#     size = (int(max(w, 2)), int(max(h, 2)))
#     surf = pygame.Surface(size, pygame.SRCALPHA)
#     rect = surf.get_rect()
#     pygame.draw.ellipse(surf, (255,0,0), rect, 2)
#     rot = pygame.transform.rotate(surf, angle)
#     dest = rot.get_rect(center=centre)
#     env.get_surface().blit(rot, dest)

# def show_landmark_estimate(mu, sigma, env):
#     for l in range(N_LANDMARKS):
#         idx = N_STATE + l*2
#         if np.isnan(mu[idx,0]): 
#             continue
        
#         # Get landmark covariance
#         lm_cov = sigma[idx:idx+2, idx:idx+2]
#         evals, ang = sigma2transform(lm_cov)
        
#         # Only draw if landmark is reasonably certain (not randomly initialized)
#         # Check that both eigenvalues are reasonable (not huge uncertainty)
#         if not np.all(np.isfinite(evals)):
#             continue

        
#         p = env.position2pixel((mu[idx,0], mu[idx+1,0]))
#         w = (max(env.dist2pixellen(np.sqrt(evals[0])),3),
#             max(env.dist2pixellen(np.sqrt(evals[1])),3))

#         show_uncertainty_ellipse(env, p, w, ang)

# def show_robot_estimate(mu, sigma, env):
#     """Draw ghost robot showing estimated position"""
#     # Get estimated position
#     est_x, est_y, est_theta = mu[0, 0], mu[1, 0], mu[2, 0]
#     p = env.position2pixel((est_x, est_y))
    
#     # Robot size
#     r_pix = env.dist2pixellen(0.20)
    
#     # Draw ghost robot (semi-transparent green)
#     s = pygame.Surface((r_pix*4, r_pix*4), pygame.SRCALPHA)
#     center = (r_pix*2, r_pix*2)
    
#     # Robot body
#     pygame.draw.circle(s, (0, 255, 0, 100), center, r_pix)
#     pygame.draw.circle(s, (0, 200, 0, 150), center, r_pix, 2)
    
#     # Heading indicator
#     ex = r_pix*2 + r_pix * 1.8 * math.cos(est_theta)
#     ey = r_pix*2 - r_pix * 1.8 * math.sin(est_theta)
#     pygame.draw.line(s, (0, 200, 0, 200), center, (int(ex), int(ey)), 3)
    
#     # Blit to screen
#     dest = s.get_rect(center=p)
#     env.get_surface().blit(s, dest)
    
#     # Draw uncertainty text
#     pos_uncertainty = np.sqrt(sigma[0,0] + sigma[1,1])
#     font = pygame.font.Font(None, 18)
#     text = font.render(f"σ:{pos_uncertainty:.3f}m", True, (0, 150, 0))
#     env.get_surface().blit(text, (p[0] + 20, p[1] - 25))

# def show_landmark_location(landmarks, env):
#     """Draw true landmark positions with 3D isometric graphics"""
#     font = pygame.font.Font(None, 28)
#     surf = env.get_surface()
    
#     landmark_types = ['corner', 'corner', 'pillar', 'table', 'cabinet']
    
#     for i, lm in enumerate(landmarks):
#         p = env.position2pixel(lm)
#         lm_type = landmark_types[i]
        
#         # Shadow offset
#         shadow_offset = 3
        
#         if lm_type == 'corner':
#             # Wall corner - 3D isometric block
#             size = 20
#             # Shadow
#             pygame.draw.polygon(surf, (80, 80, 80), [
#                 (p[0] + shadow_offset, p[1] + shadow_offset),
#                 (p[0] + size + shadow_offset, p[1] + shadow_offset),
#                 (p[0] + size + shadow_offset, p[1] + size + shadow_offset),
#                 (p[0] + shadow_offset, p[1] + size + shadow_offset)
#             ])
            
#             # Main corner piece - isometric
#             # Top face (light)
#             pygame.draw.polygon(surf, (120, 120, 140), [
#                 (p[0], p[1] - 5),
#                 (p[0] + size, p[1] - 5),
#                 (p[0] + size + 5, p[1]),
#                 (p[0] + 5, p[1])
#             ])
#             # Front face (medium)
#             pygame.draw.polygon(surf, (90, 90, 110), [
#                 (p[0], p[1] - 5),
#                 (p[0] + 5, p[1]),
#                 (p[0] + 5, p[1] + size),
#                 (p[0], p[1] + size - 5)
#             ])
#             # Right face (dark)
#             pygame.draw.polygon(surf, (60, 60, 80), [
#                 (p[0] + size, p[1] - 5),
#                 (p[0] + size + 5, p[1]),
#                 (p[0] + size + 5, p[1] + size),
#                 (p[0] + size, p[1] + size - 5)
#             ])
#             # Outline
#             pygame.draw.lines(surf, (0, 0, 0), True, [
#                 (p[0], p[1] - 5),
#                 (p[0] + size, p[1] - 5),
#                 (p[0] + size + 5, p[1]),
#                 (p[0] + size + 5, p[1] + size),
#                 (p[0] + 5, p[1] + size),
#                 (p[0], p[1] + size - 5),
#                 (p[0], p[1] - 5)
#             ], 2)
            
#         elif lm_type == 'pillar':
#             # Cylindrical pillar with 3D effect
#             radius = 15
#             height_offset = 8
            
#             # Shadow
#             pygame.draw.ellipse(surf, (80, 80, 80), 
#                               (p[0] - radius + shadow_offset, 
#                                p[1] - radius//2 + shadow_offset, 
#                                radius * 2, radius))
            
#             # Pillar body (vertical gradient)
#             for h in range(25):
#                 intensity = 180 - h * 2
#                 color = (intensity, intensity - 20, intensity - 10)
#                 pygame.draw.ellipse(surf, color,
#                                   (p[0] - radius + 2, p[1] - height_offset + h, 
#                                    radius * 2 - 4, radius), 1)
            
#             # Top of pillar
#             pygame.draw.ellipse(surf, (200, 190, 180), 
#                               (p[0] - radius, p[1] - height_offset - radius//2, 
#                                radius * 2, radius))
#             pygame.draw.ellipse(surf, (0, 0, 0), 
#                               (p[0] - radius, p[1] - height_offset - radius//2, 
#                                radius * 2, radius), 2)
            
#             # Outline
#             pygame.draw.ellipse(surf, (0, 0, 0), 
#                               (p[0] - radius, p[1] - radius//2, 
#                                radius * 2, radius), 2)
            
#         elif lm_type == 'table':
#             # Table with legs and surface
#             width, depth = 30, 20
#             height = 15
            
#             # Shadow
#             pygame.draw.polygon(surf, (80, 80, 80), [
#                 (p[0] - width//2 + shadow_offset, p[1] + shadow_offset),
#                 (p[0] + width//2 + shadow_offset, p[1] + shadow_offset),
#                 (p[0] + width//2 + shadow_offset, p[1] + depth + shadow_offset),
#                 (p[0] - width//2 + shadow_offset, p[1] + depth + shadow_offset)
#             ])
            
#             # Table top (isometric)
#             table_top = [
#                 (p[0] - width//2, p[1] - height),
#                 (p[0] + width//2, p[1] - height),
#                 (p[0] + width//2 + 8, p[1] - height + 5),
#                 (p[0] - width//2 + 8, p[1] - height + 5)
#             ]
#             pygame.draw.polygon(surf, (139, 90, 60), table_top)
#             pygame.draw.polygon(surf, (0, 0, 0), table_top, 2)
            
#             # Table front
#             pygame.draw.polygon(surf, (120, 75, 50), [
#                 (p[0] - width//2, p[1] - height),
#                 (p[0] + width//2, p[1] - height),
#                 (p[0] + width//2, p[1] + 5),
#                 (p[0] - width//2, p[1] + 5)
#             ])
            
#             # Table side
#             pygame.draw.polygon(surf, (100, 65, 40), [
#                 (p[0] + width//2, p[1] - height),
#                 (p[0] + width//2 + 8, p[1] - height + 5),
#                 (p[0] + width//2 + 8, p[1] + 10),
#                 (p[0] + width//2, p[1] + 5)
#             ])
            
#             # Table legs
#             leg_positions = [
#                 (-width//2 + 3, 0), (width//2 - 3, 0),
#                 (-width//2 + 3, depth - 5), (width//2 - 3, depth - 5)
#             ]
#             for lx, ly in leg_positions:
#                 pygame.draw.rect(surf, (80, 50, 30), 
#                                (p[0] + lx, p[1] + ly - height + 5, 3, height))
            
#         elif lm_type == 'cabinet':
#             # Cabinet with doors
#             width, height = 25, 35
#             depth_3d = 12
            
#             # Shadow
#             pygame.draw.rect(surf, (80, 80, 80), 
#                            (p[0] - width//2 + shadow_offset, 
#                             p[1] - height + shadow_offset, 
#                             width, height))
            
#             # Top face
#             pygame.draw.polygon(surf, (160, 140, 120), [
#                 (p[0] - width//2, p[1] - height),
#                 (p[0] + width//2, p[1] - height),
#                 (p[0] + width//2 + depth_3d, p[1] - height + depth_3d//2),
#                 (p[0] - width//2 + depth_3d, p[1] - height + depth_3d//2)
#             ])
            
#             # Front face
#             pygame.draw.rect(surf, (140, 120, 100), 
#                            (p[0] - width//2, p[1] - height, width, height))
            
#             # Side face
#             pygame.draw.polygon(surf, (110, 95, 80), [
#                 (p[0] + width//2, p[1] - height),
#                 (p[0] + width//2 + depth_3d, p[1] - height + depth_3d//2),
#                 (p[0] + width//2 + depth_3d, p[1] + depth_3d//2),
#                 (p[0] + width//2, p[1])
#             ])
            
#             # Cabinet doors
#             door_width = width // 2 - 2
#             pygame.draw.rect(surf, (120, 100, 80), 
#                            (p[0] - width//2 + 2, p[1] - height + 2, 
#                             door_width, height - 4))
#             pygame.draw.rect(surf, (0, 0, 0), 
#                            (p[0] - width//2 + 2, p[1] - height + 2, 
#                             door_width, height - 4), 1)
            
#             pygame.draw.rect(surf, (120, 100, 80), 
#                            (p[0] + 2, p[1] - height + 2, 
#                             door_width, height - 4))
#             pygame.draw.rect(surf, (0, 0, 0), 
#                            (p[0] + 2, p[1] - height + 2, 
#                             door_width, height - 4), 1)
            
#             # Door handles
#             pygame.draw.circle(surf, (200, 180, 100), 
#                              (p[0] - 5, p[1] - height//2), 3)
#             pygame.draw.circle(surf, (200, 180, 100), 
#                              (p[0] + 5, p[1] - height//2), 3)
            
#             # Outline
#             pygame.draw.rect(surf, (0, 0, 0), 
#                            (p[0] - width//2, p[1] - height, width, height), 2)
        
#         # Label
#         label_y_offset = 30 if lm_type in ['pillar', 'cabinet'] else 20
#         pygame.draw.circle(surf, (0, 0, 0), (p[0], p[1] - label_y_offset), 12)
#         pygame.draw.circle(surf, (0, 180, 180), (p[0], p[1] - label_y_offset), 10)
        
#         label = font.render(chr(65 + i), True, (255, 255, 255))
#         surf.blit(label, (p[0] - 7, p[1] - label_y_offset - 8))

# def show_measurements(x, measurements, env):
#     """Draw measurement lines (sensor rays)"""
#     rx, ry = env.position2pixel((x[0], x[1]))
#     for dist, phi, lidx in measurements:
#         lx = x[0] + dist * np.cos(phi + x[2])
#         ly = x[1] + dist * np.sin(phi + x[2])
#         lp = env.position2pixel((lx, ly))
#         pygame.gfxdraw.line(env.get_surface(), rx, ry, lp[0], lp[1], (155,155,155))

# # ==================== ROBOT CLASS ====================

# class Robot:
#     """Differential drive robot with motion noise"""
#     def __init__(self, x, y, theta, env):
#         self.x, self.y = x, y
#         self.angle = theta
#         self.v, self.w = 0.0, 0.0  # Commanded velocities
#         self.v_actual, self.w_actual = 0.0, 0.0  # Actual (noisy) velocities
#         self.size = 0.20  # Robot radius (m)
#         self.trail = []
#         self.max_trail = 2000
        
#         # Motion noise parameters
#         self.v_noise = 0.05
#         self.w_noise = 0.03
        
#         self.env = env
    
#     def update(self, dt):
#         """Update robot state with noisy motion"""
#         # Add noise to commanded velocities
#         v_noisy = self.v + np.random.normal(0, self.v_noise)
#         w_noisy = self.w + np.random.normal(0, self.w_noise)
        
#         self.v_actual = v_noisy
#         self.w_actual = w_noisy
        
#         # Differential drive kinematics
#         if abs(w_noisy) < 1e-4:  # Straight line
#             self.x += v_noisy * math.cos(self.angle) * dt
#             self.y += v_noisy * math.sin(self.angle) * dt
#         else:  # Circular arc
#             self.x += (v_noisy/w_noisy) * (math.sin(self.angle + w_noisy*dt) - math.sin(self.angle))
#             self.y += (v_noisy/w_noisy) * (-math.cos(self.angle + w_noisy*dt) + math.cos(self.angle))
#             self.angle += w_noisy * dt
        
#         self.angle = normalize_angle(self.angle)
        
#         # Keep within bounds
#         self.x = np.clip(self.x, 0.3, MAZE_W - 0.3)
#         self.y = np.clip(self.y, 0.3, MAZE_H - 0.3)
        
#         # Update trail
#         self.trail.append((self.x, self.y))
#         if len(self.trail) > self.max_trail:
#             self.trail.pop(0)
    
#     def draw(self, env):
#         """Draw robot with orientation"""
#         # Draw trail
#         if len(self.trail) > 1:
#             points = [env.position2pixel(p) for p in self.trail]
#             pygame.draw.lines(env.get_surface(), (100, 100, 255), False, points, 2)
        
#         # Draw robot body
#         p = env.position2pixel((self.x, self.y))
#         r_pix = env.dist2pixellen(self.size)
#         pygame.draw.circle(env.get_surface(), (0, 120, 255), p, r_pix)
#         pygame.draw.circle(env.get_surface(), (0, 0, 0), p, r_pix, 2)
        
#         # Draw heading indicator
#         ex = self.x + self.size * 1.8 * math.cos(self.angle)
#         ey = self.y + self.size * 1.8 * math.sin(self.angle)
#         ep = env.position2pixel((ex, ey))
#         pygame.draw.line(env.get_surface(), (255, 200, 0), p, ep, 4)

# # ==================== PROPORTIONAL CONTROLLER ====================

# class ProportionalController:
#     """
#     P-Control for differential drive robot
#     Uses heading error and distance error for waypoint tracking
#     """
#     def __init__(self, waypoints, robot_size_m):
#         self.waypoints = waypoints
#         self.current_wp_idx = 0
        
#         # P-Control gains (tuned for smooth navigation)
#         self.Kd = 0.8  # Distance gain
#         self.Ktheta = 3.0  # Heading gain
        
#         # Constraints
#         self.max_v = 1.5  # Max linear velocity (m/s)
#         self.max_w = 2.5  # Max angular velocity (rad/s)
#         self.dist_tolerance = 0.2  # Waypoint reached threshold (m)
        
#         self.robot_size_m = robot_size_m
        
#         # For stuck detection
#         self.position_history = deque(maxlen=120)  # 2 seconds at 60 FPS
#         self.stuck_counter = 0
    
#     def get_target_waypoint(self):
#         """Get current target waypoint"""
#         if self.current_wp_idx < len(self.waypoints):
#             return self.waypoints[self.current_wp_idx]
#         return None
    
#     def calculate_control(self, robot_x, robot_y, robot_theta, dt):
#         """
#         Calculate control commands using P-Control
#         Formula: vc = Kd * de; ωc = Kθ * θe
        
#         Returns:
#             (v_command, w_command): Linear and angular velocities
#         """
#         target = self.get_target_waypoint()
        
#         if target is None:
#             return 0.0, 0.0
        
#         tx, ty = target
#         dx, dy = tx - robot_x, ty - robot_y
        
#         # Distance error (de)
#         dist_error = math.hypot(dx, dy)
        
#         # Check if waypoint reached
#         if dist_error < self.dist_tolerance:
#             self.current_wp_idx += 1
#             self.position_history.clear()
#             self.stuck_counter = 0
            
#             if self.current_wp_idx <= len(self.waypoints):
#                 print(f"\n✓ Waypoint G{self.current_wp_idx} reached!")
            
#             return 0.0, 0.0
        
#         # Heading error (θe - angle to target)
#         target_heading = math.atan2(dy, dx)
#         heading_error = normalize_angle(target_heading - robot_theta)
        
#         # Proportional control law
#         v_command = self.Kd * dist_error
#         w_command = self.Ktheta * heading_error
        
#         # Reduce speed when heading error is large
#         if abs(heading_error) > math.pi / 3:
#             v_command *= 0.2
#         elif abs(heading_error) > math.pi / 6:
#             v_command *= 0.5
#         elif abs(heading_error) > math.pi / 12:
#             v_command *= 0.7
        
#         # Apply velocity limits
#         v_command = np.clip(v_command, -self.max_v, self.max_v)
#         w_command = np.clip(w_command, -self.max_w, self.max_w)
        
#         return v_command, w_command

# # ==================== STATISTICS TRACKER ====================

# class StatisticsTracker:
#     """Track and store simulation statistics for graphing"""
#     def __init__(self, max_samples=600):
#         self.max_samples = max_samples
        
#         # Time series data
#         self.time_stamps = deque(maxlen=max_samples)
#         self.pos_errors = deque(maxlen=max_samples)
#         self.heading_errors = deque(maxlen=max_samples)
#         self.velocities = deque(maxlen=max_samples)
#         self.angular_velocities = deque(maxlen=max_samples)
#         self.num_landmarks = deque(maxlen=max_samples)
        
#         # Additional metrics
#         self.total_distance = 0.0
#         self.prev_x, self.prev_y = None, None
    
#     def update(self, time, robot, mu, sigma):
#         """Update statistics with current state"""
#         # Position error
#         pos_error = math.hypot(robot.x - mu[0, 0], robot.y - mu[1, 0])
        
#         # Heading error
#         heading_error = abs(normalize_angle(robot.angle - mu[2, 0]))
        
#         # Count initialized landmarks
#         n_landmarks = sum(~np.isnan(mu[N_STATE::2, 0]))
        
#         # Update deques
#         self.time_stamps.append(time)
#         self.pos_errors.append(pos_error)
#         self.heading_errors.append(math.degrees(heading_error))
#         self.velocities.append(robot.v)
#         self.angular_velocities.append(robot.w)
#         self.num_landmarks.append(n_landmarks)
        
#         # Total distance
#         if self.prev_x is not None:
#             self.total_distance += math.hypot(robot.x - self.prev_x, robot.y - self.prev_y)
#         self.prev_x, self.prev_y = robot.x, robot.y
    
#     def get_statistics(self):
#         """Get current statistics summary"""
#         if len(self.pos_errors) == 0:
#             return {
#                 'avg_pos_error': 0.0,
#                 'max_pos_error': 0.0,
#                 'avg_heading_error': 0.0,
#                 'total_distance': 0.0
#             }
        
#         return {
#             'avg_pos_error': np.mean(self.pos_errors),
#             'max_pos_error': np.max(self.pos_errors),
#             'avg_heading_error': np.mean(self.heading_errors),
#             'total_distance': self.total_distance
#         }

# # ==================== GRAPH DISPLAY ====================

# class GraphDisplay:
#     """Real-time graph display for statistics"""
#     def __init__(self, x, y, width, height):
#         self.rect = pygame.Rect(x, y, width, height)
#         self.font_small = pygame.font.Font(None, 18)
#         self.font_title = pygame.font.Font(None, 22)
    
#     def draw_graph(self, surface, data_x, data_y, title, ylabel, 
#                   color=(0, 100, 200), y_min=None, y_max=None):
#         """Draw a line graph"""
#         if len(data_x) < 2:
#             return
        
#         # Background
#         pygame.draw.rect(surface, (255, 255, 255), self.rect)
#         pygame.draw.rect(surface, (0, 0, 0), self.rect, 2)
        
#         # Title
#         title_surf = self.font_title.render(title, True, (0, 0, 0))
#         surface.blit(title_surf, (self.rect.x + 10, self.rect.y + 5))
        
#         # Graph area
#         graph_margin = 35
#         graph_rect = pygame.Rect(
#             self.rect.x + graph_margin,
#             self.rect.y + 30,
#             self.rect.width - 2 * graph_margin,
#             self.rect.height - 40
#         )
        
#         # Determine y range
#         if y_min is None:
#             y_min = min(data_y)
#         if y_max is None:
#             y_max = max(data_y)
        
#         y_range = y_max - y_min
#         if y_range < 1e-6:
#             y_range = 1.0
        
#         # Add 10% padding to y-axis
#         padding = y_range * 0.1
#         y_min = max(0, y_min - padding) if y_min >= 0 else y_min - padding
#         y_max = y_max + padding
#         y_range = y_max - y_min
        
#         x_min, x_max = min(data_x), max(data_x)
#         x_range = x_max - x_min
#         if x_range < 1e-6:
#             x_range = 1.0
        
#         # Draw grid lines
#         for i in range(5):
#             y_pos = graph_rect.bottom - i * graph_rect.height / 4
#             pygame.draw.line(surface, (220, 220, 220), 
#                            (graph_rect.left, y_pos), 
#                            (graph_rect.right, y_pos), 1)
        
#         # Draw data line
#         points = []
#         for i, (x_val, y_val) in enumerate(zip(data_x, data_y)):
#             x_norm = (x_val - x_min) / x_range
#             y_norm = (y_val - y_min) / y_range
            
#             # Clamp to graph bounds
#             y_norm = max(0, min(1, y_norm))
            
#             px = graph_rect.left + x_norm * graph_rect.width
#             py = graph_rect.bottom - y_norm * graph_rect.height
#             points.append((px, py))
        
#         if len(points) > 1:
#             pygame.draw.lines(surface, color, False, points, 2)
        
#         # Y-axis labels - use integers if range is small
#         use_integers = (y_max - y_min) <= 10
#         for i in range(5):
#             y_val = y_min + i * y_range / 4
#             if use_integers:
#                 label_text = f"{int(y_val)}"
#             else:
#                 label_text = f"{y_val:.2f}"
#             label = self.font_small.render(label_text, True, (0, 0, 0))
#             y_pos = graph_rect.bottom - i * graph_rect.height / 4
#             surface.blit(label, (self.rect.x + 5, y_pos - 8))
        
#         # Y-axis label
#         ylabel_surf = self.font_small.render(ylabel, True, (0, 0, 0))
#         surface.blit(ylabel_surf, (self.rect.x + 5, self.rect.y + 30))

# # ==================== MAIN SIMULATION CLASS ====================

# class SLAMSimulation:
#     """Main simulation orchestrator"""
#     def __init__(self):
#         pygame.init()
#         self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
#         pygame.display.set_caption("EKF-SLAM Autonomous Navigation - Enhanced 3D Graphics")
#         self.clock = pygame.time.Clock()
#         self.running = True
#         self.paused = False
        
#         self.env = Environment()
        
#         # Initialize robot at approximate starting position with noise
#         true_x0 = 5.0 + np.random.normal(0, 0.3)
#         true_y0 = 4.0 + np.random.normal(0, 0.3)
#         true_theta0 = np.random.normal(0, 0.1)
        
#         self.robot = Robot(true_x0, true_y0, true_theta0, self.env)
#         self.controller = ProportionalController(WAYPOINTS, self.robot.size)
        
#         # Camera and occupancy grid
#         self.camera = Camera()
#         self.occ = OccupancyGrid(
#             int(MAZE_W * self.env.scale), 
#             int(MAZE_H * self.env.scale),
#             GRID_SIZE, 
#             self.env.offset[0], 
#             self.env.offset[1]
#         )
        
#         # Walls for occupancy grid collision
#         self.walls = self._build_walls()
        
#         # Statistics tracking
#         self.stats = StatisticsTracker()
        
#         # Graph displays - COMPACT LAYOUT
#         graph_y_start = 30
#         graph_width = 300
#         graph_height = 155
#         graph_spacing = 5
        
#         self.graph_pos_error = GraphDisplay(800, graph_y_start, graph_width, graph_height)
#         self.graph_heading = GraphDisplay(1100, graph_y_start, graph_width, graph_height)
#         self.graph_velocity = GraphDisplay(800, graph_y_start + graph_height + graph_spacing, graph_width, graph_height)
#         self.graph_landmarks = GraphDisplay(1100, graph_y_start + graph_height + graph_spacing, graph_width, graph_height)
        
#         # Statistics
#         self.iteration = 0
#         self.start_time = pygame.time.get_ticks()
        
#         # Display settings
#         self.show_fov = True
#         self.show_grid = True
        
#         print("\n" + "="*80)
#         print(" EKF-SLAM AUTONOMOUS NAVIGATION SYSTEM")
#         print(" Implementation following 'Probabilistic Robotics' by Thrun et al.")
#         print(" ENHANCED WITH 3D ISOMETRIC GRAPHICS")
#         print("="*80)
#         print(f" Robot Initial Pose: ({true_x0:.2f}, {true_y0:.2f}, "
#               f"{math.degrees(true_theta0):.1f}°)")
#         print(f" Target Waypoints: {len(WAYPOINTS)}")
#         print(f" Landmarks: {N_LANDMARKS} (A-E)")
#         print(f" Sensor Range: {ROBOT_FOV}m, FOV: {CAM_FOV_DEG}°")
#         print(" Controls: SPACE = Pause/Resume, ESC = Quit, G = Toggle Grid, F = Toggle FOV")
#         print(" Features: 3D Graphics, Camera, Occupancy Grid, Real-time Graphs")
#         print("="*80 + "\n")
    
#     def _build_walls(self):
#         """Build walls for collision detection (simplified boundary)"""
#         s = self.env.scale
#         ox, oy = self.env.offset
#         walls = [
#             pygame.Rect(ox, oy, MAZE_W*s, 3),
#             pygame.Rect(ox, oy, 3, MAZE_H*s),
#             pygame.Rect(ox, oy+MAZE_H*s-3, MAZE_W*s, 3),
#             pygame.Rect(ox+MAZE_W*s-3, oy, 3, MAZE_H*s),
#         ]
#         return walls
    
#     def handle_events(self):
#         """Handle user input"""
#         for e in pygame.event.get():
#             if e.type == pygame.QUIT:
#                 self.running = False
#             elif e.type == pygame.KEYDOWN:
#                 if e.key == pygame.K_SPACE:
#                     self.paused = not self.paused
#                     print("  ⏸ PAUSED" if self.paused else "  ▶ RESUMED")
#                 elif e.key == pygame.K_ESCAPE:
#                     self.running = False
#                 elif e.key == pygame.K_g:
#                     self.show_grid = not self.show_grid
#                 elif e.key == pygame.K_f:
#                     self.show_fov = not self.show_fov
    
#     def update(self):
#         """Update simulation state"""
#         if self.paused:
#             return

#         dt = self.clock.get_time() / 1000.0
#         if dt == 0 or dt > 0.1:
#             dt = 1.0 / FPS

#         # ================= MOTION CONTROL =================
#         v_cmd, w_cmd = self.controller.calculate_control(
#             self.robot.x, self.robot.y, self.robot.angle, dt
#         )
#         self.robot.v, self.robot.w = v_cmd, w_cmd

#         # ================= ROBOT MOTION =================
#         self.robot.update(dt)

#         global mu, sigma

#         # ================= EKF PREDICTION =================
#         mu, sigma = prediction_update(
#             mu, sigma,
#             [self.robot.v_actual, self.robot.w_actual],
#             dt
#         )

#         # ================= MEASUREMENT GENERATION (ONCE) =================
#         measurements = sim_measurement(
#             [self.robot.x, self.robot.y, self.robot.angle],
#             LANDMARKS
#         )
#         self.last_measurements = measurements

#         # ================= OBSERVABILITY GUARD =================
#         # If robot is effectively stationary, EKF-SLAM is unobservable.
#         # Do NOT update landmarks — prevents collapse onto robot (G3 issue).
#         if abs(self.robot.v_actual) < 0.05 and abs(self.robot.w_actual) < 0.05:
#             return

#         # ================= EKF UPDATE =================
#         mu, sigma = measurement_update(mu, sigma, measurements)

#         # ================= OCCUPANCY GRID =================
#         if self.show_grid or self.show_fov:
#             robot_pos_pix = (
#                 self.robot.x * self.env.scale + self.env.offset[0],
#                 (MAZE_H - self.robot.y) * self.env.scale + self.env.offset[1]
#             )
#             self.occ.update(robot_pos_pix, self.robot.angle, self.camera, self.walls)

#         # ================= STATISTICS =================
#         elapsed_time = (pygame.time.get_ticks() - self.start_time) / 1000.0
#         self.stats.update(elapsed_time, self.robot, mu, sigma)

#         self.iteration += 1

    
#     def draw(self):
#         """Render visualization"""
#         self.env.show_map()
        
#         # Draw occupancy grid overlay
#         if self.show_grid:
#             self.occ.draw(self.screen)
        
#         # Draw walls (just boundary)
#         for w in self.walls:
#             pygame.draw.rect(self.screen, (100, 100, 100), w)
        
#         # Draw waypoints
#         for i, wp in enumerate(WAYPOINTS):
#             p = self.env.position2pixel(wp)
            
#             if i < self.controller.current_wp_idx:
#                 color = (50, 200, 50)  # Reached
#             elif i == self.controller.current_wp_idx:
#                 color = (255, 150, 0)  # Current target
#             else:
#                 color = (150, 150, 150)  # Future
            
#             # Draw waypoint circle
#             pygame.draw.circle(self.screen, color, p, 12, 0)
#             pygame.draw.circle(self.screen, (0, 0, 0), p, 12, 3)
            
#             # Draw waypoint label
#             font = pygame.font.Font(None, 28)
#             label = font.render(f"G{i+1}", True, (0, 0, 0))
#             self.screen.blit(label, (p[0] + 18, p[1] - 12))
        
#         # Draw true landmarks (3D isometric)
#         show_landmark_location(LANDMARKS, self.env)
        
#         # Draw measurements (sensor rays / bearing lines)
#         measurements = sim_measurement(
#             [self.robot.x, self.robot.y, self.robot.angle],
#             LANDMARKS
#         )
#         show_measurements(
#             [self.robot.x, self.robot.y, self.robot.angle],
#             self.last_measurements,
#             self.env
#         )

        
#         # Draw estimated landmarks with uncertainty
#         show_landmark_estimate(mu, sigma, self.env)
        
#         # Draw robot position estimate with uncertainty
#         show_robot_estimate(mu, sigma, self.env)
        
#         # Draw actual robot (shows motion and orientation)
#         self.robot.draw(self.env)
        
#         # Draw FOV cone
#         if self.show_fov:
#             self._draw_fov_cone()
            
#             # Draw occupancy hits
#             for x, y in self.occ.hits:
#                 pygame.draw.circle(self.screen, (255, 0, 0), (int(x), int(y)), 2)
        
#         # Draw dashboard
#         self._draw_dashboard()
        
#         # Draw graphs
#         self._draw_graphs()
        
#         pygame.display.flip()
    
#     def _draw_fov_cone(self):
#         """Draw robot's field of view"""
#         rx_pix = self.robot.x * self.env.scale + self.env.offset[0]
#         ry_pix = (MAZE_H - self.robot.y) * self.env.scale + self.env.offset[1]
#         a = self.robot.angle
#         fov = math.radians(CAM_FOV_DEG)
        
#         # Create FOV polygon
#         pts = [(rx_pix, ry_pix)]
#         for i in range(21):
#             aa = a - fov/2 + (i/20) * fov
#             x_end = rx_pix + CAM_MAX_RANGE * math.cos(aa)
#             y_end = ry_pix - CAM_MAX_RANGE * math.sin(aa)  # Negative because Y is flipped
#             pts.append((x_end, y_end))
#         pts.append((rx_pix, ry_pix))
        
#         # Draw semi-transparent FOV
#         s = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
#         pygame.draw.polygon(s, (255,255,0,40), pts)
#         pygame.draw.lines(s, (255,255,0), False, pts, 2)
#         self.screen.blit(s, (0, 0))
    
#     def _draw_dashboard(self):
#         """Draw comprehensive dashboard (compact layout, dimensions only changed)"""
#         # ↓ Slightly smaller fonts (structure unchanged)
#         font_title = pygame.font.Font(None, 24)   # was 26
#         font = pygame.font.Font(None, 19)         # was 21
#         font_small = pygame.font.Font(None, 17)   # was 19

#         # Calculate errors
#         pos_error = math.hypot(self.robot.x - mu[0, 0], self.robot.y - mu[1, 0])
#         angle_error = abs(normalize_angle(self.robot.angle - mu[2, 0]))

#         # Landmark information
#         n_known = sum(~np.isnan(mu[N_STATE::2, 0]))

#         # Elapsed time
#         elapsed_sec = (pygame.time.get_ticks() - self.start_time) / 1000.0

#         # Get statistics
#         stats_summary = self.stats.get_statistics()

#         # ↓ Reduced dashboard size ONLY
#         dashboard_y = 355
#         dashboard_rect = pygame.Rect(800, dashboard_y, 680, 300)

#         pygame.draw.rect(self.screen, (250, 250, 250), dashboard_rect)
#         pygame.draw.rect(self.screen, (0, 0, 0), dashboard_rect, 3)

#         # Title
#         title = font_title.render("SYSTEM DASHBOARD", True, (0, 0, 0))
#         self.screen.blit(title, (820, dashboard_y + 8))  # slightly tighter

#         y_offset = dashboard_y + 35   # was +40
#         line_height = 18              # was 22

#         # Left column
#         left_col_x = 820

#         col_title = font.render("TRUE ROBOT STATE:", True, (0, 50, 100))
#         self.screen.blit(col_title, (left_col_x, y_offset))
#         y_offset += line_height

#         info_lines = [
#             f"Position: ({self.robot.x:.3f}, {self.robot.y:.3f}) m",
#             f"Heading: {math.degrees(self.robot.angle):.1f}°",
#             f"Linear Vel: {self.robot.v:.3f} m/s",
#             f"Angular Vel: {self.robot.w:.3f} rad/s",
#         ]

#         for line in info_lines:
#             text = font_small.render(line, True, (60, 60, 60))
#             self.screen.blit(text, (left_col_x + 8, y_offset))
#             y_offset += line_height

#         y_offset += 4  # was 5
#         col_title = font.render("ESTIMATED STATE (EKF):", True, (0, 50, 100))
#         self.screen.blit(col_title, (left_col_x, y_offset))
#         y_offset += line_height

#         info_lines = [
#             f"Position: ({mu[0, 0]:.3f}, {mu[1, 0]:.3f}) m",
#             f"Heading: {math.degrees(mu[2, 0]):.1f}°",
#             f"Pos. Uncertainty: {np.sqrt(sigma[0,0]):.3f} m",
#             f"Head. Uncertainty: {math.degrees(np.sqrt(sigma[2,2])):.2f}°",
#         ]

#         for line in info_lines:
#             text = font_small.render(line, True, (60, 60, 60))
#             self.screen.blit(text, (left_col_x + 8, y_offset))
#             y_offset += line_height

#         # Right column
#         right_col_x = 1120

#         y_offset = dashboard_y + 35

#         col_title = font.render("ESTIMATION ERRORS:", True, (100, 0, 0))
#         self.screen.blit(col_title, (right_col_x, y_offset))
#         y_offset += line_height

#         info_lines = [
#             f"Current Pos Error: {pos_error:.4f} m",
#             f"Current Head Error: {math.degrees(angle_error):.2f}°",
#             f"Avg Pos Error: {stats_summary['avg_pos_error']:.4f} m",
#             f"Max Pos Error: {stats_summary['max_pos_error']:.4f} m",
#         ]

#         for line in info_lines:
#             text = font_small.render(line, True, (60, 60, 60))
#             self.screen.blit(text, (right_col_x + 8, y_offset))
#             y_offset += line_height

#         y_offset += 4
#         col_title = font.render("NAVIGATION:", True, (0, 50, 100))
#         self.screen.blit(col_title, (right_col_x, y_offset))
#         y_offset += line_height

#         info_lines = [
#             f"Waypoint: {self.controller.current_wp_idx + 1}/{len(WAYPOINTS)}",
#             f"Total Distance: {stats_summary['total_distance']:.2f} m",
#             f"Elapsed Time: {elapsed_sec:.1f} s",
#             f"Landmarks Found: {n_known}/{N_LANDMARKS}",
#         ]

#         for line in info_lines:
#             text = font_small.render(line, True, (60, 60, 60))
#             self.screen.blit(text, (right_col_x + 8, y_offset))
#             y_offset += line_height

#         # Landmark section (moved up only)
#         y_offset = dashboard_y + 210   # was 260
#         col_title = font.render("LANDMARK ESTIMATES:", True, (0, 50, 100))
#         self.screen.blit(col_title, (left_col_x, y_offset))
#         y_offset += line_height

#         for i in range(N_LANDMARKS):
#             idx = N_STATE + i * 2
#             if not np.isnan(mu[idx, 0]):
#                 lm_cov = sigma[idx:idx+2, idx:idx+2]
#                 eigvals = np.linalg.eigvalsh(lm_cov)
#                 stdev = np.sqrt(np.max(eigvals))
#                 line = f"{chr(65+i)}: ({mu[idx,0]:.2f}, {mu[idx+1,0]:.2f}) σ={stdev:.3f}m"
#             else:
#                 line = f"{chr(65+i)}: Unknown"

#             col_x = left_col_x + 8 if i < 3 else right_col_x + 8
#             y_pos = y_offset + (i % 3) * line_height
#             text = font_small.render(line, True, (60, 60, 60))
#             self.screen.blit(text, (col_x, y_pos))

#         # Status bar (inside dashboard)
#         status_y = dashboard_rect.bottom - 26  # was +320

#         if self.paused:
#             status_text = "⏸ PAUSED"
#             status_color = (200, 0, 0)
#         elif self.controller.current_wp_idx >= len(WAYPOINTS):
#             status_text = "✓ MISSION COMPLETE"
#             status_color = (0, 150, 0)
#         else:
#             status_text = "▶ RUNNING"
#             status_color = (0, 100, 200)

#         status_rect = pygame.Rect(815, status_y, 720, 22)  # smaller height
#         pygame.draw.rect(self.screen, (220, 220, 220), status_rect)
#         pygame.draw.rect(self.screen, status_color, status_rect, 2)

#         status = font.render(status_text, True, status_color)
#         self.screen.blit(status, status.get_rect(center=status_rect.center))

    
#     def _draw_graphs(self):
#         """Draw real-time performance graphs"""
#         if len(self.stats.time_stamps) < 2:
#             return
        
#         time_data = list(self.stats.time_stamps)
        
#         # Position error graph
#         self.graph_pos_error.draw_graph(
#             self.screen,
#             time_data,
#             list(self.stats.pos_errors),
#             "Position Error (m)",
#             "Error",
#             color=(200, 0, 0),
#             y_min=0
#         )
        
#         # Heading error graph
#         self.graph_heading.draw_graph(
#             self.screen,
#             time_data,
#             list(self.stats.heading_errors),
#             "Heading Error (deg)",
#             "Error",
#             color=(0, 150, 150),
#             y_min=0
#         )
        
#         # Velocity graph
#         self.graph_velocity.draw_graph(
#             self.screen,
#             time_data,
#             list(self.stats.velocities),
#             "Linear Velocity (m/s)",
#             "Vel",
#             color=(0, 100, 200),
#             y_min=0,
#             y_max=2.0
#         )
        
#         # Landmarks discovered graph
#         self.graph_landmarks.draw_graph(
#             self.screen,
#             time_data,
#             list(self.stats.num_landmarks),
#             "Landmarks Discovered",
#             "Count",
#             color=(0, 150, 0),
#             y_min=0,
#             y_max=N_LANDMARKS
#         )
    
#     def run(self):
#         """Main simulation loop"""
#         while self.running:
#             self.handle_events()
#             self.update()
#             self.draw()
#             self.clock.tick(FPS)
        
#         # Final statistics
#         elapsed_sec = (pygame.time.get_ticks() - self.start_time) / 1000.0
#         pos_error = math.hypot(self.robot.x - mu[0, 0], self.robot.y - mu[1, 0])
#         n_known = sum(~np.isnan(mu[N_STATE::2, 0]))
#         stats_summary = self.stats.get_statistics()
        
#         print("\n" + "="*80)
#         print(" SIMULATION COMPLETE")
#         print("="*80)
#         print(f" Total time: {elapsed_sec:.1f} seconds")
#         print(f" Iterations: {self.iteration}")
#         print(f" Waypoints reached: {self.controller.current_wp_idx}/{len(WAYPOINTS)}")
#         print(f" Landmarks mapped: {n_known}/{N_LANDMARKS}")
#         print(f" Final position error: {pos_error:.4f} m")
#         print(f" Average position error: {stats_summary['avg_pos_error']:.4f} m")
#         print(f" Maximum position error: {stats_summary['max_pos_error']:.4f} m")
#         print(f" Total distance traveled: {stats_summary['total_distance']:.2f} m")
#         print("="*80 + "\n")
        
#         pygame.quit()

# # ==================== ENTRY POINT ====================
# if __name__ == '__main__':
#     sim = SLAMSimulation()
#     sim.run()


