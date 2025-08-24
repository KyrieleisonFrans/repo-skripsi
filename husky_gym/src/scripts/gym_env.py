''' Code for training environment'''

import gymnasium as gym
import numpy as np
import random
import rospy
import time
import tf

from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped
from gymnasium import spaces
from move_base_msgs.msg import MoveBaseActionResult
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion

from config import ENV_PARAMS
from cylinder_static import randomize_cylinders

class GymEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # gymnasium setup
        # Observation space set for [0, 1] normalization
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(13,), dtype=np.float32)
        self.action_space = spaces.Discrete(8)

        # Temporal filtering parameters
        self.scan_history = []  # Store last N scans
        self.scan_history_size = 7  # Number of scans to consider
        self.collision_threshold = 0.4  # Minimum distance to consider collision
        self.collision_confirmation_steps = 5  # Number of consecutive detections needed

        # ROS setup
        self.rate = rospy.Rate(5)
        self.tf_listener = tf.TransformListener()
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.laser_sub = rospy.Subscriber('/front/scan', LaserScan, self.laser_callback)
        self.odom_sub = rospy.Subscriber('/odometry/filtered', Odometry, self.odom_callback)
        self.dwa_vel_sub = rospy.Subscriber('/dwa/cmd_vel', Twist, self.dwa_callback)
        self.goal_sub = rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_callback)
        self.goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=1, latch=True)
        self.initial_pose_pub = rospy.Publisher('/initialpose', PoseWithCovarianceStamped, queue_size=10)

        laser_data = rospy.wait_for_message('/front/scan', LaserScan)

        self.goal_status_sub = rospy.Subscriber('/move_base/result', MoveBaseActionResult, self.goal_status_callback)
        self.dwa_goal_reached = False
        self.goal_points_gazebo = [
            (-2.08, 4.2),
            (-4.46, 5.52),
            (-5.64, 4.42),
            (-7.75, 5.5),
            (-8.0, 5.00)
        ]

        # ddqn setup
        num_laser_readings = len(laser_data.ranges)
        self.laser_range_max = laser_data.range_max
        self.laser_range_min = laser_data.range_min

        self.current_scan = np.zeros(num_laser_readings)
        self.dwa_velocity = np.zeros(2)
        self.robot_pose = None

        self.goal_position = None
        self.gazebo_goal_position = None
        self.start_position = None
        self.previous_distance = None
        self.has_goal = False
        self.last_cmd_vel = None
        self.episode_start = True
        self.steps_without_progress = 0
        self.previous_time = rospy.Time.now()
        self.prev_obstacle_distances = None
        self.consecutive_zero_dwa_velocities = 0

        self.static_obs = ENV_PARAMS["static_obs"]

        # Normalization ranges (Based on training map)
        self.map_x_min = -10.0
        self.map_x_max = 10.0
        self.map_y_min = 3.0
        self.map_y_max = 8.0

        # Max distance (Based on map diagonal)
        self.max_distance_to_goal = np.sqrt((self.map_x_max - self.map_x_min)**2 + (self.map_y_max - self.map_y_min)**2)
        self.min_distance_to_goal = 0.0

        # Angles from [-pi, pi] to [0, 1]
        self.max_relative_heading = np.pi
        self.min_relative_heading = -np.pi

        self.lidar_angle_min = -2.356  # -135° in rad
        self.lidar_angle_max = 2.356   # +135° in rad

        # Max vel
        self.max_linear_vel = 1.0 
        self.min_linear_vel = -1.0 
        self.max_angular_vel = 1.5 
        self.min_angular_vel = -1.5 

        # max vel obs = max vel robot
        self.max_v_obs = self.max_linear_vel 
        self.min_v_obs = -self.max_linear_vel 

        # Reset robot position
        self.reset_pos_x = 9.0
        self.reset_pos_y = 5.0
        self.reset_yaw_radians = np.radians(180)

        self.obstacle_range_min = 0.3 
        self.obstacle_range_max = 2.0 


    # =================================
    # callback Methods
    # =================================
    def dwa_callback(self, data):
        self.dwa_velocity = np.array([data.linear.x, data.angular.z])

    def goal_callback(self, msg):
        self.goal_position = np.array([
            msg.pose.position.x,
            msg.pose.position.y
        ])

        orientation_q = msg.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        _, _, self.goal_orientation = euler_from_quaternion(orientation_list)

        self.has_goal = True
        self.wait_for_new_goal = False
        rospy.loginfo(f"New goal received at: {self.goal_position}")

    def goal_status_callback(self, msg):
        self.dwa_goal_reached = (msg.status.status == 3)  # 3 = SUCCESS

    def laser_callback(self, data):
        # Store the full LaserScan message
        self.laser_data = data

        current_scan = np.array(data.ranges)
        self.laser_range_max = data.range_max
        self.laser_range_min = data.range_min

        # First replace inf and nan with max range
        current_scan = np.where(
            np.isinf(current_scan) | np.isnan(current_scan),
            data.range_max,
            current_scan
        )

        # Filter values below minimum range
        current_scan = np.where(
            current_scan < data.range_min,
            data.range_max,
            current_scan
        )
        current_scan = np.clip(current_scan, data.range_min, data.range_max)
        
        self.scan_history.append(current_scan)
        if len(self.scan_history) > self.scan_history_size:
            self.scan_history.pop(0)
            
        if len(self.scan_history) > 0:
            self.current_scan = np.median(self.scan_history, axis=0)
        else:
            self.current_scan = current_scan

    def odom_callback(self, data):
        try:
            self.tf_listener.waitForTransform("map", "base_link", rospy.Time(0), rospy.Duration(1.0))
            (trans, rot) = self.tf_listener.lookupTransform("map", "base_link", rospy.Time(0))

            self.robot_pose = type('RobotPose', (), {
                'position': type('Position', (), {
                    'x': trans[0],
                    'y': trans[1],
                    'z': trans[2]
                }),
                'orientation': type('Orientation', (), {
                    'x': rot[0],
                    'y': rot[1],
                    'z': rot[2],
                    'w': rot[3]
                })
            })

            orientation_list = [rot[0], rot[1], rot[2], rot[3]]
            _, _, self.robot_orientation = euler_from_quaternion(orientation_list)

        except (tf.Exception, tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logwarn_throttle(5, f"TF transform from map to base_link failed: {str(e)}")
            self.robot_pose = None
            self.robot_orientation = 0.0
            return

        if self.robot_pose is None:
            return

        if self.goal_position is not None:
            current_position = np.array([
                self.robot_pose.position.x,
                self.robot_pose.position.y
            ])
            if self.previous_distance is None:
                self.previous_distance = np.linalg.norm(current_position - self.goal_position)
                self.start_position = current_position.copy()


    # =================================
    # State setting Methods
    # =================================
    def reset_internal_state(self):
        self.start_position = None
        self.previous_distance = None
        self.steps_without_progress = 0
        self.previous_time = rospy.Time.now()
        self.consecutive_zero_dwa_velocities = 0
        self.prev_obstacle_distances = None
        self.robot_pose = None 
        self.has_goal = False
        self.collision_counter = 0  
        self.scan_history = [] 

    def reset_robot_position(self):
        # Reset in Gazebo
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            robot_state = ModelState()
            robot_state.model_name = 'husky'
            robot_state.pose.position.x = self.reset_pos_x
            robot_state.pose.position.y = self.reset_pos_y
            robot_state.pose.position.z = 0.15

            rviz_yaw = self.reset_yaw_radians + np.pi
            robot_state.pose.orientation.x = 0.0
            robot_state.pose.orientation.y = 0.0
            robot_state.pose.orientation.z = np.sin(self.reset_yaw_radians/2)
            robot_state.pose.orientation.w = np.cos(self.reset_yaw_radians/2)

            response = set_model_state(robot_state)
            if not response.success:
                rospy.logwarn("Failed to reset robot in Gazebo")

        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s" % e)
        rospy.sleep(0.5)

        # Update RViz 
        self.publish_initial_pose(0.0, 0.0, rviz_yaw)
        rospy.sleep(0.5)


    def reset_dwa_planner(self):
        try:
            rospy.wait_for_service('/move_base/clear_costmaps', timeout=0.5)
            clear_costmaps = rospy.ServiceProxy('/move_base/clear_costmaps', Empty)
            clear_costmaps()
            rospy.loginfo("DWA Planner reset: Costmaps cleared!")
        except (rospy.ServiceException, rospy.ROSException) as e:
            rospy.logwarn(f"Failed to clear costmaps: {e}")

        # Reset environment tracking variables
        self.previous_distance = None
        self.steps_without_progress = 0
        self.previous_time = rospy.Time.now()

        # Stop any residual motion
        cmd_vel = Twist()
        self.cmd_vel_pub.publish(cmd_vel)

        rospy.sleep(0.5)

    def publish_initial_pose(self, x, y, yaw):
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = "map"

        pose_msg.pose.pose.position.x = x
        pose_msg.pose.pose.position.y = y
        pose_msg.pose.pose.position.z = 0.0

        pose_msg.pose.pose.orientation.x = 0.0
        pose_msg.pose.pose.orientation.y = 0.0
        pose_msg.pose.pose.orientation.z = np.sin(yaw/2)
        pose_msg.pose.pose.orientation.w = np.cos(yaw/2)

        pose_msg.pose.covariance = [
            0.25, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.25, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.06853891945200942
        ]

        self.initial_pose_pub.publish(pose_msg)

    def set_random_predefined_goal(self):
        gazebo_x, gazebo_y = random.choice(self.goal_points_gazebo)
        rospy.loginfo(f"[Random Goal] Goal terpilih: ({gazebo_x:.2f}, {gazebo_y:.2f})")
        self.set_goal(x=gazebo_x, y=gazebo_y, gazebo_goal=True)

    def set_goal(self, x=17.50, y=-1.00, z=0.0, qx=0.0, qy=0.0, qz=0.039, qw=0.999, gazebo_goal=True):
        # Convert to map frame if it's a Gazebo goal
        if gazebo_goal:
            self.gazebo_goal_position = np.array([x, y])
            x_map = -1*(x - self.reset_pos_x)
            y_map = y - self.reset_pos_y
        else:
            self.gazebo_goal_position = None
            x_map = x
            y_map = y

        # If this is a new goal:
        if not self.has_goal or not np.array_equal(self.goal_position, [x_map, y_map]):
            self.goal_position = np.array([x_map, y_map])
            self.goal_orientation = euler_from_quaternion([qx, qy, qz, qw])[2]

            goal_msg = PoseStamped()
            goal_msg.header.stamp = rospy.Time.now()
            goal_msg.header.frame_id = "map"
            goal_msg.pose.position.x = x_map
            goal_msg.pose.position.y = y_map
            goal_msg.pose.position.z = z
            goal_msg.pose.orientation.x = qx
            goal_msg.pose.orientation.y = qy
            goal_msg.pose.orientation.z = qz
            goal_msg.pose.orientation.w = qw

            self.goal_pub.publish(goal_msg)
            self.has_goal = True
            rospy.loginfo(f"New goal set to: Position({x_map}, {y_map}, {z})")
            return True

        return False 

    def detect_obstacles(self):
        # Use the stored LaserScan message
        if not hasattr(self, 'laser_data') or self.laser_data is None:
            rospy.logwarn_throttle(1, "No laser data available for obstacle detection.")
            return np.zeros(3), np.full(3, self.laser_range_max), np.zeros(3) 

        scan_data = self.laser_data

        # Convert scan data to Cartesian coordinates
        angles = np.linspace(scan_data.angle_min, scan_data.angle_max, len(scan_data.ranges))
        ranges = np.array(scan_data.ranges)

        # Filter invalid readings
        valid_mask = (ranges >= scan_data.range_min) & (ranges <= scan_data.range_max)
        ranges = ranges[valid_mask]
        angles = angles[valid_mask]

        # If no valid obstacles found, return zeros (or max range for distance)
        if len(ranges) == 0:
            return np.zeros(3), np.full(3, scan_data.range_max), np.zeros(3)

        # Find closest obstacles
        closest_indices = np.argsort(ranges)[:3]
        obstacle_distances = ranges[closest_indices]
        obstacle_angles = angles[closest_indices]

        # relative obstacle velocities
        relative_velocities = np.zeros_like(obstacle_distances)
        if self.prev_obstacle_distances is not None and len(self.prev_obstacle_distances) == len(obstacle_distances):
            time_elapsed = (rospy.Time.now() - self.previous_time).to_sec()
            if time_elapsed > 0:
                min_len = min(len(self.prev_obstacle_distances), len(obstacle_distances))
                relative_velocities[:min_len] = (self.prev_obstacle_distances[:min_len] - obstacle_distances[:min_len]) / time_elapsed

        self.prev_obstacle_distances = obstacle_distances.copy()
        self.previous_time = rospy.Time.now()

        if len(obstacle_distances) < 3:
            pad_size = 3 - len(obstacle_distances)
            obstacle_distances = np.pad(obstacle_distances, (0, pad_size), 'constant', constant_values=scan_data.range_max)
            obstacle_angles = np.pad(obstacle_angles, (0, pad_size), 'constant')
            relative_velocities = np.pad(relative_velocities, (0, pad_size), 'constant')

        normalized_theta = (obstacle_angles - self.lidar_angle_min) / (self.lidar_angle_max - self.lidar_angle_min)
        normalized_theta = np.clip(normalized_theta, 0.0, 1.0) 

        return relative_velocities, obstacle_distances, normalized_theta

    def get_state(self):
        if self.robot_pose is None or not self.has_goal:
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        v_obs, l, theta = self.detect_obstacles()

        # Goal info
        current_position = np.array([
            self.robot_pose.position.x,
            self.robot_pose.position.y
        ])
        goal_vector = self.goal_position - current_position
        distance_to_goal = np.linalg.norm(goal_vector)

        # Relative heading to goal
        goal_angle = np.arctan2(goal_vector[1], goal_vector[0])
        relative_heading = goal_angle - self.robot_orientation
        relative_heading = (relative_heading + np.pi) % (2 * np.pi) - np.pi

        # Get current velocities
        current_linear_vel = self.dwa_velocity[0] if self.dwa_velocity is not None else 0.0
        current_angular_vel = self.dwa_velocity[1] if self.dwa_velocity is not None else 0.0

        # Normalize distance_to_goal
        normalized_distance_to_goal = (distance_to_goal - self.min_distance_to_goal) / (self.max_distance_to_goal - self.min_distance_to_goal)
        normalized_distance_to_goal = np.clip(normalized_distance_to_goal, 0.0, 1.0)

        # Normalize relative_heading (from [-pi, pi] to [0, 1])
        normalized_relative_heading = (relative_heading - self.min_relative_heading) / (self.max_relative_heading - self.min_relative_heading)
        normalized_relative_heading = np.clip(normalized_relative_heading, 0.0, 1.0)

        # Normalize distances to obstacles (l)
        if (self.laser_range_max - self.laser_range_min) > 0:
            normalized_l = (l - self.laser_range_min) / (self.laser_range_max - self.laser_range_min)
            normalized_l = np.clip(normalized_l, 0.0, 1.0)
        else:
            normalized_l = np.zeros_like(l)

        # Normalize angles to obstacles (theta) (from [-pi, pi] to [0, 1])
        normalized_theta = (theta - self.lidar_angle_min) / (self.lidar_angle_max - self.lidar_angle_min)
        normalized_theta = np.clip(normalized_theta, 0.0, 1.0)

        # Normalize velocities (from [min_vel, max_vel] to [0, 1])
        normalized_current_linear_vel = (current_linear_vel - self.min_linear_vel) / (self.max_linear_vel - self.min_linear_vel)
        normalized_current_linear_vel = np.clip(normalized_current_linear_vel, 0.0, 1.0)

        normalized_current_angular_vel = (current_angular_vel - self.min_angular_vel) / (self.max_angular_vel - self.min_angular_vel)
        normalized_current_angular_vel = np.clip(normalized_current_angular_vel, 0.0, 1.0)

        # Normalize relative velocities of obstacles (from [min_v_obs, max_v_obs] to [0, 1])
        if (self.max_v_obs - self.min_v_obs) > 0:
            normalized_v_obs = (v_obs - self.min_v_obs) / (self.max_v_obs - self.min_v_obs)
            normalized_v_obs = np.clip(normalized_v_obs, 0.0, 1.0)
        else:
            normalized_v_obs = np.zeros_like(v_obs) 

        # Construct state space
        state = np.concatenate([
            normalized_v_obs,                   # Relative velocities of 3 closest obstacles
            normalized_l,                       # Distances to 3 closest obstacles
            normalized_theta,                   # Angles to 3 closest obstacles
            [normalized_distance_to_goal],      # Distance to goal
            [normalized_relative_heading],      # Relative heading to goal
            [normalized_current_linear_vel],    # Current linear velocity
            [normalized_current_angular_vel]    # Current angular velocity
        ])

        return state.astype(np.float32)

    # =================================
    # Action setting Methods
    # =================================
    def move(self, action):
        v_acc = 0.8
        w_acc = 1.0

        dwa_is_zero = np.allclose(self.dwa_velocity, [0.0, 0.0], atol=1e-2)

        if dwa_is_zero:
            rospy.logwarn_throttle(5, "DWA is not moving, Letting agent act freely.")
            v_target = action[0]
            w_target = action[1]
        else:
            v_target = self.dwa_velocity[0] + action[0]
            w_target = self.dwa_velocity[1] + action[1]

        current_time = rospy.Time.now()
        dt = (current_time - self.previous_time).to_sec()
        dt = max(dt, 0.05)
        self.previous_time = current_time

        if self.episode_start:
            new_v = v_target
            new_w = w_target
            self.last_cmd_vel = [new_v, new_w]
            self.episode_start = False
        else:
            if self.last_cmd_vel is None:
                self.last_cmd_vel = [0.0, 0.0]

            dv_max = v_acc * dt
            dw_max = w_acc * dt

            delta_v = v_target - self.last_cmd_vel[0]
            delta_w = w_target - self.last_cmd_vel[1]

            delta_v = np.clip(delta_v, -dv_max, dv_max)
            delta_w = np.clip(delta_w, -dw_max, dw_max)

            new_v = self.last_cmd_vel[0] + delta_v
            new_w = self.last_cmd_vel[1] + delta_w
            self.last_cmd_vel = [new_v, new_w]

        cmd_vel = Twist()
        cmd_vel.linear.x = new_v
        cmd_vel.angular.z = new_w
        self.cmd_vel_pub.publish(cmd_vel)

        print(f"\n[move()] DWA vel: {self.dwa_velocity[0]:.3f}, {self.dwa_velocity[1]:.3f}, Action: {action}, Final cmd: {new_v:.2f}, {new_w:.2f}")

        return self.get_state(), *self.calculate_reward()


    def calculate_reward(self):
        info = {}
        done = False
        total_reward = 0.0

        if self.episode_start:
            return 0.0, False, {"reason": "episode_start"}

        if self.robot_pose is None:
            rospy.logwarn_throttle(1, "Robot pose not available for reward calculation.")
            return total_reward, True, {"reason": "no_robot_pose"}

        current_position = np.array([
            self.robot_pose.position.x,
            self.robot_pose.position.y
        ])
        distance_to_goal = np.linalg.norm(self.goal_position - current_position)

        goal_direction = np.arctan2(
            self.goal_position[1] - current_position[1],
            self.goal_position[0] - current_position[0]
        )
        angle_error = abs(goal_direction - self.robot_orientation)
        angle_error = min(angle_error, 2 * np.pi - angle_error)

        # The time penalty is not used, since the goal is randomized
        # total_reward -= 0.01

        # Distance improvement reward
        distance_improvement = 0.0
        if self.previous_distance is None:
            self.previous_distance = distance_to_goal
        else:
            distance_improvement = self.previous_distance - distance_to_goal
            self.previous_distance = distance_to_goal
        total_reward += distance_improvement * ENV_PARAMS["distance_improvement_weight"]

        # Stuck detection
        current_velocity = abs(self.last_cmd_vel[0]) if self.last_cmd_vel is not None else 0.0
        dwa_effectively_zero = abs(self.dwa_velocity[0]) < 0.01 and abs(self.dwa_velocity[1]) < 0.01
        if (abs(distance_improvement) < 0.03 and current_velocity < 0.05) or dwa_effectively_zero:
            self.steps_without_progress += 1
        else:
            self.steps_without_progress = max(0, self.steps_without_progress - 2)

        # Penalize idling
        if current_velocity < 0.05:
            total_reward += ENV_PARAMS["idle_penalty_weight"]

        # Obstacle penalty
        min_distance = np.min(self.current_scan) if self.current_scan is not None else self.laser_range_max
        if min_distance < self.obstacle_range_max:
            normalized_distance = (min_distance - self.obstacle_range_min) / (self.obstacle_range_max - self.obstacle_range_min)
            normalized_distance = np.clip(normalized_distance, 0.0, 1.0)
            obstacle_penalty = ENV_PARAMS["obstacle_penalty_weight"] * (1.0 - normalized_distance)
        else:
            obstacle_penalty = 0.0

        total_reward += obstacle_penalty

        # Collision detection
        if min_distance < self.collision_threshold:
            if not hasattr(self, 'collision_counter'):
                self.collision_counter = 0
            self.collision_counter += 1
        else:
            self.collision_counter = 0
        collision_detected = (self.collision_counter >= self.collision_confirmation_steps)

        # Debug info
        info["distance_to_goal"] = distance_to_goal
        info["progress_improvement"] = distance_improvement
        info["closest_obstacle_distance"] = min_distance
        info["robot_orientation_deg"] = np.degrees(self.robot_orientation)
        info["steps_stuck"] = self.steps_without_progress
        info["current_linear_velocity"] = current_velocity
        info["dwa_linear_velocity"] = self.dwa_velocity[0]
        info["dwa_angular_velocity"] = self.dwa_velocity[1]

        print(f"Distance to goal: {distance_to_goal:.2f}, "
            f"Progress: {distance_improvement:.4f}, "
            f"Closest obstacle: {min_distance:.2f}, "
            f"Robot angle: {np.degrees(self.robot_orientation):.1f}°, "
            f"Steps stuck: {self.steps_without_progress}, "
            f"Velocity: {current_velocity:.2f} m/s")

        if self.dwa_goal_reached and distance_to_goal > 0.5:
            rospy.logwarn("Ignoring DWA goal_reached (robot too far from goal).")
            self.dwa_goal_reached = False

        # Termination conditions
        if collision_detected and min_distance < 0.5:
            print("Episode ended: Collision detected!")
            total_reward += ENV_PARAMS["collision_penalty"]
            done = True
            info["reason"] = "collision"
            info["collision_distance"] = min_distance
            info["collision_steps"] = self.collision_counter

        if distance_to_goal < 0.3:
            print("Episode ended: Goal reached!")
            total_reward += ENV_PARAMS["goal_reward"] 
            done = True
            info["reason"] = "goal_reached"

        if self.steps_without_progress > 200:
            print("Episode ended: No progress detected for 200 steps!")
            done = True
            info["reason"] = "no_progress"

        return total_reward, done, info


    # Wait for robot localization
    def _wait_for_robot_localization(self, timeout=10.0, pos_tolerance=0.5, yaw_tolerance=np.radians(10)):
        rospy.loginfo("Waiting for robot localization after reset...")
        start_time = rospy.Time.now()
        
        while (rospy.Time.now() - start_time).to_sec() < timeout:
            try:
                self.tf_listener.waitForTransform("map", "base_link", rospy.Time(0), rospy.Duration(0.5))

                if self.robot_pose is not None:
                    current_x = self.robot_pose.position.x
                    current_y = self.robot_pose.position.y
                    current_yaw = self.robot_orientation
                    
                    dist_diff = np.sqrt((current_x - 0.0)**2 + (current_y - 0.0)**2)
                    yaw_diff = abs(current_yaw - (self.reset_yaw_radians + np.pi))
                    yaw_diff = min(yaw_diff, 2*np.pi - yaw_diff)
                    
                    if dist_diff < pos_tolerance and yaw_diff < yaw_tolerance:
                        rospy.loginfo(f"Robot localized successfully at ({current_x:.2f}, {current_y:.2f}, yaw={np.degrees(current_yaw):.1f} deg)")
                        return True 
                    else:
                        rospy.logwarn_throttle(2, f"Robot pose received but not yet close to target. Current: ({current_x:.2f}, {current_y:.2f}, yaw={np.degrees(current_yaw):.1f} deg). Target: (0.0, 0.0, yaw={np.degrees(self.reset_yaw_radians+np.pi):.1f} deg). Dist diff: {dist_diff:.2f}, Yaw diff: {np.degrees(yaw_diff):.1f} deg.")
                else:
                    rospy.logwarn_throttle(2, "Robot pose not yet received from odom_callback.")
                    
            except (tf.Exception, tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
                rospy.logwarn_throttle(2, f"Waiting for TF transform 'map' to 'base_link': {e}")
            
            self.rate.sleep()

        rospy.logerr(f"Robot localization timed out after {timeout} seconds.")
        return False


    # =================================
    # Gymnasium Function
    # =================================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset internal variables
        self.reset_internal_state()
        self.dwa_goal_reached = False
        self.has_goal = False 
        
        # Reset robot in Gazebo
        self.reset_robot_position()
        rospy.sleep(1.0) 
        
        # Wait for localization
        if not self._wait_for_robot_localization(timeout=15.0):
            rospy.logerr("Failed to localize robot after reset within timeout.")
        
        # Select a new random goal
        gazebo_goal = random.choice(self.goal_points_gazebo)
        rospy.loginfo(f"[Random Goal] Selected new goal: {gazebo_goal}")
        
        # Randomize obstacles 
        if self.static_obs:
            randomize_cylinders(self.reset_pos_x, self.reset_pos_y, 
                            gazebo_goal[0], gazebo_goal[1])
        
        # Set the new goal
        self.set_goal(x=gazebo_goal[0], y=gazebo_goal[1], gazebo_goal=True)
        
        # Reset DWA planner
        self.reset_dwa_planner()
        
        # Get initial observation
        observation = self.get_state()
        info = {"initial_reset": True}

        if self.robot_pose is None:
            rospy.logerr("Robot pose unavailable after reset.")
            observation = np.zeros(self.observation_space.shape, dtype=np.float32)
            info["reset_error"] = "no_robot_pose_after_reset"

        self.episode_start = True
        rospy.sleep(0.5) 
        
        return observation.astype(np.float32), info

    def step(self, action_idx):
        action_space_values = [
            (-0.25, 0), (0.25, 0), (0.5, 0), (0, 0),
            (0, -0.4), (0, -0.2), (0, 0.2), (0, 0.4)
        ]

        action = action_space_values[action_idx]

        # Move and compute reward
        observation, reward, done, info = self.move(action)
        self.rate.sleep()

        # Determine termination vs truncation
        reason = info.get("reason", "")
        terminated = reason in ["goal_reached", "collision"]
        truncated = reason  == "no_progress"

        # print(f"DDQN chose action index: {action_idx + 1}, value: {action}")
        return observation.astype(np.float32), reward, terminated, truncated, info