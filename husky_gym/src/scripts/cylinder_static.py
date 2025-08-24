#!/usr/bin/env python3

''' Code for static obstacle '''

import numpy as np
import random
import rospy

from gazebo_msgs.msg import LinkState
from gazebo_msgs.srv import SetLinkState
from geometry_msgs.msg import Pose

CYLINDER_RADIUS = 0.35
BUFFER_DISTANCE_FROM_START_GOAL = 3.0

def position_check(x, y, start_x, start_y, goal_x, goal_y, buffer_distance, cylinder_radius):
    start = np.array([start_x, start_y])
    goal = np.array([goal_x, goal_y])
    point = np.array([x, y])

    return (
        np.linalg.norm(point - start) < (buffer_distance + cylinder_radius)
        or np.linalg.norm(point - goal) < (buffer_distance + cylinder_radius)
    )

def create_link_state(name, x, y):
    state = LinkState()
    state.link_name = name

    state.pose = Pose()
    state.pose.position.x = x
    state.pose.position.y = y
    state.pose.position.z = 0.15
    state.pose.orientation.x = 0.0
    state.pose.orientation.y = 0.0
    state.pose.orientation.z = 0.0
    state.pose.orientation.w = 1.0
    return state

def randomize_cylinders(start_x, start_y, goal_x, goal_y, seed=None):
    rospy.loginfo("Randomizing cylinder positions...")

    if not rospy.core.is_initialized():
        rospy.init_node('cylinder_static_randomizer', anonymous=True)

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    else:
        random.seed()
        np.random.seed()

    rospy.wait_for_service('/gazebo/set_link_state')
    set_link_state = rospy.ServiceProxy('/gazebo/set_link_state', SetLinkState)

    rospy.sleep(1.0)

    x_min, x_max = goal_x, 8.0 #-10.0, 8.0
    y_min, y_max = 3.0, 7.0

    max_attempts = 50
    placed = [False, False, False]
    poses = [None, None]

    for _ in range(max_attempts):
        if not placed[0]:
            x1, y1 = random.uniform(x_min, x_max), random.uniform(y_min, y_max)
            if not position_check(x1, y1, start_x, start_y, goal_x, goal_y, BUFFER_DISTANCE_FROM_START_GOAL, CYLINDER_RADIUS):
                state1 = create_link_state("cylinders::cylinder1", x1, y1)
                try:
                    result = set_link_state(state1)
                    if result.success:
                        rospy.loginfo(f"Cylinder1 placed at: ({x1:.2f}, {y1:.2f})")
                        placed[0] = True
                        poses[0] = np.array([x1, y1])
                    else:
                        rospy.logerr(f"Failed to place cylinder1: {result.status_message}")
                except rospy.ServiceException as e:
                    rospy.logerr(f"Service error on cylinder1: {e}")

        if not placed[1]:
            x2, y2 = random.uniform(x_min, x_max), random.uniform(y_min, y_max)
            if not position_check(x2, y2, start_x, start_y, goal_x, goal_y, BUFFER_DISTANCE_FROM_START_GOAL, CYLINDER_RADIUS):
                if placed[0] and poses[0] is not None:
                    if np.linalg.norm(np.array([x2, y2]) - poses[0]) < (2 * CYLINDER_RADIUS + 2.5):
                        continue
                state2 = create_link_state("cylinders::cylinder2", x2, y2)
                try:
                    result = set_link_state(state2)
                    if result.success:
                        rospy.loginfo(f"Cylinder2 placed at: ({x2:.2f}, {y2:.2f})")
                        placed[1] = True
                        poses[1] = np.array([x2, y2])
                    else:
                        rospy.logerr(f"Failed to place cylinder2: {result.status_message}")
                except rospy.ServiceException as e:
                    rospy.logerr(f"Service error on cylinder2: {e}")

        if not placed[2]:
            x3, y3 = random.uniform(x_min, x_max), random.uniform(y_min, y_max)
            if not position_check(x3, y3, start_x, start_y, goal_x, goal_y, BUFFER_DISTANCE_FROM_START_GOAL, CYLINDER_RADIUS):
                too_close = False
                if placed[0] and poses[0] is not None:
                    if np.linalg.norm(np.array([x3, y3]) - poses[0]) < (2 * CYLINDER_RADIUS + 2.5):
                        too_close = True
                if placed[1] and poses[1] is not None:
                    if np.linalg.norm(np.array([x3, y3]) - poses[1]) < (2 * CYLINDER_RADIUS + 2.5):
                        too_close = True
                if too_close:
                    continue
                state3 = create_link_state("cylinders::cylinder3", x3, y3)
                try:
                    result = set_link_state(state3)
                    if result.success:
                        rospy.loginfo(f"Cylinder3 placed at: ({x3:.2f}, {y3:.2f})")
                        placed[2] = True
                    else:
                        rospy.logerr(f"Failed to place cylinder3: {result.status_message}")
                except rospy.ServiceException as e:
                    rospy.logerr(f"Service error on cylinder3: {e}")

        if all(placed):
            break

    if not all(placed):
        rospy.logwarn("Some cylinders couldn't be placed after multiple attempts.")

    rospy.sleep(0.5)

# if __name__ == '__main__':
#     try:
#         randomize_cylinders(9.0, 5.0, -9.0, 5.0) #157, 175, 343 
#     except rospy.ROSInterruptException:
#         pass
