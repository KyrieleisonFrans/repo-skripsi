#!/usr/bin/env python3

''' Code for dynamic obstacle (linear movement) '''

import rospy

from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState, GetModelState


def cylinder_dynamic():
    rospy.init_node('cylinder_dynamic')
    
    x_start = -9.0
    y_start = 5.0
    z_start = 0.5
    
    rospy.wait_for_service('/gazebo/set_model_state')
    set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
    
    rate = rospy.Rate(10)
    speed = 0.1     
    direction = 1   
    distance = 15
    
    x_current = x_start
    
    while not rospy.is_shutdown():
        # Move forward/backward along X-axis
        x_current += direction * speed
        
        # Reverse direction when reaching limits
        if x_current > x_start + distance:
            direction = -1  # backward
        elif x_current < x_start:
            direction = 1    # forward
        
        state = ModelState()
        state.model_name = "cylinder_dynamic"
        state.pose.position.x = x_current
        state.pose.position.y = y_start  # Y no moving
        state.pose.position.z = z_start  # Z no moving
        state.pose.orientation.w = 1     # No rotation
        
        try:
            set_state(state)
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s", e)
        
        rate.sleep()

if __name__ == '__main__':
    try:
        cylinder_dynamic()
    except rospy.ROSInterruptException:
        pass

