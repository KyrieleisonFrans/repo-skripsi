#!/usr/bin/env python3

''' Code for dynamic obstacle (circular movement) '''

import math
import rospy

from gazebo_msgs.msg import LinkState
from gazebo_msgs.srv import SetLinkState



def cylinder_dynamic():
    rospy.init_node('cylinder_dynamic')
    rospy.wait_for_service('/gazebo/set_link_state')
    set_link_state = rospy.ServiceProxy('/gazebo/set_link_state', SetLinkState)
    
    rate = rospy.Rate(30)
    
    # Cylinder parameters
    cylinder_radius = 2.0           # Circle size
    angular_speed = 0.5    # Radians/sec
    
    # Center positions for each cylinder (X, Y, Z)
    center1 = (2.0, 5.0, 0.15)  # Pink cylinder's circle center
    center2 = (-4.0, 5.0, 0.15) # Blue cylinder's circle center
    
    while not rospy.is_shutdown():
        t = rospy.get_time()
        
        # Cylinder 1 (Pink): Circular motion around center1
        state1 = LinkState()
        state1.link_name = "cylinders::cylinder1"
        state1.pose.position.x = center1[0] + cylinder_radius * math.cos(angular_speed * t)
        state1.pose.position.y = center1[1] + cylinder_radius * math.sin(angular_speed * t)
        state1.pose.position.z = center1[2]  # Z fixed
        state1.pose.orientation.w = 1.0      # No rotation
        
        # Cylinder 2 (Blue): Circular motion around center2 (90° offset)
        state2 = LinkState()
        state2.link_name = "cylinders::cylinder2"
        state2.pose.position.x = center2[0] + cylinder_radius * math.cos(angular_speed * t + math.pi/2)
        state2.pose.position.y = center2[1] + cylinder_radius * math.sin(angular_speed * t + math.pi/2)
        state2.pose.position.z = center2[2]  
        state2.pose.orientation.w = 1.0
        
        try:
            set_link_state(state1)
            set_link_state(state2)
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
        
        rate.sleep()

if __name__ == '__main__':
    try:
        cylinder_dynamic()
    except rospy.ROSInterruptException:
        pass