#!/usr/bin/env python3

''' Code for checking environment compatibility with stable_baselines3 '''

import rospy

from stable_baselines3.common.env_checker import check_env

from gym_env import GymEnv


# Initialize ROS node
rospy.init_node('gym_env_check', anonymous=True)

# Create environment 
env = HuskyEnv()

# Check the environment
check_env(env, warn=True)