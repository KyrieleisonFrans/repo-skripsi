#!/usr/bin/env python3

''' Dummy training for checking code. '''


import rospy
import torch.nn as nn

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

from config import DDQN_PARAMS, TOTAL_TIMESTEPS
from ddqn_sb3 import DoubleDQN
from gym_env import GymEnv 

def train_dummy_agent():
    # Initialized ROS Node
    if not rospy.core.is_initialized():
        rospy.init_node('husky_dummy_training_node', anonymous=True)

    rospy.loginfo("Starting dummy training...")

    # Env
    env = GymEnv()
    env = Monitor(env) 

    # Simple policy setting
    policy_kwargs = dict(
        net_arch=[],
        activation_fn=nn.ReLU,
    )

    # DDQN model
    model = DoubleDQN(
        DDQN_PARAMS["policy_type"],
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=DDQN_PARAMS["learning_rate"],
        buffer_size=DDQN_PARAMS["buffer_size"],
        learning_starts=DDQN_PARAMS["learning_starts"],
        batch_size=DDQN_PARAMS["batch_size"],
        tau=DDQN_PARAMS["tau"],
        gamma=DDQN_PARAMS["gamma"],
        train_freq=DDQN_PARAMS["train_freq"],
        gradient_steps=DDQN_PARAMS["gradient_steps"],
        exploration_fraction=DDQN_PARAMS["exploration_fraction"],
        exploration_initial_eps=DDQN_PARAMS["exploration_initial_eps"],
        exploration_final_eps=DDQN_PARAMS["exploration_final_eps"],
        max_grad_norm=DDQN_PARAMS["max_grad_norm"],
        tensorboard_log=dir_manager.project_dirs['tensorboard'],
        verbose=DDQN_PARAMS["verbose"],
    )

    # Training simulation
    try:
        model.learn(total_timesteps=1000, log_interval=10)
    except Exception as e:
        rospy.logwarn(f"Training dummy stopped: {e}")
    finally:
        env.close()
        rospy.loginfo("Dummy training done. No saved result")

if __name__ == "__main__":
    train_dummy_agent()
