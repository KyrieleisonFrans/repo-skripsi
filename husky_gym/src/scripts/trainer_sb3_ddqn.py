#!/usr/bin/env python3

''' Code for DDQN Training '''

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import rospy
import torch as th
import torch.nn as nn

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, BaseCallback, CheckpointCallback, ProgressBarCallback
from stable_baselines3.common.env_util import make_vec_env 
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from config import DDQN_PARAMS, TOTAL_TIMESTEPS
from ddqn_sb3 import DoubleDQN
from gym_env import GymEnv 


class DirectoryManager:
    def __init__(self, base_ws_dir="/home/husky_ws/src/husky_gym/src/ddqn_husky/"):
        self.base_ws_dir = os.path.join(base_ws_dir, "ddqn_sb3")
        self.project_dirs = {}
        
    def setup_dir(self, project_name="ddqn_sb3"):
        os.makedirs(self.base_ws_dir, exist_ok=True)
    
        train_dir = os.path.join(self.base_ws_dir, "train")
        results_dir = os.path.join(train_dir, "results")
        plot_parent_dir = os.path.join(train_dir, "plot")
        model_parent_dir = os.path.join(train_dir, "model")
        tensorboard_parent_dir = os.path.join(train_dir, "tensorboard_logs")
        
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(plot_parent_dir, exist_ok=True)
        os.makedirs(model_parent_dir, exist_ok=True)
        os.makedirs(tensorboard_parent_dir, exist_ok=True)
        
        # Check for latest directory and if training is complete
        latest_dir, is_complete = self._check_latest_training_status(model_parent_dir, project_name)
        
        if latest_dir and not is_complete:
            tne_number = latest_dir.split('_')[-1]
            rospy.loginfo(f"Resuming training in existing directory: {latest_dir}")
        else:
            tne_number = self._get_next_tne_number(model_parent_dir, project_name)
            rospy.loginfo(f"Starting new training in directory: {project_name}_{tne_number}")
        
        self.project_dirs = {
            'results': results_dir,
            'plot': os.path.join(plot_parent_dir, f"{project_name}_{tne_number}"),
            'model': os.path.join(model_parent_dir, f"{project_name}_{tne_number}"),
            'tensorboard': os.path.join(tensorboard_parent_dir, f"{project_name}_{tne_number}"),
        }
        
        for path in self.project_dirs.values():
            os.makedirs(path, exist_ok=True)
            
        return self.project_dirs
    
    def _check_latest_training_status(self, parent_dir, base_name):
        dirs = []
        for d in os.listdir(parent_dir):
            if d.startswith(base_name):
                try:
                    dir_num = int(d.split('_')[-1])
                    dirs.append((dir_num, d))
                except (ValueError, IndexError):
                    continue
        
        if not dirs:
            return None, False
        
        dirs.sort()
        latest_dir_num, latest_dir = dirs[-1]
        latest_dir_path = os.path.join(parent_dir, latest_dir)
        
        final_model_path = os.path.join(latest_dir_path, "sb3_final_model.zip")
        if os.path.exists(final_model_path):
            return latest_dir, True
        
        checkpoint_path = os.path.join(latest_dir_path, f"husky_ddqn_checkpoint_300000.zip")
        if os.path.exists(checkpoint_path):
            return latest_dir, True
        
        eval_path = os.path.join(parent_dir.replace("model", "results"), "evaluations.npz")
        if os.path.exists(eval_path):
            try:
                data = np.load(eval_path)
                if len(data['timesteps']) > 0 and data['timesteps'][-1] >= TOTAL_TIMESTEPS:
                    return latest_dir, True
            except:
                pass
        
        return latest_dir, False
    
    def _get_next_tne_number(self, parent_dir, base_name):
        counter = 1
        while True:
            dir_name = f"{base_name}_{counter}"
            full_path = os.path.join(parent_dir, dir_name)
            if not os.path.exists(full_path):
                return counter
            counter += 1
    
    def _get_next_numbered_file(self, directory, base_name, extension):
        counter = 1
        while True:
            file_name = f"{base_name}_{counter}.{extension}"
            full_path = os.path.join(directory, file_name)
            if not os.path.exists(full_path):
                return full_path
            counter += 1

class EpisodeStopCallback(BaseCallback):
    def __init__(self, max_episodes=3000, verbose=0):
        super().__init__(verbose)
        self.max_episodes = max_episodes
        self.episode_count = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.last_buffer_length = 0 

    def _on_step(self) -> bool:
        if len(self.model.ep_info_buffer) > self.last_buffer_length:
            for i in range(self.last_buffer_length, len(self.model.ep_info_buffer)):
                episode_info = self.model.ep_info_buffer[i]
                if 'r' in episode_info:
                    self.episode_count += 1
                    self.episode_rewards.append(episode_info['r'])
                    self.episode_lengths.append(episode_info['l'])
                    
                    if self.episode_count >= self.max_episodes:
                        rospy.loginfo(f"EpisodeStopCallback: Maximum number of episodes ({self.max_episodes}) reached. Stopping training.")
                        return False
            
            self.last_buffer_length = len(self.model.ep_info_buffer)
            
        return True


class TimestepLoggerCallback(BaseCallback):
    def __init__(self, total_timesteps, verbose=1):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps

    def _on_step(self) -> bool:
        if self.num_timesteps % 1000 == 0:
            rospy.loginfo(f"Training progress: {self.num_timesteps}/{self.total_timesteps} timesteps")
        return True

def setup_ros_node():
    if not rospy.core.is_initialized():
        rospy.init_node('husky_ddqn_training_node', anonymous=True, log_level=rospy.INFO)
        rospy.loginfo("ROS training node 'husky_ddqn_training_node' initialized.")
    else:
        rospy.loginfo("ROS node already initialized. Continuing...")

def setup_model_and_callbacks(env, eval_env, dir_manager):
    policy_kwargs = dict(
        net_arch=[256, 256],    # 2 hidden layers 
        activation_fn=nn.ReLU,  # ReLU activation
        normalize_images=False
    )

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

    # Callbacks
    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=50.0, verbose=1)
    episode_stop_callback = EpisodeStopCallback(max_episodes=1000)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=dir_manager.project_dirs['model'], 
        log_path=dir_manager.project_dirs['results'],          
        eval_freq=5000,                    
        n_eval_episodes=20,                  
        deterministic=True,                 
        render=False,                       
        callback_after_eval=stop_callback,
        verbose=1
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=dir_manager.project_dirs['model'],
        name_prefix="husky_ddqn_checkpoint",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    return model, [eval_callback, episode_stop_callback, checkpoint_callback]


def train_ddqn_agent():
    # Setup Dir
    dir_manager = DirectoryManager()
    dirs = dir_manager.setup_dir(project_name="ddqn_sb3")

    # Log dir
    for name, path in dirs.items():
        rospy.loginfo(f"{name.capitalize()} Directory: {path}")

    # Setup ROS Node
    setup_ros_node()

    # Create environment
    env = GymEnv()
    eval_env = GymEnv()

    monitor_file = os.path.join(dirs['results'], "monitor.csv")
    env = Monitor(env, filename=monitor_file, info_keywords=('reason',))

    rospy.loginfo(f"HuskyEnv training environment created and wrapped with Monitor. Logs saved to: {monitor_file}")

    # Setup model and callbacks
    model, callbacks = setup_model_and_callbacks(env, eval_env, dir_manager)
    
    rospy.loginfo("DDQN agent initialized.")

    # Check for existing checkpoints
    checkpoint_path = None
    completed_timesteps = 0
    remaining_timesteps = TOTAL_TIMESTEPS

    model_dir = dirs['model']
    if os.path.exists(model_dir):
        checkpoints = []
        for f in os.listdir(model_dir):
            if f.startswith("husky_ddqn_checkpoint") and f.endswith(".zip"):
                try:
                    step_num = int(f.split("_")[-2])
                    checkpoints.append((step_num, f))
                except (ValueError, IndexError):
                    rospy.logwarn(f"Skipping malformed checkpoint file: {f}")
        
        if checkpoints:
            checkpoints.sort()
            checkpoint_path = os.path.join(model_dir, checkpoints[-1][1])
            completed_timesteps = checkpoints[-1][0]
            remaining_timesteps = max(0, TOTAL_TIMESTEPS - completed_timesteps)
    
    timestep_callback = TimestepLoggerCallback(total_timesteps=remaining_timesteps)
    callbacks.append(timestep_callback)

    if checkpoint_path: 
        try:
            rospy.loginfo(f"Loading checkpoint: {checkpoint_path}")
            model = DoubleDQN.load(checkpoint_path, env=env)
            initial_episode_count = completed_timesteps // 200
        except Exception as e:
            rospy.logerr(f"Failed to load checkpoint: {e}")
            rospy.loginfo("Starting fresh training instead")
            remaining_timesteps = TOTAL_TIMESTEPS

    # Train
    rospy.loginfo("Starting DDQN training...")
    try:
        model.learn(
            total_timesteps=remaining_timesteps,
            callback=callbacks,
            log_interval=4,
            reset_num_timesteps=False if checkpoint_path else True
        )
    except KeyboardInterrupt:
        rospy.loginfo("Training interrupted by user (KeyboardInterrupt).")
    except rospy.ROSInterruptException:
        rospy.loginfo("Training interrupted by ROS shutdown (ROSInterruptException).")
    finally:
        # Save final model
        final_model_path = os.path.join(dirs['model'], "sb3_final_model")
        model.save(final_model_path)
        rospy.loginfo(f"Final model saved to: {final_model_path}.zip")
        
        env.close()
        eval_env.close()
        
        # Generate plots
        plot_results(dirs['results'], dirs['plot'], 
                    episode_rewards=callbacks[1].episode_rewards,
                    episode_lengths=callbacks[1].episode_lengths)
        
        rospy.signal_shutdown("Training complete or interrupted.")



def plot_results(log_results_dir, plot_output_dir, episode_rewards=None, episode_lengths=None):
    rospy.loginfo("Generating plots from training and evaluation results...")
    # Plot Training Rewards and Lengths from Monitor CSV (Timestep-based)
    monitor_csv_path = os.path.join(log_results_dir, "monitor.csv")
    if os.path.exists(monitor_csv_path):
        try:
            df = pd.read_csv(monitor_csv_path, skiprows=1) 
            x, y = ts2xy(load_results(log_results_dir), 'timesteps')

            # Plot Episode Rewards (Timesteps)
            plt.figure(figsize=(12, 6))
            plt.plot(x, y)
            plt.xlabel("Timesteps")
            plt.ylabel("Episode Reward")
            plt.title("Training Episode Rewards over Timesteps (from Monitor)")
            plt.grid(True)
            plt.tight_layout()
            plot_path = os.path.join(plot_output_dir, "training_episode_rewards_timesteps.png")
            plt.savefig(plot_path)
            plt.close()
            rospy.loginfo(f"Saved training episode rewards (timesteps) plot to {plot_path}")

            # Plot Episode Lengths (Timesteps)
            plt.figure(figsize=(12, 6))
            plt.plot(x, df['l']) 
            plt.xlabel("Timesteps")
            plt.ylabel("Episode Length (Steps)")
            plt.title("Training Episode Lengths over Timesteps (from Monitor)")
            plt.grid(True)
            plt.tight_layout()
            plot_path = os.path.join(plot_output_dir, "training_episode_lengths_timesteps.png")
            plt.savefig(plot_path)
            plt.close()
            rospy.loginfo(f"Saved training episode lengths (timesteps) plot to {plot_path}")

        except Exception as e:
            rospy.logwarn(f"Could not load or plot monitor.csv: {e}")
    else:
        rospy.logwarn(f"monitor.csv not found at {monitor_csv_path}. Skipping timestep-based training plots.")

    # Plot Evaluation Rewards and Lengths from EvalCallback .npz
    eval_npz_path = os.path.join(log_results_dir, "evaluations.npz")
    if os.path.exists(eval_npz_path):
        try:
            data = np.load(eval_npz_path)
            eval_timesteps = data['timesteps']
            eval_rewards = data['results']
            eval_lengths = data['ep_lengths']

            mean_rewards = np.mean(eval_rewards, axis=1)
            std_rewards = np.std(eval_rewards, axis=1)
            mean_lengths = np.mean(eval_lengths, axis=1)
            std_lengths = np.std(eval_lengths, axis=1)

            plt.figure(figsize=(12, 6))
            plt.plot(eval_timesteps, mean_rewards, label="Mean Reward")

            plt.fill_between(eval_timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.2, label="Std Dev")
            plt.xlabel("Timesteps")
            plt.ylabel("Mean Episode Reward")
            plt.title("Evaluation Mean Episode Rewards over Timesteps")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plot_path = os.path.join(plot_output_dir, "evaluation_mean_rewards.png")
            plt.savefig(plot_path)
            plt.close()
            rospy.loginfo(f"Saved evaluation mean rewards plot to {plot_path}")

            # Plot 
            plt.figure(figsize=(12, 6))
            plt.plot(eval_timesteps, mean_lengths, label="Mean Length")
            plt.fill_between(eval_timesteps, mean_lengths - std_lengths, mean_lengths + std_lengths, alpha=0.2, label="Std Dev")
            plt.xlabel("Timesteps")
            plt.ylabel("Mean Episode Length (Steps)")
            plt.title("Evaluation Mean Episode Lengths over Timesteps")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plot_path = os.path.join(plot_output_dir, "evaluation_mean_lengths.png")
            plt.savefig(plot_path)
            plt.close()
            rospy.loginfo(f"Saved evaluation mean lengths plot to {plot_path}")

        except Exception as e:
            rospy.logwarn(f"Could not load or plot evaluations.npz: {e}")
    else:
        rospy.logwarn(f"evaluations.npz not found at {eval_npz_path}. Skipping evaluation plots.")

    rospy.loginfo("Plot generation complete.")

# Main execution block
if __name__ == '__main__':
    train_ddqn_agent()
