#!/usr/bin/env python3

'''Code for DDQN testing'''

import csv
import json
import matplotlib.pyplot as plt
import os
import numpy as np
import rospy
import signal

from gym import Wrapper
from stable_baselines3 import DQN

from gym_env import GymEnv

# Configuration
TEST_DIR = "/home/husky_ws/src/husky_gym/src/ddqn_husky/ddqn_sb3/test_8"
MODEL_PATH = "/home/husky_ws/src/husky_gym/src/ddqn_husky/ddqn_sb3/train/model/ddqn_sb3_8/sb3_final_model.zip"
NUM_EPISODES = 1000
RENDER = False

class SB3CompatibleWrapper(Wrapper):
    '''convert Gymnasium API (5-tuple) to SB3 API (4-tuple) '''
    def __init__(self, env):
        super().__init__(env)
        self._last_info = {} 
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_info = info
        return obs 
        
    def step(self, action):
        if isinstance(action, np.ndarray):
            action = int(action.item())
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._last_info = info
        done = terminated or truncated
        return obs, reward, done, info
        
    def get_last_info(self):
        return self._last_info

def setup_test_directory():
    os.makedirs(TEST_DIR, exist_ok=True)
    rospy.loginfo(f"Test results will be saved to: {TEST_DIR}")

def save_results(data, filename):
    path = os.path.join(TEST_DIR, filename)
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
    rospy.loginfo(f"Saved results to {path}")

def save_to_csv(results, filename):
    path = os.path.join(TEST_DIR, filename)
    with open(path, 'w', newline='') as csvfile:
        fieldnames = ['episode', 'reward', 'length', 'termination_reason', 'success']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for i, episode in enumerate(results['episodes']):
            reason = episode['termination_reason'] or episode['truncation_reason']
            success = 1 if reason == 'goal_reached' else 0
            writer.writerow({
                'episode': i+1,
                'reward': episode['reward'],
                'length': episode['steps'],
                'termination_reason': reason,
                'success': success
            })
    rospy.loginfo(f"Saved CSV data to {path}")

def save_plots(fig, filename):
    path = os.path.join(TEST_DIR, filename)
    fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    rospy.loginfo(f"Saved plot to {path}")

class GracefulInterruptHandler:
    def __init__(self, sig=signal.SIGINT):
        self.sig = sig
        self.interrupted = False
        self.released = False
        self.original_handler = None
        
    def __enter__(self):
        self.original_handler = signal.getsignal(self.sig)
        
        def handler(signum, frame):
            self.release()
            self.interrupted = True
            
        signal.signal(self.sig, handler)
        return self
        
    def __exit__(self, type, value, tb):
        self.release()
        
    def release(self):
        if self.released:
            return False
            
        signal.signal(self.sig, self.original_handler)
        self.released = True
        return True

def evaluate_model(model, env, results):
    with GracefulInterruptHandler() as h:
        for i in range(len(results['episodes']), NUM_EPISODES):
            if h.interrupted:
                rospy.loginfo("Interrupt detected, saving current results...")
                break
                
            obs = env.reset()
            obs = np.array(obs, dtype=np.float32).flatten()
            
            done = False
            episode = {
                'reward': 0, 
                'steps': 0, 
                'termination_reason': None,
                'truncation_reason': None,
                'actions': [],
                'positions': []
            }
            
            while not done and not h.interrupted:
                if not isinstance(obs, np.ndarray):
                    obs = np.array(obs, dtype=np.float32)
                if len(obs.shape) == 1:
                    obs = obs.reshape(1, -1)
                    
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                
                # Record episode data
                episode['reward'] += reward
                episode['steps'] += 1
                episode['actions'].append(int(action))
                
                if hasattr(env.env, 'robot_pose') and env.env.robot_pose is not None:
                    episode['positions'].append([
                        float(env.env.robot_pose.position.x),
                        float(env.env.robot_pose.position.y)
                    ])
                    
                if RENDER:
                    env.render()
                    
                if done:
                    # Get the termination/truncation reason
                    final_info = env.get_last_info()
                    episode['termination_reason'] = final_info.get('reason', None)
                    episode['truncation_reason'] = 'timeout' if 'no_progress' in str(final_info.get('reason', '')) else None
                    episode['final_info'] = final_info
            
            results['episodes'].append(episode)
            rospy.loginfo(
                f"Episode {i+1}/{NUM_EPISODES}: "
                f"Reward={episode['reward']:.2f}, "
                f"Steps={episode['steps']}, "
                f"Termination={episode['termination_reason']}, "
                f"Truncation={episode['truncation_reason']}"
            )
            
            # Save results after each episode
            save_results(results, 'evaluation_results.json')
            save_to_csv(results, 'episode_data.csv')
    
    return results

def generate_plots(results):
    if not results['episodes']:
        return

    rewards = [e['reward'] for e in results['episodes']]
    steps = [e['steps'] for e in results['episodes']]
    
    termination_reasons = {
        'goal_reached': 0,
        'collision': 0,
        'timeout': 0,
        'other': 0
    }
    
    for episode in results['episodes']:
        reason = episode['termination_reason'] or episode['truncation_reason']
        if reason == 'goal_reached':
            termination_reasons['goal_reached'] += 1
        elif reason == 'collision':
            termination_reasons['collision'] += 1
        elif reason == 'no_progress' or episode['truncation_reason'] == 'timeout':
            termination_reasons['timeout'] += 1
        else:
            termination_reasons['other'] += 1
    
    num_episodes = len(results['episodes'])
    results['summary'] = {
        'mean_reward': float(np.mean(rewards)) if num_episodes > 0 else 0,
        'std_reward': float(np.std(rewards)) if num_episodes > 0 else 0,
        'min_reward': float(np.min(rewards)) if num_episodes > 0 else 0,
        'max_reward': float(np.max(rewards)) if num_episodes > 0 else 0,
        'mean_steps': float(np.mean(steps)) if num_episodes > 0 else 0,
        'std_steps': float(np.std(steps)) if num_episodes > 0 else 0,
        'success_rate': float(termination_reasons['goal_reached'] / num_episodes * 100) if num_episodes > 0 else 0,
        'collision_rate': float(termination_reasons['collision'] / num_episodes * 100) if num_episodes > 0 else 0,
        'timeout_rate': float(termination_reasons['timeout'] / num_episodes * 100) if num_episodes > 0 else 0,
        'other_rate': float(termination_reasons['other'] / num_episodes * 100) if num_episodes > 0 else 0
    }
    
    # Calculate success rate over episodes
    success_rates = []
    success_count = 0
    for i, episode in enumerate(results['episodes']):
        reason = episode['termination_reason'] or episode['truncation_reason']
        if reason == 'goal_reached':
            success_count += 1
        success_rates.append(success_count / (i + 1) * 100)
    
    # Plot styling
    plt.style.use('seaborn')
    colors = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2']
    
    # Combined Reward and Steps Plot
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Reward plot
    ax1.plot(rewards, 'o-', color=colors[0], label='Episode Reward')
    ax1.axhline(results['summary']['mean_reward'], color=colors[1], linestyle='--', 
               label=f'Mean: {results["summary"]["mean_reward"]:.2f}')
    ax1.fill_between(
        range(len(rewards)),
        results['summary']['mean_reward'] - results['summary']['std_reward'],
        results['summary']['mean_reward'] + results['summary']['std_reward'],
        color=colors[1], alpha=0.1
    )
    ax1.set_title('Episode Rewards with Standard Deviation', pad=20)
    ax1.set_ylabel('Reward')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Steps plot
    ax2.plot(steps, 'o-', color=colors[2], label='Episode Steps')
    ax2.axhline(results['summary']['mean_steps'], color=colors[3], linestyle='--', 
               label=f'Mean: {results["summary"]["mean_steps"]:.1f}')
    ax2.fill_between(
        range(len(steps)),
        results['summary']['mean_steps'] - results['summary']['std_steps'],
        results['summary']['mean_steps'] + results['summary']['std_steps'],
        color=colors[3], alpha=0.1
    )
    ax2.set_title('Episode Steps with Standard Deviation', pad=20)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_plots(fig1, 'combined_metrics.png')
    
    # Success Rate Over Episodes
    if len(success_rates) > 0:
        fig2, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(1, num_episodes+1), success_rates, 'o-', color=colors[0], label='Success Rate')
        ax.axhline(results['summary']['success_rate'], color=colors[1], linestyle='--', 
                  label=f'Final: {results["summary"]["success_rate"]:.1f}%')
        ax.set_title('Success Rate Over Episodes', pad=20)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Success Rate (%)')
        ax.set_ylim(0, 110)
        ax.legend()
        ax.grid(True, alpha=0.3)
        save_plots(fig2, 'success_rate.png')
    
    # Reward Distribution
    fig3, ax = plt.subplots(figsize=(10, 6))
    ax.hist(rewards, bins=15, color=colors[0], edgecolor='white', alpha=0.7)
    ax.set_title('Reward Distribution Across Episodes', pad=20)
    ax.set_xlabel('Reward')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)
    save_plots(fig3, 'reward_distribution.png')

def main():
    rospy.init_node('husky_dqn_tester')
    setup_test_directory()
    
    # Initialize results structure
    results = {
        'episodes': [],
        'summary': {},
        'config': {
            'model_path': MODEL_PATH,
            'num_episodes': NUM_EPISODES,
            'test_timestamp': rospy.get_time()
        }
    }
    
    try:
        # Load model
        model = DQN.load(MODEL_PATH)
        rospy.loginfo(f"Successfully loaded model from {MODEL_PATH}")
        
        # Create and wrap environment
        original_env = GymEnv()
        env = SB3CompatibleWrapper(original_env)
        rospy.loginfo("SB3-compatible environment initialized")
        
        # Run evaluation
        results = evaluate_model(model, env, results)
        
        # Generate summary statistics
        generate_plots(results)
        
        # Save final results
        save_results(results, 'evaluation_results.json')
        save_to_csv(results, 'episode_data.csv')
        
        # Print summary
        if results['episodes']:
            rospy.loginfo("\n" + "="*50)
            rospy.loginfo("=== Evaluation Summary ===".center(50))
            rospy.loginfo("="*50)
            rospy.loginfo(f"Mean Reward: {results['summary']['mean_reward']:.2f} ± {results['summary']['std_reward']:.2f}")
            rospy.loginfo(f"Reward Range: [{results['summary']['min_reward']:.2f}, {results['summary']['max_reward']:.2f}]")
            rospy.loginfo(f"Mean Steps: {results['summary']['mean_steps']:.1f} ± {results['summary']['std_steps']:.1f}")
            rospy.loginfo("-"*50)
            rospy.loginfo(f"Success Rate: {results['summary']['success_rate']:.1f}%")
            rospy.loginfo(f"Collision Rate: {results['summary']['collision_rate']:.1f}%")
            rospy.loginfo(f"Timeout Rate: {results['summary']['timeout_rate']:.1f}%")
            rospy.loginfo(f"Other Termination: {results['summary']['other_rate']:.1f}%")
            rospy.loginfo("="*50)
        
    except Exception as e:
        rospy.logerr(f"Error during testing: {str(e)}")
        # Save results 
        if results['episodes']:
            save_results(results, 'evaluation_results.json')
            save_to_csv(results, 'episode_data.csv')
        raise
    finally:
        try:
            env.close()
        except:
            pass
        rospy.signal_shutdown("Testing completed")

if __name__ == '__main__':
    main()