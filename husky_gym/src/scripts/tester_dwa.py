#!/usr/bin/env python3

''' Code for DWA testing '''

import os
import pandas as pd
import rospy
import time

from gym_env import GymEnv

def test_dwa_only():
    # Initialize ROS node
    rospy.init_node('dwa_test_node', anonymous=True)
    rospy.loginfo("Testing DWA planner only with randomized goals...")

    results_dir = "/home/husky_ws/src/husky_gym/src/ddqn_husky/dwa_only"
    os.makedirs(results_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f"dwa_test_results_{timestamp}.csv")
    
    columns = ['episode', 'step', 'reward', 'termination_reason']
    pd.DataFrame(columns=columns).to_csv(results_file, index=False)
    
    # Create the environment
    env = GymEnv()
    
    rate = rospy.Rate(10) 
    episode = 0
    max_episodes = 1000 

    try:
        while not rospy.is_shutdown() and episode < max_episodes:
            episode += 1
            rospy.loginfo(f"Starting episode {episode}")
            
            observation, info = env.reset()
            
            step = 0
            terminated = False
            truncated = False
            episode_reward = 0.0
            
            while not rospy.is_shutdown() and not terminated and not truncated:
                step += 1
                
                # Let DWA take control by publishing zero action (action index 3 is (0,0))
                observation, reward, terminated, truncated, info = env.step(3)
                episode_reward += reward
                
                if terminated or truncated:
                    termination_reason = info.get('reason', 'unknown')
                    formatted_reward = float(f"{episode_reward:.4f}")
                    
                    new_row = {
                        'episode': episode,
                        'step': step,
                        'reward': formatted_reward,
                        'termination_reason': termination_reason
                    }
                    
                    pd.DataFrame([new_row]).to_csv(results_file, mode='a', header=False, index=False)
                    
                    rospy.loginfo(f"Episode {episode} finished in {step} steps: {termination_reason}")
                
                rate.sleep()

    except rospy.ROSInterruptException:
        rospy.loginfo("Test interrupted by ROS shutdown.")
    except Exception as e:
        rospy.logerr(f"Error during testing: {str(e)}")
    finally:
        results_df = pd.read_csv(results_file)
        # print("\nFinal Results:")
        # print(results_df.to_string(index=False))
        rospy.loginfo(f"Test completed. Results saved to {results_file}")

if __name__ == '__main__':
    test_dwa_only()