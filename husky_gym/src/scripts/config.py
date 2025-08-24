''' Model hyperparameter & reward shaping '''

# Parameter Training
TOTAL_TIMESTEPS = 500000

# Parameter model DQN
DQN_PARAMS = {
    "policy_type": "MlpPolicy",
    "learning_rate": 0.0003,
    "buffer_size": 50000,
    "learning_starts": 5000,
    "batch_size": 128,
    "tau": 0.005,
    "gamma": 0.99,
    "train_freq": 4,
    "gradient_steps": 2,
    "exploration_fraction": 0.5,
    "exploration_initial_eps": 1.0,
    "exploration_final_eps": 0.05,
    "max_grad_norm": 10,
    "verbose": 1,
}

# Parameter model DDQN
DDQN_PARAMS = {
    "policy_type": "MlpPolicy",
    "learning_rate": 0.0003,
    "buffer_size": 50000,
    "learning_starts": 5000,
    "batch_size": 128,
    "tau": 0.005,
    "gamma": 0.99,
    "train_freq": 4,
    "gradient_steps": 2,
    "exploration_fraction": 0.5,
    "exploration_initial_eps": 1.0,
    "exploration_final_eps": 0.05,
    "max_grad_norm": 10,
    "verbose": 1,
}

# Parameter Env
ENV_PARAMS = {
    "static_obs": False,                    # True jika lingkungan statis, False jika lingkungan dinamis
    "time_weight":-0.0,                     # Penalti waktu
    "distance_improvement_weight": 0.5,     # Reward semakin dekat dengan goal
    "idle_penalty_weight":-0.02,            # Penalti diam
    "obstacle_penalty_weight": -0.1,        # Penalti mendekati halangan 
    "collision_penalty": -0.8,              # Penalti tabrakan
    "goal_reward": 1.0,                     # Reward mencapai goal
}