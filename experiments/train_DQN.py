import imp
import torch
import gym
import numpy as np
import osmnx as ox
import datetime
from pathlib import Path

from models.DQN import DriverAgent
from models import DriverNet_v1 as DriverNet
from logger.DQNLogger import DQNLogger

# Set numpy seed
SEED = 12345
np.random.seed(SEED)

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

print(f"Using CUDA: {torch.cuda.is_available()}")

save_dir = Path("checkpoints") / datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)

# ------------------------------------------------------------------------------
# Load graph
graph_path = 'data/andalusia-edited-multigraph-numerical-fullyconnected.graphml'

G = ox.load_graphml(graph_path, 
                    node_dtypes={'type':int,
                                 'open_time': int,
                                 'close_time': int},
                    edge_dtypes={'speed': int})

# ------------------------------------------------------------------------------
# Set up environment
config = {
    'network': G,
    'max_fuel': 3000
}

env = gym.make('TruckRouting', config)

# ------------------------------------------------------------------------------
# Instantiate agent

options = {
    "dqn": DriverNet,

    "state_dim": (7,1), # TODO: Set input dimensions of the graph 
    "action_dim": env.action_space.n,
    "save_dir": save_dir,

    "exploration_rate": 1,
    "exploration_rate_decay": 0.99999975,
    "exploration_rate_min": 0.1,

    "save_every": 3e4,
    "gamma": 0.9,
    "memory_size": 100000,
    "batch_size": 32,
    "lr": 0.00025,

    "burnin": 1e4,
    "learn_every": 3,
    "sync_every": 1e3
}

agent = DriverAgent(**options)

# ------------------------------------------------------------------------------
# Set logger

description = f"""
First experiment not in notebook.
Base DDQN with no graph information, only agent status

------------------------------------------
OPTIONS

Seed: {SEED}
Cuda: {torch.cuda.is_available()}
{options}
"""
logger = DQNLogger(save_dir, description)

# ------------------------------------------------------------------------------
# Train

EPISODES = 1500
MAX_STEPS = 300

for e in range(EPISODES):

    # Define RL model
    obs, info = env.reset(return_info=True)

    done = False
    step = 0
    
    # Until game finished or enough steps taken during the episode
    while not done and step < MAX_STEPS:
        step += 1
        
        # Sample a random action from the entire action space
        # action = env.action_space_sample()

        # Run agent on the state and get action
        action = agent.act(obs, info)
        # print(f"Current node {obs['current_node']}")
        # print(f"Taking action {action}")

        # Perform action and get the new observation space
        next_obs, reward, done, next_info = env.step(action)

        # Remember
        agent.cache(obs, next_obs, info, next_info, action, reward, done)

        # Learn
        q, loss = agent.learn()

        # Logging
        logger.log_step(reward, loss, q)

        # Update state
        obs = next_obs
        info = next_info

    # --------------------------------------------------------------------------

    logger.log_episode() # env.render(mode="rgb_array")

    # Display results
    print(f"Ended episode {e+1}/{EPISODES} in {step} steps")
    print(f"\nGoal achieved? {done}\nFinal reward: {reward}")

    if e % 20 == 0:
        env.render()
        logger.record(episode=e, epsilon=agent.exploration_rate, step=agent.curr_step)

# ------------------------------------------------------------------------------

print("Training completed")