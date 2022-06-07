# from pathlib import Path
from collections import deque
import numpy as np
import torch
from torch import nn
from random import sample

# Using Double DQN
class DQN:
    """Double Deep Q Network, using network received as argument"""

    def __init__(self,
        dqn,

        state_dim, 
        action_dim, 
        save_dir, 

        exploration_rate = 1,
        exploration_rate_decay = 0.99999975,
        exploration_rate_min = 0.1,

        save_every = 5e5,
        gamma = 0.9,
        memory_size = 100000,
        batch_size = 32,
        lr = 0.00025,

        burnin = 1e4,
        learn_every = 3,
        sync_every = 1e4):
        """
        dqn (model): Q value predictor model

        state_dim (tuple): Dimensions of the input space
        action_dim (int): Dimensions of the output space
        save_dir (str): Path where model will be saved

        exploration_rate (float)
        exploration_rate_decay (float)
        exploration_rate_min (float)

        save_every (int)
        gamma (float)
        memory_size (int)
        batch_size (int)
        lr (float)

        burnin (int)
        learn_every (int)
        sync_every (int)
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir
        self.use_cuda = torch.cuda.is_available()

        # self.gnn = Meta(...)
        self.dqn = dqn(self.state_dim, self.action_dim).float()

        if self.use_cuda:
            self.dqn = self.dqn.to(device="cuda")

        # Hyperparameters
        self.exploration_rate = exploration_rate
        self.exploration_rate_decay = exploration_rate_decay # TODO: To CAPS to indicate constant
        self.exploration_rate_min = exploration_rate_min
        self.curr_step = 0

        self.save_every = save_every  # no. of experiences between saving target net

        self.gamma = gamma  # RL

        # Memory
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size

        # Optimization
        self.lr = lr
        self.optimizer = torch.optim.Adam(self.dqn.parameters(), lr=self.lr)
        self.loss_fn = nn.SmoothL1Loss()

        # Learning
        self.burnin = burnin  # min. experiences before training
        self.learn_every = learn_every  # no. of experiences between updates to Q_online
        self.sync_every = sync_every  # no. of experiences between Q_target & Q_online sync


    # --------------------------------------------------------------------------

    def act(self, obs, info):
        """
        Given a state, choose an epsilon-greedy action and update value of step.

        Inputs:
        - obs (dict): Observation returned by the environment
        - info (dict): Additional information about the observation

        Outputs:
        action_idx (int): An integer representing which action will perform
        """
        # EXPLORE
        if np.random.rand() < self.exploration_rate:
            action_idx = np.int64(np.random.choice(np.where(obs['available_actions'] == 1)[0]))

        # EXPLOIT
        else:
            if self.use_cuda:
                state = self._obs_to_tensor(obs, info).cuda()
            else:
                state = self._obs_to_tensor(obs, info)
                
            # state = state.unsqueeze(0)
            action_values = self.dqn(state, model="online")

            # Multiply by the mask to select only available actions 
            # TODO: Consider if passing the mask to the dqn
            # TODO: Moving tensor to CPU due to tranformation into numpy, maybe not optimal
            masked_action_values = action_values.detach().cpu().numpy() * obs['available_actions']
            # TODO: Previous array could have a value of 0 as the mask.
            # So it's needed to apply mask making sure previous value is not zero
            masked_action_values = np.ma.masked_equal(masked_action_values, 0, copy=False) # Remove zeroes
            
            # TODO: Change axis because action_values is 1D. Check if this is appropiate
            # action_idx = torch.argmax(action_values, axis=1).item()
            # action_idx = torch.argmax(action_values, axis=0).item()
            action_idx = np.argmax(masked_action_values, axis=0)

        # Decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # Increment step
        self.curr_step += 1

        return action_idx

    # --------------------------------------------------------------------------

    def _obs_to_tensor(self, obs, info):
        """ Transform observation (+ info) to tensor for memory storage """
        return torch.tensor([
            obs['current_node'],
            obs['target_node'],
            obs['fuel'][0],
            obs['seq_t'][0],
            obs['day_t'][0],
            obs['week_t'][0],
            (info['deadline'] - info['current_time']).total_seconds() // 3600 # Rounded hours
        ])

    def cache(self, obs, next_obs, info, next_info, action, reward, done):
        """
        Store the experience to self.memory (replay buffer)

        Inputs:
        - obs (dict),
          - current_node
          - target_node
          - fuel
          - seq_t
          - day_t
          - week_t
        - info (dict)
          - current_time ???
          - deadline ???
        #   - remaining_time (int): Difference between deadline and current_time (hours)
        - next_obs (dict),
        - action (int),
        - reward (float),
        - done (bool)
        """

        if self.use_cuda:
            state = self._obs_to_tensor(obs, info).cuda()
            next_state = self._obs_to_tensor(next_obs, next_info).cuda()

            action = torch.tensor([action]).cuda()
            reward = torch.tensor([reward]).cuda()
            done = torch.tensor([done]).cuda()
        else:
            state = self._obs_to_tensor(obs, info)
            next_state = self._obs_to_tensor(next_obs, next_info)

            action = torch.tensor([action])
            reward = torch.tensor([reward])
            done = torch.tensor([done])

        self.memory.append((state, next_state, action, reward, done,))

    # --------------------------------------------------------------------------

    def recall(self):
        """
        Retrieve a batch of experiences from memory
        """
        batch = sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    # --------------------------------------------------------------------------

    # TODO: state should be a tensor (X_t probably)
    def td_estimate(self, state, action):
        current_Q = self.dqn(state, model="online")[
            np.arange(0, self.batch_size), action
        ]  # Q_online(s,a)

        return current_Q

    # TODO: state should be a tensor (X_t probably)
    @torch.no_grad() # We don't propagete on theta target
    def td_target(self, reward, next_state, done):
        next_state_Q = self.dqn(next_state, model="online")
        best_action = torch.argmax(next_state_Q, axis=1)
        
        next_Q = self.dqn(next_state, model="target")[
            np.arange(0, self.batch_size), best_action
        ]

        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self.dqn.target.load_state_dict(self.dqn.online.state_dict())

    def learn(self):
        """Update online action value (Q) function with a batch of experiences"""
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        # Sample from memory
        state, next_state, action, reward, done = self.recall()

        # Get TD Estimate
        td_est = self.td_estimate(state, action)

        # Get TD Target
        td_tgt = self.td_target(reward, next_state, done)

        # Backpropagate loss through Q_online
        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), loss)

    # --------------------------------------------------------------------------

    def save(self):
        save_path = (
            self.save_dir / f"driver_net_{int(self.curr_step // self.save_every)}.chkpt"
        )
        torch.save(
            dict(model=self.dqn.state_dict(), exploration_rate=self.exploration_rate),
            save_path,
        )
        print(f"DriverNet saved to {save_path} at step {self.curr_step}")