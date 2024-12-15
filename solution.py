import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium.utils import seeding
from utils import ReplayBuffer, get_env, run_episode


###############################################
# NEW: Ornstein-Uhlenbeck Noise Class
###############################################
class OUNoise:  # NEW
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2):  # NEW
        self.action_dim = action_dim  # NEW
        self.mu = mu  # NEW
        self.theta = theta  # NEW
        self.sigma = sigma  # NEW
        self.reset()  # NEW

    def reset(self):  # NEW
        self.x = np.ones(self.action_dim) * self.mu  # NEW

    def sample(self):  # NEW
        dx = self.theta * (self.mu - self.x) + self.sigma * np.random.randn(self.action_dim)  # NEW
        self.x = self.x + dx  # NEW
        return self.x  # NEW


class MLP(nn.Module):
    def __init__(self, sizes):
        super().__init__()
        layers = []
        for i, (in_size, out_size) in enumerate(zip(sizes[:-1], sizes[1:])):
            layers.append(nn.Linear(in_size, out_size))
            if i < len(sizes) - 2:
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)


class Critic(nn.Module):
    def __init__(self, obs_size, action_size, num_layers, num_units):
        super().__init__()
        self.net = MLP([obs_size + action_size] + ([num_units] * num_layers) + [1])
    
    def forward(self, x, a):
        inputs = torch.cat([x, a], dim=-1)
        value = self.net(inputs)
        return value


class Actor(nn.Module):
    def __init__(self, action_low, action_high, obs_size, action_size, num_layers, num_units):
        super().__init__()
        self.net = MLP([obs_size] + ([num_units] * num_layers) + [action_size])
        self.action_scale = (action_high - action_low) / 2
        self.action_bias = (action_high + action_low) / 2

    def forward(self, x):
        action = torch.tanh(self.net(x))
        return action * self.action_scale + self.action_bias


class Agent:

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    buffer_size: int = 50_000

    # Hyperparameters
    learning_rate = 1e-3
    num_layers = 2
    num_units = 256
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005  # NEW: soft update factor for target networks (wasn't in the original code)

    def __init__(self, env):

        self.obs_size = np.prod(env.observation_space.shape)
        self.action_size = np.prod(env.action_space.shape)
        self.action_low = torch.tensor(env.action_space.low, device=self.device).float()   # CHANGED: moved to device
        self.action_high = torch.tensor(env.action_space.high, device=self.device).float() # CHANGED: moved to device

        # Original actor and critic
        self.actor = Actor(self.action_low, self.action_high, self.obs_size, self.action_size,
                           self.num_layers, self.num_units).to(self.device)
        self.critic = Critic(self.obs_size, self.action_size, self.num_layers, self.num_units).to(self.device)

        ###############################################
        # NEW: Target networks for stable training
        ###############################################
        self.actor_target = Actor(self.action_low, self.action_high, self.obs_size, self.action_size,
                                  self.num_layers, self.num_units).to(self.device)  # NEW
        self.critic_target = Critic(self.obs_size, self.action_size, self.num_layers, self.num_units).to(self.device)  # NEW

        # NEW: Initialize target networks with the same weights as the main networks
        self.actor_target.load_state_dict(self.actor.state_dict())  # NEW
        self.critic_target.load_state_dict(self.critic.state_dict())  # NEW

        # Optimizers
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.learning_rate)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate)

        # Replay buffer
        self.buffer = ReplayBuffer(self.buffer_size, self.obs_size, self.action_size, self.device)

        ###############################################
        # NEW: OU Noise for better exploration
        ###############################################
        self.noise = OUNoise(self.action_size)  # NEW

    ###############################################
    # NEW: Soft update function for target networks
    ###############################################
    def soft_update(self, source, target, tau):  # NEW
        # Reason: softly blend target parameters with the source parameters for stability
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def train(self):
        # CHANGED: Check if we have enough samples to train
        if self.buffer.size < self.batch_size:  # NEW
            return  # NEW

        obs, action, next_obs, done, reward = self.buffer.sample(self.batch_size)

        # Removed unnecessary clone() calls. Using obs.detach() is fine, but generally 
        # not needed if we're just doing forward passes and not backward through these.
        done = done.unsqueeze(1)
        reward = reward.unsqueeze(1)

        with torch.no_grad():
            # Compute target actions using the target actor
            next_action = self.actor_target(next_obs)  # CHANGED: using target actor, was self.actor before
            # Compute target Q-values using the target critic
            target_q = self.critic_target(next_obs, next_action)  # CHANGED: using target critic
            y = reward + (1 - done) * self.gamma * target_q

        # Update critic
        q_values = self.critic(obs, action)
        critic_loss = nn.MSELoss()(q_values, y)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor using policy gradient
        self.actor_optimizer.zero_grad()
        actor_loss = -self.critic(obs, self.actor(obs)).mean()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft-update the target networks for stability
        self.soft_update(self.actor, self.actor_target, self.tau)   # NEW
        self.soft_update(self.critic, self.critic_target, self.tau) # NEW

    def get_action(self, obs, train):
        obs = torch.tensor(obs, device=self.device).float().unsqueeze(0)

        with torch.no_grad():
            action = self.actor(obs).squeeze(0).cpu().numpy()

        if train:
            # CHANGED: Use OU noise instead of simple Gaussian noise
            action = action + self.noise.sample()  # NEW

        return np.clip(action, self.action_low.cpu().numpy(), self.action_high.cpu().numpy())

    def store(self, transition):
        obs, action, reward, next_obs, terminated = transition
        self.buffer.store(obs, next_obs, action, reward, terminated)


if __name__ == '__main__':
    WARMUP_EPISODES = 10
    TRAIN_EPISODES = 50
    TEST_EPISODES = 300
    save_video = False
    verbose = True
    seeds = np.arange(10)

    start = time.time()
    print(f'Running public evaluation.') 
    test_returns = {k: [] for k in seeds}

    for seed in seeds:
        seed = int(seed)
        for fn in [random.seed, np.random.seed, torch.manual_seed]:
            fn(seed)
        torch.backends.cudnn.deterministic = True

        env = get_env()
        env.action_space.seed(seed)
        env.np_random, _ = seeding.np_random(seed)

        agent = Agent(env)

        for _ in range(WARMUP_EPISODES):
            run_episode(env, agent, mode='warmup', verbose=verbose, rec=False)

        for _ in range(TRAIN_EPISODES):
            run_episode(env, agent, mode='train', verbose=verbose, rec=False)

        for n_ep in range(TEST_EPISODES):
            video_rec = (save_video and n_ep == TEST_EPISODES - 1)
            with torch.no_grad():
                episode_return = run_episode(env, agent, mode='test', verbose=verbose, rec=video_rec)
            test_returns[seed].append(episode_return)

    avg_test_return = np.mean([np.mean(v) for v in test_returns.values()])
    within_seeds_deviation = np.mean([np.std(v) for v in test_returns.values()])
    across_seeds_deviation = np.std([np.mean(v) for v in test_returns.values()])
    print(f'Score for public evaluation: {avg_test_return}')
    print(f'Deviation within seeds: {within_seeds_deviation}')
    print(f'Deviation across seeds: {across_seeds_deviation}')

    print("Time :", (time.time() - start)/60, "min")
