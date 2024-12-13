import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium.utils import seeding
from utils import ReplayBuffer, get_env, run_episode


class MLP(nn.Module):
    '''
    A simple ReLU MLP constructed from a list of layer widths.
    '''
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
    '''
    Simple MLP Q-function.
    '''
    def __init__(self, obs_size, action_size, num_layers, num_units):
        super().__init__()
        #####################################################################
        # TODO: add components as needed (if needed)

        self.net = MLP([obs_size + action_size] + ([num_units] * num_layers) + [1])

        #####################################################################

    def forward(self, x, a):
        #####################################################################
        # TODO: code the forward pass
        # the critic receives a batch of observations and a batch of actions
        # of shape (batch_size x obs_size) and batch_size x action_size) respectively
        # and output a batch of values of shape (batch_size x 1)

        # concatenates observations and the action take together
        inputs = torch.cat([x, a], dim=-1)
        value = self.net(inputs)

        #####################################################################
        return value


class Actor(nn.Module):
    '''
    Simple Tanh deterministic actor.
    '''
    def __init__(self, action_low, action_high,  obs_size, action_size, num_layers, num_units):
        super().__init__()
        #####################################################################
        # TODO: add components as needed (if needed)

        self.net = MLP([obs_size] + ([num_units] * num_layers) + [action_size])

        #####################################################################
        # store action scale and bias: the actor's output can be squashed to [-1, 1]
        self.action_scale = (action_high - action_low) / 2
        self.action_bias = (action_high + action_low) / 2

    def forward(self, x):
        #####################################################################
        # TODO: code the forward pass
        # the actor will receive a batch of observations of shape (batch_size x obs_size)
        # and output a batch of actions of shape (batch_size x action_size)

        action = torch.tanh(self.net(x))
        return action * self.action_scale + self.action_bias

        #####################################################################


class Agent:

    # automatically select compute device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    buffer_size: int = 50_000  # no need to change

    #########################################################################
    # TODO: store and tune hyperparameters here

    learning_rate = 1e-3
    num_layers = 2
    num_units = 256
    batch_size: int = 256
    gamma: float = 0.99  # MDP discount factor, 
    exploration_noise: float = 0.1  # epsilon for epsilon-greedy exploration
    
    #########################################################################

    def __init__(self, env):

        # extract informations from the environment
        self.obs_size = np.prod(env.observation_space.shape)  # size of observations
        self.action_size = np.prod(env.action_space.shape)  # size of actions
        # extract bounds of the action space
        self.action_low = torch.tensor(env.action_space.low).float()
        self.action_high = torch.tensor(env.action_space.high).float()

        #####################################################################
        # TODOish: initialize actor, critic and attributes

        self.critic = Critic(self.obs_size, self.action_size, self.num_layers, self.num_units).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.learning_rate)

        self.actor = Actor(self.action_low, self.action_high, self.obs_size, self.action_size, self.num_layers, self.num_units).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate)

        #####################################################################
        # create buffer
        self.buffer = ReplayBuffer(self.buffer_size, self.obs_size, self.action_size, self.device)
        self.train_step = 0
    
    def train(self):
        '''
        Updates actor and critic with one batch from the replay buffer.
        '''
        obs, action, next_obs, done, reward = self.buffer.sample(self.batch_size)

        #####################################################################
        # TODO: code training logic

        obs = obs.clone().detach().float()
        action = action.clone().detach().float()
        next_obs = next_obs.clone().detach().float()
        done = done.clone().detach().float().unsqueeze(1)
        reward = reward.clone().detach().float().unsqueeze(1)

        with torch.no_grad():
            next_action = self.actor.forward(next_obs)
            target_q_values = self.critic.forward(next_obs,next_action)
            target_q_values = reward + (1 - done) * self.gamma * target_q_values

        current_q_values = self.critic(obs,action)
        
        critic_loss = nn.MSELoss()(current_q_values, target_q_values)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update Actor
        self.actor_optimizer.zero_grad()

        actor_loss = -self.critic(obs, self.actor(obs)).mean()
        actor_loss.backward()
        self.actor_optimizer.step()



        #####################################################################

    def get_action(self, obs, train):
        '''
        Returns the agent's action for a given observation.
        The train parameter can be used to control stochastic behavior.
        '''
        #####################################################################
        # TODO: return the agent's action for an observation (np.array
        # of shape (obs_size, )). The action should be a np.array of
        # shape (act_size, )
        #####################################################################

        obs = torch.tensor(obs, device=self.device).float().unsqueeze(0)


        with torch.no_grad():
            # gets action from actor - squeeze reshapes - cpu changes to run on cpu
            action = self.actor(obs).squeeze(0).cpu().numpy()

        # if we are training, we want to add random noise
        if train:
            # exploration noise is from epsilon greedy
            action += self.exploration_noise * np.random.randn(self.action_size)
        
        # makes sure action is within bounds
        return np.clip(action, self.action_low.cpu().numpy(), self.action_high.cpu().numpy())


    def store(self, transition):
        '''
        Stores the observed transition in a replay buffer containing all past memories.
        '''
        obs, action, reward, next_obs, terminated = transition
        self.buffer.store(obs, next_obs, action, reward, terminated)


# This main function is provided here to enable some basic testing. 
# ANY changes here WON'T take any effect while grading.
if __name__ == '__main__':

    WARMUP_EPISODES = 10  # initial episodes of uniform exploration
    TRAIN_EPISODES = 50  # interactive episodes
    TEST_EPISODES = 300  # evaluation episodes
    save_video = False
    verbose = True
    seeds = np.arange(10)  # seeds for public evaluation

    start = time.time()
    print(f'Running public evaluation.') 
    test_returns = {k: [] for k in seeds}

    for seed in seeds:

        # seeding to ensure determinism
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
            video_rec = (save_video and n_ep == TEST_EPISODES - 1)  # only record last episode
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
