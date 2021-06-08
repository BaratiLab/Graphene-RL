import os
import gym
import math
import argparse
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
import time
from time import gmtime, strftime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from graphene_env import GrapheneEnv
from utils import episode_finished, episode_finished_dense
from model import DQN


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0, help="random seed")
args = parser.parse_args()
print(args)

seed = args.seed
np.random.seed(seed)
torch.random.manual_seed(seed)

start_time = time.time()
start_time_str = strftime("%m%d%H%M", gmtime())
save_dir = os.path.join('./save', start_time_str)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# env = gym.make('CartPole-v1')
env = GrapheneEnv(max_timesteps=80)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


BATCH_SIZE = 128
GAMMA = 1
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 80000
TARGET_UPDATE = 10
LEARNING_RATE = 0.0005

n_actions = env.action_space.n
n_states = env.observation_space.shape[0]

print('act dim:', n_actions)
print('obs dim:', n_states)

policy_net = DQN(n_states, n_actions)
target_net = DQN(n_states, n_actions)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
memory = ReplayMemory(10000)

steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], dtype=torch.long)


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


num_episodes = 2000
ep_reward = []
ep_flux = []
ep_rej = []
for i_episode in range(num_episodes):
    # Initialize the environment and state
    state = env.reset()
    state = torch.from_numpy(state).float().view(1, -1)
    acc_rew = 0.0
    for t in count():
        # Select and perform an action
        action = select_action(state)
        new_state, reward, done, _ = env.step(action.item())
        acc_rew += reward
        reward = torch.tensor([reward])

        # # Observe new state
        if not done:
            next_state = torch.from_numpy(new_state).float().view(1, -1)
        else:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        optimize_model()
        if done:
            # episode_durations.append(t + 1)
            ep_reward.append(acc_rew)
            flux, rej = env.get_flux_rej()
            ep_flux.append(flux)
            ep_rej.append(rej)
            print("Episode: {}, reward: {}".format(i_episode, acc_rew))
            # episode_finished(target_net, policy_net, env, i_episode, save_dir, start_time_str, ep_reward)
            episode_finished_dense(target_net, policy_net, env, i_episode, save_dir, start_time_str, ep_reward)
            break


    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())


torch.save(target_net.state_dict(), os.path.join(save_dir, 'target_net.ckpt'))
torch.save(policy_net.state_dict(), os.path.join(save_dir, 'policy_net.ckpt'))
np.save(os.path.join(save_dir, 'rew.npy'), np.array(ep_reward))
np.save(os.path.join(save_dir, 'flux.npy'), np.array(ep_flux))
np.save(os.path.join(save_dir, 'rej.npy'), np.array(ep_rej))

env.close()
