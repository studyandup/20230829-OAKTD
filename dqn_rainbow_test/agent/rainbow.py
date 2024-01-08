import numpy as np
import random
import itertools
from collections import namedtuple, deque

from dqn_rainbow_test.model import NEURAL as nn
from dqn_rainbow_test.model.linear_layer import LinearLayer

import torch
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Agent(object):

    def __init__(self,
                 network,
                 state_shape,
                 feature_dim,
                 action_size,
                 seed,
                 lr,
                 lr_decay,
                 buffer_size,
                 batch_size,
                 update_size,
                 gamma,
                 ):

        self.state_shape = state_shape
        # print(state_shape)
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.gamma = gamma
        self.batch_size = batch_size
        self.update_size = update_size
        self.t = 0

        # Q-Network
        print(state_shape)
        self.feature_layer_local = nn[network](state_shape, feature_dim, seed).to(device)
        self.linear_layer_local = LinearLayer(feature_dim, action_size, seed).to(device)
        self.feature_layer_target = nn[network](state_shape, feature_dim, seed).to(device)
        self.linear_layer_target = LinearLayer(feature_dim, action_size, seed).to(device)
        self.optimizer = optim.Adam(
            itertools.chain(self.feature_layer_local.parameters(), self.linear_layer_local.parameters()), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=lr_decay)

        # Replay memory
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, seed)

    def step(self, state, action, reward, next_state, done, t, eps, next_action):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, next_action, done)

        self.learn(t, eps)
        # if self.t == 0:
        #     self.learn(t, eps)
        # self.t = (self.t + 1) % 10

    def act(self, state, eps=0.):

        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.feature_layer_local.eval()
        self.linear_layer_local.eval()
        with torch.no_grad():
            feature_values = self.feature_layer_local(state)
            action_values = self.linear_layer_local(feature_values)
        self.feature_layer_local.train()
        self.linear_layer_local.train()
        action_greedy = np.argmax(action_values.cpu().data.numpy())

        # Epsilon-greedy action selection
        if random.random() > eps:
            action = np.argmax(action_values.cpu().data.numpy())
        else:
            action = random.choice(np.arange(self.action_size))

        return action

    def learn(self, write_step=False, eps=0.):
        if len(self.memory) < self.batch_size:
            return
        experiences = self.memory.sample()

        self.feature_layer_local.train()
        self.linear_layer_local.train()
        states, actions, rewards, next_states, next_action, dones = experiences

        # Get max predicted Q values (for next states) from target model
        next_state_actions = self.linear_layer_local(self.feature_layer_local(next_states)).max(1)[1]
        
        Q_targets_next = self.linear_layer_target(self.feature_layer_target(next_states)).detach(). \
            gather(1, next_state_actions.unsqueeze(-1))
        # Compute Q targets for current states
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.linear_layer_local(self.feature_layer_local(states)).gather(1, actions)
        #print(Q_expected)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        # ------------------- update target network ------------------- #
        self.t = (self.t + 1) % self.update_size
        if self.t == 0:
            self.soft_update(self.feature_layer_local, self.feature_layer_target)
            self.soft_update(self.linear_layer_local, self.linear_layer_target)

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(local_param.data)

    def save_model(self, path):
        torch.save({'feature_layer': self.feature_layer_target.state_dict(),
                    'linear_layer': self.linear_layer_target.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict()},
                   path)
        print('checkpoint saved!')

    def load_checkpoint(self, path):
        model_CKPT = torch.load(path)
        self.feature_layer_local.load_state_dict(model_CKPT['feature_layer'])
        self.feature_layer_target.load_state_dict(model_CKPT['feature_layer'])
        self.linear_layer_local.load_state_dict(model_CKPT['linear_layer'])
        self.linear_layer_target.load_state_dict(model_CKPT['linear_layer'])
        print('loading checkpoint!')
        self.optimizer.load_state_dict(model_CKPT['optimizer'])
        self.scheduler.load_state_dict(model_CKPT['scheduler'])


class ReplayBuffer:

    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience",
                                     field_names=["state", "action", "reward", "next_state", "next_action",
                                                   "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, next_action, done):
        e = self.experience(state, action, reward, next_state, next_action,  done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.stack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.stack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        next_action = torch.from_numpy(np.vstack([e.next_action for e in experiences if e is not None])).float().to(
            device)

        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return states, actions, rewards, next_states, next_action, dones

    def __len__(self):
        return len(self.memory)
