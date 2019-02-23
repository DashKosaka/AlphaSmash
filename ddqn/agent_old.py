import random
import numpy as np
from collections import deque
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from memory import ReplayMemory
from model import *
from utils import *
from config import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    def __init__(self, action_size, epsilon=1.0, load_model=False):
        self.load_model = load_model

        self.action_size = action_size

        # These are hyper parameters for the DQN
        self.discount_factor = 0.99
        # self.epsilon = 0.0
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.explore_step = 1000000
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / self.explore_step
        self.train_start = 100000
        self.update_target = 1000

        # Generate the memory
        self.memory = ReplayMemory()

        # Create the policy net and the target net
        self.policy_net = DQN(action_size)
        self.policy_net.to(device)
        self.target_net = DQN(action_size)
        self.target_net.to(device)

        self.optimizer = optim.Adam(params=self.policy_net.parameters(), lr=learning_rate)

        # initialize target net
        self.update_target_net()

        if self.load_model:
            self.policy_net = torch.load('save_model/breakout_dqn')
            self.target_net = torch.load('save_model/breakout_dqn')

    # after some time interval update the target net to be same with policy net
    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    """Get action using policy net using epsilon-greedy policy"""
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            ### CODE ####
            # Choose a random action
            a = random.choice(list(range(self.action_size)))
        else:
            ### CODE ####
            # Obtain from policy_net
            # state_t = torch.Tensor(state)
            # print(state_t.size())
            print('forward network')
            with torch.no_grad():
                a = int(self.policy_net.forward(torch.Tensor(state).unsqueeze(0).to(device)).max(1)[1])
                # a = torch.max(self.policy_net.forward(state_t.view(1, *tuple(state_t.size())).to(device)))[1]
        return a

    # pick samples randomly from replay memory (with batch_size)
    def train_policy_net(self, frame):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

        mini_batch = self.memory.sample_mini_batch(frame)
        mini_batch = np.array(mini_batch).transpose()

        # History is the state, action, reward history
        history = np.stack(mini_batch[0], axis=0)

        # States are 4 frames statcked together
        states = np.float32(history[:, :4, :, :]) / 255.
        # Actions are a list of ints that were taken
        actions = list(mini_batch[1])

        # Rewards are the rewards we got from taking action a at state s
        rewards = list(mini_batch[2])
        # Next States are the states we wound up at by taking action a at state s
        next_states = np.float32(history[:, 1:, :, :]) / 255.

        # Done signifies if the game is over (1=Done & don't compute, 0=Playing & compute)
        # print(mini_batch[3])
        dones = torch.Tensor([int(i) for i in mini_batch[3]]).to(device)
        
        # Compute Q(s_t, a) - Q of the current state
        ### CODE ####
        self.policy_net.zero_grad()
        pred_q = self.policy_net.forward(torch.Tensor(states).to(device))
        
        # Compute Q function of next state
        ### CODE ####
        # with torch.no_grad():
        #     next_q = self.policy_net.forward(torch.Tensor(next_states).to(device))
        with torch.no_grad():
            next_q = self.target_net.forward(torch.Tensor(next_states).to(device))

        # Find maximum Q-value of action at next state from target net
        ### CODE ####
        # print(actions)
        # taken_q = pred_q.view(3, -1)[torch.Tensor(actions).long().to(device)]
        action_tensor = torch.Tensor(actions).view(-1, 1).long().to(device)
        taken_q = pred_q.gather(1, action_tensor).view(-1)[dones==0]
        with torch.no_grad():
            target_q = (self.discount_factor * torch.max(next_q, dim=1)[0] + torch.Tensor(rewards).to(device))[dones==0]

        # Compute the Huber Loss
        ### CODE ####
        loss = F.smooth_l1_loss(taken_q, target_q)
        
        # Optimize the model 
        ### CODE ####
        loss.backward()
        self.optimizer.step()


