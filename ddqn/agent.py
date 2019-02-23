import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from ddqn.memory import ReplayMemory
from ddqn.models import *
from config import *

if DEVICE == 'cpu':
    device = torch.device(DEVICE)
else:
    if torch.cuda.is_available():
        try:
            device = torch.device(DEVICE)
        except:
            print('GPU not available')
            device = torch.device('cpu')
    else:
        print('GPU not available')
        device = torch.device('cpu')

class Agent():
    def __init__(self, action_size=ACTION_SIZE, epsilon=EPSILON_START, load_model=False):
        ### Exploration vs. Exploitation ###

        # Exploration Rate
        self.epsilon = epsilon          
        # Minimum Exploration Rate
        self.epsilon_min = EPSILON_MIN
        # Discount Factor
        self.discount_factor = DISCOUNT_FACTOR
        # Exploration Rate Decay
        self.explore_decay = (self.epsilon - self.epsilon_min) / EXPLORE_STEP
        # Number of controller combinations
        self.action_size = action_size

        ### DDQN ###        
        
        # Experience Memory
        self.memory = ReplayMemory()

        # Policy Net
        self.policy_net = DDQN()
        self.policy_net.to(device)
        # Target Net
        self.target_net = DDQN()
        self.target_net.to(device)

        # Optimizing method
        self.optimizer = optim.Adam(params=self.policy_net.parameters(), lr=LEARNING_RATE)

        # Equalize the networks
        self.update_target_net()

        # Load in pre-trained networks
        if load_model:
            self.policy_net = torch.load(MODEL_PATH)
            self.target_net = torch.load(MODEL_PATH)

    def update_target_net(self):
        # Unfreeze the target network
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            # Exploration
            action = random.choice(list(range(self.action_size)))
        else:
            # Exploitation
            # print(torch.Tensor(state).unsqueeze(0).size())
            with torch.no_grad():
                action = int(self.policy_net(torch.Tensor(state).unsqueeze(0).to(device)).max(1)[1])
        return action

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
        alive = torch.Tensor([int(i) for i in mini_batch[3]]).to(device)
        
        # Compute Q(s_t, a) - Q of the current state
        self.policy_net.zero_grad()
        pred_q = self.policy_net.forward(torch.Tensor(states).to(device))
        
        # Compute Q function of next state
        with torch.no_grad():
            next_q = self.target_net.forward(torch.Tensor(next_states).to(device))

        # Find maximum Q-value of action at next state from target net
        action_tensor = torch.Tensor(actions).view(-1, 1).long().to(device)
        taken_q = pred_q.gather(1, action_tensor).view(-1)[alive==1]
        with torch.no_grad():
            target_q = (self.discount_factor * torch.max(next_q, dim=1)[0] + torch.Tensor(rewards).to(device))[alive==1]

        # Compute the Huber Loss
        loss = F.smooth_l1_loss(taken_q, target_q)
        
        # Optimize the model 
        loss.backward()
        self.optimizer.step()

class StateAgent():
    def __init__(self, load_model=False):

        # Discount Factor
        self.discount_factor = DISCOUNT_FACTOR

        ### DDQN ###        
        
        # Sizes
        input_size = 64 * 34 * 64
        
        # Experience Memory
        self.memory = ReplayMemory()

        # Policy Net
        self.policy_net = StateDQN()
        self.policy_net.to(device)
        # Target Net
        self.target_net = StateDQN()
        self.target_net.to(device)

        # Optimizing method
        self.optimizer = optim.Adam(params=self.policy_net.parameters(), lr=LEARNING_RATE)

        # Equilize the networks
        self.update_target_net()

        # Load in pre-trained networks
        if load_model:
            self.policy_net = torch.load(STATE_DQN_PATH)
            self.target_net = torch.load(STATE_DQN_PATH)

    def update_target_net(self):
        # Unfreeze the target network
        self.target_net.load_state_dict(self.policy_net.state_dict())

#    def get_action(self, state):
#        if np.random.rand() <= self.epsilon:
#            # Exploration
#            action = random.choice(list(range(self.action_size)))
#        else:
#            # Exploitation
#            with torch.no_grad():
#                action = int(self.policy_net.forward(torch.Tensor(state).unsqueeze(0).to(device)).max(1)[1])
#                # a = torch.max(self.policy_net.forward(state_t.view(1, *tuple(state_t.size())).to(device)))[1]
#        return action

    def train_policy_net(self, frame):
        mini_batch = self.memory.sample_mini_batch(frame)
        mini_batch = np.array(mini_batch).transpose()

        # History is the state, action, reward history
        history = np.stack(mini_batch[0], axis=0)

        # States are 4 frames statcked together
        states = np.float32(history[:, :4, :, :]) / 255.

        # Rewards are the rewards we got from taking action a at state s
        rewards = list(mini_batch[2])

        # Next States are the states we wound up at by taking action a at state s
        next_states = np.float32(history[:, 1:, :, :]) / 255.

        # Done signifies if the game is over (1=Done & don't compute, 0=Playing & compute)
        alive = torch.Tensor([int(i) for i in mini_batch[3]]).to(device)
        
        # Compute Q(s_t, a) - Q of the current state
        self.policy_net.zero_grad()
        pred_q = self.policy_net.forward(torch.Tensor(states).to(device))
        taken_q = pred_q.view(-1)
        
        # Compute Q function of next state
        with torch.no_grad():
            next_q = self.target_net.forward(torch.Tensor(next_states).to(device))

        # Find maximum Q-value of action at next state from target net
        with torch.no_grad():
            target_q = (self.discount_factor * torch.max(next_q, dim=1)[0] + torch.Tensor(rewards).to(device))[alive==1]

        # Compute the Huber Loss
        loss = F.smooth_l1_loss(taken_q, target_q)
        
        # Optimize the model 
        loss.backward()
        self.optimizer.step()