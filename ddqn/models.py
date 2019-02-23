import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *

'''
class DQN(nn.Module):
    def __init__(self, action_size):
        super(ODQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc = nn.Linear(3136, 512)
        self.head = nn.Linear(512, action_size)

    def forward(self, x):
        x = nn.ReLU(self.bn1(self.conv1(x)))
        x = nn.ReLU(self.bn2(self.conv2(x)))
        x = nn.ReLU(self.bn3(self.conv3(x)))
        print(x.size())
        x = nn.ReLU(self.fc(x.view(x.size(0), -1)))
        return self.head(x)
'''

class DDQN(nn.Module):
    def __init__(self):
        super(DDQN, self).__init__()
        self.conv = ConvNet(4, 64)
        
        input_size = 64 * 34 * 64

        self.action_size = ACTION_SIZE

        self.state = StateNet(input_size)
        self.action = ActionNet(input_size)
        self.qnet = QNet()
        
    def forward(self, x):
        conv_out = self.conv(x)
#        print(conv_out.size())
        conv_out = conv_out.view(conv_out.size(0), -1)

        state_q = self.state(conv_out)
        state_q = state_q.expand(state_q.size(0), self.action_size)
        action_q = self.action(conv_out)
        
        q_values = state_q + (action_q - action_q.mean(1).unsqueeze(1).expand(-1, self.action_size))

        return q_values
            
class StateDQN(nn.Module):
    def __init__(self):
        super(StateDQN, self).__init__()
        self.conv = ConvNet(4, 64)
        
        input_size = 64 * 34 * 64

        self.state = StateNet(input_size, 1)
        
    def forward(self, x):
        conv_out = self.conv(x)

        conv_out = conv_out.view(conv_out.size(0), -1)

        state_q = self.state(conv_out)

        return state_q

class ConvNet(nn.Module):
    def __init__(self, input_channels=4, output_channels=64):
        super(ConvNet, self).__init__()
        self.model = nn.Sequential(
                nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
                nn.BatchNorm2d(32),
                nn.ReLU(),

                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.BatchNorm2d(64),
                nn.ReLU(),

                nn.Conv2d(64, output_channels, kernel_size=3, stride=1),
                nn.BatchNorm2d(output_channels),
                nn.ReLU())

    def forward(self, x):
        return self.model(x)        
    
class StateNet(nn.Module):
    def __init__(self, input_size, output_size=1):
        super(StateNet, self).__init__()
        self.model = nn.Sequential(
                nn.Linear(input_size, 512),
                nn.ReLU(),
                nn.Linear(512, output_size))
        
    def forward(self, x):
        return self.model(x)
    
class ActionNet(nn.Module):
    def __init__(self, input_size, output_size=10):
        super(ActionNet, self).__init__()
        self.model = nn.Sequential(
                nn.Linear(input_size, 512),
                nn.ReLU(),
                nn.Linear(512, output_size))
        
    def forward(self, x):
        return self.model(x)
    
class QNet(nn.Module):
    def __init__(self, input_size=1+10, output_size=10):
        super(QNet, self).__init__()
        self.model = nn.Sequential(
                nn.Linear(input_size, output_size))
        
    def forward(self, x):
        return self.model(x)
    
    
    