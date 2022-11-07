import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.cv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2)
        self.cv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.cv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2)
        self.cv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2)

    def forward(self, obs):
        hidden = F.elu(self.cv1(obs))
        hidden = F.elu(self.cv2(hidden))
        hidden = F.elu(self.cv3(hidden))
        embedded_obs = F.elu(self.cv4(hidden)).reshape(hidden.size(0), -1)
        return embedded_obs

class Decoder(nn.Module):

    def __init__(self, state_dim):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(state_dim, 1024)
        self.dc1 = nn.ConvTranspose2d(1024, 128, kernel_size=5, stride=2)
        self.dc2 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2)
        self.dc3 = nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2)
        self.dc4 = nn.ConvTranspose2d(32, 3, kernel_size=6, stride=2)

    def forward(self, state):
        hidden = self.fc(state)
        hidden = hidden.view(hidden.size(0), 1024, 1, 1)
        hidden = F.elu(self.dc1(hidden))
        hidden = F.elu(self.dc2(hidden))
        hidden = F.elu(self.dc3(hidden))
        obs = self.dc4(hidden)
        return obs

class Transition(nn.Module):
    def __init__(self, state_dim, act_dim):
        super(Transition, self).__init__()
        self.fc1 = nn.Linear(state_dim + act_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 2 * state_dim)
        self.state_dim = state_dim

    def forward(self, state, action):
        hidden = torch.cat([state, action], dim=-1)
        hidden = F.elu(self.fc1(hidden))
        hidden = F.elu(self.fc2(hidden))
        hidden = F.elu(self.fc3(hidden))
        mean, pre_stddev = torch.split(hidden, self.state_dim, dim=-1)
        stddev = F.softplus(pre_stddev) + 1e-5
        mean = mean + state
        return Normal(mean, stddev)


class Posterior(nn.Module):
    def __init__(self, state_dim):
        super(Posterior, self).__init__()
        self.fc1 = nn.Linear(2 * state_dim + 1024, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 2 * state_dim)
        self.state_dim = state_dim

    def forward(self, prior_mean, prior_stddev, embedded_obs):
        hidden = torch.cat([prior_mean, prior_stddev, embedded_obs], dim=-1)
        hidden = F.elu(self.fc1(hidden))
        hidden = F.elu(self.fc2(hidden))
        hidden = F.elu(self.fc3(hidden))
        mean, pre_stddev = torch.split(hidden, self.state_dim, dim=-1)
        stddev = F.softplus(pre_stddev) + 1e-5
        return Normal(mean, stddev)

class Discriminator(nn.Module):
    def __init__(self, state_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(2 * state_dim, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, 1)

    def forward(self, state, goal):
        hidden = torch.cat([state, goal], dim=-1)
        hidden = F.elu(self.fc1(hidden))
        hidden = F.elu(self.fc2(hidden))
        out = torch.sigmoid(self.fc3(hidden))
        out = 1e-5 + (1. - 2e-5) * out
        return out

class Metric(nn.Module):
    def __init__(self, state_dim, metric_dim):
        super(Metric, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, metric_dim)

    def forward(self, state):
        hidden = F.elu(self.fc1(state))
        hidden = F.elu(self.fc2(hidden))
        out = self.fc3(hidden)
        return out
