import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np


# BUILD THE ACTOR AND CRITIC NETWORKS:
class ActorNetwork(nn.Module):
    def __init__(self, input_dims, n_actions, fc1_dims=400, fc2_dims=300):
        super(ActorNetwork, self).__init__()

        # Create the actor neural network
        self.l1 = nn.Linear(input_dims, fc1_dims)
        self.l2 = nn.Linear(fc1_dims, fc2_dims)
        self.mu_head = nn.Linear(fc2_dims, n_actions)
        self.log_std_head = nn.Linear(fc2_dims, n_actions)

    def forward(self, state):
        a = torch.tanh(self.l1(state))
        a = torch.tanh(self.l2(a))
        # a = torch.relu(self.l1(state))
        # a = torch.relu(self.l2(a))
        # print(f'Mean: {self.mu_head(a)}')
        # print(torch.exp(self.log_std_head(a)))
        mu = torch.tanh(self.mu_head(a))
        log_std = torch.tanh(self.log_std_head(a))
        std = torch.exp(log_std)
        # print(f'Mean: {mu}, STD: {std}')

        dist = Normal(mu, std)
        action = dist.sample()

        return action, dist


class Critic(nn.Module):
    def __init__(self, in_dim: int):
        """Initialize."""
        super(Critic, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(in_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 1)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""

        value = self.critic(state)

        return value


def init_weights(m):
    if type(m) in (nn.Linear, nn.Conv2d):
        nn.init.orthogonal_(m.weight.data, np.sqrt(float(2)))
        if m.bias is not None:
            m.bias.data.fill_(0)
