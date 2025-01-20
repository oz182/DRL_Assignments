import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
import numpy as np

class ProgressiveNetwork(nn.Module):
    def __init__ (self, input_dim, output_dim, pretrained_weights_paths):
        super(ProgressiveNetwork, self).__init__()

        self.columns = nn.ModuleList()
        self.lateral_connections = nn.ModuleList()

        for weight_path in pretrained_weights_paths:
            policy = self._load_pretrained_policy(input_dim, output_dim, weight_path)
            for param in policy.parameters():
                param.requires_grad = False
            self.columns.append(policy)

        for _ in pretrained_weights_paths:
            self.lateral_connections.append(nn.Linear(128, 128))

        self.actor_critic_column = self._create_actor_critic_column(input_dim, output_dim)

    def 