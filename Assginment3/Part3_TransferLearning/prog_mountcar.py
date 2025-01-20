import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from datetime import datetime
import optuna
import matplotlib.pyplot as plt
from plotly.io import show
import sklearn
import time
from torch.nn import functional as F

class PolicyNetwork(nn.Module):
    """
    Simple policy network for both Acrobot and MountainCarContinuous.
    """
    def __init__(self, state_size, hidden_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        output = self.fc3(x)
        return output

class ValueNetwork(nn.Module):
    def __init__(self, state_size, learning_rate):
        super().__init__()
        self.state_size = state_size

        # Define layers
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Pad state with zeros
def pad_with_zeros(v, pad_size):
    v = np.asarray(v, dtype=np.float32)  # Ensure v is a NumPy array
    v_t = np.hstack((v, np.zeros(pad_size)))
    return v_t.reshape((1, v_t.shape[0]))



class ProgressivePolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size, learning_rate):
        super(ProgressivePolicyNetwork, self).__init__()

        self.path_acrobot = 'C:/Users/idogu/PycharmProjects/PythonProject/weights/acrobot_policy.pth'
        self.path_cartpole = 'C:/Users/idogu/PycharmProjects/PythonProject/weights/cartpole_policy.pth'
        self.acrobot_model = self.load_pretrained_model(self.path_acrobot, state_size, hidden_size=12,
                                                        action_size=action_size)
        self.Cartpole_model = self.load_pretrained_model( self.path_cartpole, state_size, hidden_size=12,
                                                            action_size=action_size)

        self.fc1_target = nn.Linear(state_size, 16)
        self.fc2_target = nn.Linear(16, 16)
        self.fc3_target = nn.Linear(16, action_size)

        # Lateral connections to map source outputs to the target network
        self.L12_lateral_cartpole = nn.Linear(12, 16)
        self.L12_lateral_acro2carmont = nn.Linear(12, 16)
        self.L23_lateral_acro2carmont = nn.Linear(12, 16)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def load_pretrained_model(self, filepath, state_size, hidden_size, action_size):
        """
        Load the pretrained model and freeze its weights.
        """
        model = PolicyNetwork(state_size, hidden_size, action_size)
        model.load_state_dict(torch.load(filepath))

        # Freeze the source model
        for param in model.parameters():
            param.requires_grad = False
        return model

    def forward(self, state):
        # state for cartpole_model
        state = pad_with_zeros(state, 6 - 2)
        state = torch.tensor(state, dtype=torch.float32)
        # the forawrd dynamics
        Cartpole_model_output_L1 = self.Cartpole_model.fc1(state)
        acrobot_output_L1 = self.acrobot_model.fc1(state)
        acrobot_model_output_L2 = self.Cartpole_model.fc2(F.relu(acrobot_output_L1))
        h1_target = torch.relu(self.fc1_target(state) +self.L12_lateral_acro2carmont(acrobot_output_L1) +self.L12_lateral_cartpole(Cartpole_model_output_L1))
        h2_target = torch.relu(self.fc2_target(h1_target)+ self.L23_lateral_acro2carmont(acrobot_model_output_L2))
        output = self.fc3_target(h2_target)
        return output[:, :2]

    def get_action(self, state): # Return action and log(prob(action))
        x = self.forward(state)
        std=torch.exp(x[:,1])
        dist = torch.distributions.Normal(x[:,0], std)
        probs=dist.sample()
        action=torch.tanh_(probs)
        return action.detach().numpy(), dist.log_prob(action)


def plot_single_reward(episode_rewards, policy_lr, value_lr, discount_factor):
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards, label='Cumulative Reward per Episode')

    # Compute averaged rewards every 20 episodes
    window_size = 20
    averaged_rewards = [
        np.mean(episode_rewards[i:i + window_size])
        for i in range(0, len(episode_rewards), window_size)
    ]
    averaged_x = [i for i in range(0, len(episode_rewards), window_size)]
    plt.plot(averaged_x, averaged_rewards, marker='o', linestyle='-', color='red', label='Average Reward (Every 20 Episodes)')

    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(f'Cumulative and Averaged Rewards\nPolicy LR: {policy_lr}, Value LR: {value_lr}, Discount: {discount_factor}')
    plt.legend()
    plt.grid(True)
    plt.show()

def train_prog(env, policy, value_network, discount_factor, max_episodes, max_steps):

    episode_rewards = []

    episode = 0

    while (episode < max_episodes):

        state,_ = env.reset()
        cumulative_reward = 0

        for step in range(max_steps):
            action,log_prob = policy.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated  # Properly handle episode termination
            reward = torch.tensor(reward, dtype=torch.float32)
            next_state[1]*=1.006

            current_value = value_network(state=torch.tensor(state, dtype=torch.float32))
            next_value = value_network(state=torch.tensor(next_state, dtype=torch.float32))
            td_target = reward + (1 - done) * discount_factor * next_value
            td_error = td_target - current_value

            # Update Value Network
            value_loss = nn.functional.mse_loss(current_value, td_target.detach())
            value_network.optimizer.zero_grad()
            value_loss.backward()
            value_network.optimizer.step()

            # Update Policy Network
            policy_loss = -log_prob * td_error.detach()
            policy.optimizer.zero_grad()
            policy_loss.backward()
            policy.optimizer.step()

            state = next_state
            cumulative_reward += reward

            if done or step == 999:
                episode_rewards.append(cumulative_reward)
                print(f"Episode {episode} Reward: {cumulative_reward}")

                if episode > 100 and np.mean(episode_rewards[-100:]) > 50:

                    return episode_rewards
                break

        if episode_rewards!=[] and ((max(episode_rewards) < 1) and (episode > 3)):
            episode = 0
            policy.reset_parameters()
            value_network.reset_parameters()
            print("Resetting the weights")
            episode_rewards = []

        episode += 1

    return episode_rewards

def main():
    np.random.seed(23)
    torch.manual_seed(23)
    env = gym.make('MountainCarContinuous-v0', render_mode=None)
    state_size = 6
    action_size = 3
    LR_progressive = 0.001
    LR_critic = 0.0005
    discount_factor = 0.999
    max_episodes = 400
    max_steps = 999
    prog_net = ProgressivePolicyNetwork(state_size, action_size, LR_progressive)
    critic = ValueNetwork(state_size - 4, LR_critic)
    start_time = time.time()
    reward = train_prog(env, prog_net, critic, discount_factor, max_episodes, max_steps)
    end_time = time.time()
    print(f'Time elapsed: {end_time - start_time}')
    plot_single_reward(reward, policy_lr=0.001, value_lr=0.0005, discount_factor=0.999)
if __name__ == '__main__':
    main()
