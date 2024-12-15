import numpy as np
import gymnasium as gym
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical


# Define the policy network
class PolicyNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], output_size)
        self.activation = nn.ReLU()

        # Storage for log probabilities and rewards
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=-1)  # Softmax across the last dimension


# Select an action based on policy probabilities
def select_action(policy_net, state):
    action_probs = policy_net(state)
    action_dist = Categorical(action_probs)
    action = action_dist.sample()
    log_prob = action_dist.log_prob(action)

    # Save the log probability of the chosen action
    policy_net.saved_log_probs.append(log_prob)
    return action.item()


# Optimize the policy network
def optimize_net(policy_net, optimizer, discount_factor):
    if len(policy_net.saved_log_probs) == 0 or len(policy_net.rewards) == 0:
        print("No log probabilities or rewards collected in this episode. Skipping optimization.")
        return

    R = 0
    G_t = deque()

    # Compute cumulative discounted rewards in reverse
    for r in policy_net.rewards[::-1]:
        R = r + discount_factor * R
        G_t.appendleft(R)

    # Convert to a tensor and normalize
    G_t = torch.tensor(G_t, dtype=torch.float32)
    #G_t = (G_t - G_t.mean()) / (G_t.std() + 1e-5) if len(G_t) > 1 else G_t
    print (G_t[0])
    # Compute policy loss
    policy_loss = []
    for log_prob, reward in zip(policy_net.saved_log_probs, G_t):
        policy_loss.append(-log_prob * reward)

    if len(policy_loss) == 0:
        print("Policy loss is empty. Skipping optimization.")
        return

    policy_loss = torch.stack(policy_loss).sum()

    # Perform backpropagation and optimization
    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()

    # Clear saved log probabilities and rewards
    del policy_net.saved_log_probs[:]
    del policy_net.rewards[:]


# Main training loop
def main():
    # Hyperparameters
    max_episodes = 1000
    max_steps = 500
    learning_rate = 0.005
    discount_factor = 0.995

    # Initialize environment and policy network
    env = gym.make('CartPole-v1', render_mode=None)
    policy_net = PolicyNet(4, [16, 16], 2)
    optimizer = optim.AdamW(policy_net.parameters(), lr=learning_rate)

    for episode in range(max_episodes):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32)

        for step in range(max_steps):
            action = select_action(policy_net, state)
            next_state, reward, done, truncated, _ = env.step(action)

            # Save reward and move to the next state
            policy_net.rewards.append(reward)
            state = torch.tensor(next_state, dtype=torch.float32)

            if done or truncated:
                break

        # Optimize policy after the episode
        optimize_net(policy_net, optimizer, discount_factor)

        # Log progress
        print(f"Episode {episode + 1} completed")

    print("Training finished!")


if __name__ == "__main__":
    main()