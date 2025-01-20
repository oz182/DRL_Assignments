#test #test
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
# Best hyperparameters: {'policy_lr': 0.005582963305227242, 'value_lr': 0.0006112590022937452, 'discount_factor': 0.9792188226037197}
# Best reward: -109.30097087378641

# Last try that we have a plot of:
# Best is trial 24 with value: -111.11650485436893.
# Best hyperparameters: {'policy_lr': 0.0011435182467994243, 'value_lr': 0.0005850808192221598, 'discount_factor': 0.9876381948631086}
# Best reward: -111.11650485436893

fine_tunining=False  # Flag to activate hyperparameter fine-tuning using Optuna
 
# Policy Network (Actor)
class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size, learning_rate):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 12)
        self.fc2 = nn.Linear(12, 12)
        self.fc3 = nn.Linear(12, action_size)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        output = self.fc3(x)
        return torch.softmax(output, dim=1)

# Value Network (Critic)
class ValueNetwork(nn.Module):
    def __init__(self, state_size, learning_rate):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        output = self.fc3(x)
        return output


# Training loop
def train(env, policy, value_network, discount_factor, max_episodes, max_steps):
    episode_rewards = []

    for episode in range(max_episodes):

        state,_ = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        cumulative_reward = 0

        for step in range(max_steps):

            action_probs = policy(state)
            action = torch.multinomial(action_probs, 1).item()
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated  # Properly handle episode termination
            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

            current_value = value_network(state)
            next_value = value_network(next_state)
            td_target = reward + (1 - done) * discount_factor * next_value
            td_error = td_target - current_value

            # Update Value Network
            value_loss = nn.functional.mse_loss(current_value, td_target.detach())
            value_network.optimizer.zero_grad()
            value_loss.backward()
            value_network.optimizer.step()

            # Update Policy Network
            log_prob = torch.log(action_probs.squeeze(0)[action])
            policy_loss = -log_prob * td_error.detach()
            policy.optimizer.zero_grad()
            policy_loss.backward()
            policy.optimizer.step()

            state = next_state
            cumulative_reward += reward

            if done:
                episode_rewards.append(cumulative_reward)
                print(f"Episode {episode} Reward: {cumulative_reward}")

                if episode > 100 and np.mean(episode_rewards[-100:]) > -100:
                    if fine_tunining:
                        torch.save(policy.state_dict(), "Assginment3/Part1_IndividualNet/Acrobot_AC/acrobot_policy.pth")
                        torch.save(value_network.state_dict(), "Assginment3/Part1_IndividualNet/Acrobot_AC/acrobot_value.pth")
                        print("Model saved!")
                    return episode_rewards
                break
    return episode_rewards


def plot_multiple_rewards(rewards_list, hyperparams_list):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    for i, (episode_rewards, (policy_lr, value_lr, discount_factor)) in enumerate(zip(rewards_list, hyperparams_list)):
        axes[i].plot(episode_rewards, label='Cumulative Reward per Episode')

        window_size = 20
        averaged_rewards = [
            np.mean(episode_rewards[j:j + window_size])
            for j in range(0, len(episode_rewards), window_size)
        ]
        averaged_x = [j for j in range(0, len(episode_rewards), window_size)]
        axes[i].plot(averaged_x, averaged_rewards, marker='o', linestyle='-', color='red', label='Average Reward (Every 20 Episodes)')

        axes[i].set_xlabel('Episode')
        axes[i].set_ylabel('Reward')
        axes[i].set_title(f'Policy LR: {policy_lr}, Value LR: {value_lr}, Discount: {discount_factor}')
        axes[i].legend()
        axes[i].grid(True)

    plt.tight_layout()
    plt.show()

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


# Optuna Objective Function
def objective(trial):
    policy_lr = trial.suggest_loguniform('policy_lr', 1e-5, 1e-3)
    value_lr = trial.suggest_loguniform('value_lr', 1e-5, 1e-3)
    discount_factor = trial.suggest_uniform('discount_factor', 0.93, 0.999)

    env = gym.make('Acrobot-v1')
    policy = PolicyNetwork(state_size=6, action_size=3, learning_rate=policy_lr)
    value_network = ValueNetwork(state_size=6, learning_rate=value_lr)

    average_reward = np.mean(train(env, policy, value_network, discount_factor, max_episodes=500, max_steps=501))
    return average_reward


def main():
    np.random.seed(23)
    torch.manual_seed(23)
    #rewards_list = []

    if fine_tunining:
        # Create the Optuna study and optimize the objective function
        study = optuna.create_study(direction='maximize')

        study.optimize(objective, n_trials=120)

        # Print the best hyperparameters
        print("Best hyperparameters:", study.best_params)
        print("Best reward:", study.best_value)

        # Visualize the study results
        fig1 = optuna.visualization.matplotlib.plot_optimization_history(study)
        fig2 = optuna.visualization.matplotlib.plot_param_importances(study)
        plt.show()

        #rewards = train(env, policy, value_network, discount_factor, max_episodes=1500, max_steps=501)
        #rewards_list.append(rewards)
        #plot_multiple_rewards(rewards_list, hyperparams_list)

    else:
        env = gym.make('Acrobot-v1')
        policy = PolicyNetwork(state_size=6, action_size=3, learning_rate=0.001)
        value_network = ValueNetwork(state_size=6, learning_rate=0.001)
        start_time=time.time()
        rewards = train(env, policy, value_network, discount_factor=0.99, max_episodes=1500, max_steps=501)
        end_time=time.time()
        print("Training time:", end_time-start_time)
        plot_single_reward(rewards, policy_lr=0.001, value_lr=0.001, discount_factor=0.99)

if __name__ == '__main__':
    main()

