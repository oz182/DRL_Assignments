import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
import optuna
from plotly.io import show
import sklearn
import time

fine_tunining=True  # Flag to activate hyperparameter fine-tuning using Optuna

#Best is trial 29 with value: 209.044.
#Best hyperparameters: {'policy_lr': 0.000490641507278229, 'value_lr': 0.0016361417772812591, 'discount_factor': 0.9916757486832062}
#Best reward: 209.044

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
        return torch.softmax(output[:, :2], dim=1)

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

# Pad state with zeros
def pad_with_zeros(v, pad_size):
    v = np.asarray(v, dtype=np.float32)  # Ensure v is a NumPy array
    v_t = np.hstack((v, np.zeros(pad_size)))
    return v_t.reshape((1, v_t.shape[0]))


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

# Training loop
def train(env, policy, value_network, discount_factor, max_episodes, max_steps):
    episode_rewards = []
    for episode in range(max_episodes):
        state,_ = env.reset()

        state = pad_with_zeros(state, 6- env.observation_space.shape[0])
        state = torch.tensor(state, dtype=torch.float32)

        cumulative_reward = 0
        for step in range(max_steps):
            action_probs = policy(state)
            action = torch.multinomial(action_probs, 1).item()
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated  # Properly handle episode termination

            next_state = pad_with_zeros(next_state, 6 -  env.observation_space.shape[0])
            next_state = torch.tensor(next_state, dtype=torch.float32)

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

                if episode > 100 and np.mean(episode_rewards[-100:]) > 475:
                    if fine_tunining:
                        torch.save(policy.state_dict(), "Assginment3/Part1_IndividualNet/CartPole_AC/cartpole_policy.pth")
                        torch.save(value_network.state_dict(), "Assginment3/Part1_IndividualNet/CartPole_AC/cartpole_value.pth")
                    return episode_rewards
                break
    return episode_rewards


# Optuna Objective Function
def objective(trial):
    policy_lr = trial.suggest_loguniform('policy_lr', 1e-5, 1e-3)
    value_lr = trial.suggest_loguniform('value_lr', 1e-5, 1e-3)
    discount_factor = trial.suggest_uniform('discount_factor', 0.93, 0.999)

    env = gym.make('CartPole-v1')
    policy = PolicyNetwork(state_size=6, action_size=3, learning_rate=policy_lr)
    value_network = ValueNetwork(state_size=6, learning_rate=value_lr)

    average_reward = np.mean(train(env, policy, value_network, discount_factor, max_episodes=500, max_steps=501))
    return average_reward


def main():
    np.random.seed(23)
    torch.manual_seed(23)

    if fine_tunining:
        # Create the Optuna study and optimize the objective function
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=130)

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
        env = gym.make('CartPole-v1')
        policy = PolicyNetwork(state_size=6, action_size=3, learning_rate=0.0001)
        value_network = ValueNetwork(state_size=6, learning_rate=0.0005)
        start_time = time.time()
        rewards = train(env, policy, value_network, discount_factor=0.99, max_episodes=1500, max_steps=501)
        end_time = time.time()
        print(f'Time elapsed: {end_time - start_time}')
        plot_single_reward(rewards, policy_lr=0.0001, value_lr=0.0005, discount_factor=0.99)



if __name__ == '__main__':
    main()




