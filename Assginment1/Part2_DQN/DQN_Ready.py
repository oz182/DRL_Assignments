import random
import numpy as np
import gymnasium as gym
from collections import defaultdict, namedtuple, deque

from tqdm import tqdm
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
device="cpu"  # Apperantly running better than mps


class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        # Define layers
        self.L1 = nn.Linear(input_size, hidden_size[0])

        self.L2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.L3 = nn.Linear(hidden_size[1], hidden_size[2])

        self.L4 = nn.Linear(hidden_size[2], output_size)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.L1(x))  # Hidden layer 1
        x = self.activation(self.L2(x))  # Hidden layer 2
        x = self.activation(self.L3(x))  # Hidden layer 3
        x = self.L4(x)  # Hidden layer 3
        return x


class Extended_DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Extended_DQN, self).__init__()
        # Define layers
        self.L1 = nn.Linear(input_size, hidden_size[0])

        self.L2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.L3 = nn.Linear(hidden_size[1], hidden_size[2])
        self.L4 = nn.Linear(hidden_size[2], hidden_size[3])
        self.L5 = nn.Linear(hidden_size[3], hidden_size[4])

        self.L6 = nn.Linear(hidden_size[4], output_size)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.L1(x))  # Hidden layer 1
        x = self.activation(self.L2(x))  # Hidden layer 2
        x = self.activation(self.L3(x))  # Hidden layer 3
        x = self.activation(self.L4(x))  # Hidden layer 5
        x = self.activation(self.L5(x))  # Hidden layer 6
        x = self.L6(x)
        return x


class ExperienceReplay:
    def __init__(self, capacity, device="cpu"):
        self.max_capacity = capacity
        self.memory = deque([], maxlen=capacity)  # Automatically handles capacity.
        self.device = device

    def __len__(self):
        return len(self.memory)

    def push(self, state,action,next_state,reward,done):
        """Store a transition in the buffer."""
        self.memory.append((state, action, next_state,reward,done))

    def sample(self, batch_size):
        """Sample a batch of transitions."""
        return random.sample(self.memory, batch_size)
    
    
def warmup_buffer(env, Expriance_buffer, warmup_steps=1000):

    # --- Warm-Up Period ---
    warmup_steps = 1000 # Number of steps for warm-up
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    print("Starting warm-up period...")
    for _ in range(warmup_steps):
        action = env.action_space.sample()  # Random action
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        observation = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        Expriance_buffer.push(state, action, observation, reward, done)
        state = observation if not done else torch.tensor(env.reset()[0], dtype=torch.float32, device=device).unsqueeze(0)
    print("Warm-up complete.")

    return Expriance_buffer


class DQN_agent():
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.99,
    ):
        self.env = env
        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.training_error = []

    def sample_action(self, obs, model):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)  # Ensure obs is a tensor

        if np.random.random() < self.epsilon:
            self.decay_epsilon()
            return self.env.action_space.sample()  # Random action
        else:
            self.decay_epsilon()
            with torch.no_grad():
                return model(obs).argmax(dim=1).item()  # Greedy action


    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)


def batch2tensors(batch):
    """
    :param batch: batch of tuples of past random transitions
    :return: tensors vectors for each thing-states, actions, rewards, next_states, dones
    """
    states, actions, next_states, rewards,dones = zip(*batch)
    states = torch.cat(states)
    actions = torch.tensor(actions, dtype=torch.long, device=device)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
    next_states = torch.cat(next_states)
    dones = torch.tensor(dones, dtype=torch.float32, device=device)
    return states, actions, rewards, next_states, dones


def train(policy_net, target_net, optimizer, criterion, discount_factor, num_batch, Memo):

    states, actions, rewards, next_states, dones = batch2tensors(Memo.sample(num_batch))

    q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1) # Q-values for taken actions
    next_q_values = target_net(next_states).max(1).values.detach()  # Detach target values

    targets = rewards + discount_factor * next_q_values * (1 - dones)

    loss = criterion(q_values, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def training_loop(env, agent, policy_net, target_net, Memo, T, num_episodes, Criterion, optimizer, reward_per_episode, loss_per_episode):
    # Loop for training the agent
    count=0
    for i_episode in range(num_episodes):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        total_reward = 0
        episode_loss = 0  # Variable to accumulate the loss for the episode

        for t in range(T):
            count+=1
            # Sample an action
            action = Agent.sample_action(state, policy_net)

            # Perform action in the environment
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store transition in memory
            observation = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            Memo.push(state, action, observation, reward, done)

            # Update state
            state = observation
            total_reward += reward

            # Train the policy network if enough samples are available
            if len(Memo) > Batch_size:
                Loss = train(policy_net, target_net, optimizer, Criterion, discount_factor, Batch_size, Memo)
                episode_loss += Loss  # Accumulate the loss for the current episode

                # Update target network periodically
                if count % C == 0:
                    target_net.load_state_dict(policy_net.state_dict())

            if done or t ==499:
                # Store the total reward and average loss for this episode
                reward_per_episode.append(total_reward)
                loss_per_episode.append(episode_loss / (t + 1))  # Average loss for the episode
                break
        # Print the current loss for the episode and step
        print(f"The current loss for episode {i_episode} and step {total_reward} is: {Loss}")

    return policy_net, reward_per_episode, loss_per_episode


def draw_graphs(num_episodes, rewards, losses):
    # Create the plot
    plt.figure(figsize=(12, 6))

    # Plot average rewards
    plt.subplot(1, 2, 1)
    plt.plot(list(range(1, num_episodes + 1)), rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('Reward per episode')
    plt.grid()

    # Plot average losses
    plt.subplot(1, 2, 2)
    plt.plot(list(range(1, num_episodes + 1)), losses, color='r')
    plt.xlabel('Episodes')
    plt.ylabel('Loss')
    plt.title('Loss per episode')
    plt.grid()

    plt.tight_layout()
    plt.show()


def test_agent(agent, policy_net):
    Rendered_env = gym.make('CartPole-v1', render_mode='human')
    state, info =Rendered_env.reset()

    # Sample an action
    action = Agent.sample_action(state, policy_net)

    # Perform action in the environment
    observation, reward, terminated, truncated, info = Rendered_env.step(action)
    done = terminated or truncated

    while not done:
        state = observation

        # Sample an action
        action = Agent.sample_action(state, policy_net)

        # Perform action in the environment
        observation, reward, terminated, truncated, info = Rendered_env.step(action)
        done = terminated or truncated

        Rendered_env.render()
        


## main ##

#parameters
discount_factor = 0.99
LR = 0.001
initial_epsilon = 0.99
Final_epsilon = 0.01
Epsilon_decay = 0.9998
Batch_size = 1000
learning_rate=0.001

# Create the environment (CartPole-v1)
env = gym.make('CartPole-v1', render_mode='rgb_array')

# Define the replay buffer capacity
Capacity = 10000

# Initialize the experience replay buffer
Memo = ExperienceReplay(Capacity, device)
Memo = warmup_buffer(env, Memo)

# Get number of actions from gym action space
n_actions = 2
n_observations = 4
hid_layers= [128, 64, 32, 64, 128]
C=300
policy_net = Extended_DQN(n_observations,hid_layers, n_actions).to(device)
target_net = Extended_DQN(n_observations,hid_layers, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
Criterion=nn.MSELoss()
num_episodes=200
T=700
Agent=DQN_agent(env,learning_rate,initial_epsilon,Epsilon_decay,Final_epsilon,discount_factor)
#Loss=1
# List to store rewards and losses for each episode
reward_per_episode = []
loss_per_episode = []

policy_net, reward_per_episode, loss_per_episode = training_loop(env, Agent, policy_net, target_net, Memo, T, num_episodes, Criterion, optimizer, reward_per_episode, loss_per_episode)


#test_agent(Agent, policy_net)

draw_graphs(num_episodes, reward_per_episode, loss_per_episode)




















