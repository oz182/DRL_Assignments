import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
import numpy as np

class ProgressiveNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, pretrained_weights_paths):
        """
        Initialize the Progressive Network.

        Args:
            input_dim (int): Dimension of the input state.
            output_dim (int): Dimension of the action space.
            pretrained_weights_paths (list): List of file paths to pre-trained policy weights.
        """
        super(ProgressiveNetwork, self).__init__()

        self.columns = nn.ModuleList()
        self.lateral_connections = nn.ModuleList()

        # Load pre-trained policy networks as frozen columns
        for weight_path in pretrained_weights_paths:
            policy = self._load_pretrained_policy(input_dim, output_dim, weight_path)
            for param in policy.parameters():
                param.requires_grad = False
            self.columns.append(policy)

        # Add lateral connections for each pre-trained policy to the new actor-critic column
        for _ in pretrained_weights_paths:
            self.lateral_connections.append(nn.Linear(128, 128))

        # Add the final actor-critic column
        self.actor_critic_column = self._create_actor_critic_column(input_dim, output_dim)

    def _load_pretrained_policy(self, input_dim, output_dim, weight_path):
        """Loads a pre-trained policy from a .pth file."""
        policy = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        # Load the state_dict and remap keys to match Sequential layers
        pretrained_state = torch.load(weight_path)
        remapped_state = {f"{i}.weight": pretrained_state[f"fc{i+1}.weight"] for i in range(3)}
        remapped_state.update({f"{i}.bias": pretrained_state[f"fc{i+1}.bias"] for i in range(3)})
        policy.load_state_dict(remapped_state)
        return policy

    def _create_actor_critic_column(self, input_dim, output_dim):
        """Creates the actor-critic column."""
        column = nn.ModuleDict({
            "policy": nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, output_dim),
            ),
            "value": nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
            )
        })
        return column

    def forward(self, x):
        """
        Forward pass through the Progressive Network.

        Args:
            x (Tensor): Input state.

        Returns:
            dict: Contains the policy (actor) and value (critic) outputs.
        """
        # Gather features from pre-trained policies
        lateral_features = 0
        for policy, lateral_connection in zip(self.columns, self.lateral_connections):
            policy_features = policy(x)
            lateral_features += F.relu(lateral_connection(policy_features))

        # Pass through the actor-critic column
        policy_features = self.actor_critic_column["policy"](x)
        value_features = self.actor_critic_column["value"](x)

        # Combine lateral contributions for the policy
        policy_output = F.softmax(policy_features + lateral_features, dim=-1)

        return {
            "policy": policy_output,  # Actor output (policy)
            "value": value_features.squeeze(-1)  # Critic output (value)
        }

def train(model, env, optimizer, num_episodes=1000, gamma=0.99):
    """
    Train the Progressive Network using the Actor-Critic algorithm on the MountainCar environment.

    Args:
        model (ProgressiveNetwork): The Progressive Network model.
        env (gym.Env): The MountainCar environment.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        num_episodes (int): Number of training episodes.
        gamma (float): Discount factor.
    """
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        log_probs = []
        values = []
        rewards = []

        done = False
        while not done:
            output = model(state)
            policy, value = output["policy"], output["value"]

            # Sample an action from the policy
            action = torch.multinomial(policy, 1).item()

            # Perform the action in the environment
            next_state, reward, done, _, _ = env.step(action)

            # Save log probabilities, values, and rewards
            log_probs.append(torch.log(policy[0, action]))
            values.append(value)
            rewards.append(reward)

            state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

        # Compute returns and losses
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32)
        values = torch.cat(values)
        log_probs = torch.stack(log_probs)

        advantage = returns - values

        policy_loss = -(log_probs * advantage.detach()).mean()
        value_loss = F.mse_loss(values, returns)
        loss = policy_loss + value_loss

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Episode {episode + 1}/{num_episodes}, Loss: {loss.item():.4f}, Total Reward: {sum(rewards):.2f}")

def main():
    # Initialize the MountainCar environment
    env = gym.make("MountainCar-v0")

    # Define the input and output dimensions
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    # Paths to pre-trained weights (mock paths for demonstration)
    pretrained_weights_paths = [
        "Assginment3/Part1_IndividualNet/CartPole_AC/cartpole_policy.pth",
        "Assginment3/Part1_IndividualNet/Acrobot_AC/acrobot_policy.pth"
    ]

    # Create the Progressive Network model
    model = ProgressiveNetwork(input_dim, output_dim, pretrained_weights_paths)

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Train the model
    train(model, env, optimizer, num_episodes=500)

if __name__ == "__main__":
    main()
