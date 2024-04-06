import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DQN(nn.Module):
    """
    The DQN model, consisting of convolutional layers for feature extraction from the state input,
    followed by fully connected layers that predict the Q-values for each action.
    """
    def __init__(self, input_channels, action_size):
        super(DQN, self).__init__()
        # Convolutional layers for processing state inputs
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)  # Dropout layer to reduce overfitting
        )

        # Fully connected layers for action value prediction
        self.fc = nn.Sequential(
            nn.Linear(32 * 10 * 10, 128),  # Adjust the input size based on your state dimensions
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, action_size)  # Output layer with a size equal to the number of actions
        )

    def forward(self, x):
        """
        Forward pass of the network. Takes a state input and predicts Q-values for each action.
        """
        x = self.conv(x)  # Pass the input through the conv layers
        x = torch.flatten(x, 1)  # Flatten the output for the fully connected layers
        x = self.fc(x)  # Pass the flattened output through the fully connected layers
        return x

class ReplayBuffer:
    """
    A class for the Replay Buffer, storing past experiences to sample and train the DQN model.
    This helps in breaking the correlation between consecutive experiences.
    """
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)  # Define a fixed-size buffer using deque

    def add(self, state, action, reward, next_state, done):
        """
        Add an experience to the buffer.
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Randomly sample a batch of experiences from the buffer.
        """
        return random.sample(self.buffer, min(len(self.buffer), batch_size))

class Agent:
    """
    The Agent class, incorporating the DQN model, Replay Buffer, and the decision policy.
    """
    def __init__(self, input_channels, state_size, action_size, batch_size=128):
        self.state_size = state_size  # The dimensions of the state
        self.action_size = action_size  # The number of possible actions
        self.memory = ReplayBuffer(500)  # Replay Buffer with a capacity of 500 experiences
        self.gamma = 0.95  # Discount factor for future rewards
        self.epsilon = 1.0  # Initial exploration rate
        self.epsilon_min = 0.01  # Minimum exploration rate
        self.epsilon_decay = 0.999  # Rate at which the exploration rate decays
        self.batch_size = batch_size  # Batch size for training from Replay Buffer

        # Instantiate the DQN model and the optimizer
        self.model = DQN(input_channels, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

    def remember(self, state, action, reward, next_state, done):
        """
        Store an experience in the Replay Buffer.
        """
        self.memory.add(state, action, reward, next_state, done)

    def act(self, state):
        """
        Decide on an action based on the current state and the exploration-exploitation tradeoff.
        """
        if random.random() <= self.epsilon:
            return random.randint(0, self.action_size - 1)  # Explore
        else:
            state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                action_values = self.model(state)
            return np.argmax(action_values.cpu().data.numpy())  # Exploit

    def replay(self):
        """
        Train the DQN model using a batch of experiences sampled from the Replay Buffer.
        """
        if len(self.memory.buffer) < self.batch_size:
            return  # Do not train if there aren't enough experiences in the buffer

        batch = self.memory.sample(self.batch_size)  # Sample a batch of experiences
        states, actions, rewards, next_states, dones = zip(*batch)

        # Prepare the tensors for PyTorch
        states = torch.stack([torch.Tensor(state).unsqueeze(0) for state in states])
        next_states = torch.stack([torch.Tensor(next_state).unsqueeze(0) for next_state in next_states])
        actions = torch.LongTensor(actions).unsqueeze(-1)
        rewards = torch.FloatTensor(rewards).unsqueeze(-1)
        dones = torch.FloatTensor(dones).unsqueeze(-1)

        # Compute the target Q values
        Q_targets_next = self.model(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        Q_expected = self.model(states).gather(1, actions)
        loss = F.mse_loss(Q_expected, Q_targets)  # Calculate the loss

        # Gradient descent
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update the exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss

    def save(self, filename):
        """
        Save the model parameters.
        """
        torch.save(self.model.state_dict(), filename)

    def load(self, filename):
        """
        Load the model parameters.
        """
        self.model.load_state_dict(torch.load(filename))
        self.model.eval()  # Set the model to evaluation mode
