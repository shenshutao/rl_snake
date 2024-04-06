# Deep Q-Network (DQN) for Snake Game
## Description
This project is an exploration of training an agent to play the classic Snake game using Deep Reinforcement Learning (specifically, Deep Q-Networks or DQNs). It includes a custom Snake game environment and a DQN agent, with the agent learning to navigate the game to survive and achieve the highest score possible.

## Features
* A custom Snake game environment implemented with Pygame.
* A DQN agent implemented using PyTorch, optimized to improve its performance in the game through learning.
* Support for saving and loading the model, facilitating the training process's interruption and resumption.
* Detailed project documentation and code comments for ease of understanding and use.

## Model Inputs and Outputs
The DQN model takes a processed version of the game state as its input. This input is a tensor representing the game grid, where the snake's position, the food's position, and the obstacles (if any) are encoded. Specifically, the input shape is tailored to the size of the game grid, transformed into a channel-wise representation suitable for the convolutional layers of the network.

The output of the model is a vector of action values, corresponding to the estimated utility of taking each possible action (e.g., moving left, right, up, or down) from the current game state. The agent selects the action with the highest value for execution in the game.

Installation
First, clone the repository to your local machine:

```bash
# Copy code
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
Install the dependencies:

# Install requirements
pip install -r requirements.txt