import os
import time

import matplotlib.pyplot as plt

from SnakeEnv import SnakeEnv
from agents.DQNAgent import Agent


def train_agent(episodes=10000, episode_max_steps=512, render=False):
    """Train the DQN agent for a specified number of episodes."""
    # Initialize the Snake game environment and the agent.
    env = SnakeEnv(render)
    input_channels, state_size, action_size = env.get_env_info()  # Dimensions of state and action.
    agent = Agent(input_channels, state_size, action_size)

    episode_rewards = []  # To store total reward for each episode.

    # Load a pre-trained model if it exists. You can repeat train a model if your computer is slow.
    model_path = "agent_model.pth"
    if os.path.exists(model_path):
        agent.load(model_path)
        print('Loaded pre-trained model.')
    else:
        print('No pre-trained model found, starting training from scratch.')

    print(f'Start training..., Render is {render}')
    start_time = time.time()

    # Training loop.
    for e in range(episodes):
        state, expect_action = env.reset()  # Reset environment to get initial state.
        current_score = 0  # Initialize total score for the episode to 0.
        previous_score = 0  # Initialize previous score.

        for s in range(episode_max_steps):
            action = agent.act(state)  # Choose action based on current state.
            next_state, expect_action, current_score, done = env.step(action)  # Execute action, get feedback.
            if render:
                env.render()  # Update the current environment image.
            reward = current_score - previous_score  # Calculate immediate reward for the current step.
            agent.remember(state, action, reward, next_state, done, expect_action)  # Save experience.
            state = next_state
            previous_score = current_score
            if done:
                break

        episode_rewards.append(current_score)  # Record total score for the episode.
        loss = agent.replay()  # Learn from experience replay.

        # Save the model and print information every 1000 episodes.
        if e % 1000 == 0:
            print(f'Start replay for episode {e}, step {s}, total_score is {current_score}, loss is {loss}')
            agent.save('agent_model.pth')  # Save model.

    end_time = time.time()
    print(f'Time cost: {end_time - start_time}')

    # Plot the trend of rewards during training.
    plt.plot(episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Score')
    plt.show()


def test_agent(render=True):
    """Test the trained DQN agent."""
    print('Start testing')
    env = SnakeEnv(render)  # Set environment to test mode.
    _, state_size, action_size = env.get_env_info()
    agent = Agent(1, state_size, action_size)
    agent.load("agent_model.pth")  # Load the model.

    state, _ = env.reset()  # Reset environment to get initial state.
    while True:
        agent.epsilon = 0  # Set epsilon=0, fully rely on the model for decision making.
        action = agent.act(state)  # Choose action.
        next_state, _, _, done = env.step(action)
        if render:
            env.render()  # Get the current environment image.

        state = next_state
        if done:
            break


if __name__ == "__main__":
    train_agent(episodes=100000, render=False)
    test_agent(render=True)
