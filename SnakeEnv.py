import random

import pygame


class SnakeEnv:
    def __init__(self, render=False, grid_size=10, block_size=20):
        """
        Initialize the Snake game environment. If 'render' is True, the game will
        use pygame to render the game environment visually. This includes setting up
        the game window size based on 'grid_size' and 'block_size'.
        """
        self.grid_size = grid_size  # Size of the game grid
        self.block_size = block_size  # Size of each block in the grid
        self.screen_size = grid_size * block_size  # Total size of the game screen

        # Initialize pygame window and settings if rendering is enabled
        if render:
            pygame.init()
            self.dis = pygame.display.set_mode((self.screen_size, self.screen_size))
            pygame.display.set_caption('Snake Game')
            self.clock = pygame.time.Clock()
            self.font_style = pygame.font.SysFont(None, 35)

        self.reset()  # Reset the game to its initial state

    def get_env_info(self):
        """
        Get environment information including the dimensions of the state representation
        and the number of actions available.
        """
        input_channels = 1  # Number of channels in the state representation
        state_size = self.grid_size * self.grid_size  # Total size of the state
        action_size = 4  # Number of possible actions (up, down, left, right)
        return input_channels, state_size, action_size

    def reset(self):
        """
        Reset the game environment to start a new episode. This initializes the
        snake's position, places the first piece of food, and sets the initial score.
        """
        # Initialize the snake in the center of the grid
        self.snake_pos = [[self.grid_size // 2, self.grid_size // 2]]
        # Randomly place the first piece of food
        self.food_pos = [random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)]
        self.direction = 'STOP'  # The snake does not move initially
        self.score = 0  # Reset the score
        self.done = False  # The game is not over
        return self._get_state()  # Return the initial state

    def _get_state(self):
        """
        Generate the current game state for the learning agent. This includes
        the positions of the snake and the food.
        """
        # Create a grid to represent the state
        game_state = [[0 for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        # Mark the food's position in the grid
        game_state[self.food_pos[0]][self.food_pos[1]] = 1
        # Mark the snake's head in the grid
        game_state[self.snake_pos[0][0]][self.snake_pos[0][1]] = 3
        # Mark the rest of the snake's body in the grid
        for pos in self.snake_pos[1:]:
            game_state[pos[0]][pos[1]] = 2

        # Determine the expected action based on the food's position relative to the snake
        x, y = self.snake_pos[0]
        fx, fy = self.food_pos
        expected_action = [fx < x, fx > x, fy < y, fy > y]

        return game_state, expected_action

    def step(self, action):
        """Update the game state based on the action and check if the game has ended."""
        directions = ['LEFT', 'RIGHT', 'UP', 'DOWN']
        dx, dy = {'LEFT': (-1, 0), 'RIGHT': (1, 0), 'UP': (0, -1), 'DOWN': (0, 1)}[directions[action]]

        previous_head = self.snake_pos[0]
        new_head = [previous_head[0] + dx, previous_head[1] + dy]

        # Check if the snake head collides with the boundaries
        if new_head[0] < 0 or new_head[0] >= self.grid_size or new_head[1] < 0 or new_head[1] >= self.grid_size:
            self.done = True

        # Check if the snake head collides with its body
        if new_head in self.snake_pos:
            self.done = True

        if not self.done:
            self.snake_pos.insert(0, new_head)  # Update the snake's position
            if new_head == self.food_pos:  # If the snake eats the food
                self.score += 1
                self.food_pos = self._generate_food()  # Place new food
            else:
                self.snake_pos.pop()  # Move the snake by removing the tail piece

            # Calculate the Manhattan distance difference before and after the move
            manhattan_dist_before = abs(previous_head[0] - self.food_pos[0]) + abs(previous_head[1] - self.food_pos[1])
            manhattan_dist_after = abs(new_head[0] - self.food_pos[0]) + abs(new_head[1] - self.food_pos[1])

            # If the snake moves closer to the food, increase the score; otherwise, decrease it
            if manhattan_dist_after < manhattan_dist_before:
                self.score += 0.1  # Reward moving closer to the food
            else:
                self.score -= 0.1  # Penalize moving away from the food

        return self._get_state()[0], self._get_state()[1], self.score, self.done

    def _generate_food(self):
        """
        Place a new piece of food in a random location that is not occupied by the snake.
        """
        while True:
            food_pos = [random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)]
            if food_pos not in self.snake_pos:
                return food_pos

    def render(self):
        """
        Render the current game state using Pygame. This method draws the snake
        and the food on the game window.
        """
        self.dis.fill((0, 0, 0))  # Fill the background with black

        # Draw the snake head
        pygame.draw.rect(self.dis, (0, 0, 255),
                         [self.snake_pos[0][0] * self.block_size, self.snake_pos[0][1] * self.block_size,
                          self.block_size, self.block_size])
        # Draw the snake body
        for part in self.snake_pos[1:]:
            pygame.draw.rect(self.dis, (0, 255, 0),
                             [part[0] * self.block_size, part[1] * self.block_size, self.block_size, self.block_size])
        # Draw the food
        pygame.draw.rect(self.dis, (255, 0, 0),
                         [self.food_pos[0] * self.block_size, self.food_pos[1] * self.block_size, self.block_size,
                          self.block_size])

        pygame.display.update()  # Update the display
        self.clock.tick(10)  # Control the game speed

    def close(self):
        """Close the Pygame window."""
        pygame.quit()
