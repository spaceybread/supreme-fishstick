from gym import spaces
from gym import Env
import numpy as np
import clear_output
from make_maze import Maze

import random
import os

EMPTY_CELL = 0
WALL = 9
PLAYER = 1
TARGET = 2

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

class MazeEnv(Env):
    def __init__(self):
        self.maze_size = 10
        self.grid_size = 2 * self.maze_size + 1

        self.observation_space = spaces.Box(0, 9, [self.grid_size * self.grid_size], dtype=np.int16)
        self.action_space = spaces.Discrete(4)

        self.player_pos = [1, 1]
        self.target_pos = [2 * self.maze_size - 1, 2 * self.maze_size - 1]

        self.reset()
        
    def step(self, action):
        moves = {UP: (-2, 0), DOWN: (2, 0), LEFT: (0, -2), RIGHT: (0, 2)}
        wall_offsets = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}

        r, c = self.player_pos
        dr, dc = moves[action]
        wr, wc = wall_offsets[action]

        wall_idx = (r + wr) * self.grid_size + (c + wc)

        if self.grid[wall_idx] != WALL:
            new_r, new_c = r + dr, c + dc

            self.grid[r * self.grid_size + c] = EMPTY_CELL
            self.grid[new_r * self.grid_size + new_c] = PLAYER
            self.player_pos = [new_r, new_c]

        done = (self.player_pos == self.target_pos)
        reward = 1.0 if done else -0.01
        return self.grid.copy(), reward, done, {}
    
    def reset(self):
        maze = Maze(self.maze_size)
        self.grid = np.array(maze.flatten(), dtype=np.int16)
        self.player_pos = [1, 1]
        return self.grid.copy()
    
    def render(self):
        ma = {0: ' ', 9: '#', 1: 'o', 2: 'x'}
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                print(ma[self.grid[i * self.grid_size + j]], end='')
            print()
