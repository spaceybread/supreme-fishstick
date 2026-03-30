from gym import spaces
from gym import Env
import numpy as np
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

        self.observation_space = self.observation_space = spaces.Box(0, 1, shape=(8,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)

        self.player_pos = [1, 1]
        self.target_pos = [2 * self.maze_size - 1, 2 * self.maze_size - 1]
        
        self.max_steps = 1000
        self.current_steps = 0

        self.reset()
        
    def step(self, action):
        self.current_steps += 1
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

        tr, tc = self.target_pos
        pr, pc = self.player_pos
        
        done = (self.player_pos == self.target_pos)
        
        max_dist = ((self.grid_size**2 + self.grid_size**2) ** 0.5)
        dist = ((pr - tr)**2 + (pc - tc)**2) ** 0.5
        reward = -(dist / max_dist)

        if (pr, pc) in self.visited_cells: reward -= 0.3
        else:
            self.visited_cells.add((pr, pc))
            reward += 0.1

        if done: reward = 10.0
        
        truncated = self.current_steps >= self.max_steps
        done = done or truncated
        
        return self.get_obs(), reward, done, {}
        
    def get_obs(self):
        pr, pc = self.player_pos
        tr, tc = self.target_pos
        r, c = self.player_pos

        wall_up    = 1 if self.grid[(r-1)*self.grid_size + c] == WALL else 0
        wall_down  = 1 if self.grid[(r+1)*self.grid_size + c] == WALL else 0
        wall_left  = 1 if self.grid[r*self.grid_size + (c-1)] == WALL else 0
        wall_right = 1 if self.grid[r*self.grid_size + (c+1)] == WALL else 0

        return np.array([
            pr / self.grid_size,
            pc / self.grid_size,
            tr / self.grid_size,
            tc / self.grid_size,
            wall_up, wall_down, wall_left, wall_right
        ], dtype=np.float32)
    
    def reset(self):
        self.current_steps = 0
        self.visited_cells = set()
        self.visited_cells.add((1, 1))
        maze = Maze(self.maze_size)
        self.grid = np.array(maze.flatten(), dtype=np.int16)
        self.player_pos = [1, 1]
        return self.get_obs()
    
    def render(self):
        ma = {0: ' ', 9: '#', 1: 'o', 2: 'x'}
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                print(ma[self.grid[i * self.grid_size + j]], end='')
            print()
