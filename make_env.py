from collections import deque
from gymnasium import spaces
from make_maze import Maze
from gymnasium import Env
import numpy as np

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
        self.maze_size = 5
        self.grid_size = 2 * self.maze_size + 1

        self.observation_space = spaces.Box(
            0, 255,
            shape=(1, self.grid_size, self.grid_size),
            dtype=np.uint8
        )
        self.action_space = spaces.Discrete(4)

        self.player_pos = [1, 1]
        self.target_pos = [2 * self.maze_size - 1, 2 * self.maze_size - 1]
        
        self.max_steps = 10000
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
        
        current_true_dist = self.dist_map[pr, pc]

        if done:
            reward = 1.0
        elif (pr, pc) not in self.visited_cells:
            reward = 0.01 
            if current_true_dist < self.best_dist:
                reward += 0.02
                self.best_dist = current_true_dist
            else: reward -= 0.01
            self.visited_cells.add((pr, pc))
        else:
            reward = -0.001 

        truncated = self.current_steps >= self.max_steps

        return self.get_obs(), reward, done, truncated, {}
        
    def get_obs(self):
        return (self.grid.reshape(1, self.grid_size, self.grid_size) * 28).astype(np.uint8)

    def _calculate_dist_map(self):
        grid_2d = self.grid.reshape(self.grid_size, self.grid_size)
        dist_map = np.full((self.grid_size, self.grid_size), float('inf'))
        
        target_r, target_c = self.target_pos
        dist_map[target_r, target_c] = 0
        queue = deque([(target_r, target_c)])
        
        while queue:
            r, c = queue.popleft()
            for dr, dc in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
                nr, nc = r + dr, c + dc
                wr, wc = r + dr//2, c + dc//2 
                
                if (0 <= nr < self.grid_size and 0 <= nc < self.grid_size and 
                    grid_2d[wr, wc] != 9 and dist_map[nr, nc] == float('inf')):
                    dist_map[nr, nc] = dist_map[r, c] + 1
                    queue.append((nr, nc))
        return dist_map
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_steps = 0
        self.visited_cells = {(1, 1)}
        
        maze = Maze(self.maze_size)
        self.grid = np.array(maze.flatten(), dtype=np.uint8)
        
        # CRITICAL: Ensure the grid has a TARGET at the expected position
        tr, tc = self.target_pos
        # Clear any other '2's the Maze generator might have placed
        self.grid[self.grid == TARGET] = EMPTY_CELL 
        # Set the target at your math-defined position
        self.grid[tr * self.grid_size + tc] = TARGET 
        
        self.player_pos = [1, 1]
        self.grid[1 * self.grid_size + 1] = PLAYER # Ensure player is visible
        
        self.dist_map = self._calculate_dist_map()
        self.best_dist = self.dist_map[1, 1]
        
        return self.get_obs(), {}
    
    def render(self):
        ma = {0: ' ', 9: '#', 1: 'o', 2: 'x'}
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                print(ma[self.grid[i * self.grid_size + j]], end='')
            print()
