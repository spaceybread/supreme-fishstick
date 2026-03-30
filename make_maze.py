import numpy as np
import random

class Maze:
    def __init__(self, size):
        self.size = size
        self.wall_symbol = '#'
        self.empt_symbol = ' '
        self.generate()
        
        self.out = self.get_2d_arr()

    def generate(self):
        self.h_walls = np.ones((self.size, self.size), dtype=bool)
        self.v_walls = np.ones((self.size, self.size), dtype=bool)

        self.visited = set()
        self.step(0, 0)

    def step(self, i, j):
        self.visited.add((i, j))

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        random.shuffle(directions)

        for dx, dy in directions:
            ni, nj = i + dx, j + dy

            if 0 <= ni < self.size and 0 <= nj < self.size:
                if (ni, nj) not in self.visited:

                    if dx == 1: self.h_walls[i][j] = False
                    if dx == -1: self.h_walls[ni][nj] = False
                    if dy == 1: self.v_walls[i][j] = False
                    if dy == -1: self.v_walls[ni][nj] = False

                    self.step(ni, nj)

    def get_2d_arr(self):
        out = []
        out += [[9] * (2 * self.size + 1)]
        
        for i in range(self.size):
            row = [9]
            for j in range(self.size):
                row += [0]
                if j < self.size - 1: row += [9] if self.v_walls[i][j] else [0]
                else: row += [9]
            out += [row]

            row = []
            for j in range(self.size):
                row += [9]
                if i < self.size - 1: row += [9] if self.h_walls[i][j] else [0]
                else: row += [9]
            row += [9]
            out += [row]
        
        out[1][1] = 1
        out[2 * self.size - 1][2 * self.size - 1] = 2
        
        # for x in out: print(*x)
        self.out = out
        return out

    def render_map(self):
        arr = self.out
        
        ma = {9: '#', 0: ' ', 1: 'o', 2: 'x'}
        
        for i in range(len(arr)):
            for j in range(len(arr[i])):
                print(ma[arr[i][j]], end='')
            print()
    
    def flatten(self):
        arr = self.out
        out = []
        
        for x in arr:
            for y in x:
                out.append(y)
        
        return out
