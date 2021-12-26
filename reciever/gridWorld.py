from typing import Iterable
import random
import numpy as np
import operator


class GridWorld:
    """
    Matrix Game environment.
    """

    def __init__(self, p_term):
        """
        In GridWorld:
        0 -> empty tile
        1 -> wall
        2 -> agent
        3 -> goal
        """
        self.p_term = p_term
        self.row = 5
        self.col = 5
        self.grid = np.zeros((self.row, self.col))
        self.agent_pos = (self.row//2, self.col//2)
        self.grid[self.agent_pos] = 2
        self.forbid = [((self.row//2, self.col//2))]
        while True:
            self.goal = (random.randint(0,self.row-1), random.randint(0,self.col-1))
            if(self.goal not in self.forbid):
                self.grid[self.goal] = 3
                break

    def act(self, action: int):
        """
        Action:
        0 -> N
        1 -> E
        2 -> S
        3 -> W
        """
        p = random.random()
        if p < self.p_term:
            return self.grid, 0, True
        else:
            next_pos = tuple(map(operator.add, self.agent_pos, move(action)))
            try:
                if(self.grid[next_pos] == 0):
                    update(next_pos)
                elif(self.grid[next_pos] == 3):
                    update(next_pos)
                    return self.grid, 1, True
            except IndexError:
                pass
            return self.grid, 0, False
            

    def move(self, action):
        if(action == 0):
            return (0,1)
        elif(action == 1):
            return (1,0)
        elif(action == 2):
            return (0,-1)
        else:
            return (-1,0)

    def update(self, next_pos):
        self.grid[self.agent_pos] = 0
        self.grid[next_pos] = 2
        self.agent_pos = next_pos


if __name__ == '__main__':
    grd = GridWorld(1)
    print(grd.grid)