from typing import Iterable
import random
from numpy import zeros
import operator


class GridWorld:
    """
    Matrix Game environment.
    """

    def __init__(self, p_term):
        self.p_term = p_term
        self.row = 5
        self.col = 5

    def step(self, action: int):
        """
        Action:
        0 -> N
        1 -> E
        2 -> S
        3 -> W
        """
        p = random.random()
        if p < self.p_term:
            return self.one_hot_encoding(), 0, True
        else:
            next_pos = tuple(map(operator.add, self.agent_pos, self.move(action)))
            try:
                if(self.grid[next_pos] == 0):
                    self.update(next_pos)
                elif(self.grid[next_pos] == 3):
                    self.update(next_pos)
                    encode = self.one_hot_encoding()
                    return encode, 1, True
            except IndexError:
                pass

            encode = self.one_hot_encoding()

            return encode, 0, False
            

    def move(self, action):
        if(action == 0):
            return (-1,0)
        elif(action == 1):
            return (0,1)
        elif(action == 2):
            return (1,0)
        else:
            return (0,-1)

    def update(self, next_pos):
        self.grid[self.agent_pos] = 0
        self.grid[next_pos] = 2
        self.agent_pos = next_pos

    def reset(self):
        """
        In GridWorld:
        0 -> empty tile
        1 -> wall
        2 -> agent
        3 -> goal
        """
        self.grid = zeros((self.row, self.col))
        self.agent_pos = (self.row//2, self.col//2)
        self.grid[self.agent_pos] = 2
        self.forbid = [((self.row//2, self.col//2))]
        """
        while True:
            self.goal = (random.randint(0,self.row-1), random.randint(0,self.col-1))
            if(self.goal not in self.forbid):
                self.grid[self.goal] = 3
                break
        """
        self.goal = (2,3)
        self.grid[self.goal] = 3

        encode = self.one_hot_encoding()

        return encode

    def one_hot_encoding(self):
        encoding = zeros(self.row * self.col)
        for i in range(self.col):
            for j in range(self.row):
                if self.grid[i,j] == 2:
                    encoding[i*self.row + j] = 1
                    
        return encoding