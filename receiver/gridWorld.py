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
        
        reward = 0
        done = False

        if p < self.p_term:
            done = True
        else:
            # Compute next_pos based on move
            next_pos = tuple(map(operator.add, self.agent_pos, self.move(action)))
            # Bound next_pos
            next_pos = (max(min(4, next_pos[0]), 0), max(min(4, next_pos[1]), 0))
            
            if(self.grid[next_pos] == 0):
                self.update(next_pos)
            elif(self.grid[next_pos] == 3):
                reward = 1
                done = True
                self.update(next_pos)

        return self.one_hot_encoding(), reward, done
            

    def move(self, action):
        if(action == 0):
            print("Action N")
            return (-1,0)
        elif(action == 1):
            print("Action E")
            return (0,1)
        elif(action == 2):
            print("Action S")
            return (1,0)
        else:
            print("Action W")
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

    def print_grid(self):
        print(" ------------")
        for x in range(self.col):
            line = " |"
            for y in range(self.row):
                if self.grid[x,y] == 0:
                    line += "  "
                elif self.grid[x,y] == 2:
                    line += u"\u2588 "
                elif self.grid[x,y] == 3:
                    line += "X "
            print(line + "|")
        print(" ------------")