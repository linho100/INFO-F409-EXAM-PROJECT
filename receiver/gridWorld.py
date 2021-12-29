from typing import Iterable
from random import random, randint
from numpy import zeros
from operator import add

class GridWorld:
    """
    Matrix Game environment.
    """

    def __init__(self, p_term):
        """
            Create a 5 by 5 grid. (y,x) -> origin corresponds to the upper left corner
        """
        self.p_term = p_term
        self.row = 5
        self.col = 5

        self.layouts = [
                            {'name': 'empty', 'walls': []},
                            {'name': 'flower', 'walls': [(0,2),(2,0),(4,2),(2,4)]},
                            {'name': 'two_room', 'walls': [(0,2),(1,2),(3,2),(4,2)]},
                            {'name': 'four_room', 'walls': [(0,2),(2,0),(4,2),(2,4),(2,1),(2,3)]},
                            {'name': 'pong', 'walls': [(1,0),(2,0),(3,0),(0,2),(1,2),(3,2),(4,2),(1,4),(2,4),(3,4)]}
                        ]

    def step(self, action: int):
        """
        :param action: The action to perform
            0 -> N
            1 -> E
            2 -> S
            3 -> W
        """
        p = random()
        
        reward = 0
        done = False

        if p < self.p_term:
            done = True
        else:
            # Compute next_pos based on move
            next_pos = tuple(map(add, self.agent_pos, self.move(action)))
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

    def reset(self, layout: int = 0):
        """
        :param layout: Layouts index from 0 to 4. 5 is random
        :return: One hot encoding of map
        In GridWorld:
            0 -> empty tile
            1 -> wall
            2 -> agent
            3 -> goal
        """
        # Create emtpy grid
        self.grid = zeros((self.row, self.col))

        # Add agent
        self.agent_pos = (self.row//2, self.col//2)
        self.grid[self.agent_pos] = 2

        # Add walls
        if layout == 5:
            layout = randint(0,4)

        layout_walls = self.layouts[layout]["walls"]
        for wall in layout_walls:
            self.grid[wall[0], wall[1]] = 1
        
        # Deal with forbidden positions --> REMOVE?
        self.forbid = [((self.row//2, self.col//2))]

        # Place goal
        self.grid[(3,3)] = 3
        """
        while True:
            self.goal = (randint(0,self.row-1), randint(0,self.col-1))
        if(self.goal not in self.forbid):
            self.grid[self.goal] = 3
        break
        """

        return self.one_hot_encoding()

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
                elif self.grid[x,y] == 1:
                    line += u"\u25A1 "
                elif self.grid[x,y] == 2:
                    line += u"\u2588 "
                elif self.grid[x,y] == 3:
                    line += "X "
            print(line + "|")
        print(" ------------")