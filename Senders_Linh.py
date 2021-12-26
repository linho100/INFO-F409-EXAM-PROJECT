import numpy as np


def vocabulary_generation(vocabulary_size: int) -> list(int):
    pass



class SenderAgent:
    """
    The sender agent class that sends a message to the receiver
    in order to find the goal location.
    """

    def __init__(self, grid_world: np.ndarray, goal_location: np.ndarray, 
                    epsilon: float, vocabulary: list(int)):
        """
        :param grid_world: The grid world.
        :param goal_location: The goal location on the grid world.
        :param epsilon: The action exploration rate.
        :param vocabulary: The given vocabulary.
        """
        self.grid_world = grid_world
        self.goal_location = goal_location
        self.epsilon = epsilon
        self.vocabulary = vocabulary
        self.loss = 0
        self.rewards = 0
        pass


    def send_message(self, set_messages: list(int)) -> list(int):
        """
        Method that returns a message from a given set of possible messages.
        
        :param set_messages: The given set of possible messages.
        :return: The message.
        """
        if np.random.rand() < self.epsilon:
            message = np.random.choice(set_messages)
        else:
            message = 0
            # message = best message in set_messages
        
        return message


    def learn(self, reward: int) -> None:
        """
        Learning method to update the rewards of the sender agent.

        :param reward: The reward received.
        """
        self.rewards += reward
        # Q = estimation function implemented as a single layer FFNN
        # self.loss = (reward - Q) ** 2
        pass

