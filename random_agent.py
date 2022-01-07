from random import randint


class RandomAgent:
    """
    The agent class
    """

    def __init__(self):
        self.n_action = 4 


    def act(self) -> int:
        """
        Return the action.
        :param obs: The observation.
        :param training: Boolean flag for training, when not training agent
        should act greedily.
        :return: The action.
        """
        return randint(0, self.n_action - 1)  

