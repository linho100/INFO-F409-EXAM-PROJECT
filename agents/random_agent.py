from random import randint


class RandomAgent:
    """
    The random agent class
    """

    def __init__(self):
        self.n_action = 4

    def act(self) -> int:
        """
        Return a random action.
        :return: The action.
        """
        return randint(0, self.n_action - 1)
