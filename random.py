from random import sample as randint

class randomAgent:
    """
    The agent class
    """

    def __init__(self):
        """
        """
        self.n_action = 4 


    def act(self) -> int:
        """
        Return the action.
        :param obs: The observation.
        :param training: Boolean flag for training, when not training agent
        should act greedily.
        :return: The action.
        """
        return randint(0, 3)  
