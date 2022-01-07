import numpy as np
from numpy import ndarray


def create_q_table(num_states: int, num_actions: int) -> ndarray:
    """
    Function that returns a q_table as an array of shape (num_states, num_actions) filled with zeros.

    :param num_states: Number of states.
    :param num_actions: Number of actions.
    :return: q_table: Initial q_table.
    """
    q_table = np.zeros((num_states, num_actions))

    return q_table


class QLearnerAgent:
    """
    The non-communicating Q-learning agent class
    """

    def __init__(self, 
                num_states: int,
                num_actions: int,
                learning_rate: float,
                gamma: float,
                epsilon: float) -> None:
        """
        :param num_states: Number of states.
        :param num_actions: Number of actions.
        :param learning_rate: The learning rate.
        :param gamma: The discount factor.
        :param epsilon: The epsilon of epsilon-greedy.
        """
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.q_table = create_q_table(num_states, num_actions)
        self.epsilon = epsilon
    
    def greedy_action(self, observation: int) -> int:
        """
        Return the greedy action.

        :param observation: The observation.
        :return: The action.
        """
        greedy_action = self.q_table[observation].argmax()

        return greedy_action

    def act(self, observation: ndarray, training: bool = True) -> int:
        """
        Return the action.

        :param observation: The observation.
        :param training: Boolean flag for training, when not training agent
        should act greedily.
        :return: The action.
        """
        if training:
            # Based on the observation and q-table
            if np.random.rand() <= self.epsilon:
                action = np.random.randint(0, 4)
            else:
                action = self.greedy_action(observation)
        else:
            action = self.greedy_action(observation)

        return action

    def learn(self, obs: ndarray, act: int, rew: float, done: bool, next_obs: ndarray) -> None:
        """
        Update the Q-Value.

        :param obs: The observation.
        :param act: The action.
        :param rew: The reward.
        :param done: Done flag.
        :param next_obs: The next observation.
        """
        self.q_table[obs, act] += self.learning_rate * (rew + self.gamma * self.q_table[next_obs].max() - self.q_table[obs, act])