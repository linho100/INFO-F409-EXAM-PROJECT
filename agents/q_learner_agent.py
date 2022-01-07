from typing import Optional
from numpy import zeros, ndarray, argmax
from random import randint, uniform


def create_q_table(num_states: int, num_actions: int) -> ndarray:
    """
    Function that returns a q_table as an array of shape (num_states, num_actions) filled with zeros.

    :param num_states: Number of states.
    :param num_actions: Number of actions.
    :return: q_table: Initial q_table.
    """
    return zeros((num_states, num_actions))


class QLearnerAgent:
    """
    The non-communicating Q-learning agent class
    """

    def __init__(self,
                 num_states: int,
                 num_actions: int = 4,
                 learning_rate: float = 1e-3,
                 gamma: float = 0.9,
                 epsilon_max: Optional[float] = 1,
                 epsilon_min: Optional[float] = 0.01,
                 epsilon_decay: Optional[float] = 0.995):
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
        self.n_actions = num_actions
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.epsilon = epsilon_max

    def greedy_action(self, obs: int) -> int:
        """
        Return the greedy action.

        :param observation: The observation.
        :return: The action.
        """
        return argmax(self.q_table[obs])

    def act(self, obs: ndarray, training: bool = True) -> int:
        """
        Return the action.

        :param observation: The observation.
        :param training: Boolean flag for training, when not training agent
        should act greedily.
        :return: The action.
        """
        obs = argmax(obs)

        epsilon_rate = uniform(0, 1)
        # Exploration-Exploitation trade-off
        if (not training) or epsilon_rate > self.epsilon:
            return self.greedy_action(obs)
        else:
            return randint(0, 3)

    def learn(self, obs: ndarray, act: int, rew: float, done: bool, next_obs: ndarray) -> None:
        """
        Update the Q-Value.

        :param obs: The observation.
        :param act: The action.
        :param rew: The reward.
        :param done: Done flag.
        :param next_obs: The next observation.
        """
        obs = argmax(obs)
        next_obs = argmax(next_obs)

        self.q_table[obs, act] += self.learning_rate * \
            (rew + self.gamma *
             max(self.q_table[next_obs, :]) - self.q_table[obs, act])
        if done:
            self.epsilon = max(
                self.epsilon * self.epsilon_decay, self.epsilon_min)
