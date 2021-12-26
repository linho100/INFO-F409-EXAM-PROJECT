from typing import Optional
import numpy as np
from numpy import ndarray
import random

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD


def create_q_table(num_states: int, num_actions: int) -> ndarray:
    """
    Function that returns a q_table as an array of shape (num_states, num_actions) filled with zeros.
    :param num_states: Number of states.
    :param num_actions: Number of actions.
    :return: q_table: Initial q_table.
    """

    q_table = np.zeros((num_states, num_actions))


    model = Sequential()
    model.add(Dense(256, input_shape=(784,), activation="sigmoid"))
    model.add(Dense(4, activation="softmax"))

    sgd = SGD(0.01)
    model.compile(loss="categorical_crossentropy", optimizer=sgd,
        metrics=["accuracy"])
    H = model.fit(trainX, trainY, validation_data=(testX, testY),
        epochs=100, batch_size=128)


    return q_table

class QLearnerAgent:
    """
    The agent class for exercise 1.
    """

    def __init__(self,
                 num_states: int,
                 num_actions: int,
                 learning_rate: float,
                 gamma: float,
                 epsilon_max: Optional[float] = 1,
                 epsilon_min: Optional[float] = 0.01,
                 epsilon_decay: Optional[float] = 0.9):
        """
        :param num_states: Number of states.
        :param num_actions: Number of actions.
        :param learning_rate: The learning rate.
        :param gamma: The discount factor.
        :param epsilon_max: The maximum epsilon of epsilon-greedy.
        :param epsilon_min: The minimum epsilon of epsilon-greedy.
        :param epsilon_decay: The decay factor of epsilon-greedy.
        """
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.q_table = create_q_table(num_states, num_actions)
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.epsilon = epsilon_max

    def greedy_action(self, observation: int) -> int:
        """
        Return the greedy action.
        :param observation: The observation.
        :return: The action.
        """
        return np.argmax(self.q_table[observation,:])

    def act(self, observation: int, training: bool = True) -> int:
        """
        Return the action.
        :param observation: The observation.
        :param training: Boolean flag for training, when not training agent
        should act greedily.
        :return: The action.
        """
        if(not training):
            return self.greedy_action(observation)
        else:
            # Exploration-Exploitation trade-off
            epsilon_rate = random.uniform(0,1)
            if epsilon_rate < self.epsilon:
                return random.randint(0,3)
            else:
                return self.greedy_action(observation)


    def learn(self, obs: int, act: int, rew: float, done: bool, next_obs: int) -> None:
        """
        Update the Q-Value.
        :param obs: The observation.
        :param act: The action.
        :param rew: The reward.
        :param done: Done flag.
        :param next_obs: The next observation.
        """
        self.q_table[obs, act] = self.q_table[obs, act] + self.learning_rate*(rew + self.gamma*np.max(self.q_table[next_obs,:]) - self.q_table[obs, act])
        # Epsilon decay
        if(done):
            self.epsilon = max(self.epsilon*self.epsilon_decay, self.epsilon_min)