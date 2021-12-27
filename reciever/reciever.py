from typing import Optional
import numpy as np
from numpy import ndarray
import random

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend
import tensorflow as tf


def create_nnet(n_cells: int, msg_dim: int) -> tf.keras.model.Sequential:
    """
    Function that returns an nnet
    :param n_cells: ....
    :param msg_dim: ....
    :return: nnet: ...
    """

    model = Sequential()
    model.add(Dense(input_shape=(n_cells + msg_dim,), activation="sigmoid"))
    model.add(Dense(4, activation="softmax")) # Number of actions

    model.compile(loss=custom_loss(), optimizer=Adam(learning_rate=0.001), metrics=["accuracy"])

    return model

class QLearnerAgent:
    """
    The agent class
    """

    def __init__(self,
                 learning_rate: float,
                 gamma: float,
                 epsilon_max: Optional[float] = 1,
                 epsilon_min: Optional[float] = 0.01,
                 epsilon_decay: Optional[float] = 0.9):
        """
        :param learning_rate: The learning rate.
        :param gamma: The discount factor.
        :param epsilon_max: The maximum epsilon of epsilon-greedy.
        :param epsilon_min: The minimum epsilon of epsilon-greedy.
        :param epsilon_decay: The decay factor of epsilon-greedy.
        """
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.epsilon = epsilon_max
        self.model = create_nnet()

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

    def custom_loss(self):
        return backend.square(2)

    def learn(self, obs: int, act: int, rew: float, done: bool, next_obs: int) -> None:
        """
        Update the Q-Value.
        :param obs: The observation.
        :param act: The action.
        :param rew: The reward.
        :param done: Done flag.
        :param next_obs: The next observation.
        """

        H = self.model.fit(trainX, trainY, validation_data=(testX, testY), epochs=100, batch_size=128)

        # Epsilon decay
        if(done):
            self.epsilon = max(self.epsilon*self.epsilon_decay, self.epsilon_min)


            
            
class CustomAccuracy(tf.keras.losses.Loss):
        
    def __init__(self):

        super().__init__()

    def call(self, y_true, y_pred):

        mse= tf.reduce_mean(tf.square(y_pred-y_true))

        rmse= tf.math.sqrt(mse)

        return rmse / tf.reduce_mean(tf.square(y_true)) - 1

    model.compile(optimizer=Adam(learning_rate=0.001), loss=CustomAccuracy(), metrics=['mae', 'mse'])