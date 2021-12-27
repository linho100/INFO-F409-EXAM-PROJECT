from typing import Optional
import numpy as np
from numpy import ndarray
import random

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend
import tensorflow as tf


def create_nnet(n_cells: int, msg_dim: int, learning_rate: float) -> tf.keras.model.Sequential:
    """
    Function that returns an nnet
    :param n_cells: ....
    :param msg_dim: ....
    :return: nnet: ...
    """

    model = Sequential()
    model.add(Dense(input_shape=(n_cells + msg_dim,), activation="sigmoid"))
    model.add(Dense(4, activation="softmax")) # Number of actions

    model.compile(loss='mse', optimizer=RMSprop(learning_rate), metrics=["accuracy"])

    return model

class DeepQLearnerAgent:
    """
    The agent class
    """

    def __init__(self,
                 env,
                 dim_msg,
                 learning_rate: float = 0.1,
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
        self.memory  = deque(maxlen=2000)

        self.gamma = gamma
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.epsilon = epsilon_max
        self.learning_rate = learning_rate
        self.model = create_nnet(env.row * env.col, dim_msg, self.learning_rate)

    def greedy_action(self, observation) -> int:
        """
        Return the greedy action.
        :param observation: The observation.
        :return: The action.
        """
        outputs = self.model.predict(observation)
        return np.argmax(outputs)

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
        self.remember(obs, act, rew, done, next_obs)
        self.replay()
        # Epsilon decay
        if(done):
            self.epsilon = max(self.epsilon*self.epsilon_decay, self.epsilon_min)

    def remember(self, obs: int, act: int, rew: float, done: bool, next_obs: int):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        batch_size = 10
        if len(self.memory) < batch_size: 
            return
        samples = random.sample(self.memory, batch_size)
        for sample in samples:
            state, action, reward, new_state, done = sample
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                Q_future = max(
                    self.model.predict(new_state)[0])
                target[0][action] = reward + Q_future * self.gamma
            self.model.fit(state, target, epochs=1, verbose=0)
