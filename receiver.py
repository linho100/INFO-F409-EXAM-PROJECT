from typing import Optional
from numpy import argmax, ndarray, array
from random import uniform, sample as rsample, randint
from collections import deque

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop


class DeepQLearnerAgent:
    """
    The agent class
    """

    def __init__(self,
                 dim_msg: int = 5,
                 n_senders: int = 1,
                 n_states: int = 25,
                 n_actions: int = 4,
                 learning_rate: float = 1e-3,
                 gamma: float = 0.95,
                 epsilon_max: Optional[float] = 1,
                 epsilon_min: Optional[float] = 0.01,
                 epsilon_decay: Optional[float] = 0.995):
        """
        :param dim_msg: The dimension of the sender's message.
        :param n_states: The number of states.
        :param n_actions: The number of actions.
        :param learning_rate: The learning rate.
        :param gamma: The discount factor.
        :param epsilon_max: The maximum epsilon of epsilon-greedy.
        :param epsilon_min: The minimum epsilon of epsilon-greedy.
        :param epsilon_decay: The decay factor of epsilon-greedy.
        """
        self.n_states = n_states + dim_msg * n_senders
        self.n_actions = n_actions

        self.alpha = learning_rate
        self.gamma = gamma

        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.epsilon = epsilon_max

        self.batch_size = 15
        self.memory = deque(maxlen=2000)
        self.model = self.create_nnet()

    def create_nnet(self) -> Sequential:
        """
        Function that returns an nnet
        :return: nnet: ...
        """
        model = Sequential(
            [
                Dense(24, input_dim=self.n_states, activation='relu', name="input"),
                Dense(24, activation='relu', name="hidden"),
                Dense(self.n_actions, activation='linear', name="output")
            ])
        model.compile(loss='mse', optimizer=RMSprop(self.alpha), metrics=["accuracy", "mse"])

        return model

    def greedy_action(self, obs) -> int:
        """
        Return the greedy action.
        :param obs: The observation.
        :return: The action.
        """
        return argmax(self.model.predict_step(obs.reshape(-1, self.n_states)))

    def act(self, obs: ndarray, training: bool = True) -> int:
        """
        Return the action.
        :param obs: The observation.
        :param training: Boolean flag for training, when not training agent
        should act greedily.
        :return: The action.
        """
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
        self.remember(obs, act, rew, done, next_obs)
        self.replay()

        # Epsilon decay
        if (done):
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def remember(self, obs: ndarray, act: int, rew: float, done: bool, next_obs: ndarray):
        self.memory.append([obs, act, rew, done, next_obs])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = rsample(self.memory, self.batch_size)
        # Get current states from minibatch, then query NN model for Q values
        current_states = array([transition[0] for transition in minibatch])
        # print("Transition: ")
        # print('len: ', len(current_states))
        # for transition in current_states:
        #     print(transition)
        current_qs_list = self.model.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = array([transition[4] for transition in minibatch])
        future_qs_list = self.model.predict(new_current_states)

        X, y = [], []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, done, _) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = max(future_qs_list[index])
                new_q = self.gamma * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(array(X), array(y), batch_size=self.batch_size, verbose=0, shuffle=False)
