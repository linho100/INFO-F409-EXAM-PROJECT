from typing import Optional
from numpy import argmax, ndarray, array
from random import uniform, sample as rsample, randint
from collections import deque

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import load_model


class DeepQLearnerAgent:
    """
    The Deep Q-learner agent class
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
                 epsilon_decay: Optional[float] = 0.995,
                 model_path=None):
        """
        :param dim_msg: The dimension of the sender's message.
        :param n_states: The number of states.
        :param n_actions: The number of actions.
        :param learning_rate: The learning rate.
        :param gamma: The discount factor.
        :param epsilon_max: The maximum epsilon of epsilon-greedy.
        :param epsilon_min: The minimum epsilon of epsilon-greedy.
        :param epsilon_decay: The decay factor of epsilon-greedy.
        :param model_path: Path to an existing model used to test or use previously trained model.
        """
        # Init new model
        if model_path is None:
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

        # Load existing model
        else:
            self.epsilon = 0.9  # Useless value
            self.n_states = n_states + dim_msg * n_senders
            self.model = load_model(model_path)

    def create_nnet(self) -> Sequential:
        """
        Function that returns a sequential nnet composed of 3 Dense layers (input-hidden-output) 
        which respectively use, as activation function : 'relu', 'relu' and 'linear'.
        The model is compiled using a mean-squared-error loss function and an RMSProp optimizer.
        :return: The model.
        """
        model = Sequential(
            [
                Dense(24, input_dim=self.n_states,
                      activation='relu', name="input"),
                Dense(24, activation='relu', name="hidden"),
                Dense(self.n_actions, activation='linear', name="output")
            ])
        model.compile(loss='mse', optimizer=RMSprop(
            self.alpha), metrics=["accuracy", "mse"])

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
        :param training: Boolean flag for training, when not training agent should always act greedily.
        :return: The action.
        """
        epsilon_rate = uniform(0, 1)
        # Exploration-Exploitation trade-off
        if (not training) or epsilon_rate > self.epsilon:
            # Greedy action
            return self.greedy_action(obs)
        else:
            # Random action
            return randint(0, 3)

    def learn(self, obs: ndarray, act: int, rew: float, done: bool, next_obs: ndarray) -> None:
        """
        Receives a transition, saves it into the memory, execute replay procedure and update the epsilon if necessary.
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
            self.epsilon = max(
                self.epsilon * self.epsilon_decay, self.epsilon_min)

    def remember(self, obs: ndarray, act: int, rew: float, done: bool, next_obs: ndarray):
        """
        Save a transition into the memory array
        :param obs: The observation.
        :param act: The action.
        :param rew: The reward.
        :param done: Done flag.
        :param next_obs: The next observation.
        """
        self.memory.append([obs, act, rew, done, next_obs])

    def replay(self):
        """
        Updates the model using a random batch extracted from the transaction memory.
        Source: https://pythonprogramming.net/training-deep-q-learning-dqn-reinforcement-learning-python-tutorial/?completed=/deep-q-learning-dqn-reinforcement-learning-python-tutorial/
        """
        # Exit if memory isn't sufficient
        if len(self.memory) < self.batch_size:
            return

        # Get a random minibatch from the memory
        minibatch = rsample(self.memory, self.batch_size)

        # Get current states from minibatch, then query the model for predictions
        current_states = array([transition[0] for transition in minibatch])
        current_qs_list = self.model.predict(current_states)

        # Get future states from minibatch, then query the model for predictions
        new_current_states = array([transition[4] for transition in minibatch])
        future_qs_list = self.model.predict(new_current_states)

        X, y = [], []

        # Enumerate our batches
        for index, (current_state, action, reward, done, _) in enumerate(minibatch):
            # Compute new q_value
            if not done:
                max_future_q = max(future_qs_list[index])
                new_q = self.gamma * max_future_q
            else:
                new_q = reward

            # Update q_value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # Append to the training data
            X.append(current_state)
            y.append(current_qs)

        # Fit on all samples as one batch
        self.model.fit(array(X), array(
            y), batch_size=self.batch_size, verbose=0, shuffle=False)
