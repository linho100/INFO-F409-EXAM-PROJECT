from typing import Optional
from numpy import argmax, ndarray, array
from random import uniform, sample as rsample, randint
from collections import deque

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import Input

class DeepQLearnerAgent:
    """
    The agent class
    """

    def __init__(self,
                 env,
                 dim_msg: int = 0,
                 n_states: int = 25, 
                 n_actions: int = 4, 
                 learning_rate: float = 1e-4,
                 gamma: float = 0.9,
                 epsilon_max: Optional[float] = 1,
                 epsilon_min: Optional[float] = 0.01,
                 epsilon_decay: Optional[float] = 0.999):
        """
        :param learning_rate: The learning rate.
        :param gamma: The discount factor.
        :param epsilon_max: The maximum epsilon of epsilon-greedy.
        :param epsilon_min: The minimum epsilon of epsilon-greedy.
        :param epsilon_decay: The decay factor of epsilon-greedy.
        """
        self.memory  = deque(maxlen=2000)
        self.n_cells = env.row * env.col
        self.n_states = 25#n_states + dim_msg
        self.n_actions = n_actions
        self.batch_size = 15
        self.gamma = gamma
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.epsilon = epsilon_max
        self.learning_rate = learning_rate
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
            ]    
        )

        model.compile(loss='mse', optimizer=RMSprop(self.learning_rate), metrics=["accuracy", "mse"])

        return model

    def greedy_action(self, obs) -> int:
        """
        Return the greedy action.
        :param observation: The observation.
        :return: The action.
        """
        q_values = self.model.predict_step(obs)
        action = argmax(q_values)
        print(f"Q-values {q_values}")
        print(f"Action: {action}")
        return action

    def act(self, obs: ndarray, training: bool = True) -> int:
        """
        Return the action.
        :param observation: The observation.
        :param training: Boolean flag for training, when not training agent
        should act greedily.
        :return: The action.
        """
        obs = obs.reshape(-1,self.n_states)

        if(not training):
            return self.greedy_action(obs)
        else:
            # Exploration-Exploitation trade-off
            epsilon_rate = uniform(0,1)
            if epsilon_rate < self.epsilon:
                return randint(0,3)
            else:
                return self.greedy_action(obs)

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
        history = self.replay()
        # Epsilon decay
        if(done):
            self.epsilon = max(self.epsilon*self.epsilon_decay, self.epsilon_min)

        return history

    def remember(self, obs: ndarray, act: int, rew: float, done: bool, next_obs: ndarray):
        self.memory.append([obs.reshape(-1,self.n_states), act, rew, done, next_obs.reshape(-1,self.n_states)])
        self.memory.append([obs, act, rew, done, next_obs])

    def replay_old(self):
        batch_size = 10
        if len(self.memory) < batch_size: 
            return
        samples = rsample(self.memory, batch_size)
        for sample in samples:
            state, action, reward, done, new_state = sample
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                Q_future = max(self.model.predict(new_state)[0])
                target[0][action] = reward + Q_future * self.gamma
            return self.model.fit(state, target, epochs=1, verbose=0)

    def replay(self):
    
        if len(self.memory) < self.batch_size: 
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = rsample(self.memory, self.batch_size)

        # Get current states from minibatch, then query NN model for Q values
        current_states = array([transition[0] for transition in minibatch])
        current_qs_list = self.model.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = array([transition[4] for transition in minibatch])
        # future_qs_list = self.target_model.predict(new_current_states)
        future_qs_list = self.model.predict(new_current_states)

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, done, new_current_state) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = max(future_qs_list[index])
                new_q = self.gamma * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(array(X), array(y), batch_size=self.batch_size, verbose=0, shuffle=False)
        # print(self.model.summary())
        # print(self.model.weights)
        # input("press enter to continue...")