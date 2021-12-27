import numpy as np
import tensorflow as tf


def loss_function(reward: int) -> float:
    """
    Method that returns the loss for a sender agent.
    :param reward: The reward received.
    """
    loss = (reward - ) ** 2

    return loss



class SenderAgent:
    """
    The sender agent class that sends a message to the receiver
    in order to find the goal location.
    """

    def __init__(self, goal_location: np.ndarray, epsilon: float, messages_nb: int):
        """
        :param goal_location: The goal location on the grid world.
        :param epsilon: The action exploration rate.
        :param messages_nb: The number of messages the sender agent can generate.
        """
        self.goal_location = goal_location
        self.epsilon = epsilon
        self.messages_nb = messages_nb
        self.loss = 0
        self.rewards = 0
        
        # Build the single layer FNN
        # inputs = tf.keras.Input(shape = (25,))
        # outputs = tf.keras.layers.Dense(self.messages_nb, activation=tf.nn.relu)(inputs)
        # self.fnn_model = tf.keras.Model(inputs=inputs, outputs=outputs)
        self.fnn_model = tf.keras.Sequential()
        self.fnn_model.add(tf.keras.layers.Dense(self.messages_nb, input_dim=25, activation='softmax'))

        pass


    def send_message(self) -> int:
        """
        Method that returns a message from a given set of possible messages
        output from a feed-forward neural network.
        
        :return: The message.
        """
        # Implementation of action value estimation function
        # as a single layer feed-forward neural network param by theta
        input = self.goal_location.flatten()

        # outputs = outputs of the FFN which is a list of messages_nb size
        # containing all the messages that the sender can emit.
        # set_messages = outputs

        if np.random.rand() < self.epsilon:
            message = np.random.choice(set_messages)
        else:
            message = 0
            # message = best message in set_messages
        
        return message


    def learn(self, reward: int) -> None:
        """
        Learning method to update the rewards of the sender agent.

        :param reward: The reward received.
        """
        self.rewards += reward
        # Q = estimation function implemented as a single layer FNN
        # self.loss = (reward - Q) ** 2
        

