import numpy as np
import tensorflow as tf


def loss_function(reward: int, set_messages: list, message_sent: int) -> float:
    """
    Method that returns the loss for a sender agent.

    :param reward: The reward received.
    :param set_messages: The possible messages.
    :param message_sent: The message sent by the sender agent.
    :return: The loss value of the sender.
    """
    def loss(y_true, y_pred):
        loss_value = (reward - set_messages[message_sent]) ** 2
        
        return loss_value

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
        self.reward = 0
        self.set_messages = 0
        self.message_sent = 0
        
        # Build the single layer FNN
        # inputs = tf.keras.Input(shape = (25,))
        # outputs = tf.keras.layers.Dense(self.messages_nb, activation=tf.nn.relu)(inputs)
        # self.fnn_model = tf.keras.Model(inputs=inputs, outputs=outputs)
        self.fnn_model = tf.keras.Sequential()
        self.fnn_model.add(tf.keras.layers.InputLayer(batch_input_shape=(25, 1)))
        self.fnn_model.add(tf.keras.layers.Dense(self.messages_nb, activation='softmax'))
        self.fnn_model.compile(optimizer='adam', loss=loss_function(self.reward, self.set_messages, self.message_sent))

        pass


    def send_message(self) -> int:
        """
        Method that returns a message from a given set of possible messages which
        the action-values are output from a feed-forward neural network.
        
        :return: The message.
        """
        inputs = self.goal_location#.flatten()
        #Outputs of the FNN containing all the q-values of the messages that the sender can emit.
        outputs = self.fnn_model.predict(inputs)
        self.set_messages = outputs[0]

        if np.random.rand() < self.epsilon:
            message = np.random.randint(self.messages_nb)
        else:
            message = np.argmax(self.set_messages)

        self.message_sent = message
        print(self.set_messages)
        print("message sent: ", message)
        return message


    def learn(self, reward: int, message_sent: int) -> None:
        """
        Learning method to update the feed-forward neural network.

        :param reward: The reward received.
        :param message_sent: The message sent during the current episode.
        """
        if reward == 1: #to check
            inputs = np.array(self.goal_location).reshape(-1,25) #.flatten()
            outputs = self.set_messages.reshape(-1, int(self.messages_nb))

            self.fnn_model.fit(inputs, outputs, epochs=1, verbose=0)
        

