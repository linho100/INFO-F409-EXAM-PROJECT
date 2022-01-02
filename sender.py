from typing import Iterable

from numpy import insert, zeros, ndarray, array, random, argmax
from numpy.core.records import array

from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential

def one_hot_encoding(message: int) -> Iterable[int]:
    max_bits_nb = 5
    binary = str(format(message, 'b'))
    encoded_message = zeros(len(binary))
    
    for bit in range(len(binary)):
        if int(binary[bit]) == 1:
            encoded_message[bit] = 1
    # encoded_message = encoded_message.astype(int)
    # Padding?
    if len(encoded_message) < max_bits_nb:
        for i in range(max_bits_nb - len(encoded_message)):
            encoded_message = insert(encoded_message, 0, 0)
    # print('encoded_message: ', encoded_message)

    return encoded_message

class SenderAgent:
    """
    The sender agent class that sends a message to the receiver
    in order to find the goal location.
    """

    def __init__(self, epsilon: float, messages_nb: int, goal_location: ndarray = [0, 0]):
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
        self.model = Sequential(
            [
                Dense(24, input_shape=(25,), activation='relu', name="input"),
                Dense(self.messages_nb, activation='softmax', name="output")
            ]
        )

        self.model.compile(optimizer='adam', loss='mse')

    def send_message(self) -> int:
        """
        Method that returns a message from a given set of possible messages which
        the action-values are output from a feed-forward neural network.
        
        :return: The message.
        """
        inputs = self.goal_location.reshape(1, -1)
        #Outputs of the FNN containing all the q-values of the messages that the sender can emit.
        outputs = self.model.predict(array(inputs))
        self.set_messages = outputs
        # print(self.set_messages)

        if random.rand() < self.epsilon:
            message = random.randint(self.messages_nb)
        else:
            message = argmax(self.set_messages)
        self.message_sent = message

        # one hot-encoding
        encoded_message = one_hot_encoding(message)
        
        return encoded_message


    def learn(self, reward: int) -> None:
        """
        Learning method to update the feed-forward neural network.

        :param reward: The reward received.
        """
        target = self.model.predict(array(self.goal_location.reshape(1, -1)))
        # print(target)
        target[0][self.message_sent] = reward

        # if reward == 1: #to check
        #     inputs = array(self.goal_location).reshape(-1,25)
        #     outputs = self.set_messages.reshape(-1, int(self.messages_nb))

        self.model.fit(array(self.goal_location.reshape(1, -1)), target, epochs=1, verbose=0)
        