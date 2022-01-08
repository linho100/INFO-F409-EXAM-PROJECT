from typing import Iterable

from numpy import insert, zeros, ndarray, array, argmax, random
from numpy.core.records import array

from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
from tensorflow.keras.models import load_model


def convert_message_to_binary(message: int) -> Iterable[int]:
    """
    Convert the message from its decimal representation to binary one in an array.
    :param message: Message as a decimal integer.
    :return: Binary representation in an array.
    """
    max_bits_nb = 5
    binary = str(format(message, 'b'))
    encoded_message = zeros(len(binary))

    for bit in range(len(binary)):
        if int(binary[bit]) == 1:
            encoded_message[bit] = 1
    if len(encoded_message) < max_bits_nb:
        for _ in range(max_bits_nb - len(encoded_message)):
            encoded_message = insert(encoded_message, 0, 0)

    return encoded_message


class SenderAgent:
    """
    The sender agent class that sends a message to the receiver
    in order to find the goal location. 
    It uses a single layer feed forward network to predict the message to send to the receiver.
    """

    def __init__(self,
                 epsilon: float = 0.9,
                 messages_nb: int = 3,
                 goal_location: ndarray = [0, 0],
                 model_path=None):
        """
        :param epsilon: The exploration rate.
        :param messages_nb: The number of messages the sender agent can generate.
        :param goal_location: The goal location on the grid world.
        :param model_path: Path to an existing model used to test or use previously trained model.
        """
        # Init new model
        if model_path is None:
            self.goal_location = goal_location
            self.epsilon = epsilon
            self.messages_nb = messages_nb
            self.reward = 0
            self.set_messages = 0
            self.message_sent = 0

            # Build the single layer model
            self.model = Sequential(
                [
                    Dense(24, input_shape=(25,),
                          activation='relu', name="input"),
                    Dense(self.messages_nb, activation='softmax', name="output")
                ]
            )
            self.model.compile(optimizer='adam', loss='mse')

        # Load existing model
        else:
            self.epsilon = 0.9  # Useless value
            self.messages_nb = messages_nb
            self.set_messages = 0
            self.message_sent = 0
            self.model = load_model(model_path)

    def send_message(self, training: bool = True) -> int:
        """
        Method that returns a message from a given set of possible messages.
        If a greedy-action is choosen, the message corresponding to the best value outputted by the feed-forward neural network is choosen.
        Otherwise, a random message is used.
        :return: The message.
        """

        inputs = self.goal_location.reshape(1, -1)
        # Outputs of the FNN containing all the q-values of the messages that the sender can emit.
        outputs = self.model.predict(array(inputs))
        self.set_messages = outputs

        if(not training) or random.rand() > self.epsilon:
            # Greedy message
            message = argmax(self.set_messages)
        else:
            # Random message
            message = random.randint(self.messages_nb)

        self.message_sent = message
        return convert_message_to_binary(message)

    def learn(self, reward: int) -> None:
        """
        Learning method to update the feed-forward neural network based on:
            - the goal location received at initialization;
            - the message sent;
            - the reward obtained at the end of the episode.
        :param reward: The reward received.
        """
        target = self.model.predict(array(self.goal_location.reshape(1, -1)))
        target[0][self.message_sent] = reward

        self.model.fit(array(self.goal_location.reshape(1, -1)),
                       target, epochs=1, verbose=0)
