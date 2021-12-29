import numpy as np
from numpy.core.numeric import _moveaxis_dispatcher
from senders_Linh import SenderAgent


def run_process(channel_capacity: int, senders_nb: int, context_vector: list, epsilon: float):
    messages_nb = (channel_capacity) ** (1/senders_nb)
    print(messages_nb)
    senders_list = list()

    for i in range(senders_nb):
        senders_list.append(SenderAgent(context_vector, epsilon, messages_nb))
    
    messages_list = list()
    for i in range(senders_nb):
        message = senders_list[i].send_message()
        messages_list.append(message)

    #Simulation of the end of an episode
    reward = 1
    
    for i in range(senders_nb):
        senders_list[i].learn(reward, messages_list[i])



if __name__ == "__main__":
    channel_capacity = 16
    senders_nb = 1
    context_vector = [0 for i in range(24)]
    context_vector.insert(0, 1)
    epsilon = 0.01

    run_process(channel_capacity, senders_nb, context_vector, epsilon)