from numpy import concatenate, delete
from math import floor, exp, log

from gridWorld import GridWorld
from .agents.sender_agent import SenderAgent
from .agents.receiver_agent import DeepQLearnerAgent


def main(layout, senders_nb, channel_capacity, senders_filepath, receiver_filepath):
    env = GridWorld(0.0001)
    messages_nb = floor(exp(log(channel_capacity) / senders_nb))

    senders = [SenderAgent(messages_nb=messages_nb, model_path=sender_filepath)
               for sender_filepath in senders_filepath]
    receiver = DeepQLearnerAgent(
        dim_msg=5, n_states=25, n_senders=senders_nb, model_path=receiver_filepath)

    for _ in range(20):
        # Reset grid
        env.reset(layout)
        print("----------------------- NEW GAME -----------------------")
        print(env)

        done = False
        step = 0

        # Senders
        messages_encoded = [-1]
        for sender in senders:
            sender.goal_location = env.one_hot_enc_goal()
            message = sender.send_message(training=False)
            messages_encoded = concatenate((messages_encoded, message))

        messages_encoded = delete(messages_encoded, 0)
        print(f"MESSAGES ENCODED : {messages_encoded}")

        # Receiver
        while not done and step < 25:
            obs = concatenate((env.one_hot_enc_player(), messages_encoded))
            action = receiver.act(obs, training=False)
            _, _, done = env.step(action)
            print(env)
            step += 1

        print(f'----------------DONE in {step} steps ----------------')


if __name__ == '__main__':
    main(layout=1,
         senders_nb=1,
         channel_capacity=16,
         senders_filepath=[
             "./experiments/experiment_1_layout_1/models/senders/s_n_1_0"
         ],
         receiver_filepath="./experiments/experiment_1_layout_1/models/receivers/r_n_1")
