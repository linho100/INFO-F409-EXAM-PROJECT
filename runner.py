from typing import Iterable, Tuple
from math import floor, exp, log
from numpy import arange, concatenate, delete
from tqdm import tqdm
from csv import writer as csv_writer
from os import mkdir

from sender import SenderAgent
from receiver import DeepQLearnerAgent
from gridWorld import GridWorld


def run_episode(env: GridWorld, receiver: DeepQLearnerAgent, senders_list: Iterable[SenderAgent], training: bool, layout_type: int) -> float:
    """
    Interact with the environment for one episode using actions derived from the nnet.
    :param env: The environment.
    :param receiver: The receiver agent.
    :param sender: The sender agent.
    :param training: If true the q_table will be updated using q-learning. The flag is also passed to the action selector.
    :return: The average cumulated reward.
    """
    done = False
    obs = env.reset(layout_type)
    context_vector = env.one_hot_enc_goal()
    step = 0
    messages_encoded = [-1]
    senders_nb = len(senders_list)

    for i in range(senders_nb):
        senders_list[i].goal_location = context_vector
        message = senders_list[i].send_message()
        messages_encoded = concatenate((messages_encoded, message)) #store the messages sent
        obs = concatenate((obs, message)) #TODO: implement msg from the i-th sender

    messages_encoded = delete(messages_encoded, 0)

    while not done:
        action = receiver.act(obs, training)
        obs_prime, reward, done = env.step(action)
        obs_prime = concatenate((obs_prime, messages_encoded)) #TODO: implement msg from senders

        # Receiver learning
        if training:
            receiver.learn(obs, action, reward, done, obs_prime)

        obs = obs_prime
        step += 1

    # Senders learning
    if training:
        for i in range(senders_nb):
            senders_list[i].learn(reward)

    return reward/step

def train(env: GridWorld, num_episodes: int, gamma: float, channel_capacity: int, senders_nb: int, epsilon_s: float, epsilon_r: float, layout_type: int) -> Tuple[list, DeepQLearnerAgent]:
    """
    Training loop.
    :param env: The environment.
    :param num_episodes: The number of episodes.
    :return: ...
    """
    receiver = DeepQLearnerAgent(n_senders=senders_nb, epsilon_decay = 1-epsilon_r, gamma=gamma)
    messages_nb = floor(exp(log(channel_capacity)/senders_nb))
    senders_list = list()

    for i in range(senders_nb):
        senders_list.append(SenderAgent(epsilon_s, messages_nb))

    avg_rewards_list = []
    for i in tqdm(range(num_episodes)):
        if i % 50 == 0:
            print("Episode {} of {}".format(i + 1, num_episodes))
            for j in range(15):
                # evaluation
                run_episode(env, receiver, senders_list, training=False, layout_type=layout_type)
                i += j
        avg_rewards_list.append(run_episode(env, receiver, senders_list, training=True, layout_type=layout_type))

    return avg_rewards_list, receiver.model, [s.model for s in senders_list]

def experiment_1(num_episodes, gamma, epsilon_s, epsilon_r, layout_type, channel_capacity):
    for senders_nb in range(1,6):
        env = GridWorld(p_term=1-gamma)
        rewards, receiver_model, senders_models = train(env, num_episodes, gamma, channel_capacity, senders_nb, epsilon_s, epsilon_r, layout_type)

        # Save results to csv file
        save_results(rewards, receiver_model, senders_models, layout=layout_type, experiment_number=1, subtitle=f"n_{senders_nb}")

def experiment_2(num_episodes, gamma, epsilon_s, epsilon_r, layout_type, senders_nb):
    for channel_capacity in [3,4,5,8,9,16,25,27,32]:
        env = GridWorld(p_term=1-gamma)
        rewards, receiver_model, senders_models = train(env, num_episodes, gamma, channel_capacity, senders_nb, epsilon_s, epsilon_r, layout_type)

        # Save results to csv file
        save_results(rewards, receiver_model, senders_models, layout=layout_type, experiment_number=2, subtitle=f"c_{channel_capacity}")

def save_results(data, receiver_model, senders_models, layout, experiment_number, subtitle):
    # Save data to csv
    with open(f"./experiments/experiment_{experiment_number}_{subtitle}.csv", 'a') as file:
        writer = csv_writer(file)
        writer.writerows(data)
    
    # Save models
    folder_name = f"experiment_{experiment_number.to_s}_layout_{layout}"

    mkdir(f"./experiments/{folder_name}")
    mkdir(f"./experiments/{folder_name}/senders")

    receiver_model.save(f"/models/{folder_name}/r_{subtitle}")
    
    counter = 0
    for sender_model in senders_models:
        sender_model.save(f"/models/{folder_name}/s_{subtitle}_{counter}")
        counter += 1

if __name__ == '__main__':
    num_episodes = int(1e4)
    gamma = 0.8
    epsilon_s = 0.005
    epsilon_r = 0.005

    layout_type = 3 # [Sara = Pong(4), Linh = 4-four(3), Ilyes = 2-room(2), JF = flower(1)]

    experiment_1(num_episodes, gamma, epsilon_s, epsilon_r, layout_type, channel_capacity=16)
    # experiment_2(num_episodes, gamma, epsilon_s, epsilon_r, layout_type, channel_capacities=[3,4,5,8,9,16,25,27,32,36,64], senders_nb=3)
    