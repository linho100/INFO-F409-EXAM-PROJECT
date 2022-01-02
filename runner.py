from typing import Iterable, Tuple
from numpy import arange, concatenate, delete
import matplotlib.pyplot as plt
from tqdm import tqdm

from sender import SenderAgent
from receiver import DeepQLearnerAgent
from gridWorld import GridWorld


def run_episode(env: GridWorld, receiver: DeepQLearnerAgent, senders_list: Iterable[SenderAgent], training: bool, print_progress: bool) -> float:
    """
    Interact with the environment for one episode using actions derived from the nnet.
    :param env: The environment.
    :param receiver: The receiver agent.
    :param sender: The sender agent.
    :param training: If true the q_table will be updated using q-learning. The flag is also passed to the action selector.
    :return: The average cumulated reward.
    """
    done = False
    obs = env.reset()
    context_vector = env.one_hot_enc_goal()
    cum_reward = 0.
    t = 0
    messages_encoded = [-1]

    for i in range(senders_nb):
        senders_list[i].goal_location = context_vector
        message = senders_list[i].send_message()
        print('MESSAGE: ', message)
        messages_encoded = concatenate((messages_encoded, message)) #store the messages sent
        obs = concatenate((obs, message)) #TODO: implement msg from the i-th sender

    messages_encoded = delete(messages_encoded, 0)
    receiver.update_states(len(messages_encoded))
    receiver.model = receiver.create_nnet()

    if print_progress is True:
        print(env)
    # print("HERE: ", len(obs), len(messages_encoded), receiver.n_states)
    while not done:
        action = receiver.act(obs, training)
        obs_prime, reward, done = env.step(action)
        # print("HERE 2: ", len(obs_prime))
        obs_prime = concatenate((obs_prime, messages_encoded)) #TODO: implement msg from senders
        # print("HERE 3: ", len(obs_prime), len(messages_encoded))
        if training:
            receiver.learn(obs, action, reward, done, obs_prime)
        obs = obs_prime
        
        if print_progress is True:
            if reward == 1:
                print("WIN")
                print(env)

            if not done:
                print(env)

        cum_reward += reward
        t += 1
    print("EPISODE DONE")
    if training:
        print("TRAINING")
        for i in range(senders_nb):
            senders_list[i].learn(reward)
    if print_progress is True: print("DONE")

    return cum_reward/t

def train(env: GridWorld, num_episodes: int, gamma: float, channel_capacity: int, senders_nb: int, epsilon_s: float) -> Tuple[list, DeepQLearnerAgent]:
    """
    Training loop.
    :param env: The environment.
    :param num_episodes: The number of episodes.
    :return: ...
    """
    receiver = DeepQLearnerAgent()
    messages_nb = (channel_capacity) ** (1/senders_nb)
    senders_list = list()

    for i in range(senders_nb):
        senders_list.append(SenderAgent(epsilon_s, messages_nb))

    avg_rewards_list = []
    for i in tqdm(range(num_episodes)):
        if i % 50 == 0:
            print("Episode {} of {}".format(i + 1, num_episodes))
            for j in range(15):
                # evaluation
                run_episode(env, receiver, senders_list, False, True)
                i += j
        avg_rewards_list.append(run_episode(env, receiver, senders_list, True, False))

    return avg_rewards_list, receiver


if __name__ == '__main__':
    num_episodes = 100
    gamma = 0.9
    epsilon_s = 0.01
    epsilon_r = None

    channel_capacity = 16
    senders_nb = 2

    env = GridWorld(p_term=1-gamma)
    rewards, agent = train(env, num_episodes, gamma, channel_capacity, senders_nb, epsilon_s)
      
    plt.plot(arange(1, len(rewards)+1), rewards)

    plt.title('Performance of DQLagent')
    plt.xlabel('episode')
    plt.ylabel('average reward per episode')
    plt.show()