from typing import Tuple
from numpy import arange, zeros, concatenate
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import sleep

from reciever import DeepQLearnerAgent
from gridWorld import GridWorld


def run_episode(env: GridWorld, agent: DeepQLearnerAgent, training: bool) -> float:
    """
    Interact with the environment for one episode using actions derived from the nnet.
    :param env: The environment.
    :param agent: The agent.
    :param training: If true the q_table will be updated using q-learning. The flag is also passed to the action selector.
    :return: The average cumulated reward.
    """
    done = False
    obs = env.reset()
    obs = concatenate((obs, [0,0])) #TODO: implement msg from sender
    cum_reward = 0.
    t = 0
    while not done:
        action = agent.act(obs, training)
        obs_prime, reward, done = env.step(action)
        obs_prime = concatenate((obs_prime, [0,0])) #TODO: implement msg from sender
        if training:
            agent.learn(obs, action, reward, done, obs_prime)
        obs = obs_prime
        if reward == 1: print("WIN")
        cum_reward += reward
        t += 1
        print(env.grid)

    print("DONE")

    return cum_reward/t

def train(env: GridWorld, num_episodes: int, gamma: float) -> Tuple[list, DeepQLearnerAgent]:
    """
    Training loop.
    :param env: The environment.
    :param num_episodes: The number of episodes.
    :return: ...
    """
    agent = DeepQLearnerAgent(env, 2, gamma)
    avg_rewards_list = [] 
    for i in tqdm(range(num_episodes)):
        if i % 100 == 0:
            print("Episode {} of {}".format(i + 1, num_episodes))
        avg_rewards_list.append(run_episode(env, agent, True))
        print("--------------")
        sleep(0.25)
    
    return avg_rewards_list, agent


if __name__ == '__main__':
    num_episodes = 15
    gamma = 0.7
    env = GridWorld(p_term=1-gamma)
    rewards, agent = train(env, num_episodes, gamma)
      
    plt.plot(arange(num_episodes), rewards)

    plt.title('Performance of DQLagent')
    plt.xlabel('episode')
    plt.ylabel('average reward per episode')
    plt.show()