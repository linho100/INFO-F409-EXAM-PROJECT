from typing import Tuple
from numpy import arange, concatenate
import matplotlib.pyplot as plt
from tqdm import tqdm

from receiver import DeepQLearnerAgent
from gridWorld import GridWorld


def run_episode(env: GridWorld, agent: DeepQLearnerAgent, training: bool, print_progress: bool) -> float:
    """
    Interact with the environment for one episode using actions derived from the nnet.
    :param env: The environment.
    :param agent: The agent.
    :param training: If true the q_table will be updated using q-learning. The flag is also passed to the action selector.
    :return: The average cumulated reward.
    """
    done = False
    obs = env.reset()
    # obs = concatenate((obs, [0,0])) #TODO: implement msg from sender
    cum_reward = 0.
    t = 0

    if print_progress is True: env.print_grid()

    while not done:
        action = agent.act(obs, training)
        obs_prime, reward, done = env.step(action)
        # obs_prime = concatenate((obs_prime, [0,0])) #TODO: implement msg from sender
        if training:
            history = agent.learn(obs, action, reward, done, obs_prime)
        obs = obs_prime
        
        if print_progress is True:
            if reward == 1:
                print("WIN")
                env.print_grid()

            if not done:
                env.print_grid()

        cum_reward += reward
        t += 1

    if print_progress is True: print("DONE")

    return cum_reward/t

def train(env: GridWorld, num_episodes: int, gamma: float) -> Tuple[list, DeepQLearnerAgent]:
    """
    Training loop.
    :param env: The environment.
    :param num_episodes: The number of episodes.
    :return: ...
    """
    agent = DeepQLearnerAgent()
    avg_rewards_list = []
    for i in tqdm(range(num_episodes)):
        if i % 50 == 0:
            print("Episode {} of {}".format(i + 1, num_episodes))
            for j in range(15):
                # evaluation
                run_episode(env, agent, False, True)
                i += j
        avg_rewards_list.append(run_episode(env, agent, True, False))

    return avg_rewards_list, agent


if __name__ == '__main__':
    num_episodes = 8
    gamma = 0.9
    env = GridWorld(p_term=1-gamma)
    rewards, agent = train(env, num_episodes, gamma)
      
    plt.plot(arange(1, len(rewards)+1), rewards)

    plt.title('Performance of DQLagent')
    plt.xlabel('episode')
    plt.ylabel('average reward per episode')
    plt.show()