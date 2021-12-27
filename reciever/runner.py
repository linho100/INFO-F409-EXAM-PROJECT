import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from reciever import DeepQLearnerAgent
from gridWorld import GridWorld


def run_episode(env: GridWorld, agents: DeepQLearnerAgent, training: bool) -> float:
    """
    Interact with the environment for one episode using actions derived from the q_table and the action_selector.
    :param env: The gym environment.
    :param agent: The agent.
    :param training: If true the q_table will be updated using q-learning. The flag is also passed to the action selector.
    :param gamma: The discount factor.
    :return: The cumulative discounted reward.
    """
    done = False
    obs = env.reset()
    cum_reward = 0.
    t = 0
    while not done:
        action = agent.act(obs, training)
        obs_prime, reward, done = env.step(actions)
        if training:
            agent.learn(obs, action, reward, done, obs_prime)
        obs = new_obs
        cum_reward += reward
        t += 1

    return cum_reward/t

def train(env: MatrixGame, num_episodes: int, evaluate_every: int, num_evaluation_episodes: int,
              epsilon_max: float = 1, epsilon_min: float = 0.01,
              epsilon_decay: float = 0.999) -> Tuple[List[IQLAgent], ndarray, ndarray, list]:
    """
    Training loop.
    :param env: The gym environment.
    :param t_max: The number of timesteps.
    :param evaluate_every: Evaluation frequency.
    :param num_evaluation_episodes: Number of episodes for evaluation.
    :param epsilon_max: The maximum epsilon of epsilon-greedy.
    :param epsilon_min: The minimum epsilon of epsilon-greedy.
    :param epsilon_decay: The decay factor of epsilon-greedy.
    :return: Tuple containing the list of agents, the returns of all training episodes, the averaged evaluation
    return of each evaluation, and the list of the greedy joint action of each evaluation.
    """
    agent = DeepQLearnerAgent(env, 2)
    evaluation_returns = np.zeros(t_max // evaluate_every)
    evaluation_joint_actions = np.zeros((t_max // evaluate_every, env.num_actions, env.num_actions))
    avg_rewards_list = [] 
    for i in range(num_episodes):
        if i % 100 == 0:
            print("Episode {} of {}".format(i + 1, num_episodes))
        avg_rewards_list.append(run_episode(env, agent, True))


if __name__ == '__main__':
    evaluate_every=5
    t_max = 1000
    env = MatrixGame()
    prob = np.zeros((t_max // evaluate_every, env.num_actions, env.num_actions))

    for _ in tqdm(range(100)):
      agents, returns, evaluation_returns, evaluation_joint_actions = train_iql(env, t_max, evaluate_every, 1)
      prob += evaluation_joint_actions
      joint_action = []
      for agent in agents:
        joint_action.append(agent.greedy_action())

    prob/=100

    row_agent_1_y = []
    row_agent_2_y = []
    row_agent_3_y = []
    
    col_agent_1_y = []
    col_agent_2_y = []
    col_agent_3_y = []

    print(prob[-1])
    print(np.transpose(prob[-1]))

    for e in prob:
      row_agent_1_y.append(sum(e[0]))
      row_agent_2_y.append(sum(e[1]))
      row_agent_3_y.append(sum(e[2]))

      col_agent_1_y.append(sum(np.transpose(e)[0]))
      col_agent_2_y.append(sum(np.transpose(e)[1]))
      col_agent_3_y.append(sum(np.transpose(e)[2]))

    plt.plot(np.arange(evaluate_every, t_max + evaluate_every, evaluate_every), row_agent_1_y, label="action 1")
    plt.plot(np.arange(evaluate_every, t_max + evaluate_every, evaluate_every), row_agent_2_y, label="action 2")
    plt.plot(np.arange(evaluate_every, t_max + evaluate_every, evaluate_every), row_agent_3_y, label="action 3")
    plt.title('Strategy of row agent')
    plt.xlabel('episode')
    plt.ylabel('probability of action')
    plt.legend(loc='best')
    plt.show()

    plt.plot(np.arange(evaluate_every, t_max + evaluate_every, evaluate_every), col_agent_1_y, label="action 1")
    plt.plot(np.arange(evaluate_every, t_max + evaluate_every, evaluate_every), col_agent_2_y, label="action 2")
    plt.plot(np.arange(evaluate_every, t_max + evaluate_every, evaluate_every), col_agent_3_y, label="action 3")
    plt.title('Strategy of column agent')
    plt.xlabel('episode')
    plt.ylabel('probability of action')
    plt.legend(loc='best')
    plt.show()