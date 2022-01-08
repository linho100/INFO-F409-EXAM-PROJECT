from typing import Iterable
from math import floor, exp, log
from numpy import concatenate, delete, array
from pandas import DataFrame
from os import mkdir, remove
from os.path import exists
from shutil import rmtree
from time import time

from gridWorld import GridWorld
from agents.sender_agent import SenderAgent
from agents.receiver_agent import DeepQLearnerAgent
from agents.random_agent import RandomAgent
from agents.q_learner_agent import QLearnerAgent


def training_episode(env: GridWorld, receiver: DeepQLearnerAgent, senders_list: Iterable[SenderAgent],
                     layout_type: int):
    """
    Interacts with the environment for one episode using actions provided by the models predictions in order to train said models.
    :param env: The environment.
    :param receiver: The receiver agent.
    :param senders_list: The list of senders agents.
    :param layout_type: Int id of the env's layout to be used.
    :return: The number of steps performed.
    """
    done = False
    obs = env.reset(layout_type)
    context_vector = env.one_hot_enc_goal()
    step = 0

    # For each sender, retrieve the message corresponding to the goal position
    # Then concatenate all the messages into a single one
    if senders_list:
        messages_encoded = [-1]
        senders_nb = len(senders_list)

        for i in range(senders_nb):
            senders_list[i].goal_location = context_vector
            message = senders_list[i].send_message()
            messages_encoded = concatenate((messages_encoded, message))
            obs = concatenate((obs, message))

        messages_encoded = delete(messages_encoded, 0)

    # Run the episode
    while not done:
        action = receiver.act(obs, True)
        obs_prime, reward, done = env.step(action)
        if senders_list:
            obs_prime = concatenate((obs_prime, messages_encoded))
        # Receiver earning
        receiver.learn(obs, action, reward, done, obs_prime)

        obs = obs_prime
        step += 1

    # Senders learning
    if senders_list:
        for i in range(senders_nb):
            senders_list[i].learn(reward)

    return step


def evaluation_episode(env: GridWorld, receiver: DeepQLearnerAgent, senders_list: Iterable[SenderAgent],
                       layout_type: int):
    """
    Interacts with the environment for one episode using actions provided by the models predictions in order to evaluate said models.
    :param env: The environment.
    :param receiver: The receiver agent.
    :param senders_list: The list of senders agents.
    :param layout_type: Int id of the env's layout to be used.
    :return: The reward received at the end of the episode divided by the number of steps performed.
    """
    done = False
    obs = env.reset(layout_type)
    context_vector = env.one_hot_enc_goal()
    step = 0

    if senders_list:
        messages_encoded = [-1]
        senders_nb = len(senders_list)
        for i in range(senders_nb):
            senders_list[i].goal_location = context_vector
            message = senders_list[i].send_message()
            messages_encoded = concatenate(
                (messages_encoded, message))  # Store the messages sent
            obs = concatenate((obs, message))
        messages_encoded = delete(messages_encoded, 0)

    while not done:
        action = receiver.act(obs, False)
        obs_prime, reward, done = env.step(action)
        if senders_list:
            obs_prime = concatenate((obs_prime, messages_encoded))
        obs = obs_prime
        step += 1

    return reward / step


def exp_0_random(layout=0):
    """
    Runs a random agent for the same number of steps as for the subsequent agents to be used as a reference point.
    :param layout: Int id of the env's layout to be used.
    """
    # Hyperparameters
    gamma = 0.9

    # Env and agent
    env = GridWorld(p_term=1 - gamma)
    agent = RandomAgent()

    # Run configuration
    steps_to_backup = 50000
    last_backup = 0
    max_step = 600000
    last_evaluation = 0
    steps_to_evaluation = 2000
    number_episode_per_evaluation = 20

    global_results = []

    step = 1
    start_t = time()
    while step < max_step:
        # No training episode required
        step += 1

        if step >= last_evaluation + steps_to_evaluation:
            # Evaluation episodes
            last_evaluation = step
            evaluation_reward = 0
            for _ in range(number_episode_per_evaluation):
                done = False
                env.reset(layout)
                ev_step = 0

                while not done:
                    _, reward, done = env.step(agent.act())
                    ev_step += 1

                evaluation_reward += reward / ev_step

            global_results.append(
                [step, evaluation_reward / number_episode_per_evaluation])

            # Feedback
            delta_t = time() - start_t
            remaining = convert_to_time(
                round(delta_t / step * (max_step - step), 0))
            print(
                f"Step: {step} ({round(step / max_step * 100, 2)}%) - Elapsed time: {convert_to_time(round(delta_t, 0))} - Estimated remaining time: {remaining}")

        # Save results
        if step >= last_backup + steps_to_backup:
            last_backup = step
            print(f"Starting backup now ... (Step: {step})")
            save_results(results=global_results, r_model=None, s_models=None,
                         experiment_number=0, layout=layout, subtitle="random")
            print("Backup is done!")


def exp_0_q_learner(layout=0):
    """
    Runs a basic q-learner agent for the same number of steps as for the subsequent agents to be used as a reference point.
    Evaluation is performed periodically during training to follow its progression.
    :param layout: Int id of the env's layout to be used.
    """

    # Hyperparameters
    gamma = 0.9
    n_states = 25

    # Env and agent
    env = GridWorld(p_term=1 - gamma)
    agent = QLearnerAgent(num_states=n_states, gamma=gamma)

    # Run configuration
    last_backup = 0
    steps_to_backup = 50000
    max_step = 600000
    last_evaluation = 0
    steps_to_evaluation = 2000
    number_episode_per_evaluation = 20

    global_results = []

    start_t = time()
    step = 1
    while step < max_step:
        # Training episode
        episode_step = training_episode(
            env, agent, senders_list=None, layout_type=layout)
        step += episode_step

        # Evaluate and feedback
        if step >= last_evaluation + steps_to_evaluation:
            last_evaluation = step
            evaluation_reward = 0
            for _ in range(number_episode_per_evaluation):
                evaluation_reward += evaluation_episode(
                    env, agent, senders_list=None, layout_type=layout)
            global_results.append(
                [step, evaluation_reward / number_episode_per_evaluation])

            # Feedback
            delta_t = time() - start_t
            remaining = convert_to_time(
                round(delta_t / step * (max_step - step), 0))
            print(
                f"Step: {step} ({round(step / max_step * 100, 2)}%) - Elapsed time: {convert_to_time(round(delta_t, 0))} - Estimated remaining time: {remaining}")

        # Save results
        if step >= last_backup + steps_to_backup:
            last_backup = step
            print(f"Starting backup now ... (Step: {step})")
            save_results(results=global_results, r_model=None, s_models=None,
                         experiment_number=0, layout=layout, subtitle="q_learner")
            print("Backup is done!")


def exp_1(layout=0, senders_nb=1):
    """
    Trains senders agents and one receiver DeepQL agent on a given grid layout.
    Evaluation is performed periodically during training to follow its progression.
    :param layout: Int id of the env's layout to be used.
    :param senders_nb: Number of senders agent to be trained.
    """
    # Hyperparameters
    gamma = 0.9
    n_states = 25
    dim_msg = 5
    channel_capacity = 16
    messages_nb = floor(exp(log(channel_capacity) / senders_nb))

    epsilon_r = 0.005
    epsilon_s = 0.005

    # Create the env and the models
    env = GridWorld(p_term=1 - gamma)
    senders = [SenderAgent(messages_nb=messages_nb, epsilon=epsilon_s)
               for _ in range(senders_nb)]
    receiver = DeepQLearnerAgent(dim_msg=dim_msg, n_states=n_states, n_senders=senders_nb, epsilon_decay=1 - epsilon_r,
                                 gamma=gamma)

    # Run configuration
    steps_to_backup = 50000
    last_backup = 0
    max_step = 600000
    last_evaluation = 0
    steps_to_evaluation = 2000
    number_episode_per_evaluation = 20

    global_results = []

    start_t = time()
    step = 1
    while step < max_step:
        # Train
        episode_step = training_episode(
            env, receiver, senders, layout_type=layout)
        step += episode_step

        # Evaluate and feedback
        if step >= last_evaluation + steps_to_evaluation:
            last_evaluation = step
            evaluation_reward = 0
            for _ in range(number_episode_per_evaluation):
                evaluation_reward += evaluation_episode(
                    env, receiver, senders, layout_type=layout)
            global_results.append(
                [step, evaluation_reward / number_episode_per_evaluation])

            # Feedback
            delta_t = time() - start_t
            remaining = convert_to_time(
                round(delta_t / step * (max_step - step), 0))
            print(
                f"Step: {step} ({round(step / max_step * 100, 2)}%) - Elapsed time: {convert_to_time(round(delta_t, 0))} - Estimated remaining time: {remaining}")

        # Save results
        if step >= last_backup + steps_to_backup:
            last_backup = step
            print(f"Starting backup now ... (Step: {step})")
            save_results(results=global_results, r_model=receiver.model, s_models=[s.model for s in senders],
                         experiment_number=1, layout=layout, subtitle=f"n_{senders_nb}")
            print("Backup is done!")


def exp_2(channel_capacity, layout):
    """
    Trains 1 sender agents and one receiver DeepQL agent on a given grid layout with a choosen communication channel capacity.
    Evaluation is performed only at the end of training
    :param layout: Int id of the env's layout to be used.
    :param channel_capacity: Capacity of the communication channel between both agents.
    """

    # Hyperparameters
    gamma = 0.9
    n_states = 25
    senders_nb = 1
    dim_msg = 5
    messages_nb = floor(exp(log(channel_capacity) / senders_nb))

    epsilon_r = 0.005
    epsilon_s = 0.005

    # Create the env and the models
    env = GridWorld(p_term=1 - gamma)
    senders = [SenderAgent(messages_nb=messages_nb, epsilon=epsilon_s)
               for _ in range(senders_nb)]
    receiver = DeepQLearnerAgent(dim_msg=dim_msg, n_states=n_states, n_senders=senders_nb, epsilon_decay=1 - epsilon_r,
                                 gamma=gamma)

    # Run configuration
    step = 1
    max_step = 250000
    start_t = time()
    steps_to_feedback = 2000
    last_feedback = 0
    while step < max_step:
        # Training episode
        episode_step = training_episode(
            env, receiver, senders, layout_type=layout)
        step += episode_step

        # Feedback
        if step >= last_feedback + steps_to_feedback:
            last_feedback = step
            # Feedback
            delta_t = time() - start_t
            remaining = convert_to_time(
                round(delta_t / step * (max_step - step), 0))
            print(
                f"Step: {step} ({round(step / max_step * 100, 2)}%) | Elapsed time: {convert_to_time(round(delta_t, 0))} - Left: {remaining}")

    evaluation_avg_reward = 0
    n_evaluation_episodes = 200
    for _ in range(n_evaluation_episodes):
        evaluation_avg_reward += evaluation_episode(
            env, receiver, senders, layout_type=layout)
    evaluation_avg_reward /= n_evaluation_episodes

    # Save results
    save_results(results=[[step, evaluation_avg_reward]],
                 r_model=receiver.model,
                 s_models=[s.model for s in senders],
                 experiment_number=2,
                 layout=layout,
                 subtitle=f"c_{channel_capacity}")


def convert_to_time(seconds):
    """
    Basic helper to convert seconds into human-readable format.
    :param seconds: Number of seconds to convert.
    :return: Time following the 'd day(s) - hh:mm:ss' format.
    """
    seconds = max(seconds, 0)
    days = seconds // (24 * 3600)
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60

    return "%d day(s) - %d:%02d:%02d" % (days, hour, minutes, seconds)


def save_results(results, r_model, s_models, experiment_number, layout, subtitle):
    """
    Saves the models and the training results in .csv files.
    :param results: Array containing the results of the evaluation of the models.
    :param r_model: Receiver model.
    :param s_models: Senders models.
    :param experiment_number: Int.
    :param layout: Int id of the env's layout used.
    :param subtitle: str.
    """
    # Create directories
    folder_name = f"experiment_{experiment_number}_layout_{layout}"
    create_directories(folder_name)

    # Create new CSV file
    csv_filename = f"./experiments/{folder_name}/experiment_{experiment_number}_{subtitle}.csv"
    if exists(csv_filename):
        remove(csv_filename)

    # Saving results to csv
    DataFrame(array(results), columns=[
              "Step", "Average reward"]).to_csv(csv_filename)

    # Delete previous models
    if s_models:
        s_filepaths = [
            f"./experiments/{folder_name}/models/senders/s_{subtitle}_{i}" for i in range(len(s_models))]
        for path in s_filepaths:
            if exists(path):
                rmtree(path, ignore_errors=True)
    r_filepath = f"./experiments/{folder_name}/models/receivers/r_{subtitle}"
    if exists(r_filepath):
        rmtree(r_filepath, ignore_errors=True)

    # Save models
    if r_model:
        r_model.save(r_filepath)

    if s_models:
        for i in range(len(s_models)):
            model = s_models[i]
            model.save(s_filepaths[i])


def create_directories(folder_name):
    """
    Helper to create all the necessary directories if they are missing
    :param folder_name: Experiment base folder
    """
    try:
        mkdir(f"./experiments/")
    except FileExistsError:
        pass

    try:
        mkdir(f"./experiments/{folder_name}/")
    except FileExistsError:
        pass

    try:
        mkdir(f"./experiments/{folder_name}/models/")
    except:
        pass

    try:
        mkdir(f"./experiments/{folder_name}/models/senders")
        mkdir(f"./experiments/{folder_name}/models/receivers")
    except FileExistsError:
        pass


if __name__ == '__main__':
    # Random and Q-learner agents "training"
    layouts = [0, 2, 4]
    for l in layouts:
        exp_0_random(l)
        exp_0_q_learner(l)

    # Extended training of various configurations of multi-senders agents with 1 DeepQL receiver agent
    senders_nb = [1, 3]
    for l in layouts:
        for n in senders_nb:
            exp_1(layout=l, senders_nb=n)

    # Training of single sender with single DeepQL receiver using different channel capacities
    channel_capacities = [3, 4, 5, 8, 9, 16]
    layouts = [1, 3]

    for l in layouts:
        for c in channel_capacities:
            exp_2(layout=l, channel_capacity=c)
