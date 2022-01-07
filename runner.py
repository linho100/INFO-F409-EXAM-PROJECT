from typing import Iterable
from math import floor, exp, log
from numpy import concatenate, delete, array
from os import mkdir, remove
import shutil
from os.path import exists
from pandas import DataFrame
from time import time
import multiprocessing

from gridWorld import GridWorld
from sender import SenderAgent
from receiver import DeepQLearnerAgent

def training_episode(env: GridWorld, receiver: DeepQLearnerAgent, senders_list: Iterable[SenderAgent], layout_type: int):
    """
    Interact with the environment for one episode using actions derived from the nnet.
    :param env: The environment.
    :param receiver: The receiver agent.
    :param senders_list:
    :param training: If true the q_table will be updated using q-learning. The flag is also passed to the action selector.
    :param layout_type:
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
        messages_encoded = concatenate((messages_encoded, message)) # Store the messages sent
        obs = concatenate((obs, message)) # TODO: implement msg from the i-th sender

    messages_encoded = delete(messages_encoded, 0)

    while not done:
        action = receiver.act(obs, True)
        obs_prime, reward, done = env.step(action)
        obs_prime = concatenate((obs_prime, messages_encoded)) # TODO: implement msg from senders
        receiver.learn(obs, action, reward, done, obs_prime)

        obs = obs_prime
        step += 1

    # Senders learning
    for i in range(senders_nb):
        senders_list[i].learn(reward)

    return step

def evaluation_episode(env: GridWorld, receiver: DeepQLearnerAgent, senders_list: Iterable[SenderAgent], layout_type: int):
    """
    Interact with the environment for one episode using actions derived from the nnet.
    :param env: The environment.
    :param receiver: The receiver agent.
    :param senders_list:
    :param training: If true the q_table will be updated using q-learning. The flag is also passed to the action selector.
    :param layout_type:
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
        messages_encoded = concatenate((messages_encoded, message)) # Store the messages sent
        obs = concatenate((obs, message)) # TODO: implement msg from the i-th sender
    messages_encoded = delete(messages_encoded, 0)

    while not done:
        action = receiver.act(obs, False)
        obs_prime, reward, done = env.step(action)
        obs_prime = concatenate((obs_prime, messages_encoded)) # TODO: implement msg from senders
        obs = obs_prime
        step += 1

    return reward / step

def exp_1(layout = 0, senders_nb = 1):
    # Hyperparameters
    gamma = 0.9
    n_states = 25
    dim_msg = 5
    channel_capacity = 16
    messages_nb = floor(exp(log(channel_capacity) / senders_nb))

    epsilon_r = 0.005
    epsilon_s = 0.005

    # Create the model
    env = GridWorld(p_term=1-gamma)
    senders = [SenderAgent(messages_nb = messages_nb, epsilon = epsilon_s) for _ in range(senders_nb)]
    receiver = DeepQLearnerAgent(dim_msg = dim_msg, n_states = n_states, n_senders = senders_nb, epsilon_decay = 1 - epsilon_r, gamma=gamma)
    
    # Training
    steps_to_backup = 50000
    last_backup = 0
    
    # Testing episodes : 20 episodes every 2000 steps
    step = 1
    max_step = 5000000
    last_evaluation = 0
    steps_to_evaluation = 2000
    number_episode_per_evaluation = 20
    global_results = []
    start_t = time()
    while step < max_step:
        # Train
        episode_step = training_episode(env, receiver, senders, layout_type=layout)
        step += episode_step

        # Evaluate and feedback
        if step >= last_evaluation + steps_to_evaluation:
            last_evaluation = step
            evaluation_reward = 0
            for _ in range(number_episode_per_evaluation):
                evaluation_reward += evaluation_episode(env, receiver, senders, layout_type=layout)
            global_results.append([step, evaluation_reward/number_episode_per_evaluation])

            # Feedback
            delta_t = time() - start_t
            remaining = convert_to_time(round(delta_t/step*(max_step-step),0))
            print(f"Step: {step} ({round(step/max_step*100,2)}%) - Elapsed time: {convert_to_time(round(delta_t, 0))} - Estimated remaining time: {remaining}")
            
        # Save results
        if step >= last_backup + steps_to_backup:
            last_backup = step
            print(f"Starting backup now ... (Step: {step})")
            save_results(results=global_results,r_model=receiver.model,s_models=[s.model for s in senders],experiment_number=1,layout=layout,subtitle=f"n_{senders_nb}")
            print("Backup is done!")
    
def exp_2(channel_capacity, layout): # Channel capacity
    # Hyperparameters
    gamma = 0.9
    n_states = 25
    senders_nb = 1
    dim_msg = 5
    messages_nb = floor(exp(log(channel_capacity) / senders_nb))

    epsilon_r = 0.005
    epsilon_s = 0.005

    # Create the model
    env = GridWorld(p_term=1-gamma)
    senders = [SenderAgent(messages_nb = messages_nb, epsilon = epsilon_s) for _ in range(senders_nb)]
    receiver = DeepQLearnerAgent(dim_msg = dim_msg, n_states = n_states, n_senders = senders_nb, epsilon_decay = 1 - epsilon_r, gamma=gamma)
    
    # Train
    step = 1
    max_step = 250000
    start_t = time()
    steps_to_feedback = 2000
    last_feedback = 0
    while step < max_step:
        episode_step = training_episode(env, receiver, senders, layout_type=layout)
        step += episode_step
        
        # Feedback
        if step >= last_feedback + steps_to_feedback:
            last_feedback = step
            # Feedback
            delta_t = time() - start_t
            remaining = convert_to_time(round(delta_t/step*(max_step-step),0))
            print(f"Step: {step} ({round(step/max_step*100,2)}%) | Elapsed time: {convert_to_time(round(delta_t, 0))} - Left: {remaining}")
            
    evaluation_avg_reward = 0
    n_evaluation_episodes = 200
    for _ in range(n_evaluation_episodes):
        evaluation_avg_reward += evaluation_episode(env, receiver, senders, layout_type=layout)
    evaluation_avg_reward /= n_evaluation_episodes
    
    # Save results
    save_results(results=[[step, evaluation_avg_reward]],
                 r_model = receiver.model,
                 s_models = [s.model for s in senders],
                 experiment_number=2,
                 layout=layout,
                 subtitle=f"c_{channel_capacity}")

def convert_to_time(seconds):
    seconds = max(seconds, 0)
    days = seconds // (24%3600)
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
      
    return "%d day(s) - %d:%02d:%02d" % (days,hour, minutes, seconds)

def save_results(results,r_model,s_models,experiment_number,layout,subtitle):
    # Create directories
    folder_name = f"experiment_{experiment_number}_layout_{layout}"
    create_directories(folder_name)
    
    # Create new CSV file
    csv_filename = f"./experiments/{folder_name}/experiment_{experiment_number}_{subtitle}.csv"
    if exists(csv_filename):
      remove(csv_filename)

    # Saving results to csv
    DataFrame(array(results), columns=["Step","Average reward"]).to_csv(csv_filename)
    
    # Delete previous models
    s_filepaths = [f"./experiments/{folder_name}/models/senders/s_{subtitle}_{i}" for i in range(len(s_models))]
    for path in s_filepaths:
        if exists(path):
            shutil.rmtree(path, ignore_errors=True)
    r_filepath = f"./experiments/{folder_name}/models/receivers/r_{subtitle}"
    if exists(r_filepath):
        shutil.rmtree(r_filepath, ignore_errors=True)

    # Save models
    r_model.save(r_filepath)
    
    for i in range(len(s_models)):
        model = s_models[i]
        model.save(s_filepaths[i])
    
def create_directories(folder_name):
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
    # In main!!
    # Type 1 experiments
    # Sara: l=0,s=[1,3]
    # Ilyes: l=2,s=[1,3]
    # Linh: l=4,s=[1,3]
    layouts = [0,2,4]
    senders_nb = [1,3]
    layout = 2
    sender_nb = 1
    sender_nb_2 = 3
    # exp_1(layout = layout, senders_nb = sender_nb)
    pool = multiprocessing.Pool(processes = 8)
    pool.starmap(exp_1, [(layout, sender_nb), (layout, sender_nb_2)])
    pool.close()
    