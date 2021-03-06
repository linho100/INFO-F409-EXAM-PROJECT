from typing import Dict, List
from math import floor, exp, log
from pandas import DataFrame, read_csv
from numpy import concatenate, array as narray, ndarray
import matplotlib.pyplot as plt
from os import mkdir

from gridWorld import GridWorld
from agents.sender_agent import SenderAgent, convert_message_to_binary
from agents.receiver_agent import DeepQLearnerAgent


def reward_step_plot(filepaths: List[str], labels: List[str], layout_name: str, plot_output_path: str):
    """
    Plots the average reward per training step.
    :param filepaths: List of filepaths to the results of experiments to plot.
    :param labels: Description of each configuration.
    :param layout_name: Title of the experiment.
    :param plot_output_path: Filepath to export the plot to.
    """
    try:
        dataframes = {labels[i]: read_csv(filepaths[i])
                      for i in range(len(filepaths))}
    except:
        print(f"One file in '{filepaths}' has not been found")
        return

    # Create plot by overlapping each df
    plt.figure()
    for label, df in dataframes.items():
        x = df["Step"]
        y = df["Average reward"].rolling(window=20).mean()
        plt.plot(x, y, label=label)

    plt.xlim([0, 6e5])
    plt.xlabel('Step')
    plt.ylabel('Average reward')
    plt.legend()
    plt.title(layout_name)
    plt.savefig(plot_output_path)


def reward_capacity_plot(filepaths: List[str], capacities: List[int], layout_name: str, plot_output_path: str):
    """
    Plots the evolution of the average reward reached at the end of training for various capacities values.
    :param filepaths: List of filepaths to the results of experiments to plot.
    :param capacities: Capacities used for each experiment.
    :param layout_name: Title of the experiment.
    :param plot_output_path: Filepath to export the plot to.
    """
    x, y = capacities, list()
    try:
        for f in filepaths:
            y.append(read_csv(f)["Average reward"][0])
    except:
        print(f"One file in '{filepaths}' has not been found")
        return

    # Create plot
    plt.figure()
    plt.plot(x, y, marker='o')
    plt.xlabel('Communication channel capacity (C)')
    plt.ylabel('Average reward')
    plt.title(layout_name)

    plt.savefig(plot_output_path)


def senders_predictions_to_csv(models_paths: Dict, layouts: List[int], capacities: List[int], output_filepath: str):
    """
    Loads the models, generate envs with the matching layout and get the prediction of the corresponding model for each possible goal location.
    The results are saved in a new CSV file to be imported into Excel in order to display them.
    :param models_paths: {keys=layout_name: values=List of paths to senders models}.
    :param layouts: List of layout_ids.
    :param capacities: List of capacities correponding to each model.
    :param output_filepath: Path to save the csv file to.
    """
    senders_nb = 1
    layouts_names = list(models_paths.keys())
    data = dict()

    # Part-1: Retrieve predictions from models
    for c in range(len(capacities)):
        data_c = dict()
        capacity = capacities[c]

        for l in range(len(layouts)):
            data_l = [[None for _ in range(5)] for _ in range(5)]
            # Setup env
            env = GridWorld(p_term=1e-10)
            layout_nb = layouts[l]
            env.reset(layout_nb)

            # Find walls, player and goals
            walls = env.layouts[layout_nb]["walls"]
            for w in walls:
                data_l[w[0]][w[1]] = -2
            player = (2, 2)
            data_l[player[0]][player[1]] = -1
            goals = list(set([(i, j) for i in range(5)
                         for j in range(5)]) - set(walls) - set([player]))

            # Setup agent
            messages_nb = floor(exp(log(capacity) / senders_nb))
            model_path = models_paths[layouts_names[l]][c]
            agent = SenderAgent(messages_nb=messages_nb, model_path=model_path)

            # For each position, retrieve message from agent
            for goal in goals:
                env.move_goal(goal)
                agent.goal_location = env.one_hot_enc_goal()
                message = convert_message_to_decimal(
                    agent.send_message(training=False))
                data_l[goal[0]][goal[1]] = message

            data_c[layouts_names[l]] = narray(data_l)
        data[capacity] = data_c

    # Part-2: Save data to csv file
    df = DataFrame.from_dict(data)
    df.to_csv(output_filepath)


def receiver_prediction_to_csv(model_path: str, messages: List[int], layout: int, output_filepath: str):
    """
    Loads the model, generates env with the matching layout and gets the prediction of the corresponding model for each possible message.
    The results are saved in a new CSV file to be imported into Excel in order to display them.
    :param model_path: Path to receiver model.
    :param messages: List of possible integer messages.
    :param layouts: layout_ids.
    :param capacities: List of capacities correponding to each model.
    :param output_filepath: Path to save the csv file to.
    """
    senders_nb = 1
    titles = [f"m = {m}" for m in messages]
    messages = [convert_message_to_binary(m) for m in messages]

    data = dict()
    # Setup env
    env = GridWorld(p_term=1e-10)
    env.reset(layout)

    # Setup agent
    agent = DeepQLearnerAgent(dim_msg=5, n_states=25,
                              n_senders=senders_nb, model_path=model_path)

    # Part-1: Retrieve predictions from models
    for m in range(len(messages)):
        message = messages[m]
        data_m = [[None for _ in range(5)] for _ in range(5)]

        # Find walls, player and goals
        walls = env.layouts[layout]["walls"]
        for w in walls:
            data_m[w[0]][w[1]] = -2
        positions = list(set([(i, j) for i in range(5)
                              for j in range(5)]) - set(walls))

        # For each position and the given message, retrieve action from agent
        for p in positions:
            env.update(p)
            obs = env.one_hot_enc_player()
            obs = concatenate((obs, message))
            action = agent.act(obs, training=False)
            data_m[p[0]][p[1]] = action

        data[titles[m]] = narray(data_m)

    # Part-2: Save data to csv file
    df = DataFrame.from_dict({4: data})
    df.to_csv(output_filepath)


def convert_message_to_decimal(message: ndarray) -> int:
    """ 
    Basic binary to decimal converted.
    :param message: One-hot-encoded Message written in an array.
    :return: The decimal integer.
    """
    decimal = 0
    for digit in message:
        decimal = decimal * 2 + int(digit)
    return decimal


if __name__ == '__main__':
    """
    Generates plots based on the results of the experiences realised with 'runner.py'.
    Makes some predictions with the trained models under given circumstances and export those to csv files.
    """

    # Make sure results directory exists
    try:
        mkdir(f"./results/")
    except FileExistsError:
        pass

    # 1
    reward_step_plot(filepaths=[
        "./experiments/experiment_1_layout_0/experiment_1_n_3.csv",
        "./experiments/experiment_1_layout_0/experiment_1_n_4.csv",
        "./experiments/experiment_1_layout_0/experiment_1_n_5.csv",
        "./experiments/experiment_0_layout_0/experiment_0_random.csv",
        "./experiments/experiment_0_layout_0/experiment_0_q_learner.csv"
    ],
        labels=["3 senders", "4 senders", "5 senders", "Random", "Q-learner"],
        layout_name="Empty room",
        plot_output_path="./results/plot_1_l_0.png")

    reward_step_plot(filepaths=[
        "./experiments/experiment_1_layout_2/experiment_1_n_1.csv",
        "./experiments/experiment_1_layout_2/experiment_1_n_3.csv",
        "./experiments/experiment_0_layout_2/experiment_0_random.csv",
        "./experiments/experiment_0_layout_2/experiment_0_q_learner.csv"
    ],
        labels=["1 sender", "3 senders", "Random", "Q-learner"],
        layout_name="Two room",
        plot_output_path="./results/plot_1_l_2.png")

    reward_step_plot(filepaths=[
        "./experiments/experiment_1_layout_4/experiment_1_n_1.csv",
        "./experiments/experiment_1_layout_4/experiment_1_n_3.csv",
        "./experiments/experiment_0_layout_4/experiment_0_random.csv",
        "./experiments/experiment_0_layout_4/experiment_0_q_learner.csv"
    ],
        labels=["1 sender", "3 senders", "Random", "Q-learner"],
        layout_name="Pong",
        plot_output_path="./results/plot_1_l_4.png")

    # 2
    reward_capacity_plot(filepaths=[
        "./experiments/experiment_2_layout_1/experiment_2_c_3.csv",
        "./experiments/experiment_2_layout_1/experiment_2_c_4.csv",
        "./experiments/experiment_2_layout_1/experiment_2_c_5.csv",
        "./experiments/experiment_2_layout_1/experiment_2_c_8.csv",
        "./experiments/experiment_2_layout_1/experiment_2_c_9.csv",
        "./experiments/experiment_2_layout_1/experiment_2_c_16.csv"
    ],
        capacities=[3, 4, 5, 8, 9, 16],
        layout_name="Flower",
        plot_output_path="./results/plot_2_l_1.png")

    reward_capacity_plot(filepaths=[
        "./experiments/experiment_2_layout_3/experiment_2_c_3.csv",
        "./experiments/experiment_2_layout_3/experiment_2_c_4.csv",
        "./experiments/experiment_2_layout_3/experiment_2_c_5.csv",
        "./experiments/experiment_2_layout_3/experiment_2_c_8.csv",
        "./experiments/experiment_2_layout_3/experiment_2_c_9.csv",
        "./experiments/experiment_2_layout_3/experiment_2_c_16.csv"
    ],
        capacities=[3, 4, 5, 8, 9, 16],
        layout_name="Four room",
        plot_output_path="./results/plot_2_l_3.png")

    # 3
    senders_predictions_to_csv(
        models_paths={
            "Flower": ["./experiments/experiment_2_layout_1/models/senders/s_c_3_0",
                       "./experiments/experiment_2_layout_1/models/senders/s_c_4_0",
                       "./experiments/experiment_2_layout_1/models/senders/s_c_9_0",
                       "./experiments/experiment_2_layout_1/models/senders/s_c_16_0"],
            "Four room": ["./experiments/experiment_2_layout_3/models/senders/s_c_3_0",
                          "./experiments/experiment_2_layout_3/models/senders/s_c_4_0",
                          "./experiments/experiment_2_layout_3/models/senders/s_c_9_0",
                          "./experiments/experiment_2_layout_3/models/senders/s_c_16_0"]
        },
        layouts=[1, 3],
        capacities=[3, 4, 9, 16],
        output_filepath="./results/sender_pred.csv")

    receiver_prediction_to_csv(model_path="./experiments/experiment_2_layout_1/models/receivers/r_c_4",
                               messages=[0, 1, 2, 3],
                               layout=1,
                               output_filepath="./results/receiver_pred.csv")
