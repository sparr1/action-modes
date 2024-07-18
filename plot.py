import matplotlib.pyplot as plt
import numpy as np
import os, glob, json
from utils import compute_rewards

#we'll proceed first by graphing the reward for a simple experiment. We first grab some key info from the settings.json, such as names of the algorithms, labels for graphs, etc.
#then we load everything into numpy, averaging and computing std over the trials if applicable and plot it using matplotlib.

#TODO we will need some kind of plotting settings

#we parsed out all the trial data and averaged it correctly. now all that's left to do is actually graphing it...
def graph_rewards(results, title):
    plt.figure(figsize=(10,6))
    for algorithm, values in results.items():
        means = values["means"]
        stds = values["stds"]
        x = np.arange(len(means))
        plt.plot(x,means,label = algorithm)
        plt.fill_between(x, means-stds, means+stds, alpha=0.2)
    plt.title(title)
    plt.xlabel('episodes')
    plt.ylabel('total reward')
    plt.legend()
    save_graph(title)
    plt.show()

def save_graph(name):
    plt.savefig('./graphs/'+name+'.png')
if __name__ == "__main__":

    # results = compute_rewards("DQNShort_old")
    # graph_rewards(results, "old DQN config override test, n = 15")

    results = compute_rewards("AntMazeDense700_2024-07-16_21-05-37")
    graph_rewards(results, "long baselines AntMazeDense 10M timesteps, n = 5")

    # results = compute_rewards("PointMazeSparse_2024-07-14_13-54-38")
    # graph_rewards(results, "long baselines PointMazeSparse, n = 5")