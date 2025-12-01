import matplotlib.pyplot as plt
import numpy as np
import os, glob, json
from utils.stats import compute_rewards

#we'll proceed first by graphing the reward for a simple experiment. We first grab some key info from the settings.json, such as names of the algorithms, labels for graphs, etc.
#then we load everything into numpy, averaging and computing std over the trials if applicable and plot it using matplotlib.

#TODO we will need some kind of plotting settings

#we parsed out all the trial data and averaged it correctly. now all that's left to do is actually graphing it...
def graph_rewards(results, keyword, title, buckets = True, base = False):
    plt.figure(figsize=(10,6))
    print("results keys", results.keys())
    if buckets:
        bucket_first = {}
        for algorithm in results[keyword]:
            print(algorithm)
            for bucket, values in results[keyword][algorithm].items():
                print(bucket)
                if bucket not in bucket_first:
                    bucket_first[bucket] = {}
                bucket_first[bucket][algorithm] = values
        print("REVERSE")
        for bucket in bucket_first:
            print(bucket)
            for algorithm, values in bucket_first[bucket].items():
                print(algorithm)
                # print("alg results keys", alg_results.keys())
                # print(values.keys())
                means = values["means"]
                stds = values["stds"]
                # if not base:
                #     means = np.maximum(-5,means)
                #     stds = np.minimum(10, stds)
                # else:
                #     means = np.maximum(-10000,means)
                #     stds = np.minimum(5000, stds)
                x = np.arange(len(means))
                plt.plot(x,means,label = algorithm+bucket)
                plt.fill_between(x, means-stds, means+stds, alpha=0.2)
            new_title = title+" "+bucket
            plt.title(new_title)
            plt.xlabel('episodes')
            plt.ylabel('total reward')
            plt.legend()
            save_graph(new_title)
            plt.show()
    else:
        for algorithm, values in results[keyword].items():
            print(algorithm)
            means = values["means"]
            stds = values["stds"]
            # means = np.maximum(-10000,means)
            # stds = np.minimum(5000, stds)
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

    # results = compute_rewards("PlatformTest_2025-01-11_20-05-32")
    # graph_rewards(results, "rewards", "Platform RandomPAMDP test, n = 3", buckets=False)

    # results = compute_rewards("MazeModes", rolling_window=25)
    # graph_rewards(results, "rewards", "Maze Modes with universal support, n = 5, t = 2e6", buckets=False)
    
    
    # results = compute_rewards("AntMazeDense700_2024-07-16_21-05-37")
    # graph_rewards(results, "long baselines AntMazeDense 10M timessteps, n = 5")

    # results = compute_rewards("PointMazeSparse_2024-07-14_13-54-38")
    # graph_rewards(results, "long baselines PointMazeSparse, n = 5")

    # results = compute_rewards("PointMazeDense_2024-07-12_17-59-20")
    # graph_rewards(results, "long baselines PointMazeDense 1M timesteps, n = 5")
    # results = compute_rewards("AntMazeSparse_2024-07-23_16-51-17")
    # graph_rewards(results, "long baselines AntMazeSparse 10M timesteps, n = 3")

    # results = compute_rewards("AntMove_2024-08-09_20-37-51")
    # graph_rewards(results, "AntMove 1M timesteps, n= 5")

    # results = compute_rewards("AntMove_2024-08-16_11-58-22")
    # graph_rewards(results, "AntMove 10M timesteps L1, n=3")

    # results = compute_rewards("AntRotate_2024-08-12_21-38-51")
    # graph_rewards(results, "AntRotate 1M timesteps, n= 5")

    # results = compute_rewards("Ant_2025-10-09_11-56-13", rolling_window=1)
    # graph_rewards(results, "rewards", "Ant (no maze) 1M timesteps, n=5", buckets = False)

    results = compute_rewards("MazeModesv2_2025-11-07_14-43-56", rolling_window=25)
    graph_rewards(results, "rewards", "Ant (no maze) 1M timesteps, n=5", buckets = False)

    # results = compute_rewards("AntPlaneMove_2024-09-10_12-12-12")
    # graph_rewards(results, "AntPlaneMove 1M (full vec vel reward, reduced action-cost) n=5")

    # results = compute_rewards("AntPlaneMove_2024-09-11_18-06-07")
    # graph_rewards(results, "AntPlaneMove 1M (full vec vel reward, reduced action-cost) n=5")
    # results = compute_rewards("AntPlaneMove_2024-09-23_15-55-54", base = True)
    # results = compute_rewards("AntPlaneMove_2024-09-23_14-47-18", base = True)
    # num_goal_buckets = 7
    # results = compute_rewards("AntPlaneMoveFinal_2024-11-11_21-03-38", base = True, num_goal_buckets = num_goal_buckets)
    # # print(results.keys())
    # buckets = True if num_goal_buckets else False
    # # print(results["base rewards"].keys())
    # graph_rewards(results, "base rewards", "AntPlaneMoveFinal T=3.5M v=(-6.0, 6.0) adapt_min = 0.01 L=L2 (base reward) n=5", base = True, buckets = buckets)
    # graph_rewards(results, "base average", "AntPlaneMoveFinal T=3.5M v=(-6.0, 6.0) adapt_min = 0.01 L=L2 (base average) n=5", base = True, buckets = buckets)
    # graph_rewards(results, "rewards", "AntPlaneMoveFinal T=3.5M v=(-6.0, 6.0) adapt_min = 0.01 L=L2 (total reward) n=5", buckets = buckets)