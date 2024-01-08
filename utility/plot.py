import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from pathlib import Path
import sys

base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

colors = [
    'lightgreen',
    'grey',
    'blue',
    'red',
    'pink',
    'green',
    'chocolate',
    'black',
    'cornsilk',
    'gold',
    'violet',
    'orange',
    'yellow',
    'purple',
    'moccasin'
]

shadow_colors = [
    'lightcoral',
    'lightblue',
    'lightgreen',
    'violet',
    'chocolate',
    'lightyellow',
    'moccasin',
    'grey',
    'linghtpink',
    'cornsilk'
]

shadow = np.array([
    0.3,
    0.3,
    0.5,
    0.2,
    0.3,
    0.3,
])
point = [
    's',
    '*',
    'o',
    'D',
    'h',
    'H',
    'p',
    '8'
]

font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 20,
         }


def plot_rewards(env_name, model_names):
    _fig = plt.figure(figsize=(8, 6))
   
    path = '../results/{}/'.format(env_name)
    for i in range(len(model_names)):
        path_i = path + model_names[i] + "/"
        # read
        rewards = []
        files = os.listdir(path_i)
        for j in range(len(files)):

            path_i_j = path_i + files[j] + "/rewards.csv"
            run_reward = pd.read_csv(path_i_j).values
            # mean_reward = np.mean(run_reward, axis=1)
            print(model_names[i], files[j], run_reward.shape)
            if j == 0:
                rewards = run_reward.T
            else:
                rewards = np.vstack((rewards, run_reward.T))
        print(rewards.shape)
        mean_rewards = np.mean(np.array(rewards), axis=0)
        std_rewards = np.std(np.array(rewards), axis=0)
        id = np.arange(len(mean_rewards))

        # mean_rewards = mean_rewards[0:100:5]
        # std_rewards = std_rewards[0:100:5]
        # id = id[0:100:5]+1
        if model_names[i] == 'TileCoding':
            plt.plot(id, mean_rewards, label='TD with Tile Coding', color=colors[i], marker=point[i])
        else:
            plt.plot(id, mean_rewards, label=model_names[i], color=colors[i], marker=point[i])
        plt.fill_between(id, mean_rewards-std_rewards, mean_rewards+std_rewards, color=colors[i], alpha=shadow[-(i+1)])
    plt.xlim((0, 100))
    # plt.ylim((-325, -100))
    # plt.yticks(np.arange(-325, -100, 25))
    # plt.yticks(np.arange(0, 120, 20))
    # plt.xlim((0, 200))
    plt.ylabel('Cumulative reward', font2)
    plt.xlabel('Training steps (in thousands)', font2)
    plt.legend()
    if env_name == "PuddleWorldEnv":
        plt.title("Puddle World", font2)
    if env_name == "MountainCarEnv":
        plt.title("Mountain Car", font2)
    if env_name == "AcrobotEnv":
        plt.title("Acrobot", font2)
    if env_name == "CartPoleEnv":
        plt.title("CartPole", font2)
    save_file = '../results/{}/{}_rewards.pdf'.format(env_name, env_name)
    plt.savefig(save_file)
    plt.show()

# def plot_trainingsteps(envname, modelnames, sigma2s=None):
#     _ts = plt.figure()
#     for i in range(len(modelnames)):
#         if sigma2s is not None:
#             filename = '../results/{}/data/{}_trainingsteps_sigma2_{}.csv'.format(envname, modelnames[i],sigma2s[i])
#         else:
#             filename = '../results/{}/data/{}_trainingsteps.csv'.format(envname, modelnames[i])
#         run_steps = pd.read_csv(filename).values
#         if modelnames[i] == 'Rainbow':
#             run_steps = run_steps[0:100, :]
#         stepsMean = np.mean(run_steps, axis=0)
#         stepsStd = np.std(run_steps, axis=0)
#         stepsMean = stepsMean[0:201:10]
#         stepsStd = stepsStd[0:201:10]
#         X = np.arange(201)[0:201:10]
#         if modelnames[i] == 'TileCoding':
#             plt.plot(X, stepsMean, label='Tile Coding',color=colors[i],marker=point[i])
#         else:
#             plt.plot(X, stepsMean, label=modelnames[i],color=colors[i],marker=point[i])
#         plt.fill_between(X,stepsMean-stepsStd,stepsMean+stepsStd,color=colors[i],alpha=shadow[-(i+1)])
#     #plt.errorbar(X, stepsMean, yerr=stepsStd,label='learningsteps',color='b')
#     plt.ylim((50, 300))
#     plt.yticks(np.arange(50, 300, 25))
#     plt.ylabel('steps', font2)
#     plt.xlabel('episodes', font2)
# #     plt.title('alpha={}'.format(alphas[0]))
#     plt.legend()
#     if sigma2s is not None:
#         figname = '../results/{}/figures/trainingsteps_sigma2_{}.pdf'.format(envname, sigma2s[0])
#     else:
#         figname = '../results/{}/figures/trainingsteps.pdf'.format(envname)
#     plt.savefig(figname)
#     plt.show()

# def plot_MC_testingsteps(envname, modelnames, positions, sigma2s):
#     X = positions
#     _fig3 = plt.figure() 
#     for i in range(len(modelnames)):
#         filename = '../results/{}/data/{}_testingsteps_sigma2_{}.csv'.format(envname, modelnames[i],sigma2s[i])
#         run_steps = pd.read_csv(filename).values  
#         stepsMean = np.mean(run_steps, axis=0)
#         stepsStd = np.std(run_steps, axis=0)
#         plt.plot(X, stepsMean, label=modelnames[i],color=colors[i])
#         plt.fill_between(X,stepsMean-stepsStd,stepsMean+stepsStd,color=shadow_colors[i],alpha=0.2)
#         #plt.errorbar(X, stepsMean, yerr=stepsStd,label=modelnames[i],color=colors[i])
#     plt.ylim((0, 1000))
#     plt.xlim((-1.2, 0.5))
#     plt.yticks(np.arange(0, 1000, 100))
#     plt.ylabel('Steps')
#     plt.xlabel('Position(Velocity=0.0)')
#     plt.legend()
#     figname = '../results/{}/figures/testingsteps_sigma2_{}.pdf'.format(envname, sigma2s[0])
#     plt.savefig(figname)
#     plt.show()
    
#Acrobot
# def plot_AB_testingsteps(envname, modelnames, sigma2s=None):
#     steps = np.zeros((len(modelnames), 1))
#     for i in range(len(modelnames)):
#         if sigma2s is not None:
#             filename = '../results/{}/data/{}_testingsteps_sigma2_{}.csv'.format(envname, modelnames[i],sigma2s[i])
#         else:
#             filename = '../results/{}/data/{}_testingsteps.csv'.format(envname, modelnames[i])
#         model_steps = pd.read_csv(filename).values
#         if i == 0:
#             steps =  model_steps
#         else:
#             steps = np.hstack((steps, model_steps))
#     stepsMean = np.mean(steps, axis=0)
#     stepsStd = np.std(steps, axis=0)
#     print(stepsMean)
#     print(stepsStd)
#     _fig0 = plt.figure(0)
#     plt.errorbar(modelnames, stepsMean, yerr=stepsStd, ecolor='b',color='r', fmt='s', elinewidth=2,capsize=4)
#     plt.ylabel('Steps')
#     plt.xlabel('Algorithm')
#     plt.legend()
#     if sigma2s is not None:
#         figname = '../results/{}/figures/testingsteps_sigma2_{}.pdf'.format(envname, sigma2s[0])
#     else:
#         figname = '../results/{}/figures/testingsteps.pdf'.format(envname)
#     plt.savefig(figname)
#     plt.show()
    




