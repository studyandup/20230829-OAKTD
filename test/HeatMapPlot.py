import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
from environment import Env

# import scipy.spatial.distance
# def gaussian_kernel(states, dictionary, sigma=1.0):
#     d= scipy.spatial.distance.cdist(states,dictionary)
#     return np.exp(-d**2/(2*sigma**2))
# samples


# AcrobotEnv CartPoleEnv MountainCarEnv PuddleWorldEnv
envname='CartPoleEnv'
env = Env[envname]()


# OAKTD TileCoding OKTD
modelname = "OAKTD"

# seed 573 61 17 22
seed = 17

# path
path = '../results/{}/{}/'.format(envname, modelname)

f1 = path + str(seed) + '/w.csv'
f2 = path + str(seed) + '/theta.csv'
f3 = path + str(seed) + '/dictionary.csv'
# f4 = path + str(seed) + '/rewards_all.csv'

w = pd.read_csv(f1).values.T
theta = pd.read_csv(f2).values.T
D = pd.read_csv(f3).values
#
# index = 0
# files = os.listdir(path)
# reward_all = 0
# for i in range(len(files)):
#     path_i = path + files[i] + "/rewards_all.csv"
#     # path_i = f4
#     if os.path.exists(path_i):
#         print(files[i])
#         if index == 0:
#             reward_all = pd.read_csv(path_i).values
#         else:
#             reward_all += pd.read_csv(path_i).values
#         index += 1
# reward_all /= index
D_numbers = []
files = os.listdir(path)
reward_all = 0
for i in range(len(files)):
    path_i = path + files[i] + '/dictionary.csv'
    # path_i = f4
    if os.path.exists(path_i):
        # print(files[i])
        if 0 < pd.read_csv(path_i).values.shape[0] < 100:
            print(files[i])
        D_numbers.append(pd.read_csv(path_i).values.shape[0])
print(D_numbers)
print("Mean={} | STD={}".format(np.mean(D_numbers), np.std(D_numbers)))

# reward_all = pd.read_csv(f4).values
position = np.arange(-1.2, 0.51, 0.01)
velocity = np.arange(-0.07, 0.071, 0.001)
X = np.around(position, decimals=3)
Y = np.around(velocity, decimals=3)[::-1]
# samples = np.transpose([np.repeat(position, len(velocity)), np.tile(velocity, len(position))])
# samples_ = samples[::-1]
# D = np.array(sorted(D, key=lambda x:(x[0],x[1])))

font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 20,
         }

low = env.observation_space.low
high = env.observation_space.high

# simple state in the whole space
# point = np.array([0, 0])
# point = (point-low)/(high-low)
# individual_attentions = np.zeros((len(velocity), len(position)))
# individual_selections = np.zeros((len(velocity), len(position)))
# for i in range(len(position)):
#     for j in range(len(velocity)):
#         state = np.array([position[i], velocity[j]])
#         state = (state-low)/(high-low)
#         individual_attentions[j][i] = np.exp(np.dot(w[0], np.abs(point-state)))
#         kernel = np.exp(-np.sum((point-state)**2)/(2.0*1**2))
#         individual_selections[j][i] = np.sqrt(2-2*kernel) < 0.2
# individual_attentions /= np.sum(individual_attentions)
# sns.set()
# f1 = plt.figure(figsize=(8, 6))
# # rainbow Greys_r
# ax1 = sns.heatmap(individual_attentions, xticklabels=20, yticklabels=20, cmap="rainbow")
# ax1.set_xlabel('Position', font2)
# ax1.set_ylabel('Velocity', font2)
# ax1.set_xticklabels(X[::20], fontsize=12, rotation=360, )
# ax1.set_yticklabels(Y[::20], fontsize=12, rotation=360, )
# plt.savefig('../results/{}/individual_attentions.pdf'.format(envname))
#


# simple states in the dictionary
# samples = np.array([[-1.0, -0.07],
#                     [0.0, 0.0],
#                     [0.5, 0.07]])
#
# samples = (samples-low)/(high-low)
# attentions = np.zeros((len(samples), D.shape[0]))
# selections = np.zeros((len(samples), D.shape[0]))
# X = [str(s*(high-low)+low) for s in D]
# Y = [str(s*(high-low)+low) for s in samples]
# print(X[::26])
# print(Y)
# X_choice = ['( -0.48,0)', '( -0.10,-0.01 )', '( -0.11,0.04)', '( -0.42,0.06 )']
# Y_choice = ["( -1,-0.07 )", "( 0,0 )", "( 0.5,0.07 )"]
# for i in range(len(samples)):
#     sample = samples[i]
#     for j in range(len(D)):
#         state = D[j]
#         attentions[i][j] = np.exp(np.dot(w[0], np.abs(sample-state)))
#         kernel = np.exp(-np.sum((sample-state)**2)/(2.0*1**2))
#         selections[i][j] = np.sqrt(2-2*kernel) < 0.2
# attentions /= np.sum(attentions)
# sns.set()
# f1 = plt.figure(figsize=(10, 6))
# ax1 = sns.heatmap(attentions,  xticklabels=26, cmap="rainbow")
# ax1.set_xlabel('Dictionary', font2)
# ax1.set_ylabel('Sample', font2)
# ax1.set_xticklabels(X_choice, fontsize=13, rotation=360)
# ax1.set_yticklabels(Y_choice, fontsize=13, rotation=60)
# plt.savefig('../results/{}/attentions.pdf'.format(envname))

#
# sns.set()
# f2 = plt.figure(figsize=(8, 6))
# ax2 = sns.heatmap(np.flip(reward_all, 0), xticklabels=20, vmax=-0, vmin=-200, yticklabels=20, cmap="rainbow_r")
# ax2.set_xlabel('Position', font2)
# ax2.set_ylabel('Velocity', font2)
# ax2.set_xticklabels(X[::20], fontsize=12, rotation=360, )
# ax2.set_yticklabels(Y[::20], fontsize=12, rotation=360, )
# if modelname == 'TileCoding':
#     plt.title('TD with Tile Coding', font2)
# else:
#     plt.title(modelname, font2)
# plt.savefig('../results/{}/{}_rewards_all.pdf'.format(envname, modelname))

plt.show()






















# norm_samples = (samples-bound0)/bound1
# #norm_D = (D-bound0)/bound1
# norm_samples_ = norm_samples[::-1]

#all states
# X = [str(np.around(sample_, decimals=3)) for sample_ in samples_]
# Y = [str(np.around(sample, decimals=3)) for sample in samples]
# #selections
# distances = graph.cal_kernel_matrix(norm_samples, norm_samples_)
# distances = np.sqrt(2-2*distances)
# selections = distances < 0.30 
# #attentions
# attentions = np.zeros((samples.shape[0],samples.shape[0]))
# for i in range(samples.shape[0]):
#     delta_d = (norm_samples[i]-norm_samples_)**2 * 2.0
#     d = np.matmul(w, delta_d.T)
#     e = np.exp(d)
#     attentions[i] = e/np.sum(e)
# # attentions = pd.DataFrame(attentions,index=Y,columns=X)
# f2 = plt.figure(2)
# ax2 = sns.heatmap(selections,xticklabels=240, yticklabels=240)
# ax2.set_xlabel('Dictionary')
# ax2.set_ylabel('State')
# ax2.set_xticklabels(X[::240], fontsize = 8, rotation = 0, horizontalalignment='center')
# ax2.set_yticklabels(Y[::240], fontsize = 5, rotation = 360)
# plt.savefig('../results/mountaincar/figures/selections,.pdf')
