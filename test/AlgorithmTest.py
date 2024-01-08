from pathlib import Path
import sys

base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

from environment import Env
import argparse
# import random as rd
from model import Agent
# import pandas as pd
import numpy as np
# from multiprocessing import Process
import os
# print (os.getcwd())




def algorithm_run(agent_name, train_env, test_env, params, runs,
                  episodes, train_steps, test_steps, path, starts=None):

    for i_run in range(runs):
        seed = np.random.randint(0, 1000)

        path_ = path + str(seed)
        if not os.path.exists(path_):
            os.makedirs(path_)

        algorithm = Agent[agent_name](**params)
        algorithm.run(train_env, test_env, train_steps, test_steps, episodes, path_, seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="transaction")
    # AcrobotEnv MountainCarEnv CartPoleEnv PuddleWorldEnv
    parser.add_argument("--environment", type=str, default="AcrobotEnv", help="Environment")
    # OAKTD OAKGDT  OAKTDC  OKTD OSKTD TileCoding
    parser.add_argument("--agent", type=str, default="OAKTD", help="Agent")
    # parser.add_argument("--agent", type=str, default="OAKGTD", help="Agent")
    parser.add_argument("--dictionary", type=str, default="MNC", help="Dictionary Construct")
    parser.add_argument("--gamma", type=float, default=0.999, help="Discount Rate. Default=0.99")
    parser.add_argument("--lamb", type=float, default=0, help="Lambda. Default=0.0")
    parser.add_argument("--alpha", type=float, default=0.05, help="Slow Learning Rate. Default=0.0001")
    parser.add_argument("--alphaDecay", type=float, default=0.99999, help="Slow Decay Rate. Default=0.995")
    parser.add_argument("--beta", type=float, default=0.01, help="Fast Learning Rate. Default=0.0001")
    parser.add_argument("--betaDecay", type=float, default=0.999999, help="Fast Decay Rate. Default=0.9995")
    parser.add_argument("--epsilonMax", type=float, default=1.0, help="Epsilon Greedy. Default=1.0")
    parser.add_argument("--epsilonDecay", type=float, default=0.9999, help="Epsilon Decay. Default=0.995")
    parser.add_argument("--mu", type=float, default=0.0001, help="ALD Threshold. Default=0.0001")
    parser.add_argument("--mu1", type=float, default=0.3, help="MNC Threshold. Default=0.1")
    parser.add_argument("--mu2", type=float, default=1.4, help="Selected Threshold. Default=0.8")
    parser.add_argument("--sigma1", type=float, default=1.0, help="Dictionary Kernel. Default=1.0")
    parser.add_argument("--sigma2", type=float, default=0.1, help="Feature Kernel. Default=1.0")

    parser.add_argument("--episodes", type=int, default=1, help="Test Episodes. Default=10")
    parser.add_argument("--trainSteps", type=int, default=100000, help="Train Steps. Default=1e6")
    parser.add_argument("--testSteps", type=int, default=1000, help="Test Every N Steps. Default=1e3")
    parser.add_argument("--runs", type=int, default=50, help="Algorithm Times. Default=1e1")

    opt = parser.parse_args()
    train_env = Env[opt.environment]()
    test_env = Env[opt.environment]()
    train_env = train_env.unwrapped
    test_env = test_env.unwrapped
    print("environment observation shape: ",  train_env.observation_space)
    print(train_env.observation_space.high)
    print(train_env.observation_space.low)
    print("environment action shape: ", train_env.action_space)
    # print(train_env.observation_space.low)
    agent = opt.agent
    runs = opt.runs
    train_steps = opt.trainSteps
    test_steps = opt.testSteps
    episodes = opt.episodes
    samples = None

    params = dict()
    params['environment'] = opt.environment
    params['dictionary'] = opt.dictionary
    params['gamma'] = opt.gamma
    params['lamb'] = opt.lamb
    params['alpha'] = opt.alpha
    params['alpha_decay'] = opt.alphaDecay
    params['beta'] = opt.beta
    params['beta_decay'] = opt.betaDecay
    params['epsilon_max'] = opt.epsilonMax
    params['epsilon_decay'] = opt.epsilonDecay
    params['states_dict'] = None
    params['mu'] = opt.mu                       # ALD
    params['mu1'] = opt.mu1                     # MNC   MountainCar:0.1, Acrobot:0.3
    params['mu2'] = opt.mu2
    params['sigma1'] = opt.sigma1               # sparse kernel
    params['sigma2'] = opt.sigma2               # feature kernel

    path = '../results/{}/{}/'.format(opt.environment, opt.agent)

    algorithm_run(agent, train_env, test_env, params, runs,
                  episodes, train_steps, test_steps, path, starts=None)

    # start_positions = np.arange(-1.2, 0.5, 0.034)
    # start_positions = np.around(start_positions, decimals=3)
    # start_positions = start_positions[np.newaxis,:].T
    # start_velocities = np.zeros((len(start_positions), 1))
    # samples = np.hstack((start_positions, start_velocities))
    #
    # start_positions = np.arange(-1.2, 0.5, 0.10) #np.ones(50)*-0.3#
    # start_positions = np.around(start_positions, decimals=2)
    # start_velocities = np.arange(-0.07, 0.07, 0.005) #np.ones(50)*-0.3#
    # start_velocities = np.around(start_velocities, decimals=3)
    # samples = np.transpose([np.repeat(start_positions, len(start_velocities)),
    #                         np.tile(start_velocities, len(start_positions))])
    
    


       


