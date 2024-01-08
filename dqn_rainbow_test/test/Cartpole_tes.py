from pathlib import Path
import sys

base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))
print(base_dir)
import os
import argparse
import gym
from dqn_rainbow_test.environment import Env
import pandas as pd
import torch
import numpy as np
import random as rd
from dqn_rainbow_test.agent import REGISTERED as Agent
import multiprocessing as mp

parser = argparse.ArgumentParser(description="transaction")
parser.add_argument("--checkpoint", type=bool, default=False, help="model loading checkpoint")
parser.add_argument("--alg", default="dqn", type=str,
                    help="algorithm of agent")  # default="dqn",default="double_dqn",default="mretrace"
parser.add_argument("--env", default="CartPoleEnv", type=str, help="env name")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning Rate. ")
parser.add_argument("--lrDecay", type=float, default=1, help="Learning Rate Decay. ")
parser.add_argument("--updateSize", type=int, default=500, help="soft update size")
parser.add_argument("--batchSize", type=int, default=32, help="batch size")
parser.add_argument("--bufferSize", type=int, default=int(100000), help="buffer size")
parser.add_argument("--gpu", default="0", type=str, help="gpu ids (default: 0)")
parser.add_argument("--gamma", type=float, default=0.99, help="discount")
parser.add_argument("--featureDim", type=int, default=16, help="Feature Dimension")
parser.add_argument("--trainSteps", type=int, default=int(200000), help="train steps")
parser.add_argument("--maxEps", type=float, default=1, help="max epsilon")
parser.add_argument("--minEps", type=float, default=0.01, help="min epsilon")
parser.add_argument("--epsDecay", type=float, default=0.99, help="epsilon decay")
parser.add_argument("--nEpochs", type=int, default=5, help="Number of epochs to train for")
parser.add_argument("--network", type=str, default="mlp", help="Representation layer")
parser.add_argument("--nEpisodes", type=int, default=1, help="testing episodes")
parser.add_argument("--testSteps", type=int, default=int(1000), help="testing each steps")

opt = parser.parse_args(args=[])

if torch.cuda.is_available():
    print("=> use gpu id: '{}'".format(opt.gpu))
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    if not torch.cuda.is_available():
        raise Exception("No GPU found or Wrong gpu id, please run without --cuda")


# def test(agent, env, n_episodes, eps):

#     scores = np.zeros(n_episodes)
#     for i in range(n_episodes):
#         obs = env.reset()
#         obs = pre_process(obs)
#         state = init_state(obs)
#         while True:
#             action = agent.act(state, eps)
#             # env.render()
#             next_state, reward, done, _ = env.step(action)
#             next_state = np.stack((state[1], state[2], state[3], pre_process(next_state)), axis=0)
#             scores[i] += reward
#             state = next_state
#             if done:
#                 break
#     return scores
def scoretest(agent, env, n_episodes, eps):
    scores = np.zeros(n_episodes)
    step = 0

    for i in range(n_episodes):
        state = env.reset()
        while True:
            step += 1
            action = agent.act(state, eps)
            # env.render()
            next_state, reward, done, _ = env.step(action)
            scores[i] += reward
            state = next_state
            if done or scores[i] >= 200:
                step = 0
                break
            # if done :
            #     break
    return scores


def train(agent, env, env1, n_episodes, test_steps, train_steps, eps_start, eps_end, eps_decay, path,
          checkpoint, path1, describe):
    scores = []
    scores_1 = []
    eps = eps_start
    rewards = 0
    t_reward = 0
    epo = 0
    path_ = path + '/save_model.pth.tar'
    # f1 = path1 + "/accumulated_rewards.csv"
    f2 = path1 + "/" + describe + ".csv"

    if checkpoint:
        agent.load_checkpoint(path_)
        eps = eps_end

    state = env.reset()
    action = agent.act(state, eps)
    T = 0

    for t in range(train_steps):
        T += 1
        next_state, reward, done, _ = env.step(action)
        next_action = agent.act(next_state, eps)
        agent.step(state, action, reward, next_state, done, t, eps, next_action)
        state = next_state
        action = next_action
        rewards += reward

        if T >= 300:
            done = True

        if (t + 1) % test_steps == 0:
            test_scores = scoretest(agent, env1, n_episodes, eps)
            scores.append(test_scores)
            # print(test_scores)
            data = pd.DataFrame(np.array(scores))
            data.to_csv(f2, index=False)
            print("After Training {} steps  \t Reward: {:.2f}".format(t, test_scores.mean()))

        eps = max(eps_end, eps_decay * eps)
        # decrease epsilon

        if done:
            T = 0
            epo += 1
            t_reward += rewards
            # scores_1.append(t_reward)
            # data = pd.DataFrame(np.array(scores_1))
            # if epo % 10 == 0:
            #     data.to_csv(f2, index=False)
            #     agent.save_model(path_)
            #     print("After Training {} Eposide/3000 \t Reward: {:.2f}".format(epo, t_reward))
            t_reward = 0
            state = env.reset()
            action = agent.act(state, eps)
            rewards = 0

        if epo >= 3001:
            break
    return scores


def run(Params: dict, alg, env, env1, path, n_epochs, n_episodes, test_steps, train_steps, eps_start, eps_end,
        eps_decay, checkpoint, path1, describe):
    if not os.path.exists(path1):
        os.makedirs(path1)
    scores = []
    for i in range(n_epochs):
        print("Start ", i, "epochs")
        seed = rd.randint(0, 1000)
        Params["seed"] = seed
        cur_path = path + str(seed)
        describe = describe + str(seed)
        print('start alg:', alg, ', save result in ', cur_path)

        if not os.path.exists(cur_path):
            os.makedirs(cur_path)
        env.seed(seed)
        agent = Agent[alg](**Params)

        scores.append(
            train(agent, env, env1, n_episodes, test_steps, train_steps, eps_start, eps_end, eps_decay,
                  cur_path, checkpoint, path1, describe))


def begin():
    alg = opt.alg
    n_epochs = opt.nEpochs
    env_name = opt.env
    n_episodes = opt.nEpisodes
    test_steps = opt.testSteps
    train_steps = opt.trainSteps
    eps_start = opt.maxEps
    eps_end = opt.minEps
    eps_decay = opt.epsDecay
    checkpoint = opt.checkpoint

    Params = dict()
    Params["network"] = opt.network
    Params["feature_dim"] = opt.featureDim
    Params["seed"] = 0
    Params["lr"] = opt.lr
    Params["lr_decay"] = opt.lrDecay
    Params["buffer_size"] = opt.bufferSize
    Params["gamma"] = opt.gamma
    Params["batch_size"] = opt.batchSize
    Params["update_size"] = opt.updateSize

    describe = alg + '-lr' + str(Params["lr"]) + '-buffer' + str(Params["buffer_size"]) + '-g' + str(
        Params["gamma"]) + '-batch_size' + str(Params["batch_size"]) + '-seed'
    path1 = str(base_dir) + '/rewards/' + env_name + '/' + alg + str(Params["lr"])

    env = Env["CartPoleEnv"]()
    env1 = Env["CartPoleEnv"]()
    # env1 = gym.make(env_name)

    Params["state_shape"] = env.observation_space.shape
    Params["action_size"] = env.action_space.n
    path = str(base_dir) + '/result/' + env_name + '/' + describe

    run(Params, alg, env, env1, path, n_epochs, n_episodes, test_steps,
        train_steps, eps_start, eps_end, eps_decay, checkpoint, path1, describe)


if __name__ == "__main__":
    # num_workers = 1

    # ctx = mp.get_context('spawn')
    # q = ctx.SimpleQueue()
    # processes = []
    # for i in range(num_workers):
    #     p = ctx.Process(
    #         target=begin,
    #         args=())
    #     p.start()
    #     processes.append(p)

    # for p in processes:
    #     p.join()
    begin()
