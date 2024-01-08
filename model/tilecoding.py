import numpy as np
import scipy.spatial.distance
from environment import Env
import pandas as pd
from utility.tiling import IHT, tiles


class TileCoding(object):
    def __init__(
            self,
            environment,
            dictionary,
            gamma,
            lamb,
            alpha,
            alpha_decay,
            beta,
            beta_decay,
            epsilon_max,
            epsilon_decay,
            states_dict,
            mu,
            mu1,
            mu2,
            sigma1,
            sigma2,
            
    ):
        self.envname = environment
        self.env = Env[environment]()
        self.n_features = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n
        self.gamma = gamma
        self.lamb = lamb
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.beta = beta
        self.beta_decay = beta_decay
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_max
        
        self.size = 8192
        self.tilings = 16
        self.eligibility = np.zeros(self.size)
        self.hashtable = IHT(self.size)
        self.numtilings = self.tilings
        self.theta = np.random.randn(self.size)*0.01
        self.bound = self.numtilings/(self.env.observation_space.high - self.env.observation_space.low)
        # self.positionscale = self.numtilings / (self.env.high[0] - self.env.low[0])
        # self.velocityscale = self.numtilings / (self.env.high[1] - self.env.low[1])

    def get_tiles(self, observation):
        # s = np.array([np.cos(observation[0]), np.sin(observation[0]),
        #               np.cos(observation[1]), np.sin(observation[1]),
        #               observation[2], observation[3]])
        return tiles(self.hashtable, self.numtilings, self.bound*observation)
        # np.array([self.positionscale, self.velocityscale])*observation)
    
    def cal_value(self, tiles):
        return np.sum(self.theta[tiles])

    def epsilon_greedy(self, observation):
        if np.random.uniform() > self.epsilon:
            q_value = np.zeros(self.n_actions)
            for i in range(self.n_actions):
                self.env.reset(observation)
                observation_, reward, done, _info = self.env.step(i)
                tc_ = self.get_tiles(observation_)
                q_value[i] = reward + self.gamma*self.cal_value(tc_)*np.abs(done-1)
            action = np.argmax(q_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action
    
    # def deterministic_policy(self, observation):
    #     q_value = np.zeros(self.n_actions)
    #     for i in range(self.n_actions):
    #         self.env.reset(observation)
    #         observation_, reward, done, _info = self.env.step(i)
    #         tc_ = self.get_tiles(observation_)
    #         q_value[i] = reward + self.gamma*self.cal_value(tc_)*np.abs(done-1)
    #     action = np.argmax(q_value)
    #     return action
    
    def learn(self, observation, action, reward, observation_, done):
        tc = self.get_tiles(observation)
        tc_ = self.get_tiles(observation_)
        if done:
            delta = reward + 0 - self.cal_value(tc)
        else:
            delta = reward + self.gamma*self.cal_value(tc_) - self.cal_value(tc)
        
        self.theta[tc] = self.theta[tc] + self.beta*delta*np.ones(self.tilings)
        self.beta *= self.beta_decay
        # self.beta *= self.beta_decay
        self.epsilon *= self.epsilon_decay
        return delta

    def run(self, train_env, test_env, train_steps, test_steps, episodes, path, seed):
        train_env.seed(seed)
        f1 = path + "/rewards.csv"
        f2 = path + "/rewards_all.csv"
        print("Data will be saved into " + f1)
        rewards_record = []
        observation = train_env.reset()
        for t in range(train_steps):
            # env.render()
            action = self.epsilon_greedy(observation)
            observation_, reward, done, _info = train_env.step(action)
            # update dictionary
            self.learn(observation, action, reward, observation_, done)
            if (t + 1) % test_steps == 0:
                test_scores = self.test(test_env, episodes)
                rewards_record.append(test_scores)

                print("Tile Coding: After Training {}k Steps "
                      "\tAverage Score: {:.2f}".format((t + 1) / 1000.0, np.mean(test_scores)))
            observation = observation_
            if done:
                observation = train_env.reset()
                # print(self.epsilon)
        data = pd.DataFrame(np.array(rewards_record))
        data.to_csv(f1, index=False)
        self.save_parameters(path)
        if self.envname == "MountainCarEnv":
            reward_all = self.rewards_all(test_env, 1)
            data = pd.DataFrame(reward_all)
            data.to_csv(f2, index=False)

    def save_parameters(self, path):
        # f2 = path + "/w.csv"
        f3 = path + "/theta.csv"
        # f4 = path + "/theta_.csv"
        # data = pd.DataFrame(np.array(self.w))
        # data.to_csv(f2, index=False)
        data = pd.DataFrame(np.array(self.theta))
        data.to_csv(f3, index=False)
        # data = pd.DataFrame(np.array(self.theta_))
        # data.to_csv(f4, index=False)

    def test(self, env, episodes, state=None):
        all_steps = np.zeros(episodes)
        scores = np.zeros(episodes)
        for episode in range(episodes):
            observation = env.reset(state)
            # print(observation)
            while True:
                # env.render()
                action = self.epsilon_greedy(observation)
                observation_, reward, done, _info = env.step(action)
                all_steps[episode] += 1.0
                scores[episode] += reward
                if done or all_steps[episode] >= 300:
                    break
                observation = observation_
        # print("OAKTD | Test start = {} | Steps = {} ".format(state, np.mean(all_steps)))
        return scores

    def rewards_all(self, env, episodes):
        positions = np.arange(-1.2, 0.51, 0.01)
        velocities = np.arange(-0.07, 0.071, 0.001)
        positions = np.around(positions, decimals=3)
        velocities = np.around(velocities, decimals=3)
        steps = np.zeros((len(velocities), len(positions)))
        for i in range(len(positions)):
            for j in range(len(velocities)):
                state = np.array([positions[i], velocities[j]])
                # kernels = self.kernels(state)
                # attentions,_ = self.attention(state)
                # states_values[j][i] = self.cal_value(kernels, attentions)
                steps[j][i] = np.mean(self.test(env, episodes, state))
            # print("Position = {} OAKTD test reward = {}".format(positions[i], steps[:, i]))
        return steps
    
    def policy(self):
        positions = np.arange(-1.2, 0.51, 0.01)
        velocities = np.arange(-0.07, 0.071, 0.001)
        positions = np.around(positions, decimals=3)
        velocities = np.around(velocities, decimals=3) 
        policy = np.zeros((len(velocities),len(positions)))
        for i in range(len(positions)):
            for j in range(len(velocities)):
                state = np.array([positions[i], velocities[j]])
#                 kernels = self.kernels(state)
#                 attentions,_ = self.attention(state)
#                 states_values[j][i] = self.cal_value(kernels, attentions)
                self.epsilon = 1.0
                policy[j][i] = self.epsilon_greedy(state)
        return policy
