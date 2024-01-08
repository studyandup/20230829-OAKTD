import numpy as np
import scipy.spatial.distance
from environment import Env
import pandas as pd


class OSKTD(object):
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
        self.env = Env[environment]()
        self.n_features = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n

        self.online_construct = getattr(self, dictionary)
        self.gamma = gamma
        self.lamb = lamb
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.beta = beta
        self.beta_decay = beta_decay
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_max
        self.mu = mu
        self.mu1 = mu1
        self.mu2 = mu2
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.gamma = gamma

        self.states_dict = states_dict
        self.w = np.random.randn(self.n_features) * 0.01

        if self.states_dict is None:
            self.theta = None
        else:
            self.theta = np.random.randn(self.states_dict.shape[0])*0.01

    def ALD(self,observation):
        if self.states_dict is None:
            self.update_params(observation)
        else:
            kernel_states = self.gaussian_kernel(self.states_dict, self.states_dict)
            kernel_state = self.gaussian_kernel(observation[np.newaxis,:], self.states_dict)
            if np.linalg.det(kernel_states) != 0:
                C = np.dot(np.linalg.inv(kernel_states), kernel_state.T)
                delta = 1-np.dot(C,kernel_state)
                if delta[0][0] >= self.mu:
                    self.update_params(observation)

    def MNC(self,observation):
        norm_observation = self.normalization(observation)
        if self.states_dict is None:
            self.update_params(norm_observation)
        else:
            kernel_state = self.gaussian_kernel(norm_observation[np.newaxis,:], self.states_dict, self.sigma1)
            if np.sqrt(np.min(2-2*kernel_state)) >= self.mu1:
                self.update_params(norm_observation)
               
    def update_params(self, observation):
        if self.states_dict is None:
            self.states_dict = observation[np.newaxis,:]
            self.theta = np.random.randn(self.states_dict.shape[0])*0.01
        else:
            self.states_dict = np.vstack((self.states_dict, observation[np.newaxis,:]))
            self.theta = np.hstack((self.theta, np.random.randn()*0.01))
    
    def selection(self, observation):
        if self.states_dict is None:
            return 0
        else:
            # d = np.abs(observation-self.states_dict)
            # bound = np.array([1.7,0.14])
            # psi = d/bound
            # w = np.ones(self.n_features)
            # e = np.matmul(w[np.newaxis,:], psi.T)
            #
            # selections = e < self.mu2
            kernel_state = self.kernels(observation)
            # kernel_state = kernel_state[0]
            selections = np.sqrt(2-2*kernel_state) < self.mu2
            # print(selections)
            return selections
        
    def kernels(self, observation):
        norm_observation=self.normalization(observation)
        if self.states_dict is None:
            return 0
        else:
            kernel_state = self.gaussian_kernel(norm_observation[np.newaxis,:], self.states_dict, self.sigma2)
            return kernel_state[0]
        
    def gaussian_kernel(self, states, dictionary, sigma=1.0):
        d = scipy.spatial.distance.cdist(states,dictionary)
        return np.exp(-d**2/(2*sigma**2))

    def normalization(self, s):
        return (s - self.env.observation_space.low) / (self.env.observation_space.high - self.env.observation_space.low)

    def cal_value(self, kernels, selections):
        if self.states_dict is None:
            return 0
        else:
            return np.dot(self.theta, kernels*selections)

    def epsilon_greedy(self, observation):
        if np.random.uniform() > self.epsilon:
            q_value = np.zeros(self.n_actions)
            for i in range(self.n_actions):
                self.env.reset(observation)
                observation_, reward, done, _info = self.env.step(i)
                selections_ = self.selection(observation_)
                # print(np.sum(selections_))
                kernels_ = self.kernels(observation_)
                q_value[i] = reward + self.gamma*self.cal_value(kernels_, selections_)*np.abs(done-1)
            action = np.argmax(q_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action
    
    def deterministic_policy(self, observation):
        q_value = np.zeros(self.n_actions)
        for i in range(self.n_actions):
            self.env.reset(observation)
            observation_, reward, done, _info = self.env.step(i)
            selections_ = self.selection(observation_)
            kernels_ = self.kernels(observation_)
            q_value[i] = reward + self.gamma*self.cal_value(kernels_, selections_)*np.abs(done-1)
        action = np.argmax(q_value)
        return action
    
    def learn(self, observation, action, reward, observation_, done):
        selections = self.selection(observation)
        kernels = self.kernels(observation)
        selections_ = self.selection(observation_)
        kernels_ = self.kernels(observation_)
        if done:
            delta = reward - self.cal_value(kernels, selections)
        else:
            delta = reward + self.gamma*self.cal_value(kernels_, selections_) - self.cal_value(kernels, selections)
        nabla_theta = kernels*selections
        self.theta += self.beta * delta * nabla_theta
        self.beta *= self.beta_decay
        self.epsilon *= self.epsilon_decay

    def run(self, train_env, test_env, train_steps, test_steps, episodes, path, seed):
        train_env.seed(seed)
        f1 = path + "/rewards.csv"
        print("Data will be saved into " + f1)
        rewards_record = []
        observation = train_env.reset()
        self.online_construct(observation)
        for t in range(train_steps):
            # env.render()
            action = self.epsilon_greedy(observation)
            observation_, reward, done, _info = train_env.step(action)
            # update dictionary
            self.online_construct(observation_)
            self.learn(observation, action, reward, observation_, done)
            if (t + 1) % test_steps == 0:
                test_scores = self.test(test_env, episodes)
                rewards_record.append(test_scores)

                print("OSKTD: After Training {}k Steps "
                      "\tDictionary memory: {}"
                      "\tAverage Score: {:.2f}".format((t + 1) / 1000.0, self.states_dict.shape[0],
                                                       np.mean(test_scores)))
            observation = observation_
            if done:
                observation = train_env.reset()
                # print(self.epsilon)
        data = pd.DataFrame(np.array(rewards_record))
        data.to_csv(f1, index=False)
        self.save_parameters(path)

    def save_parameters(self, path):
        # f2 = path + "/w.csv"
        f3 = path + "/theta.csv"
        # f4 = path + "/theta_.csv"
        f5 = path + "/dictionary.csv"
        # data = pd.DataFrame(np.array(self.w))
        # data.to_csv(f2, index=False)
        data = pd.DataFrame(np.array(self.theta))
        data.to_csv(f3, index=False)
        # data = pd.DataFrame(np.array(self.theta_))
        # data.to_csv(f4, index=False)
        data = pd.DataFrame(np.array(self.states_dict))
        data.to_csv(f5, index=False)

    def test(self, env, episodes, state=None):
        all_steps = np.zeros(episodes)
        scores = np.zeros(episodes)
        for episode in range(episodes):
            observation = env.reset(state)
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
    

