import numpy as np
import scipy.spatial.distance
from environment import Env
import pandas as pd


class OAKGTD(object):
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

        self.online_construct = getattr(self, dictionary)
        self.gamma = gamma
        self.lamb = lamb
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.beta = beta
        self.beta_decay = beta_decay
        self.beta_gtd = 0.01
        self.beta_decay_gtd = 0.999999
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_max
        self.mu = mu
        self.mu1 = mu1
        self.mu2 = mu2
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.states_dict = states_dict
        self.w = np.random.randn(self.n_features) * 0.01
        self.w_gtd = np.random.randn(self.n_features) * 0
        # np.array([-3.4897304009482237,-3.254886410409489])
        # np.random.randn(self.n_features)*0.01
        # -np.ones(self.n_features)
        # np.zeros(self.n_features)
        if self.states_dict is None:
            self.theta = None
            self.theta_ = None
            # self.eligibility = None
        else:
            self.theta = np.random.randn(self.states_dict.shape[0]) * 0.000001
            self.theta_ = self.theta
            # self.eligibility = np.random.randn(self.states_dict.shape[0])

    def gaussian_kernel(self, states, dictionary, sigma=1.0):
        d = scipy.spatial.distance.cdist(states, dictionary)
        return np.exp(-d ** 2 / (2 * sigma ** 2))

    def ALD(self, observation):
        norm_observation = self.normalization(observation)
        if self.states_dict is None:
            self.update_params(norm_observation)
        else:
            kernel_states = self.gaussian_kernel(self.states_dict, self.states_dict)
            kernel_state = self.gaussian_kernel(norm_observation[np.newaxis, :], self.states_dict)
            if np.linalg.det(kernel_states) != 0:
                C = np.matmul(np.linalg.inv(kernel_states), kernel_state.T)
                delta = 1 - np.matmul(kernel_state, C)
                if np.sqrt(np.abs(delta[0][0])) >= self.mu:
                    self.update_params(norm_observation)

    # dictionary
    def MNC(self, observation):
        norm_observation = self.normalization(observation)  # 归一化
        if self.states_dict is None:
            self.update_params(norm_observation)
        else:
            kernel_state = self.gaussian_kernel(norm_observation[np.newaxis, :], self.states_dict, self.sigma1)
            if np.sqrt(np.min(2 - 2 * kernel_state)) >= self.mu1:  # self.mu1 = 0.3
                self.update_params(norm_observation)

    def normalization(self, s):
        '''
        非"CartPoleEnv"环境中，将状态值映射到0到1之间的范围。 归一化
        '''
        # observation = np.array([np.cos(s[0]), np.sin(s[0]), np.cos(s[1]), np.sin(s[1]), s[2], s[3]])
        if self.envname == "CartPoleEnv":
            return s
        return (s - self.env.observation_space.low) / (self.env.observation_space.high - self.env.observation_space.low)

    def update_params(self, observation):
        if self.states_dict is None:
            self.states_dict = observation[np.newaxis, :]
            self.theta = np.random.randn(self.states_dict.shape[0]) * 0.000001
            self.theta_ = self.theta
            # self.eligibility = np.zeros(self.states_dict.shape[0])
        else:
            self.states_dict = np.vstack((self.states_dict, observation[np.newaxis, :]))
            new_value = 0.0
            # np.random.randn()*0.01
            self.theta = np.hstack((self.theta, new_value))
            # (np.random.uniform()-0.5)*0.1
            self.theta_ = np.hstack((self.theta_, new_value))
            # self.eligibility = np.hstack((self.eligibility, 0))

    def kernels(self, observation):
        norm_observation = self.normalization(observation)
        if self.states_dict is None:
            return 0
        else:
            kernel_state = self.gaussian_kernel(norm_observation[np.newaxis, :], self.states_dict, self.sigma2)
            return kernel_state[0]

    def attention(self, observation):
        norm_observation = self.normalization(observation)  # 归一化
        if self.states_dict is None:
            return 0
        else:
            # Cosine Similarity
            # if np.sqrt(np.sum(observation**2)) !=0:
            #     norm_observation = observation/np.sqrt(np.sum(observation**2))
            # square_sum = np.sqrt(np.sum(self.states_dict**2, axis=1))
            # index = np.where(square_sum==0.0)
            # square_sum[index] = 1.0
            # norm_states_dict = self.states_dict/square_sum[:,np.newaxis]
            # psi = norm_observation * norm_states_dict

            # Euclidean Distance L1
            # 计算norm_observation和self.states_dict之间的绝对差值，将结果存储在d中。
            # d是一个数组，其中每个元素表示对应样本之间的欧氏距离。
            d = np.abs(norm_observation - self.states_dict)
            # norm
            # square_sum = np.sqrt(np.sum(d**2, axis=1))
            # index = np.where(square_sum==0.0)
            # square_sum[index] = 1.0
            # psi = d/square_sum[:,np.newaxis]
            # bound = np.array([1.7,0.14])
            # bound = np.array([2*np.pi, 2*np.pi, 8*np.pi, 18*np.pi])
            psi = d
            # /bound
            # 矩阵乘法运算，将self.w与psi的转置相乘，结果存储在e中。
            # self.w是一个权重向量，psi是一个矩阵。矩阵乘法的结果是一个具有相应维度的数组。
            e = np.matmul(self.w[np.newaxis, :], psi.T)
            # 计算e的指数函数，并将结果除以指数函数的总和，得到注意力权重。
            # 指数函数的计算将增强大的值，并抑制小的值，从而得到归一化的注意力权重。
            a = np.exp(e) / np.sum(np.exp(e))
            # 返回注意力权重数组和psi，a[0]表示注意力权重，psi表示样本之间的关系或相似度。
            return a[0], psi

    def cal_value(self, kernels, attention, theta):
        if self.states_dict is None:
            return 0
        else:
            assert np.sum(attention) > 0.99999
            return np.dot(theta, kernels * attention)

    def epsilon_greedy(self, observation):
        if np.random.uniform() > self.epsilon:
            q_value = np.zeros(self.n_actions)
            for i in range(self.n_actions):
                self.env.reset(observation)
                observation_, reward, done, _info = self.env.step(i)
                kernels_ = self.kernels(observation_)
                a_, _ = self.attention(observation_)
                q_value[i] = (reward + self.gamma * self.cal_value(kernels_, a_, self.theta)) * np.abs(done - 1)
            action = np.argmax(q_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn_TD(self, observation, action, reward, observation_, done):
        kernels = self.kernels(observation)
        kernels_ = self.kernels(observation_)
        a, psi = self.attention(observation)  # 计算观测observation与状态字典之间的注意力权重a和样本之间的关系psi
        a_, psi_ = self.attention(observation_)

        if done:
            delta_ = 0 - self.cal_value(kernels, a, self.theta_)
            delta = 0 - self.cal_value(kernels, a, self.theta)
            # print(a)
        else:
            delta_ = reward + self.gamma * self.cal_value(kernels_, a_, self.theta_) - self.cal_value(kernels, a,
                                                                                                      self.theta_)
            delta = reward + self.gamma * self.cal_value(kernels_, a_, self.theta) - self.cal_value(kernels, a,
                                                                                                    self.theta)

        n = self.states_dict.shape[0]  # 字典中状态的个数
        # slow s
        A = np.matmul(a[np.newaxis, :], psi)
        A = psi - A
        A = A.reshape((n, -1))
        weight = kernels * a * self.theta_
        nabla_w = np.matmul(weight[np.newaxis, :], A).reshape(self.w.shape)
        nabla_theta = kernels * a  # 更新 theta 和 theta_

        # slow s'
        B = np.matmul(a_[np.newaxis, :], psi_)
        B = psi_ - B
        B = B.reshape((n, -1))
        weight_ = kernels_ * a_ * self.theta_
        nabla_w_ = np.matmul(weight_[np.newaxis, :], B).reshape(self.w.shape)
        nabla_theta_ = kernels_ * a_

        self.w += self.alpha * delta_ * (nabla_w - self.gamma * nabla_w_)

        self.theta_ += self.alpha * delta_ * (nabla_theta - self.gamma * nabla_theta_)
        # self.w += self.alpha * delta_ * nabla_w
        # self.theta_ += self.alpha * delta_ * nabla_theta

        # fast    beta : Fast Learning Rate
        # delta :  TD error   reward + self.gamma * self.cal_value(kernels_, a_, self.theta) - self.cal_value(kernels, a,
        #                                                                                                     self.theta)
        # nabla_theta
        self.theta += self.beta * delta * nabla_theta

        self.alpha *= self.alpha_decay
        self.beta *= self.beta_decay
        self.epsilon *= self.epsilon_decay
        return delta, delta_

        # 入口

    def learn_GTD2(self, observation, action, reward, observation_, done):
        kernels = self.kernels(observation)
        kernels_ = self.kernels(observation_)
        a, psi = self.attention(observation)  # 计算观测observation与状态字典之间的注意力权重a和样本之间的关系psi
        a_, psi_ = self.attention(observation_)

        if done:
            delta_ = 0 - self.cal_value(kernels, a, self.theta_)
            delta = 0 - self.cal_value(kernels, a, self.theta)
            # print(a)
        else:
            delta_ = reward + self.gamma * self.cal_value(kernels_, a_, self.theta_) - self.cal_value(kernels, a,
                                                                                                      self.theta_)
            delta = reward + self.gamma * self.cal_value(kernels_, a_, self.theta) - self.cal_value(kernels, a,
                                                                                                    self.theta)

        n = self.states_dict.shape[0]  # 字典中状态的个数
        # slow s
        A = np.matmul(a[np.newaxis, :], psi)
        A = psi - A
        A = A.reshape((n, -1))
        weight = kernels * a * self.theta_
        nabla_w = np.matmul(weight[np.newaxis, :], A).reshape(self.w.shape)

        nabla_theta = kernels * a  # 更新 theta 和 theta_


        # slow s'
        B = np.matmul(a_[np.newaxis, :], psi_)
        B = psi_ - B
        B = B.reshape((n, -1))
        weight_ = kernels_ * a_ * self.theta_
        nabla_w_ = np.matmul(weight_[np.newaxis, :], B).reshape(self.w.shape)
        nabla_theta_ = kernels_ * a_

        self.w += self.alpha * delta_ * (nabla_w - self.gamma * nabla_w_)

        self.theta_ += self.alpha * delta_ * (nabla_theta - self.gamma * nabla_theta_)
        # self.w += self.alpha * delta_ * nabla_w
        # self.theta_ += self.alpha * delta_ * nabla_theta

        # fast
        phi_feature = nabla_theta
        phi_feature_ = nabla_theta_

        self.w_gtd += self.beta_gtd * (delta - np.dot(phi_feature[np.newaxis, :], self.w_gtd)) * phi_feature

        self.theta += self.beta * (phi_feature-self.gamma*phi_feature_)*np.dot(phi_feature[np.newaxis, :],self.w_gtd)
        # self.theta +=

        self.alpha *= self.alpha_decay
        self.beta *= self.beta_decay
        self.beta_gtd *= self.beta_decay_gtd
        self.epsilon *= self.epsilon_decay
        return delta, delta_

    def learn_TDC(self, observation, action, reward, observation_, done):
        kernels = self.kernels(observation)
        kernels_ = self.kernels(observation_)
        a, psi = self.attention(observation)  # 计算观测observation与状态字典之间的注意力权重a和样本之间的关系psi
        a_, psi_ = self.attention(observation_)

        if done:
            delta_ = 0 - self.cal_value(kernels, a, self.theta_)
            delta = 0 - self.cal_value(kernels, a, self.theta)
            # print(a)
        else:
            delta_ = reward + self.gamma * self.cal_value(kernels_, a_, self.theta_) - self.cal_value(kernels, a,
                                                                                                      self.theta_)
            delta = reward + self.gamma * self.cal_value(kernels_, a_, self.theta) - self.cal_value(kernels, a,
                                                                                                    self.theta)

        n = self.states_dict.shape[0]  # 字典中状态的个数
        # slow s
        A = np.matmul(a[np.newaxis, :], psi)
        A = psi - A
        A = A.reshape((n, -1))
        weight = kernels * a * self.theta_
        nabla_w = np.matmul(weight[np.newaxis, :], A).reshape(self.w.shape)
        nabla_theta = kernels * a  # 更新 theta 和 theta_

        # slow s'
        B = np.matmul(a_[np.newaxis, :], psi_)
        B = psi_ - B
        B = B.reshape((n, -1))
        weight_ = kernels_ * a_ * self.theta_
        nabla_w_ = np.matmul(weight_[np.newaxis, :], B).reshape(self.w.shape)
        nabla_theta_ = kernels_ * a_

        self.w += self.alpha * delta_ * (nabla_w - self.gamma * nabla_w_)

        self.theta_ += self.alpha * delta_ * (nabla_theta - self.gamma * nabla_theta_)
        # self.w += self.alpha * delta_ * nabla_w
        # self.theta_ += self.alpha * delta_ * nabla_theta

        # fast    beta : Fast Learning Rate
        # delta :  TD error   reward + self.gamma * self.cal_value(kernels_, a_, self.theta) - self.cal_value(kernels, a,
        #                                                                                                     self.theta)
        # nabla_theta
        self.theta += self.beta * delta * nabla_theta

        self.alpha *= self.alpha_decay
        self.beta *= self.beta_decay
        self.epsilon *= self.epsilon_decay
        return delta, delta_

    def run(self, train_env, test_env, train_steps, test_steps, episodes, path, seed):
        train_env.seed(seed)
        # test_env.seed(seed)
        f1 = path + "/rewards.csv"
        f2 = path + "/rewards_all.csv"
        print("Data will be saved into " + f1)
        rewards_record = []
        observation = train_env.reset()
        print(observation)
        # 在线构造字典
        self.online_construct(observation)
        for t in range(train_steps):
            # train_env.render()
            action = self.epsilon_greedy(observation)
            observation_, reward, done, _info = train_env.step(action)
            # print(observation_)
            # update dictionary
            self.online_construct(observation_)
            # learn
            # self.learn_TD(observation, action, reward, observation_, done)
            self.learn_GTD2(observation, action, reward, observation_, done)
            # self.learn_TDC(observation, action, reward, observation_, done)

            if (t + 1) % test_steps == 0:
                # print(self.theta)
                test_scores = self.test(test_env, episodes)
                rewards_record.append(test_scores)

                print("OAKTD: After Training {}k Steps "
                      "\tDictionary memory: {}"
                      "\tAverage Score: {:.2f}".format((t + 1) / 1000.0, self.states_dict.shape[0],
                                                       np.mean(test_scores)))
            observation = observation_
            if done:
                observation = train_env.reset()
                # self.env.reset()
                # self.env.reset()
                # print(observation)
                # print(self.epsilon)
        data = pd.DataFrame(np.array(rewards_record))
        data.to_csv(f1, index=False)
        self.save_parameters(path)
        if self.envname == "MountainCarEnv":
            reward_all = self.rewards_all(test_env, 1)
            data = pd.DataFrame(reward_all)
            data.to_csv(f2, index=False)

    def save_parameters(self, path):
        f2 = path + "/w.csv"
        f3 = path + "/theta.csv"
        f4 = path + "/theta_.csv"
        f5 = path + "/dictionary.csv"
        data = pd.DataFrame(np.array(self.w))
        data.to_csv(f2, index=False)
        data = pd.DataFrame(np.array(self.theta))
        data.to_csv(f3, index=False)
        data = pd.DataFrame(np.array(self.theta_))
        data.to_csv(f4, index=False)
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

#     def policy(self):
#         positions = np.arange(-1.2, 0.51, 0.01)
#         velocities = np.arange(-0.07, 0.071, 0.001)
#         positions = np.around(positions, decimals=3)
#         velocities = np.around(velocities, decimals=3)
#         policy = np.zeros((len(velocities),len(positions)))
#         for i in range(len(positions)):
#             for j in range(len(velocities)):
#                 state = np.array([positions[i],velocities[j]])
# #                 kernels = self.kernels(state)
# #                 attentions,_ = self.attention(state)
# #                 states_values[j][i] = self.cal_value(kernels, attentions)
#                 self.epsilon = 1.0
#                 policy[j][i] = self.epsilon_greedy(state)
#         return policy
