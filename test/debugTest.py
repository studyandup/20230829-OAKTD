# import tensorflow as tf
# import numpy as np
# import pandas as pd
# model = "Rainbow"#TileCoding OAKTD Rainbow MountainCarEnv AcrobotEnv
# f1 = '../results/AcrobotEnv/{}_testingsteps.csv'.format(model)
# f2 = '../results/MountainCarEnv/data/{}_trainingsteps.csv'.format(model)
# f3 = '../results/new/{}_testingsteps_2.csv'.format(model)
# f4 = '../results/new/{}_trainingsteps_2.csv'.format(model)
# f5 = '../results/new/{}_testingsteps.csv'.format(model)
# f6 = '../results/new/{}_trainingsteps.csv'.format(model)
# testingsteps_1 = pd.read_csv(f1).values
# trainingsteps_1 = pd.read_csv(f2).values  #64 71
# testingsteps_2 = pd.read_csv(f3).values   
# trainingsteps_2 = pd.read_csv(f4).values  #54 60
# 
# testingsteps_3 = (testingsteps_1*71.0+testingsteps_2*60.0)/(71.0+60.0)
# trainingsteps_3 = np.vstack((trainingsteps_1[0:71,:], trainingsteps_2[0:60,:]))
# 
# data = pd.DataFrame(testingsteps_3) 
# data.to_csv(f5,index=False)
# data = pd.DataFrame(trainingsteps_3[0:100]) 
# data.to_csv(f6,index=False)
# testingsteps = pd.read_csv(f1).values
# 
# 
# trainingsteps = pd.read_csv(f2).values
# n,m = trainingsteps.shape
# for i in range(n):
#     for j in range(m):
#         if trainingsteps[i][j] >2000:
#             trainingsteps[i][j] = 2000.0
#     
# data = pd.DataFrame(trainingsteps) 
# data.to_csv(f2,index=False)
# print(trainingsteps[0:5,:])











# from environment.acrobot import AcrobotEnv
# from environment.cartpole import CartPoleEnv
# # env = AcrobotEnv()
# env = CartPoleEnv()
# observation = env.reset()
# print(env.observation_space)
# print(env.action_space)
# while True:
#     env.render()
#     action = np.random.randint(2)
#     observation_, reward, done, _ =  env.step(action)
#     print(observation)
#     print(reward)
#     if done:
#         env.close()
#         break
    
    
    
# import time
# a = tf.constant(np.arange(1, 36, dtype=np.float32), shape=[3, 2, 2, 3])
# b = tf.constant(np.arange(1, 36, dtype=np.float32), shape=[3, 2, 3, 2])
# c = tf.matmul(a, b)
# sess = tf.Session()
# print("a*b = ", sess.run(c))
# print(c.shape)
# c1 = tf.matmul(a[0, 0, :, :], b[0, 0, :, :])
# print("a[1]*b[1] = ", sess.run(c1))


# from environment.tetris import Tetris
# from utility.feature import TetrisDTFeature
# 
# T = Tetris()
# F = TetrisDTFeature()
# 
# board = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#                [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
#                [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
#                [0., 0., 0., 0., 1., 1., 0., 0., 0., 0.],
#                [0., 0., 0., 1., 1., 0., 0., 0., 0., 0.],
#                [0., 0., 0., 0., 1., 1., 1., 1., 1., 0.],
#                [0., 0., 0., 0., 1., 1., 0., 0., 1., 0.],
#                [0., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
#                [0., 1., 1., 1., 1., 1., 1., 1., 1., 1.]])
# 
# piece_id = 0 
# observation = T.reset(board, piece_id)
# action = [1,0]
# print(observation[0])
# observation_, reward, done, info = T.step(action)
# after_state = observation_[0][4:,:]
# print(reward)
# print(info)
# print(after_state)
# features = F.feature(after_state, info)
# print(features)
# print(observation_, reward, done, info)
# observation = T.reset()
# print(observation[0])
# print(T.piece)
# for i in range(1000):
#     valid_action = T.valid_action()
#     rotate = np.random.randint(0, len(valid_action))
#     tanslation = np.random.randint(0, valid_action[rotate]+1)
#     action = [rotate, tanslation]
#     print(action)
#     
#     time1 = time.clock()
#     feature = F.feature(observation, action)
#     time2 = time.clock()
#     print("feature time: ", (time2-time1)*1000)
#     print(feature)
#     
#     
#     observation_, reward, done, info = T.step(action)
#     print(observation_[0])
#     print(T.piece)
#     if done:
#         print(i+1)
#         break
#     observation = observation_


#adjacency_knn test
# from utility import graph
# A  =  np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#                 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#                 [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
#                 [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
#                 [0., 0., 0., 0., 1., 1., 0., 0., 0., 0.],
#                 [0., 0., 0., 1., 1., 0., 0., 0., 0., 0.],
#                 [0., 0., 0., 0., 1., 1., 1., 1., 1., 0.],
#                 [0., 0., 0., 0., 1., 1., 0., 0., 1., 0.],
#                 [0., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
#                 [0., 1., 1., 1., 1., 1., 1., 1., 1., 1.]])
# B = graph.adjacency_knn(A,4)
# print(B.todense())

# a = np.array([[1,2,3,4],
#               [5,6,7,8]])
# b = a[0,:].reshape((-1,2))
# c = a[1,:].reshape((-1,2))
# aid = np.array([[1,2]])
# bid = np.array([[1,2],
#                 [2,3],
#                 [1,1]])
# # d = np.expand_dims(d,1)
# # d = d.reshape((-1,2,2))
# print(aid-bid)
# print(b)
# print(c)
# print(a.reshape((1,-1)))

# positions = np.arange(-1.2, 0.51, 0.01)
# velocities = np.arange(-0.07, 0.071, 0.001)
# print(positions)

# a = np.array([[2,1],[1,2]])
# b = np.array([2,3])
# c = a+b[np.newaxis,:]#np.repeat(b[np.newaxis,:].T,2,axis=1)
# d = a+b
# print(a)
# print(b)
# print(c)
# print(d)
# print(b**a)
# print(np.prod(b**a,axis=1))

import numpy as np
from ple import PLE

# from ple.games.raycastmaze import RaycastMaze
# from ple.games.flappybird import FlappyBird
from ple.games.catcher import Catcher


class NaiveAgent():
    """
            This is our naive agent. It picks actions at random!
    """

    def __init__(self, actions):
        self.actions = actions

    def pickAction(self, reward, obs):
        return self.actions[np.random.randint(0, len(self.actions))]


###################################
game = Catcher(
    height=400,
    width=400
)  # create our game

fps = 30  # fps we want to run at
frame_skip = 2
num_steps = 1
force_fps = False  # slower speed
display_screen = True

reward = 0.0
max_noops = 20
nb_frames = 15000

# make a PLE instance.
p = PLE(game, fps=fps, frame_skip=frame_skip, num_steps=num_steps,
        force_fps=force_fps, display_screen=display_screen)

# our Naive agent!
agent = NaiveAgent(p.getActionSet())

# init agent and game.
p.init()

# lets do a random number of NOOP's
for i in range(np.random.randint(0, max_noops)):
    reward = p.act(p.NOOP)

# start our training loop
for f in range(nb_frames):
    # if the game is over
    if p.game_over():
        p.reset_game()

    # obs = p.getScreenRGB()
    obs = p.getGameState()
    print(obs)
    action = agent.pickAction(reward, obs)
    reward = p.act(action)

    if f % 50 == 0:
        p.saveScreen("screen_capture.png")
