# import gym
import gym_puddle
#
# env = gym.make('PuddleWorld-v0')
# # env = gym.make('Catcher-v0')
#
# env.seed(17)
# # env = env.unwrapped
# for i_episode in range(500):
#     all_reward = 0
#     steps = 0
#     observation = env.reset()
#     # observation = np.array([0.2, 0])
#     # observation = env.reset(observation)
#     # print(env.observation_space.high)
#     # print(env.observation_space.low)
#     print(env.action_space)
#     # env.render()
#     while True:
#         env.render()
#         print(observation)
#         action = env.action_space.sample()
#         observation, reward, done, info = env.step(action)
#         # print(observation)
#         all_reward += reward
#         steps += 1
#         if done:
#             print("Episode finished after {} timesteps".format(steps))
#             print(all_reward)
#             break
#     # env.close()

import logging
import os, sys

import gym
from gym.wrappers import Monitor
from environment import Env

# import gym_ple

# The world's simplest agent!
class RandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()


if __name__ == '__main__':
    # You can optionally set up the logger. Also fine to set the level
    # to logging.DEBUG or logging.WARN if you want to change the
    # amount of output.
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # 如果在运行脚本时给定了命令行参数，那么将使用该参数作为游戏环境的名称；
    # 否则，将使用默认的'PuddleWorld-v0'作为游戏环境的名称
    # env = gym.make('PuddleWorld-v0' if len(sys.argv) < 2 else sys.argv[1])
    env = Env["MountainCarEnv"]()  # ok
    env = Env["CartPoleEnv"]()    # ok
    # env = Env["PuddleWorldEnv"]()  # 有问题？

    # env = Env["PuddleWorldEnv"]() # ？？？？？
    env = env.unwrapped

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    outdir = '/tmp/random-agent-results'
    env = Monitor(env, directory=outdir, force=True)

    # This declaration must go *after* the monitor call, since the
    # monitor's seeding creates a new action_space instance with the
    # appropriate pseudorandom number generator.
    env.seed(0)
    agent = RandomAgent(env.action_space)

    episode_count = 100
    reward = 0
    done = False

    for i in range(episode_count):
        ob = env.reset()

        while True:
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            if done:
                break
            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.

    # Dump result info to disk
    env.close()

    # Upload to the scoreboard. We could also do this from another
    # process if we wanted.
    logger.info(
        "Successfully ran RandomAgent. Now trying to upload results to the scoreboard. If it breaks, you can always just try re-uploading the same results.")
#    gym.upload(outdir)
# Syntax for uploading has changed
