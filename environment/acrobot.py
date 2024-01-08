# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# import math
# from gym import spaces
from gym.envs.classic_control import acrobot
from gym import spaces
from gym.envs.classic_control.acrobot import rk4, wrap, bound
import numpy as np


class AcrobotEnv(acrobot.AcrobotEnv):

  def __init__(self, ):
    super(AcrobotEnv, self).__init__()
    high = np.array([2*np.pi, 2*np.pi, self.MAX_VEL_1, self.MAX_VEL_2]) # , dtype=np.float32
    low = -high
    self.observation_space = spaces.Box(low=low, high=high) #, dtype=np.float32

  def reset(self, observation=None):
    '''
    self.state = self.np_random.uniform(low=-0.1, high=0.1, size=(4,))
    return self._get_ob()
    '''
    if observation is None:
      self.state = self.np_random.uniform(low=-0.0000, high=0.0000, size=(4,))
    else:
      self.state = observation
    return self.state

  def step(self, a):
    s = self.state
    torque = self.AVAIL_TORQUE[a]

    # Add noise to the force action
    if self.torque_noise_max > 0:
      torque += self.np_random.uniform(-self.torque_noise_max, self.torque_noise_max)

    # Now, augment the state with our force action so it can be passed to
    # _dsdt
    s_augmented = np.append(s, torque)

    ns = rk4(self._dsdt, s_augmented, [0, self.dt])
    # only care about final timestep of integration returned by integrator
    ns = ns[-1]
    ns = ns[:4]  # omit action
    # ODEINT IS TOO SLOW!
    # ns_continuous = integrate.odeint(self._dsdt, self.s_continuous, [0, self.dt])
    # self.s_continuous = ns_continuous[-1] # We only care about the state
    # at the ''final timestep'', self.dt

    ns[0] = wrap(ns[0], -np.pi, np.pi)
    ns[1] = wrap(ns[1], -np.pi, np.pi)
    ns[2] = bound(ns[2], -self.MAX_VEL_1, self.MAX_VEL_1)
    ns[3] = bound(ns[3], -self.MAX_VEL_2, self.MAX_VEL_2)
    self.state = ns
    terminal = self._terminal()
    reward = -1. if not terminal else 0.
    # return (self._get_ob(), reward, terminal, {}) # 原版
    # print("self._get_ob(): ", self._get_ob())  # np.array([cos(s[0]), sin(s[0]), cos(s[1]), sin(s[1]), s[2], s[3]])
    # print("self.state: ",self.state)
    return (self.state, reward, terminal, {}) # (self._get_ob(), reward, terminal, {})

