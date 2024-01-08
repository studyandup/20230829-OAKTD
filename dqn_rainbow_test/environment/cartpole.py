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
from gym.envs.classic_control import cartpole
from gym import spaces
import numpy as np


class CartPoleEnv(cartpole.CartPoleEnv):

  def __init__(self, ):
    super(CartPoleEnv, self).__init__()
    high = np.array([
        self.x_threshold * 2,
        np.finfo(np.float32).max,
        self.theta_threshold_radians * 2,
        np.finfo(np.float32).max])

    self.observation_space = spaces.Box(-high, high)

  def reset(self, observation=None):
    if observation is None:
        self.state = np.array(super(CartPoleEnv, self).reset())
    else:
        self.state = observation
    self.steps_beyond_done = None
    return self.state


