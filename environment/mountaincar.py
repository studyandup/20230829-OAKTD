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
from gym.envs.classic_control import mountain_car
import numpy as np


class MountainCarEnv(mountain_car.MountainCarEnv):

  def __init__(self, ):
    super(MountainCarEnv, self).__init__()

  def reset(self, observation=None):
    if observation is None:
      self.state = np.array(super(MountainCarEnv, self).reset())
    else:
      self.state = observation
    return self.state

