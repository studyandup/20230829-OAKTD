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

from gym_puddle.envs import puddle_env
import numpy as np


class PuddleWorldEnv(puddle_env.PuddleEnv):

  def __init__(self, ):
    super(PuddleWorldEnv, self).__init__()
    # self.start = None

  def reset(self, observation=None):
    if observation is None:
      self.pos = np.array(super(PuddleWorldEnv, self).reset())
    else:
      self.pos = observation
    return self.pos

