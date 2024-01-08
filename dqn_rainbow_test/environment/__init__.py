from .mountaincar import MountainCarEnv
from .acrobot import AcrobotEnv
from .cartpole import CartPoleEnv
from .puddleworld import PuddleWorldEnv
# from .catcher import CatcherEnv

Env = {"MountainCarEnv": MountainCarEnv,
       "AcrobotEnv": AcrobotEnv,
       "CartPoleEnv": CartPoleEnv,
       "PuddleWorldEnv": PuddleWorldEnv,
       # "CatcherEnv": CatcherEnv,
       }

