
from .dqn import Agent as dqn
from .rainbow import Agent as rainbow

REGISTERED = {"dqn": dqn,
              "rainbow": rainbow,
              }
