from pathlib import Path
import sys

base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))
from utility import plot
# import numpy as np
from pathlib import Path
import sys

base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

if __name__ == '__main__':
    # MountainCarEnv AcrobotEnv CartPoleEnv PuddleWorldEnv
    env_name = 'AcrobotEnv'

    model_names = [
        # 'OKTD',
        # 'OSKTD',
        # 'TileCoding',
        'OAKTD',
        # 'dqn',
        # 'rainbowdqn',

    ]
    plot.plot_rewards(env_name, model_names)
    
    
    



