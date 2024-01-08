import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import copy


class PuddleEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, start=[0.2, 0.4], goal=[1.0, 1.0], goal_threshold=0.1,
            noise=0.01, thrust=0.05, puddle_center=[[.1, .75], [.45, .75], [.45, .4], [.45, .8]],
            puddle_radius = 0.1):
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.goal_threshold = goal_threshold
        self.noise = noise
        self.thrust = thrust
        self.puddle_center = [np.array(center) for center in puddle_center]
        self.puddle_radius = puddle_radius

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(2,))

        self.actions = [np.zeros(2) for i in range(5)]
        for i in range(4):
            self.actions[i][i//2] = thrust * (i%2 * 2 - 1)
        # print(self.actions)

        self.seed()
        self.viewer = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

        self.pos += self.actions[action] + self.np_random.randn(2)*self.noise
        # print(self.np_random.randn(2)*self.noise)
        # self.np_random.uniform(low=-self.noise, high=self.noise, size=(2,))
        self.pos = np.clip(self.pos, 0.0, 1.0)

        reward = self._get_reward(self.pos)
        # print(reward)

        done = np.linalg.norm((self.pos - self.goal), ord=1) < self.goal_threshold

        return self.pos, reward, done, {}

    def _get_reward(self, pos):
        reward = -1.
        distance_x = 0
        if 0<self.pos[0]<0.1 and 0.65<self.pos[1]<0.85:
            distance = np.linalg.norm((self.pos - self.puddle_center[0]), ord=2)
            # print()
            if distance<self.puddle_radius:
                distance_x = self.puddle_radius-distance
        if 0.45<self.pos[0]<0.55 and 0.65<self.pos[1]<0.85:
            distance = np.linalg.norm((self.pos - self.puddle_center[1]), ord=2)
            if distance < self.puddle_radius:
                distance_x = self.puddle_radius - distance
        if .1<=self.pos[0]<=0.45 and 0.65<self.pos[1]<0.85:
            distance = np.abs(0.75-self.pos[0])
            if distance < self.puddle_radius:
                distance_x = self.puddle_radius-distance

        distance_y = 0
        if 0.3 < self.pos[1] < 0.4 and 0.35 < self.pos[0] < 0.55:
            distance = np.linalg.norm((self.pos - self.puddle_center[2]), ord=2)
            if distance < self.puddle_radius:
                distance_y = self.puddle_radius - distance
        if 0.8 < self.pos[1] < 0.9 and 0.35 < self.pos[0] < 0.55:
            distance = np.linalg.norm((self.pos - self.puddle_center[3]), ord=2)
            if distance < self.puddle_radius:
                distance_y = self.puddle_radius - distance
        if 0.4 <= self.pos[1] <= 0.8 and 0.35 < self.pos[0] < 0.55:
            distance = np.abs(0.45 - self.pos[1])
            if distance < self.puddle_radius:
                distance_y = self.puddle_radius - distance

        distance_min = min(distance_x, distance_y)
        # elif self.pos[0]<.45:
        #     min_x = np.abs(self.pos[0]-0.75) - self.puddle_radius
        # for cen, wid in zip(self.puddle_center, self.puddle_width):
        #     reward -= 2. * self._gaussian1d(pos[0], cen[0], wid[0]) * \
        #         self._gaussian1d(pos[1], cen[1], wid[1])

        return -400*distance_min + reward

    def _gaussian1d(self, p, mu, sig):
        return np.exp(-((p - mu)**2)/(2.*sig**2)) / (sig*np.sqrt(2.*np.pi))

    def reset(self):
        if self.start is None:
            self.pos = self.observation_space.sample()
        else:
            self.pos = copy.copy(self.start)
        return self.pos

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 600
        screen_height = 400

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            from gym_puddle.shapes.image import Image
            self.viewer = rendering.Viewer(screen_width, screen_height)

            import pyglet
            img_width = 100
            img_height = 100
            fformat = 'RGB'
            pixels = np.zeros((img_width, img_height, len(fformat)))
            for i in range(img_width):
                for j in range(img_height):
                    x = float(i)/img_width
                    y = float(j)/img_height
                    pixels[j,i,:] = self._get_reward(np.array([x,y]))

            pixels -= pixels.min()
            pixels *= 255./pixels.max()
            pixels = np.floor(pixels)

            img = pyglet.image.create(img_width, img_height)
            img.format = fformat
            data=[chr(int(pixel)) for pixel in pixels.flatten()]

            img.set_data(fformat, img_width * len(fformat), ''.join(data))
            bg_image = Image(img, screen_width, screen_height)
            bg_image.set_color(1.0,1.0,1.0)

            self.viewer.add_geom(bg_image)

            thickness = 5
            agent_polygon = rendering.FilledPolygon([(-thickness,-thickness),
             (-thickness,thickness), (thickness,thickness), (thickness,-thickness)])
            agent_polygon.set_color(0.0,1.0,0.0)
            self.agenttrans = rendering.Transform()
            agent_polygon.add_attr(self.agenttrans)
            self.viewer.add_geom(agent_polygon)

        self.agenttrans.set_translation(self.pos[0]*screen_width, self.pos[1]*screen_height)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')