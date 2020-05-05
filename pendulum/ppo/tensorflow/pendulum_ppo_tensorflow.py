import os
import random
from collections import namedtuple, deque

import cv2
import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as K


Transition = namedtuple('Transition', ('s', 'a', 's_next', 'r', 'done'))


class ReplayMemoryBuffer():

    def __init__(self, buffer_size):
        self.memory = deque(maxlen=buffer_size)
    
    def push(self, *args):
        self.memory.append(*args)
    
    def sample(self, batch_size):
        random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)


class Observer():

    def __init__(self, env):
        self._env = env
    
    @property
    def action_space(self):
        return self._env.action_space
    
    @property
    def observation_space(self):
        return self._env.action_space
    
    def reset(self):
        return self.transform(self._env.reset())
    
    def step(self, a):
        o_next, r, done, info = self._env.step(a[0])
        return self.transform(o_next), self.transform(r, is_scaler=True)
                self.transform(done, is_scaler=True), info
    
    def render(self):
        return self._env.render(mode="rgb_array")
    
    def transform(self, x, is_scaler=False):
        raise NotImplementedError("You have to implement transform method")
    

class PendulumObserver(Observer):

    def transform(self, x, is_scaler=False):
        if is_scaler:
            return np.array(x, dtype=np.float32).reshape=(1, -1)
        else:
            return np.reshape(x, (1, -1)).astype(np.float32)


class PPOModel(K.Model):

    def __init__(self, n_actions):
        super(PPOModel, self).__init__()

    def call(self, inputs):
        mu = None
        sigma = None
        q = None
        return mu, sigma, q


class FNAgent():

    def __init__(self, epsilon, n_actions):
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.model = None
        self.estimate_probs = False
        self.initialized = False
    
    def save(self, model_path):
        self.model.save(model_path, overwrite=True, include_optimizer=False)
    
    @classmethod
    def load(cls, env, model_path, epsilon=1e-4):
        n_actions = env.action_space.shape[0]
        agent = cls(epsilon, n_actions)
        agent.model = K.models.load_model(model_path)
        agent.initialized = True
        return agent
    
    def initialize(self, experiences):
        raise NotImplementedError("You have to implement initialize method.")
    
    def estimate(self, s):
        raise NotImplementedError("You have to implement estimate method.")
    
    def update(self, experiences, gamma):
        raise NotImplementedError("You have to implement update method.")

    def policy(self, s):
        raise NotImplementedError("You have to implement policy method.")

    def play(self, env, episode=5, render=True):
        print("Play Mode:")
        frames_rmax = None
        for e in range(episode):
            done = False
            r_total = 0
            s = env.reset()
            frames = []
            while not done:
                if render:
                    frames.append(env.render())
                a = self.policy(s)
                s_next, r, done, _ = env.step(s)
                r_total += r
                s = s_next
            else:
                if (e == 0) or (r_total > r_total_ref):
                    r_total_ref = r_total
                    frames_rmax = frames.copy()

                print("Episode: {}, Total Reward: {}".format(e, r_total))
        return frames_rmax



class PPOAgent(FNAgent):

    def __init__(self, n_actions):
        super(PPOAgent, self).__init__(epsilon=0.0, n_actions=n_actions)
        self.n_actions = n_actions
        self.model = PPOModel(n_actions=n_actions)

    def initialize(self, experiences):
        pass
    
    def estimate(self, s, s_next, r, gamma):
        _, v = self.model(s)
        _, v_next = self.model(s_next)
        q_v = r + gamma * v_next
        return v, q_v
    
    def update(self, experiences, gamma):
        raise NotImplementedError("You have to implement update method.")

    def policy(self, s):
         mu, sigma, _ = self.model(s)
         a = np.random.normal(mu, sigma)
         return a


class Trainer():

    def __init__(self, buffer_size=10000, batch_size=64, gamma=0.99, report_interval=10, log_dir=""):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.report_interval = report_interval
        self.logger = Logger(log_dir, self.trainer_name)
        self.experiences = ReplayMemoryBuffer(buffer_size)
        self.reward_log = []
    
    @property
    def trainer_name(self):
        class_name = self.__class__.__name__
        snaked = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", class_name)
        snaked = re.sub("([a-z0-9])([A-Z])", r"\1_\2", snaked).lower()
        snaked = snaked.replace("_trainer", "")
        return snaked
    
    def train_loop(self, env, agent, episode=200, initial_count=-1, render=False, observe_interval=0):
        self.experiences = ReplayMemoryBuffer(self.buffer_size)
        self.reward_log = []
        frames = []

        for e in range(episode):
            s = env.reset()
            done = False
            r_total = 0
            while not done:
                if render:
                    env.render()
                
                a = agent.policy(s)
                s_next, r, done, _ = env.step(a)

                self.experiences.push(Transition(s, a, s_next, r, done))

                s = s_next
                r_total += r

                if len(self.experiences) < self.batch_size:
                    continue

                self.step(agent, self.experiences)
            else:
                self.reward_log.append(r_total)
                print("Episode: {}, Total Reward: {}".format(e, r_total))
    
    def step(self, agent, experiences):
        pass


class PPOTrainer(Trainer):

    def step(self, agent, experiences):
        transitions = experiences.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        batch_s = np.vstack(batch.s)
        batch_a = np.vstack(batch_a)
        batch_s_next = np.vstack(batch.s_next)
        batch_r = np.vstack(batch.r)
        batch_not_done = 1 - np.vstack(batch.done)

        agent.update(batch)
    


class Logger():

    def __init__(self, log_dir="", dir_name=""):
        self.log_dir = log_dir
        if not log_dir:
            self.log_dir = os.path.join(os.path.dirname(os.getcwd()), "logs")

        if not os.path.exists(self.log_dir):
            os.makedir(self.log_dir)
        
        if dir_name:
            self.log_dir = os.path.join(self.log_dir, dir_name)
            if not os.path.exists(self.log_dir):
                os.mkdir(self.log_dir)
        
        self._summary_writer = tf.summary.create_file_writer(self.log_dir)
    
    @property
    def writer(self):
        return self._summary_writer
    
    def path_of(self, file_name):
        return os.path.join(self.log_dir, file_name)
    
    def describe(self, name, values, episode=-1, step=-1):
        mean = np.round(np.mean(values), 3)
        std = np.round(np.std(values), 3)
        desc = "{} is {} (+/-{})".format(name, mean, std)
        if episode > 0:
            print("At episode {}, {}".format(episode, desc))
        elif step > 0:
            print("At step {}, {}".format(step, desc))
    
    def plot(self, name, values, interval=10):
        indices = list(range(0, len(values), interval))
        means = []
        stds = []
        for i in indices:
            _values = values[i:i+interval]
            means.append(np.mean(_values))
            stds.append(np.std(_values))
        means = np.array(means)
        stds = np.array(stds)

        fig = plt.figure(figsize=(6,4))
        ax = fig.add_subplot(111)
        ax.fill_between(indices, means - stds, means + stds, alpha=0.1, color="g")
        ax.plot(indices, means, "o-", color="g", label="{} per {} episode".format(name.lower(), interval))
        ax.set_xlim(0, len(values))
        ax.legend(loc="best")
        plt.show()
    
    def write(self, index, name, value):
        with self._summary_writer.as_default():
            tf.summary.scalar(name, value, step=index)


class PPOLogger(Logger):
    pass


def main():
    pass

if __name__ == "__main__":
    env = PendulumObserver(gym.make("Pendulum-v0"))
    agent = PPOAgent()
    logger = PPOLogger()

    main()
    