import os
import re
import random
from collections import namedtuple, deque

import cv2
import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as K


EPISODE_SIZE = 10000
EPOCH_SIZE = 10
BATCH_SIZE = 64
TRAJECTORY_SIZE = 2048
GAMMA = 0.99
LAMBDA = 0.95


Transition = namedtuple('Transition', ('s', 'a', 'r', 'done'))


class ReplayMemoryBuffer():

    def __init__(self, buffer_size):
        self.memory = deque(maxlen=buffer_size)
    
    def push(self, *args):
        self.memory.append(Transition(*args))
    
    def get(self):
        return self.memory
    
    def clear(self):
        self.memory.clear()
    
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
        return self.transform(o_next), r, done, info
    
    def render(self):
        return self._env.render(mode="rgb_array")
    
    def transform(self, x):
        raise NotImplementedError("You have to implement transform method")
    

class PendulumObserver(Observer):

    def transform(self, x):
        return np.reshape(x, (1, -1)).astype(np.float32)


class Actor(K.Model):

    def __init__(self, n_actions):
        super(Actor, self).__init__()
        normal = K.initializers.glorot_normal()
        self.fc1 = K.layers.Dense(64, activation="tanh", kernel_initializer=normal)
        self.fc2 = K.layers.Dense(64, activation="tanh", kernel_initializer=normal)
        self.mu = K.layers.Dense(n_actions, activation="tanh", kernel_initializer=normal)
        self.logstd = np.zeros((n_actions,)).astype(np.float32)

    def call(self, inputs):
        x = self.fc1(inputs)
        x = self.fc2(x)
        return self.mu(x)


class Critic(K.Model):

    def __init__(self):
        super(Critic, self).__init__()
        normal = K.initializers.glorot_normal()
        self.fc1 = K.layers.Dense(64, activation="relu", kernel_initializer=normal)
        self.fc2 = K.layers.Dense(64, activation="relu", kernel_initializer=normal)
        self.out = K.layers.Dense(1, kernel_initializer=normal)
    
    def call(self, inputs):
        x = self.fc1(inputs)
        x = self.fc2(x)
        return self.out(x)


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
    
    def initialize(self, optimizer):
        raise NotImplementedError("You have to implement initialize method.")
    
    def estimate(self, s):
        raise NotImplementedError("You have to implement estimate method.")
    
    def update(self, experiences):
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
        self.actor = Actor(n_actions)
        self.critic = Critic()
    
    @classmethod
    def load(cls, env, actor_path, critic_path, epsilon=1e-4):
        n_actions = env.action_space.shape[0]
        agent = cls(epsilon, n_actions)
        agent.actor = K.models.load_model(actor_path)
        agent.critic = K.models.load_model(critic_path)
        agent.initialized = True
        return agent

    def initialize(self, optim_actor, optim_critic):
        self.optim_actor = optim_actor
        self.optim_critic = optim_critic
    
    def estimate(self, s):
        return self.critic(s)

    def calc_logprob(self, s, a):
        mu = self.actor(s)
        logstd = self.actor.logstd * np.ones(mu.shape)
        return -((a - mu)**2 )/ (2 * np.exp(logstd)) - np.log(np.sqrt(2 * np.exp(logstd)))
    
    def update(self, batch_s_v, batch_a_v, batch_adv_v, batch_ref_v, batch_old_logprob_v, eps):
        with tf.GradientTape() as tape_critic:
            val_v = self.estimate(batch_s_v)
            loss_critic = tf.reduce_mean((val_v - tf.stop_gradient(batch_ref_v))**2)
        
        grads_critic = tape_critic.gradient(loss_critic, self.critic.trainable_weights)
        self.optim_critic.apply_gradients(zip(grads_critic, self.critic.trainable_weights))
        
        with tf.GradientTape() as tape_actor:
            logprob_v = self.calc_logprob(batch_s_v, tf.stop_gradient(batch_a_v))
            ratio_v = tf.exp(logprob_v - batch_old_logprob_v)
            surr_obj_v = ratio_v * tf.stop_gradient(batch_adv_v)
            c_ratio_v = tf.clip_by_value(ratio_v, 1-eps, 1+eps)
            c_surr_obj_v = c_ratio_v * tf.stop_gradient(batch_adv_v)
            loss_actor = -tf.math.segment_min([surr_obj_v, c_surr_obj_v], [0, 0])

        grads_actor = tape_actor.gradient(loss_actor, self.actor.trainable_weights)
        self.optim_actor.apply_gradients(zip(grads_actor, self.actor.trainable_weights))


    def policy(self, s):
         mu = self.actor(s)
         logstd = self.actor.logstd * np.ones(mu.shape)
         a = np.random.normal(mu, np.exp(logstd)).astype(np.float32)
         return a


class Trainer():

    def __init__(self, buffer_size=10000, trajectory_size=2049, epoch_size=10, batch_size=64, gamma=0.99,
                 gae_lambda=0.95, lr_actor=1e-4, lr_critic=1e-3, eps=0.2, report_interval=10, log_dir=""):
        self.buffer_size = buffer_size
        self.epoch_size = epoch_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.trajectory_size = trajectory_size
        self.gae_lambda = gae_lambda
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.eps = eps
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
        agent.initialize(optim_actor=K.optimizers.Adam(learning_rate=self.lr_actor),
                         optim_critic=K.optimizers.Adam(learning_rate=self.lr_critic))
        
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

                self.experiences.push(s, a, r, done)

                s = s_next
                r_total += r

                if len(self.experiences) < self.trajectory_size:
                    continue

                transitions = self.experiences.get()
                trajs = Transition(*zip(*transitions))

                self.step(agent, trajs)

                self.experiences.clear()
            else:
                self.reward_log.append(r_total)
                print("Episode: {}, Total Reward: {}".format(e, r_total))
    
    def step(self, agent, trajs):
        raise NotImplementedError("You have to implement step method.")


class PPOTrainer(Trainer):

    def step(self, agent, trajs):
        trajs_s_v = np.vstack(trajs.s)
        trajs_a_v = np.vstack(trajs.a)
        trajs_r_v = np.vstack(trajs.r)
        trajs_done = np.array(trajs.done)
        trajs_adv_v, trajs_ref_v = self.calc_adv(agent, trajs_s_v, trajs_r_v, trajs_done)
        trajs_old_logprob_v = agent.calc_logprob(trajs_s_v, trajs_a_v)

        for epoch in range(self.epoch_size):
            for i in range(0, len(trajs)-1, self.batch_size):
                batch_s_v = trajs_s_v[i:i+self.batch_size]
                batch_a_v = trajs_a_v[i:i+self.batch_size]
                batch_r_v = trajs_r_v[i:i+self.batch_size]
                batch_adv_v = trajs_adv_v[i:i+self.batch_size]
                batch_ref_v = trajs_ref_v[i:i+self.batch_size]
                batch_old_logprob_v = trajs_old_logprob_v[i:i+self.batch_size]

                agent.update(batch_s_v, batch_a_v, batch_adv_v, batch_ref_v, batch_old_logprob_v, self.eps)
    
    def calc_adv(self, agent, trajs_s_v, trajs_r_v, trajs_done):
        val_v = agent.estimate(trajs_s_v)
        last_gae = np.array([0.0], dtype=np.float32)
        adv_v = []
        ref_v = []
        for r, val, val_next, done in zip(reversed(trajs_r_v[:-1]), reversed(val_v[:-1]), reversed(val_v[1:]), reversed(trajs_done[:-1])):
            if done:
                delta = r - val
                last_gae = delta
            else:
                delta = r + self.gamma * val_next - val
                last_gae = delta + self.gamma * self.gae_lambda * last_gae
            adv_v.append(last_gae)
            ref_v.append(last_gae + val)
        adv_v = np.array(list(reversed(adv_v)), dtype=np.float32)
        ref_v = np.array(list(reversed(ref_v)), dtype=np.float32)

        return adv_v, ref_v
    


class Logger():

    def __init__(self, log_dir="", dir_name=""):
        self.log_dir = log_dir
        if not log_dir:
            self.log_dir = os.path.join(os.path.dirname(os.getcwd()), "logs")

        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
        
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


def main(env, agent, logger):
    trainer = PPOTrainer()
    trainer.train_loop(env, agent)

if __name__ == "__main__":
    env = PendulumObserver(gym.make("Pendulum-v0"))
    n_actions = env.action_space.shape[0]
    agent = PPOAgent(n_actions)
    logger = PPOLogger()

    main(env, agent, logger)
    