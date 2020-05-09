import os
import random
from collections import deque, namedtuple

import cv2
import gym
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as K


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input: 
        vector x, 
        [x0, 
         x1, 
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class PPOBuffer():

    def __init__(self, size, n_observations, n_actions, gamma=0.99, lam=0.95):
        self.s_buffer = np.zeros((size, n_observations))
        self.a_buffer = np.zeros((size, n_actions))
        self.r_buffer = np.zeros((size,))
        self.adv_buffer = np.zeros((size,))
        self.ret_buffer = np.zeros((size,))
        self.v_buffer = np.zeros((size,))
        self.logp_buffer = np.zeros((size,))
        self.gamma = gamma
        self.lam = lam
        self.ptr = 0
        self.path_start_idx = 0
        self.max_size = size
    
    def store(self, s, a, r, v, logp):
        self.s_buffer[self.ptr] = s
        self.a_buffer[self.ptr] = a
        self.r_buffer[self.ptr] = r
        self.v_buffer[self.ptr] = v
        self.logp_buffer[self.ptr] = logp
        self.ptr += 1
    
    def finish_path(self, last_val=0):
        path_slice = slice(self.path_start_idx, self.ptr)
        rs = np.append(self.r_buffer[path_slice], last_val)
        vs = np.append(self.v_buffer[path_slice], last_val)

        deltas = rs[:-1] + self.gamma * vs[1:] - vs[:-1]
        self.adv_buffer[path_slice] = np.array([np.sum([(self.gamma*self.lam)**(j-t) * deltas[j]\
                                                for j in range(t, len(deltas))]) for t in range(len(deltas))],\
                                                dtype=np.float32)
        self.ret_buffer[path_slice] = np.array([np.sum([self.gamma**(j-t) * rs[j] for j in range(t, len(rs)-1)])\
                                                for t in range(len(rs)-1)])
        # self.adv_buffer[path_slice] = discount_cumsum(deltas, self.gamma*self.lam)
        # self.ret_buffer[path_slice] = discount_cumsum(rs, self.gamma)[:-1]
        self.path_start_idx = self.ptr
    
    def get(self):
        self.ptr = 0
        self.path_start_idx = 0

        adv_mean = np.mean(self.adv_buffer)
        adv_std = np.std(self.adv_buffer)
        self.adv_buffer = (self.adv_buffer - adv_mean) / adv_std
        # ret_mean = np.mean(self.ret_buffer)
        # ret_std = np.std(self.ret_buffer)
        # self.ret_buffer = (self.ret_buffer - ret_mean) / ret_std
        data = dict(s=self.s_buffer, a=self.a_buffer, ret=self.ret_buffer.reshape(-1,1),
                    adv=self.adv_buffer.reshape(-1,1), logp=self.logp_buffer.reshape(-1,1))
        return data


class PPOObserver():

    def __init__(self, env):
        self._env = env
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
    
    def reset(self):
        s, _, _, _ = self.transform(self._env.reset())
        return s
    
    def step(self, a):
        o_next, r, done, info = self._env.step(a[0])
        return self.transform(o_next, r, done, info)
    
    def transform(self, o_next, r=None, done=None, info=None):
        s_next = np.reshape(o_next, (1, -1)).astype(np.float32)
        r = np.array([[r]], dtype=np.float32)
        return s_next, r, done, info


class Actor(K.Model):

    def __init__(self, n_actions):
        super(Actor, self).__init__()
        normal = K.initializers.glorot_normal()
        self.fc1 = K.layers.Dense(64, activation="relu", kernel_initializer=normal)
        self.fc2 = K.layers.Dense(64, activation="relu", kernel_initializer=normal)
        self.out = K.layers.Dense(n_actions, activation="tanh", kernel_initializer=normal)
        self.logstd = -0.5

    def call(self, inputs):
        x = self.fc1(inputs)
        x = self.fc2(x)
        mu = self.out(x)
        logstd = self.logstd * np.ones(mu.shape)
        logstd = logstd.astype(np.float32)
        return mu, logstd


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


class PPOAgent():

    def __init__(self, n_actions):
        self.actor = Actor(n_actions)
        self.critic = Critic()
    
    def __call__(self, inputs):
        mu, log_std = self.actor(inputs)
        std = np.exp(log_std)
        a = np.random.normal(mu, std)
        logp = np.sum(-0.5 * (((a - mu) / (std + 1e-8))**2 + 2*log_std + np.log(2*np.pi)))
        logp = np.reshape(logp, (-1, 1))
        v = self.critic(inputs)
        return a, logp, v


if __name__ == "__main__":
    GAMMA = 0.99
    LAMMDA = 0.95
    EPOCH = 2049
    TRAJECTORY_SIZE = 200
    MAX_EP_LEN = 200
    TRAIN_PI_ITER = 10
    TRAIN_V_ITER = 10
    ACTOR_LR = 1e-5
    CRITIC_LR = 1e-4
    CLIP_RATIO = 0.2

    env = PPOObserver(gym.make('Pendulum-v0'))
    n_observations = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    agent = PPOAgent(n_actions)
    buffer = PPOBuffer(TRAJECTORY_SIZE, n_observations, n_actions, gamma=GAMMA, lam=LAMMDA)

    optim_actor = K.optimizers.Adam(learning_rate=ACTOR_LR)
    optim_critic = K.optimizers.Adam(learning_rate=CRITIC_LR)

    def loss_fn_actor(mu, log_std, logp_old, adv, clip_ratio=0.2):
        std = tf.stop_gradient(tf.math.exp(log_std))
        a = np.random.normal(tf.stop_gradient(mu), std)
        logp = tf.reduce_sum(-0.5 * (((a - mu) / (std+1e-8))**2 + 2*log_std + tf.math.log(2*np.pi)), axis=1)
        logp = tf.reshape(logp, (-1, 1))
        ratio = tf.math.exp(logp - tf.stop_gradient(logp_old))
        clip_adv = tf.clip_by_value(ratio, 1-clip_ratio, 1+clip_ratio) * adv
        return -tf.reduce_mean(tf.reduce_min(tf.concat([ratio*adv, clip_adv], axis=1),axis=1))
    loss_fn_critic = K.losses.MeanSquaredError()
    
    s = env.reset()
    ep_rt = 0
    ep_len = 0
    for e in range(EPOCH):
        for t in range(TRAJECTORY_SIZE):
            a, logp, v = agent(s)
            s_next, r, done, _ = env.step(a)

            ep_rt += r.item()
            ep_len += 1

            buffer.store(s, a, r, v, logp)
            
            s = s_next

            timeout = (ep_len == MAX_EP_LEN)
            terminal = (done or timeout)
            epoch_ended = (t == (TRAJECTORY_SIZE - 1))

            if terminal or epoch_ended:
                if timeout or epoch_ended:
                    _, _, v_next = agent(s)
                else:
                    v_next = 0
                buffer.finish_path(v_next)
                s = env.reset()
                print("Total Reward: {}".format(ep_rt))
                ep_rt = 0
                ep_len = 0
            
        data = buffer.get()
        data_s = data['s'].astype(np.float32)
        data_a = data['a'].astype(np.float32)
        data_ret = data['ret'].astype(np.float32)
        data_adv = data['adv'].astype(np.float32)
        data_logp_old = data['logp'].astype(np.float32)

        for i in range(TRAIN_PI_ITER):
            with tf.GradientTape() as tape_actor:
                data_mu_, data_log_std_ = agent.actor(data_s)
                loss_actor = loss_fn_actor(data_mu_, data_log_std_, data_logp_old, data_adv)
            grads_actor = tape_actor.gradient(loss_actor, agent.actor.trainable_variables)
            # print(grads_actor)
            optim_actor.apply_gradients(zip(grads_actor, agent.actor.trainable_variables))
            # print("Training Pi: {}".format(i))

        for i in range(TRAIN_V_ITER):
            with tf.GradientTape() as tape_critic:
                data_v = agent.critic(data_s)
                loss_critic = loss_fn_critic(data_v, data_ret)
            grads_critic = tape_critic.gradient(loss_critic, agent.critic.trainable_weights)
            # print(grads_critic)
            optim_critic.apply_gradients(zip(grads_critic, agent.critic.trainable_variables))
            # print("Training V: {}".format(i))

        print("{} epoch done, loss_actor: {}, loss_critic: {}".format(e+1, loss_actor, loss_critic))

        