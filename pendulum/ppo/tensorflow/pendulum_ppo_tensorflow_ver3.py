import os
import random

import cv2
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from gym.spaces import Box, Discrete

import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras as K

tfd = tfp.distributions

def mlp(sizes, activation, output_activation=tf.identity):
    layers = []
    for i in range(len(sizes)):
        activation_func = activation if i < len(sizes) else output_activation
        layers += [K.layers.Dense(sizes[i], activation_func)]
    return K.Sequential(layers=layers)

def discount_cumsum(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::1]

class PPOObserver(object):

    def __init__(self, env):
        self._env = env
    
    @property
    def observation_space(self):
        return self._env.observation_space
    
    @property
    def action_space(self):
        return self._env.action_space
    
    def render(self):
        return self._env.render(mode="rgb_array")
    
    def reset(self):
        observation = self._env.reset()
        observation = self.transform(observation)
        return observation
    
    def step(self, action):
        observation_next, r, done, info = self._env.step(action)
        observation_next = self.transform(observation_next)

        return observation_next, r, done, info
    
    def transform(self, observation):
        observation = np.reshape(observation, (1, -1)).astype(np.float32)
        return observation

    def close(self):
        self._env.close()

class Actor(K.Model):

    def _distribution(self, observation):
        raise NotImplementedError

    def _log_prob_from_distribution(self, observation, action):
        raise NotImplementedError

    def call(self, observation, action=None):
        logp = None
        if action is not None:
            logp = self._log_prob_from_distribution(observation, action)
        return logp

class CategoricalActor(Actor):

    def __init__(self, n_observation, n_action, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp(list(hidden_sizes) + [n_action], activation)
    
    def _distribution(self, observation):
        logits = self.logits_net(observation)
        distribution = tfd.Categorical(logits=logits)
        return distribution.sample()
    
    def _log_prob_from_distribution(self, observation, action):
        logits = self.logits_net(observation)
        distribution = tfd.Categorical(logits=logits)
        prob = distribution.prob(action)
        return tf.math.log(prob)


class GaussianActor(Actor):

    def __init__(self, n_observation, n_action, hidden_sizes, activation):
        super().__init__()
        self.log_std = -0.5
        self.mu_net = mlp(list(hidden_sizes) + [n_action], activation)
    
    def _distribution(self, observation):
        mu = self.mu_net(observation)
        log_std = tf.ones(mu.shape) * self.log_std
        std = np.exp(log_std)
        distribution = tfd.Normal(loc=mu, scale=std)
        return distribution.sample()
    
    def _log_prob_from_distribution(self, observation, action):
        mu = self.mu_net(observation)
        log_std = tf.ones(mu.shape) * self.log_std
        std = np.exp(log_std)
        distribution = tfd.Normal(loc=mu, scale=std)
        prob = distribution.prob(action)
        return tf.math.log(prob)
    
class Critic(K.Model):

    def __init__(self, n_observation, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp(list(hidden_sizes) + [1], activation)
    
    def call(self, observation):
        return self.v_net(observation)

class ActorCritic(K.Model):

    def __init__(self, observation_space, action_space,
                 hidden_sizes=(64, 64), activation="tanh"):
        super().__init__()

        n_observation = observation_space.shape[0]

        if isinstance(action_space, Box):
            self.pi = GaussianActor(n_observation, action_space.shape[0],
                                    hidden_sizes, activation)
        elif isinstance(action_space, Discrete):
            self.pi = CategoricalActor(n_observation, action_space.n,
                                       hidden_sizes, activation)
        
        self.v = Critic(n_observation, hidden_sizes, activation)
    
    def step(self, observation):
        action = self.pi._distribution(observation)
        logp = self.pi._log_prob_from_distribution(observation, action)
        v = self.v(observation)

        return (action.numpy().reshape(-1), v.numpy().reshape(-1), logp.numpy().reshape(-1))
    
    def act(self, observation):
        return self.step(observation)[0]

class ReplayBuffer(object):

    def __init__(self, n_observation, n_action, size,
                 gamma=0.99, lam=0.95):
        self.observation_buffer = np.zeros((size, *n_observation), dtype=np.float32)
        self.action_buffer = np.zeros((size, *n_action), dtype=np.float32)
        self.adv_buffer = np.zeros(size, dtype=np.float32)
        self.reward_buffer = np.zeros(size, dtype=np.float32)
        self.return_buffer = np.zeros(size, dtype=np.float32)
        self.v_buffer = np.zeros(size, dtype=np.float32)
        self.logp_buffer = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size
    
    def store(self, observation, action, reward, v, logp):
        assert self.ptr < self.max_size
        self.observation_buffer[self.ptr] = observation
        self.action_buffer[self.ptr] = action
        self.reward_buffer[self.ptr] = reward
        self.v_buffer[self.ptr] = v
        self.logp_buffer[self.ptr] = logp
        self.ptr += 1
    
    def finish_path(self, last_v=0):
        path_slice = slice(self.path_start_idx, self.ptr)
        rewards = np.append(self.reward_buffer[path_slice], last_v)
        vs = np.append(self.v_buffer[path_slice], last_v)

        deltas = rewards[:-1] + self.gamma * vs[1:] - vs[:-1]
        self.adv_buffer[path_slice] = discount_cumsum(deltas, self.gamma*self.lam)

        self.return_buffer[path_slice] = discount_cumsum(rewards, self.gamma)[:-1]

        self.path_start_idx = self.ptr
    
    def get(self):
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0
        adv_mean = np.mean(self.adv_buffer)
        adv_std = np.std(self.adv_buffer)
        self.adv_buffer = (self.adv_buffer - adv_mean) / (adv_std + 1e-11)
        data = dict(observation=self.observation_buffer,
                    action=self.action_buffer,
                    ret=self.return_buffer,
                    adv=self.adv_buffer,
                    logp=self.logp_buffer)
        return data


class PPOMaster(object):

    def __init__(self,
                 actor_critic,
                 gamma=0.99,
                 clip_ratio=0.2,
                 pi_learning_rate=3e-4,
                 v_learning_rate=1e-3,
                 train_pi_iters=80,
                 train_v_iters=80,
                 lam=0.97,
                 target_kl=0.01):
        self.actor_critic = actor_critic
        self.gamma=gamma
        self.clip_ratio=clip_ratio
        self.pi_learning_rate=pi_learning_rate
        self.v_learning_rate=v_learning_rate
        self.train_pi_iters=train_pi_iters
        self.train_v_iters=train_v_iters
        self.lam=lam
        self.target_kl=target_kl
    
        self.optim_pi = K.optimizers.Adam(learning_rate=self.pi_learning_rate)
        self.optim_v = K.optimizers.Adam(learning_rate=v_learning_rate)

    def compute_loss_pi(self, data):
        observation, action, adv, logp_old = data['observation'], data['action'], data['adv'], data['logp']
        
        logp = self.actor_critic.pi(observation, action)
        ratio = tf.math.exp(logp - logp_old)
        clip_adv = tf.clip_by_value(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * adv
        loss_pi = -tf.reduce_mean(tf.reduce_min([ratio*adv, clip_adv], axis=0))

        return loss_pi
    
    def compute_loss_v(self, data):
        observation, ret = data['observation'], data['ret']
        ret = np.reshape(ret, (-1, 1)).astype(np.float32)
        return tf.reduce_mean((self.actor_critic.v(observation) - ret)**2)
    
    def update(self, data):
        loss_pis = []
        for _ in range(self.train_pi_iters):
            with tf.GradientTape() as tape_pi:
                loss_pi = self.compute_loss_pi(data)
            grads_pi = tape_pi.gradient(loss_pi, self.actor_critic.pi.trainable_variables)
            self.optim_pi.apply_gradients(zip(grads_pi, self.actor_critic.pi.trainable_variables))
            loss_pis.append(loss_pi)
        
        loss_vs = []
        for _ in range(self.train_v_iters):
            with tf.GradientTape() as tape_v:
                loss_v = self.compute_loss_v(data)
            grads_v = tape_v.gradient(loss_v, self.actor_critic.v.trainable_variables)
            self.optim_v.apply_gradients(zip(grads_v, self.actor_critic.v.trainable_variables))
            loss_vs.append(loss_v)

        return loss_pi.numpy().mean(), loss_v.numpy().mean()

class PPOWorker(object):

    def __init__(self):
        pass