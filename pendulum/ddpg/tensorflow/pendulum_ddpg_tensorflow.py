import os
import random
from collections import deque, namedtuple

import cv2
import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as K
tf.compat.v1.disable_eager_execution()


Transition = namedtuple('Transition', ('s', 'a', 's_next', 'r', 'done'))


class ReplayMemory():

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
    
    def push(self, *args):
        self.memory.append(*args)
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)


class DDPG():

    def __init__(self, n_observation, n_actions):
        self.n_observation = n_observation
        self.n_actions = n_actions
        normal = K.initializers.glorot_normal()

        actor_net = K.Sequential()
        actor_net.add(K.layers.Dense(400, input_shape=(n_observation,), activation="relu", kernel_initializer=normal))
        actor_net.add(K.layers.Dense(300, activation="relu", kernel_initializer=normal))
        actor_net.add(K.layers.Dense(n_actions, kernel_initializer=normal))
        self.actor_net = K.Model(inputs=actor_net.input, outputs=actor_net.output)

        obs_input = K.layers.Input(shape=(n_observation,), dtype="float32")
        actions_input = K.layers.Input(shape=(n_actions,), dtype="float32")
        obs_net = K.layers.Dense(400, input_shape=(n_observation,), activation="relu", kernel_initializer=normal)
        critic_input = K.layers.Concatenate()([obs_net(obs_input), actions_input])
        critic_net = K.Sequential()
        critic_net.add(K.layers.Dense(32, activation="relu", kernel_initializer=normal))
        critic_net.add(K.layers.Dense(32, activation="relu", kernel_initializer=normal))
        critic_net.add(K.layers.Dense(32, activation="relu", kernel_initializer=normal))
        critic_net.add(K.layers.Dense(1, kernel_initializer=normal))
        critic_eval = critic_net(critic_input)
        self.critic_net = K.Model(inputs=[obs_input, actions_input], outputs=critic_net.output)

        print(self.actor_net.summary())
        print(self.critic_net.summary())


if __name__ == "__main__":
    CAPACITY = 10000
    EPISODE = 1
    BATCH_SIZE = 64

    env = gym.make("Pendulum-v0")
    n_observation = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    memory = ReplayMemory(CAPACITY)

    agent = DDPG(n_observation, n_actions)

    for e in range(EPISODE):
        o = env.reset()
        done = False
        step = 0
        r_total = 0
        while not done:
            step += 1
            s = np.reshape(o, (1, -1))
            a = env.action_space.sample()
            
            o_next, r, done, _ = env.step(a)
            
            r_total += r
            r = np.array([[r]])
            s_next = np.reshape(o_next, (1, -1))
            memory.push(Transition(s, a, s_next, r, done))

            o = o_next

            if len(memory) < BATCH_SIZE:
                continue
            
            transitions = memory.sample(BATCH_SIZE)
            batch = Transition(*zip(*transitions))
            batch_s = np.vstack(batch.s)
            batch_a = np.vstack(batch.a)
            batch_s_next = np.vstack(batch.s_next)
            batch_r = np.vstack(batch.r)
            batch_done = np.array(batch.done)

        else:
            print("Episode: {}, Total Reward: {}".format(e, r_total))
