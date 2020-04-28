import os
import random
from collections import deque, namedtuple

import cv2
import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras as K

tf.compat.v1.disable_eager_execution()

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


def make_graph(steps, savedir="img", savefile="results_cart_pole_dqn.png"):
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.plot(np.arange(1, len(steps)+1, 1), steps)
    ax.set_xlim(0, len(steps))
    ax.set_ylim(0, 210)
    path = os.path.join(os.getcwd(), savedir)
    if not os.path.exists(path):
        os.mkdir(path)
    path = os.path.join(path, savefile)
    plt.savefig(path, dpi=300)
    plt.show()


def make_movie(frames, savedir="movie", savefile="movie_cartpole_dqn.mp4"):
    path = os.path.join(os.getcwd(), savedir)
    if not os.path.exists(path):
        os.mkdir(path)
    path = os.path.join(path, savefile)

    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
    video = cv2.VideoWriter(path, fourcc, 50.0, (600, 600))

    for frame in frames:
        frame = cv2.resize(frame, (600,600))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video.write(frame)

    video.release()


class PolicyGradient():

    def __init__(self, input_shape, n_actions, optimizer):
        self.n_actions = n_actions
        self.normal = K.initializers.glorot_normal()
        self.model = K.Sequential()
        self.model.add(K.layers.Dense(10, input_shape=(input_shape,), activation='relu', kernel_initializer=self.normal))
        self.model.add(K.layers.Dense(10, activation='relu', kernel_initializer=self.normal))
        self.model.add(K.layers.Dense(n_actions, activation='softmax', kernel_initializer=self.normal))
        self.set_updater(optimizer)
    
    def set_updater(self, optimizer):
        actions = tf.compat.v1.placeholder(shape=(None), dtype="int32")
        rewards = tf.compat.v1.placeholder(shape=(None), dtype="float32")
        one_hot_actions = tf.one_hot(actions, self.n_actions, axis=1)
        action_probs = self.model.output
        selected_action_probs = tf.reduce_sum(one_hot_actions * action_probs, axis=1)
    
        clipped = tf.clip_by_value(selected_action_probs, 1e-10, 1.0)
        loss = -tf.math.log(clipped) * rewards
        loss = tf.reduce_mean(loss)

        updates = optimizer.get_updates(loss=loss, params=self.model.trainable_weights)
        self._updater = K.backend.function(
            inputs=[self.model.input, actions, rewards],
            outputs=[loss],
            updates=updates
        )
    
    def update(self, states, actions, rewards):
        loss = self._updater([states, actions, rewards])
        return loss


if __name__ == "__main__":
    LEARNING_RATE = 0.01
    EPISODE = 400
    GAMMA = 0.99
    MAXLEN = 10

    env = gym.make("CartPole-v0").unwrapped
    n_observation = env.observation_space.shape[0]
    n_actions = env.action_space.n
    
    optimizer = K.optimizers.Adam(learning_rate=LEARNING_RATE)
    agent = PolicyGradient(n_observation, n_actions, optimizer)

    successes = deque(maxlen=MAXLEN)
    steps = []
    for e in range(EPISODE):
        o = env.reset()
        done = False
        step = 0
        memory = deque()
        while not done:
            step += 1

            s = np.reshape(o, (1, -1))
            a_probs = agent.model.predict(s)
            a = np.random.choice([0, 1], p=a_probs[0])
            o_next, _, done, _ = env.step(a)
            a = np.array([a], dtype=np.int32)

            if step >= 200:
                done = True
            
            if done:
                s_next = None
                if step < 200:
                    r = np.array([-1.0])
                    successes.append(0)
                else:
                    r = np.array([1.0])
                    successes.append(1)
            else:
                s_next = np.reshape(o_next, (1, -1))
                r = np.array([0.0])
            
            memory.append(Transition(s, a, s_next, r))

            o = o_next
        else:
            steps.append(step)
            print("Episode: {}, Total Step: {}".format(e+1, step))

            transitions = Transition(*zip(*memory))
            states = np.vstack(transitions.state)
            actions = np.hstack(transitions.action)
            rewards = np.hstack(transitions.reward)
            rewards_mean = rewards.mean()
            rewards_std = rewards.std()
            rewards = (rewards - rewards_mean) / rewards_std

            g_t = np.zeros(step, dtype=np.float32)
            for t in range(step):
                g_t[t] = sum([r*(GAMMA**i) for i, r in enumerate(rewards[t:])])
            
            loss = agent.update(states, actions, g_t)
        
        if sum(successes) == MAXLEN:
            print("{} times success!".format(MAXLEN))
            break

    
    savedir = "img"
    savefile = "result_cart_pole_policy_gradient_tensorflow.png"
    make_graph(steps, savedir=savedir, savefile=savefile)


    o = env.reset()
    done = False
    step = 0
    frames = []
    while not done:
        step += 1
        frames.append(env.render(mode="rgb_array"))
        s = np.reshape(o, (1, -1))
        a_probs = agent.model.predict(s)
        a = np.random.choice([0, 1], p=a_probs[0])
        o_next, _, done, _ = env.step(a)
        o = o_next
        if step >= 1000:
            break
    else:
        print("Total Step: {}".format(step))
    savedir = "movie"
    savefile = "movie_cart_pole_policy_gradient_tensorflow.mp4"
    make_movie(frames, savedir=savedir, savefile=savefile)
    env.close()