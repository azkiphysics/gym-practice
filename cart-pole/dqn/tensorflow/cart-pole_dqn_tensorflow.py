import os
import random
from collections import namedtuple, deque

import numpy as np
import matplotlib.pyplot as plt
import gym
import cv2
from tensorflow import keras as K


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


class ReplayMemory():

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
    
    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)


if __name__ == "__main__":
    EPS_START = 0.9
    EPS_END = 0.005
    EPS_DECAY = 50
    BATCH_SIZE = 128
    LEARNING_RATE = 0.0001
    EPISODE = 200
    CAPACITY = 10000
    GAMMA = 0.99
    TARGET_UPDATE = 10

    env = gym.make("CartPole-v0").unwrapped

    n_observation = env.observation_space.shape[0]
    n_action = env.action_space.n

    normal = K.initializers.glorot_normal()
    policy_network = K.Sequential()
    policy_network.add(K.layers.Dense(32, input_shape=(n_observation,), activation="relu", kernel_initializer=normal))
    policy_network.add(K.layers.Dense(32, activation="relu", kernel_initializer=normal))
    policy_network.add(K.layers.Dense(n_action, activation="linear", kernel_initializer=normal))

    target_network = K.models.clone_model(policy_network)
    
    optimizer = K.optimizers.Adam(learning_rate=LEARNING_RATE)
    policy_network.compile(optimizer=optimizer, loss="huber_loss")

    memory = ReplayMemory(CAPACITY)

    print("Start: Training")
    successes = deque(maxlen=10)
    steps = []
    for e in range(EPISODE):
        o = env.reset()
        done = False
        step = 0
        epsilon = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * e / EPS_DECAY)
        while not done:
            #env.render()
            step += 1
            s = np.reshape(o, (1, -1))
            if np.random.rand() < epsilon:
                a = env.action_space.sample()
            else:
                a = policy_network.predict(s).argmax(axis=1)[0]
            o_next, _, done, _ = env.step(a)
            if step >= 200:
                done = True
            if done:
                s_next = None
                if step < 200:
                    r = -1.0
                    successes.append(0)
                else:
                    r = 1.0
                    successes.append(1)
            else:
                s_next = np.reshape(o_next, (1, -1))
                r = 0.0
                o = o_next
            memory.push(s, a, s_next, r)

            if len(memory) < BATCH_SIZE:
                continue

            transitions = memory.sample(BATCH_SIZE)
            batch = Transition(*zip(*transitions))
            non_final_mask = np.array(tuple(map(lambda s: s is not None, batch.next_state)))
            non_final_batch_s_next = np.vstack([s_next for s_next in batch.next_state if s_next is not None])
            batch_s = np.vstack(batch.state)
            batch_a = np.hstack(batch.action)
            batch_r = np.hstack(batch.reward)
            
            estimateds = policy_network.predict(batch_s)
            future = np.zeros(BATCH_SIZE)
            future[non_final_mask] = target_network.predict(non_final_batch_s_next).max(axis=1)
            estimateds_ = batch_r + GAMMA * future
            
            for i in range(len(estimateds_)):
                estimateds[i][batch_a[i]] = estimateds_[i]
            loss = policy_network.train_on_batch(batch_s, estimateds)
        else:
            steps.append(step)
            print("episode: {}, step: {}".format(e, step))
        
        if e % TARGET_UPDATE == 0:
            target_network.set_weights(policy_network.get_weights())
        
        if sum(successes) == 10:
            print("10 Times Success!!")
            break
    print("End: Training")
    

    savedir = "img"
    savefile = "result_cart_pole_dqn_tensorflow.png"
    make_graph(steps, savedir=savedir, savefile=savefile)


    o = env.reset()
    done = False
    step = 0
    frames = []
    while not done:
        step += 1
        frames.append(env.render(mode='rgb_array'))
        s = np.reshape(o, (1, -1))
        a = policy_network.predict(s).argmax(axis=1)[0]
        o_next, _, done, _ = env.step(a)
        o = o_next
        if step >= 1000:
            break
    else:
        print("Total Step: {}.".format(step))
        savedir="movie"
        savefile="movie_cart_pole_dqn_tensorflow.mp4"
        make_movie(frames, savedir=savedir, savefile=savefile)
    
    env.close()