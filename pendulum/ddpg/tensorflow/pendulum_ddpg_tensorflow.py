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


def make_graph(rewards, savedir="img", savefile="graph.png"):
    path = os.path.join(os.getcwd(), savedir)
    if not os.path.exists(path):
        os.mkdir(path)
    path = os.path.join(path, savefile)
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(111)
    ax.plot(np.arange(1, len(rewards)+1, 1), rewards)
    ax.set_xlim(0, len(rewards))
    plt.savefig(path, dpi=300)

def make_movie(frames, savedir="movie", savefile="movie.mp4"):
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

def save_model(model, savedir="model", savefile="model.h5"):
    path = os.path.join(os.getcwd(), savedir)
    if not os.path.exists(path):
        os.mkdir(path)
    path = os.path.join(path, savefile)
    model.save(path)


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

    def __init__(self, n_observation, n_actions, actor_optimizer, critic_optimizer):
        self.n_observation = n_observation
        self.n_actions = n_actions
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.make_model()
        self.set_updater()

    def make_model(self):
        normal = K.initializers.glorot_normal()
        actor_net = K.Sequential()
        actor_net.add(K.layers.Dense(400, input_shape=(self.n_observation,), activation="relu", kernel_initializer=normal))
        actor_net.add(K.layers.Dense(300, activation="relu", kernel_initializer=normal))
        actor_net.add(K.layers.Dense(self.n_actions, activation="tanh", kernel_initializer=normal))
        self.actor_net = K.Model(inputs=actor_net.input, outputs=actor_net.output)
        self.actor_target_net = K.models.clone_model(self.actor_net)

        obs_input = K.layers.Input(shape=(self.n_observation,), dtype="float32")
        actions_input = K.layers.Input(shape=(self.n_actions,), dtype="float32")
        obs_net = K.layers.Dense(400, input_shape=(self.n_observation,), activation="relu", kernel_initializer=normal)
        critic_input = K.layers.Concatenate()([obs_net(obs_input), actions_input])
        critic_net = K.Sequential()
        critic_net.add(K.layers.Dense(300, activation="relu", kernel_initializer=normal))
        critic_net.add(K.layers.Dense(1, kernel_initializer=normal))
        critic_eval = critic_net(critic_input)
        self.critic_net = K.Model(inputs=[obs_input, actions_input], outputs=critic_eval)
        self.critic_target_net = K.models.clone_model(self.critic_net)

        print(self.actor_net.summary())
        print(self.critic_net.summary())

    def set_updater(self):
        s = tf.compat.v1.placeholder(shape=(None), dtype="float32")
        a = tf.compat.v1.placeholder(shape=(None), dtype="float32")
        estimated = tf.compat.v1.placeholder(shape=(None), dtype="float32")

        q = self.critic_net([s, a])
        critic_loss = tf.reduce_mean((q-estimated)**2)
        critic_updates = self.critic_optimizer.get_updates(loss=critic_loss, params=self.critic_net.trainable_weights)
        self._critic_updater = K.backend.function(
            inputs = [s, a, estimated],
            outputs = [critic_loss],
            updates = critic_updates
        )

        a_ = self.actor_net(s)
        actor_loss = -tf.reduce_mean(self.critic_net([s, a_]))
        actor_updates = self.actor_optimizer.get_updates(loss=actor_loss, params=self.actor_net.trainable_weights)
        self._actor_updater = K.backend.function(
            inputs = [s],
            outputs = [actor_loss],
            updates = actor_updates
        )
    
    def update(self, s, a, estimated, tau=0.01):
        self._critic_updater([s, a, estimated])
        self.critic_net.trainable = False
        self._actor_updater([s])
        self.critic_net.trainable = True
        self.soft_update(tau=tau)
    
    def soft_update(self, tau=0.01):
        weights = np.array(self.actor_net.get_weights())
        target_weights = np.array(self.actor_target_net.get_weights())
        self.actor_target_net.set_weights(weights*tau + target_weights*(1-tau))
        weights = np.array(self.critic_net.get_weights())
        target_weights = np.array(self.critic_target_net.get_weights())
        self.critic_target_net.set_weights(weights*tau + target_weights*(1-tau))
    
    def get_action(self, s, noise_scale=0.1, a_limit=1.0):
        a = self.actor_net.predict(s) + noise_scale * np.array([[np.random.randn()]])
        a = np.clip(a, -a_limit, a_limit)
        return a



if __name__ == "__main__":
    CAPACITY = 10000
    EPISODE = 200
    BATCH_SIZE = 64
    GAMMA = 0.99
    ACTOR_LR = 1e-4
    CRITIC_LR = 1e-3

    env = gym.make("Pendulum-v0")
    n_observation = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    memory = ReplayMemory(CAPACITY)

    actor_optimizer = K.optimizers.Adam(learning_rate=ACTOR_LR)
    critic_optimizer = K.optimizers.Adam(learning_rate=CRITIC_LR)
    agent = DDPG(n_observation, n_actions, actor_optimizer, critic_optimizer)
    
    r_totals = []
    for e in range(EPISODE):
        o = env.reset()
        done = False
        step = 0
        r_total = 0
        while not done:
            step += 1
            s = np.reshape(o, (1, -1))
            a = agent.get_action(s)
            
            o_next, r, done, _ = env.step(a[0])
            
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

            batch_a_next = agent.actor_target_net.predict([batch_s_next])
            q_next = agent.critic_target_net.predict([batch_s_next, batch_a_next])
            q_next[batch_done] = 0.0

            estimateds = batch_r + GAMMA * q_next
            estimateds = tf.stop_gradient(estimateds)

            agent.update(batch_s, batch_a, estimateds)

        else:
            r_totals.append(r_total)
            print("Episode: {}, Total Reward: {}".format(e+1, r_total))
    

    savedir = "img"
    savefile = "reward_pendulum_ddpg_tensorflow.png"
    make_graph(r_totals, savedir=savedir, savefile=savefile)


    o = env.reset()
    done = False
    r_total = 0
    frames = []
    while not done:
        frames.append(env.render(mode="rgb_array"))
        s = np.reshape(o, (1, -1))
        a = agent.get_action(s)
        o, r, done, _ = env.step(a[0])
        r_total += r
    else:
        print("Total Rewards: {}".format(r_total))
    
    savedir = "movie"
    savefile = "movie_pendulum_ddpg_tensorflow.mp4"
    make_movie(frames, savedir=savedir, savefile=savefile)

    savedir = "model"
    savefile = "model_pendulum_ddpg_actor_net_tensorflow.h5"
    save_model(agent.actor_net, savedir=savedir, savefile=savefile)
    savefile = "model_pendulum_ddpg_critic_net_tensorflow.h5"
    save_model(agent.critic_net, savedir=savedir, savefile=savefile)

    # agent_test = DDPG(n_observation, n_actions, actor_optimizer, critic_optimizer)
    # savefile = "model_pendulum_ddpg_actor_net_tensorflow.h5"
    # agent_test.actor_net.load_weights(os.path.join(os.path.join(os.getcwd(), savedir),savefile))
    # savefile = "model_pendulum_ddpg_critic_net_tensorflow.h5"
    # agent_test.critic_net.load_weights(os.path.join(os.path.join(os.getcwd(), savedir),savefile))