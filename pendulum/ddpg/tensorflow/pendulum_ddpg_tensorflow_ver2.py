import os
import random
from collections import deque, namedtuple

import cv2
import gym
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import layers

Transition = namedtuple('Transition', ('s', 'a', 's_next', 'r', 'done'))

class ReplayMemoryBuffer():

    def __init__(self, buffer_size):
        self.memory = deque(maxlen=buffer_size)
    
    def push(self, *args):
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)


class PendulumObserver():

    def __init__(self, env):
        self._env = env
    
    @property
    def action_space(self):
        return self._env.action_space

    @property
    def obsevation_space(self):
        return self._env.observation_space
    
    def reset(self):
        return self.transform(self._env.reset())
    
    def step(self, a):
        o_next, r, done, info = self._env.step(a[0])
        return self.transform(o_next), np.array([[r]]), np.array([[done]]), info
    
    def transform(self, x):
        return np.reshape(x, (1, -1)).astype(np.float32)
    
    def render(self, mode="rgb_array"):
        return self._env.render(mode=mode)
    
    def close(self):
        self._env.close()


class Actor(K.Model):

    def __init__(self, n_actions):
        super(Actor, self).__init__()
        self.fc1 = layers.Dense(400)
        self.fc2 = layers.Dense(300)
        self.out = layers.Dense(n_actions)
    
    def call(self, inputs):
        x = tf.nn.relu(self.fc1(inputs))
        x = tf.nn.relu(self.fc2(x))
        return K.activations.tanh(self.out(x))


class Critic(K.Model):

    def __init__(self):
        super(Critic, self).__init__()
        self.s_block = layers.Dense(400)
        self.q_block = layers.Dense(300)
        self.out = layers.Dense(1)
    
    def call(self, inputs):
        s, a = inputs
        s_output = tf.nn.relu(self.s_block(s))
        q_input = tf.concat([s_output, a], 1)
        x = tf.nn.relu(self.q_block(q_input))
        return self.out(x)


class DDPGAgent():

    def __init__(self, n_actions):
        self.actor = Actor(n_actions)
        self.actor_target = Actor(n_actions)
        self.actor_target.set_weights(np.array(self.actor.get_weights()))
        self.actor_target.trainable = False

        self.critic = Critic()
        self.critic_target = Critic()
        self.critic_target.set_weights(np.array(self.critic.get_weights()))
        self.critic_target.trainable = False
    
    def policy(self, s, is_noise=True, noise_scale=0.1, clip_min=-1.0, clip_max=1.0):
        mu = self.actor(s)

        if is_noise:
            noise = np.random.randn(*mu.shape).astype(np.float32)
        else:
            noise = np.zeros(mu.shape).astype(np.float32)

        a = mu + noise_scale * noise

        return np.clip(a, clip_min, clip_max)
    
    def target_policy(self, s):
        return self.actor_target(s)
    
    def target_evaluate(self, s, a):
        return self.critic_target([s, a])
    
    def update(self, states, actions, estimateds, actor_optimizer, critic_optimizer, tau):
        def actor_loss_func(critic, states, mus):
            qs = critic([states, mus])
            return -tf.reduce_mean(qs)
        
        def critic_loss_func(expecteds, estimateds):
            return tf.reduce_mean((expecteds - tf.stop_gradient(estimateds))**2)

        with tf.GradientTape() as critic_tape, tf.GradientTape() as actor_tape:
            expecteds = self.critic([states, actions])
            mus = self.actor(states)
            
            critic_loss = critic_loss_func(expecteds, estimateds)
            actor_loss = actor_loss_func(self.critic, states, mus)
        
        critic_grads = critic_tape.gradient(critic_loss, self.critic.trainable_variables)
        actor_grads = actor_tape.gradient(actor_loss, self.actor.trainable_variables)

        critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
        actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        actor_weights = np.array(self.actor.get_weights())
        actor_target_weights = np.array(self.actor_target.get_weights())
        self.actor_target.set_weights(tau*actor_weights + (1-tau)*actor_target_weights)

        critic_weights = np.array(self.critic.get_weights())
        critic_target_weights = np.array(self.critic_target.get_weights())
        self.critic_target.set_weights(tau*critic_weights + (1-tau)*critic_target_weights)


class DDPGTrainer():

    def __init__(self, buffer_size=10000, batch_size=64, gamma=0.99, tau=0.001):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
    
    def train(self, env, episode=200):
        n_actions = env.action_space.shape[0]
        agent = DDPGAgent(n_actions)
        r_totals = self.train_loop(env, agent, episode)

        return agent, r_totals
    
    def train_loop(self, env, agent, episode):
        experiences = ReplayMemoryBuffer(self.buffer_size)
        r_totals = []
        for e in range(episode):
            s = env.reset()
            done = False
            r_total = 0
            while not done:
                a = agent.policy(s)
                s_next, r, done, _ = env.step(a)
                experiences.push(s, a, s_next, r, done)
                s = s_next
                r_total += r[0,0]

                if len(experiences) < self.batch_size:
                    continue

                self.train_step(agent, experiences, tau=self.tau)
            else:
                r_totals.append(r_total)
                print("Episode: {}, Total Rewards: {}".format(e+1, r_total))

        return r_totals
    
    def train_step(self, agent, experiences, tau):
        actor_optimizer = K.optimizers.Adam(learning_rate=1e-4)
        critic_optimizer = K.optimizers.Adam(learning_rate=1e-3)

        transitions = experiences.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        batch_s = np.vstack(batch.s)
        batch_a = np.vstack(batch.a)
        batch_r = np.vstack(batch.r)
        batch_s_next = np.vstack(batch.s_next)
        batch_not_done = 1.0 - np.vstack(batch.done)

        batch_mu_next = agent.target_policy(batch_s_next)
        q_next = agent.target_evaluate(batch_s_next, batch_mu_next)
        estimateds = batch_r + self.gamma * batch_not_done * q_next
        
        agent.update(batch_s, batch_a, estimateds, actor_optimizer, critic_optimizer, tau)


class DDPGLogger():

    def __init__(self, dirnames=["img", "movie", "model"]):
        self.dirnames = dirnames
        for dirname in dirnames:
            path = os.path.join(os.getcwd(), dirname)
            if not os.path.exists(path):
                os.mkdir(path)
    
    def make_graph(self, r_totals, filename="total_rewards.png"):
        path = os.path.join(os.getcwd(),self.dirnames[0])
        path = os.path.join(path, filename)

        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111)
        ax.plot(np.arange(1, len(r_totals)+1, 1), r_totals)
        ax.set_xlim(0, len(r_totals))
        plt.savefig(path, dpi=300)
    
    def make_movie(self, frames, filename="movie.mp4"):
        path = os.path.join(os.getcwd(), self.dirnames[1])
        path = os.path.join(path, filename)

        fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
        video = cv2.VideoWriter(path, fourcc, 50.0, (600, 600))

        for frame in frames:
            frame = cv2.resize(frame, (600,600))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video.write(frame)

        video.release()
    
    def save_model(self, agent, dirname_actor="model_actor", dirname_critic="model_critic"):
        path = os.path.join(os.getcwd(), self.dirnames[2])
        path_actor = os.path.join(path, dirname_actor)
        path_critic = os.path.join(path, dirname_critic)

        tf.saved_model.save(agent.actor, path_actor)
        tf.saved_model.save(agent.critic, path_critic)

def test(env, agent):
    s = env.reset()
    done = False
    frames = []
    while not done:
        frames.append(env.render(mode="rgb_array"))
        a = agent.policy(s)
        s, r, done, _ = env.step(a)
    return frames

def main():
    env = PendulumObserver(gym.make("Pendulum-v0"))
    trainer = DDPGTrainer(buffer_size=10000, batch_size=64, gamma=0.99, tau=0.001)
    agent, r_totals = trainer.train(env, episode=500)
    frames = test(env, agent)
    env.close()
    logger = DDPGLogger()
    logger.make_graph(r_totals, filename="reward_pendulum_ddpg_tensorflow_v2.png")
    logger.make_movie(frames, filename="movie_pendulum_ddpg_tensorflow_v2.mp4")
    logger.save_model(agent, dirname_actor="model_actor_pendulum_ddpg_tensorflow_v2", dirname_critic="model_critic_pendulum_ddpg_tensorflow_v2")


if __name__ == "__main__":
    main()