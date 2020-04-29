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


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


def make_graph(steps, savedir="img", savefile="results_cart_pole.png"):
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


def make_movie(frames, savedir="movie", savefile="movie_cart_pole.mp4"):
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


def save_model(model, savedir="model", savefile="model_cart_pole.h5"):
    path = os.path.join(os.getcwd(), savedir)
    if not os.path.exists(path):
        os.mkdir(path)
    path = os.path.join(path, savefile)
    model.save(path)


class ActorCriticAgent():

    def __init__(self, n_observation, n_actions, optimizer):
        self.n_observation = n_observation
        self.n_actions = n_actions
        self.make_model()
        self.set_updater(optimizer)
        print(self.model.summary())
    
    def make_model(self):
        normal = K.initializers.glorot_normal()
        model = K.Sequential()
        model.add(K.layers.Dense(10, input_shape=(self.n_observation, ), kernel_initializer=normal, activation="relu"))
        model.add(K.layers.Dense(10, kernel_initializer=normal, activation="relu"))

        actor_layer = K.layers.Dense(self.n_actions, kernel_initializer=normal)
        action_evals = actor_layer(model.output)
        actions = SampleLayer()(action_evals)

        critic_layer = K.layers.Dense(1, kernel_initializer=normal)
        values = critic_layer(model.output)

        self.model = K.Model(inputs=model.input, outputs=[actions, action_evals, values])

    def set_updater(self, optimizer, value_loss_weight=1.0, entropy_weight=0.1):
        actions = tf.compat.v1.placeholder(shape=(None), dtype="int32")
        values = tf.compat.v1.placeholder(shape=(None), dtype="float32")

        _, action_evals, estimateds = self.model.output

        neg_logs = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=action_evals, labels=actions
        )
        advantages = values - tf.stop_gradient(estimateds)

        policy_loss = tf.reduce_mean(neg_logs * advantages)
        value_loss = tf.keras.losses.MeanSquaredError()(values, estimateds)
        action_entropy = tf.reduce_mean(self.categorical_entropy(action_evals))

        loss = policy_loss + value_loss_weight * value_loss
        loss -= entropy_weight * action_entropy

        updates = optimizer.get_updates(loss=loss, params=self.model.trainable_weights)

        self._updater = K.backend.function(
            inputs=[self.model.input, actions, values],
            outputs=[loss, policy_loss, value_loss, 
                        tf.reduce_mean(neg_logs), tf.reduce_mean(advantages), action_entropy],
            updates=updates
        )
    
    def categorical_entropy(self, logits):
        a0 = logits - tf.reduce_max(logits, axis=-1, keepdims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (tf.math.log(z0) - a0), axis=-1)

    def policy(self, state):
        actions, action_evals, values = self.model.predict(state)
        return actions[0]
    
    def estimate(self, state):
        actions, action_evals, values = self.model.predict(state)
        return values[0][0]
    
    def update(self, states, actions, rewards):
        return self._updater([states, actions, rewards])


class SampleLayer(K.layers.Layer):

    def __init__(self, **kwargs):
        self.output_dim = 1
        super(SampleLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        super(SampleLayer, self).build(input_shape)
    
    def call(self, x):
        noise = tf.random.uniform(tf.shape(x))
        return tf.argmax(x - tf.math.log(-tf.math.log(noise)), axis=1)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


if __name__ == "__main__":
    BUFFER_SIZE = 256
    BATCH_SIZE = 1
    MAXLEN = 10
    EPISODE = 1000
    LEARNING_RATE = 8e-4
    GAMMA = 0.99

    env = gym.make("CartPole-v0").unwrapped
    n_observation = env.observation_space.shape[0]
    n_actions = env.action_space.n

    optimizer = K.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=5.0)
    agent = ActorCriticAgent(n_observation, n_actions, optimizer)

    steps = []
    memory = deque(maxlen=BUFFER_SIZE)
    successes = deque(maxlen=MAXLEN)
    for e in range(EPISODE):
        o = env.reset()
        done = False
        step = 0

        while not done:
            # env.render()
            step += 1
            s = np.reshape(o, (1, -1))
            a = agent.policy(s)

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
            memory.append(Transition(s, a, s_next, r))
            o = o_next
            
            if len(memory) < BATCH_SIZE:
                continue

            batch = Transition(*zip(*memory))
            batch_states = np.vstack(batch.state)
            batch_actions = np.hstack(batch.action)
            batch_next_states = batch.next_state
            batch_rewards = np.array(batch.reward)

            values = []
            future = batch_rewards[-1] if batch_next_states[-1] is None else agent.estimate(batch_next_states[-1])
            for next_state, reward in zip(reversed(batch_next_states), reversed(batch_rewards)):
                value = reward
                if next_state is not None:
                    value += GAMMA * future
                values.append(value)
                future = value
            values = np.array(list(reversed(values)))
            values = np.reshape(values, (-1, 1))

            loss, lp, lv, p_ng, p_ad, p_en = agent.update(batch_states, batch_actions, values)

            memory.clear()
        else:
            steps.append(step)
            print("Episode: {}, Total Step: {}".format(e, step))

    
    savedir = "img"
    savefile = "result_cart_pole_a2c_tensorflow.png"
    make_graph(steps, savedir=savedir, savefile=savefile)


    o = env.reset()
    done = False
    step = 0
    frames = []
    while not done:
        step += 1
        frames.append(env.render(mode="rgb_array"))
        s = np.reshape(o, (1, -1))
        a = agent.policy(s)
        o_next, _, done, _ = env.step(a)
        o = o_next
        if step >= 1000:
            break
    else:
        print("Total Step: {}".format(step))
    savedir = "movie"
    savefile = "movie_cart_pole_a2c_tensorflow.mp4"
    make_movie(frames, savedir=savedir, savefile=savefile)
    env.close()


    savedir = "model"
    savefile = "model_cart_pole_a2c_tensorflow.h5"
    save_model(agent.model, savedir=savedir, savefile=savefile)
    
    env.close()