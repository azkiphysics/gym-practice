import os
import random
from collections import deque, namedtuple

import gym
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

EPISODE_SIZE = 1000
BUFFER_SIZE = 10000
TRAJECTORY_SIZE = 1024
EPOCH_SIZE = 10
BATCH_SIZE = 128
GAMMA = 0.99
LAMMDA = 0.95
EPS = 0.2
LR_ACTOR = 1e-4
LR_CRITIC = 1e-3
MAXLEN = 20

Transition = namedtuple('Transition', ('s', 'a', 'pi', 'r', 'done'))


def make_graph(steps, savedir="img", savefile="graph.png"):
    path = os.path.join(os.getcwd(), savedir)
    if not os.path.exists(path):
        os.mkdir(path)
    path = os.path.join(path, savefile)

    N = len(steps)
    batch_size = 20
    steps_mean = []
    steps_std = []
    for i in range(0, N, batch_size):
        steps_mean.append(np.mean(steps[i:i+batch_size]))
        steps_std.append(np.std(steps[i:i+batch_size]))
    steps_mean = np.array(steps_mean)
    steps_std = np.array(steps_std)

    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(111)
    ax.fill_between(np.arange(0, N, batch_size), steps_mean+steps_std, steps_mean-steps_std, color="blue", alpha=0.1)
    ax.plot(np.arange(0, N, batch_size), steps_mean, color="blue")
    ax.set_xlim(0, len(steps))
    plt.savefig(path, dpi=300)
    plt.show()


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


def save_model(model, savedir="model", savefile="model.pth"):
    path = os.path.join(os.getcwd(), savedir)
    if not os.path.exists(path):
        os.mkdir(path)
    path = os.path.join(path, savefile)
    torch.save(model.state_dict(), path)


class MemoryBuffer():

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
    def observation_space(self):
        return self._env.observation_space
    
    @property
    def action_space(self):
        return self._env.action_space
    
    def reset(self):
        return self.transform(self._env.reset())
    
    def render(self, mode="rgb_array"):
        return self._env.render(mode=mode)
    
    def step(self, a):
        o_next, r, done, info = self._env.step(a)
        return self.transform(o_next), r, done, info
    
    def close(self):
        self._env.close()
    
    def transform(self, o):
        s = torch.from_numpy(o).type(torch.FloatTensor)
        return s.unsqueeze(0).to(device)


class Actor(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(n_observations, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, n_actions)
    
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return F.softmax(self.fc3(x), dim=1)

class Critic(nn.Module):

    def __init__(self, n_observations):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(n_observations, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def calc_adv(critic, traj_s_v, traj_r_v, traj_done):
    val_v = critic(traj_s_v)

    last_gae = torch.FloatTensor([0.0])
    adv_v = []
    ref_v = []
    for r, done, v, v_next in zip(reversed(traj_r_v[:-1]), reversed(traj_done[:-1]), reversed(val_v[:-1]), reversed(val_v[1:])):
        if done:
            delta = r - v
            last_gae = delta
        else:
            delta = r + GAMMA * v_next - v
            last_gae = delta + GAMMA * LAMMDA * last_gae
        adv_v.append(last_gae)
        ref_v.append(last_gae + v)
    adv_v = torch.cat(tuple(reversed(adv_v))).unsqueeze(1)
    adv_v = adv_v.to(device)
    ref_v = torch.cat(tuple(reversed(ref_v))).unsqueeze(1)
    ref_v = ref_v.to(device)

    return adv_v.detach(), ref_v.detach()


if __name__ == "__main__":
    env = Observer(gym.make("CartPole-v0").unwrapped)
    n_observations = env.observation_space.shape[0]
    n_actions = env.action_space.n

    actor = Actor(n_observations, n_actions).to(device)
    critic = Critic(n_observations).to(device)

    optim_actor = optim.Adam(actor.parameters(), lr=LR_ACTOR)
    optim_critic = optim.Adam(critic.parameters(), lr=LR_CRITIC)

    buffer = MemoryBuffer(BUFFER_SIZE)

    steps = []
    successes = deque(maxlen=MAXLEN)
    for e in range(EPISODE_SIZE):
        s = env.reset()
        done = False
        step = 0
        while not done:
            step += 1

            actor.eval()
            with torch.no_grad():
                a_probs = actor(s)
                a = a_probs.multinomial(num_samples=1)
            s_next, r, done, _ = env.step(a.item())
            
            if step >= 200:
                done = True
            if done:
                if step < 200:
                    r = -1
                else:
                    r = 1
            else:
                r = 0
            
            buffer.push(s, a, a_probs[:,a.item()], r, done)
            
            s = s_next

            if len(buffer) < TRAJECTORY_SIZE:
                continue

            if sum(successes) == MAXLEN:
                print("{} times success!".format(successes))
                break
            
            trajs = buffer.get()
            trajs = Transition(*zip(*trajs))
            traj_s_v = torch.cat(trajs.s)
            traj_a_v = torch.cat(trajs.a)
            traj_r_v = torch.FloatTensor(trajs.r).unsqueeze(1).to(device)
            traj_done = trajs.done
            
            traj_adv_v, traj_ref_v = calc_adv(critic, traj_s_v, traj_r_v, traj_done)

            traj_adv_v = (traj_adv_v - torch.mean(traj_adv_v)) / torch.std(traj_adv_v)
            
            traj_s_v = traj_s_v[:-1]
            traj_a_v = traj_a_v[:-1]
            traj_old_pi_v = torch.cat(trajs.pi[:-1]).unsqueeze(1)
            
            actor.train()
            critic.train()
            for epoch in range(EPOCH_SIZE):
                for i_s in range(0, traj_s_v.shape[0], BATCH_SIZE):
                    i_e = i_s + BATCH_SIZE
                    batch_slice = slice(i_s, i_e)
                    batch_s_v = traj_s_v[batch_slice]
                    batch_a_v = traj_a_v[batch_slice]
                    batch_old_pi_v = traj_old_pi_v[batch_slice].detach()
                    batch_adv_v = traj_adv_v[batch_slice]
                    batch_ref_v = traj_ref_v[batch_slice]

                    optim_critic.zero_grad()
                    batch_val_v = critic(batch_s_v)
                    loss_critic = F.mse_loss(batch_val_v, batch_ref_v)
                    loss_critic.backward()
                    optim_critic.step()

                    optim_actor.zero_grad()
                    batch_pi_v = actor(batch_s_v).gather(1, batch_a_v)
                    ratio_v = batch_pi_v / batch_old_pi_v
                    surr_obj_v = ratio_v * batch_adv_v
                    c_ratio_v = torch.clamp(ratio_v, 1-EPS, 1+EPS)
                    c_surr_obj_v = c_ratio_v * batch_adv_v
                    loss_actor = -torch.min(surr_obj_v, c_surr_obj_v).mean()
                    loss_actor.backward()
                    optim_actor.step()
            buffer.clear()
        else:
            steps.append(step)
            print("Episode: {}, Total Step: {}".format(e+1, step))
    steps = np.array(steps)

    dirname = "img"
    filename = "cart_pole_pendulum_pytorch.png"
    make_graph(steps, savedir=dirname, savefile=filename)


    s = env.reset()
    done = False
    step = 0
    frames = []
    while not done:
        step += 1
        frames.append(env.render())
        with torch.no_grad():
            a_probs = actor(s)
            a = a_probs.multinomial(num_samples=1)
        s_next, r, done, _ = env.step(a.item())
        s = s_next

        if step > 1000:
            break
    else:
        print("Test:\nTotal Step: {}".format(step))

    dirname = "movie"
    filename = "cart_pole_pendulum_pytorch.mp4"
    make_movie(frames, savedir=dirname, savefile=filename)


    dirname = "model"
    filename = "cart_pole_pendulum_pytorch_actor.pth"
    save_model(actor, savedir=dirname, savefile=filename)
    filename = "cart_pole_pendulum_pytorch_critic.pth"
    save_model(critic, savedir=dirname, savefile=filename)

    env.close()