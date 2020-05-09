import os
import random
from collections import namedtuple, deque

import cv2
import gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


Transition = namedtuple('Transition', ('s', 'a', 'r', 'done'))


class MemoryBuffer():

    def __init__(self, size):
        self.size = size
        self.memory = deque(maxlen=size)

    def push(self, *args):
        self.memory.append(*args)

    def clear(self):
        self.memory.clear()
    
    def get(self):
        return self.memory
    
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
    
    def render(self):
        return self._env.render(mode="rgb_array")
    
    def reset(self):
        return self.transform(self._env.reset())
    
    def step(self, a):
        o_next, r, done, info = self._env.step(a[0])
        return self.transform(o_next), r, done, info
    
    def transform(self, x):
        x = torch.from_numpy(x).type(torch.FloatTensor)
        return torch.unsqueeze(x, 0).to(device)


class Actor(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(n_observations, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, n_actions)
        self.logstd = nn.Parameter(torch.zeros(n_actions))
    
    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        return F.tanh(self.out(x))


class Critic(nn.Module):

    def __init__(self, n_observations):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(n_observations, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)


def calc_adv(critic, traj_s_v, traj_r, traj_done, gamma, lam, device):
    values_v = critic(traj_s_v)
    values = values_v.squeeze().data.cpu().numpy()

    last_gae = 0.0
    result_adv = []
    result_ref = []
    for val, val_next, r, done in zip(reversed(values[:-1]), reversed(values[1:]), reversed(traj_r[:-1]), reversed(traj_done[:-1])):
        if done:
            delta = r - val
            last_gae = delta
        else:
            delta = r + gamma * val_next - val
            last_gae = delta + (gamma*lam) * last_gae
        result_adv.append(last_gae)
        result_ref.append(delta + val)
    adv_v = torch.FloatTensor(list(reversed(result_adv))).unsqueeze(1)
    ref_v = torch.FloatTensor(list(reversed(result_ref))).unsqueeze(1)
    return adv_v.to(device), ref_v.to(device)


def calc_logprob(mu_v, logstd_v, traj_a_v):
    return -((traj_a_v - mu_v)**2) / (2 * torch.exp(logstd_v).clamp(min=1e-3)) - torch.log(torch.sqrt(2 * np.pi * torch.exp(logstd_v)))



if __name__ == "__main__":
    LR_ACTOR = 1e-9
    LR_CRITIC = 1e-8
    EPISODE = 10000
    TRAJECTORY_SIZE = 1024
    EPOCH_SIZE = 5
    BATCH_SIZE = 64
    GAMMA = 0.99
    LAMMDA=0.95
    EPS = 0.2

    env = Observer(gym.make("Pendulum-v0"))
    n_observations = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    actor = Actor(n_observations, n_actions).to(device)
    critic = Critic(n_observations).to(device)

    optim_actor = optim.Adam(actor.parameters(), lr=LR_ACTOR)
    optim_critic = optim.Adam(critic.parameters(), lr=LR_CRITIC)

    buffer = MemoryBuffer(TRAJECTORY_SIZE)

    r_totals = []
    for e in range(EPISODE):
        s = env.reset()
        done = False
        r_total = 0
        while not done:
            # env.render()
            mu = actor.forward(s)
            with torch.no_grad():
                a = mu + torch.exp(actor.logstd) * np.random.normal()
                a = torch.clamp(a, -1.0, 1.0).to(device)
            s_next, r, done, _ = env.step(a.detach())
            v = critic(s)

            buffer.push(Transition(s, a, r, done))

            r_total += r
            s = s_next

            if len(buffer) < TRAJECTORY_SIZE:
                continue

            trajs = Transition(*zip(*buffer.get()))
            traj_s_v = torch.cat(trajs.s).to(device)
            traj_a_v = torch.cat(trajs.a).to(device)
            traj_r = trajs.r
            traj_done = trajs.done
            
            traj_adv_v, traj_ref_v = calc_adv(critic, traj_s_v, traj_r, traj_done, GAMMA, LAMMDA, device)
            mu_v = actor(traj_s_v)
            old_logprob_v = calc_logprob(mu_v, actor.logstd, traj_a_v).to(device)
            
            traj_adv_v = traj_adv_v - torch.mean(traj_adv_v)
            traj_adv_v /= torch.std(traj_adv_v)

            traj_s_v = traj_s_v[:-1]
            traj_a_v = traj_a_v[:-1]
            old_logprob_v = old_logprob_v[:-1].detach()

            for epoch in range(EPOCH_SIZE):
                for batch_ofs in range(0, TRAJECTORY_SIZE-1, BATCH_SIZE):
                    batch_l = batch_ofs + BATCH_SIZE
                    batch_s_v = traj_s_v[batch_ofs:batch_l]
                    batch_a_v = traj_a_v[batch_ofs:batch_l]
                    batch_adv_v = traj_adv_v[batch_ofs:batch_l]
                    batch_ref_v = traj_ref_v[batch_ofs:batch_l]
                    batch_logprob_v = old_logprob_v[batch_ofs:batch_l]

                    optim_critic.zero_grad()
                    value_v = critic(batch_s_v)
                    loss_critic_v = F.mse_loss(value_v, batch_ref_v.detach())
                    loss_critic_v.backward()
                    optim_critic.step()

                    optim_actor.zero_grad()
                    batch_mu_v = actor(batch_s_v)
                    logprob_pi_v = calc_logprob(batch_mu_v, actor.logstd, batch_a_v)
                    ratio_v = torch.exp(logprob_pi_v - batch_logprob_v)
                    surr_obj_v = batch_adv_v * ratio_v
                    c_ratio_v = torch.clamp(ratio_v, 1.0 - EPS, 1.0 + EPS)
                    clipped_surr_obj_v = batch_adv_v * c_ratio_v
                    loss_actor_v = -torch.min(surr_obj_v, clipped_surr_obj_v).mean()
                    loss_actor_v.backward()
                    optim_actor.step()
            print(loss_critic_v)
            buffer.clear()
        else:
            print("Episode: {}, Total Reward: {}".format(e, r_total))