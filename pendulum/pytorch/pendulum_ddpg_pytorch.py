import os
import random
from collections import deque, namedtuple

import cv2
import gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',('s', 'a', 's_next', 'r', 'done'))


def make_graph(rewards, savedir="img", savefile="graph.png"):
    pass

def make_movie(frames, savedir="movie", savefile="movie.mp4"):
    pass

def save_model(agent, savedir="model", savefile="model.pth"):
    pass

def ornstein_uhlenbeck(x, theta=0.15, mu=0, sigma=0.2, clip_min=-2.0, clip_max=2.0):
    x_next = x + theta * (mu - x) + sigma * np.random.normal()
    return np.clip(x_next, clip_min, clip_max)


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


class DDPG_Actor(nn.Module):

    def __init__(self, n_observation, n_actions):
        super(DDPG_Actor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(n_observation, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, n_actions),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.net(x)


class DDPG_Critic(nn.Module):

    def __init__(self, n_observation, n_actions):
        super(DDPG_Critic, self).__init__()

        self.obs_net = nn.Sequential(
            nn.Linear(n_observation, 400),
            nn.ReLU()
        )

        self.out_net = nn.Sequential(
            nn.Linear(400 + n_actions, 300),
            nn.ReLU(),
            nn.Linear(300, 1)
        )

    def forward(self, x, a):
        obs = self.obs_net(x)
        return self.out_net(torch.cat([obs, a], dim=1))


if __name__ == "__main__":
    CAPACITY = 10000
    BATCH_SIZE = 64
    EPISODE = 10000
    GAMMA = 0.99
    ACTOR_LEARNING_RATE = 1e-4
    CRITIC_LEARNING_RATE = 1e-4

    env = gym.make("Pendulum-v0")
    n_observation = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    actor_net = DDPG_Actor(n_observation, n_actions).to(device)
    critic_net = DDPG_Critic(n_observation, n_actions).to(device)

    actor_optim = optim.Adam(actor_net.parameters(), lr=ACTOR_LEARNING_RATE)
    critic_optim = optim.Adam(critic_net.parameters(), lr=CRITIC_LEARNING_RATE)

    memory = ReplayMemory(CAPACITY)

    r_totals = []
    for e in range(EPISODE):
        o = env.reset()
        done = False
        r_total = 0
        step = 0
        while not done:
            step += 1
            s = torch.from_numpy(o).type(torch.FloatTensor)
            s = torch.unsqueeze(s, 0).to(device)
            actor_net.eval()
            with torch.no_grad():
                a = actor_net.forward(s)
                a = torch.FloatTensor([[ornstein_uhlenbeck(a.item())]]).to(device)
            o_next, r, done, _ = env.step(np.array([a.item()]))
            r_total += r
            r = torch.FloatTensor([[r]]).to(device)

            s_next = torch.from_numpy(o_next).type(torch.FloatTensor)
            s_next = torch.unsqueeze(s_next, 0).to(device)
            
            memory.push(Transition(s, a, s_next, r, done))
            o = o_next
            
            if len(memory) < BATCH_SIZE:
                continue
            
            transitions = memory.sample(BATCH_SIZE)
            batch = Transition(*zip(*transitions))
            batch_non_final_mask = torch.BoolTensor(batch.done)
            batch_s = torch.cat(batch.s)
            batch_a = torch.cat(batch.a)
            batch_s_next = torch.cat(batch.s_next)
            batch_r = torch.cat(batch.r)

            critic_net.train()
            q = critic_net(batch_s, batch_a)
            q_next = critic_net.forward(batch_s_next, actor_net.forward(batch_s_next))
            q_next[batch_non_final_mask] = 0
            q_ref = batch_r + GAMMA * q_next
            critic_loss = F.mse_loss(q, q_ref.detach())
            critic_optim.zero_grad()
            critic_loss.backward()
            critic_optim.step()

            actor_net.train()
            q = critic_net(batch_s, batch_a)
            actor_loss = -q.mean()
            actor_optim.zero_grad()
            actor_loss.backward()
            actor_optim.step()
            

        else:
            r_totals.append(r_total)
            print("Episode: {}, Total Reward: {}".format(e, r_total))

    savedir = "img"
    savefile = "reward_pendulum_ddpg_pytorch.png"
    make_graph(r_totals, savedir=savedir, savefile=savefile)

    o = env.reset()
    done = False
    r_total = 0
    frames = []
    while not done:
        frames.append(env.render(mode="rgb_array"))
        s = torch.from_numpy(o).type(torch.FloatTensor)
        s = torch.unsqueeze(s, 0).to(device)
        with torch.no_grad():
            a = actor_net.forward(s)
            a = torch.FloatTensor([[ornstein_uhlenbeck(a.item())]]).to(device)
        o_next, r, done, _ = env.step(np.array([a_with_noise.item()]))
        r_total += r
        o = o_next
    else:
        print("Total Reward: {}".format(r_total))
    
    savedir = "movie"
    savefile = "movie_pendulum_ddpg_pytorch.mp4"
    make_movie(frames, savedir=savedir, savefile=savefile)

    savedir = "model"
    savefile = "model_pendulum_ddpg_actor_net_pytorch.pth"
    save_model(actor_net, savedir=savedir, savefile=savefile)
    savefile = "model_pendulum_ddpg_critic_net_pytorch.pth"
    save_model(critic_net, savedir=savedir, savefile=savefile)

    env.close()