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

def save_model(model, savedir="model", savefile="model.pth"):
    path = os.path.join(os.getcwd(), savedir)
    if not os.path.exists(path):
        os.mkdir(path)
    path = os.path.join(path, savefile)
    torch.save(model.state_dict(), path)

def ornstein_uhlenbeck(x, theta=0.15, mu=0, sigma=0.2, dt=1e-2):
    x_next = x + theta * (mu - x) * dt + sigma * np.sqrt(dt) * np.random.normal()
    return x_next

def soft_update(target, source, tau=0.001):
    for target_key, source_key in zip(target.state_dict(), source.state_dict()):
        target.state_dict()[target_key] = target.state_dict()[target_key] * (1 - tau) + source.state_dict()[source_key] * tau


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
    EPISODE = 100
    GAMMA = 0.99
    ACTOR_LEARNING_RATE = 1e-4
    CRITIC_LEARNING_RATE = 1e-3

    tau = 0.001

    env = gym.make("Pendulum-v0")
    n_observation = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    actor_net = DDPG_Actor(n_observation, n_actions).to(device)
    actor_net_target = DDPG_Actor(n_observation, n_actions).to(device)
    critic_net = DDPG_Critic(n_observation, n_actions).to(device)
    critic_net_target = DDPG_Critic(n_observation, n_actions).to(device)

    soft_update(actor_net_target, actor_net, tau=1.0)
    soft_update(critic_net_target, critic_net, tau=1.0)

    actor_optim = optim.Adam(actor_net.parameters(), lr=ACTOR_LEARNING_RATE)
    critic_optim = optim.Adam(critic_net.parameters(), lr=CRITIC_LEARNING_RATE)

    memory = ReplayMemory(CAPACITY)

    r_totals = []
    for e in range(EPISODE):
        o = env.reset()
        done = False
        r_total = 0
        step = 0
        noise_prev = torch.FloatTensor([[0.0]])
        while not done:
            step += 1
            s = torch.from_numpy(o).type(torch.FloatTensor)
            s = torch.unsqueeze(s, 0).to(device)

            actor_net.eval()
            with torch.no_grad():
                a = actor_net.forward(s)
                # noise = torch.FloatTensor([[ornstein_uhlenbeck(noise_prev.item())]]).to(device)
                noise = torch.FloatTensor([[np.random.randn() * 0.01]]).to(device)
                a += noise
                a = torch.clamp(a, -1, 1).to(device)
                noise_prev = noise

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
            batch_done = torch.BoolTensor(batch.done)
            batch_s = torch.cat(batch.s)
            batch_a = torch.cat(batch.a)
            batch_s_next = torch.cat(batch.s_next)
            batch_r = torch.cat(batch.r)

            actor_net.eval()
            critic_net.train()
            critic_optim.zero_grad()
            q = critic_net(batch_s, batch_a)
            q_next = critic_net_target.forward(batch_s_next, actor_net_target.forward(batch_s_next))
            q_next[batch_done] = 0.0
            q_ref = batch_r + GAMMA * q_next
            critic_loss = F.mse_loss(q, q_ref.detach())
            critic_loss.backward()
            critic_optim.step()

            actor_net.train()
            critic_net.eval()
            actor_optim.zero_grad()
            batch_cur_a = actor_net.forward(batch_s)
            q = critic_net.forward(batch_s, batch_cur_a).to(device)
            actor_loss = -q.mean()
            actor_loss.backward()
            actor_optim.step()
            
            soft_update(actor_net_target, actor_net, tau=tau)
            soft_update(critic_net_target, critic_net, tau=tau)

        else:
            r_totals.append(r_total)
            print("Episode: {}, Total Reward: {}".format(e+1, r_total))

    savedir = "img"
    savefile = "reward_pendulum_ddpg_pytorch.png"
    make_graph(r_totals, savedir=savedir, savefile=savefile)

    o = env.reset()
    done = False
    r_total = 0
    frames = []
    noise_prev = torch.FloatTensor([[0.0]])
    while not done:
        frames.append(env.render(mode="rgb_array"))
        s = torch.from_numpy(o).type(torch.FloatTensor)
        s = torch.unsqueeze(s, 0).to(device)
        with torch.no_grad():
            a = actor_net.forward(s)
            # noise = torch.FloatTensor([[ornstein_uhlenbeck(noise_prev.item())]]).to(device)
            noise = torch.FloatTensor([[np.random.randn() * 0.01]]).to(device)
            a += noise
            a = torch.clamp(a, -1, 1).to(device)
            noise_prev = noise
        o_next, r, done, _ = env.step(np.array([a.item()]))
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