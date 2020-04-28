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


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class PolicyGradient(nn.Module):

    def __init__(self, input_shape, n_actions):
        super(PolicyGradient, self).__init__()
        self.fc1 = nn.Linear(input_shape, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, n_actions)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x


def make_graph(steps, savedir="movie", savefile="movie_cartpole.dqn.mp4"):
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


if __name__ == "__main__":
    EPISODE = 400
    GAMMA = 0.99
    LEARNING_RATE = 0.01
    MAXLEN = 20

    env = gym.make("CartPole-v0").unwrapped
    n_observation = env.observation_space.shape[0]
    n_actions = env.action_space.n

    network = PolicyGradient(n_observation, n_actions).to(device)

    optimizer = optim.Adam(network.parameters(), lr=LEARNING_RATE)

    successes = deque(maxlen=MAXLEN)
    steps = []
    for e in range(EPISODE):
        o = env.reset()
        done = False
        step = 0
        memory = deque()
        while not done:
            step += 1

            s = torch.from_numpy(o).type(torch.FloatTensor)
            s = torch.unsqueeze(s, 0).to(device)

            network.eval()
            with torch.no_grad():
                probs = np.array(network.forward(s).view(-1))
                a = np.random.choice([0, 1], p=probs)
                a = torch.LongTensor([[a]])
            
            o_next, _, done, _ = env.step(a.item())

            if step >= 200:
                done = True
            
            if done:
                s_next = None
                if step < 200:
                    r = torch.FloatTensor([-1.0]).to(device)
                    successes.append(0)
                else:
                    r = torch.FloatTensor([1.0]).to(device)
                    successes.append(1)
            else:
                s_next = torch.from_numpy(o_next).type(torch.FloatTensor)
                s_next = torch.unsqueeze(s_next, 0).to(device)
                r = torch.FloatTensor([0.0]).to(device)
            
            memory.append(Transition(s, a, s_next, r))
            
            o = o_next
        else:
            steps.append(step)
            print("Episode: {}, Step: {}".format(e+1, step))

            network.eval()
            transitions = Transition(*zip(*memory))
            states = torch.cat(transitions.state)
            actions = torch.cat(transitions.action)
            rewards = torch.cat(transitions.reward)
            rewards_mean = rewards.mean()
            rewards_std = rewards.std()
            rewards = (rewards - rewards_mean) / rewards_std
            
            g_t = torch.zeros(step)
            for t in range(step):
                g = torch.FloatTensor([r * (GAMMA**i) for i, r in enumerate(rewards[t:])])
                g_t[t] = torch.sum(g)
            g_t = torch.unsqueeze(g_t, 1).to(device)

            pi = network.forward(states).gather(1, actions)
            
            network.train()
            loss = -torch.mean(torch.log(pi) * g_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if sum(successes) == MAXLEN:
            print("{} times success !".format(MAXLEN))
            break


    savedir = 'img'
    savefile = 'result_cart_pole_policy_gradient_pytorch.png'
    make_graph(steps, savedir=savedir, savefile=savefile)


    o = env.reset()
    done = False
    step = 0
    frames = []
    while not done:
        step += 1

        frames.append(env.render(mode="rgb_array"))

        s = torch.from_numpy(o).type(torch.FloatTensor)
        s = torch.unsqueeze(s, 0)

        with torch.no_grad():
            probs = np.array(network.forward(s).view(-1))
            a = np.random.choice([0, 1], p=probs)
        
        o_next, _, done, _ = env.step(a)

        o = o_next
    else:
        print("Total Step: {}".format(step))
        savedir = "movie"
        savefile = "movie_cart_pole_policy_gradient_pytorch.mp4"
        make_movie(frames, savedir=savedir, savefile=savefile)
    
    env.close()