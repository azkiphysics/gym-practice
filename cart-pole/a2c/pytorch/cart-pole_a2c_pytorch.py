import os
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

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))


def make_graph(steps, savedir="movie", savefile="results_cart_pole.png"):
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


def save_model(model, savedir="model", savefile="model_cart_pole.pth"):
    path = os.path.join(os.getcwd(), savedir)
    if not os.path.exists(path):
        os.mkdir(path)
    path = os.path.join(path, savefile)
    torch.save(model.state_dict(), path)


class ActorCriticAgent(nn.Module):

    def __init__(self, input_shape, n_actions):
        super(ActorCriticAgent, self).__init__()
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.fc1 = nn.Linear(input_shape, 10)
        self.fc2 = nn.Linear(10, 10)
        self.actor = nn.Linear(10, n_actions)
        self.critic = nn.Linear(10, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.actor(x)
        v = self.critic(x)
        return q, v
    
    def policy(self, x):
        q, _ = self.forward(x)
        a_probs = F.softmax(q, dim=1)
        a = a_probs.multinomial(num_samples=1)
        return a


if __name__ == "__main__":
    EPISODE = 2000
    BUFFER_SIZE = 1024
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-2
    GAMMA = 0.99

    value_loss_coef = 0.5
    entropy_coef = 0.01
    max_grad_norm = 0.5

    env = gym.make("CartPole-v0").unwrapped
    n_observation = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = ActorCriticAgent(n_observation, n_actions)
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE)

    steps = []
    experiences = deque(maxlen=BUFFER_SIZE)
    for e in range(EPISODE):
        o = env.reset()
        done = False
        step = 0
        while not done:
            # env.render()

            step += 1
            s = torch.from_numpy(o).type(torch.FloatTensor)
            s = torch.unsqueeze(s, 0).to(device)

            with torch.no_grad():
                a = agent.policy(s)

            o_next, _, done, _ = env.step(a.item())

            if step >= 200:
                done = True
            
            if done:
                s_next = None
                if step < 200:
                    r = torch.FloatTensor([[-1.0]])
                else:
                    r = torch.FloatTensor([[1.0]])
            else:
                s_next = torch.from_numpy(o_next).type(torch.FloatTensor)
                s_next = torch.unsqueeze(s_next, 0).to(device)
                r = torch.FloatTensor([[0.0]])
            
            experiences.append(Transition(s, a, s_next, r, done))
            o = o_next
            
            if len(experiences) < BATCH_SIZE:
                continue

            batch = list(experiences)
            batch_s = torch.cat([val.state for val in batch]).to(device)
            batch_a = torch.cat([val.action for val in batch]).to(device)

            q_values = []
            last = batch[-1]
            future = last.reward if last.done else agent.forward(last.next_state)[1]
            for val in reversed(batch):
                value = val.reward
                if not val.done:
                    value += GAMMA * future
                q_values.append(value)
                future = value
            q_values = torch.FloatTensor(list(reversed(q_values)))
            q_values = torch.unsqueeze(q_values, 1).to(device)

            q_values_, v_values = agent.forward(batch_s)
            ad_values = q_values - v_values
            log_probs = F.log_softmax(q_values_, dim=1)
            a_log_probs = log_probs.gather(1, batch_a)
            policy_loss = (-a_log_probs * ad_values.detach()).mean()

            value_loss = ad_values.pow(2).mean()

            probs = F.softmax(q_values_, dim=1)
            entropy = (log_probs * probs).sum(dim=1).mean()

            loss = policy_loss + value_loss_coef * value_loss + entropy_coef * entropy
            agent.train()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
            optimizer.step()
            
            experiences.clear()
        else:
            steps.append(step)
            print("Episode: {}, Total Step: {}".format(e+1, step))
    
    savedir = 'img'
    savefile = 'result_cart_pole_a2c_pytorch.png'
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
            a = agent.policy(s)
        
        o_next, _, done, _ = env.step(a.item())

        o = o_next
    else:
        print("Total Step: {}".format(step))
        savedir = "movie"
        savefile = "movie_cart_a2c_gradient_pytorch.mp4"
        make_movie(frames, savedir=savedir, savefile=savefile)


    savedir = "model"
    savefile = "model_cart_pole_a2c_pytorch.pth"
    save_model(agent, savedir=savedir, savefile=savefile)

    env.close()