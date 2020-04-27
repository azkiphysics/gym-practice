import os
import random
from collections import namedtuple, deque

import cv2
from PIL import Image
import gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory():

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
    
    def push(self, *args):
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
    
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
    
    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)


resize = T.Compose([
    T.ToPILImage(),
    T.Resize(40, interpolation=Image.CUBIC),
    T.ToTensor()
])


def get_cart_location(screen_width):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)


def get_screen():
    screen = env.render(mode="rgb_array").transpose((2, 0, 1))
    _, screen_height, screen_width = screen.shape
    screen = screen[:, int(0.4*screen_height):int(0.8*screen_height), :]
    view_width = int(screen_width * 0.6)
    cart_location = get_cart_location(screen_width)

    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2, cart_location + view_width // 2)

    screen = screen[:, :, slice_range]
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255.0
    screen = torch.from_numpy(screen)
    return resize(screen).unsqueeze(0).to(device)


def save_movie(frames, savedir="movie", savefile="movie_cartpole_dqn.mp4"):
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
    CAPACITY = 10000
    EPISODE = 50000
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 10000
    BATCH_SIZE = 128
    GAMMA = 0.999
    TARGET_UPDATE = 10

    env = gym.make("CartPole-v0").unwrapped

    env.reset()
    input_shape = get_screen().shape[1:]
    n_actions = env.action_space.n

    policy_network = DQN(input_shape, n_actions).to(device)
    target_network = DQN(input_shape, n_actions).to(device)
    target_network.load_state_dict(policy_network.state_dict())
    target_network.eval()

    optimizer = optim.RMSprop(policy_network.parameters())
    memory = ReplayMemory(capacity=CAPACITY)
    
    steps = []
    successes = deque(maxlen=10)
    for e in range(EPISODE):
        epsilon = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * e / EPS_DECAY)
        env.reset()
        done = False
        step = 0
        last_screen = get_screen()
        current_screen = get_screen()
        s = current_screen - last_screen
        while not done:
            step += 1

            if np.random.rand() < epsilon:
                a = torch.LongTensor([[env.action_space.sample()]]).to(device)
            else:
                policy_network.eval()
                with torch.no_grad():
                    a = policy_network.forward(s).max(1)[1].view(1, 1)
            
            _, _, done, _ = env.step(a.item())

            if step >= 200:
                done = True

            if done:
                s_next = None
                if step < 200:
                    r = torch.LongTensor([-1.0]).to(device)
                    successes.append(0)
                else:
                    r = torch.LongTensor([1.0]).to(device)
                    successes.append(1)
            else:
                last_screen = current_screen
                current_screen = get_screen()
                s_next = current_screen - last_screen
                r = torch.LongTensor([0.0]).to(device)
            memory.push(s, a, s_next, r)

            if not done:
                s = s_next

            if len(memory) < BATCH_SIZE:
                continue
            
            policy_network.eval()
            transitions = memory.sample(BATCH_SIZE)
            batch = Transition(*zip(*transitions))
            non_final_mask = torch.BoolTensor(tuple(map(lambda s: s is not None, batch.next_state)))
            non_final_batch_s_next = torch.cat([s for s in batch.next_state if s is not None])
            batch_s = torch.cat(batch.state)
            batch_a = torch.cat(batch.action)
            batch_r = torch.cat(batch.reward)

            estimateds = policy_network.forward(batch_s).gather(1, batch_a)
            expecteds = torch.zeros(BATCH_SIZE).to(device)
            expecteds[non_final_mask] = target_network(non_final_batch_s_next).max(1)[0].detach()
            expecteds = batch_r + GAMMA * expecteds

            policy_network.train()
            loss = F.smooth_l1_loss(estimateds, expecteds.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            steps.append(step)
            print("Episode: {}, Total Step: {}".format(e+1, step))
        
        if e % TARGET_UPDATE == 0:
            target_network.load_state_dict(policy_network.state_dict())
        
        if sum(successes) == 10:
            print("10times success!")
            break


    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.plot(np.arange(1, len(steps)+1, 1), steps)
    ax.set_xlim(0, len(steps))
    ax.set_ylim(0, 210)
    savedir = "img"
    savefile = "result_cart_pole_dqn_state_image_pytorch.png"
    path = os.path.join(os.getcwd(), "img")
    if not os.path.exists(path):
        os.mkdir(path)
    path = os.path.join(path, savefile)
    plt.savefig(path, dpi=300)
    plt.show()


    o = env.reset()
    done = False
    step = 0
    frames = []
    last_screen = get_screen()
    current_screen = get_screen()
    s = current_screen - last_screen
    while not done:
        step += 1
        frames.append(env.render(mode='rgb_array'))
        policy_network.eval()
        with torch.no_grad():
            a = policy_network.forward(s).max(1)[1].view(1, 1)
        _, _, done, _ = env.step(a.item())
        last_screen = current_screen
        current_screen = get_screen()
        s = current_screen - last_screen
    else:
        print("Total Step: {}.".format(step))
        savedir="movie"
        savefile="movie_cart_pole_dqn_state_image_pytorch.mp4"
        save_movie(frames, savedir=savedir, savefile=savefile)


    env.close()
    
    