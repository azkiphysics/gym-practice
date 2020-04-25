from collections import namedtuple
import random
import time

import gym
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory():

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
    
    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
        super(DQN, self).__init__()
        self.hidden1 = nn.Linear(input_size, hidden1_size)
        self.hidden2 = nn.Linear(hidden1_size, hidden2_size)
        self.output = nn.Linear(hidden2_size, output_size)
    
    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.output(x)
        return x


if __name__ == "__main__":
    EPISODE = 200
    BATCH_SIZE = 32
    CAPACITY = 10000
    GAMMA = 0.99
    EPS_START = 0.9
    EPS_END = 0.005
    EPS_DECAY = 50
    TARGET_UPDATE = 10

    memory = ReplayMemory(CAPACITY)

    env = gym.make("CartPole-v0").unwrapped
    n_observation = env.observation_space.shape[0]
    n_action = env.action_space.n

    policy_net = DQN(n_observation, 32, 32, n_action).to(device)
    target_net = DQN(n_observation, 32, 32, n_action).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=0.0001)

    print("Start: Pre-training")
    o = env.reset()
    step = 0
    while len(memory) < BATCH_SIZE:
        step += 1
        # env.render()
        s = torch.from_numpy(o).type(torch.FloatTensor)
        s = torch.unsqueeze(s, 0)
        a = torch.LongTensor([[env.action_space.sample()]])
        o_next, _, done, _ = env.step(a.item())
        if step >= 200:
            done = True
        if done:
            if step < 195:
                r = torch.FloatTensor([-1.0])
            else:
                r = torch.FloatTensor([1.0])
            s_next = None
            memory.push(s, a, s_next, r)
            # print("(observation: {}, action: {}, next_observation: {}, reward: {})".format(s, a, s_next, r))
            o = env.reset()
            step = 0
        else:
            r = torch.FloatTensor([0.0])
            s_next = torch.from_numpy(o_next).type(torch.FloatTensor)
            s_next = torch.unsqueeze(s_next, 0)
            memory.push(s, a, s_next, r)
            # print("(observation: {}, action: {}, next_observation: {}, reward: {})".format(s, a, s_next, r))
            o = o_next
    print("Finish: Pre-training")
    # env.close()

    print("Start: Training")
    successes = np.zeros(10)
    for e in range(EPISODE):
        o = env.reset()
        done = False
        eps = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * e / EPS_DECAY)
        step = 0
        r_total = 0
        while not done:
            step += 1
            s = torch.from_numpy(o).type(torch.FloatTensor)
            s = torch.unsqueeze(s, 0)
            policy_net.eval()
            if np.random.rand() < eps:
                a = torch.LongTensor([[env.action_space.sample()]])
            else:
                with torch.no_grad():
                    a = policy_net.forward(s).max(1)[1].view(1, 1)
            o_next, _, done, _ = env.step(a.item())

            if step >= 200:
                done = True

            if done:
                if step < 200:
                    r = torch.FloatTensor([-1.0])
                    successes = np.zeros(10)
                else:
                    successes[1:] = successes[:-1]
                    successes[0] = 1
                    r = torch.FloatTensor([1.0])
                s_next = None
                memory.push(s, a, s_next, r)
            else:
                r = torch.FloatTensor([0.0])
                s_next = torch.from_numpy(o_next).type(torch.FloatTensor)
                s_next = torch.unsqueeze(s_next, 0)
                memory.push(s, a, s_next, r)
                o = o_next
            
            policy_net.eval()
            transitions = memory.sample(BATCH_SIZE)
            batch = Transition(*zip(*transitions))
            non_final_mask = torch.BoolTensor(tuple(map(lambda x: x is not None, batch.next_state)))
            non_final_batch_s_next = torch.cat([s_next for s_next in batch.next_state if s_next is not None])
            batch_s = torch.cat(batch.state)
            batch_a = torch.cat(batch.action)
            batch_r = torch.cat(batch.reward)
        
            estimateds = policy_net.forward(batch_s).gather(1, batch_a)
            expecteds = torch.zeros(BATCH_SIZE)
            expecteds[non_final_mask] = target_net.forward(non_final_batch_s_next).max(1)[0].detach()
            expecteds = batch_r + GAMMA * expecteds
        
            policy_net.train()
            loss = F.smooth_l1_loss(estimateds, expecteds.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            print("Episode", e, "Total Step: ", step)
        
        if sum(successes) == 10:
            print("10 Times Success!!")
            break

        if e % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
    print("Finish: Training")