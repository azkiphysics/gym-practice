import os
import random
from collections import deque, namedtuple

import cv2
import gym
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras as K

if __name__ == "__main__":
    EPISODE = 200

    env = gym.make("CartPole-v0").unwrapped

    for e in range(EPISODE):
        