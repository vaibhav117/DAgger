import tqdm
import hydra
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision.transforms as T

import numpy as np

from reacher_env import ReacherDaggerEnv
from utils import weight_init, ExpertBuffer
import matplotlib.pyplot as plt


env = ReacherDaggerEnv()
env.reset() 

print(f"action_space: {env.action_space.shape[0]}")

def try2():
    for _ in range(1000):
        action = env.get_expert_action()
        visual_obs, reward, done, info = env.step(action)
        visual_obs = np.moveaxis(visual_obs, 0, -1)
        plt.imshow(visual_obs)
        plt.show()
        
        print(f"shape of visual_obs:{visual_obs.shape}\n")
        print(f"reward:{reward} , action:{action}\n")



def try1():
    env.reset()
    for i in range(100):
        env.step(env.action_space.sample())
        rgb_array_large = env.render(
            mode="rgb_array",
        )
        plt.imshow(rgb_array_large)
        plt.show()

try1()