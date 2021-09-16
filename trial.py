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


env = ReacherDaggerEnv()
env.reset() 

print(f"action_space: {env.action_space.shape[0]}")

for _ in range(1000):
    action = env.get_expert_action()
    visual_obs, reward, done, info = env.step(action)
    print(f"shape of visual_obs:{visual_obs.shape}\n")
    print(f"reward:{reward} , action:{action}\n")