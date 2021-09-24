"""
[CSCI-GA 3033-090] Special Topics: Deep Reinforcement Learning

Homework - 1, DAgger
Deadline: Sep 17, 2021 11:59 PM.

Complete the code template provided in dagger.py, with the right 
code in every TODO section, to implement DAgger. Attach the completed 
file in your submission.
"""

from numpy.core.fromnumeric import argmax, take
from numpy.lib.function_base import copy
import tqdm
import hydra
import os
import wandb
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision.transforms as T

import numpy as np

from reacher_env import ReacherDaggerEnv
from utils import weight_init, ExpertBuffer


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # TODO define your own network
        input_shape = ReacherDaggerEnv().observation_space.shape
        n_space = ReacherDaggerEnv().action_space.shape[0]
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=16, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=8, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_space),
            nn.Tanh()
        )

        self.apply(weight_init)

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
    
    def forward(self, x):
        # Normalize
        x = x / 255.0 - 0.5
        # TODO pass it forward through your network.
        x = self.fc(self.conv(x))
        return x



def initialize_model_and_optim(cfg):
    # TODO write a function that creates a model and associated optimizer
    # given the config object.
    net = CNN().to(cfg.device)
    return net , optim.Adam(net.parameters(), lr=cfg.lr)


class Workspace:
    def __init__(self, cfg):
        self._work_dir = os.getcwd()
        print(f'workspace: {self._work_dir}')

        self.cfg = cfg

        self.device = torch.device(cfg.device)
        self.train_env = ReacherDaggerEnv()
        self.eval_env = ReacherDaggerEnv()

        self.expert_buffer = ExpertBuffer(cfg.experience_buffer_len, 
                                          self.train_env.observation_space.shape,
                                          self.train_env.action_space.shape)
        
        self.model, self.optimizer = initialize_model_and_optim(cfg)

        # TODO: define a loss function
        self.loss_function = F.mse_loss

        self.transforms = T.Compose([
            T.RandomResizedCrop(size=(224, 224)),
        ])
        self.eval_transforms = T.Compose([
            T.Resize(size=(224, 224))
        ])

    def eval(self):
        # A function that evaluates the 
        # Set the DAgger model to evaluation
        self.model.eval()

        avg_eval_reward = 0.
        avg_episode_length = 0.
        for _ in range(self.cfg.num_eval_episodes):
            eval_reward = 0.
            ep_length = 0.
            obs_np = self.eval_env.reset()
            # Need to be moved to torch from numpy first
            obs = torch.from_numpy(obs_np).float().to(self.device).unsqueeze(0)
            t_obs = self.eval_transforms(obs)
            with torch.no_grad():
                action = self.model(t_obs)
            done = False
            while not done:
                # Need to be moved to numpy from torch
                action = action.squeeze().detach().cpu().numpy()
                obs, reward, done, info = self.eval_env.step(action)
                obs = torch.from_numpy(obs).float().to(self.device).unsqueeze(0)
                t_obs = self.eval_transforms(obs)
                with torch.no_grad():
                    action = self.model(t_obs)
                eval_reward += reward
                ep_length += 1.
            avg_eval_reward += eval_reward
            avg_episode_length += ep_length
        avg_eval_reward /= self.cfg.num_eval_episodes
        avg_episode_length /= self.cfg.num_eval_episodes
        return avg_eval_reward, avg_episode_length


    def model_training_step(self):
        # A function that optimizes the model self.model using the optimizer 
        # self.optimizer using the experience  stored in self.expert_buffer.
        # Number of optimization step should be self.cfg.num_training_steps.

        # Set the model to training.
        self.model.train()
        # For num training steps, sample data from the training data.
        avg_loss = 0.
        for _ in range(self.cfg.num_training_steps):
            # TODO write the training code.
            # Hint: use the self.transforms to make sure the image observation is of the right size.
            batch_obs, batch_action = self.expert_buffer.sample()

            t_batch_obs = torch.from_numpy(batch_obs).float().to(self.device)
            t_batch_obs = self.transforms(t_batch_obs)
            t_batch_action = torch.from_numpy(batch_action).float().to(self.device)                 

            t_actions = self.model(t_batch_obs)

            loss = self.loss_function(t_actions,t_batch_action)

            avg_loss += loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        avg_loss /= self.cfg.num_training_steps
        wandb.log({"avg_loss":avg_loss})
        return avg_loss

    def alpha_decay_policy_selector(self,ep_num):
        random_num = random.randint(0,self.cfg.total_training_episodes)
        take_expert_action = True if random_num > ep_num else False
        return take_expert_action

    def run(self):
        train_loss, eval_reward, episode_length = 1., 0, 0
        iterable = tqdm.trange(self.cfg.total_training_episodes)
        for ep_num in iterable:
            iterable.set_description('Collecting exp')
            # Set the DAGGER model to evaluation
            self.model.eval()
            ep_train_reward = 0.
            ep_length = 0.

            # TODO write the training loop.
            # 1. Roll out your current model on the environment.
            # 2. On each step, after calling either env.reset() or env.step(), call 
            #    env.get_expert_action() to get the expert action for the current 
            #    state of the environment.
            # 3. Store that observation alongside the expert action in the buffer.
            # 4. When you are training, use the stored obs and expert action.

            # Hints:
            # 1. You will need to convert your obs to a torch tensor before passing it
            #    into the model.
            # 2. You will need to convert your action predicted by the model to a numpy
            #    array before passing it to the environment.
            # 3. Make sure the actions from your model are always in the (-1, 1) range.
            # 4. Both the environment observation and the expert action needs to be a
            #    numpy array before being added to the environment.
            # 5. Use the self.transforms to make sure the image observation is of the right size.
            
            # TODO training loop here.
            obs = self.train_env.reset()
            done = False
            while not done:

                expert_action = self.train_env.get_expert_action()
                if self.alpha_decay_policy_selector(ep_num):
                    policy_action = self.train_env.get_expert_action()
                else:
                    policy_action = self.model( self.transforms(torch.from_numpy(obs).float().to(self.device).unsqueeze(0)) )
                    policy_action = policy_action.squeeze().detach().cpu().numpy()
                
                self.expert_buffer.insert(np.array(obs, copy=False),np.array(expert_action,copy=False))
                wandb.log({"replay_buffer_len":self.expert_buffer.__len__()})

                obs, reward, done, info = self.train_env.step(policy_action)
                ep_train_reward += reward
                ep_length += ep_length

            train_reward = ep_train_reward
            train_episode_length = ep_length

            if (ep_num + 1) % self.cfg.train_every == 0:
                # Reinitialize model every time we are training
                iterable.set_description('Training model')
                # TODO train the model and set train_loss to the appropriate value.
                # Hint: in the DAgger algorithm, when do we initialize a new model?
                train_loss = self.model_training_step()

            if (ep_num + 1) % self.cfg.eval_every == 0:
                # Evaluation loop
                iterable.set_description('Evaluating model')
                eval_reward, episode_length = self.eval()

            iterable.set_postfix({
                'Train loss': train_loss,
                'Train reward': train_reward,
                'Eval reward': eval_reward
            })
            wandb.log({"train_reward":train_reward, "eval_reward":eval_reward, "expert_calls": self.train_env.expert_calls})


@hydra.main(config_path='.', config_name='train')
def main(cfg):
    # In hydra, whatever is in the train.yaml file is passed on here
    # as the cfg object. To access any of the parameters in the file,
    # access them like cfg.param, for example the learning rate would
    # be cfg.lr
    wandb.init(project="deep-rl-hw1", name=f"Dagger-{cfg.run_name}")
    workspace = Workspace(cfg)
    workspace.run()


if __name__ == '__main__':
    main()
