import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import cv2 

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython: from IPython import display

class EnvManager():
    def __init__(self,\
                 num_state_imgs = 2, \
                 env_name = 'CartPole-v1',\
                 device = "cpu", 
                 save_trials = False,\
                 save_freq = 100):
        
        self.device = device
        self.env = gym.make(env_name)
        self.done = False
        self.trial_num = 0
        self.trial_imgs = []
        self.save_freq = 100
        self.num_state_imgs = num_state_imgs
        self.save_trials = save_trials
        self.prev_imgs = []

    def get_state(self):
        if self.done:
            self.reset_prev_imgs() 
            return torch.cat(self.prev_imgs,2)
        self.prev_imgs.append(self.get_processed_screen())
        self.prev_imgs.pop(0)
        output = torch.cat(self.prev_imgs,2)
        print(output.shape)
        return output

    def take_action(self, action):
        state, reward, done, _ = self.env.step(action.item())
        return torch.tensor([reward], device = self.device)
    
        
    # env utilities 
    def reset(self):
        state = self.env.reset()
        if self.save_trials and self.trial_num % self.save_freq == 0:
            self.save_trial_video()
        self.trial_num += 1
        self.reset_prev_imgs()
        
    def reset_prev_imgs(self):
        self.prev_imgs.clear()
        for i in range(self.num_state_imgs):
            self.prev_imgs.append(torch.zeros_like(self.get_processed_screen()))
        
    def close(self):
        self.env.close()

    def render(self, mode='human'):
        return self.env.render(mode)

    def num_actions_available(self):
        return self.env.action_space.n


    #self.trial_images.append(cv2.cvtColor(self.render('rgb_array'), cv2.COLOR_BGR2RGB))
    # Image Utilities
    def get_screen_height(self):
        screen = self.get_processed_screen()
        return screen.shape[2]

    def get_screen_width(self):
        screen = self.get_processed_screen()
        return screen.shape[3]

    def get_processed_screen(self):
        screen = self.render('rgb_array').transpose((2,0,1))
        screen = self.crop_screen(screen)
        return self.transform_screen_data(screen)

    def crop_screen(self, screen):
        screen_height = screen.shape[1]

        top = int(screen_height * 0.4)
        bottom = int(screen_height * 0.8)
        screen = screen[:, top:bottom, :]
        
        return screen

    def transform_screen_data(self, screen):
        # convert to float, rescale, convert to tensor
        screen = np.ascontiguousarray(screen, dtype=np.float32)/255
        screen = torch.from_numpy(screen)

        # Use torchvision package to compose image transforms
        resize = T.Compose([
            T.ToPILImage()
            , T.Resize((40,90))
            , T.ToTensor()
        ])

        return resize(screen).unsqueeze(0).to(self.device) 

    #data utilities
    def save_trial_video(self, filePath = "output/trial_videos/", filePrefix  = "trial_"):
        if len(self.trial_images) == 0:
            return
        height, width, layers  = self.trial_imgs[0].shape
        vid = cv2.VideoWriter(filePath + filePrefix + str(self.trial_num) + ".avi", 0, 25, (width,height))

        for img in self.trial_imgs:
            vid.write(img)


if __name__ == "__main__":
    em = EnvManager()
    em.reset()

    screen = em.render('rgb_array')

    plt.figure()
    plt.imshow(screen)
    plt.title("Non-processed screen example")
    #plt.show()
    
    screen = em.get_state()
    
    plt.figure()
    print(type(screen))
    plt.imshow(screen.squeeze(0).permute(1,2,0), interpolation='none')
    #plt.imshow(screen)
    plt.title('Initial screen example')
    #plt.show()
    
    for i in range(5):
        em.take_action(torch.tensor([1]))
    screen = em.get_state()

    plt.figure()
    plt.imshow(screen.squeeze(0).permute(1,2,0), interpolation = 'none')
    plt.title('Processed screen example')
    #plt.show()

    em.done = True
    screen = em.get_state()

    plt.figure()
    plt.imshow(screen.squeeze(0).permute(1,2,0), interpolation = 'none')
    plt.title('End State Screen Example')
    plt.show()
    
    em.close()
    

  
