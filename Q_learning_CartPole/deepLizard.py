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


class DQN(nn.Module):
    def __init__(self, img_height, img_width):
        super().__init__()

        self.fc1 = nn.Linear(in_features=img_height*img_width*3, out_features=24)
        torch.nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity = 'relu')
        #torch.nn.init.uniform_(self.fc1.weight)
        #torch.nn.init.ones_(self.fc1.bias)
        self.fc2 = nn.Linear(in_features=24, out_features=32)
        torch.nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity = 'relu')
        #torch.nn.init.uniform_(self.fc2.weight)
        #torch.nn.init.ones_(self.fc2.bias)
        self.out = nn.Linear(in_features=32, out_features=2)
        torch.nn.init.kaiming_uniform_(self.out.weight, nonlinearity = 'relu')
        #torch.nn.init.uniform_(self.out.weight)
        #torch.nn.init.ones_(self.out.bias)

    def forward(self, t):
        t = t.flatten(start_dim = 1)
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        t = self.out(t)
        return t

class DQNJoint(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(in_features=4, out_features=24)
        torch.nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity = 'relu')
        #torch.nn.init.ones_(self.fc1.bias)
        self.fc2 = nn.Linear(in_features=24, out_features=24)
        torch.nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity = 'relu')
        #torch.nn.init.uniform_(self.fc2.weight)
        #torch.nn.init.ones_(self.fc2.bias)
        self.fc3 = nn.Linear(in_features=24, out_features=24)
        torch.nn.init.kaiming_uniform_(self.fc3.weight, nonlinearity = 'relu')
        #torch.nn.init.uniform_(self.fc3.weight)
        #torch.nn.init.ones_(self.fc3.bias)
       # self.fc4 = nn.Linear(in_features=48, out_features=48)
       # torch.nn.init.kaiming_uniform_(self.fc4.weight, nonlinearity = 'relu')
        #torch.nn.init.uniform_(self.fc4.weight)
        #torch.nn.init.ones_(self.fc4.bias)
        self.out = nn.Linear(in_features=24, out_features=2)
        torch.nn.init.kaiming_uniform_(self.out.weight, nonlinearity = 'relu')
        #torch.nn.init.uniform_(self.out.weight)
        #torch.nn.init.ones_(self.out.bias)

    def forward(self, t):
        #t = t.flatten(start_dim = 1) 
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        t = F.relu(self.fc3(t))
        #t = F.relu(self.fc4(t))
        t = self.out(t)
        return t
    
Experience = namedtuple(
    'Experience',
    ('state', 'action', 'next_state', 'reward')
)

if __name__ == "__main__":
    e = Experience(2,3,1,4)
    print(e)

class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0

    def push(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.push_count % self.capacity] = experience
        self.push_count += 1

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def can_provide_sample(self, batch_size):
        return len(self.memory) >= batch_size


class EpsilonGreedyStrategy():
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay

    def get_exploration_rate(self, current_step):
        return self.end + (self.start - self.end) * \
            math.exp(-1. * current_step * self.decay)

class Agent():
    def __init__(self, strategy, num_actions, device):
        self.current_step = 0
        self.strategy = strategy
        self.num_actions = num_actions
        self.device = device

    def select_action(self, state, policy_net, state_type="image_diff"):
        rate = self.strategy.get_exploration_rate(self.current_step)
        self.current_step += 1

        if rate > random.random():
            action =  random.randrange(self.num_actions) 
            return torch.tensor([action]).to(self.device) # explore
        else:
            with torch.no_grad():
                if state_type == "image_diff":
                    return policy_net(state).argmax(dim=1).to(self.device) # exploit
                elif state_type == "joint":
                    action =  policy_net(state).argmax()#.to(self.device)
                    return torch.tensor([action]).to(self.device)


class CartPoleEnvManager():
    def __init__(self, device, state_type='image_diff', saveTrials = True):
        self.device = device
        self.env = gym.make('CartPole-v1')#.unwrapped
        self.joint_state = self.env.reset()
        self.current_screen = None
        self.done = False
        self.state_type = state_type
        self.trial_images = []
        self.trial_num = 0

    def reset(self):
        self.joint_state = self.env.reset()
        self.save_trial_video()
        self.trial_num += 1
        self.current_screen = None

    def close(self):
        self.env.close()

    def render(self, mode='human'):
        return self.env.render(mode)

    def num_actions_available(self):
        return self.env.action_space.n

    def take_action(self, action):
        self.joint_state, reward, self.done, _ = self.env.step(action.item())
        return torch.tensor([reward], device= self.device)

    def just_starting(self):
        return self.current_screen is None

    def save_trial_video(self, filePath = "output/trial_videos/trial"):
        if len(self.trial_images) == 0:
            return
        height, width, layers  = self.trial_images[0].shape
        vid = cv2.VideoWriter(filePath + str(self.trial_num) + ".avi", 0, 25, (width,height))

        for img in self.trial_images:
            vid.write(img)

        self.trial_images.clear()
    
    def get_state(self):
        if self.state_type == 'image_diff':
            if self.just_starting() or self.done:
                self.current_screen = self.get_processed_screen()
                black_screen = torch.zeros_like(self.current_screen)
                return black_screen
            else:
                s1 = self.current_screen
                s2 = self.get_processed_screen()
                self.current_screen = s2
                #self.trial_images.append(self.current_screen.cpu().numpy())
                return s2-s1
        elif self.state_type == 'joint':
            self.trial_images.append(cv2.cvtColor(self.render('rgb_array'), cv2.COLOR_BGR2RGB))
            return torch.from_numpy(self.joint_state).to(self.device)

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


def evoManagerTest():
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    em = CartPoleEnvManager(device)
    em.reset()

    screen = em.render('rgb_array')

    plt.figure()
    plt.imshow(screen)
    plt.title("Non-processed screen example")
    #plt.show()
    
    screen = em.get_state()
    
    plt.figure()
    plt.imshow(screen.squeeze(0).permute(1,2,0), interpolation='none')
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

if __name__ == "__main__":
    evoManagerTest()

# Helpful utility functions
def plot(values, moving_avg_period, folderPath='output.png'):
    plt.figure(2)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(values)
    plt.plot([0, len(values)], [195,195])
    moving_avg = get_moving_average(moving_avg_period, values)
    plt.plot(moving_avg)
    if len(values) % moving_avg_period == 0:
        plt.savefig(folderPath)
    plt.pause(0.001)
    print("Episode", len(values), "\n", \
          moving_avg_period, "episode moving avg:", moving_avg[-1])

    if is_ipython: display.clear_output(wait=True)

def get_moving_average(period, values):
    values = torch.tensor(values, dtype=torch.float)
    if len(values)>= period:
        moving_avg = values.unfold(dimension=0, size=period, step=1) \
            .mean(dim=1).flatten(start_dim=0)
        moving_avg = torch.cat((torch.zeros(period-1), moving_avg))
        return moving_avg.numpy()
    else:
        moving_avg = torch.zeros(len(values))
        return moving_avg.numpy()

def extract_tensors(experiences):

    batch = Experience(*zip(*experiences))
    
    t1 = torch.stack(batch.state)
    t2 = torch.cat(batch.action)
    t3 = torch.cat(batch.reward)
    t4 = torch.stack(batch.next_state)

    return(t1, t2, t3, t4)

class QValues():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def get_current(policy_net, states, actions):
        return policy_net(states).gather(dim=1, index = actions.unsqueeze(-1))

    @staticmethod
    def get_next(target_net, next_states):
        final_state_locations = next_states.flatten(start_dim=1)\
            .max(dim=1)[0].eq(0).type(torch.bool)
        non_final_state_locations = (final_state_locations == False)
        non_final_states = next_states[non_final_state_locations]
        batch_size = next_states.shape[0]
        values = torch.zeros(batch_size).to(QValues.device)
        values[non_final_state_locations] = target_net(non_final_states).max(dim=1)[0].detach()
        return values
    
class QValuesJoint():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def get_current(policy_net, states, actions):
        return policy_net(states).gather(dim = 1, index = actions.unsqueeze(-1))

    @staticmethod
    def get_next(target_net, next_states):
        final_state_locations = next_states.flatten(start_dim=1).max(dim=1)[0].eq(0).type(torch.bool)
        non_final_state_locations = (final_state_locations == False)
        non_final_states = next_states[non_final_state_locations]
        batch_size = next_states.shape[0]
        values = torch.zeros(batch_size).to(QValuesJoint.device)
        values[non_final_state_locations] = target_net(non_final_states).max(dim=1)[0].detach()
        return values

if __name__ == "__main__":
    plot(np.random.rand(300), 100)
    plt.show()
