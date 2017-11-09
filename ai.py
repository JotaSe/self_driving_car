# AI for Self Driving Car

# Importing the libraries

import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

# Creating the architecture of the Neural Network
class Network(nn.Module):
    
    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        self.fc1 = nn.Linear(input_size, 30)
        self.fc2 = nn.Linear(30, 20)
        self.fc3 = nn.Linear(20, nb_action)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values

# Implementing Experience Replay
class ReplayMemory(object):
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
    
    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]
    
    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x, 0)), samples)

# Implementing Deep Q Learning

class Dqn():
    
    def __init__(self, input_size, nb_action, gamma, default_action= 0, 
                 memory_size= 100000, learning_rate= 0.001, temp_value= 100,
                 min_memory_to_learn= 100, reward_windows_capacity= 1000,
                 model_path= 'last_brain.pth'):
        self.gamma = gamma
        self.reward_window = []
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(memory_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr =learning_rate)
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = default_action
        self.last_reward = 0
        self.T = temp_value
        self.min_memory_to_learn = min_memory_to_learn
        self.reward_windows_capacity = reward_windows_capacity
        self.model_path = model_path
    
    # Select the best action by the best Q-value
    def select_action(self, state):
        probs = F.softmax(self.model(Variable(state, volatile = True))*self.T)
        action = probs.multinomial()
        return action.data[0,0]
    
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        # get states by a given batch
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        
        # get the next states
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        
        # Calculate the target
        target = self.gamma*next_outputs + batch_reward
        
        # Best stop loss for Q-learning `smooth_l1_loss`
        td_loss = F.smooth_l1_loss(outputs, target)
        
        # Reinitialize our optimizer
        self.optimizer.zero_grad()
        
        # Backpropagation
        td_loss.backward(retain_variables = True)
        
        self.optimizer.step()
    
    def update(self, reward, new_signal):
        # Create a new state with the new signal, unqueeze for flat dimension
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        
        # add the new event to the memory, with the last state, the new state,
        # the last action in a LongTensor object, and the last reward
        # as a Tensor object
        self.memory.push((self.last_state, new_state, 
                          torch.LongTensor([int(self.last_action)]), 
                          torch.Tensor([self.last_reward])))
        
        # Select the best action given by the new state
        action = self.select_action(new_state)
        
        # Learn if the agent has enough memory 
        if len(self.memory.memory) > self.min_memory_to_learn:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(self.min_memory_to_learn)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
            
        # Set the new action, state and reward as the lastest
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        
        # Add the new reward to the reward_windows and delete the first if
        # the size of the reward_windows is greater than 1000
        self.reward_window.append(reward)
        if len(self.reward_window) > self.reward_windows_capacity:
            del self.reward_window[0]
        return action
    
    # Return the mean score stored in the reward_windows
    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1.)
    
    # Save function to save states and the optimizer
    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict(),
                   }, self.model_path)
    
    # Load function to load stored state's weights and the optimizer
    def load(self):
        if os.path.isfile(self.model_path):
            print("=> loading checkpoint... ")
            checkpoint = torch.load(self.model_path)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done !")
        else:
            print("no checkpoint found...")