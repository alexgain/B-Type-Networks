import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import copy
import numpy as np
import matplotlib.pyplot as plt
import time

class ATypeNetwork(nn.Module):
    def __init__(self, neurons = 3, p_initial = 0.5):
        super(ATypeNetwork, self).__init__()
        
        self.adj = []
        for neuron in range(neurons):
            adj_row = np.zeros(neurons)
            adj_row[np.random.choice(neurons,2,replace=False)] = 1
            self.adj.append(adj_row)
        self.adj = torch.tensor(np.array(self.adj),requires_grad=False).float()
        self.states = torch.tensor(np.random.choice([0, 1], size=(neurons,), p=[p_initial, 1-p_initial]),requires_grad=False).float()
        
    
    def forward(self):
        self.states = 1 - ((1 - self.adj) + self.adj*self.states).prod(dim=1)
        return self.states

class BTypeBlock(nn.Module):
    def __init__(self,p_initial = 0.5):
        super(BTypeBlock,self).__init__()
        
        self.states = torch.tensor(np.random.choice([0, 1], size=(3,), p=[p_initial, 1-p_initial]),requires_grad=False).float()
        
    def forward(self,input1,input2=1,input3=1):    
        B_temp = 1 - self.states[1]*input2
        C_temp = 1 - self.states[2]*input3
            
        self.states[1] = B_temp
        self.states[2] = C_temp
        self.states[0] = self.states[2]
        
        return 1 - self.states[0]*input1
        
class BTypeNetwork(nn.Module):
    def __init__(self, neurons = 3, p_initial = 0.5):
        super(BTypeNetwork, self).__init__()
        
        self.adj = []
        self.blocks = nn.ModuleList([])
        for neuron in range(neurons):
            adj_row = np.zeros(neurons)
            adj_row[np.random.choice(neurons,2,replace=False)] = 1
            self.adj.append(adj_row)
            self.blocks.append(BTypeBlock(p_initial))
            
        self.adj = torch.tensor(np.array(self.adj),requires_grad=False).float()
        self.states = torch.tensor(np.random.choice([0, 1], size=(neurons,), p=[p_initial, 1-p_initial]),requires_grad=False).float()                
    
    def forward_input(self):
        intermediate = 1 - ((1 - self.adj) + self.adj*self.states).prod(dim=1)
        for i, state in enumerate(intermediate):
            block = self.blocks[i]
            self.states[i] = block.forward(state)
        
        return self.states        
        
class BITypeNetwork(nn.Module):
    def __init__(self, neurons = 3, p_initial = 0.5):
        super(BITypeNetwork, self).__init__()
        self.p_initial = p_initial
        self.neurons = neurons        
        
        self.adj = []
        self.blocks = nn.ModuleList([])
        for neuron in range(neurons):
            adj_row = np.zeros(neurons)
            adj_row[np.random.choice(neurons,2,replace=False)] = 1
            self.adj.append(adj_row)
            self.blocks.append(BTypeBlock(p_initial))
        
        self.adj = torch.tensor(np.array(self.adj),requires_grad=False).float()
        self.states = torch.tensor(np.random.choice([0, 1], size=(neurons,), p=[p_initial, 1-p_initial]),requires_grad=False).float()                
    
    def forward(self,x):
        x = x.view(-1)
        N = x.shape[0]
        intermediate = 1 - ((1 - self.adj) + self.adj*self.states).prod(dim=1)
        for i, state in enumerate(intermediate):
            block = self.blocks[i]
            self.states[i] = block.forward(state,x[i%N],x[(i+1)%N])
        
        return self.states
    
    def reinit(self, p_initial = None):
        if p_initial is None:
            p_initial = self.p_initial
        self.states = torch.tensor(np.random.choice([0, 1], size=(self.neurons,), p=[self.p_initial, 1-self.p_initial]),requires_grad=False).float()
        
        
        
        
        
        








