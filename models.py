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
        
        # self.states = nn.Parameter(torch.tensor(np.random.choice([0, 1], size=(3,), p=[p_initial, 1-p_initial]).astype(np.float32),requires_grad=True).float())
        self.states = torch.tensor(np.random.choice([0, 1], size=(3,), p=[p_initial, 1-p_initial]).astype(np.float32),requires_grad=False).float()
        
    def forward(self,input1,input2=1,input3=1):    
        B_temp = 1 - self.states[1]*input2
        C_temp = 1 - self.states[2]*input3
            
        self.states[1] = B_temp
        self.states[2] = C_temp
        self.states[0] = self.states[2]
        
        return (1 - self.states[0]*input1)
        
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
    def __init__(self, neurons = 3, p_initial = 0.5, grad = False):
        super(BITypeNetwork, self).__init__()
        self.p_initial = p_initial
        self.neurons = neurons        
        self.grad = grad
        self.adj = []
        self.blocks = nn.ModuleList([])
        for neuron in range(neurons):
            adj_row = np.zeros(neurons)
            adj_row[np.random.choice(neurons,2,replace=False)] = 1
            self.adj.append(adj_row)
            self.blocks.append(BTypeBlock(p_initial))
        
        if self.grad:
            self.adj = nn.Parameter(torch.tensor(np.array(self.adj),requires_grad=self.grad).float())
            self.states = nn.Parameter(torch.tensor(np.random.choice([0, 1], size=(neurons,), p=[p_initial, 1-p_initial]).astype(np.float32),requires_grad=self.grad).float())
        else:
            self.adj = torch.tensor(np.array(self.adj),requires_grad=self.grad).float()
            self.states = torch.tensor(np.random.choice([0, 1], size=(neurons,), p=[p_initial, 1-p_initial]).astype(np.float32),requires_grad=self.grad).float()
    
        # self.W = nn.Parameter(torch.ones(self.adj.shape))
        self.W = nn.Parameter(torch.ones(self.states.shape))
    
    def forward(self,x):
        x = x.view(-1)
        N = x.shape[0]
        intermediate = 1 - ((1 - self.adj) + self.adj*self.states).prod(dim=1)
        for i, state in enumerate(intermediate):
            block = self.blocks[i]
            self.states[i] = block.forward(state,x[i%N],x[(i+1)%N])
        
        return self.states
    
    def forward_sequence(self, x, init_p=0):
        #Assumes: (1) K = features, (2) T = sequence length, (3) # of neurons >= K
        if type(init_p) != type(''):
            if init_p>0:
                self.reinit(p_initial = init_p)
        elif init_p == 'ones':
            self.reinit(ones = True)
        elif init_p == 'zeros':
            self.reinit(zeros = True)            
        
        K = x.shape[0]
        T = x.shape[-1]
        output = []
        for t in range(T):
            x_t = x[:,t]
            # intermediate = 1 - ((1 - self.W*self.adj) + self.W*self.adj*self.states)
            # intermediate = intermediate.prod(dim=1)
            intermediate = (1 - ((1 - self.adj) + self.adj*self.states)).prod(dim=1)
            for i, state in enumerate(intermediate):
                block = self.blocks[i]
                self.states[i] = self.W[i] * torch.Tensor(block.forward(state,x_t[i%K],x_t[(i+1)%K]).cpu().data.numpy())
                # self.states[i] = self.W[i] * block.forward(state,x_t[i%K],x_t[(i+1)%K])
            output.append(self.states[:K])        
        
        return torch.stack(output).t()
                    
    def reinit_(self, ones = False, zeros = False, p_initial = None):
        if ones:
            self.states = nn.Parameter(torch.ones(self.neurons).float())
        elif zeros:
            self.states = nn.Parameter(torch.zeros(self.neurons).float())
        else:
            if p_initial is None:
                p_initial = self.p_initial
            self.states = nn.Parameter(torch.tensor(np.random.choice([0, 1], size=(self.neurons,), p=[self.p_initial, 1-self.p_initial]).astype(np.float32),requires_grad=self.grad).float())


    def reinit(self, ones = False, zeros = False, p_initial = None):
        self.reinit_()
        return self
        
class MLP(nn.Module):
    def __init__(self, input_size, width=500, num_classes=10):
        super(MLP, self).__init__()

        ##feedfoward layers:

        bias_ind = True

        self.ff1 = nn.Linear(input_size, width, bias = bias_ind) #input

        self.ff2 = nn.Linear(width, width, bias = bias_ind) #hidden layers
        self.ff3 = nn.Linear(width, width, bias = bias_ind)
        self.ff4 = nn.Linear(width, width, bias = bias_ind)
        self.ff5 = nn.Linear(width, width, bias = bias_ind)

##        self.ff_out = nn.Linear(width, num_classes, bias = bias_ind) #output     
        self.ff_out = nn.Linear(width, num_classes, bias = bias_ind) #output     
        
        ##activations:
        self.do = nn.Dropout()
        self.relu = nn.ReLU()
        self.sm = nn.Softmax()
        
        ##BN:
        self.bn1 = nn.BatchNorm1d(width)
        self.bn2 = nn.BatchNorm1d(width)
        self.bn3 = nn.BatchNorm1d(width)
        self.bn4 = nn.BatchNorm1d(width)
        self.bn5 = nn.BatchNorm1d(width)
        
    def forward(self, input_data):

        ##forward pass computation:
        
        out = self.relu(self.ff1(input_data)) #input

        out = self.relu(self.ff2(out)) #hidden layers
        out = self.relu(self.ff3(out))
        out = self.relu(self.ff4(out))
        out = self.relu(self.ff5(out))

        out = self.ff_out(out)

        return out #returns class probabilities for each image
        
        
        








