import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import copy
import numpy as np
import matplotlib.pyplot as plt
import time

from models import *

## hyperparams:
neurons = 50
p_initial = 0.2
N = 1 #number of datapoints
K = 50 #number of features
timesteps = 200
loops = 10

binary = True

for loop in range(loops):
    ## data generation and preprocessing:
    xdata = torch.randn(N,K)
    xdata -= xdata.min()
    xdata /= xdata.max()
    if binary:
        xdata = torch.round(xdata)
    
    ## defining the net:
    net = BITypeNetwork(neurons = neurons, p_initial = p_initial)
    
    differences = []
    for t in range(timesteps):
        prev_state = copy.copy(net.states.cpu().data.numpy())
        differences.append((np.abs(net(xdata).cpu().data.numpy() - prev_state).sum()).item())
    
    plt.plot(differences,'o',label='absolute difference for each time step')
    plt.plot([], [], ' ', label='neurons: %d'%(neurons))
    plt.plot([], [], ' ', label='p_initial: %.2f'%(p_initial))
    plt.plot([], [], ' ', label='features: %d'%(K))
    if binary:
        plt.plot([], [], ' ', label='Binary inputs')
    plt.xlabel('timestep')
    plt.ylabel('absolute state difference sum')
    plt.legend()
    plt.savefig('./figures/difference_fig%d.png'%loop,dpi=500)
    plt.show()

    

        
        
        








