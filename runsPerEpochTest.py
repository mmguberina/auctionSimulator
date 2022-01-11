import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import statistics

import matplotlib.pyplot as plt
import math
import pickle
import os

from visualising import *

payment_methods = ["uniform_pricing"]
auctions_per_strategy_update = 1200
max_epochs = 1
whole_epochs_runs = 10
n_agents = 5

auctions_per_strategy_update_slice = [10, 20, 50, 100, 150, 200, 300, 400, 600]



for payment_method in payment_methods:
    plt.figure()
    for n in auctions_per_strategy_update_slice:
        x = []
        y = []
        for epochs_run in range(whole_epochs_runs):
            with open('Results\\agents_' + payment_method + "_" + str(max_epochs) + \
                      "epochs_" + str(auctions_per_strategy_update) + 'runs_EpochsRun' + str(epochs_run) + '.pkl', 'rb') as inp:
                agents = pickle.load(inp)
        for agent in agents:
            for i in range(math.floor(len(agent.payoff_history)/n)):
                if len(agent.payoff_history[i*n:(i+1)*n]) == 0:
                    print(i,n)
                mean = statistics.mean(agent.payoff_history[i*n:(i+1)*n])
                x.append(n)
                y.append(mean)
        plt.scatter(x,y)
        plt.xlim(0,650)
        plt.ylim(4,12)
    plt.show()

"""
results = pandas_results(n_agents, payment_methods, max_epochs, auctions_per_strategy_update, whole_epochs_runs)
for payment_method in results.index.levels[0]:
    plt.figure()
    for epochs_run in results.index.levels[1]:
        for agent_number in results.index.levels[2]:
            x = [i[0] for i in results.loc[(payment_method,epochs_run,agent_number),"s_mix_2D"]]
            y = [i[1] for i in results.loc[(payment_method,epochs_run,agent_number),"s_mix_2D"]]
            plt.scatter(x,y,s=1,label='agent ' + str(agent_number),c=[i for i in range(len(x))],cmap='Blues')
    
    
    # ploting the edges of the triangle
    plt.plot([0, 1], [0, 0], 'k', [0, 0.5], [0, math.sin(math.pi / 3)], 'k', [1, 0.5], [0, math.sin(math.pi / 3)],
             'k')
    cbar = plt.colorbar()
    cbar.set_label('Epoch')
    plt.show()
"""
