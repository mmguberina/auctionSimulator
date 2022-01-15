import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import statistics
from collections import namedtuple

import matplotlib.pyplot as plt
import math
import pickle
import os

from visualising import *

Parameters = namedtuple('Parameters', ['experiment_id', 'run_ids', 'payment_methods',
                                    'max_epochs', 'auctions_per_strategy_update',
                                    'demand_curve', 'n_agents', 'strategy'])
# load
experiment_id = "031"

parameters_file_name = "Results/Experiment_" + experiment_id + "/" + "parameters.pkl"
parameters_file = open(parameters_file_name, 'rb')
parameters = pickle.load(parameters_file)
parameters_file.close()

experiment_id = parameters.experiment_id
payment_methods = parameters.payment_methods
max_epochs = parameters.max_epochs
auctions_per_strategy_update = parameters.auctions_per_strategy_update
demand_curve = parameters.demand_curve
n_agents = parameters.n_agents
strategy = parameters.strategy
run_ids = parameters.run_ids

payment_methods = ["uniform_pricing"]

auctions_per_strategy_update_slice = [10, 20, 50, 100, 150, 200, 300, 400, 600, 1000]


results = {payment_method : [] for payment_method in payment_methods}
# z value for .95 confidence intervals is
z = 1.64

for payment_method in payment_methods:
    plt.figure()
    for n in auctions_per_strategy_update_slice:
#        x = []
#        y = []
        means = []
        stds = []
        for epochs_run in run_ids:
            with open('Results/Experiment_' + experiment_id + '/agents_' + payment_method + "_" + str(max_epochs) + \
                      "epochs_" + str(auctions_per_strategy_update) + 'runs_EpochsRun' + str(epochs_run) + '.pkl', 'rb') as inp:
                agents = pickle.load(inp)
        for agent in agents:
            for i in range(math.floor(len(agent.payoff_history)/n)):
                if len(agent.payoff_history[i*n:(i+1)*n]) == 0:
                    print(i,n)
                mean = np.mean(np.array(agent.payoff_history[i*n:(i+1)*n]))
                means.append(mean)
                std = np.std(np.array(agent.payoff_history[i*n:(i+1)*n]))
                stds.append(std)
#                print(n, mean, std)
                #x.append(n)
        mean = np.mean(np.array(means))
        avg_std = np.mean(stds)
        # calculating half of the confidence interval size
        range_for_95_perc_conf = (z * avg_std) / np.sqrt(n)

        results[payment_method].append(range_for_95_perc_conf)
#        print(n, avg_std)
#        plt.scatter(x,y, s=0.1)
#        plt.xlim(0,650)
#        plt.ylim(4,12)
#    plt.show()


save_file = open("auctionsPerEpochData_031.pkl", 'wb')
pickle.dump(results, save_file)
parameters_file.close()

print(results)

fig, ax = plt.subplots()
labels = auctions_per_strategy_update_slice
x = np.arange(n_agents)
rects = []
width = 0.35
for i, payment_method in enumerate(results):
    #if yerr_flag:
    #    rects.append(ax.bar(x - width/2 + i * width, results[payment_method]['means'],
    #        width, yerr=results[payment_method]['stds'], label=payment_method))
    #else:
    rects.append(ax.bar(x - width/2 + i * width, np.array(results[payment_method]),
        width, label=payment_method))

ax.set_ylabel('Half of 95% confidence interval')
ax.set_xlabel('Number of auctions in a slice')
ax.set_title('Half of average payoff 95% confidence interval for different number of auctions')
ax.set_xticks(x, labels)
ax.legend()

fig.tight_layout()
plt.savefig("conf_interval_per_n_of_auctions.png")
plt.savefig("conf_interval_per_n_of_auctions.pdf")
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
