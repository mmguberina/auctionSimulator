import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import statistics

from agents import *
from market_clearing import *
from mechanisms import *
from strategies import *
from strategy_updating_algorithms import *
from visualising import *

runs_per_strategy_update = 10000
max_epochs = 1
epoch = 0

if __name__ == "__main__":
    n_of_demand_bids = 20
    demand_curve = [[25/n_of_demand_bids,i] for i in list(np.linspace(5, 1, num = n_of_demand_bids))]

    n_agents = 5
    agents = []
    init_strategy_mix = [] 
    for agent_num in range(n_agents):
        agents.append(Agent("all_the_same", init_strategy_mix, agent_num, max_epochs, runs_per_strategy_update))


    # Run runs_per_strategy_update times
    for run_of_strategy in range(runs_per_strategy_update):
        for a in agents:
            a.generateBid()

#        supply_quantities_cleared_solution, demand_quantities_cleared_solution, m = marketClearing(agents,
#                                                                                                   demand_curve)

#        uniformPricing(agents, supply_quantities_cleared_solution, demand_quantities_cleared_solution, m, \
#                       epoch, runs_per_strategy_update, run_of_strategy)

        supply_quantities_cleared_solution, demand_quantities_cleared_solution, result = marketClearingSciPy(agents,
                                                                                                   demand_curve)
#        exit()
#        SW_history[runs_per_strategy_update * epoch + run_of_strategy] = copy.deepcopy(m.ObjVal)
    # Update strategy position
#    for agent in agents:
#        agent.epoch_payoff_history[epoch] = \
#            statistics.mean(agent.payoff_history[runs_per_strategy_update*epoch:runs_per_strategy_update*(epoch+1)])
#    PSO(agents, max_epochs, epoch)
#    epoch += 1

#    SavingAgents(agents, SW_history, payment_method, max_epochs, runs_per_strategy_update, epochs_run)
#    results, results_SW = pandas_results(n_agents, payment_methods, max_epochs, runs_per_strategy_update, whole_epochs_runs)
    #plotAgentsChanges2D_all(results,plot_saving_switch)
        #plotSupplyDemand(agents, demand_curve, payment_method, epochs_run)
        #plotAgentChanges2D(agents, payment_method, epochs_run)
        #plotSW(SW_history, runs_per_strategy_update, payment_method, epochs_run)
        #plotPayoffs(agents, payment_method, runs_per_strategy_update, epochs_run)

