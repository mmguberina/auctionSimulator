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

payment_methods = ["uniform_pricing"]
# Might want to move this to the runs.py file,
runs_per_strategy_update = 1200  # Example for simple strategies
# (might need other criteria with more complex strategies)
# define termination criteria
max_epochs = 1  # Just for testing with simple termination criteria
whole_epochs_runs = 10

# let's start with the following
# 1 buyer, static demand curve
# same form as bid curve
if __name__ == "__main__":
    # Create demand curve
    #n_of_demand_bids = 5
    # only 1 buyer
    #demand_curve = [[5, (n_of_demand_bids - i)*1.2] for i in range(n_of_demand_bids)]
    #demand_curve[2][1] += 0.01
    n_of_demand_bids = 20
    demand_curve = [[25/n_of_demand_bids,i] for i in list(np.linspace(5, 1, num = n_of_demand_bids))]

    for epochs_run in range(whole_epochs_runs):
        # Initialize agents
        n_agents = 5
        init_strategy_mix = [] #the initial strategy mix for the agents
        for payment_method in payment_methods:
            agents = []
            for agent_num in range(n_agents):
                agents.append(Agent("all_the_same", init_strategy_mix, agent_num, max_epochs, runs_per_strategy_update))
                #if it is the first payment method, we save the init_strategy_mix to be used later as the initial for the
                #   other payment methods
                if len(init_strategy_mix) != n_agents:
                    init_strategy_mix.append(agents[agent_num].strategy_mix)
            # Social welfare history

            SW_history = [None] * max_epochs * runs_per_strategy_update

            epoch = 0

            while epoch < max_epochs:
                print("[epoch run, payment, epoch]: ", [epochs_run,payment_method,epoch])

                # Run runs_per_strategy_update times
                for run_of_strategy in range(runs_per_strategy_update):
                    for a in agents:
                        a.generateBid()

                    # Market clearing function
                    # supply_bids = [a.bids_curve for a in agents]
                    supply_quantities_cleared_solution, demand_quantities_cleared_solution, m = marketClearing(agents,
                                                                                                               demand_curve)
                    SW_history[runs_per_strategy_update * epoch + run_of_strategy] = copy.deepcopy(m.ObjVal)
                    if payment_method == "uniform_pricing":
                        uniformPricing(agents, supply_quantities_cleared_solution, demand_quantities_cleared_solution, m, \
                                       epoch, runs_per_strategy_update, run_of_strategy)
                    if payment_method == "VCG_nima_NoCost":
                        VCG_nima_NoCost(agents, demand_curve, m, supply_quantities_cleared_solution, epoch, \
                                        runs_per_strategy_update, run_of_strategy)
                    if payment_method == "VCG_nima":
                        VCG_nima(agents, demand_curve, m, supply_quantities_cleared_solution, epoch, \
                                 runs_per_strategy_update, run_of_strategy)
                # Update strategy position
                for agent in agents:
                    agent.epoch_payoff_history[epoch] = \
                        statistics.mean(agent.payoff_history[runs_per_strategy_update*epoch:runs_per_strategy_update*(epoch+1)])
                PSO(agents, max_epochs, epoch)
                epoch += 1

            SavingAgents(agents, SW_history, payment_method, max_epochs, runs_per_strategy_update, epochs_run)
    results = pandas_results(n_agents, payment_methods, max_epochs, runs_per_strategy_update, whole_epochs_runs)
    plotAgentsChanges2D_all(results)
        #plotSupplyDemand(agents, demand_curve, payment_method, epochs_run)
        #plotAgentChanges2D(agents, payment_method, epochs_run)
        #plotSW(SW_history, runs_per_strategy_update, payment_method, epochs_run)
        #plotPayoffs(agents, payment_method, runs_per_strategy_update, epochs_run)

