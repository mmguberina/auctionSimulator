import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt

from agents import *
from market_clearing import *
from mechanisms import *
from strategies import *
from strategy_updating_algorithms import *
from visualising import *

payment_methods = ["uniform_pricing","VCG_nima"]
# Might want to move this to the runs.py file,
runs_per_strategy_update = 200  # Example for simple strategies
# (might need other criteria with more complex strategies)
# define termination criteria
max_epochs = 30  # Just for testing with simple termination criteria



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

    # Initialize agents
    n_agents = 5
    init_strategy_mix = [] #the initial strategy mix for the agents
    for payment_method in payment_methods:
        agents = []
        for agent_num in range(n_agents):
            agents.append(Agent("all_the_same", init_strategy_mix, agent_num, max_epochs, runs_per_strategy_update))
            #if it is the first payment method, we save the init_strategy_mix to be used later as the initial for the
            #   other payment methods
            if len(init_strategy_mix) != 5:
                init_strategy_mix.append(agents[agent_num].strategy_mix)
        # Social welfare history
        SW_history = []

        epoch = 0
        while epoch < max_epochs:
            print(epoch)
            epoch += 1
            # Run runs_per_strategy_update times
            for run_of_strategy in range(runs_per_strategy_update):
                for a in agents:
                    a.generateBid()

                # Market clearing function
                # supply_bids = [a.bids_curve for a in agents]
                supply_quantities_cleared_solution, demand_quantities_cleared_solution, m = marketClearing(agents,
                                                                                                           demand_curve)
                SW_history.append(m.ObjVal)
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
            PSO(agents, max_epochs, epoch)

        plotSupplyDemand(agents, demand_curve, payment_method)
        plotAgentChanges2D(agents,payment_method)
        plotSW(SW_history, runs_per_strategy_update,payment_method)
        #plotPayoffs(agents,payment_method)