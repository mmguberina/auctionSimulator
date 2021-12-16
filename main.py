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

# let's start with the following
# 1 buyer, static demand curve
# same form as bid curve
if __name__ == "__main__":
    # Create demand curve
    n_of_demand_bids = 5
    # only 1 buyer

    demand_curve = [[5, n_of_demand_bids - i] for i in range(n_of_demand_bids)]
    demand_curve[2][1] += 0.01

    # Initialize agents
    n_agents = 5
    agents = []
    for i in range(n_agents):
        agents.append(Agent("all_the_same"))

    # Might want to move this to the runs.py file,
    runs_per_strategy_update = 200   # Example for simple strategies
                                    # (might need other criteria with more complex strategies)
    # define termination criteria
    max_epochs = 200 # Just for testing with simple termination criteria
    epoch = 0
    while epoch < max_epochs:
        epoch += 1
        # Run runs_per_strategy_update times
        for i in range(runs_per_strategy_update):
            for a in agents:
                a.generateBid()

            # Market clearing function
            #supply_bids = [a.bids_curve for a in agents]
            supply_quantities_cleared_solution, demand_quantities_cleared_solution,m = marketClearing(agents, demand_curve)
            uniformPricing(agents, supply_quantities_cleared_solution, demand_quantities_cleared_solution,m)

        # Update strategy position
        PSO(agents)


    #plot1AgentChanges(agents[0])
    plotSupplyDemand(agents,demand_curve)
    plotAgentsChanges2D(agents)
    plotAgentChanges2D(agents)

