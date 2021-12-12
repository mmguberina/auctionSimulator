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

# let's start with the following
# 1 buyer, static demand curve
# same form as bid curve
if __name__ == "__main__":
    # Create demand curve
    n_of_demand_bids = 5
    demand_curve = [[n_of_demand_bids - i, i] for i in range(n_of_demand_bids)]

    # Initialize agents
    n_agents = 5
    agents = []
    for i in range(n_agents):
        agents.append(Agent("all_the_same"))

    # Might want to move this to the runs.py file,
    runs_per_strategy_update = 10   # Example for simple strategies
                                    # (might need other criteria with more complex strategies)
    # define termination criteria
    max_epochs = 100 # Just for testing with simple termination criteria
    epoch = 0
    while epoch < max_epochs:
        epoch += 1
        # Run runs_per_strategy_update times
        for i in range(runs_per_strategy_update):
            for a in agents:
                a.generateBid()
            # Market clearing function
            supply_bids = [a.bids_curve for a in agents]
            m,x = primal_multibid(demand_curve,supply_bids)
            #cleared_bids_demand, cleared_bids_supply = clearing_results_processing_multibid(m,x,demand_curve,supply_bids,...)
            # Payment distribution
            #price_clearing_results = uniformPricing(m,,demand_curve,supply_bids,epoch,cleared_bids_demand,cleared_bids_supply)
            # Accumulate payments/utility

        # Update strategy position
        #PSO(agents,price_clearing_results)