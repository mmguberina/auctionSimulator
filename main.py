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
    n_of_demand_bids = 5
    demand_curve = [[n_of_demand_bids - i, i] for i in range(n_of_demand_bids)]

    n_agents = 5
    agents = []
    for i in range(n_agents):
        agents.append(Agent("all_the_same"))

