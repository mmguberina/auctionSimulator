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

# Create demand curve
n_of_demand_bids = 5
demand_curve = [[n_of_demand_bids - i, i] for i in range(n_of_demand_bids)]

# Initialize agents
n_agents = 5
agents = []
agent_utility = []
def test_util_func(agent):
    return -pow((agent.strategy_mix[0]-0.5),2) - pow((agent.strategy_mix[1]-0.5),2)

for i in range(n_agents):
    agents.append(Agent("all_the_same"))
    agents[i].strategy_mix = [random.random()*5,random.random()*5]
    agents[i].best_strategy = copy.deepcopy(agents[i].strategy_mix)
    agents[i].best_utility = test_util_func(agents[i])
    agent_utility.append(test_util_func(agents[i]))

for i in range(20):
    agent_utility = [test_util_func(a) for a in agents]
    PSO(agents,agent_utility)
    print(max(agent_utility))
    print(agents[agent_utility.index(max(agent_utility))].strategy_mix)