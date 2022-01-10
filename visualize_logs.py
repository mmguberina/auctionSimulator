"""
here write functions which generate graphs based on the logs of runs
"""

from visualising import *


n_agents = 5
payment_methods = ["uniform_pricing", "VCG_nima"]
max_epochs = 500
runs_per_strategy_update = 100
n_of_demand_bids = 20
experiment_ids = [0,1,5,6,7,10,11,12,15,16,17,20,21,22]

saving_switch = 0
results, results_SW = pandas_results(n_agents, payment_methods, max_epochs, runs_per_strategy_update, experiment_ids)

plotAgentsChanges2D_all(results, saving_switch)

#plotAgentsChanges2D_all_histogram(results)

#plotSW_all(results_SW)

#plotPayoffs_all(results)
