import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import statistics
import multiprocessing as mp
from progress.bar import Bar
from collections import namedtuple

from runs import *
from utils import *
from market_clearing import *
from visualize_logs import savePlotsForExperiment

if __name__ == "__main__":

    # collect all parameters into a single data structure
    # now it's easy to write it down into a file
    # and it nicely separates logic from parameters
    # so that you only need to change this file to create a run*
    # *up to a certain point, but close enough
    Parameters = namedtuple('Parameters', ['experiment_id', 'run_ids', 'payment_methods',
                                        'max_epochs', 'auctions_per_strategy_update',
                                        'demand_curve', 'n_agents', 'strategy'])
    # experiment_id denotes the entire experiment so that everything
    # related to it can be put in the same folder for easy access
    experiment_id = getExperimentID() # note: this is a string denoting the number
    payment_methods = ["uniform_pricing", "VCG_nima"]
    auctions_per_strategy_update = 10
    max_epochs = 5 

    # only 1 buyer
    n_of_demand_bids = 20
    demand_curve = [[25 / n_of_demand_bids, i] for i in list(np.linspace(5, 1, num = n_of_demand_bids))]

    n_agents = 5
    strategy = [pureStrategyBidTruthfully, pureStrategy15PercentHigher, priceAdjusting]

    run_ids = [0,1,5,6,7,10,11,12,15,16,17,20,21,22]

    parameters = Parameters(experiment_id, run_ids, payment_methods, max_epochs,
                        auctions_per_strategy_update, demand_curve, n_agents,
                        strategy)
    savingExperimentParameters(parameters)

    # each run is made into a single task for multiprocessing
    # because that's the easiest way to implement paralelisation in this case
    task_queue = mp.Queue()
    for task in run_ids:
        task_queue.put(task)
    
    progress_queue = mp.Queue()
    bar = Bar('Processing', max=len(run_ids) * len(payment_methods) + 1) # +1 for plot saving

    n_of_cores = mp.cpu_count()
    print("you have", n_of_cores, "cpu cores, select the number of processes you want - choose n <=", n_of_cores)
    n_of_processes = int(input("number of processes: "))

    processes = []
    for i in range(n_of_processes):
        processes.append(mp.Process(target=oneRun, args=(parameters, task_queue, progress_queue)))

    for p in processes:
        p.start()

    for i in range(len(run_ids) * len(payment_methods)):
        progress_queue.get()
        bar.next()
    
    for p in processes:
        p.join()

    savePlotsForExperiment(parameters) 
    bar.next()
    bar.finish()
