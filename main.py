import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import statistics
import multiprocessing as mp
from progress.bar import Bar

from runs import *


# let's start with the following
# 1 buyer, static demand curve
# same form as bid curve
if __name__ == "__main__":
    experiment_ids = [0,1,5,6,7,10,11,12,15,16,17,20,21,22]

    task_queue = mp.Queue()
    for task in experiment_ids:
        task_queue.put(task)
    
    progress_queue = mp.Queue()
    bar = Bar('Processing', max=len(experiment_ids) * 2) # times 2 for payment methods

    n_of_cores = mp.cpu_count()
    print("you have", n_of_cores, "cpu cores, select the number of processes you want - choose n <=", n_of_cores)
    n_of_processes = int(input("number of processes: "))

    processes = []
    for i in range(n_of_processes):
        processes.append(mp.Process(target=oneExperiment, args=(task_queue, progress_queue)))

    for p in processes:
        p.start()

    for i in range(len(experiment_ids) * 2):
        progress_queue.get()
        bar.next()
    bar.finish()
    
    for p in processes:
        p.join()




    #results, results_SW = pandas_results(n_agents, payment_methods, max_epochs, runs_per_strategy_update, whole_epochs_runs)
    #plotAgentsChanges2D_all(results,plot_saving_switch)
        #plotSupplyDemand(agents, demand_curve, payment_method, epochs_run)
        #plotAgentChanges2D(agents, payment_method, epochs_run)
        #plotSW(SW_history, runs_per_strategy_update, payment_method, epochs_run)
        #plotPayoffs(agents, payment_method, runs_per_strategy_update, epochs_run)

