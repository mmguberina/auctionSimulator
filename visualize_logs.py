"""
here write functions which generate graphs based on the logs of runs
"""

from visualising import *
import pickle
from collections import namedtuple

def savePlotsForExperiment(parameters):
    experiment_id = parameters.experiment_id

    saving_switch = 1
    # TODO make this comply with this scheme
    results, results_SW = pandas_results(parameters)

    plotAgentsChanges2D_all(experiment_id, results, saving_switch)
    
    # TODO make these work and save themselves as well
    # also, save it in Results/Experiment_experiment_id/Plots

    #plotAgentsChanges2D_all_histogram(results)

    plotSW_all(experiment_id, results_SW, saving_switch)

    plotPayoffs_all(experiment_id, results, saving_switch)


if __name__ == "__main__":

    Parameters = namedtuple('Parameters', ['experiment_id', 'run_ids', 'payment_methods',
                                        'max_epochs', 'auctions_per_strategy_update',
                                        'demand_curve', 'n_agents', 'strategy'])
    experiment_id = "011"
    parameters_file_name = "Results/Experiment_" + experiment_id + "/" + "parameters.pkl"
    parameters_file = open(parameters_file_name, 'rb')
    parameters = pickle.load(parameters_file)
    parameters_file.close()

    saving_switch = 0
    results, results_SW = pandas_results(parameters)
    plotAgentsChanges2D_all(experiment_id, results, saving_switch)
    plotAgentsChanges2D_all_histogram(experiment_id, results, saving_switch)
    plotSW_all(experiment_id, results_SW, saving_switch)
    plotPayoffs_all(experiment_id, results, saving_switch)
