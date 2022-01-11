"""
here write functions which generate graphs based on the logs of runs
"""

from visualising import *
import pickle

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
    experiment_id = "003"
    parameters_file_name = "Results/Experiment_" + experiment_id + "/" + "parameters"
    parameters_file = open(parameters_file_name, 'rb')
    parameters = pickle.load(parameters_file)
    parameters_file.close()

    saving_switch = 0
    results, results_SW = pandas_results(parameters)
    plotAgentsChanges2D_all(experiment_id, results, saving_switch)
    plotSW_all(experiment_id, results_SW, saving_switch)
    plotPayoffs_all(experiment_id, results, saving_switch)
