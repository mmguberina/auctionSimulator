# random functions moved here so that they don't polute the rest of the code
import subprocess
import re
from os import makedirs
import pickle

def getExperimentID():
    child = subprocess.Popen(['ls', './Results'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    result_folder_ls = child.stdout.read().decode('utf-8').split("\n")
    regex_name = re.compile("Experiment_.*")
    regex_number = re.compile("[0-9]+")

    max_index = 0
    for name in result_folder_ls:
        r1 = regex_name.search(name)
        if r1 != None:
            r2 = regex_number.search(r1.string)
            index = int(r2.group(0))
            if index > max_index:
                max_index = index

    str_prefix = ""
    if max_index < 10:
        str_prefix = "00"
    else:
        if max_index < 100:
            str_prefix = "0"

    return str_prefix + str(max_index + 1)


def savingExperimentParameters(parameters):
    experiment_id = parameters.experiment_id
    dir_string_plots = "Results/Experiment_" + experiment_id + "/Plots"
    dir_string = "Results/Experiment_" + experiment_id + "/"
    makedirs(dir_string_plots) # will make all necessary dies

    parameters_file = open(dir_string + "parameters.pkl", 'wb')
    pickle.dump(parameters, parameters_file)
    parameters_file.close()



def SavingAgents(experiment_id, agents, SW_history, payment_method, max_epochs, auctions_per_strategy_update, epochs_run, parameters):
    
    dir_string = "Results/Experiment_" + experiment_id + "/"

    with open (dir_string + 'agents_'+payment_method+"_"+str(max_epochs)+\
               "epochs_"+str(auctions_per_strategy_update)+'runs_EpochsRun'+str(epochs_run)+'.pkl', 'wb') as outp:
        pickle.dump(agents, outp)
        pickle.dump(SW_history, outp)
