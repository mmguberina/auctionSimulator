import pickle
import numpy as np
import matplotlib.pyplot as plt
from agents import *
from market_clearing import *
from mechanisms import *
from strategies import *
from collections import namedtuple


def deviationFromTruth(true_evaluation, bids_curve):
    percentage_of_lying_per_bid = []
    for i in range(len(true_evaluation)):
        percentage_of_lying_per_bid.append(100 * (1 - true_evaluation[i][1] / bids_curve[i][1]))
# altervative 1 - (true_evaluation[i][0] * true_evaluation[i][1]) / (bids_curve[i][0] * bids_curve[i][1])))
# if you want to include quantities as well
    return percentage_of_lying_per_bid


def plotTruthfulnessResults(results, experiment_id, n_agents, clearing_criterion, show_plot_flag):
    fig, ax = plt.subplots()
    labels = [str(i) for i in range(n_agents)]
    x = np.arange(n_agents)
    rects = []
    width = 0.35
    for i, payment_method in enumerate(results):
        #if yerr_flag:
        #    rects.append(ax.bar(x - width/2 + i * width, results[payment_method]['means'],
        #        width, yerr=results[payment_method]['stds'], label=payment_method))
        #else:
        rects.append(ax.bar(x - width/2 + i * width, results[payment_method]['means'],
            width, label=payment_method))

    ax.set_ylabel('Percentage of deviating from truth')
    if clearing_criterion:
        ax.set_title('Average percentage of deviating from truth in cleared bids for every agent')
    else:
        ax.set_title('Average percentage of deviating from truth for every agent')
    ax.set_xticks(x, labels)
    ax.legend()

    fig.tight_layout()
#    if yerr_flag:
#        plt.savefig("Results/Experiment_" + experiment_id + "/Plots/Truthfulness_final_with_stderr.png")
#    else:
    help_str = ""
    if clearing_criterion:
        help_str = "in_cleared"
    plt.savefig("Results/Experiment_" + experiment_id + "/Plots/Truthfulness_" + help_str + "final_no_stderr.png")
    plt.savefig("Results/Experiment_" + experiment_id + "/Plots/Truthfulness_" + help_str + "final_no_stderr.pdf")
    if show_plot_flag:
        plt.show()



def plotTruthfulnessAnalysis(parameters, show_plot_flag):
    Parameters = namedtuple('Parameters', ['experiment_id', 'run_ids', 'payment_methods',
                                        'max_epochs', 'auctions_per_strategy_update',
                                        'demand_curve', 'n_agents', 'strategy'])

    experiment_id = parameters.experiment_id
    run_ids = parameters.run_ids
    #epochs_run = run_ids[-1]
    payment_methods = parameters.payment_methods
    max_epochs = parameters.max_epochs
    auctions_per_strategy_update = parameters.auctions_per_strategy_update
    n_agents = parameters.n_agents
    demand_curve = parameters.demand_curve


    dir_string = "Results/Experiment_" + experiment_id + "/"
    # now run the agents and check what they actually bid
    n_of_auctions = 1000
    results = {payment_method : {'means' : [], 'stds' : []} for payment_method in payment_methods}
    results_only_cleared = {payment_method : {'means' : [], 'stds' : []} for payment_method in payment_methods}

    for payment_method in payment_methods:
        agents_percentage_of_lying_per_bid = {i : [] for i in range(n_agents)}
        agents_percentage_of_lying_per_cleared_bid = {i : [] for i in range(n_agents)}
        for epochs_run in run_ids:
            with open(dir_string + 'agents_' + payment_method + "_" + str(max_epochs) + \
                      "epochs_" + str(auctions_per_strategy_update) + 'runs_EpochsRun' + str(epochs_run) + '.pkl', 'rb') as inp:
                agents = pickle.load(inp)
                SW_history = pickle.load (inp)

            for auction_id in range(n_of_auctions):
                for i, agent in enumerate(agents):
                    agent.generateBid()
                    # we're adding a list to a list, resulting in a longer list
                    agents_percentage_of_lying_per_bid[i] += \
                            deviationFromTruth(agent.true_evaluation, agent.bids_curve)

                # Market clearing function
                # supply_bids = [a.bids_curve for a in agents]
                supply_quantities_cleared, objective_value, uniform_price = marketClearingSciPy(agents, demand_curve)
                # in supply_quantities_cleared:
                # [..., [quantity_i, price_i, cleared_amount, [agent_index, bid_index]],...]
                for i, agent in enumerate(agents):
                    only_cleared_bids_i = [[bid[0], bid[1]] for bid in supply_quantities_cleared if bid[2] > 0 and bid[3][0] == i]
                    corresponding_true_evaluations_i = [agent.true_evaluation[bid[3][1]] for bid in supply_quantities_cleared if bid[2] > 0 and bid[3][0] == i]
#                    exit()
                    agents_percentage_of_lying_per_cleared_bid[i] += \
                            deviationFromTruth(corresponding_true_evaluations_i, only_cleared_bids_i)


                if payment_method == "uniform_pricing":
                    payoffs = uniformPricing(agents, supply_quantities_cleared, uniform_price)
                if payment_method == "VCG_nima":
                    payoffs = VCG_nima(agents, demand_curve, supply_quantities_cleared, objective_value)

                for i, agent in enumerate(agents):
                    if agent.last_strategy == 2:
                        agent.last_adjusting_payoff = payoffs[i]

        for i in agents_percentage_of_lying_per_bid:
            agents_percentage_of_lying_per_bid[i] = np.array(agents_percentage_of_lying_per_bid[i])
            results[payment_method]['means'].append(np.mean(agents_percentage_of_lying_per_bid[i]))
            results[payment_method]['stds'].append(np.std(agents_percentage_of_lying_per_bid[i]))

            agents_percentage_of_lying_per_cleared_bid[i] = np.array(agents_percentage_of_lying_per_cleared_bid[i])
            results_only_cleared[payment_method]['means'].append(np.mean(agents_percentage_of_lying_per_cleared_bid[i]))
            results_only_cleared[payment_method]['stds'].append(np.std(agents_percentage_of_lying_per_cleared_bid[i]))

            
    for payment_method in payment_methods:
        results[payment_method]['means'] = np.array(results[payment_method]['means'])
        results[payment_method]['stds'] = np.array(results[payment_method]['stds'])
        results_only_cleared[payment_method]['means'] = np.array(results_only_cleared[payment_method]['means'])
        results_only_cleared[payment_method]['stds'] = np.array(results_only_cleared[payment_method]['stds'])

    plotTruthfulnessResults(results, experiment_id, n_agents, False, show_plot_flag)
    plotTruthfulnessResults(results_only_cleared, experiment_id, n_agents, True, show_plot_flag)







if __name__ == "__main__":

    Parameters = namedtuple('Parameters', ['experiment_id', 'run_ids', 'payment_methods',
                                        'max_epochs', 'auctions_per_strategy_update',
                                        'demand_curve', 'n_agents', 'strategy'])
    experiment_id = "029"
    parameters_file_name = "Results/Experiment_" + experiment_id + "/" + "parameters.pkl"
    parameters_file = open(parameters_file_name, 'rb')
    parameters = pickle.load(parameters_file)
    parameters_file.close()

    show_plot_flag = True

    plotTruthfulnessAnalysis(parameters, show_plot_flag)


#    experiment_id = parameters.experiment_id
#    run_ids = parameters.run_ids
#    #epochs_run = run_ids[-1]
#    payment_methods = parameters.payment_methods
#    max_epochs = parameters.max_epochs
#    auctions_per_strategy_update = parameters.auctions_per_strategy_update
#    n_agents = parameters.n_agents
#    demand_curve = parameters.demand_curve



    #plotTruthfulnessAnalysis(False)
    #plotTruthfulnessAnalysis(False)
    #plt.show()
    #plotTruthfulnessAnalysis(True)
    #plt.show()




