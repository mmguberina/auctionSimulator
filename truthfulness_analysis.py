import pickle
import numpy as np
import matplotlib.pyplot as plt
from agents import *
from market_clearing import *
from mechanisms import *
from strategies import *
from collections import namedtuple

Parameters = namedtuple('Parameters', ['experiment_id', 'run_ids', 'payment_methods',
                                    'max_epochs', 'auctions_per_strategy_update',
                                    'demand_curve', 'n_agents', 'strategy'])
experiment_id = "011"
parameters_file_name = "Results/Experiment_" + experiment_id + "/" + "parameters.pkl"
parameters_file = open(parameters_file_name, 'rb')
parameters = pickle.load(parameters_file)
parameters_file.close()


experiment_id = parameters.experiment_id
run_ids = parameters.run_ids
#epochs_run = run_ids[-1]
payment_methods = parameters.payment_methods
max_epochs = parameters.max_epochs
auctions_per_strategy_update = parameters.auctions_per_strategy_update
n_agents = parameters.n_agents
demand_curve = parameters.demand_curve




def deviationFromTruth(agent):
    percentage_of_lying_per_bid = []
    for i in range(len(agent.true_evaluation)):
        percentage_of_lying_per_bid.append(100 * (1 - agent.true_evaluation[i][1] / agent.bids_curve[i][1]))
    return percentage_of_lying_per_bid


dir_string = "Results/Experiment_" + experiment_id + "/"
# now run the agents and check what they actually bid
n_of_auctions = 1000
results = {payment_method : {'means' : [], 'stds' : []} for payment_method in payment_methods}

for payment_method in payment_methods:
    agents_percentage_of_lying_per_bid = {i : [] for i in range(n_agents)}
    for epochs_run in run_ids:
        with open(dir_string + 'agents_' + payment_method + "_" + str(max_epochs) + \
                  "epochs_" + str(auctions_per_strategy_update) + 'runs_EpochsRun' + str(epochs_run) + '.pkl', 'rb') as inp:
            agents = pickle.load(inp)
            SW_history = pickle.load (inp)

        for auction_id in range(n_of_auctions):
            for i, agent in enumerate(agents):
                agent.generateBid()
                agents_percentage_of_lying_per_bid[i].append(deviationFromTruth(agent))

            # Market clearing function
            # supply_bids = [a.bids_curve for a in agents]
            supply_quantities_cleared, objective_value, uniform_price = marketClearingSciPy(agents, demand_curve)

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
        
for payment_method in payment_methods:
    results[payment_method]['means'] = np.array(results[payment_method]['means'])
    results[payment_method]['stds'] = np.array(results[payment_method]['stds'])

def plotTruthfulnessAnalysis(yerr_flag):
    fig, ax = plt.subplots()
    labels = [str(i) for i in range(n_agents)]
    x = np.arange(n_agents)
    rects = []
    width = 0.35
    for i, payment_method in enumerate(payment_methods):
        if yerr_flag:
            rects.append(ax.bar(x - width/2 + i * width, results[payment_method]['means'],
                width, yerr=results[payment_method]['stds'], label=payment_method))
        else:
            rects.append(ax.bar(x - width/2 + i * width, results[payment_method]['means'],
                width, label=payment_method))

    ax.set_ylabel('Percentage of deviating from truth')
    ax.set_title('Average percentage of deviating from truth for every agent')
    ax.set_xticks(x, labels)
    ax.legend()

    fig.tight_layout()
    if yerr_flag:
        plt.savefig("Results/Experiment_" + experiment_id + "/Plots/Truthfulness_final_with_stderr.png")
    else:
        plt.savefig("Results/Experiment_" + experiment_id + "/Plots/Truthfulness_final_no_stderr.png")
#    plt.show()

plotTruthfulnessAnalysis(False)
plotTruthfulnessAnalysis(True)
