"""
here write different runs
make that multiprocessed (marko's job)

you can call this from main
but first write a generic main for a single auction-market_clearing-mechanism_allocation iteration
"""


import statistics
from agents import *
from market_clearing import *
from mechanisms import *
from strategies import *
from strategy_updating_algorithms import *
from visualising import *
from os import getpid

# it's by far the simplest to paralelize over every epoch
# otherwise updating stuff gets complicated

# put results into the queue when done
def oneExperiment(task_queue, progress_queue):
    print("hello from worker process", getpid(), "! :)")

    while not task_queue.empty():
        experiment_id  = task_queue.get()

        payment_methods = ["uniform_pricing", "VCG_nima"]
        runs_per_strategy_update = 100
        max_epochs = 500 

        n_of_demand_bids = 20
        # only 1 buyer
        demand_curve = [[25 / n_of_demand_bids, i] for i in list(np.linspace(5, 1, num = n_of_demand_bids))]
        # Initialize agents
        n_agents = 5

        # we want to start with the same strategy mixes for each payment method
        strategy = [pureStrategyBidTruthfully, pureStrategy15PercentHigher, priceAdjusting]
        init_strategy_mixes = []
        for i in range(n_agents):
            init_strategy_mixes.append(list( # casting as list for proper initialization in agent class
                np.random.dirichlet(np.ones(len(strategy))/2,size=1)[0]))

        for payment_method in payment_methods:
            agents = []
            for i in range(n_agents):
                agents.append(Agent("all_the_same", max_epochs, runs_per_strategy_update, init_strategy_mix=init_strategy_mixes[i], strategy=strategy))
            
            # Social welfare history
            SW_history = [None] * max_epochs * runs_per_strategy_update

            epoch = 0

            while epoch < max_epochs:
                #print("[experiment_id , payment, epoch]: ", [experiment_id, payment_method, epoch])

                # Run runs_per_strategy_update times
                for run_of_strategy in range(runs_per_strategy_update):
                    for a in agents:
                        a.generateBid()

                    # Market clearing function
                    # supply_bids = [a.bids_curve for a in agents]
                    supply_quantities_cleared, objective_value, uniform_price = marketClearingSciPy(agents, demand_curve)
                    SW_history[runs_per_strategy_update * epoch + run_of_strategy] = objective_value
                    if payment_method == "uniform_pricing":
                        payoffs = uniformPricing(agents, supply_quantities_cleared, uniform_price)
                    if payment_method == "VCG_nima":
                        payoffs = VCG_nima(agents, demand_curve, supply_quantities_cleared, objective_value)
    #                    if payment_method == "VCG_nima_NoCost":
    #                        VCG_nima_NoCost(agents, demand_curve, m, supply_quantities_cleared, epoch, \
    #                                        runs_per_strategy_update, run_of_strategy)

                    for i, agent in enumerate(agents):
                        agent.payoff_history[runs_per_strategy_update * epoch + run_of_strategy] = payoffs[i]
                        if agent.last_strategy == 2:
                            agent.last_adjusting_payoff = payoffs[i]
                            
                # Update strategy position
                for agent in agents:
                    agent.epoch_payoff_history[epoch] = \
                        statistics.mean(agent.payoff_history[runs_per_strategy_update * epoch : runs_per_strategy_update*(epoch+1)])
                PSO(agents, max_epochs, epoch)
                epoch += 1

            SavingAgents(agents, SW_history, payment_method, max_epochs, runs_per_strategy_update, experiment_id)
            progress_queue.put(experiment_id)
    print("process", getpid(), "is done!")
