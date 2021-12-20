from strategies import *
import random


class Agent:
    """
     variables
     ================
     true_evaluation (curve)
     strategy
     strategy_mix
     best_strategy (For PSO)
     (velocity might be nice for the PSO)
     strategy_parameters
     bids_curve
     payoff_history
     strategy_mix_history

     functions
     ================
     generateBid - changes bid variable based on current strategy


     descriptions of variables
     =========================
     true_evaluation
     ---------------
     - list of truthful bids
     - a bid i is [quantity_i, price_i]
     - when combining these in the aggregate supply, you'll need to index each of these

     strategy
     --------
     - function pointer to strategy function to be called to create a bid

     strategy_mix
     --------
     - List of probabilities of choosing each strategy

     best_strategy
     --------
     - Mix of strategies that gave the greatest utility that agent has had

     best_utility
     --------
     - The top utility for updating best_strategy
     
     payoff_history
     ---------------
     - a list of every payoff ever

     strategy_mix_history
     ----------------
     - history of strategy mixes
    """

    def __init__(self, initType, init_strategy_mix, agent_num, max_epochs, runs_per_strategy_update):
        if initType == "all_the_same":
            self.true_evaluation = [[1, i+1] for i in range(5)]
            # if you want a mixed strategy,
            # create self.strategies = [strategy1, ...]
            # and a list of equal length denoting how probable each strategy should be.
            # don't forget that the probabilities must sum to 1!
            self.strategy = [pureStrategy5PercentHigher, pureStrategy15PercentHigher, pureStrategyBidTruthfully]
            #if it is the first payment method, we randomize the initial point
            if len(init_strategy_mix) != 5:
                #rand_init = [random.randint(0, 10) for i in range(len(self.strategy))]
                #self.strategy_mix = [p / sum(rand_init) for p in rand_init]
                self.strategy_mix = np.random.dirichlet(np.ones(len(self.strategy))/2,size=1)[0]

            else:
                #if it is not the first payment method, we used the initial strategy mix that was used in the first
                    #payment method
                self.strategy_mix = copy.deepcopy(init_strategy_mix[agent_num])
            self.best_strategy = copy.deepcopy(self.strategy_mix)
            self.best_utility = 0
            self.bids_curve = self.strategy[0](self.true_evaluation)
            self.payoff_history = [None]*(max_epochs * runs_per_strategy_update + 1)
            #self.strategy_mix_history = [copy.deepcopy(self.strategy_mix)]
            self.strategy_mix_history = [None]*(max_epochs+1)
            self.strategy_mix_history[0] = copy.deepcopy(self.strategy_mix)
        else:
            raise NotImplementedError("Not all the same not implemented")


    def generateBid(self):
        chosen_strategy = random.choices(self.strategy, weights=self.strategy_mix,k=1)[0]
        self.bids_curve = chosen_strategy(self.true_evaluation)

def updateStrategy(self):
    """
    update strategy parameters with PSO or something.
    that should probably be a standalone function tho
    --------
    Thinking this function should be for the internal strategy updates.
    PSO/EA takes care of agents positioning among the "pure strategies".
    """
    pass
