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
    """

    def __init__(self, initType):
        if initType == "all_the_same":
            self.true_evaluation = [[i, i] for i in range(5)]
            # if you want a mixed strategy,
            # create self.strategies = [strategy1, ...]
            # and a list of equal length denoting how probable each strategy should be.
            # don't forget that the probabilities must sum to 1!
            self.strategy = [pureStrategy5PercentHigher]
            self.strategy_mix = [1]
            self.best_strategy = copy.deepcopy(self.strategy_mix)
            self.best_utility = 0
            self.bids_curve = self.strategy[0](self.true_evaluation)
        else:
            raise NotImplementedError("Not all the same not implemented")


    def generateBid(self):
        chosen_strategy = random.choices(self.strategy,weights=self.strategy_mix,k=1)
        self.bids_curve = chosen_strategy(self.true_evaluation) #Not sure which is the nicer solution
        #return chosen_strategy(self.true_evaluation)

def updateStrategy(self):
    """
    update strategy parameters with PSO or something.
    that should probably be a standalone function tho
    --------
    Thinking this function should be for the internal strategy updates.
    PSO/EA takes care of agents positioning among the "pure strategies".
    """
    pass
