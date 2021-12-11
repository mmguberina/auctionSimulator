from strategies import *

class Agent:
    """
     variables
     ================
     true_evaluation (curve)
     strategy
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
    """
    

    def __init__(self, initType):
        if initType == "all_the_same":
            self.true_evaluation = [[i, i] for i in range(5)]
            # if you want a mixed strategy,
            # create self.strategies = [strategy1, ...]
            # and a list of equal length denoting how probable each strategy should be.
            # don't forget that the probabilities must sum to 1!
            self.strategy = pureStrategy5PercentHigher
            self.bids_curve = self.strategy(self.true_evaluation)
        else:
            raise NotImplemented

    
   def updateStrategy(self):
        """
        update strategy parameters with PSO or something.
        that should probably be a standalone function tho
        """
        pass


