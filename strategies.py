"""
define strategies here
each agent can have his/her paramenters for the specific strategy
here are then the functions which generate bids based on:
    - those parameters
    - true evaluation 
and then return the bidding curve
"""
import copy
import random
import numpy as np


# pure strategies
def pureStrategy5PercentHigher(true_evaluation):
    bids_curve = copy.deepcopy(truthful_bid)
    for bid in bids_curve:
        bid[1] *= 1.05
    return bids_curve


def pureStrategy10PercentHigher(true_evaluation):
    bids_curve = copy.deepcopy(truthful_bid)
    for bid in bids_curve:
        bid[1] *= 1.10
    return bids_curve

def pureStrategy15PercentHigher(true_evaluation):
    bids_curve = copy.deepcopy(truthful_bid)
    for bid in bids_curve:
        bid[1] *= 1.10
    return bids_curve

def pureStrategyBidTruthfully(true_evaluation):
    return copy.deepcopy(true_evaluation)


# probabilistic mix of pure strategies
# input sorted probabilities so that the numbers
# so that this can be interpreted as an interval
# and an outcome can be selected by sampling once from a uniform distribution
def mixPureStrategies(strategies, probabilities):
    """
    roll the dice depending on the probability of mixing
    call the selected pure strategy function 
    return the bid
    """
    outcome = random.uniform(0, 1)
    cdf = np.cumsum(np.array(probabilities))
    past_p = 0
    for i in range(len(probabilities)):
        if past_p < outcome and cdf[i] > outcome:
            return strategies[i]
        past_p = cdf[i]




def compicatedStrategy1():
    pass


