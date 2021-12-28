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
def pureStrategy5PercentHigher(agent):
    bids_curve = copy.deepcopy(agent.true_evaluation)
    for bid in bids_curve:
        bid[1] *= 1.05
    return bids_curve


def pureStrategy10PercentHigher(agent):
    bids_curve = copy.deepcopy(agent.true_evaluation)
    for bid in bids_curve:
        bid[1] *= 1.10
    return bids_curve

def pureStrategy15PercentHigher(agent):
    bids_curve = copy.deepcopy(agent.true_evaluation)
    for bid in bids_curve:
        bid[1] *= 1.15
    return bids_curve

def pureStrategyBidTruthfully(agent):
    return copy.deepcopy(agent.true_evaluation)

def priceAdjusting(agent):
    bids = copy.deepcopy(agent.bids_curve)
    if agent.last_adjusting_payoff is not None:
        last_payoff = agent.last_adjusting_payoff
        for i, bid in enumerate(bids):
            last_payoff -= bid
            if last_payoff >= 0:
                bids[i] *= 1.05
            else:
                bids[i] *= 0.95
    agent.last_adjusting_bid = copy.deepcopy(bids)
    return bids

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


