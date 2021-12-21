import random
import copy
import numpy as np
"""
here define algorithms which update the strategy parameters
based on the results of the previous run(s)

input:
    - agent objects (contains parameters of strategies for each agent)
    - payoffs of run(s)

output:
    - none, update agent's strategy parameters in place (in the object)
"""
# Assumes that the agent strategy_mix parameter is a list of probabilities of choosing each strategy
def PSO(agents, epoch_length, epoch):
    #epoch_utility = [agent.payoff_history[-1] for agent in agents]
    #epoch_utility = [sum(agent.payoff_history[-epoch_length:]) for agent in agents]
    epoch_utility = [sum(agent.payoff_history[:epoch]) for agent in agents]
    # Weight parameters for movement components (dividing by ten to start conservatively)
    swarm_best_weight = random.random()/10
    agent_best_weight = random.random()/10
    random_movement_weight = random.random()/10

    # Find top agents (might want to swap to swarm best known including history)
    max_util = max(epoch_utility)
    top_index = [i for i, j in enumerate(epoch_utility) if j == max_util]
    top_agents = [agents[i] for i in top_index]
    # Movement based on a few factors
    for idx, a in enumerate(agents):
        n_strategies = len(a.strategy_mix)
        # Swarm best (Average of vectors to agents with highest scores)
        swarm_best_vector = [0]*n_strategies
        for a_max in top_agents:
            swarm_best_vector = [swarm_best_vector[i]+a_max.strategy_mix[i]-a.strategy_mix[i] for i in range(n_strategies)]
        swarm_best_vector = [(swarm_best_vector[i]/len(top_agents)*swarm_best_weight) for i in range(n_strategies)]


        # Own best
        if a.best_utility < epoch_utility[idx]:
            a.best_strategy = copy.deepcopy(a.strategy_mix)
            a.best_utility = epoch_utility[idx]
        agent_best_vector = [(a.best_strategy[i] - a.strategy_mix[i]) for i in range(n_strategies)]
        agent_best_vector = [agent_best_vector[i]*agent_best_weight for i in range(n_strategies)]


        # Random movement
        random_movement_vector = [(random.random()*2-1)*random_movement_weight for i in range(n_strategies)]

        #Update agent position (might want to change to update agent velocity)
        strategy_mix = [0]*n_strategies
        for i in range(n_strategies):
            strategy_mix[i] = a.strategy_mix[i] + swarm_best_vector[i] + agent_best_vector[i] + random_movement_vector[i]
        strategy_mix = [strategy_mix[i]/sum(strategy_mix) for i in range(n_strategies)]

        # Project back on feasible set (specifically for n_strategies = 3)
        plane_points = [[1,0,0],[0,1,0],[0,0,1]]
        if n_strategies == 3:
            for i, mix in enumerate(strategy_mix):
                if mix < 0: # If negative project back onto the line
                    AB = np.subtract(plane_points[(i+2)%n_strategies],plane_points[(i+1)%n_strategies])
                    AS = np.subtract(strategy_mix,plane_points[(i+1)%n_strategies])
                    strategy_mix = np.add(plane_points[(i+1)%n_strategies], [np.dot(AS,AB)/np.dot(AB,AB)*ab for ab in AB])
                    break
            negative = [i for i, j in enumerate(strategy_mix) if j < 0]
            if len(negative) != 0:
                strategy_mix = [1,1,1]
                strategy_mix[i] = 0
                strategy_mix[negative[0]] = 0


        a.strategy_mix = copy.deepcopy(strategy_mix)
        #a.strategy_mix_history.append(copy.deepcopy(a.strategy_mix))
        a.strategy_mix_history[epoch+1] = a.strategy_mix

def GA():
    pass

def somethingElse():
    pass
