import random
import copy
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
def PSO(agents):
    epoch_utility = [agent.payoff_history[-1] for agent in agents]
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
        random_movement_vector = [random.random()*random_movement_weight for i in range(n_strategies)]

        #Update agent position (might want to change to update agent velocity)
        for i in range(n_strategies):
            a.strategy_mix[i] = a.strategy_mix[i] + swarm_best_vector[i] + agent_best_vector[i] + random_movement_vector[i]
        a.strategy_mix = [a.strategy_mix[i]/sum(a.strategy_mix) for i in range(n_strategies)]
        a.strategy_mix_history.append(copy.deepcopy(a.strategy_mix))


def GA():
    pass

def somethingElse():
    pass
