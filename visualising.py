import numpy as np
import matplotlib.pyplot as plt

def plot1AgentChanges(agent):
    strategy_mix_history = np.array(agent.strategy_mix_history)
    t = np.arange(len(strategy_mix_history))

    for i in range(len(strategy_mix_history[0])):
        plt.plot(t, strategy_mix_history[:, i])

    plt.show()



def plotAgentsChanges3D(agents):
    ax = plt.figure().add_subplot(projection='3d')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_zlim([0, 1])

    ax.set_xlabel("pureStrategy5PercentHigher")
    ax.set_ylabel("pureStrategy15PercentHigher")
    ax.set_zlabel("pureStrategyBidTruthfully")

    t = np.arange(len(agents[0].strategy_mix_history))
    for i, agent in enumerate(agents):
        strategy_mix_history = np.array(agent.strategy_mix_history)
        ax.plot(strategy_mix_history[:, 0], strategy_mix_history[:, 1], strategy_mix_history[:, 2], label='agent_' + str(i))


    plt.show()
