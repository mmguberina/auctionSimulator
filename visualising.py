import numpy as np
import matplotlib.pyplot as plt
import math

def plot1AgentChanges(agents):

    #mapping coefficients: (x,y,z) --> (x_2D=ax1+by1+cz1, y_2D=dx1+ey1+fy1)
        # 6 eq. 6 unknowns --solution--> a=0,b=1,c=0.5 d=0,e=0,f=sin(pi/3)
    mapping_coeffs = np.array([[0,1,0.5],[0,0,math.sin(math.pi/3)]])
    #strategy mix history in 2D
    strategy_mix_history_2D = np.array([[list(np.matmul(strategy_mix,mapping_coeffs.T)) for strategy_mix in agent.strategy_mix_history] for agent in agents])

    plt.figure()
    #for each agent plot the 2D, extract the x and y values and plot
    for agent_number in range(len(agents)):
        x = strategy_mix_history_2D[agent_number][:,0]
        y = strategy_mix_history_2D[agent_number][:,1]
        plt.plot(x,y, label='agent_' + str(agent_number))
    #ploting the edges of the triangle
    plt.plot([0,1],[0,0],'k',[0,0.5],[0,math.sin(math.pi/3)],'k',[1,0.5],[0,math.sin(math.pi/3)],'k')
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
