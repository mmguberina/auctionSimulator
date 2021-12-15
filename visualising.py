import numpy as np
import matplotlib.pyplot as plt
import math

def plotAgentChanges2D(agents):

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

"""
Not fully functional yet, seems the simple way with a rotational matrix didn't work out yet
For now same as plotAgentsChange3D but with locked axis to 0-1 range
"""
def plotAgentsChanges2D(agents):


    R = np.array([[1/12*(6 + 2*np.sqrt(3)) , 1/12*(2*np.sqrt(3) - 6) , -1/np.sqrt(3)],
                    [1/12*(2*np.sqrt(3) - 6), 1/12*(6 + 2*np.sqrt(3)), -1/np.sqrt(3)],
                    [1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)]])

    ax = plt.figure().add_subplot(projection='3d')

    t = np.arange(len(agents[0].strategy_mix_history))
    for i, agent in enumerate(agents):
        strategy_mix_history = np.array(agent.strategy_mix_history)
        #strategy_mix_history = np.array([np.matmul(A,R) for A in strategy_mix_history])
        ax.plot(strategy_mix_history[:, 0], strategy_mix_history[:, 1], strategy_mix_history[:, 2], label='agent_' + str(i))

    ax.set_xlim(xmin=0,xmax=1)
    ax.set_ylim(ymin=0,ymax=1)
    ax.set_zlim(zmin=0,zmax=1)
    plt.show()


"""
Plots the latest supply/demand curves
"""
def plotSupplyDemand(agents, demand_curve):

    #t = range(sum([demand[1]] for demand in demand_curve))#np.arange(len(strategy_mix_history))
    tmp = 0
    x_points_demand = []
    y_points_demand = []
    for demand in demand_curve:
        x_points_demand.append(tmp)
        y_points_demand.append((demand[1]))
        tmp += demand[0]
        x_points_demand.append(tmp)
        y_points_demand.append(demand[1])

    plt.plot(x_points_demand, y_points_demand, label='Demand')

    all_bids = []
    for agent in agents:
        for bid in agent.bids_curve:
            # now bid is [quantity_i, price_i,]
            all_bids.append(bid)

    sort_by_price = lambda solution : solution[1]
    all_bids = sorted(all_bids, key=sort_by_price)
    x_points_supply = []
    y_points_supply = []
    tmp = 0
    for bid in all_bids:
        x_points_supply.append(tmp)
        y_points_supply.append(bid[1])
        tmp += bid[0]
        x_points_supply.append(tmp)
        y_points_supply.append(bid[1])

    plt.plot(x_points_supply, y_points_supply, label='Supply')

    plt.show()