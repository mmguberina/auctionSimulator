import copy

import numpy as np
import matplotlib.pyplot as plt
import math
import pickle
import os
import pandas as pd

def pandas_results(n_agents, payment_methods, max_epochs, runs_per_strategy_update, whole_epochs_runs):
    column_names = ["s_mix","s_mix_2D","payoff"]
    agents_list = [i for i in range (n_agents)]
    #whole_epochs_runs_list = [i for i in range(whole_epochs_runs)]
    epochs_list = [i for i in range(max_epochs)]

    #mapping coefficients: (x,y,z) --> (x_2D=ax1+by1+cz1, y_2D=dx1+ey1+fy1)
        # 6 eq. 6 unknowns --solution--> a=0,b=1,c=0.5 d=0,e=0,f=sin(pi/3)
    mapping_coeffs = np.array([[0,1,0.5],[0,0,math.sin(math.pi/3)]])

    #results for the agent data
    MultiIndex_obj = pd.MultiIndex.from_product([payment_methods,  whole_epochs_runs ,agents_list, epochs_list],\
                                                names=["payment_method", "n_whole_epoch","agent","epoch"])
    results = pd.DataFrame(np.empty((len(MultiIndex_obj), len(column_names))) * np.nan, \
                                   columns=column_names, index=MultiIndex_obj)
    #results for the SW
    column_names = ["SW"]
    MultiIndex_obj =pd.MultiIndex.from_product([payment_methods,  whole_epochs_runs , epochs_list],\
                                                names=["payment_method", "n_whole_epoch","epoch"])
    results_SW = pd.DataFrame(np.empty((len(MultiIndex_obj), len(column_names))) * np.nan, \
                                   columns=column_names, index=MultiIndex_obj)
    for payment_method in payment_methods:
        for epochs_run in whole_epochs_runs:
            with open('Results\\agents_' + payment_method + "_" + str(max_epochs) + \
                      "epochs_" + str(runs_per_strategy_update) + 'runs_EpochsRun' + str(epochs_run) + '.pkl', 'rb') as inp:
                agents = pickle.load(inp)
                SW_history = pickle.load (inp)

                np_SW_history = np.array(SW_history)
                results_SW.loc[(payment_method,epochs_run),"SW"] = \
                    np.average(np_SW_history.reshape(-1, runs_per_strategy_update), axis=1)
                for agent_number, agent in enumerate(agents):
                    results.loc[(payment_method,epochs_run,agent_number),"s_mix"] =\
                        pd.Series(agent.strategy_mix_history[:-1]).values
                    results.loc[(payment_method,epochs_run,agent_number),"payoff"] =\
                        np.average(np.array(agent.payoff_history).reshape(-1, runs_per_strategy_update), axis=1)
                    results.loc[(payment_method, epochs_run, agent_number), "s_mix_2D"] = \
                        pd.Series([np.matmul(strategy_mix, mapping_coeffs.T) for strategy_mix in
                          agent.strategy_mix_history[:-1]]).values
    return results, results_SW


def plotAgentsChanges2D_all(results,saving_switch):
    fig_size_all_agents_plot = (15,10)
    fig_size_each_agent_plot = (15,15)

    #all the agents with the same color_with final and start
    for payment_method in results.index.levels[0]:
        plt.figure(figsize=fig_size_all_agents_plot)
        for epochs_run in results.index.levels[1]:
            for agent_number in results.index.levels[2]:
                x = [i[0] for i in results.loc[(payment_method,epochs_run,agent_number),"s_mix_2D"]]
                y = [i[1] for i in results.loc[(payment_method,epochs_run,agent_number),"s_mix_2D"]]
                plt.scatter(x[0:1], y[0:1],marker="x", label="start", color="r",zorder=4,alpha=0.6)
                plt.scatter(x[-1], y[-1],marker="s", label="end", color= "r",zorder=4,alpha=0.6)
                plt.scatter(x, y, s=1, c=[i**2 for i in range(len(x))], cmap='Blues', alpha=0.6)
                #plt.plot(x,y,linewidth=1,label='agent ' + str(agent_number), alpha=0.6)
        # ploting the edges of the triangle
        plt.plot([0, 1], [0, 0], 'k', [0, 0.5], [0, math.sin(math.pi / 3)], 'k', [1, 0.5], [0, math.sin(math.pi / 3)],
                 'k')
        cbar = plt.colorbar()
        cbar.set_label('Epoch^2')
        plt.xlim(-0.1,1.1)
        plt.ylim(-0.1,math.sin(math.pi / 3)+0.1)
        plt.annotate("TT",(-0.03,0),annotation_clip=False)
        plt.annotate("+15%", (1.01, 0),annotation_clip=False)
        plt.annotate("PA",(0.5,math.sin(math.pi / 3)+0.03),annotation_clip=False)
        plt.title("Strategy mix for all agents, payment method: "+payment_method)
        plt.show()
        if saving_switch == 1:
            plt.savefig("Results\Results_nima_5000epochs_1000RunsPerStrategyUpdate\Plots\Strategy mix for all agents_"+payment_method+"_with initial and final strategies.pdf")
            plt.savefig(
                "Results\Results_nima_5000epochs_1000RunsPerStrategyUpdate\Plots\Strategy mix for all agents_" + payment_method + "_with initial and final strategies.png")
    # all the agents with the same color_withOUT final and start
    for payment_method in results.index.levels[0]:
        plt.figure(figsize=fig_size_all_agents_plot)
        for epochs_run in results.index.levels[1]:
            for agent_number in results.index.levels[2]:
                x = [i[0] for i in results.loc[(payment_method,epochs_run,agent_number),"s_mix_2D"]]
                y = [i[1] for i in results.loc[(payment_method,epochs_run,agent_number),"s_mix_2D"]]
                #plt.scatter(x[0:1], y[0:1],marker="x", label="start", color="r",zorder=4,alpha=0.6)
                #plt.scatter(x[-1], y[-1],marker="s", label="end", color= "r",zorder=4,alpha=0.6)
                plt.scatter(x, y, s=1, c=[i**2 for i in range(len(x))], cmap='Blues', alpha=0.6)
                #plt.plot(x,y,linewidth=1,label='agent ' + str(agent_number), alpha=0.6)
        # ploting the edges of the triangle
        plt.plot([0, 1], [0, 0], 'k', [0, 0.5], [0, math.sin(math.pi / 3)], 'k', [1, 0.5], [0, math.sin(math.pi / 3)],
                 'k')
        cbar = plt.colorbar()
        cbar.set_label('Epoch^2')
        plt.xlim(-0.1,1.1)
        plt.ylim(-0.1,math.sin(math.pi / 3)+0.1)
        plt.annotate("TT",(-0.03,0),annotation_clip=False)
        plt.annotate("+15%", (1.01, 0),annotation_clip=False)
        plt.annotate("PA",(0.5,math.sin(math.pi / 3)+0.03),annotation_clip=False)
        plt.title("Strategy mix for all agents, payment method: "+payment_method)
        plt.show()
        if saving_switch == 1:
            plt.savefig("Results\Results_nima_5000epochs_1000RunsPerStrategyUpdate\Plots\Strategy mix for all agents_"+payment_method+".pdf")
            plt.savefig(
                "Results\Results_nima_5000epochs_1000RunsPerStrategyUpdate\Plots\Strategy mix for all agents_" + payment_method + ".png")

    #different colors for different agents to check why some agents' payoff is less, with initial and final strategies
    cmap_list = [ 'Blues', 'Oranges', 'Greens', 'Reds',"Purples"]
    for payment_method in results.index.levels[0]:
        fig, axs = plt.subplots(3, 2,figsize=fig_size_each_agent_plot)
        for agent_number in results.index.levels[2]:
            for epochs_run in results.index.levels[1]:
                x = [i[0] for i in results.loc[(payment_method, epochs_run, agent_number), "s_mix_2D"]]
                y = [i[1] for i in results.loc[(payment_method, epochs_run, agent_number), "s_mix_2D"]]
                axs[agent_number//2,agent_number%2].scatter(x[0:1], y[0:1], marker="x", label="start", color="r",zorder=5)
                axs[agent_number//2,agent_number%2].scatter(x[-1], y[-1], marker="s", label="end", color="r",zorder=5)
                im = axs[agent_number//2,agent_number%2].scatter(x, y, s=1, c=[i**2 for i in range(len(x))], cmap=cmap_list[agent_number], alpha=0.6)
                # plt.plot(x,y,linewidth=1,label='agent ' + str(agent_number), alpha=0.6)
            # ploting the edges of the triangle
            axs[agent_number//2,agent_number%2].plot([0, 1], [0, 0], 'k', [0, 0.5], [0, math.sin(math.pi / 3)], 'k', [1, 0.5], [0, math.sin(math.pi / 3)],
                     'k')
            cbar = fig.colorbar(im,ax=axs[agent_number//2,agent_number%2])
            cbar.set_label('Epoch^2')
            axs[agent_number//2,agent_number%2].set_xlim([-0.1, 1.1])
            axs[agent_number//2,agent_number%2].set_ylim([-0.1, math.sin(math.pi / 3) + 0.1])
            axs[agent_number//2,agent_number%2].annotate("TT", (-0.03, 0), annotation_clip=False)
            axs[agent_number//2,agent_number%2].annotate("+15%", (1.01, 0), annotation_clip=False)
            axs[agent_number//2,agent_number%2].annotate("PA", (0.5, math.sin(math.pi / 3) + 0.03), annotation_clip=False)
            axs[agent_number//2,agent_number%2].set_title("Agent: "+str(agent_number))
        fig.suptitle("Strategy mix for each agent, payment method: "+payment_method)
        fig.delaxes(axs[2,1])
        if saving_switch == 1:
            plt.savefig("Results\Results_nima_5000epochs_1000RunsPerStrategyUpdate\Plots\Strategy mix for each agent_"+payment_method+"_with initial and final strategies.pdf")
            plt.savefig(
                "Results\Results_nima_5000epochs_1000RunsPerStrategyUpdate\Plots\Strategy mix for each agents_" + payment_method + "_with initial and final strategies.png")
        fig.show()

    #different colors for different agents to check why some agents' payoff is less, withOUT initial and final strategies
    cmap_list = [ 'Blues', 'Oranges', 'Greens', 'Reds',"Purples"]
    for payment_method in results.index.levels[0]:
        fig, axs = plt.subplots(3, 2,figsize=fig_size_each_agent_plot)
        for agent_number in results.index.levels[2]:
            for epochs_run in results.index.levels[1]:
                x = [i[0] for i in results.loc[(payment_method, epochs_run, agent_number), "s_mix_2D"]]
                y = [i[1] for i in results.loc[(payment_method, epochs_run, agent_number), "s_mix_2D"]]
                #axs[agent_number//2,agent_number%2].scatter(x[0:1], y[0:1], marker="x", label="start", color="r",zorder=5)
                #axs[agent_number//2,agent_number%2].scatter(x[-1], y[-1], marker="s", label="end", color="r",zorder=5)
                im = axs[agent_number//2,agent_number%2].scatter(x, y, s=1, c=[i**2 for i in range(len(x))], cmap=cmap_list[agent_number], alpha=0.6)
                # plt.plot(x,y,linewidth=1,label='agent ' + str(agent_number), alpha=0.6)
            # ploting the edges of the triangle
            axs[agent_number//2,agent_number%2].plot([0, 1], [0, 0], 'k', [0, 0.5], [0, math.sin(math.pi / 3)], 'k', [1, 0.5], [0, math.sin(math.pi / 3)],
                     'k')
            cbar = fig.colorbar(im,ax=axs[agent_number//2,agent_number%2])
            cbar.set_label('Epoch^2')
            axs[agent_number//2,agent_number%2].set_xlim([-0.1, 1.1])
            axs[agent_number//2,agent_number%2].set_ylim([-0.1, math.sin(math.pi / 3) + 0.1])
            axs[agent_number//2,agent_number%2].annotate("TT", (-0.03, 0), annotation_clip=False)
            axs[agent_number//2,agent_number%2].annotate("+15%", (1.01, 0), annotation_clip=False)
            axs[agent_number//2,agent_number%2].annotate("PA", (0.5, math.sin(math.pi / 3) + 0.03), annotation_clip=False)
            axs[agent_number//2,agent_number%2].set_title("Agent: "+str(agent_number))
        fig.suptitle("Strategy mix for each agent, payment method: "+payment_method)
        fig.delaxes(axs[2,1])
        if saving_switch == 1:
            plt.savefig("Results\Results_nima_5000epochs_1000RunsPerStrategyUpdate\Plots\Strategy mix for each agent_"+payment_method+".pdf")
            plt.savefig(
                "Results\Results_nima_5000epochs_1000RunsPerStrategyUpdate\Plots\Strategy mix for each agents_" + payment_method + ".png")
        fig.show()

    #Plotting only the lastest epochs
    '''
    epoch_range = np.arange(4999,5000)
    for payment_method in results.index.levels[0]:
        plt.figure()
        for epochs_run in results.index.levels[1]:
            x = [i[0] for i in results.loc[(payment_method,epochs_run,results.index.levels[2],epoch_range),"s_mix_2D"]]
            y = [i[1] for i in results.loc[(payment_method,epochs_run,results.index.levels[2],epoch_range),"s_mix_2D"]]
            plt.scatter(x,y,label='Run' + str(epochs_run))
            #plt.plot(x,y,linewidth=1,label='agent ' + str(agent_number), alpha=0.6)
        # ploting the edges of the triangle
        plt.plot([0, 1], [0, 0], 'k', [0, 0.5], [0, math.sin(math.pi / 3)], 'k', [1, 0.5], [0, math.sin(math.pi / 3)],
                 'k')
        plt.legend()
        plt.title(payment_method)
        plt.show()
    

    #mapping coefficients: (x,y,z) --> (x_2D=ax1+by1+cz1, y_2D=dx1+ey1+fy1)
        # 6 eq. 6 unknowns --solution--> a=0,b=1,c=0.5 d=0,e=0,f=sin(pi/3)
    mapping_coeffs = np.array([[0,1,0.5],[0,0,math.sin(math.pi/3)]])
    #strategy mix history in 2D
    colormaps=['Purples','Blues','Greens','Oranges','Reds','YlOrBr','PuBu']

    for payment_method in payment_methods:
        plt.figure()
        for epochs_run in range(whole_epochs_runs):
            # to load
            with open ('Results\\agents_'+payment_method+"_"+str(max_epochs)+\
                       "epochs_"+str(runs_per_strategy_update)+'runs_EpochsRun'+str(epochs_run)+'.pkl', 'rb') as inp:
                agents = pickle.load(inp)
            #SW_history = pickle.load(inp)
            strategy_mix_history_2D = np.array(
                [[list(np.matmul(strategy_mix, mapping_coeffs.T)) for strategy_mix in agent.strategy_mix_history] for
                 agent in agents])
            #for each agent plot the 2D, extract the x and y values and plot
            for agent_number in range(len(agents)):
                x = strategy_mix_history_2D[agent_number][:,0]
                y = strategy_mix_history_2D[agent_number][:,1]
                plt.scatter(x,y,s=1,label='agent ' + str(agent_number),c=[i for i in range(len(x))],cmap='Blues')
            #ploting the edges of the triangle
            plt.plot([0,1],[0,0],'k',[0,0.5],[0,math.sin(math.pi/3)],'k',[1,0.5],[0,math.sin(math.pi/3)],'k')
    plt.title(payment_method)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    cbar = plt.colorbar()
    cbar.set_label('Epoch')
    #plt.legend()
    plt.show()


def ContourPlotAgentsChanges2D_all(payment_methods, max_epochs, runs_per_strategy_update, whole_epochs_runs):
    import scipy.interpolate
    for payment_method in results.index.levels[0]:
        x = [i[0] for i in results.loc[(payment_method),"s_mix_2D"]]
        y = [i[1] for i in results.loc[(payment_method),"s_mix_2D"]]

        deltaX = (max(x) - min(x)) / 10
        deltaY = (max(y) - min(y)) / 10
        xmin = min(x) - deltaX
        xmax = max(x) + deltaXymin = min(y) - deltaY
        ymax = max(y) + deltaYprint(xmin, xmax, ymin, ymax)  # Create meshgrid
        xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]


    try:
        from astropy.convolution import Gaussian2DKernel, convolve
        astro_smooth = True
    except ImportError as IE:
        astro_smooth = False
    x = [i[0] for i in results.loc[:,"s_mix_2D"]]
    y = [i[1] for i in results.loc[:,"s_mix_2D"]]

    H, xedges, yedges = np.histogram2d(x, y, bins=(50, 40))
    xmesh, ymesh = np.meshgrid(xedges[:-1], yedges[:-1])

    # Smooth the contours (if astropy is installed)
    #if astro_smooth:
    #    kernel = Gaussian2DKernel(stddev=1.)
    #    H = convolve(H, kernel)


    fig, ax = plt.subplots(1, figsize=(7, 6))
    clevels = ax.contour(xmesh, ymesh, H.T, lw=.9, cmap='winter')  # ,zorder=90)

    # Identify points within contours
    p = clevels.collections[0].get_paths()
    inside = np.full_like(x, False, dtype=bool)
    for level in p:
        inside |= level.contains_points(zip(*(x, y)))

    ax.plot(x[~inside], y[~inside], 'kx')
    plt.show(block=False)
'''
def SavingAgents(agents,SW_history, payment_method, max_epochs, runs_per_strategy_update, epochs_run):
    if not os.path.exists('Results'):
        os.makedirs('Results')
    with open ('Results\\agents_'+payment_method+"_"+str(max_epochs)+\
               "epochs_"+str(runs_per_strategy_update)+'runs_EpochsRun'+str(epochs_run)+'.pkl', 'wb') as outp:
        pickle.dump(agents,outp)
        pickle.dump(SW_history, outp)



def plotSW_all(results_SW):
    #Plotting the average of SW over all the whole-epoch-runs
    results_SW_grouped_mean = results_SW.groupby(level=[0,2])["SW"].mean()
    plt.figure()
    for payment_method in results_SW.index.levels[0]:
        plt.plot(results_SW_grouped_mean.loc[payment_method],label=payment_method)
    plt.xlabel("Epoch")
    plt.ylabel("SW [SEK]")
    plt.legend()
    plt.title("Average of SW over the runs")
    plt.show()
"""
    # Plotting the SW for each whole-epoch-run
    for epochs_run in results_SW.index.levels[1]:
        plt.figure()
        for payment_method in results_SW.index.levels[0]:
                plt.plot(results_SW.loc[(payment_method, epochs_run), "SW"],label=payment_method)
        plt.xlabel("Epoch")
        plt.ylabel("SW [SEK]")
        plt.title("Run: "+str(epochs_run))
        plt.legend()
        plt.show()
"""

def plotPayoffs_all (results):

    results_grouped_mean = results.groupby(level=[0,2,3])["payoff"].mean()
    plt.figure()
    color_list= ['tab:blue', 'tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray',\
                 'tab:olive','tab:cyan']
    for payment_method in results.index.levels[0]:
        for agent_number in results.index.levels[2]:
            if payment_method == results.index.levels[0][0]:
                plt.plot(results_grouped_mean.loc[payment_method,agent_number], linewidth=0.8, linestyle="-",\
                         color=color_list[agent_number], label=payment_method+str(agent_number))
            else:
                plt.plot(results_grouped_mean.loc[payment_method, agent_number], linewidth=0.8, linestyle=":",\
                         color=color_list[agent_number], label=payment_method+str(agent_number))
    plt.xlabel("Epoch")
    plt.ylabel("Agents Payoff [SEK]")
    plt.legend()
    plt.title("Average of payoffs over the runs")
    plt.show()

'''
    for payment_method in results.index.levels[0]:
        for epochs_run in results.index.levels[1]:
            plt.figure()
            for agent_number in results.index.levels[2]:
                plt.plot(results.loc[(payment_method,epochs_run,agent_number),"payoff"],linewidth=1, label='agent ' + str(agent_number))
                plt.xscale('log')
                plt.xlabel("Epoch")
                plt.ylabel("Payoffs")
                plt.title(payment_method+"_"+str(epochs_run))
                #plt.legend()
'''

"""                
Plots the latest supply/demand curves
"""
def plotSupplyDemand(agents, demand_curve,payment_method):

    #t = range(sum([demand[1]] for demand in demand_curve))#np.arange(len(strategy_mix_history))
    plt.figure()
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
    plt.title(payment_method)
    plt.show(block=False)
"""__________________________________________________
-----------------------old plots
__________________________________________________


def plotSW(SW_history,runs_per_strategy_update,payment_method):
    np_SW_history = np.array(SW_history)
    epoch_SW_average = np.average(np_SW_history.reshape(-1, runs_per_strategy_update), axis=1)
    plt.figure()
    x = np.array([i for i in range(len(epoch_SW_average))])
    y = epoch_SW_average
    plt.plot(x,y)

    #model1 = np.poly1d(np.polyfit(x, y, 1))
    #model2 = np.poly1d(np.polyfit(x, y, 2))
    model3 = np.poly1d(np.polyfit(x, y, 3))
    #model4 = np.poly1d(np.polyfit(x, y, 4))
    #model5 = np.poly1d(np.polyfit(x, y, 5))

    #plt.plot(x, model1(x), color='green')
    #plt.plot(x, model2(x), color='red')
    plt.plot(x, model3(x), '--' , color='purple')
    #plt.plot(x, model4(x), color='blue')
    #plt.plot(x, model5(x), color='orange')

    plt.xlabel("epoch")
    plt.ylabel("Social welfare")
    plt.title(payment_method)

    plt.show()
def plotPayoffs (agents,payment_method,runs_per_strategy_update):
    plt.figure()
    for idx, agent in enumerate(agents):
        epoch_agent_payoff_average = np.average(np.array(agent.payoff_history).reshape(-1, runs_per_strategy_update), axis=1)
        plt.plot([i for i in range(len(epoch_agent_payoff_average))],epoch_agent_payoff_average, label='agent_' + str(idx))
        plt.xlabel("Epoch")
        plt.ylabel("Payoffs")
        plt.title(payment_method)
        plt.legend()
def plotAgentChanges2D(agents,payment_method):

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
    plt.title(payment_method)
    plt.legend()
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


#Not fully functional yet, seems the simple way with a rotational matrix didn't work out yet
#For now same as plotAgentsChange3D but with locked axis to 0-1 range

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