import random

import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt

"""
take bids and the demand
calculate the price
(calculate marginal contributions for each agent)
calculate social welfare
return final price(s) and social welfare
"""



def clearing_results_processing_multibid(m,x,bids_demand,bids_supply,path_cleared_bid_demand,path_cleared_bid_supply):

    #-----finding the cleared bids (the ones having x!=0, i.e., x>=0)
    cleared_bids_indices_demand = [idx for idx in bids_demand.index \
                                   if x[idx].x != 0]
    cleared_bids_indices_supply = [idx for idx in bids_supply.index \
                                   if x[idx].x != 0]

    cleared_bids_demand = bids_demand.loc[(cleared_bids_indices_demand)]
    cleared_bids_demand['x'] = [x[idx].x for idx in cleared_bids_demand.index]
    cleared_bids_demand.to_excel(path_cleared_bid_demand)

    cleared_bids_supply = bids_supply.loc[(cleared_bids_indices_supply)]
    cleared_bids_supply['x'] = [x[idx].x for idx in cleared_bids_supply.index]
    cleared_bids_supply.to_excel(path_cleared_bid_supply)

    #Set of T includes only the hours in this day that are cleared so that we don't have to plot all the unnessary plots fo
    T = list(cleared_bids_demand.groupby(level=[1]).groups.keys()) # the set cleared of hours

    for t in T:
        #sorting the cleared bids in each hour
        bids_demand_sorted_t = bids_demand.loc[bids_demand.groupby(level=[1]).groups[t]].sort_values(by=['u'], ascending=False)
        bids_supply_sorted_t = bids_supply.loc[bids_supply.groupby(level=[1]).groups[t]].sort_values(by=['u'])

        #aggregated the sorted bids to be able to plot a stacked line plot for the supply and demand curves
        for bid_idx in bids_demand_sorted_t.index:
            if bids_demand_sorted_t.index.get_loc(bid_idx) == 0:
                bids_demand_sorted_t.loc \
                    [bid_idx, "q_aggregated"] \
                    = bids_demand_sorted_t.loc [bid_idx, "q"]
            else:
                bids_demand_sorted_t.loc \
                    [bid_idx, "q_aggregated"] \
                    = bids_demand_sorted_t.iloc [bids_demand_sorted_t.index.get_loc(bid_idx)-1, 4] \
                      + bids_demand_sorted_t.loc [bid_idx, "q"]

        for bid_idx in bids_supply_sorted_t.index:
            if bids_supply_sorted_t.index.get_loc(bid_idx) == 0:
                bids_supply_sorted_t.loc \
                    [bid_idx, "q_aggregated"] \
                    = bids_supply_sorted_t.loc [bid_idx, "q"]
            else:
                bids_supply_sorted_t.loc \
                    [bid_idx, "q_aggregated"] \
                    = bids_supply_sorted_t.iloc [bids_supply_sorted_t.index.get_loc(bid_idx)-1, 4] \
                      + bids_supply_sorted_t.loc [bid_idx, "q"]



    return cleared_bids_demand, cleared_bids_supply








def marketClearingProcessing(m,x,bids_demand,bids_supply,path_cleared_bid_demand,path_cleared_bid_supply):

    #-----finding the cleared bids (the ones having x!=0, i.e., x>=0)
    cleared_bids_indices_demand = [idx for idx in bids_demand.index \
                                   if x[idx].x != 0]
    cleared_bids_indices_supply = [idx for idx in bids_supply.index \
                                   if x[idx].x != 0]

    cleared_bids_demand = bids_demand.loc[(cleared_bids_indices_demand)]
    cleared_bids_demand['x'] = [x[idx].x for idx in cleared_bids_demand.index]
    cleared_bids_demand.to_excel(path_cleared_bid_demand)

    cleared_bids_supply = bids_supply.loc[(cleared_bids_indices_supply)]
    cleared_bids_supply['x'] = [x[idx].x for idx in cleared_bids_supply.index]
    cleared_bids_supply.to_excel(path_cleared_bid_supply)

    #Set of T includes only the hours in this day that are cleared so that we don't have to plot all the unnessary plots fo
    T = list(cleared_bids_demand.groupby(level=[1]).groups.keys()) # the set cleared of hours

    for t in T:
        #sorting the cleared bids in each hour
        bids_demand_sorted_t = bids_demand.loc[bids_demand.groupby(level=[1]).groups[t]].sort_values(by=['u'], ascending=False)
        bids_supply_sorted_t = bids_supply.loc[bids_supply.groupby(level=[1]).groups[t]].sort_values(by=['u'])

        #aggregated the sorted bids to be able to plot a stacked line plot for the supply and demand curves
        for bid_idx in bids_demand_sorted_t.index:
            if bids_demand_sorted_t.index.get_loc(bid_idx) == 0:
                bids_demand_sorted_t.loc \
                    [bid_idx, "q_aggregated"] \
                    = bids_demand_sorted_t.loc [bid_idx, "q"]
            else:
                bids_demand_sorted_t.loc \
                    [bid_idx, "q_aggregated"] \
                    = bids_demand_sorted_t.iloc [bids_demand_sorted_t.index.get_loc(bid_idx)-1, 4] \
                      + bids_demand_sorted_t.loc [bid_idx, "q"]

        for bid_idx in bids_supply_sorted_t.index:
            if bids_supply_sorted_t.index.get_loc(bid_idx) == 0:
                bids_supply_sorted_t.loc \
                    [bid_idx, "q_aggregated"] \
                    = bids_supply_sorted_t.loc [bid_idx, "q"]
            else:
                bids_supply_sorted_t.loc \
                    [bid_idx, "q_aggregated"] \
                    = bids_supply_sorted_t.iloc [bids_supply_sorted_t.index.get_loc(bid_idx)-1, 4] \
                      + bids_supply_sorted_t.loc [bid_idx, "q"]



    return cleared_bids_demand, cleared_bids_supply



def primal_multibid(bids_demand,bids_supply):

    m = gp . Model ("CL_activation_primal_multibid")
    m.Params.LogToConsole = 0
    # j: the participant
    # t: hour of the day
    # g: the grandchildren bid, or the granularities of the bids in each hour

    ##########----Defining the variables
    x = {} # how much CL is cleared from each bid x[j,c,t,g]

    for idx in bids_demand.index:
        x[idx] = m.addVar(name ="x[%a]" %str(idx))
    for idx in bids_supply.index:
        x[idx] = m.addVar(name ="x[%a]" %str(idx))


    #########-----Defining constraints
    for idx in bids_demand.index:
        # cleared quantity should be limited to quantity that is bidded
        m.addConstr(x[idx] <= \
                    bids_demand.loc[idx,'q'], name="q_constaint [%a]" %(str(idx)))

    for idx in bids_supply.index:
        # cleared quantity should be limited to quantity that is bidded
        m.addConstr(x[idx] <= \
                    bids_supply.loc[idx,'q'] , name="q_constaint [%a]" %(str(idx)))

    #The hourly balance of demand and supply
    #!!!! this can be done also only for the hours that are not equal to zero
    #but have to check how it would impact the other contraints and the OF
    T = list(bids_demand.groupby(level=[1]).indices.keys()) # the set of hours
    for t in T:
        m . addConstr ( gp.quicksum(x[idx] for idx in bids_demand.index if idx[1]==t) \
                        - gp.quicksum(x[idx] for idx in bids_supply.index if idx[1]==t) == 0, "balance_constraint[%a]"%t)

    ##########----- Set objective : maximize social welfare
    obj = gp.quicksum(bids_demand.loc[idx,'u'] * x[idx] for idx in bids_demand.index) \
          - gp.quicksum(bids_supply.loc[idx,'u'] * x[idx] for idx in bids_supply.index)
    m . setObjective (obj, GRB . MAXIMIZE )

    m.optimize()

    return m,x



def marketClearing(agents, demand_curve):
    random.shuffle(agents)
    m = gp . Model ("CL_activation_primal_multibid")
    m.Params.LogToConsole = 0
    # j: the participant
    # t: hour of the day
    # g: the grandchildren bid, or the granularities of the bids in each hour

    ##########----Defining the variables
    supply_quantities_cleared = [] #how much CL is cleared from each bid x[j,c,t,g] (supply side)
    demand_quantities_cleared = [] #how much CL is cleared from each bid x[j,c,t,g] (demand side)

    # since there are multiple sellers, their bids need to be indexed so that we 
    # know whose bid a bid is,
    # meaning allBids = [..., [agent_index, quantity_i, price_i],...]
    # however, there is only 1 buyer so those bids need not be indexed, 
    # meaning demand = [..., [quantity_i, price_i],...]

    allBids = []
    for i, agent in enumerate(agents):
        for j, bid in enumerate(agent.bids_curve):
            # now bid is [quantity_i, price_i, agent_index]
            allBids.append(bid + [i] )
            supply_quantities_cleared.append(m.addVar(name="s_" + str(i) + "_" + str(j)))
            m.addConstr(supply_quantities_cleared[-1] <= bid[0], \
                    name="qs_" + str(i) + "_" + str(j)) 
    
    for i, bid in enumerate(demand_curve):
        demand_quantities_cleared.append(m.addVar(name="d_" + str(i)))
        m.addConstr(demand_quantities_cleared[-1] <= bid[0], \
                name="qd_" + str(i))

    m.addConstr(gp.quicksum(demand_quantities_cleared) \
            - gp.quicksum(supply_quantities_cleared) == 0,
            "balance_constraint")

    ##########----- Set objective : maximize social welfare
    obj = gp.quicksum([quantity * demand_curve[i][1] \
        for i, quantity in enumerate(demand_quantities_cleared) ]) \
          - gp.quicksum([quantity * allBids[i][1] \
        for i, quantity in enumerate(supply_quantities_cleared) ])

    m.setObjective(obj, GRB . MAXIMIZE )
    m.optimize()

    supply_quantities_cleared_solution = []
    demand_quantities_cleared_solution = []

# NOTE FINISH
    for i, var in enumerate(supply_quantities_cleared):
        # = supply_quantities_cleared_solution = [..., [quantity_i, price_i, cleared_amount, agent_index],...]
        supply_quantities_cleared_solution.append(allBids[i][0:2] + [var.x] + [allBids[i][2]])

    for i, var in enumerate(demand_quantities_cleared):
        # = deman_quantities_cleared_solution = [..., [quantity_i, price_i, cleared_amount],...]
        demand_quantities_cleared_solution.append(demand_curve[i] + [var.x])
    return supply_quantities_cleared_solution, demand_quantities_cleared_solution, m


def market_clearing_probability_based(agents, demand_curve):
    random.shuffle(agents)
    m = gp.Model("CL_activation_primal_multibid_binary")
    m.Params.LogToConsole = 0
    # j: the participant
    # t: hour of the day
    # g: the grandchildren bid, or the granularities of the bids in each hour

    ##########----Defining the variables
    X_s = []  # how much CL is cleared from each bid x[j,c,t,g] (supply side)
    X_d = []  # how much CL is cleared from each bid x[j,c,t,g] (demand side)
    Y_s = []  # how much CL is cleared from each bid x[j,c,t,g] (supply side)
    Y_d = []  # how much CL is cleared from each bid x[j,c,t,g] (demand side)

    # since there are multiple sellers, their bids need to be indexed so that we
    # know whose bid a bid is,
    # meaning allBids = [..., [agent_index, quantity_i, price_i],...]
    # however, there is only 1 buyer so those bids need not be indexed,
    # meaning demand = [..., [quantity_i, price_i],...]

    allBids = []
    for i, agent in enumerate(agents):
        for j, bid in enumerate(agent.bids_curve):
            # now bid is [quantity_i, price_i, agent_index]
            allBids.append(bid + [i])
            X_s.append(m.addVar(name="x_s_" + str(i) + "_" + str(j)))
            Y_s.append(m.addVar(name="y_s_" + str(i) + "_" + str(j), vtype=GRB.BINARY))
            m.addConstr(X_s[-1] <= bid[0] * Y_s[-1], \
                        name="qs_" + str(i) + "_" + str(j))
            m.addConstr(Y_s[-1] * sum([b[0] for b in allBids[-(j+1):-1]]) <= sum(X_s[-(j+1):-1]), \
                        name="ys_" + str(i)+"_"+str(j))

    for i, bid in enumerate(demand_curve):
        X_d.append(m.addVar(name="x_d_" + str(i)))
        Y_d.append(m.addVar(name="y_d_" + str(i) , vtype=GRB.BINARY))
        m.addConstr(X_d[-1] <= bid[0] * Y_d[-1], \
                    name="qd_" + str(i))
        m.addConstr(Y_d[-1] * sum([b[0] for b in demand_curve[0:i]]) <= sum(X_d[0:-1]), \
                    name="yd_" + str(i))

    m.addConstr(gp.quicksum(X_d) \
                - gp.quicksum(X_s) == 0,
                "balance_constraint")

    ##########----- Set objective : maximize social welfare
    obj = gp.quicksum([quantity * demand_curve[i][1] \
                       for i, quantity in enumerate(X_d)]) \
          - gp.quicksum([quantity * allBids[i][1] \
                         for i, quantity in enumerate(X_s)])

    m . setObjective (obj, GRB . MAXIMIZE)
    m.optimize()

    supply_quantities_cleared_solution = []
    demand_quantities_cleared_solution = []

    # NOTE FINISH
    for i, var in enumerate(X_s):
        # = supply_quantities_cleared_solution = [..., [quantity_i, price_i, cleared_amount, agent_index],...]
        supply_quantities_cleared_solution.append(allBids[i][0:2] + [var.x] + [allBids[i][2]])

    for i, var in enumerate(X_d):
        # = deman_quantities_cleared_solution = [..., [quantity_i, price_i, cleared_amount],...]
        demand_quantities_cleared_solution.append(demand_curve[i] + [var.x])

    return supply_quantities_cleared_solution, demand_quantities_cleared_solution, m

