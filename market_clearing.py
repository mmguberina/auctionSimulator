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



def primal_multibid(bids_demand,bids_supply):

    m = gp . Model ("CL_activation_primal_multibid")
    m.Params.LogToConsole = 0
    # j: the participant
    # t: hour of the day
    # g: the grandchildren bid, or the granularities of the bids in each hour

    ##########----Defining the variables
    x = {} #how much CL is cleared from each bid x[j,c,t,g]

    for idx in bids_demand.index:
        x[idx] = m . addVar (name ="x[%a]" %str(idx))
    for idx in bids_supply.index:
        x[idx] = m . addVar (name ="x[%a]" %str(idx))


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
