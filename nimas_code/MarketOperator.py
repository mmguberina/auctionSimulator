# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 09:48:28 2021

@author: alavijeh

Agent: Market operator
Aim: Clearing for activation of capacity-limit products and payment allocation

Includes three functions:
    - Solving the Primal problem of market clearing
    - Calculating the payments based on VCG and Shapley
    - Post processing and structuring the results
"""


import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt


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
                bids_demand_sorted_t.loc\
                    [bid_idx, "q_aggregated"]\
                        = bids_demand_sorted_t.loc [bid_idx, "q"]
            else:
                bids_demand_sorted_t.loc\
                    [bid_idx, "q_aggregated"]\
                        = bids_demand_sorted_t.iloc [bids_demand_sorted_t.index.get_loc(bid_idx)-1, 4]\
                            + bids_demand_sorted_t.loc [bid_idx, "q"]
        
        for bid_idx in bids_supply_sorted_t.index:
            if bids_supply_sorted_t.index.get_loc(bid_idx) == 0:
                bids_supply_sorted_t.loc\
                    [bid_idx, "q_aggregated"]\
                        = bids_supply_sorted_t.loc [bid_idx, "q"]
                # plt.plot([0,bid.loc[0,'q']],[bid.loc[0,'u'],bid.loc[0,'u']])
            else:
                bids_supply_sorted_t.loc\
                    [bid_idx, "q_aggregated"]\
                        = bids_supply_sorted_t.iloc [bids_supply_sorted_t.index.get_loc(bid_idx)-1, 4]\
                            + bids_supply_sorted_t.loc [bid_idx, "q"]
        
        date = bids_demand_sorted_t.iloc[0,0].date()
        #plotting the supply-demand curve
        plt.figure()
        plt.title ("Demand-supply curves: %a %a" %(str(date),str(t)))
        #adding the first point in the curves as [0,u[0]] to get the full supply/demand curves
        plt.step(np.insert(bids_demand_sorted_t.q_aggregated.values,0,0), np.insert(bids_demand_sorted_t.u.values,0,bids_demand_sorted_t.u[0]), label="Demand")
        plt.step(np.insert(bids_supply_sorted_t.q_aggregated.values,0,0), np.insert(bids_supply_sorted_t.u.values,0,bids_supply_sorted_t.u[0]), label="Supply")
        plt.xlabel("Quantity [kW]")
        plt.ylabel("Valuation [SEK/kW]")
        plt.legend()
        
        
    return cleared_bids_demand, cleared_bids_supply
 
#---------------------------- Market clearing algorithm------------------------
def primal_multibid(bids_demand,bids_supply):

    m = gp . Model ("CL_activation_primal_multibid")
    m.Params.LogToConsole = 0
    # j: the participant
    # t: hour of the day
    # g: the grandchildren bid, or the granularities of the bids in each hour
    
    ##########----Defining the variables
    x = {} #howmuch CL is cleared from each bid x[j,c,t,g]

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


#----------------------------Calculating the payments-----------------------    
def payment_allocation_multibid (model_grand_coalition, x_grand_coalition, bids_demand, bids_supply,\
                                 date, cleared_bids_demand, cleared_bids_supply):
    
    # bids_demand = pd.read_excel(path_bid_demand,index_col=[0])
    # bids_supply = pd.read_excel(path_bid_supply,index_col=[0])

    demand=bids_demand.index.unique(0).values #set of buyer agents
    supply=bids_supply.index.unique(0).values #set of seller agents 
    
    N = np.concatenate((demand,supply)) # The players in the grand coalition
    
    #____________________________________VCG_______________________________
    # model_grand_coalition, x_grand_coalition = primal_multibid(bids_demand,bids_supply)
    SW_N = model_grand_coalition.ObjVal #the value of the grand coalition
    T = list(bids_demand.groupby(level=[1]).groups.keys()) # the set of the cleared hours in a day
    uniform_price_clearing_results = pd.DataFrame(columns=["unifrom_price_SEKperkW"])
    
    #reading the uniform-price from the dual variable (Pi) of the balance constraint
    for t in T:
        uniform_price_clearing_results.loc[t,"unifrom_price_SEKperkW"] = \
            model_grand_coalition.getConstrByName("balance_constraint[%a]"%t).Pi
    
    # models ["Dual_GrandCoalition"] = DualProblem(bids_demand, bids_supply)
    # # dual_balance_equality = models ['Dual_GrandCoalition'] . getVarByName ( 'lambda' ).x
    # uniform_price_GrandCoalition = models ['Dual_GrandCoalition'] . getVarByName ( 'lambda' ).x
    
    
    column_names = ['SW_N','SW_N\{p}','marg_contribution','cost_or_value' ,'VCG_payment', 'uniform_price_payment']
    results = pd.DataFrame(columns=column_names)
    
    for participant in N:
        bids_demand_S = bids_demand.drop(participant, errors='ignore')
        bids_supply_S = bids_supply.drop(participant, errors='ignore')
        #since here, only the objval is important, the dual is faster to solve than primal. however, the objval should be multiplied by -1
        m,x = primal_multibid(bids_demand_S,bids_supply_S)
        results.loc[participant,'SW_N\{p}'] = m.ObjVal
        results.loc[participant,'SW_N'] = SW_N
        #decision of the cleared value for each participant in the grand coalition optimization  
        # x_star_N_participant = model_grand_coalition . getVarByName ( 'x[\''+str(participant)+'\']' ).x
        # results.loc[participant, 'x_star_N'] = x_star_N_participant
        #calculating the marginal contribution of each participant
        results.loc[participant,'marg_contribution'] = SW_N - results.loc[participant, 'SW_N\{p}']
        
        # Calculating the VCG payment:
        #   the cost part of VCG payment is different for demand and supply.\
        #   So, a negative sign is multiplied to the cost part of VCG payment to supply
        if participant in demand:
            results.loc[participant, 'cost_or_value' ] = -(cleared_bids_demand.loc[participant,'u'] \
                                                          * cleared_bids_demand.loc[participant,'x']).sum()            
            results.loc[participant, 'VCG_payment'] = results.loc[participant, 'marg_contribution'] \
                + results.loc[participant, 'cost_or_value' ]
            # #payments in case of uniform market clearing (to compare)
            results.loc[participant, 'uniform_price_payment'] \
                 = -(cleared_bids_demand.loc[participant,'x']\
                         * [uniform_price_clearing_results.loc[t,"unifrom_price_SEKperkW"] \
                            for t in cleared_bids_demand.loc[participant,'x'].index.get_level_values(0)]).sum()
        else:
            results.loc[participant, 'cost_or_value' ] = (cleared_bids_supply.loc[participant,'u'] \
                                                          * cleared_bids_supply.loc[participant,'x']).sum()             
            results.loc[participant, 'VCG_payment'] = results.loc[participant, 'marg_contribution'] \
                - results.loc[participant, 'cost_or_value' ] * (-1)
            # #payments in case of uniform market clearing (to compare)
            results.loc[participant, 'uniform_price_payment'] \
                 = (cleared_bids_supply.loc[participant,'x']\
                         * [uniform_price_clearing_results.loc[t,"unifrom_price_SEKperkW"] \
                            for t in cleared_bids_supply.loc[participant,'x'].index.get_level_values(0)]).sum()
                
    
    
    #checking the budget balance
    budget_balance = results['VCG_payment'].sum()
    print('\n -------------Checking if VCG is budget-balanced ------------------- \n')
    print ('\nBudget Balance:' + str(budget_balance))
    if budget_balance < 0:
        print('We have EXCESS in the budget!')
    if budget_balance > 0:
        print('We have DEFICIT in the budget!')
        
   
    #___________________________________Shapley______________________________
        
    print('\n---------Calculating Shapley values---------\n')
    
    #we will divide all participants into four big groups of players as below:
    players_names_in_GrandCoalition = N #The name of the big players of the game
    # indexes_of_participants_under_each_player = dict.fromkeys(players_names_in_GrandCoalition)
    
    #Find all the subsets of the grand coalition
    from itertools import chain, combinations
    
    def powerset(iterable):
        "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
    
    all_subsets = list(powerset(players_names_in_GrandCoalition))
    
    #Calculate Shapley
    from math import factorial
    Shapley_value = {}
    SW_S = {}
    for player in players_names_in_GrandCoalition:
        print("Player: %a" %player)
        Shapley_value [player] = 0
        #iterating over all the subsets (S) of the Grand coalition
        for S in all_subsets:
            if player in S:
                coeff = factorial(len(S)-1) * factorial(len(players_names_in_GrandCoalition) - len(S)) / factorial(len(players_names_in_GrandCoalition))
                # #slicing the bids for the players
                # indexes_in_S =list()
                # for i in S:
                #     indexes_in_S.extend(i)
                idx_slc = pd.IndexSlice
                bids_demand_S = bids_demand.loc[idx_slc[list(S),:,:]]
                bids_supply_S = bids_supply.loc[idx_slc[list(S),:,:]]
                #calculating SW for the subset S
                m_S,x_S = primal_multibid(bids_demand_S, bids_supply_S)
                nu_S = m_S.ObjVal
                SW_S [S] = nu_S #save the SW of each coalition to calculated the excess
                
                #removing player from the subset S to calculate the marginal contribution
                S_wo_player = list(S)
                S_wo_player.remove(player)
                
                # indexes_in_S_wo_player =list()
                # for i in S_wo_player:
                #     indexes_in_S_wo_player.extend(i)
    
                bids_demand_S_wo_player = bids_demand.loc[idx_slc[list(S_wo_player),:,:]]
                bids_supply_S_wo_player = bids_supply.loc[idx_slc[list(S_wo_player),:,:]]
                m_S_wo_player, x_S_wo_player = primal_multibid(bids_demand_S_wo_player, bids_supply_S_wo_player)
                nu_S_wo_player = m_S_wo_player.ObjVal
                
                Shapley_value [player] += coeff * (nu_S - nu_S_wo_player)
     
    #Checking if the calculation of the Shapley is correct
    #Sum of Shapley values should be equal to the SW of the Grand Coalition
    print('\n---------Checking if Shapley calculations add up \n')
    print('Sum of all Shapley values: ' + str(sum(Shapley_value.values())))
    print('SW in the grand coalition: ' + str(SW_N))
    #print('\n---------------------------------------------------------')
    
    
    results_shapley = pd.DataFrame.from_dict(Shapley_value,orient='index',columns=['Shapley_value'])
    #calculating the payments (r) by --> shapley values +/- the declared cost/utility
    for i in players_names_in_GrandCoalition:
        if i in demand:
                            
            results_shapley.loc [i, 'cost_or_utility'] = results.loc[i, 'cost_or_value']
            results_shapley.loc [i, 'Shapley_payment'] \
                = results_shapley.loc [i,'Shapley_value'] \
                    + results_shapley.loc [i, 'cost_or_utility']
        else: #if a seller(i.e. in "supply")
            
            results_shapley.loc [i, 'cost_or_utility'] = results.loc[i, 'cost_or_value']
            results_shapley.loc [i, 'Shapley_payment'] \
                = results_shapley.loc [i,'Shapley_value'] \
                    + results_shapley.loc [i, 'cost_or_utility']
                    
        #bring the VCG payments for each player (just to compare with Shapley payments)
        results_shapley.loc[i,'VCG_payment'] = results.loc[i,'VCG_payment']   
                    
    print('\n---------Checking if Shapley-based payement is budget balanced \n')
    print('Sum of Shapley payments (including cost/utility): ' + str(results_shapley['Shapley_payment'].sum()))
    
    #Calculate and checking the stability of the grand coalition by calculating the "excess": e^x_S
    excess_S = {}
    
    for S in SW_S.keys():
        sum_shapley_values = results_shapley.loc[[i for i in S],'Shapley_value'].sum()
        excess_S[S] = SW_S[S] - sum_shapley_values
    
    #plotting excess
    excess_S_df = pd.DataFrame.from_dict(excess_S,orient='index',columns=['Excess'])
    excess_S_df=excess_S_df.sort_values(by=['Excess'])
    excess_S_df.plot(kind='bar')
    
    return results, results_shapley, excess_S_df
