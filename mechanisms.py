import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import copy
from math import factorial

from market_clearing import marketClearing
from market_clearing import market_clearing_probability_based
from itertools import chain, combinations

"""
take agent contributions and social welfare
return payoff for each agent

possibly you'll need to rerun market clearing with different participants
to calculate this (certainly will have to do it for shapley)
"""


def uniformPricing(agents, supply_quantities_cleared_solution, demand_quantities_cleared_solution, m, epoch,\
                   runs_per_strategy_update, run_of_strategy):
    uniform_price = m.getConstrByName("balance_constraint").Pi

#     sortByPrice = lambda solution : solution[1]
#     supply_quantities_cleared_sorted = sorted(supply_quantities_cleared_solution, key=sortByPrice)
# #    print(supply_quantities_cleared_sorted)
#
#     for i, solution in enumerate(supply_quantities_cleared_sorted):
# #        print(solution)
#         if solution[2] == 0 and i==0:
#             uniform_price = 0
#             break
#         elif solution[2] == 0:
#             uniform_price = supply_quantities_cleared_sorted[i-1][1]
#             break
#
#     price_interval_cuttoff = 0.5
#
#     demand_quantities_cleared_sorted = sorted(demand_quantities_cleared_solution, key=sortByPrice)
#     for i, solution in enumerate(demand_quantities_cleared_sorted):
#         if solution[2] == 0 and i==0:
#             break
#         if solution[2] == 0:
#             uniform_price = uniform_price + price_interval_cuttoff * (demand_quantities_cleared_sorted[i-1][1] - uniform_price)
#             break

    payoffs = [0] * len(agents)
    for solution in supply_quantities_cleared_solution:
#        print(solution)
        payoffs[solution[3]] += uniform_price * solution[2]

    for i, agent in enumerate(agents):
        #agent.payoff_history.append(payoffs[i])
        agent.payoff_history[runs_per_strategy_update * epoch + run_of_strategy] = payoffs[i]
        if agent.last_strategy == 2:
            agent.last_adjusting_payoff = payoffs[i]


def VCG(agents, demand_curve, m_grand_coalition, cleared_bids_supply):

    SW_N = m_grand_coalition.ObjVal
    for i, agent in enumerate(agents):
        grand_coalition_wo_agent = copy.deepcopy(agents)
        grand_coalition_wo_agent.remove(grand_coalition_wo_agent[i])


        supply_cleared_wo_agent, demand_cleared_wo_agent, m_wo_agent = marketClearing(grand_coalition_wo_agent, demand_curve)
        sw_wo_agent = m_wo_agent.ObjVal
        agent_marginal_contribution = SW_N - sw_wo_agent

        agent_cost_or_value = sum([bid[1]*bid[2] for bid in cleared_bids_supply if bid[3]==i])
        agent_vcg_payment = agent_marginal_contribution + agent_cost_or_value #Simplified marginal_contribution - cost_or_value*(-1)
        agent.payoff_history.append(agent_vcg_payment)

def VCG_nima (agents, demand_curve, m, supply_quantities_cleared_solution, epoch,\
              runs_per_strategy_update, run_of_strategy,market_clearing_method):
    SW_grand_coalition = m.ObjVal

    payoffs = [0] * len(agents)
    marg_contribution = [0] * len(agents)
    declared_cost = [0] * len(agents)
    for i,agent in enumerate(agents):
        #calculating the marginal contribution to the social welfare
        #agents_without_i = copy.deepcopy(agents)
        #del agents_without_i[i]
        if market_clearing_method == "impact_based":
            _,_,m_without_i = marketClearing(agents[:i]+agents[i+1:],demand_curve)
        elif market_clearing_method == "probability_based":
            _, _, m_without_i = market_clearing_probability_based(agents[:i] + agents[i + 1:], demand_curve)
        marg_contribution [i] = SW_grand_coalition - m_without_i.ObjVal
        #to calculate the declared cost
        filtered_solution_with_i = list(filter(lambda solution: solution[3] == i, supply_quantities_cleared_solution))
        declared_cost [i] = sum([i_bids [1]*i_bids [2] for i_bids in filtered_solution_with_i])
        #payoff calculation
        payoffs [i] = marg_contribution[i] + declared_cost[i]

    for i, agent in enumerate(agents):
        #agent.payoff_history.append(payoffs[i])
        agent.payoff_history[runs_per_strategy_update * epoch + run_of_strategy] = payoffs[i]
        if agent.last_strategy == 2:
            agent.last_adjusting_payoff = payoffs[i]

def VCG_nima_NoCost (agents, demand_curve, m, supply_quantities_cleared_solution, epoch,\
                     runs_per_strategy_update, run_of_strategy,market_clearing_method):
    #!! not speeded up yet like VCG_nima
    SW_grand_coalition = m.ObjVal

    payoffs = [0] * len(agents)
    marg_contribution = [0] * len(agents)
    for i,agent in enumerate(agents):
        #calculating the marginal contribution to the social welfare
        agents_without_i = copy.deepcopy(agents)
        del agents_without_i[i]
        #del agents_without_i[i]
        if market_clearing_method == "impact_based":
            _,_,m_without_i = marketClearing(agents[:i]+agents[i+1:],demand_curve)
        elif market_clearing_method == "probability_based":
            _, _, m_without_i = market_clearing_probability_based(agents[:i] + agents[i + 1:], demand_curve)
        marg_contribution [i] = SW_grand_coalition -  m_without_i.ObjVal
        #payoff calculation
        payoffs [i] = marg_contribution[i]

    for i, agent in enumerate(agents):
        #agent.payoff_history.append(payoffs[i])
        agent.payoff_history[runs_per_strategy_update * epoch + run_of_strategy] = payoffs[i]
        if agent.last_strategy == 2:
            agent.last_adjusting_payoff = payoffs[i]

def uniformPricingOld(model_grand_coalition, x_grand_coalition, bids_demand, bids_supply, \
                           date, cleared_bids_demand, cleared_bids_supply):
    """
    alocate payments based on the calculated price
    """
    demand = bids_demand.index.unique(0).values #set of buyer agents
    supply = bids_supply.index.unique(0).values #set of seller agents

    uniform_price_clearing_results = pd.DataFrame(columns=["unifrom_price_SEKperkW"])
    return uniform_price_clearing_results


def Shapley_nima (agents, demand_curve, m, supply_quantities_cleared_solution, epoch,\
              runs_per_strategy_update, run_of_strategy, market_clearing_method):
    SW_grand_coalition = m.ObjVal

    agents_list = np.arange(len(agents)) #The name of the big players of the game
    #Find all the subsets of the grand coalition
    #DSO's ID is "1000"
    #players_names_in_GrandCoalition = np.append(players_names_in_GrandCoalition,1000)
    def powerset(iterable):
        "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

    all_subsets = list(powerset(agents_list))

    payoffs = [0] * len(agents)
    declared_cost = [0] * len(agents)
    Shapley_value = {}
    SW_S = {}
    for i, agent in enumerate(agents):
        print("Player: %a" % i)
        Shapley_value[i] = 0
        # iterating over all the subsets (S) of the Grand coalition
        for S in all_subsets:
            if i in S:
                coeff = factorial(len(S) - 1 +1) * factorial(len(agents)+1 - (len(S)+1)) / factorial(
                    len(agents)+1) #+1 because of the DSO
                # calculating SW for the subset S
                if market_clearing_method == "impact_based":
                    _, _, m_S = marketClearing([agents[j] for j in S], demand_curve)
                elif market_clearing_method == "probability_based":
                    _, _, m_S = market_clearing_probability_based([agents[j] for j in S], demand_curve)
                nu_S = m_S.ObjVal
                SW_S[S] = copy.deepcopy(nu_S)  # save the SW of each coalition to calculated the excess

                # removing player from the subset S to calculate the marginal contribution
                S_wo_i = list(S)
                S_wo_i.remove(i)

                if market_clearing_method == "impact_based":
                    _, _, m_S_wo_i = marketClearing([agents[j] for j in S_wo_i], demand_curve)
                elif market_clearing_method == "probability_based":
                    _, _, m_S_wo_i = market_clearing_probability_based([agents[j] for j in S_wo_i], demand_curve)
                nu_S_wo_i = m_S_wo_i.ObjVal

                Shapley_value[i] += coeff * (nu_S - nu_S_wo_i)

                # to calculate the declared cost
        filtered_solution_with_i = list(
            filter(lambda solution: solution[3] == i, supply_quantities_cleared_solution))
        declared_cost[i] = sum([i_bids[1] * i_bids[2] for i_bids in filtered_solution_with_i])
        # payoff calculation
        payoffs[i] = Shapley_value[i] + declared_cost[i]

        #agent.payoff_history.append(payoffs[i])
        agent.payoff_history[runs_per_strategy_update * epoch + run_of_strategy] = payoffs[i]
        if agent.last_strategy == 2:
            agent.last_adjusting_payoff = payoffs[i]

"""

def VCG_OLD(model_grand_coalition, x_grand_coalition, bids_demand, bids_supply, \
                date, cleared_bids_demand, cleared_bids_supply):
    demand=bids_demand.index.unique(0).values #set of buyer agents
    supply=bids_supply.index.unique(0).values #set of seller agents

    N = np.concatenate((demand,supply)) # The players in the grand coalition

    #____________________________________VCG_______________________________
    SW_N = model_grand_coalition.ObjVal #the value of the grand coalition
    T = list(bids_demand.groupby(level=[1]).groups.keys()) # the set of the cleared hours in a day
    uniform_price_clearing_results = pd.DataFrame(columns=["unifrom_price_SEKperkW"])

    #reading the uniform-price from the dual variable (Pi) of the balance constraint
    for t in T:
        uniform_price_clearing_results.loc[t,"unifrom_price_SEKperkW"] = \
            model_grand_coalition.getConstrByName("balance_constraint[%a]"%t).Pi


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
                = -(cleared_bids_demand.loc[participant,'x'] \
                    * [uniform_price_clearing_results.loc[t,"unifrom_price_SEKperkW"] \
                       for t in cleared_bids_demand.loc[participant,'x'].index.get_level_values(0)]).sum()
        else:
            results.loc[participant, 'cost_or_value' ] = (cleared_bids_supply.loc[participant,'u'] \
                                                          * cleared_bids_supply.loc[participant,'x']).sum()
            results.loc[participant, 'VCG_payment'] = results.loc[participant, 'marg_contribution'] \
                                                      - results.loc[participant, 'cost_or_value' ] * (-1)
            # #payments in case of uniform market clearing (to compare)
            results.loc[participant, 'uniform_price_payment'] \
                = (cleared_bids_supply.loc[participant,'x'] \
                   * [uniform_price_clearing_results.loc[t,"unifrom_price_SEKperkW"] \
                      for t in cleared_bids_supply.loc[participant,'x'].index.get_level_values(0)]).sum()

    return results


def shapley(model_grand_coalition, x_grand_coalition, bids_demand, bids_supply, \
            date, cleared_bids_demand, cleared_bids_supply):

    demand=bids_demand.index.unique(0).values #set of buyer agents
    supply=bids_supply.index.unique(0).values #set of seller agents
    N = np.concatenate((demand,supply)) # The players in the grand coalition

    SW_N = model_grand_coalition.ObjVal #the value of the grand coalition

    column_names = ['SW_N','SW_N\{p}','marg_contribution','cost_or_value']
    results = pd.DataFrame(columns=column_names)

    #___________________________________Shapley______________________________

    #we will divide all participants into four big groups of players as below:
    players_names_in_GrandCoalition = N #The name of the big players of the game
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



#----------------------------Calculating the payments-----------------------
def payment_allocation_multibid (model_grand_coalition, x_grand_coalition, bids_demand, bids_supply, \
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
                = -(cleared_bids_demand.loc[participant,'x'] \
                    * [uniform_price_clearing_results.loc[t,"unifrom_price_SEKperkW"] \
                       for t in cleared_bids_demand.loc[participant,'x'].index.get_level_values(0)]).sum()
        else:
            results.loc[participant, 'cost_or_value' ] = (cleared_bids_supply.loc[participant,'u'] \
                                                          * cleared_bids_supply.loc[participant,'x']).sum()
            results.loc[participant, 'VCG_payment'] = results.loc[participant, 'marg_contribution'] \
                                                      - results.loc[participant, 'cost_or_value' ] * (-1)
            # #payments in case of uniform market clearing (to compare)
            results.loc[participant, 'uniform_price_payment'] \
                = (cleared_bids_supply.loc[participant,'x'] \
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
    """

