# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 09:48:28 2021

@author: alavijeh

Agent: Market operator
Aim: Clearing for activation of capacity-limit products 

Includes three functions:
    - Solving the Primal
    - Solving the Dual
    - Solving the KKTs
"""


import numpy as np
import pandas as pd

import gurobipy as gp
from gurobipy import GRB

#_____________________________________________________________________________
    
#---------------------------Primal problem--------------------------------------
#_____________________________________________________________________________

def PrimalProblem (bids_buyers, bids_sellers):
    
    #___________________________importing data______________________________
    # In these two dataframes: 
        # agent_tag: the name of the agent
        # k: max quantity that the capacity of flexible assets can be reduced [kW]
        # u: price per unit of reduction (valuation) [€/kW]
        # location: where the CL service is bidded
    # bids_buyers = pd.read_excel(path_bid_buyers,index_col=[0])
    # bids_sellers = pd.read_excel(path_bid_sellers,index_col=[0])
    
    B=bids_buyers.index.unique(0).values #set of buyer agents
    S=bids_sellers.index.unique(0).values #set of seller agents
    
    
    #___________________________Optimization_______________________________
    
    # Create a new model
    m = gp . Model ("CL_activation_primal")
    
    ##########-----Creating variables
    x={} #howmuch CL is cleared from each bid
    for b in B:
        # variable that shows how much of a bid is cleared
        x[b] = m . addVar (name ="x[%a]" %b)
    
    for s in S:
        # variable that shows how much of a bid is cleared
        x[s] = m . addVar (name ="x[%a]" %s)
    
    #########-----Defining constraints
    for b in B:
        # cleared quantity should be limited to quantity that is bidded
        m.addConstr(x[b]<=bids_buyers.loc[b,'k'], name="cap_constaint [%a]" %(b))
                
    for s in S:
        # cleared quantity should be limited to quantity that is bidded
        m.addConstr(x[s]<=bids_sellers.loc[s,'k'], name="cap_constaint [%a]" %(s))   
    
    #The balance of demand and supply
        m . addConstr ( gp.quicksum(x[b] for b in B) \
                      - gp.quicksum(x[s] for s in S) == 0, "balance_constraint")
            
            
    ##########----- Set objective : maximize social welfare
    obj = gp.quicksum(bids_buyers.loc[b,'u']*x[b] for b in B) \
                      - gp.quicksum(bids_sellers.loc[s,'u']*x[s] for s in S)
    m . setObjective (obj, GRB . MAXIMIZE )
    
    m.optimize()
    
    return m
#_____________________________________________________________________________
    
#-----------------------------------KKTs--------------------------------------
#_____________________________________________________________________________

def KKTs (path_bid_buyers, path_bid_sellers):
    
    #___________________________importing data______________________________
    
    # In these two dataframes: 
        # agent_tag: the name of the agent
        # k: max quantity that the capacity of flexible assets can be reduced [kW]
        # u: price per unit of reduction (valuation) [€/kW]
        # location: where the CL service is bidded
    bids_buyers = pd.read_excel(path_bid_buyers,index_col=[0])
    bids_sellers = pd.read_excel(path_bid_sellers,index_col=[0])
    
    B=bids_buyers.index.unique(0).values #set of buyer agents
    S=bids_sellers.index.unique(0).values #set of seller agents
    
    
    #___________________________Optimization_______________________________
    
    # Create a new model
    m = gp . Model ("CL_activation_KKTs")
    m.params.NonConvex=2
    
    ##########-----Creating variables
    mu_1={}
    mu_2={}
    mu_3={}
    mu_4={} 
    
    lmbda=m.addVar(lb=-gp.GRB.INFINITY,name="lambda") #dual variable of the balance equation (free variable)
    
    for b in B:
        # dual variables for x_b constraints 
        mu_2[b] = m . addVar (name ="mu_2[%a]" %b) #constraint for lower bound x_b
        mu_4[b] = m . addVar (name ="mu_4[%a]" %b) #constraint for higher bound x_b
    
    for s in S:
        # dual variables for x_s constraints 
        mu_1[s] = m . addVar (name ="mu_1[%a]" %s) #constraint for lower bound x_s
        mu_3[s] = m . addVar (name ="mu_3[%a]" %s) #constraint for higher bound x_s
    
    
    x={} #howmuch CL is cleared from each bid
    for b in B:
        # variable that shows how much of a bid is cleared
        x[b] = m . addVar (name ="x[%a]" %b)
    
    for s in S:
        # variable that shows how much of a bid is cleared
        x[s] = m . addVar (name ="x[%a]" %s)
    
    ###########-----Defining constraints
    for b in B:
        # cleared quantity should be limited to quantity that is bidded
        m.addConstr(x[b]<=bids_buyers.loc[b,'k'], name="cap_constaint [%a]" %(b))
        m.addConstr(-bids_buyers.loc[b,'u']+mu_4[b]+lmbda>=0,\
                    name="derivative to x[%a]" %(b))
        m.addConstr(x[b]*mu_2[b]==0, name="complemntary constrint mu_2 [%a]" %(b))
        m.addConstr(mu_4[b]*(x[b]-bids_buyers.loc[b,'k'])==0,\
                    name="complemntary constrint mu_4 [%a]" %(b))
            
    for s in S:
        # cleared quantity should be limited to quantity that is bidded
        m.addConstr(x[s]<=bids_sellers.loc[s,'k'], name="cap_constaint [%a]" %(s))   
        m.addConstr(bids_sellers.loc[s,'u']+mu_3[s]-lmbda>=0, \
                    name="derivative to x[%a]" %(s))
        m.addConstr(x[s]*mu_1[s]==0, name="complemntary constrint mu_1 [%a]" %(s))
        m.addConstr(mu_3[s]*(x[s]-bids_sellers.loc[s,'k'])==0,\
                    name="complemntary constrint mu_3 [%a]" %(s))
    
    #The balance of demand and supply
        m . addConstr ( gp.quicksum(x[b] for b in B) \
                      - gp.quicksum(x[s] for s in S) == 0, "balance_constraint")
            
            
    ###########----- Set objective : maximize social welfare
    obj = 1
    m . setObjective (obj, GRB . MAXIMIZE )
    
    m.optimize()
    
    return m

def KKTs_bigM (path_bid_buyers, path_bid_sellers):
    
    #___________________________importing data______________________________
    
    # In these two dataframes: 
        # agent_tag: the name of the agent
        # k: max quantity that the capacity of flexible assets can be reduced [kW]
        # u: price per unit of reduction (valuation) [€/kW]
        # location: where the CL service is bidded
    bids_buyers = pd.read_excel(path_bid_buyers,index_col=[0])
    bids_sellers = pd.read_excel(path_bid_sellers,index_col=[0])
    
    B=bids_buyers.index.unique(0).values #set of buyer agents
    S=bids_sellers.index.unique(0).values #set of seller agents
    
    
    #___________________________Optimization_______________________________
    
    # Create a new model
    m = gp . Model ("CL_activation_KKTs_bigM")
    #m.params.NonConvex=2
    
    ##########-----Creating variables
    mu_1={}
    mu_2={}
    mu_3={}
    mu_4={} 
    
    Y_mu_1={}
    Y_mu_2={}
    Y_mu_3={}
    Y_mu_4={}
    
    M=99999 #big-M (for linearization of complementary slackness conditions)
    
    lmbda = m.addVar(lb=-gp.GRB.INFINITY,name="lambda") #dual variable of the balance equation (free variable)
    
    for b in B:
        # dual variables for x_b constraints 
        mu_2[b] = m . addVar (name ="mu_2[%a]" %b) #constraint for lower bound x_b
        mu_4[b] = m . addVar (name ="mu_4[%a]" %b) #constraint for higher bound x_b
        Y_mu_2[b] = m . addVar (name="Y_mu_2[%a]"%b, vtype=GRB.BINARY)
        Y_mu_4[b] = m . addVar (name="Y_mu_4[%a]"%b, vtype=GRB.BINARY)
    for s in S:
        # dual variables for x_s constraints 
        mu_1[s] = m . addVar (name ="mu_1[%a]" %s) #constraint for lower bound x_s
        mu_3[s] = m . addVar (name ="mu_3[%a]" %s) #constraint for higher bound x_s
        Y_mu_1[s] = m . addVar (name="Y_mu_1[%a]"%s, vtype=GRB.BINARY)
        Y_mu_3[s] = m . addVar (name="Y_mu_3[%a]"%s, vtype=GRB.BINARY)
    
    x={} #howmuch CL is cleared from each bid
    for b in B:
        # variable that shows how much of a bid is cleared
        x[b] = m . addVar (name ="x[%a]" %b)
    
    for s in S:
        # variable that shows how much of a bid is cleared
        x[s] = m . addVar (name ="x[%a]" %s)
    
    ###########-----Defining constraints
    for b in B:
        # cleared quantity should be limited to quantity that is bidded
        m.addConstr(x[b]<=bids_buyers.loc[b,'k'], name="cap_constaint [%a]" %(b))
        m.addConstr(-bids_buyers.loc[b,'u']-mu_2[b]+mu_4[b]+lmbda==0,\
                    name="derivative to x[%a]" %(b))
        #m.addConstr(x[b]*mu_2[b]==0, name="complemntary constrint mu_2 [%a]" %(b))
        m.addConstr(mu_2[b] <= M * Y_mu_2[b], name="complemntary constrint1 mu_2 [%a]" %(b))
        m.addConstr(x[b] <= M * (1 - Y_mu_2[b]), name="complemntary constrint2 mu_2 [%a]" %(b))

        #m.addConstr(mu_4[b]*(x[b]-bids_buyers.loc[b,'k'])==0,\
        #            name="complemntary constrint mu_4 [%a]" %(b))
        m.addConstr(mu_4[b] <= M * Y_mu_4[b], name="complemntary constrint1 mu_4 [%a]" %(b))
        m.addConstr(bids_buyers.loc[b,'k'] - x[b] <= M * (1 - Y_mu_4[b]), name="complemntary constrint2 mu_4 [%a]" %(b))            
            
    for s in S:
        # cleared quantity should be limited to quantity that is bidded
        m.addConstr(x[s]<=bids_sellers.loc[s,'k'], name="cap_constaint [%a]" %(s))   
        m.addConstr(bids_sellers.loc[s,'u']-mu_1[s]+mu_3[s]-lmbda==0, \
                    name="derivative to x[%a]" %(s))
        #m.addConstr(x[s]*mu_1[s]==0, name="complemntary constrint mu_1 [%a]" %(s))
        m.addConstr(mu_1[s] <= M * Y_mu_1[s], name="complemntary constrint1 mu_1 [%a]" %(s))
        m.addConstr(x[s] <= M * (1 - Y_mu_1[s]), name="complemntary constrint2 mu_1 [%a]" %(s))
        
        #m.addConstr(mu_3[s]*(x[s]-bids_sellers.loc[s,'k'])==0,\
        #            name="complemntary constrint mu_3 [%a]" %(s))
        m.addConstr(mu_3[s] <= M * Y_mu_3[s], name="complemntary constrint1 mu_3 [%a]" %(s))
        m.addConstr(bids_sellers.loc[s,'k'] - x[s] <= M * (1 - Y_mu_3[s]), name="complemntary constrint2 mu_3 [%a]" %(s))  
    
    #The balance of demand and supply
        m . addConstr ( gp.quicksum(x[b] for b in B) \
                      - gp.quicksum(x[s] for s in S) == 0, "balance_constraint")
            
            
    ###########----- Set objective : maximize social welfare
    obj = 1
    m . setObjective (obj, GRB . MAXIMIZE )
    
    m.optimize()
    
    return m

#_____________________________________________________________________________
    
#---------------------------Dual problem--------------------------------------
#_____________________________________________________________________________    

def DualProblem (bids_buyers, bids_sellers):
    
    #___________________________importing data______________________________
    
    # In these two dataframes: 
        # agent_tag: the name of the agent
        # k: max quantity that the capacity of flexible assets can be reduced [kW]
        # u: price per unit of reduction (valuation) [€/kW]
        # location: where the CL service is bidded
    # bids_buyers = pd.read_excel(path_bid_buyers,index_col=[0])
    # bids_sellers = pd.read_excel(path_bid_sellers,index_col=[0])
    
    B=bids_buyers.index.unique(0).values #set of buyer agents
    S=bids_sellers.index.unique(0).values #set of seller agents
    
    
    #___________________________Optimization_______________________________
    
    # Create a new model
    m = gp . Model ("CL_activation_dual")
    
    ##-----Creating variables
    mu_1={}
    mu_2={}
    mu_3={}
    mu_4={} 
    
    lmbda=m.addVar(lb=-gp.GRB.INFINITY,name="lambda") #dual variable of the balance equation (free variable)
    
    for b in B:
        # dual variables for x_b constraints 
        mu_2[b] = m . addVar (name ="mu_2[%a]" %b) #constraint for lower bound x_b
        mu_4[b] = m . addVar (name ="mu_4[%a]" %b) #constraint for higher bound x_b
    
    for s in S:
        # dual variables for x_s constraints 
        mu_1[s] = m . addVar (name ="mu_1[%a]" %s) #constraint for lower bound x_s
        mu_3[s] = m . addVar (name ="mu_3[%a]" %s) #constraint for higher bound x_s
    
    ##-----Defining constraints
    for b in B:
        m.addConstr(-bids_buyers.loc[b,'u']+mu_4[b]+lmbda>=0, name="derivative to x[%a]" %(b))
    for s in S:
        m.addConstr(bids_sellers.loc[s,'u']+mu_3[s]-lmbda>=0, name="derivative to x[%a]" %(s))
            
            
    ##----- Set objective : maximize social welfare
    obj = gp.quicksum(-bids_buyers.loc[b,'k']*mu_4[b] for b in B) \
                      + gp.quicksum(-bids_sellers.loc[s,'k']*mu_3[s] for s in S)
    m . setObjective (obj, GRB . MAXIMIZE )
    
    m.optimize()

    return m

