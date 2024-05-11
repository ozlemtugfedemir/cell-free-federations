#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import numpy.matlib
import pickle
import matplotlib.pyplot as plt
import scipy.linalg as sl
import time
import cvxpy as cvx
import copy
from k_means_constrained import KMeansConstrained

# using loadtxt()
betasdB0 = np.loadtxt("pathloss_factory_30CSPs.csv",
                 delimiter=",", dtype=str)

data_rate_th = 60e6

multiplying_factor = 3000+2500*np.random.rand() 

#Number of UEs 
K = 24

#Number of federations 
F = 4

#Number of ECSPs
ES = 5

HowManyCSPperECSP = 6

#Number of CSPs
S = ES*HowManyCSPperECSP

betasdB = np.zeros((K,S))
for s in range(S):
    for k in range(K):
        
        betasdB[k,s] = -float(betasdB0[s,k])
    
#Select length of pilot of UEs
tau_p = int(K/F)

#Number of CSP antennas
M = 16

## Propagation parameters

#Communication bandwidth
B = 20e6

#Total uplink transmit power per UE (mW)
p = 100

#Maximum downlink transmit power per CSP (mW)
Pmax = 200

#Define noise figure at AP (in dB)
noiseFigure = 7

#Compute noise power
noiseVariancedBm = -174 + 10*np.log10(B) + noiseFigure

#Select length of coherence block
tau_c = 200

samp_rate = 600e+6

betas0 = 10**(betasdB/10)

#Pilot SNR
rhop = p/(10**((noiseVariancedBm/10)))

gammas0 = tau_p*rhop*betas0**2/(tau_p*rhop*betas0+1)

betas = betas0/(10**((noiseVariancedBm/10)))
gammas = gammas0/(10**((noiseVariancedBm/10)))

SINRth = 2**((data_rate_th/B)*tau_c/(tau_c-tau_p))-1

RSbetas = np.sqrt(betas)
LSgammas =  np.sqrt(M)*np.sqrt(1/tau_p)*np.sqrt(gammas)/np.sqrt(SINRth)


t_dl = 1/B*(tau_c-tau_p)
t_ch = 1/B*tau_p

num_users = tau_p

FOMW = 34.4e-15  # DAC FoM factor
b = 12  # effective bit resolution
   
power_adc =  M*FOMW * 2**b * samp_rate

energy_channel_est =  power_adc * t_ch

# DL communication PowerAmplifier, DAC, EthernetPort, PLL, DLProcessor

eta_max= 0.34
Ptmax = 3000 #mW

power_multiplier = t_dl/eta_max*np.sqrt(Ptmax)
    
power_dac = power_adc

E_mac = 3.1e-12
E_sram = 5e-12
E_dram = 640e-12

alpha = 0.1
beta = 0.01

energy_overhead = 0.2

operation_per_second = 2 * M * num_users * tau_c

E_op = (1 + energy_overhead) * (E_mac  + alpha * E_sram + beta * E_dram)

E_proc = E_op * operation_per_second

energy_dl = (power_dac)*t_dl + E_proc

energy_WR  = (7 + 2.2)*(t_dl+t_ch)

energy_per_csp = energy_dl + energy_WR + energy_channel_est


#Solution accuracy
delta = 0.01

objPrevious = 100000000

objDifference = 10000 

xPrev = np.random.rand(K,F)
yPrev = np.random.rand(S,F)

    
RSmatrix = np.zeros((S+1,F,K))
LSmatrix = np.zeros((S+1,F,K))

for k in range(K):
    RSmatrix[0:S,:,k] = RSbetas[k:k+1,:].T@np.ones((1,F))
    LSmatrix[0:S,:,k] = LSgammas[k:k+1,:].T@np.ones((1,F))
    RSmatrix[S,:,k] = 1
iterr=1


#Continue iterations until stopping criterion is satisfied 
while objDifference  > delta: 
   
    t_CVX = cvx.Variable((S+1,F))
    epsilon_CVX = cvx.Variable((S,F))
    epsilon2_CVX = cvx.Variable((K,F))


    Constraints = []
    Objective = []
    Constraints.append(t_CVX[S,:]==1)
    Constraints.append(t_CVX>=0)
    Constraints.append(epsilon_CVX>=0)
    Constraints.append(epsilon2_CVX>=0)

 
    for k in range(K):
        for f in range(F):

            Constraints.append( 0.1*cvx.norm(cvx.multiply(t_CVX[:,f],RSmatrix[:,f,k]))*xPrev[k,f] <= 0.1*xPrev[k,f]*cvx.sum(cvx.multiply(t_CVX[:,f],LSmatrix[:,f,k]) )+0.1*epsilon2_CVX[k,f])

    for s in range(S):
        Objective.append(100*power_multiplier*cvx.norm((t_CVX[s,:])))

        for f in range(F):
            Constraints.append(0.1*t_CVX[s,f]<=0.1*(np.sqrt(Pmax)*yPrev[s,f])+0.1*epsilon_CVX[s,f])

    Objective.append(multiplying_factor*cvx.sum(cvx.sum((epsilon_CVX)))+multiplying_factor*cvx.sum(cvx.sum((epsilon2_CVX))))                       
                           
    prob = cvx.Problem(cvx.Minimize(cvx.sum(Objective)),Constraints)
    prob.solve(solver=cvx.GUROBI)
    tPrev = t_CVX.value
    
    
    
    x_CVX = cvx.Variable((K,F),boolean=True)

    y_CVX = cvx.Variable((S,F),boolean=True)
    z_CVX = cvx.Variable((ES,1),boolean=True)
        
    epsilon_CVX = cvx.Variable((S,F))
    epsilon2_CVX = cvx.Variable((K,F))

    Constraints = []
    Objective = []
    
    Constraints.append(epsilon_CVX>=0)
    Constraints.append(epsilon2_CVX>=0)

   
    for es in range(ES):
        for l in range(HowManyCSPperECSP):
            Constraints.append(cvx.sum(y_CVX[es*HowManyCSPperECSP+l,:])<=z_CVX[es])
    for k in range(K):
        for f in range(F):

            Constraints.append( 0.1*cvx.norm(cvx.multiply(tPrev[:,f],RSmatrix[:,f,k]))*x_CVX[k,f] <= 0.1*x_CVX[k,f]*cvx.sum(cvx.multiply(tPrev[:,f],LSmatrix[:,f,k]) )+0.1*epsilon2_CVX[k,f])

    
    for f in range(F):
        Constraints.append(cvx.sum(x_CVX[:,f])<=tau_p)
    for k in range(K):
        Constraints.append(cvx.sum(x_CVX[k,:])==1)
    for s in range(S):
        for f in range(F):
            Constraints.append(0.1*tPrev[s,f]<=0.1*(np.sqrt(Pmax)*y_CVX[s,f])+0.1*epsilon_CVX[s,f])

    Objective.append(multiplying_factor*cvx.sum(cvx.sum((epsilon_CVX)))+multiplying_factor*cvx.sum(cvx.sum((epsilon2_CVX))))                       
    Objective.append(100000*energy_WR*cvx.sum(cvx.sum(z_CVX))+100000*energy_per_csp*cvx.sum(cvx.sum(y_CVX)))
    
            

    prob = cvx.Problem(cvx.Minimize(cvx.sum(Objective)),Constraints)
    prob.solve(solver=cvx.GUROBI)
    yPrev = y_CVX.value
    xPrev = x_CVX.value
    zPrev = z_CVX.value
    objCurrent = 100000*energy_WR*np.sum(np.sum(zPrev))+100000*energy_per_csp*np.sum(np.sum(yPrev))
    for s in range(S):
        objCurrent = objCurrent+ 100*power_multiplier*np.linalg.norm((tPrev[s,:]))
    objDifference = np.abs(objPrevious-objCurrent)
    objPrevious = objCurrent
    print(objPrevious)
door = 0
iterr = 0

while door  ==  0 and iterr<20:
    t_CVX = cvx.Variable((S+1,F))
    iterr = iterr+1

     
    Constraints = []
    Objective = []

    Constraints.append(t_CVX[S,:]==1)
    Constraints.append(t_CVX>=0)


    for k in range(K):
        for f in range(F):

            Constraints.append( 0.1*cvx.norm(cvx.multiply(t_CVX[:,f],RSmatrix[:,f,k]))*xPrev[k,f] <= 0.1*xPrev[k,f]*cvx.sum(cvx.multiply(t_CVX[:,f],LSmatrix[:,f,k]) ))
    for s in range(S):
        Objective.append(100*power_multiplier*cvx.norm((t_CVX[s,:])))
        for f in range(F):
        
            Constraints.append(t_CVX[s,f]<=np.sqrt(Pmax)*yPrev[s,f])

                      
    prob3 = cvx.Problem(cvx.Minimize(cvx.sum(Objective)),Constraints)
    prob3.solve(solver=cvx.GUROBI,verbose=True)
    status = prob3.status
    if status == 'optimal':
        door = 1
    else:
        door2 = 0
        while door2==0:
            random_number1 = np.random.randint(0, S)
            random_number2 = np.random.randint(0, F)
            if np.sum(yPrev[random_number1,:])==0:
                yPrev[random_number1,random_number2] = 1
                door2 = 1
            if np.sum(np.sum(yPrev))==S:
                door2 = 1

tPrev = t_CVX.value
for es in range(ES):
    zPrev[es] = min(1,np.sum(np.sum(yPrev[es*HowManyCSPperECSP:(es+1)*HowManyCSPperECSP,:])))
    
objCurrent = 100000*energy_WR*np.sum(np.sum(zPrev))+100000*energy_per_csp*np.sum(np.sum(yPrev))
for s in range(S):
    objCurrent = objCurrent+ 100*power_multiplier*np.linalg.norm((tPrev[s,:]))

print(objCurrent/100000/(t_dl+t_ch))
print(np.sum(np.sum(yPrev)))



