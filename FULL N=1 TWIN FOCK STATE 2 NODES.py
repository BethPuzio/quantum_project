#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 08:24:26 2021

@author: Beth
"""
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import sympy as sym
import scipy.linalg as la
from numpy.linalg import inv

H = [[0,1],
     [1,0]]

node0 = [1,0]
node1 = [0,1]
node11 = [1,1]

s = (1/np.sqrt(2))

A = nx.from_numpy_matrix(np.array(H))  
nx.draw(A, with_labels=True, node_color="lightgreen")
plt.show()

t = sym.Symbol('t', real=True)

tstart = 0
tend = 10
time = np.linspace(tstart,tend,(tend-tstart)*20)

#This function combine the probability distribution calculations
def prob_dist(H, start_node, final_node):
    p_array = []
    t = sym.Symbol('t', real=True)
    #Matrix exponential diagonalisation
    eigvals, eigvecs = la.eig(H)
    diag_vals = []
    for i in range(len(eigvals)):
        diag_vals.append(sym.exp(1j*eigvals[i]*t))
    W = eigvecs
    X = inv(W)
    D = sym.diag(*diag_vals)
    M = sym.Matrix(W)*D*sym.Matrix(X)
    Y = M*sym.Matrix(start_node)
    Z = sym.Matrix(final_node).dot(Y)
    #Defining the probability function
    prob_func = sym.Abs(sym.simplify(Z))**2
    for i in range(len(time)):
        p = prob_func.evalf(subs={t:time[i]})
        p_array.append(sym.Abs(p))
      
    return p_array, prob_func

def Fisher(prob_func):
    f_array = []
    diff = sym.diff(prob_func, t)
    FI = sym.simplify((diff**2)/(prob_func))
    for i in range (len(time)):
        f = FI.evalf(subs={t:time[i]})
        f_array.append(sym.Abs(f)) 
    return f_array

#This is for twin fock states of the form |N,N>
#In this case we specifically consider an N=1 twin fock state
#First particle
prob_0, pf_00 = prob_dist(H, node0, node0)
prob_1, pf_01 = prob_dist(H, node0, node1)
#second particle
prob_10, pf_10 = prob_dist(H, node1, node0)
prob_11, pf_11 = prob_dist(H, node1, node1)

#Probability distribution arrays
prob20 = np.multiply(prob_0,prob_10)
prob21 = np.multiply(prob_1, prob_11)
prob11 = np.add(np.multiply(prob_1, prob_10),np.multiply(prob_0, prob_11))

#Probability distribution functions
pf20 = pf_00*pf_10
pf02 = pf_01*pf_11
pf211 = (pf_01*pf_10) + (pf_00*pf_11)

#Calculating Fisher information distributions from the porb distribution functions
FI20 = Fisher(pf20)
FI02 = Fisher(pf02)
FI211 = Fisher(pf211)

plt.title("Two Particles on Two Nodes")
plt.xlabel("Time in seconds")
plt.ylabel("Probability")
plt.plot(time, prob20, label="Both on node 0", linewidth=5)
plt.plot(time, prob21, label = "Both on node 1", c='gold')
plt.plot(time, prob11, label = "One on each node", c='magenta')
plt.legend(["Both on node 0" , 'Both on node 1', 'One on each node'], loc='upper right')
plt.show()

plt.title("Fisher Information for two particle Quantum Walk on Graph")
plt.xlabel("Time in seconds")
plt.ylabel("Fisher Information")
plt.plot(time, FI20, label="Both on node 0", linewidth =5)
plt.plot(time, FI02, label = "Both on node 1", c='gold')
plt.plot(time, FI211, label = "One on each node")
plt.plot(time, np.sum([FI20, FI02, FI211], axis=0), label = "Total", c='darkviolet')
plt.legend(["Both on node 0" , 'Both on node 1','One on each node', 'Total Fisher Info'], loc='upper right')
plt.show()

def data(T,n):
    node_20 = abs((pf20).evalf(subs={t:T}))
    node_02 = abs((pf02).evalf(subs={t:T}))
    node_11 = abs((pf211).evalf(subs={t:T}))
    nodes = [20,2,11]
    data = np.random.choice(nodes, n, p=[node_20, node_02, node_11])
    return data

def MLF(T,n):
    data_set = data(T,n)
    MLF=1
    for i in range(len(data_set)):
        if data_set[i]==20:
            MLF = MLF*(pf20)
        if data_set[i]==2:
            MLF = MLF*(pf02)
        if data_set[i]==11:
            MLF = MLF*(pf211)
    return MLF

timespan = np.linspace(tstart,tend/4,(tend/4-tstart)*2000)

#Using a less efficient method here, was struggling getting python to evaluate complicated expressions for zero
def turning_points(F):
    time_values = []
    for i in range(1, len(timespan)-1):
        f_low = sym.Abs(F.subs(t,timespan[i-1]))
        f = sym.Abs(F.subs(t,timespan[i]))
        f_high = sym.Abs(F.subs(t,timespan[i+1]))
        if f_low < f and f_high < f:
            time_values.append(timespan[i])
    return time_values                                                              

    #n defines the number of points in each data set produced
def repeat(N,T,n):
    time_values=turning_points(MLF(T,n))
    j=0
    while j<(N-1):
        appended = turning_points(MLF(T,n))
        if len(appended)==3:
            time_values = np.vstack((time_values,appended))
            print(time_values)
            j=j+1
        print(j)
        continue 
    mean = []
    var =[]
    for i in range(len(time_values[0])):
        c = time_values[:,[i]]
        mean.append(np.mean(c))
        var.append((np.std(c))**2)
    return mean, var

#N is the number of times a data set is produced
N=50
#Time trying to be estimated
T=1.5
#Number of points in a data set
n=50

mean, var = repeat(N,T,n)
MLF = MLF(T,n)
mlf_arr=[]
for i in range(len(timespan)):
    mlf_arr.append(abs(MLF.evalf(subs={t:timespan[i]})))
        
plt.title("Maximum Likelihood Function for substituted time {}s".format(T), pad=20)
plt.xlabel("Time in seconds")
plt.ylabel("Maximum Likelihood")
plt.plot(timespan, mlf_arr, label='N=1 twin fock state MLF')
plt.show()

print(mean, var)
