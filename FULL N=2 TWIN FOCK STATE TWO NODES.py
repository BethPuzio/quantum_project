#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 17:53:02 2021

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
time = np.linspace(tstart,tend,(tend-tstart)*15)

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
        
#This is for dual fock states of the form |N,N>
#First particle
prob_00, pf_00 = prob_dist(H, node0, node0)
prob_01, pf_01 = prob_dist(H, node0, node1)
#second particle
prob_10, pf_10 = prob_dist(H, node1, node0)
prob_11, pf_11 = prob_dist(H, node1, node1)

prob20 = np.multiply(prob_00,prob_10)
prob02 = np.multiply(prob_10, prob_11)
prob11 = np.add(np.multiply(prob_01, prob_10),np.multiply(prob_00, prob_11))

prob40 = np.multiply(prob20,prob20)
prob04 = np.multiply(prob02, prob02)
prob31 = np.add(np.multiply(prob11,prob20),np.multiply(prob20,prob11))
prob13 = np.add(np.multiply(prob11,prob02),np.multiply(prob02,prob11))
prob22 = np.add(np.add(np.multiply(prob02,prob20),np.multiply(prob20,prob02)), np.multiply(prob11,prob11))

pf20 = pf_00*pf_10
pf02 = pf_01*pf_11
pf40 = pf20*pf20
pf04 = pf02*pf02
pf11 = (pf_01*pf_10) + (pf_00*pf_11)
pf31 = (pf11*pf20)+(pf20*pf11)
pf13 = (pf11*pf02)+(pf02*pf11)
pf22 = (pf02*pf20)+(pf20*pf02)+(pf11*pf11)
print(pf22)

farr40 = Fisher(pf40)
farr04 = Fisher(pf04)
farr31 = Fisher(pf31)
farr13 = Fisher(pf13)
farr22 = Fisher(pf22)

plt.title("Probability distribution of a given node superposition, at time t")
plt.xlabel("Time in seconds")
plt.ylabel("Probability")
plt.plot(time, prob40, label="|4,0>", linewidth = 5)
plt.plot(time, prob04, label="|0,4>")
plt.plot(time, prob31, label="|3,1>", linewidth = 5)
plt.plot(time, prob13, label="|1,3>", c='gold')
plt.plot(time, prob22, label="|2,2>", c='darkred')
plt.legend(["|4,0>" ,'|0,4>','|3,1>','|1,3>','|2,2>'], loc='upper right')
plt.show()

plt.title("Fisher Information for Quantum Walk on Graph")
plt.xlabel("Time in seconds")
plt.ylabel("Fisher Information")
plt.plot(time, farr40, label="|4,0>", linewidth = 5)
plt.plot(time, farr04, label="|0,4>")
plt.plot(time, farr31, label="|3,1>", linewidth = 5)
plt.plot(time, farr13, label="|1,3>", c='gold')
plt.plot(time, farr22, label="|2,2>", c='darkred')
plt.plot(time, np.sum([farr40, farr04, farr31, farr13, farr22], axis=0), label = "Total", c='mediumpurple')
plt.legend(["|4,0>" ,'|0,4>','|3,1>','|1,3>','|2,2>', 'Total Fisher Info'], loc='upper right')
plt.show()

def data(T,n):
    node40 = (pf40).evalf(subs={t:T})
    node04 = (pf04).evalf(subs={t:T})
    node31 = (pf31).evalf(subs={t:T})
    node13 = (pf13).evalf(subs={t:T})
    node22 = (pf22).evalf(subs={t:T})
    nodes = [40,4,31,13,22]
    data = np.random.choice(nodes, n, p=[node40, node04, node31, node13, node22])
    return data

def MLF(T,n):
    data_set = data(T,n)
    MLF=1
    for i in range(len(data_set)):
        if data_set[i]==40:
            MLF = MLF*(pf40)
        if data_set[i]==4:
            MLF = MLF*(pf04)
        if data_set[i]==31:
            MLF = MLF*(pf31)
        if data_set[i]==13:
            MLF = MLF*(pf13)
        if data_set[i]==22:
            MLF = MLF*(pf22)
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
plt.plot(timespan, mlf_arr, label='N=2 twin fock state MLF')
plt.legend(["N=2 twin fock state MLF"], loc='upper right')
plt.show()

print(mean, var)
