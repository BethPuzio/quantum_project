#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 14:59:23 2021

@author: Beth
"""
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import sympy as sym
import scipy.linalg as la
from numpy.linalg import inv
from sympy import I

#This code currently specifically applies to the two node graph, adjacency matrix defined as H
H = [[0,1],
     [1,0]]

node0 = [1,0]
node1 = [0,1]
node11 = [1,1]

H_2 = [[0,1,1,1,0],
       [1,0,0,1,1],
       [1,0,0,0,1],
       [1,1,0,0,0],
       [0,1,1,0,0]]

#three node ring
H_3 = [[0,1,1],
       [1,0,1],
       [1,1,0]]

#three node line
H_4 = [[0,1,0],
       [1,0,1],
       [0,1,0]]

H_5 = [[0,1,0,0,0],
       [1,0,1,0,0],
       [0,1,0,1,0],
       [0,0,1,0,1],
       [0,0,0,1,0]]

H_52 = [[0,1,0,0,1],
        [1,0,1,0,0],
        [0,1,0,1,0],
        [0,0,1,0,1],
        [1,0,0,1,0]]

node30 = [1,0,0]
node31 = [0,1,0]
node32 = [0,0,1]

node50 = [1,0,0,0,0]
node51 = [0,1,0,0,0]
node52 = [0,0,1,0,0]
node53 = [0,0,0,1,0]
node54 = [0,0,0,0,1]

s = (1/np.sqrt(2))

A = nx.from_numpy_matrix(np.array(H_5))  
nx.draw(A, with_labels=True, node_color="lightgreen")
plt.show()

t = sym.Symbol('t')

tstart = 0
tend = 20
time = np.linspace(tstart,tend,(tend-tstart)*10)

#This function combine the probability distribution calculations
def prob_dist(H, start_node, final_node):
    p_array = []
    f_array = []
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
    
    diff = sym.diff(prob_func, t)
    FI = (diff**2)/(prob_func)
    for i in range (len(time)):
        f = FI.evalf(subs={t:time[i]})
        f_array.append(sym.Abs(f))    
    return p_array, f_array

#Defining a function that returns node column vectors 
def nodes(H):
    M = np.eye(len(H), len(H))
    return np.hsplit(M, len(H))


prob_0, fi_0 = prob_dist(H_5, node50, node50)
prob_1, fi_1 = prob_dist(H_5, node50, node51)
prob_2, fi_2 = prob_dist(H_5, node50, node52)
prob_3, fi_3 = prob_dist(H_5, node50, node53)
#In order to model loss, we could say that we are unable to detect from node 4
prob_4, fi_4 = prob_dist(H_5, node50, node54)


plt.title("Probability distribution of being at given node on graph, at time t")
plt.xlabel("Time in seconds")
plt.ylabel("Probability")
plt.scatter(time, prob_0, label="Node 0", s=60, marker  = 'x')
plt.scatter(time, prob_1, label = "Node 1", s=40, marker = 'x')
plt.scatter(time, prob_2, label = "Node 2", s=30, marker = 'x')
plt.scatter(time, prob_3, label = "Node 3", s=30, marker = 'x')
plt.legend(["Node 0" , 'Node 1', 'Node 2', 'Node3'], loc='upper right')
plt.show()

plt.title("Fisher Information for Quantum Walk on Graph")
plt.xlabel("Time in seconds")
plt.ylabel("Fisher Information")
plt.scatter(time, fi_0, label="Node 0", s=70, marker  = 'x')
plt.scatter(time, fi_1, label = "Node 1", s=40, marker = 'x')
plt.scatter(time, fi_2, label = "Node 2", s=40,c='black', marker = 'x')
plt.scatter(time, fi_2, label = "Node 3", s=10, marker = 'x')
plt.scatter(time, np.sum([fi_0, fi_1, fi_2, fi_3], axis=0), label = "Total", s=10, marker = 'x')
plt.legend(["Node 0" , 'Node 1', 'Node 2','Node3', 'Total Fisher Info'], loc='upper right')
plt.show()

total_prob = np.sum([prob_0,prob_1,prob_2,prob_3], axis=0)
plt.title("Total probability across nodes against time")
plt.xlabel("Time in seconds")
plt.ylabel("Total probability")
plt.plot(time,total_prob)
plt.show()
