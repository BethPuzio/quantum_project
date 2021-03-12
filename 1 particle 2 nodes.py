#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 10:22:22 2021

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

s = (1/np.sqrt(2))

A = nx.from_numpy_matrix(np.array(H))  
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
    return p_array, f_array, prob_func

#Defining a function that returns node column vectors 
def nodes(H):
    M = np.eye(len(H), len(H))
    return np.hsplit(M, len(H))


prob_0, fi_0, nodefunc0 = prob_dist(H, node0, node0)
prob_1, fi_1, nodefunc1 = prob_dist(H, node0, node1)


plt.title("Probability distribution of being at given node on graph, at time t")
plt.xlabel("Time in seconds")
plt.ylabel("Probability")
plt.scatter(time, prob_0, label="Node 0", s=20, marker  = 'x')
plt.scatter(time, prob_1, label = "Node 1", s=10, marker = 'x')
plt.legend(["Node 0" , 'Node 1'], loc='upper right')
plt.show()

plt.title("Fisher Information for Quantum Walk on Graph")
plt.xlabel("Time in seconds")
plt.ylabel("Fisher Information")
plt.scatter(time, fi_0, label="Node 0", s=20, marker  = 'x')
plt.scatter(time, fi_1, label = "Node 1", s=10, marker = 'x')
plt.scatter(time, np.sum([fi_0, fi_1], axis=0), label = "Total", s=10, marker = 'x')
plt.legend(["Node 0" , 'Node 1', 'Total Fisher Info'], loc='upper right')
plt.show()

print(nodefunc0, nodefunc1)