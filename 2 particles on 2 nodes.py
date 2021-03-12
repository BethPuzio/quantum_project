#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 19:43:37 2021

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

t = sym.Symbol('t')


tstart = 0
tend = 10
time = np.linspace(tstart,tend,(tend-tstart)*15)

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
    return p_array, f_array, prob_func, FI

#Defining a function that returns node column vectors 
def nodes(H):
    M = np.eye(len(H), len(H))
    return np.hsplit(M, len(H))


prob_0, fi_0, pf_00, FI00 = prob_dist(H, node0, node0)
prob_1, fi_1, pf_01, FI01 = prob_dist(H, node0, node1)
prob_10, fi_10, pf_10, FI10 = prob_dist(H, node1, node0)
prob_11, fi_11, pf_11, FI11 = prob_dist(H, node1, node1)

print((pf_01*pf_10)+(pf_00*pf_11))

prob20 = np.multiply(prob_0,prob_10)
prob21 = np.multiply(prob_1, prob_11)
prob11 = np.add(np.multiply(prob_1, prob_10),np.multiply(prob_0, prob_11))

FI20 = np.multiply(fi_0,fi_10)
FI02 = np.multiply(fi_1, fi_11)
FI211 = np.add(np.multiply(fi_1, fi_10),np.multiply(fi_0, fi_11))

plt.title("Two Particles on Two Nodes")
plt.xlabel("Time in seconds")
plt.ylabel("Probability")
plt.scatter(time, prob20, label="Both on node 0", s=60, marker  = 'x')
plt.scatter(time, prob21, label = "Both on node 1", s=10, marker = 'x')
plt.scatter(time, prob11, label = "one on each node", s=30, marker = 'x')
plt.legend(["Both on node 0" , 'Both on node 1', 'one on each node'], loc='upper right')
plt.show()

plt.title("Fisher Information for two particle Quantum Walk on Graph")
plt.xlabel("Time in seconds")
plt.ylabel("Fisher Information")
plt.scatter(time, FI20, label="Both on node 0", s=60, marker  = 'x')
plt.scatter(time, FI02, label = "Both on node 1", s=10, marker = 'x')
plt.scatter(time, FI211, label = "One on each node", s=30, marker = 'x')
plt.scatter(time, np.sum([FI20, FI02, FI211], axis=0), label = "Total", s=10, marker = 'x')
plt.legend(["Node 0" , 'Node 1','One on each node', 'The Fisher Info'], loc='upper right')
plt.show()


