#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 11:12:58 2021

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

A = nx.from_numpy_matrix(np.array(H_4))  
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
    return p_array, f_array, prob_func

#Defining a function that returns node column vectors 
def nodes(H):
    M = np.eye(len(H), len(H))
    return np.hsplit(M, len(H))


prob_0, fi_0, pfunc00 = prob_dist(H_3, node30, node30)
prob_1, fi_1, pfunc01 = prob_dist(H_3, node30, node31)
prob_2, fi_2, pfunc02 = prob_dist(H_3, node30, node32)

print(pfunc00,pfunc01,pfunc02)

plt.title("Probability distribution of being at given node on graph, at time t")
plt.xlabel("Time in seconds")
plt.ylabel("Probability")
plt.scatter(time, prob_0, label="Node 0", s=60, marker  = 'x')
plt.scatter(time, prob_1, label = "Node 1", s=40, marker = 'x')
plt.scatter(time, prob_2, label = "Node 2", s=30, marker = 'x')
plt.legend(["Node 0" , 'Node 1', 'Node 2'], loc='upper right')
plt.show()

plt.title("Fisher Information for Quantum Walk on Graph")
plt.xlabel("Time in seconds")
plt.ylabel("Fisher Information")
plt.scatter(time, fi_0, label="Node 0", s=70, marker  = 'x')
plt.scatter(time, fi_1, label = "Node 1", s=40, marker = 'x')
plt.scatter(time, fi_2, label = "Node 2", s=35, marker = 'x')
plt.scatter(time, np.sum([fi_0, fi_1, fi_2], axis=0), label = "Total", s=10, marker = 'x')
plt.legend(["Node 0" , 'Node 1', 'Node 2', 'Total Fisher Info'], loc='upper right')
plt.show()
