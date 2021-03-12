#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 14:49:21 2021

@author: Beth
"""
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import sympy as sym
import scipy.linalg as la
from numpy.linalg import inv

#This code currently specifically applies to the two node graph, adjacency matrix defined as H

#Here we are using the code to model 2 particles on 3 nodes
#three node ring
H_3 = [[0,1,1],
       [1,0,1],
       [1,1,0]]

#three node line
H_4 = [[0,1,0],
       [1,0,1],
       [0,1,0]]

node30 = [1,0,0]
node31 = [0,1,0]
node32 = [0,0,1]


s = (1/np.sqrt(2))

A = nx.from_numpy_matrix(np.array(H_3))  
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
    return p_array, f_array

#Defining a function that returns node column vectors 
def nodes(H):
    M = np.eye(len(H), len(H))
    return np.hsplit(M, len(H))

#Probability and Fisher information distributions for the first particle 
prob_10, fi_10 = prob_dist(H_4, node30, node30)
prob_11, fi_11 = prob_dist(H_4, node30, node31)
prob_12, fi_12 = prob_dist(H_4, node30, node32)

#Probability and Fisher Info distributions for the second particle 
prob_20, fi_20 = prob_dist(H_4, node31, node30)
prob_21, fi_21 = prob_dist(H_4, node31, node31)
prob_22, fi_22 = prob_dist(H_4, node31, node32)

#both particles on node 0
prob2node0 = np.multiply(prob_10,prob_20)
fi2node0 = np.multiply(fi_10,fi_20)
#both particles on node 1
prob2node1 = np.multiply(prob_11, prob_21)
fi2node1 = np.multiply(fi_11, fi_21)
#both particles on node 2
prob2node2 = np.multiply(prob_12, prob_22)
fi2node2 = np.multiply(fi_12, fi_22)
#Particles on different nodes
arr = np.array([[np.multiply(prob_10,prob_21)],
                [np.multiply(prob_10,prob_22)],
                [np.multiply(prob_11,prob_20)],
                [np.multiply(prob_11,prob_22)],
                [np.multiply(prob_12,prob_20)],
                [np.multiply(prob_12,prob_21)]])
probdiffnode = arr.sum(axis=0)
fiarr = np.array([[np.multiply(fi_10,fi_21)],
                  [np.multiply(fi_10,fi_22)],
                  [np.multiply(fi_11,fi_20)],
                  [np.multiply(fi_11,fi_22)],
                  [np.multiply(fi_12,fi_20)],
                  [np.multiply(fi_12,fi_21)]])
fidiffnode = fiarr.sum(axis=0)

plt.title("Probability distribution of being at given node on graph, at time t")
plt.xlabel("Time in seconds")
plt.ylabel("Probability")
plt.scatter(time, prob2node0, label="Both particles on node 0", s=80, marker  = 'x')
plt.scatter(time, prob2node1, label = "Both particles on node 1", s=20, marker = 'x')
plt.scatter(time, prob2node2, label = "Both particles on node 2", s=20,c='magenta', marker = 'x')
plt.scatter(time, probdiffnode, label = "Particles on different nodes", s=20, marker = 'x')
plt.legend(["Both on node 0" , 'Both on node 1', 'Both on node 2','Different Nodes'], loc='upper right')
plt.show()

plt.title("Fisher Information for Quantum Walk on Graph")
plt.xlabel("Time in seconds")
plt.ylabel("Fisher Information")
plt.scatter(time, fi2node0, label="Both particles on node 0", s=80, marker  = 'x')
plt.scatter(time, fi2node1, label = "Both particles on node 1", s=20, marker = 'x')
plt.scatter(time, fi2node2, label = "Both particles on node 2", s=10, c='magenta', marker = 'x')
plt.scatter(time, fidiffnode, label = "Particles on different nodes", s=20, marker = 'x')
plt.scatter(time, np.sum([fi2node0, fi2node1, fi2node2, fidiffnode], axis=0), label = "Total", s=10, marker = 'x')
plt.legend(["Both on node 0" , 'Both on node 1', 'Both on node 2', 'Different nodes','The Fisher Info'], loc='upper right')
plt.show()


print(np.max(np.sum([fi2node0, fi2node1, fi2node2, fidiffnode], axis=0)))


