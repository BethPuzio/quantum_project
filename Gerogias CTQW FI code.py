#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 13:38:06 2021

@author: Beth
"""
import numpy as np
import math
from scipy.linalg import expm
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.animation as ani 
import sympy as sym


def line_hadamard(N):
    a = np.zeros((1, N))[0]
    b = np.ones((1, N-1))[0]
    H = np.diag(a, 0) + np.diag(b, -1) + np.diag(b, 1)

    A = nx.from_numpy_matrix(np.array(H))  
    #nx.draw_spectral(A, with_labels=True)
    #plt.title('Line Graph')
    #plt.show()

    return H 

t = sym.Symbol('t', real=True)
#simulating 2 node quantum walk symbolically 
H = sym.Matrix([[0,1],[1,0]])
u_ctqw = (H*1j*t).exp()

#Defining the start and end nodes as vectors k and j respectively
a = sym.Symbol('a', real=True)
b = sym.Symbol('b', real=True)
A = sym.Symbol('A', real=True)
B = sym.Symbol('B', real=True)

#defining the combinations of nodes and the matrix exponential
psi_k = sym.Matrix([a,b]) # starting at 0> 
psi_kt = u_ctqw*psi_k
psi_j = sym.Matrix([[A,B]])

#Probability of positioning on desired node, from calculating evolution from initial node
#multiplied by desired node 
prob = psi_j*psi_kt
#redefining probability distribution as magnitude squared of previous definitio
prob = sym.simplify(sym.Abs(prob)**2)
#differentiating probability distribution wrt t for the fisher information
dprob = sym.diff(prob,t)
fish_xx = (prob*dprob)[0]
fish_xx = sym.simplify(fish_xx)

#Defining nodes as vectors for the two node graph
state_0 = [1,0]
state_1 = [0,1]
states = [state_0, state_1]
fish = 0
#for the number of vector states in the two node matrix
for i in range(len(states)):
    #initial state
    psi_k = sym.Matrix(states[int(np.floor(2/2))])
    #initial state combined with unitary time evolution
    psi_kt = u_ctqw*psi_k
    #extracting final position node from the graph matrix
    psi_j = sym.Matrix(states[i]).T

    prob = psi_j*psi_kt
    prob = sym.simplify(sym.Abs(prob)**2)
    prob = prob[0]
    dprob = sym.diff(prob,t)
    fish_xx = ((1/prob)*(dprob**2))
    print(sym.simplify(fish_xx))
    fish += fish_xx

nodes = []
F = []

#Looping through for different sized graphs
for l in range(2,6):
    N = l
    nodes.append(N)
    states = []
    #Loop to define the node states corresponding to a graph of a given size
    for n in range(N):
        state = np.zeros(N)
        state[n] = 1
        states.append(state)
    
    #Defining the adjacency matrix using the function that returns it for a line of a given length
    H = sym.Matrix(line_hadamard(N))
    u_ctqw = (H*1j*t).exp()

    fish = 0
    for i in range(len(states)):
        #Taking last element in the list
        #Starting at the final node
        psi_k = sym.Matrix(states[-1])
        psi_kt = u_ctqw*psi_k
        #Looping through position nodes
        psi_j = sym.Matrix(states[i]).T
        
        #probability distribution for given node, starting at the final node
        prob = psi_j*psi_kt
        prob = sym.simplify(sym.Abs(prob)**2)
        prob = prob[0]
        #calculating the fihser info for given node
        dprob = sym.diff(prob,t)
        fish_xx = ((1/prob)*(dprob**2))
        fish += fish_xx
    print(N)
    #Returns the real part of the Fisher information equation, substituting 1 in for t
    print(sym.re(fish.evalf(subs={t:1})))

    F.append(sym.re(fish.evalf(subs={t:1})))