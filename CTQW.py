#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 10:17:28 2021

@author: Beth
"""
import numpy as np
import math
from scipy.linalg import expm
import matplotlib.pyplot as plt
import networkx as nx


#Hamiltonian matrix that is also the adjacency matrix of the graph
#The adjacency matrix most important for defining the directions in which the particle can evolve
#H_1 adjacency matrix for the 1D line
H_1 = [[0,1,0,0,0],
       [1,0,1,0,0],
       [0,1,0,1,0],
       [0,0,1,0,1],
       [0,0,0,1,0]]
#H_2 slightly more complicated graph
H_2 = [[0,1,1,1,0],
       [1,0,0,1,1],
       [1,0,0,0,1],
       [1,1,0,0,0],
       [0,1,1,0,0]]

H_3 = [[0,1,0,0,0,0,0,1],
       [1,0,1,0,0,0,0,0],
       [0,1,0,1,0,0,0,0],
       [0,0,1,0,1,0,0,0],
       [0,0,0,1,0,1,0,0],
       [0,0,0,0,1,0,1,0],
       [0,0,0,0,0,1,0,1],
       [1,0,0,0,0,0,1,0]]

#Defining initial vertex position of the particle on the graph
#Defining the vertices of the graph on which the walk takes place
#For any size Hamiltonian/adjacency matrix
def positioning(H):
    P=np.zeros(len(H))
    P[math.ceil(len(P)/2)-1] = 1
    G=np.zeros(len(H))
    for i in range(len(H)):
        G[i]=i
    return P,G

#Setting the Hamiltonian to be used
H=H_3

P,G=positioning(H)


#Defining the unitary evolution of the walk
def unitary(P,H,t):
    x = np.dot(1j,H)
    U = expm(np.dot(t,x))
    psi = np.matmul(P,U)
    prob=0
    for i in range(len(psi)):
        psi[i]=abs(psi[i])**2
        prob+=psi[i]
    return psi,prob
    

t=10
psi_out,prob = unitary(P,H,t)
#check that probabilities sum to one
print(prob)

A = nx.from_numpy_matrix(np.array(H))  
nx.draw(A, with_labels=True, node_color="lightgreen")

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.set_ylabel('Probability of detection at given position')
ax.set_title('Position on graph at time {}s, particle starting at position {}'.format(t, G[math.ceil(len(P)/2)-1]))
ax.bar(G,psi_out)