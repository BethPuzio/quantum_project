#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 12:23:45 2020

@author: Beth
"""

import numpy as np
import matplotlib.pyplot as plt

#code to plot probability distribution for a classical random walk along a one dimensional line

#Defining the arrays for position and probabilities
line = list(range(-100,101))
prob = np.zeros(201)
P=1000

for j in range(P):
    position = np.zeros(201)
    N=100
    position[N] = 1
    # 100 coin flips before defining final position
    for i in range(100):
        #coin flip
        coin = np.random.randint(0,2)
        if coin == 0:
            N+= -1
        if coin == 1:
            N+=1
    prob[N]+=1

#function to remove zero values from arrays to neaten plots
def remove_zeros(x,y):
    x=list(x)
    y=list(y)
    n=0
    for i in range(len(y)):
        if y[i-n] == 0:
            y.remove(y[i-n])
            x.remove(x[i-n])
            n+=1
    return x,y
    

line, prob = remove_zeros(line,prob)

plt.title("Random Walk Probability Distribution")
plt.xlabel("Position along line")
plt.ylabel("Number of times finally found at given position")
plt.scatter(line, prob, marker  = 'x')
plt.show()