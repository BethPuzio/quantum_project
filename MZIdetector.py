#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 18:11:30 2020

@author: Beth
"""

import numpy as np
import math
import cmath
import matplotlib.pyplot as plt

sqrt = math.sqrt
s = (1/sqrt(2))
e = cmath.exp
np.set_printoptions(suppress=True)

#define beam splitters 

def beam_splitter_1(path): 
    b = [[s,s], [s,-s]] 
    return np.matmul(path,b)

def phase_shifter(path, phase): 
    d = e(phase*math.pi*1j)
    m = [[1,0], [0, d]]
    return np.matmul(path,m)

phase_array = []
d1_array = []
d2_array = []


for N in range(500):
    i = np.random.random()*math.pi 
    phase_array.append(i)
    
    x = [[0,1]]
    x2 = beam_splitter_1(x)
    x3 = phase_shifter(x2, i)
    x4 = beam_splitter_1(x3)

    d1 = x4[0][0].real.round(2)
    d2 = x4[0][1].real.round(2)

    d1_array.append(d1)
    d2_array.append(d2)
 
   
plt.title("Phase and detector probability distribution")
plt.xlabel("Phase introduced by the phase shifter in radians")
plt.ylabel("Probability of detection")
plt.scatter(phase_array, d1_array, label="Detector 1", marker  = 'x')
plt.scatter(phase_array, d2_array, label = "Detector 2", marker = 'x')
plt.legend(["Detector 1" , 'Detector 2'], loc='upper right')
plt.show()
#output plot shows a periodic trigonometric plot, as expected from the output state
# we calculated by hand. The first output mode and detector one shows a sine curve
# as predicted, and detector 2 shows a cos curve as predicted.