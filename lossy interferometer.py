#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 17:54:46 2021

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

def MZI_components(path, phase):
    #Beam splitter matrix
    b = [[s,(s*1j)], [(s*1j),s]] 
    d = e(phase*math.pi*1j)
    #Phase shifter matrix
    m = [[1,0], [0, d]]
    #Lossy beam splitter example
    l = [[0.5,0], [0,1]]
    #Multiplying together the interferometer matrix elements to apply
    x = np.matmul(b,m)
    y=np.matmul(x,l)
    z=np.matmul(y,b)
    return np.matmul(path,z)

phase_array = []
d1_array = []
d2_array = []


for N in range(500):
    i = np.random.random()*math.pi 
    phase_array.append(i)
    
    x = [s,s]
    xf = MZI_components(x, i)

    d1 = abs(xf[0])**2
    d2 = abs(xf[1])**2

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