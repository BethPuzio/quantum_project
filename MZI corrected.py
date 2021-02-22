#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 17:50:34 2021

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
    b = [[s,(s*1j)], [(s*1j),s]] 
    d = e(phase*1j*math.pi)
    m = [[1,0], [0, d]]
    x = np.matmul(b,m)
    y=np.matmul(x,b)
    return np.matmul(path,y)

phase_array = []
d1_array = []
d2_array = []


for N in range(100):
    i = np.random.random()*math.pi 
    phase_array.append(i)
    
    # note, inout [s,s] for the two photon input so that the state is normalised
    x = [s,s]
    print(x)
    xf = MZI_components(x, i)
    print(xf[0])
    

    d1 = abs(xf[0])**2
    d2 = abs(xf[1])**2

    d1_array.append(d1)
    d2_array.append(d2)
 
   
plt.title("Phase and detector probability distribution")
plt.xlabel("Phase introduced by the phase shifter in radians")
plt.ylabel("Probability of detection")
plt.scatter(phase_array, d1_array, label="Detector 1", s=10, marker  = 'x')
plt.scatter(phase_array, d2_array, label = "Detector 2", s=10, marker = 'x')
plt.legend(["Detector 1" , 'Detector 2'], loc='upper right')
plt.show()






