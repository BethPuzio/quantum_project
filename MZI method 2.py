#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 18:06:04 2021

@author: Beth
"""
import numpy as np
import math
import cmath
import matplotlib.pyplot as plt
import sympy as sym


sqrt = math.sqrt
s = (1/sqrt(2))
e = sym.exp
np.set_printoptions(suppress=True)
p = sym.Symbol('p') 

def MZI_components(path): 
    b = [[s,(s*1j)], [(s*1j),s]] 
    d = e(p*1j)
    m = np.array([[1,0], [0, d]])
    x = np.matmul(b,m)
    y=np.matmul(x,b)
    return np.matmul(path,y)

phase_array = []
p10_array = []
p01_array = []

for N in range(100):
    i = np.random.random()*math.pi 
    phase_array.append(i)
    
    # note, inout [s,s] for the two photon input so that the state is normalised
    x = [1,0]
    xf = MZI_components(x)
    p10 = abs((np.conj(xf[0].subs(p,i)))*(xf[0].subs(p,i)))
    p10 = p10.rewrite(sym.exp).simplify()
    p10_array.append(p10)

    p01 = abs((np.conj(xf[1].subs(p,i)))*(xf[1].subs(p,i)))
    p01 = p01.rewrite(sym.exp).simplify()
    p01_array.append(p01)

plt.title("Phase and detector probability distribution")
plt.xlabel("Phase introduced by the phase shifter in radians")
plt.ylabel("Probability of detection")
plt.scatter(phase_array, p10_array, label="Detector 1", s=10, marker  = 'x')
plt.scatter(phase_array, p01_array, label = "Detector 2", s=10, marker = 'x')
plt.legend(["Detector 1" , 'Detector 2'], loc='upper right')
plt.show()