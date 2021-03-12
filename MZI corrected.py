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
    d = e(phase*1j)
    m = [[1,0], [0, d]]
    x = np.matmul(b,m)
    y=np.matmul(x,b)
    return np.matmul(path,y)

#Continuous time quantum walk on two nodes
def CTQW_components(path,time):
    b = [[s,s],[s,-s]]
    m = [[e(time*1j),0],[0,e(time*(-1j))]]
    x = np.matmul(b,m)
    y=np.matmul(x,b)
    return np.matmul(path,y)   
    
phase_array = []
d1_array = []
d2_array = []
n20_array = []
n02_array = []
n11_array = []


#Code to simulate two node quantum walk as comparison with the interferometer setup
for N in range(100):
    i = np.random.random()*math.pi 
    phase_array.append(i)
    
    # note, inout [s,s] for the two photon input so that the state is normalised
    x = [s,s]
    xf = MZI_components(x, i)
    
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

tstart = 0
tend = 10
time = np.linspace(tstart,tend,(tend-tstart)*20)
for i in range(len(time)):
    x = [[1,0],[0,1]]
    xf = CTQW_components(x, time[i])

    n20 = (abs(xf[0][0])**2)*(abs(xf[0][1])**2)
    n02 = (abs(xf[1][0])**2)*(abs(xf[1][1])**2)
    n11 = ((abs(xf[1][0])**2)*(abs(xf[0][1])**2))+((abs(xf[0][0])**2)*(abs(xf[1][1])**2))

    n20_array.append(n20)
    n02_array.append(n02)
    n11_array.append(n11)

plt.title("Two particle CTQW, MZI representation")
plt.xlabel("Time in seconds")
plt.ylabel("Probability")
plt.scatter(time, n20_array, label="Both at detector 1", s=50, marker  = 'x')
plt.scatter(time, n02_array, label = "Both at detector 2", s=10, marker = 'x')
plt.scatter(time, n11_array, label = "Coincidence detetion", s=10, marker = 'x')
plt.legend(["Node 1" , 'Node 2', 'Same Node'], loc='upper right')
plt.show()


