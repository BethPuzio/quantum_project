#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 12:04:38 2021

@author: Beth
"""
import numpy as np
import math
import matplotlib.pyplot as plt
import sympy as sym

#Calulcating Fisher information distribution based on given probability distribution
def FisherInfo(prob_function, phase_array):
    a = sym.Symbol('a')
    FI_array = []
    diff = sym.diff(prob_function, a)
    for i in range (len(phase_array)):
        FI = (diff**2)/(prob_function)
        FI_array.append(FI.subs(a,phase_array[i]))
    return FI_array

#Defining probability distributions based given detection scenario
a = sym.Symbol('a')
T = 0.8
prob_10 = T*sym.sin(a/2)**2
prob_01 = T*sym.cos(a/2)**2
prob_02 = T*0.5*sym.sin(a)**2
prob_20 = T*0.5*sym.sin(a)**2
prob_11 = T*sym.cos(a)**2

p10_array = []
p01_array = []
p02_array = []
p20_array = []
p11_array = []
phase_array = []

for N in range(100):
    i = np.random.random()*math.pi 
    phase_array.append(i)

for i in range(len(phase_array)):
    p10_array.append(prob_10.subs(a, phase_array[i]))
    p01_array.append(prob_01.subs(a, phase_array[i]))
    p02_array.append(prob_02.subs(a, phase_array[i]))
    p20_array.append(prob_20.subs(a, phase_array[i]))
    p11_array.append(prob_11.subs(a, phase_array[i]))

plt.title("Phase and detector probability distribution for single photon input")
plt.xlabel("Phase introduced by the phase shifter in radians")
plt.ylabel("Probability of detection")
plt.scatter(phase_array, p10_array, label="Detector 1", s=10, marker  = 'x')
plt.scatter(phase_array, p01_array, label = "Detector 2", s=10, marker = 'x')
plt.legend(["Detector 1" , 'Detector 2'], loc='upper right')
plt.show()

plt.title("Phase and detector probability distribution for two photon input")
plt.xlabel("Phase introduced by the phase shifter in radians")
plt.ylabel("Probability of detection")
plt.scatter(phase_array, p02_array, label="Detector 1", s=30, marker  = 'x')
plt.scatter(phase_array, p20_array, label = "Detector 2", s=10, marker = 'x')
plt.scatter(phase_array, p11_array, label = "Coincidence detection", s=10, marker = 'x')
plt.legend(["Detector 1" , 'Detector 2', 'Coincidence detection'], loc='upper right')
plt.show()

plt.title("Fisher Information for single photon input")
plt.xlabel("Phase introduced by the phase shifter in radians")
plt.ylabel("Probability of detection")
plt.scatter(phase_array, FisherInfo(prob_10, phase_array), label="Detector 1", s=10, marker  = 'x')
plt.scatter(phase_array, FisherInfo(prob_01, phase_array), label = "Detector 2", s=10, marker = 'x')
plt.scatter(phase_array, np.sum([FisherInfo(prob_01, phase_array), FisherInfo(prob_10, phase_array)], axis=0), label = "Detector 2", s=10, marker = 'x')
plt.legend(["Detector 1" , 'Detector 2', 'Total Fisher Info'], loc='upper right')
plt.show()

plt.title("Fisher Information for two photon input")
plt.xlabel("Phase introduced by the phase shifter in radians")
plt.ylabel("Probability of detection")
plt.scatter(phase_array, FisherInfo(prob_20, phase_array), label="Detector 1", s=50, marker  = 'x')
plt.scatter(phase_array, FisherInfo(prob_02, phase_array), label = "Detector 2", s=10, marker = 'x')
plt.scatter(phase_array, FisherInfo(prob_11, phase_array), label = "Coincidence Detection", s=10, marker = 'x')
plt.scatter(phase_array, np.sum([FisherInfo(prob_20, phase_array), FisherInfo(prob_02, phase_array), FisherInfo(prob_11, phase_array)], axis=0), label = "Total Fisher Info", s=10, marker = 'x')
plt.legend(["Detector 1" , 'Detector 2', 'Coincidence Detection', 'Total Fisher info'], loc='upper right')
plt.show()
