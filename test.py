import numpy
import cv2
import random

n1 = 100
n2 = 75
bruit = 1/10
center = [random.random()*(2-1)*3,random.random()*(2-1)*3]
radius = random.random()
deformation = 2

template1 = numpy.array([
    [numpy.cos(i*2*numpy.pi/n1)*radius*deformation for i in range(n1)], 
    [numpy.sin(i*2*numpy.pi/n1)*radius for i in range(n1)]
])

data1 = numpy.array([
    [numpy.cos(i*2*numpy.pi/n2)*radius*(1+random.random()*bruit)+center[0] for i in range(n2)], 
    [numpy.sin(i*2*numpy.pi/n2)*radius*deformation*(1+random.random()*bruit)+center[1] for i in range(n2)]
])
template2 = cv2.imread("map00.png")

data2 = cv2.imread("map02.png")
data2 = cv2.resize(data2,(2048,2048))
new = [[],[]]

import matplotlib.pyplot

for x,v_x in enumerate(data2):
    for y,v_y in enumerate(v_x):
        if len(new[0]) == 0 and sum(v_y) != 0:
            new[0].append(x-1024)
            new[1].append(y-1024)
        elif sum(v_y) != 0 and ( (new[0][len(new[0])-1] - (x-1024))**2 + (new[1][len(new[1])-1] - (y-1024))**2  )**0.5 < 10:
            new[0].append(x-1024)
            new[1].append(y-1024)
        elif sum(v_y) != 0:
            matplotlib.pyplot.plot(new[0], new[1], 'r')
            new[0] = []
            new[1] = []
            new[0].append(x-1024)
            new[1].append(y-1024)
            

matplotlib.pyplot.show()

# from numpy import *
# import math
# import matplotlib.pyplot as plt

# t = linspace(0, 2*math.pi, 400)
# a = sin(t)
# b = cos(t)
# c = a + b

# plt.plot(t, a, 'r') # plotting t, a separately 
# plt.plot(t, b, 'r') # plotting t, b separately 
# plt.plot(t, c, 'g') # plotting t, c separately 
# plt.show()