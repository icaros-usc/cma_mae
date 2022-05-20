import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import functools

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

filepath = 'obj_func.pdf'

def sampleFunction(f, start, end, numSamples):
    xs = []
    ys = []
    for i in range(numSamples):
        x = (end-start) * (i * 1.0 / numSamples) + start
        xs.append(x)
        ys.append(f(x))
    return xs, ys

startX = -8.0
endX = 8.0
numSamples = 10000

def sphere(x):
    return x ** 2

def rastrigin(x):
    return 10 + x ** 2 - 10 * np.cos(2 * np.pi * x ** 2)

def plateau(x):
    if abs(x) <= 5.12:
        return 0
    return (abs(x) - 5.12) ** 2

# Pick the function to show
f = plateau

xs, ys = sampleFunction(f, startX, endX, numSamples)

print(xs)
print(ys)

plt.plot(xs, ys, 'k')
plt.xticks([-8, 0, 8], fontsize=20)
plt.yticks([0, 90], fontsize=20)

# Draw plot 1 with uniform lines accross a value
plt.tight_layout()
#plt.show()

plt.savefig(filepath)

plt.close('all')
