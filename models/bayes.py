import sys
import os

import cv2
import sklearn
import numpy as np

import sklearn.naive_bayes as snb

def img_to_lr_point(img):
    sum_cols = img.sum(axis=0)
    return [sum_cols[0:25].sum(), sum_cols[25:50].sum()]

data = []
labels = []

print('[LOG] Loading labels')
with open('userin.txt', 'r') as f:
    ln = f.readline()

for i in range(1,2001):
    img = cv2.imread('processed/{}/1.jpg'.format(i), cv2.IMREAD_GRAYSCALE)
    lrpair = img_to_lr_point(img)

    # 's' is right, 'h' is left, 'b' both, 'c' neither
    if ln[i] == 's':
        data.append(lrpair)
        labels.append(1)
    elif ln[i] == 'h':
        data.append(lrpair)
        labels.append(0)

data = np.array(data)
labels = np.array(labels)

# Plot
colors = ['#FF0000' if x == 0 else '#0000FF' for x in labels]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig = plt.figure()
plt.scatter(data[:,0], data[:,1], c=colors)
plt.savefig('plot_alldata.png')
plt.close(fig)
