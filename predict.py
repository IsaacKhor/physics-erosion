import cv2
import numpy as np
import sys
import keras.models as kmodels

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

classifier = kmodels.load_model('attempt-1.h5')

def load_sim(sim_no):
    data = []
    for step_no in range(1,202):
        img = cv2.imread('processed/Image-{}-{}.jpg'.format(sim_no, step_no),
            cv2.IMREAD_GRAYSCALE)
        img = img.reshape(50,50,1)
        data.append(img)

    return np.array(data)

for sim_no in range(2003, 2007):
    data = load_sim(sim_no)
    predictions = classifier.predict(data)

    fig = plt.figure()
    plt.plot(np.arange(0,201), predictions[:,0], label='left')
    plt.plot(np.arange(0,201), predictions[:,1], label='right')
    plt.legend()
    plt.savefig('figs/sim-{}.png'.format(sim_no))
    plt.close(fig)
