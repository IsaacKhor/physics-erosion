import cv2
import numpy as np
import sys
import keras.models as kmodels

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

classifier = kmodels.load_model(sys.argv[1])

def load_img(sim_no, step):
    img = cv2.imread('processed/{}/{}.jpg'.format(sim_no, step),
        cv2.IMREAD_GRAYSCALE)
    return img.reshape(50,50,1)

def load_sim(sim_no):
    data = [load_img(sim_no, x) for x in range(1,201)]
    return np.array(data)

def plot_sim_preds_over_time(sim_no, name):
    data = load_sim(sim_no)
    predictions = classifier.predict(data)
    fig = plt.figure()
    plt.plot(np.arange(0,201), predictions[:,0], label='left')
    plt.plot(np.arange(0,201), predictions[:,1], label='right')
    plt.legend()
    plt.savefig('model-evals/{}/sim-{}.png'.format(sim_no, name))
    plt.close(fig)


def plot_step_only(step, start, end, figname):
    data = np.array([load_img(x, step) for x in range(start,end)])
    data = data.reshape(start-end, 50,50,1)
    preds = classifier.predict(data)
    fig = plt.figure()
    plt.scatter(preds[:,0], preds[:,1], s=2)
    plt.savefig('figs/{}'.format(figname))
    plt.close(fig)
