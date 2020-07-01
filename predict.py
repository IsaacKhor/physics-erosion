import math
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
    plt.plot(np.arange(1,201), predictions[:,0], label='left')
    plt.plot(np.arange(1,201), predictions[:,1], label='right')
    plt.legend()
    plt.savefig('model-evals/{}/sim-{}.png'.format(name, sim_no))
    plt.close(fig)

def plot_step_only(step, start, end, figname):
    data = np.array([load_img(x, step) for x in range(start,end)])
    data = data.reshape(start-end, 50,50,1)
    preds = classifier.predict(data)
    fig = plt.figure()
    plt.scatter(preds[:,0], preds[:,1], s=2)
    plt.savefig('figs/{}'.format(figname))
    plt.close(fig)

# Measure shortest distance from (1,0), (0,1), (1,1)
def dists_to_extreme(pp):
    d10 = math.sqrt((pp[0] - 1) ** 2 + pp[1] ** 2)
    d01 = math.sqrt((pp[0]) ** 2 + (pp[1] - 1) ** 2)
    d11 = math.sqrt((pp[0] - 1) ** 2 + (pp[1] - 1) ** 2)
    return np.array([d10, d01, d11, min(d10, d01, d11)])

def avgdist_stddev_to_pred(step):
    data = np.array([load_img(x, step) for x in range(2001,4001)])
    data = data.reshape(2000, 50, 50, 1)
    preds = classifier.predict(data)
    mindists = [dists_to_extreme(x) for x in preds]
    return (np.average(mindists, axis=0), np.std(mindists, axis=0))

def scatter_step_md_dist(step, path):
    data = np.array([load_img(x,step) for x in range(2001, 4001)])
    data.reshape(2000, 50, 50, 1)
    preds = classifier.predict(data)
    mindists = np.array([dists_to_extreme(x) for x in preds])
    fig = plt.figure()
    plt.hist(mindists[:,3], bins='auto')
    plt.savefig(path)
    plt.close(fig)

def plot_avgmd_ot(path):
    avgs = np.array([avgdist_stddev_to_pred(x) for x in range(1, 201)])
    fig = plt.figure()
    plt.plot(np.arange(1,201), avgs[:,0,0], label='1,0')
    plt.plot(np.arange(1,201), avgs[:,0,1], label='0,1')
    plt.plot(np.arange(1,201), avgs[:,0,2], label='1,1')
    plt.plot(np.arange(1,201), avgs[:,0,3], label='min')
    plt.plot(np.arange(1,201), avgs[:,1,0], label='sd 1,0')
    plt.plot(np.arange(1,201), avgs[:,1,1], label='sd 0,1')
    plt.plot(np.arange(1,201), avgs[:,1,2], label='sd 1,1')
    plt.plot(np.arange(1,201), avgs[:,1,3], label='sd min')
    plt.legend()
    plt.savefig(path)
    plt.close(fig)

def mindist_to_extreme(pp):
    d10 = math.sqrt((pp[0] - 1) ** 2 + pp[1] ** 2)
    d01 = math.sqrt((pp[0]) ** 2 + (pp[1] - 1) ** 2)
    d11 = math.sqrt((pp[0] - 1) ** 2 + (pp[1] - 1) ** 2)
    return min(d10, d01, d11)

# Measure as a ratio of stream progression
def progression_row(img, thres):
    num_over_thres = np.asarray([(img[i] < thres).sum() for i in range(0,50)])
    return 49 - np.argmax(num_over_thres > 1)

def plot_perf_to_ratio(path, start_sim, end_sim):
    prog_to_preds = [list() for i in range(0,50)]
    for i in range(start_sim, end_sim):
        sim_imgs = load_sim(i)
        progs = [progression_row(x, 50) for x in sim_imgs]
        preds = classifier.predict(sim_imgs)
        for j in range(0,len(progs)):
            progression = progs[j]
            prediction = preds[j]
            pred_mindist = mindist_to_extreme(prediction)
            prog_to_preds[progression].append(pred_mindist)

    # Delete last 2 elements (for reasons)
    del prog_to_preds[-1]
    del prog_to_preds[-1]

    md_avgs = [np.average(x) for x in prog_to_preds]
    md_median = [np.percentile(x, 50) for x in prog_to_preds]
    md_stddev = [np.std(x) for x in prog_to_preds]
    md_ci_upper = [ mean + sd for (mean,sd) in zip(md_avgs, md_stddev)]
    md_ci_lower = [ mean - sd for (mean,sd) in zip(md_avgs, md_stddev)]
    md_80_pc = [np.percentile(x, 80) for x in prog_to_preds]
    md_20_pc = [np.percentile(x, 20) for x in prog_to_preds]

    fig = plt.figure()
    plt.plot(np.arange(0,48), md_avgs, label='avg')
    plt.plot(np.arange(0,48), md_median, label='median')
    plt.plot(np.arange(0,48), md_stddev, label='stddev')
    plt.plot(np.arange(0,48), md_ci_upper, label='66% CI')
    plt.plot(np.arange(0,48), md_ci_lower, label='66% CI')
    plt.plot(np.arange(0,48), md_80_pc, label='80th percentile')
    plt.plot(np.arange(0,48), md_20_pc, label='20th percentile')
    plt.legend()
    plt.savefig(path)
    plt.close(fig)

    return prog_to_preds
