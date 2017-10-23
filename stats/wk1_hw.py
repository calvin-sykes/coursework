#!/usr/bin/python3

import numpy as np
from scipy.stats import norm
import math
import matplotlib.pyplot as plt

SEP='-'*37

def q1():
    print(SEP, 'Q1', SEP)
    data = np.array((11.45,10.91,11.60,10.59,10.32,10.34,11.00,10.94,11.67,11.06,10.57),dtype='f')
    mean = np.mean(data)
    sigma = np.std(data, ddof=1)
    stderr = sigma / np.sqrt(len(data))
    print('mean={}, sigma={}, stderr={}'.format(mean,sigma,stderr))

def q3():
    print(SEP, 'Q3', SEP)
    sigmas = range(1, 6)
    for sigma in sigmas:
        prob = norm.cdf(sigma, 0, 1) - norm.cdf(-sigma, 0, 1)
        print('Lies within {}σ {}% of the time'.format(sigma, prob * 100))
    print()
    fractions = [0.5, 0.99]
    for f in fractions:
        rge = norm.interval(f, 0, 1)
        print('Range containing {}% of data is +-{}σ'.format(f * 100, rge[1]))

def q4():
   print(SEP, 'Q4', SEP)
   data = np.array((45.7, 53.2, 48.4, 45.1, 51.4, 62.1, 49.3), dtype='f')
   mean = np.mean(data)
   dev = np.std(data, ddof=1)
   anomaly = data[5]
   prob = 1 - (norm.cdf(anomaly, mean, dev) - norm.cdf(-anomaly, mean, dev))
   crit = prob * len(data)
   print('Result from applying Chauvenet criterion is {}'.format(crit))
   if crit < 0.5:
       print('So datum should be treated as an outlier')
       np.delete(data, 5)
       newmean = np.mean(data)
       newdev = np.std(data, ddof=1)
       err = newdev / math.sqrt(len(data))
       print('Statistics following removal of outlier:\nmean is {}\nSD is {}\nerror is {}'
             .format(newmean, newdev, err))

def q5():
    print(SEP, 'Q5', SEP)
    data = np.random.poisson(35, 100000)
    normx = np.linspace(0, 70, 1000)
    normy = norm.pdf(normx, 35, math.sqrt(35))
    plt.figure()
    ax = plt.gca()
    ax.set_xlabel('x')
    ax.set_ylabel('Probability')
    plt.hist(data, bins = 50, normed=True, label='Poisson')
    plt.plot(normx, normy, c='r', label='Normal')
    plt.legend()
    plt.savefig('q5.pdf')
    plt.close('all')
    print('Saved plot as q5.pdf')

def q6():
    print(SEP, 'Q6', SEP)
    sigmas = range(2, 6)
    for sigma in sigmas:
        prob = 1 - (norm.cdf(sigma, 0, 1) - norm.cdf(-sigma, 0, 1))
        events = 1e5 * prob
        print('There will be {} {}σ events in a year'.format(events, sigma))
    prob2sigma = (1 - (norm.cdf(2, 0, 1) - norm.cdf(-2, 0, 1)))**2 * 0.01**2
    print('Probability of two 2σ events in same bin is {}\n\t(assuming uniform distribution)'
          .format(prob2sigma))
def qcode():
    print(SEP, 'QC', SEP)
    nums = np.random.normal(0.5, 0.2, 1000)
    plt.figure()
    ax = plt.gca()
    ax.set_xlabel('x')
    ax.set_ylabel('Frequency')
    ax.set_title('Histogram of values from normal distribution with mean = 0.5, σ = 0.2')
    plt.hist(nums, bins=100, range=[0,1])
    plt.savefig('normhist.pdf')
    plt.close('all')

    NSUB = 1000

    def pickn(data, n):
        return np.random.choice(data, n, replace=True)

    # take n randomly selected values and average them
    # repeat NSUB times and return as a list
    def sample_n_and_average(n):
        subnums = np.zeros(NSUB)
        for i in range(0, NSUB):
            subnums[i] = np.mean(pickn(nums, n))
        assert(len(subnums)==NSUB)
        return subnums

    plt.figure()
    
    for n in range(5, 1, -1):
        subnums = sample_n_and_average(n)
        plt.figure()
        ax = plt.gca()
        ax.set_xlabel('Mean x value')
        ax.set_ylabel('Frequency')
        ax.set_title('Histogram of means averaging over {} values'.format(n))
        plt.hist(subnums, bins = 20, range=[0,1])
        plt.savefig('normhist_avg_{}.pdf'.format(n))

if __name__ == '__main__':
    q1()
    q3()
    q4()
    q5()
    q6()
    qcode()
