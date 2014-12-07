#!/usr/bin/python

import random
import collections
import math
import sys
import pdb
from collections import Counter
from util import *
from config import *

def predictor(x, weight):
    if dotProduct(weight, x) <= 0:
        return -1
    else:
        return 1

def evaluatePredictor(examples, cache, weight):
    '''
    predictor: a function that takes an x and returns a predicted y.
    Given a list of examples (x, y), makes predictions based on |predict| and returns the fraction
    of misclassiied examples.
    '''
    error = 0
    for i in range(len(examples)):
        if predictor(cache[i], weight) != examples[i][1]:
            error += 1
    return 1.0 * error / len(examples)

def converges(weightPrev, weight):
    # print weight
    # pdb.set_trace()
    for key in weight:
        # print key
        # print weight[key]
        # print weightPrev[key]
        if abs(weightPrev[key] - weight[key]) > CONVERGENCE_MAX:
            return False
    return True

def converges_avg(weightPrev, weight):
    summ = 0
    for key in weight:
        summ += abs(weightPrev[key] - weight[key])
    if 1.0 * summ / len(weight) > CONVERGENCE_MAX:
            return False
    return True

def learnPredictor(trainExamples, testExamples, featureExtractor):
    '''
    Given |trainExamples| and |testExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of iterations to
    train |numIters|, return the weight vector (sparse feature vector) learned.

    '''

    global stepSize

    weight = {}
    weightPrev = None


    print "Caching..."
    trainCache = [featureExtractor(example[0]) for example in trainExamples]
    testCache = [featureExtractor(example[0]) for example in testExamples]
    print "Done Caching"

    for k in range(numIters):
        for i in range(len(trainExamples)):
##            x = trainExamples[i][0]
            y = trainExamples[i][1]
            phi = trainCache[i]
            score = dotProduct(weight, phi)
            margin = score * y
            coef = 0
            if margin <= 1:
                coef = -1
            increment(weight, -1*stepSize*coef*y,phi)
        trainError = evaluatePredictor(trainExamples, trainCache, weight)
        testError = evaluatePredictor(testExamples, testCache, weight)
        stepSize = stepSize * dampeningFactor
        print ("Step %d: train error = %s, test error = %s \\\\" % (k + 1, trainError, testError))
        # if weightPrev and converges(weightPrev, weight):
        #     print "Converged"
        #     break
        # if weightPrev and converges_avg(weightPrev, weight):
        #     print "Converged average"
        #     break
        weightPrev = dict(weight)
    return weight
