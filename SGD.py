#!/usr/bin/python

import random
import collections
import math
import sys
from collections import Counter
from util import *

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

def learnPredictor(trainExamples, testExamples, featureExtractor):
    '''
    COMMENTS FROM HOMEWORK CODE
    Given |trainExamples| and |testExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of iterations to
    train |numIters|, return the weight vector (sparse feature vector) learned.

    '''
    weight = {}
    numIters = 40
    stepSize = .05

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
        devError = evaluatePredictor(testExamples, testCache, weight)
        print ("Step %d: train error = %s, test error = %s \\\\" % (k + 1, trainError, devError))
    return weight
