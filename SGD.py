#!/usr/bin/python

import random
import collections
import math
import sys
from collections import Counter
from util import *

############################################################
# Problem 2: binary classification
############################################################

############################################################
# Problem 2a: feature extraction

def extractWordFeatures(x):
    """
    Extract word features for a string x.
    @param string x: 
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_CODE (around 5 lines of code expected)
    phi = dict()
    y = x.split()
    for s in y:
        if s not in phi:
            phi[s] = 0
        phi[s] += 1
    return phi       
    # END_YOUR_CODE

############################################################
# Problem 2b: stochastic gradient descent

def learnPredictor(trainExamples, testExamples, featureExtractor):
    '''
    Given |trainExamples| and |testExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of iterations to
    train |numIters|, return the weight vector (sparse feature vector) learned.

    You should implement stochastic gradient descent.

    Note: only use the trainExamples for training!
    You should call evaluatePredictor() on both trainExamples and testExamples
    to see how you're doing as you learn after each iteration.
    '''
    weight = {}  # feature => weight
    # BEGIN_YOUR_CODE (around 15 lines of code expected)
    numIters = 15
    stepSize = .05
    for k in range(numIters):
        for example in trainExamples:
            x = example[0]
            y = example[1]
            phi = featureExtractor(x)
            score = dotProduct(weight, phi)
            margin = score * y
            coef = 0
            if margin <= 1:
                coef = -1
            increment(weight, -1*stepSize*coef*y,phi)
        def predictor(x):
            if dotProduct(weight, featureExtractor(x)) <= 0:
                return -1
            else:
                return 1
        trainError = evaluatePredictor(trainExamples, predictor)
        devError = evaluatePredictor(testExamples, predictor)
        print("Step %d: train error = %s, test error = %s \\\\" % (k + 1, trainError, devError))
    # END_YOUR_CODE
    return weight

############################################################
# Problem 2c: generate test case

def generateDataset(numExamples, weights):
    '''
    Return a set of examples (phi(x), y) randomly which are classified correctly by
    |weights|.
    '''
    random.seed(42)
    # Return a single example (phi(x), y).
    # phi(x) can be anything (randomize!) with a nonzero score under the given weight vector
    # y should be 1 or -1 as classified by the weight vector.
    def generateExample():
        # BEGIN_YOUR_CODE (around 5 lines of code expected)
        phi = {}
        for key in weights:
            phi[key] = random.randint(1,10)
        y = 1
        if dotProduct(weights, phi) < 0:
            y = -1
        # END_YOUR_CODE
        return (phi, y)
    return [generateExample() for _ in range(numExamples)]

############################################################
# Problem 2f: character features

def extractCharacterFeatures(n):
    '''
    Return a function that takes a string |x| and returns a sparse feature
    vector consisting of all n-grams of |x| without spaces.
    EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    '''
    def extract(x):
        # BEGIN_YOUR_CODE (around 10 lines of code expected)
        y = x.replace(" ","").replace("\t","")
        v = {}
        for i in range(len(y) - n + 1):
            ngram = y[i:i+n]
            if ngram not in v:
                v[ngram] = 0
            v[ngram] += 1
        return v
        # END_YOUR_CODE
    return extract

############################################################
# Problem 2h: extra credit features

def extractExtraCreditFeatures(x):
    # BEGIN_YOUR_CODE (around 5 lines of code expected)
    raise Exception("Not implemented yet")
    # END_YOUR_CODE

############################################################
# Problem 3: k-means
############################################################


def kmeans(examples, K, maxIters):
    '''
    examples: list of examples, each example is a string-to-double dict representing a sparse vector.
    K: number of desired clusters
    maxIters: maximum number of iterations to run for (you should terminate early if the algorithm converges).
    Return: (length K list of cluster centroids,
            list of assignments, (i.e. if examples[i] belongs to centers[j], then assignments[i] = j)
            final reconstruction loss)
    '''
    # BEGIN_YOUR_CODE (around 35 lines of code expected)
    # raise Exception("Not implemented yet")

    def dist2(v, w):
        sum = 0
        for a in v :
            if a in w:
                sum += (v[a]-w[a])**2
            else:
                sum += v[a]**2
        for a in w:
            if a not in v:
                sum += w[a]**2
        return sum
        
    def assign(examples, assignments, centers):
        for i in range(len(examples)):
            maxDistance = dist2(examples[i],centers[0])
            for j in range(len(centers)):
                distance = dist2(examples[i],centers[j])
                if distance <= maxDistance:
                    maxDistance = distance
                    assignments[i] = j
    
    def center(examples, assignments, centers):
        for j in range(len(centers)):
            vec = {}
            count = 0
            for i in range(len(assignments)):
                if assignments[i] == j :
                    count += 1
                    increment(vec, 1, examples[i])
            if count != 0:
                for a in vec:
                    vec[a] = 1.0*vec[a]/count
                centers[j]=vec

    def loss(assignments, centers):
        sum = 0
        for i in range(len(assignments)):
           sum += dist2(examples[i],centers[assignments[i]])
        return sum
               
    centers = random.sample(examples, K)
    assignments = {}
    oldAssignments = {}
    oldCenters = {}
    for i in range(maxIters):
        assign(examples, assignments, centers)
        center(examples, assignments, centers)
        if oldAssignments == assignments and oldCenters == centers:
            break
        oldAssignments = assignments
        oldCenters = centers     
    return (centers, assignments, loss(assignments, centers))
    # END_YOUR_CODE
