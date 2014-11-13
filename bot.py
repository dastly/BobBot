import SGD
import random
import collections
import math
import sys
import nltk
from nltk.corpus import switchboard
from SGD import learnPredictor

def sampleFeatureExtractor(x):
    phi = dict()
    turnA = x[0]
    turnB = x[1]
    phi["Has same word"] = 0
    for wordA in turnA:
        for wordB in turnB:
            if wordA == wordB:
                phi["Has same word"] += 1
    return phi

def runBot():

    TRAIN_SET_SIZE = 100
    DEV_SET_SIZE = 30
    
    
    def speakerSeq(turn1, turn2):
        if turn1.speaker == turn2.speaker:
            return False
        return turn1.id < turn2.id

    tagged_turns = switchboard.tagged_turns()
    turns = switchboard.turns()
    trainExamples = []
    #for i in range(1, len(turns)):
    for i in range(1, TRAIN_SET_SIZE):
        if speakerSeq(turns[i-1], turns[i]):
    ##        print(((turns[i-1], turns[i]), 1))
            trainExamples.append(((turns[i-1], turns[i]), 1))
    for i in range(1, TRAIN_SET_SIZE):
        randomInt = random.randint(0, TRAIN_SET_SIZE - 1)
        if speakerSeq(turns[i-1], turns[randomInt]):
    ##        print(((turns[i-1], turns[i]), 1))
            trainExamples.append(((turns[i-1], turns[randomInt]), -1))
    ##print(trainExamples)
    testExamples = []
    for i in range(1 + TRAIN_SET_SIZE, DEV_SET_SIZE + TRAIN_SET_SIZE):
        if speakerSeq(turns[i-1], turns[i]):
            testExamples.append(((turns[i-1], turns[i]), 1))
    for i in range(1 + TRAIN_SET_SIZE, DEV_SET_SIZE + TRAIN_SET_SIZE):
        randomInt = random.randint(TRAIN_SET_SIZE, DEV_SET_SIZE + TRAIN_SET_SIZE - 1)
        if speakerSeq(turns[i-1], turns[randomInt]):
    ##        print(((turns[i-1], turns[i]), 1))
            trainExamples.append(((turns[i-1], turns[randomInt]), -1))
    weights = learnPredictor(trainExamples, testExamples, sampleFeatureExtractor)

runBot()
