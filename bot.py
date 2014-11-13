import SGD
import random
import collections
import math
import sys
import nltk
from nltk.corpus import switchboard
from nltk.corpus import nps_chat
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
##    print(turnA)
    if "?" in turnA and "yeah" in turnB:
##        print("QnA")
        phi["QnA"] = 1
    return phi

def runBot():

    TRAIN_SET_SIZE = 100
    DEV_SET_SIZE = 30
    random.seed(10)
    
    def speakerSeq(turn1, turn2):
        if turn1.speaker == turn2.speaker:
            return False
        return turn1.id < turn2.id

    tagged_turns = switchboard.tagged_turns()
    turns = switchboard.turns()
    trainExamples = []
    trainExamplesPos = []
    trainExamplesNeg = []
    #for i in range(1, len(turns)):
    for i in range(1, TRAIN_SET_SIZE):
        if speakerSeq(turns[i-1], turns[i]):
    ##        print(((turns[i-1], turns[i]), 1))
            trainExamples.append(((turns[i-1], turns[i]), 1))
            trainExamplesPos.append(((turns[i-1], turns[i]), 1))
    for i in range(1, TRAIN_SET_SIZE):
        randomInt = random.randint(0, TRAIN_SET_SIZE - 1)
        #if speakerSeq(turns[i-1], turns[randomInt]):
    ##        print(((turns[i-1], turns[i]), 1))
        while turns[i-1].speaker == turns[randomInt].speaker:
            randomInt = random.randint(0, TRAIN_SET_SIZE - 1)
        trainExamples.append(((turns[i-1], turns[randomInt]), -1))
        trainExamplesNeg.append(((turns[i-1], turns[randomInt]), -1))
##    print(trainExamplesPos)
##    print("*********")
##    print(trainExamplesNeg)
    testExamples = []
    testExamplesPos = []
    testExamplesNeg = []
    for i in range(1 + TRAIN_SET_SIZE, DEV_SET_SIZE + TRAIN_SET_SIZE):
        if speakerSeq(turns[i-1], turns[i]):
            testExamples.append(((turns[i-1], turns[i]), 1))
            testExamplesPos.append(((turns[i-1], turns[i]), 1))
    for i in range(1 + TRAIN_SET_SIZE, DEV_SET_SIZE + TRAIN_SET_SIZE):
        randomInt = random.randint(TRAIN_SET_SIZE, DEV_SET_SIZE + TRAIN_SET_SIZE - 1)
        #if speakerSeq(turns[i-1], turns[randomInt]):
    ##        print(((turns[i-1], turns[i]), 1))
        while turns[i-1].speaker == turns[randomInt].speaker:
            randomInt = random.randint(0, TRAIN_SET_SIZE - 1)
        testExamples.append(((turns[i-1], turns[randomInt]), -1))
        testExamplesNeg.append(((turns[i-1], turns[randomInt]), -1))
##    print(testExamplesPos)
##    print("*********")
##    print(testExamplesNeg)

    weights = learnPredictor(trainExamples, testExamples, sampleFeatureExtractor)

runBot()
