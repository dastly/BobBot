
#!/usr/bin/python
# Filename: bot.py

import random, collections, math, sys, getopt, pdb
from swda import Transcript
from swda import CorpusReader
from SGD import learnPredictor

from features import swda_feature_extractor
from features import baseline_feature_extractor

from util import dotProduct
from bot_utils import *     #processUtterances, getPosExamples, isBadTrun, getNegExamples, printExamples

from evaluation import *
from actual_chat_bot import swda_chat

from config import *

def runBot(train_flag): 

    random.seed(RANDOM_SEED)
    
    trainExamples = []
    trainExamplesPosList = []
    trainExamplesNegList = []

    testExamples = []
    testExamplesPosList = []
    testExamplesNegList = []

    turnSet = []

    print "Generating Examples..."
    count = 0
    for transcript in CorpusReader('swda').iter_transcripts(display_progress=False):
        turns = processUtterances(transcript)
        turnSet.append(turns)
        if count < TRAIN_SET_SIZE:
##        if count >= TRAIN_SET_SIZE and count <  TEST_SET_SIZE + TRAIN_SET_SIZE:
            trainExamplesPos = getPosExamples(turns)   
            trainExamplesNeg = getNegExamples(turns)
            trainExamplesPosList.append(trainExamplesPos)
            trainExamplesNegList.append(trainExamplesNeg)
            trainExamples.extend(trainExamplesPos)
            trainExamples.extend(trainExamplesNeg)
            count = count + 1
        elif count < TEST_SET_SIZE + TRAIN_SET_SIZE:
            count = count + 1
        elif count < TEST_SET_SIZE + TRAIN_SET_SIZE + FINAL_TEST_SET_SIZE:
##        elif count < TRAIN_SET_SIZE:
            testExamplesPos = getPosExamples(turns)   
            testExamplesNeg = getNegExamples(turns)
            testExamplesPosList.append(testExamplesPos)
            testExamplesNegList.append(testExamplesNeg)
            testExamples.extend(testExamplesPos)
            testExamples.extend(testExamplesNeg)
            count = count + 1
        else:
            break
    
    printTagCount(turnSet)
    printTagSetCount(turnSet)
    printNumBadTurns(turnSet)
    printLengthStats()
    printInterruptionStats(turnSet)
    
    weights = None
    if train_flag:
        print "Training Predictor..."
        calculatedWeights = learnPredictor(trainExamples, testExamples, swda_feature_extractor)
        save_weights_to_file(WEIGHTS_FILENAME, calculatedWeights)
        weights = calculatedWeights

    if not weights:
        weights = read_weights_from_file(WEIGHTS_FILENAME)

    #swda_chat returns false if the chatting is totally over
    while swda_chat(weights, swda_feature_extractor, turnSet):
        print "Finished a conversation"

    findExampleStats(testExamples, weights, swda_feature_extractor)
    printExamples(testExamples, weights, swda_feature_extractor)
    printWeightStatistics(weights, 5)

    print "Evaluating Bot..."
    print "Train Choosing Score: {0}".format(evaluate(trainExamplesPosList, chooseEval, weights))
    print "Test Choosing Score: {0}".format(evaluate(testExamplesPosList, chooseEval, weights))
    print "Train Guessing Score: {0}".format(evaluate(trainExamplesPosList, guessEval, weights))
    print "Test Guessing Score: {0}".format(evaluate(testExamplesPosList, guessEval, weights))

def usage():
    print 'Usage: python bot.py [-t]'
    print '-t OR --train = to train the chatbot'
    
if __name__ == "__main__":
    opts, args = getopt.getopt(sys.argv[1:], "t", ["train"])
    train_flag = False
    for opt, arg in opts:
        if opt in ('-t', '--train'):
            train_flag = True
    runBot(train_flag)
