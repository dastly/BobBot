##from swda import Transcript
##trans = Transcript('swda/sw00utt/sw_0001_4325.utt.csv', 'swda/swda-metadata.csv')
##print trans

#!/usr/bin/python
# Filename: bot.py

import random
import collections
import math
import sys

import swda
from swda import Transcript
from swda import CorpusReader
import pdb

import SGD
from SGD import learnPredictor

from features import swda_feature_extractor
from features import baseline_feature_extractor

from util import dotProduct
from bot_utils import *     #processUtterances, getPosExamples, isBadTrun, getNegExamples, printExamples

from actual_chat_bot import swda_chat

weights = trainExamplesPosList = testExamplesPosList = None
 
def guessEval(examples):
    global weights
    correct = 0
    for i in range(len(examples)):
        prompt = examples[i][0][0]
        maxScore = 0
        maxResponse = examples[0][0][1]
        for j in range(len(examples)):
            response = examples[j][0][1]
            guess = (prompt, response)
            phi = swda_feature_extractor(guess)
            score = dotProduct(weights, phi)
            if score > maxScore:
                maxScore = score
                maxResponse = response
        if maxResponse == examples[i][0][1]:
            correct = correct + 1
    return 1.0 * correct / len(examples)
        
def humanScore():
    humanTestList = []
    humanTestList.extend(trainExamplesPosList[5])
    humanTestList.extend(trainExamplesNegList[5])
    random.shuffle(humanTestList)
    print "NUM EXAMPLES"
    print len(humanTestList)
    yourCorrect = 0
    soFar = 0
    for example in humanTestList:
        soFar = soFar + 1
        print soFar
        print "A"
        for utt in example[0][0]:
            print utt.text_words();
        print "B"
        for utt in example[0][1]:
            print utt.text_words();
        user_input = raw_input("Score: ")
        if int(user_input) == example[1]:
            print "Correct!"
            yourCorrect = yourCorrect + 1
        else:
            print "Incorrect."
        print "Your score:"
        print 1.0*yourCorrect/soFar

def chooseEval(examples):
    correct = 0
    for i in range(len(examples)):
        prompt = examples[i][0][0]
        response1 = examples[i][0][1]
        randomInt = random.randint(0, len(examples)-1)
        response2 = examples[randomInt][0][1]
        while response1 == response2:
            randomInt = random.randint(0, len(examples)-1)
            response2 = examples[randomInt][0][1]
        guess1 = (prompt, response1)
        phi1 = swda_feature_extractor(guess1)
        score1 = dotProduct(weights, phi1)
        guess2 = (prompt, response2)
        phi2 = swda_feature_extractor(guess2)
        score2 = dotProduct(weights, phi2)
        #The following lines had guess instead of score previously!
        if(score1 > score2):
            correct = correct + 1
        if(score1 == score2):
            correct = correct + .5 
    return 1.0 * correct / len(examples)


def humanChoice():
    global trainExamplesPosList
    humanTestList = []
    humanTestList.extend(trainExamplesPosList[10])
    random.shuffle(humanTestList)
    print "NUM EXAMPLES"
    print len(humanTestList)
    yourCorrect = 0
    soFar = 0
    for example in humanTestList:
        soFar = soFar + 1
        prompt = example[0][0]
        response1 = example[0][1]
        randomInt = random.randint(0, len(humanTestList)-1)
        response2 = humanTestList[randomInt][0][1]
        print soFar
        print "Prompt"
        for utt in prompt:
            print utt.text_words();
        randomInt =random.randint(1,2)
        if randomInt == 1:
            print "Response 1"
            for utt in response1:
                print utt.text_words();
            print "Response 2"
            for utt in response2:
                print utt.text_words();
        else:
            print "Response 1"
            for utt in response2:
                print utt.text_words();
            print "Response 2"
            for utt in response1:
                print utt.text_words();
        user_input = raw_input("Response: ")
        if int(user_input) == randomInt:
            print "Correct!"
            yourCorrect = yourCorrect + 1
        else:
            print "Incorrect."
        print "Your score:"
        print 1.0*yourCorrect/soFar

    
def runBot():

    global weights, trainExamplesPosList, testExamplesPosList
    
    #number of transcripts
    TRAIN_SET_SIZE = 1000
    TEST_SET_SIZE = 10
    random.seed(5)
    
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
            trainExamplesPos = getPosExamples(turns)   
            trainExamplesNeg = getNegExamples(turns)
            trainExamplesPosList.append(trainExamplesPos)
            trainExamplesNegList.append(trainExamplesNeg)
            trainExamples.extend(trainExamplesPos)
            trainExamples.extend(trainExamplesNeg)
            count = count + 1
        elif count < TEST_SET_SIZE + TRAIN_SET_SIZE:
            testExamplesPos = getPosExamples(turns)   
            testExamplesNeg = getNegExamples(turns)
            testExamplesPosList.append(testExamplesPos)
            testExamplesNegList.append(testExamplesNeg)
            testExamples.extend(trainExamplesPos)
            testExamples.extend(trainExamplesNeg)
            count = count + 1
        else:
            break
    print "Finding Tag Counts..."
    printTagCount(turnSet)

    print "Finding Tag Set Counts..."
    printTagSetCount(turnSet)

    print "Finding Bad Turns..."
    printNumBadTurns(turnSet)

    print "Lenght Statistics..."
    printAvgStats()

    print "Interruption Statistics..."
    printInterruptionStats(turnSet)
    
    print "Training Predictor..."
    weights = learnPredictor(trainExamples, testExamples, swda_feature_extractor)

    #swda_chat returns false if the chatting is totally over
    while swda_chat(weights, swda_feature_extractor, turnSet):
        print "Finished a conversation"

    print "Finding Interesting Examples..."
    printExamples(testExamples, weights, swda_feature_extractor)

    print "Finding highest and lowest weights..."
    printWeightStatistics(weights, 5)

    print "Evaluating Bot..."
    print "Train Choosing..."
    summ = 0
    for trainExamplesPos in trainExamplesPosList:
        summ = summ + chooseEval(trainExamplesPos)
    print "Train Choosing Score"
    print 1.0 * summ/len(trainExamplesPosList)
    print "Test Choosing..."
    summ = 0
    for testExamplesPos in testExamplesPosList:
        summ = summ + chooseEval(testExamplesPos)
    print "Test Choosing Score"
    print 1.0 * summ/len(testExamplesPosList)
    print "Train Guessing..."
    summ = 0
    for trainExamplesPos in trainExamplesPosList:
        summ = summ + guessEval(trainExamplesPos)
    print "Train Guessing Score"
    print 1.0 * summ/len(trainExamplesPosList)
    print "Test Guessing..."
    summ = 0
    for testExamplesPos in testExamplesPosList:
        summ = summ + guessEval(testExamplesPos)
    print "Test Guessing Score"
    print 1.0 * summ/len(testExamplesPosList)
    
##    humanChoice()

if __name__ == "__main__":
    runBot()
