#!/usr/bin/python
# Filename: bot.py

import SGD
import random
import collections
import math
import sys
import nltk
from nltk.corpus import switchboard
from nltk.corpus import nps_chat
from SGD import learnPredictor
from util import dotProduct

##print(switchboard.discourses()[0])

def sampleFeatureExtractor(x):
    phi = dict()
    turnA = x[0]
    turnB = x[1]
    return phi

def runBot():
    
    def getPosExamples(turns):
        posExamples = []
        for i in range(1, len(turns)):
            posExamples.append(((turns[i-1], turns[i]), 1))
        return posExamples

    def getNegExamples(turns):
        negExamples = []
        for i in range(1, len(turns)):
            randomInt = random.randint(0, len(turns) - 1)
            while turns[i-1].speaker == turns[randomInt].speaker:
                randomInt = random.randint(0, len(turns) - 1)
            negExamples.append(((turns[i-1], turns[randomInt]), -1))
        return negExamples

    TRAIN_SET_SIZE = 10
    TEST_SET_SIZE = 10
    random.seed(5)
    
    trainExamples = []
    trainExamplesPosList = []
    trainExamplesNegList = []

    testExamples = []
    testExamplesPosList = []
    testExamplesNegList = []

    count = 0
    for turns in switchboard.discourses():
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
    """
    weights = learnPredictor(trainExamples, testExamples, sampleFeatureExtractor)

    def guessEval(examples):
        correct = 0
        for i in range(len(examples)):
            prompt = examples[i][0][0]
            maxScore = 0
            maxResponse = examples[0][0][1]
            for j in range(len(examples)):
                response = examples[j][0][1]
                guess = (prompt, response)
                phi = sampleFeatureExtractor(guess)
                score = dotProduct(weights, phi)
                if score > maxScore:
                    maxScore = score
                    maxResponse = response
            if maxResponse == examples[i][0][1]:
                correct = correct + 1
        return 1.0 * correct / len(examples)

    def chooseEval(examples):
        correct = 0
        for i in range(len(examples)):
            prompt = examples[i][0][0]
            response1 = examples[i][0][1]
            randomInt = random.randint(0, len(examples)-1)
            response2 = examples[randomInt][0][1]
            guess1 = (prompt, response1)
            phi1 = sampleFeatureExtractor(guess1)
            score1 = dotProduct(weights, phi1)
            
            guess2 = (prompt, response2)
            phi2 = sampleFeatureExtractor(guess2)
            score2 = dotProduct(weights, phi2)
            if(guess1 > guess2):
                correct = correct + 1
            if(guess2 == guess1):
                correct = correct + .5
        return 1.0 * correct / len(examples)
    summ = 0
    for trainExamplesPos in trainExamplesPosList:
        summ = summ + guessEval(trainExamplesPos)
    print("Train Guessing")
    print(1.0 * summ/len(trainExamplesPosList))
    summ = 0
    for testExamplesPos in testExamplesPosList:
        summ = summ + guessEval(testExamplesPos)
    print("Test Guessing")
    print(1.0 * summ/len(testExamplesPosList))
    summ = 0
    for trainExamplesPos in trainExamplesPosList:
        summ = summ + chooseEval(trainExamplesPos)
    print("Train Choosing")
    print(1.0 * summ/len(trainExamplesPosList))
    summ = 0
    for testExamplesPos in testExamplesPosList:
        summ = summ + chooseEval(testExamplesPos)
    print("Test Choosing")
    print(1.0 * summ/len(testExamplesPosList))
    """
    def humanChoice():
        humanTestList = []
        humanTestList.extend(trainExamplesPosList[5])
        random.shuffle(humanTestList)
        print("NUM EXAMPLES")
        print(len(humanTestList))
        yourCorrect = 0
        soFar = 0
        for example in humanTestList:
            soFar = soFar + 1
            prompt = example[0][0]
            responseT1 = example[0][1]
            response1 = repr(responseT1)
            response1 = response1[response1.find(":"):]
            randomInt = random.randint(0, len(humanTestList)-1)
            responseT2 = humanTestList[randomInt][0][1]
            response2 = repr(responseT2)
            response2 = response2[response2.find(":"):]
            print(soFar)
            print("Prompt")
            print(prompt)
            randomInt =random.randint(1,2)
            if randomInt == 1:
                print("Response 1")
                print(response1)
                print("Response 2")
                print(response2)
            else:
                print("Response 1")
                print(response2)
                print("Response 2")
                print(response1)
            user_input = input("Response: ")
            if int(user_input) == randomInt:
                print("Correct!")
                yourCorrect = yourCorrect + 1
            else:
                print("Incorrect.")
            print("Your score:")
            print(1.0*yourCorrect/soFar)
    humanChoice()
        
runBot()
