##from swda import Transcript
##trans = Transcript('swda/sw00utt/sw_0001_4325.utt.csv', 'swda/swda-metadata.csv')
##print trans

#!/usr/bin/python
# Filename: bot.py

import swda
from swda import Transcript
from swda import CorpusReader
import SGD
import random
import collections
import math
import sys
from SGD import learnPredictor
from util import dotProduct

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

def sampleFeatureExtractor(x):
    phi = dict()
    turnA = x[0]
    turnB = x[1]
    tagA = turnA[len(turnA)-1].act_tag
    tagB = turnB[0].act_tag
    phi[tagA + ", " + tagB] = 1
    if len(turnA) + len(turnB) < 4:
        phi["Short"] = 1
    return phi


def processUtterances(transcript):
    turns = []
    turn = []
    utterances = transcript.utterances
    for utterance in utterances:
        if len(turn) == 0:
            turn.append(utterance)
        else:
            if turn[0].caller == utterance.caller:
                turn.append(utterance)
            else:
                turns.append(turn)
                turn = []
                turn.append(utterance)
    turns.append(turn)
    return turns
        
def getPosExamples(turns):
    posExamples = []
    for i in range(1, len(turns)):
        posExamples.append(((turns[i-1], turns[i]), 1))
    return posExamples

def getNegExamples(turns):
    negExamples = []
    for i in range(1, len(turns)):
        randomInt = random.randint(0, len(turns) - 1)
        while turns[i-1][0].caller == turns[randomInt][0].caller:
            randomInt = random.randint(0, len(turns) - 1)
        negExamples.append(((turns[i-1], turns[randomInt]), -1))
    return negExamples

def humanScore():
    humanTestList = []
    humanTestList.extend(trainExamplesPosList[5])
    humanTestList.extend(trainExamplesNegList[4])
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


def humanChoice():
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

    count = 0
    for transcript in CorpusReader('swda').iter_transcripts(display_progress=False):
        turns = processUtterances(transcript)
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
    
    weights = learnPredictor(trainExamples, testExamples, sampleFeatureExtractor)

    for example in testExamples:
        print example[1]
        phi = sampleFeatureExtractor(example[0])
        print phi
        for key in phi:
            print key
            print weights[key]
        score = dotProduct(weights, phi)
        if score < -.5 and example[1] == -1:
            print "FOUND"
            print "Prompt"
            for utt in example[0][0]:
                print utt.text_words();
            print example[0][0][len(example[0][0])-1].act_tag
            print "Response"
            for utt in example[0][1]:
                print utt.text_words();
            print example[0][1][0].act_tag
            break

    summ = 0
    for trainExamplesPos in trainExamplesPosList:
        summ = summ + guessEval(trainExamplesPos)
    print "Train Guessing"
    print 1.0 * summ/len(trainExamplesPosList)
    summ = 0
    for testExamplesPos in testExamplesPosList:
        summ = summ + guessEval(testExamplesPos)
    print "Test Guessing"
    print 1.0 * summ/len(testExamplesPosList)
    summ = 0
    for trainExamplesPos in trainExamplesPosList:
        summ = summ + chooseEval(trainExamplesPos)
    print "Train Choosing"
    print 1.0 * summ/len(trainExamplesPosList)
    summ = 0
    for testExamplesPos in testExamplesPosList:
        summ = summ + chooseEval(testExamplesPos)
    print "Test Choosing"
    print 1.0 * summ/len(testExamplesPosList)
        
    humanChoice()

if __name__ == "__main__":
    runBot()
