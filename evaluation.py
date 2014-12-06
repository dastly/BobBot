import random, collections, math, sys, getopt, pdb
from util import dotProduct
from features import swda_feature_extractor
from bot_utils import *     #processUtterances, getPosExamples, isBadTrun, getNegExamples, printExamples

from config import *

### COMPUTER EVALUATION ###

# Averages evaluation run over a set of discourses.
# discourses is a list of discourses, where each discourse is a list of examples
def evaluate(discourses, evalFn, weights):
    summ = 0
    for examples in discourses:
        summ = summ + evalFn(examples, weights)
    return 1.0 * summ/len(discourses)

# Evaluation Function 1: chooseEval
# Uses the weights and feature extractor to choose the best scoring response to a prompt.
# Here it guesses between the actual response and a random response from the discourse.
def chooseEval(examples, weights):
    correct = 0
    for i in range(len(examples)):
        prompt = examples[i][0][0]
        response1 = examples[i][0][1]
        randomInt = random.randint(0, len(examples)-1)
        response2 = examples[randomInt][0][1]
        # Relies on random to break loop
        while response1 == response2 or (neg_restrict_bad and isBadTrun(response2)) or response1[0].caller == response2[0].caller:
            randomInt = random.randint(0, len(examples)-1)
            response2 = examples[randomInt][0][1]
        guess1 = (prompt, response1)
        phi1 = swda_feature_extractor(guess1)
        score1 = dotProduct(weights, phi1)
        guess2 = (prompt, response2)
        phi2 = swda_feature_extractor(guess2)
        score2 = dotProduct(weights, phi2)
        if(score1 > score2):
            correct = correct + 1
        if(score1 == score2):
            correct = correct + .5 
    return 1.0 * correct / len(examples)

# Evaluation Function 1: guessEval
# Uses the weights and feature extractor to choose the best scoring response to a prompt.
# Here it guesses among all possibly responses in the discourse.
def guessEval(examples, weights):
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

### HUMAN ORACLE TESTING ###

# More so for CS 221, these functions allow a human to attempt similar evaluations
# This is slightly unfair to the human, since they only get to see part of the discourse

# Given a set of (prompt, response examples) guess if it is real or random
def humanScore(trainExamplesPosList, trainExamplesNegList):
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

# Choose between the actual response and a random response
def humanChoice(trainExamplesPosList):
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