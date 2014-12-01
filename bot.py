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
import sys, getopt
from SGD import learnPredictor
from util import dotProduct
from features import swda_feature_extractor
import parse_xml as parse

weights = trainExamplesPosList = testExamplesPosList = None
#number of transcripts
TRAIN_SET_SIZE = 20
TEST_SET_SIZE = 20
random.seed(5)

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

def getNegExamplesSwitchboard(turns):
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
        phi1 = swda_feature_extractor(guess1)
        score1 = dotProduct(weights, phi1)

        guess2 = (prompt, response2)
        phi2 = swda_feature_extractor(guess2)
        score2 = dotProduct(weights, phi2)
        if(guess1 > guess2):
            correct = correct + 1
        if(guess2 == guess1):
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

def get_transcripts(use_subtitles):
    if use_subtitles:
        for movie in parse.tag_all_movies():
            yield movie
    else:
        for transcript in CorpusReader('swda').iter_transcripts(display_progress=True):
            yield transcript

            
def runBot(use_subtitles):

    global weights, trainExamplesPosList, testExamplesPosList
        
    trainExamples = []
    trainExamplesPosList = []
    trainExamplesNegList = []

    testExamples = []
    testExamplesPosList = []
    testExamplesNegList = []

    count = 0

    if use_subtitles:
        getNegExamplesFn = getNegExamplesSubtitles
        featureExtractor = subtitles_feature_extractor
        processFn = lambda x : x
    else:
        getNegExamplesFn = getNegExamplesSwitchboard
        featureExtractor = swda_feature_extractor
        processFn = processUtterances

    for transcript in get_transcripts(use_subtitles):
        turns = processFn(transcript)
        if count < TRAIN_SET_SIZE:
            trainExamplesPos = getPosExamples(turns)   
            trainExamplesNeg = getNegExamplesFn(turns)
            trainExamplesPosList.append(trainExamplesPos)
            trainExamplesNegList.append(trainExamplesNeg)
            trainExamples.extend(trainExamplesPos)
            trainExamples.extend(trainExamplesNeg)
            count = count + 1
        elif count < TEST_SET_SIZE + TRAIN_SET_SIZE:
            testExamplesPos = getPosExamples(turns)   
            testExamplesNeg = getNegExamplesFn(turns)
            testExamplesPosList.append(testExamplesPos)
            testExamplesNegList.append(testExamplesNeg)
            testExamples.extend(trainExamplesPos)
            testExamples.extend(trainExamplesNeg)
            count = count + 1
        else:
            break

    weights = learnPredictor(trainExamples, testExamples, featureExtractor)

    for example in testExamples:
        print example[1]
        phi = swda_feature_extractor(example[0])
        print phi
        for key in phi:
            print key
            print weights[key]
        score = dotProduct(weights, phi)
        if score < -.5 and example[1] == -1:
            # print "FOUND"
            # print "Prompt"
            # for utt in example[0][0]:
            #     print utt.text_words();
            # print example[0][0][len(example[0][0])-1].act_tag
            # print "Response"
            # for utt in example[0][1]:
            #     print utt.text_words();
            # print example[0][1][0].act_tag
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

    
def usage():
    print 'Usage: python bot.py [-p] [--phone] [-s] [--subtitles]'
    print '-p OR --phone = to train with the Switchboard phone conversation corpus'
    print '-s OR --subtitles = to train with the open subtitles corpus'

if __name__ == "__main__":
    if len(sys.argv) != 2:
        usage()
        sys.exit(2)
    try:
        argv = sys.argv[1:]
        opts, args = getopt.getopt(argv, "ps", ["subtitles", "phone"])
        use_subtitles = None
        for opt, arg in opts:
            if opt in ('-p', '--phone'):
                use_subtitles = False
            if opt in ('-s', '--subtitles'):
                use_subtitles = True
        runBot(use_subtitles)
    except getopt.GetoptError:
        usage()
        sys.exit(2)
