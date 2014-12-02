import random
from util import dotProduct


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

#Returns true if turn only contains utterances that are backchannels, turn-exits, or noise
def isBadTurn(turn):
    for utt in turn:
        if utt.act_tag not in ["b", "%", "x"]:
            return False
    return True

#Returns true if example should be filtered
def filterNeg(turnA, turnB):
    if turnA[0].caller == turnB[0].caller: return True
    if isBadTurn(turnB): return True
    return False

"""
def getNegExamples(turns):
    negExamples = []
    badTurnCount = 0
    for i in range(1, len(turns)):
        randomInt = random.randint(0, len(turns) - 1)
        while turns[i-1][0].caller == turns[randomInt][0].caller:
            randomInt = random.randint(0, len(turns) - 1)
        negExamples.append(((turns[i-1], turns[randomInt]), -1))
    return negExamples
"""

#Extension of the original getNegExamples gets rid of badTurns
#It still adds the same number of negative examples by adding extras at the end
def getNegExamples(turns):
    negExamples = []
    badTurnCount = 0
    for i in range(1, len(turns)):
        if isBadTurn(turns[i-1]):
            badTurnCount = badTurnCount + 1
            continue
        randomInt = random.randint(0, len(turns) - 1)
        while filterNeg(turns[i-1], turns[randomInt]) or randomInt == i:
            randomInt = random.randint(0, len(turns) - 1)
        negExamples.append(((turns[i-1], turns[randomInt]), -1))
    while badTurnCount > 0:
        randomInt1 = random.randint(0, len(turns) - 1)
        if isBadTurn(turns[randomInt1]):
            continue
        randomInt2 = random.randint(0, len(turns) - 1)
        while filterNeg(turns[randomInt1], turns[randomInt2]) or randomInt1 + 1 == randomInt2:
            randomInt2 = random.randint(0, len(turns) - 1)
        negExamples.append(((turns[randomInt1], turns[randomInt2]), -1))
        badTurnCount = badTurnCount - 1
    return negExamples

def printExamples(examples, weights, featureExtractor):
    tpFound = fpFound = tnFound = fnFound = False
    for example in examples:
        phi = featureExtractor(example[0])
        score = dotProduct(weights, phi)
        if not tpFound and score > .5 and example[1] == 1:
            print "FOUND: True Positive"
            print "Prompt"
            for utt in example[0][0]:
                print utt.text_words();
            print example[0][0][len(example[0][0])-1].act_tag
            print "Response"
            for utt in example[0][1]:
                print utt.text_words();
            print example[0][1][0].act_tag
            tpFound = True
        if not fpFound and score > -.5 and example[1] == -1:
            print "FOUND: False Positive"
            print "Prompt"
            for utt in example[0][0]:
                print utt.text_words();
            print example[0][0][len(example[0][0])-1].act_tag
            print "Response"
            for utt in example[0][1]:
                print utt.text_words();
            print example[0][1][0].act_tag
            fpFound = True
        if not tnFound and score < -.5 and example[1] == -1:
            print "FOUND: True Negative"
            print "Prompt"
            for utt in example[0][0]:
                print utt.text_words();
            print example[0][0][len(example[0][0])-1].act_tag
            print "Response"
            for utt in example[0][1]:
                print utt.text_words();
            print example[0][1][0].act_tag
            tnFound = True
        if not fnFound and score < -.5 and example[1] == 1:
            print "FOUND: False Negative"
            print "Prompt"
            for utt in example[0][0]:
                print utt.text_words();
            print example[0][0][len(example[0][0])-1].act_tag
            print "Response"
            for utt in example[0][1]:
                print utt.text_words();
            print example[0][1][0].act_tag
            fnFound = True
        if tpFound and fpFound and tnFound and fnFound:
            break
