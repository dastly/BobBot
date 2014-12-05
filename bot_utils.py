import random
from util import dotProduct
from collections import Counter


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
            print "Response"
            for utt in example[0][1]:
                print utt.text_words();
            for key in phi:
                print "{0}: {1}".format(key, weights[key])
            tpFound = True
        if not fpFound and score > -.5 and example[1] == -1:
            print "FOUND: False Positive"
            print "Prompt"
            for utt in example[0][0]:
                print utt.text_words();
            print "Response"
            for utt in example[0][1]:
                print utt.text_words();
            for key in phi:
                print "{0}: {1}".format(key, weights[key])
            fpFound = True
        if not tnFound and score < -.5 and example[1] == -1:
            print "FOUND: True Negative"
            print "Prompt"
            for utt in example[0][0]:
                print utt.text_words();
            print "Response"
            for utt in example[0][1]:
                print utt.text_words();
            for key in phi:
                print "{0}: {1}".format(key, weights[key])
            tnFound = True
        if not fnFound and score < -.5 and example[1] == 1:
            print "FOUND: False Negative"
            print "Prompt"
            for utt in example[0][0]:
                print utt.text_words();
            print "Response"
            for utt in example[0][1]:
                print utt.text_words();
            for key in phi:
                print "{0}: {1}".format(key, weights[key])
            fnFound = True
        if tpFound and fpFound and tnFound and fnFound:
            break

def printTurns(turns, print_act_tags = False, num_turns = 0):
    if num_turns == 0:
        num_turns = len(turns)
    for i in range(num_turns):
        if i%2 == 0:
            speaker = "YOU"
        else:
            speaker = "ME"
        for utt in turns[i]:
            act_tag = ""
            if print_act_tags:
                act_tag = " (" + utt.act_tag + ")"
            print "{0}{1}: {2}".format(speaker, act_tag, utt.text)
                        
def printWeightStatistics(weightsIn, NUM_FEATURES = 5):
    weights = {}
    weightsInv = {}
    for k,v in weightsIn.items():
        if  "'b'" in k or "'%'" in k or "'x'" in k: continue
        weights[k] = v
        if "utt_length" in k: continue
        weightsInv[k] = -v
    c = Counter(weights)
    print c.most_common(NUM_FEATURES)
    c = Counter(weightsInv)
    print c.most_common(NUM_FEATURES)

def printTagCount(turnSet):
    c = Counter()
    for discourse in turnSet:
        for turn in discourse:
            for utt in turn:
                c[utt.act_tag] += 1
    print c
    
def printNumBadTurns(turnSet):
    badTurnCount = 0
    turnCount = 0
    averageBadTurnDensity = 0
    for discourse in turnSet:
        discourseTurnCount = 0
        discourseBadTurnCount = 0
        for turn in discourse:
            turnCount += 1
            discourseTurnCount += 1
            if isBadTurn(turn):
                badTurnCount += 1
                discourseBadTurnCount += 1
        averageBadTurnDensity += 1.0 * discourseBadTurnCount / discourseTurnCount
    print "NUM BAD TURNS: {0}".format(badTurnCount)
    print "NUM TURNS: {0}".format(turnCount)
    print "AVG BAD TURN DENSITY: {0}".format(1.0 * badTurnCount / turnCount)
    print "AVG BAD TURN DENSITY: {0}".format(1.0 * averageBadTurnDensity / len(turnSet))

def chooseFromDistribution(distribution):
  "Takes either a counter or a list of (prob, key) pairs and samples"
  summ = 0
  for element, score in distribution:
    summ += score
  for element, score, index in enumerate(distribution):
    distribution[index] = (element, 1.0 * score / summ)

  r = random.random()
  base = 0.0
  for element, prob in distribution:
    base += prob
    if r <= base: return element