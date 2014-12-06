import random, json, os
from util import dotProduct
from collections import Counter
from sets import Set

from config import * # restrict_bad_utterances

num_utterances = total_utt_length = 0
num_turns = total_turn_length = 0
num_turn_pairs = total_turn_pair_diff = 0

"""
Processing Utterances and Generating Examples
"""

# Takes a transcript, and converts the set of utterances into a set of turns.
# A turn is a set of consecutively spoken utterances by one speaker.
# This represents one person's turn in a conversation.
def processUtterances(transcript):
    global num_utterances, total_utt_length, num_turns, total_turn_length
    turns = []
    turn = []
    utterances = transcript.utterances
    num_utterances += len(utterances)
    for utterance in utterances:
        if restrict_bad_utterances and utterance.act_tag in PRE_PROCESS_NOISE: continue
        total_utt_length += len(utterance.text_words())
        if len(turn) == 0:
            turn.append(utterance)
        else:
            if turn[0].caller == utterance.caller:
                turn.append(utterance)
            else:
                turns.append(turn)
                turn = []
                turn.append(utterance)
    num_turns += 1
    total_turn_length += len(turn)
    turns.append(turn)
    return turns

# Positive example = (prompt, response)
# where prompt and response are two consecutive turns
def getPosExamples(turns):
    global num_turn_pairs, total_turn_pair_diff
    posExamples = []
    for i in range(1, len(turns)):
        total_turn_pair_diff += abs(len(turns[i-1]) - len(turns[i]))
        posExamples.append(((turns[i-1], turns[i]), 1))
    num_turn_pairs += len(posExamples)
    return posExamples

# Returns true if turn only contains utterances that are backchannels, turn-exits, or noise
def isBadTurn(turn):
    for utt in turn:
        if utt.act_tag not in BAD_TURN_NOISE:
            return False
    return True

#Returns true if example should be filtered
def filterNeg(turnA, turnB):
    if turnA[0].caller == turnB[0].caller: return True
    if isBadTurn(turnB): return True
    return False

# Negative example = (turn, randomTurn)
# Random pairs give an estimation of what constitutes unlikely conversation

def getNegExamples(turns):
    if restrict_bad_turns:
        return getNegExamples_restricted(turns)
    else:
        return getNegExamples_unrestricted(turns)

def getNegExamples_unrestricted(turns):
    negExamples = []
    badTurnCount = 0
    for i in range(1, len(turns)):
        randomInt = random.randint(0, len(turns) - 1)
        # Relies on random to break loop
        while turns[i-1][0].caller == turns[randomInt][0].caller:
            randomInt = random.randint(0, len(turns) - 1)
        negExamples.append(((turns[i-1], turns[randomInt]), -1))
    return negExamples

# Extension of the original getNegExamples gets rid of badTurns
# It still adds the same number of negative examples by adding extras at the end
def getNegExamples_restricted(turns):
    global num_turn_pairs, total_turn_pair_diff
    negExamples = []
    badTurnCount = 0
    for i in range(1, len(turns)):
        if isBadTurn(turns[i-1]):
            badTurnCount = badTurnCount + 1
            continue
        randomInt = random.randint(0, len(turns) - 1)
        # Relies on random to break loop
        while filterNeg(turns[i-1], turns[randomInt]) or randomInt == i:
            randomInt = random.randint(0, len(turns) - 1)
        negExamples.append(((turns[i-1], turns[randomInt]), -1))
        total_turn_pair_diff += abs(len(turns[i-1]) - len(turns[randomInt]))
    # Relies on random to break loop
    while badTurnCount > 0:
        randomInt1 = random.randint(0, len(turns) - 1)
        if isBadTurn(turns[randomInt1]):
            continue
        randomInt2 = random.randint(0, len(turns) - 1)
        # Relies on random to break loop
        while filterNeg(turns[randomInt1], turns[randomInt2]) or randomInt1 + 1 == randomInt2:
            randomInt2 = random.randint(0, len(turns) - 1)
        negExamples.append(((turns[randomInt1], turns[randomInt2]), -1))
        total_turn_pair_diff += abs(len(turns[randomInt2]) - len(turns[randomInt1]))
        badTurnCount = badTurnCount - 1
    num_turn_pairs += len(negExamples)
    return negExamples

"""
Print Examples and Statistics
"""

# Will find and print a TRUE POSITIVE, TRUE NEGATIVE, FALSE POSITIVE, and FALSE NEGATIVE
def printExamples(examples, weights, featureExtractor):
    print "Finding Interesting Examples..."
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
                if key in weights:
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
                if key in weights:
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
                if key in weights:
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

# Prints a set of turns in conversation format
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

# Prints set of utterances
def utt_list_to_string(utt_list):
    return map(lambda utt: reduce(lambda x, y : x + y + ' ', utt.text_words(), "").strip(), utt_list)

# Prints set of candidate responses and the associated scores
def print_candidates_and_scores(candidates_and_scores):
    for cand, score in candidates_and_scores:
        print 'Candidate = {0}, Score = {1}'.format(utt_list_to_string(cand), score)

def printLengthStats():
    print "Length Statistics..."
    print 'Average Utterance Length = ', float(total_utt_length)/float(num_utterances)
    print 'Average Turn Length = ', float(total_turn_length)/float(num_turns)
    print 'Average Difference in Turn Length in a pair = ', float(total_turn_pair_diff)/float(num_turn_pairs)

# Prints most common and least common features (excluding some)                       
def printWeightStatistics(weightsIn, NUM_FEATURES = 5, ignore_noise = True, ignore_length = True):
    print "Finding highest and lowest weights..."
    weights = {}
    weightsInv = {}
    for k,v in weightsIn.items():
        if ignore_noise and "'b'" in k or "'%'" in k or "'x'" in k: continue
        weights[k] = v
        if ignore_length and "length" in k: continue
        weightsInv[k] = -v
    c = Counter(weights)
    print c.most_common(NUM_FEATURES)
    c = Counter(weightsInv)
    print c.most_common(NUM_FEATURES)

# Prints count of all act_tags
def printTagCount(turnSet):
    print "Finding Tag Counts..."
    c = Counter()
    numUtts = 0
    for discourse in turnSet:
        for turn in discourse:
            for utt in turn:
                c[utt.act_tag] += 1
                numUtts += 1
    print c
    for item in c:
        c[item] = round(100.0*c[item]/numUtts, 2)
    print c

# Prints count of different sets of act_tags all from one turn
def printTagSetCount(turnSet, NUM_MOST_COMMON = 10):
    print "Finding Tag Set Counts..."
    c = Counter()
    numTurns = 0
    for discourse in turnSet:
        for turn in discourse:
            tagSet = ""
            for utt in turn:
                tagSet += utt.act_tag + " ; "
            c[tagSet] += 1
            numTurns += 1
    print c.most_common(NUM_MOST_COMMON)
    for item in c:
        c[item] = round(100.0*c[item]/numTurns, 2)
    print c.most_common(NUM_MOST_COMMON)
    print c["sd ; sd ; sd ; "]
    
def printNumBadTurns(turnSet):
    print "Finding Bad Turns..."
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

def printInterruptionStats(turnSet, NUM_MOST_COMMON = 10):
    print "Interruption Statistics..."
    backChannelPrevCounter = Counter()
    collabPrevCounter = Counter()
    numInterrupts = 0
    for discourse in turnSet:
        for i in range(len(discourse)):
            if i > 0 and discourse[i][0].act_tag == 'b':
                prevTurn = discourse[i - 1]
                if prevTurn[len(prevTurn) - 1].text.find("--") > -1:
                    prevTagList = ""
                    for utt in prevTurn:
                        prevTagList += utt.act_tag + " ; "
                    backChannelPrevCounter[prevTagList] += 1
            if i > 0 and discourse[i][0].act_tag == '^2':
                prevTurn = discourse[i - 1]
                prevTagList = ""
                for utt in prevTurn:
                    prevTagList += utt.act_tag + " ; "
                collabPrevCounter[prevTagList] += 1
    print "Tags Before Backchannel"
    print backChannelPrevCounter.most_common(NUM_MOST_COMMON)
    print "Tags Before Collaborative Completion"
    print collabPrevCounter.most_common(NUM_MOST_COMMON)

"""
File reading for weights vector
"""

def read_weights_from_file(filename):
    try:
        with open(filename, 'r') as weights_file:
            weights = json.load(weights_file)
            return weights
    except Exception:
        print 'ERROR: Weights file does not exist or is malformed'
        raise

def save_weights_to_file(filename, weights):
    try:
        os.remove(filename)
    except:
        pass
    with open(filename, 'w') as weights_file:
        weights_file.write(json.dumps(weights))
