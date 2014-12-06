import random, pdb

from nltk.grammar import PCFG, induce_pcfg, toy_pcfg1, toy_pcfg2
from nltk.parse.generate import generate
import swda
from swda import Transcript
from swda import CorpusReader
from swda import Utterance

from util import * # dotProduct, chooseFromDistribution
from bot_utils import * # printTurns, print_candidates_and_scores
from features import swda_feature_extractor

# Global variables
turns = transcript_metadata = None
get_act_tag = use_specific_discourse = use_distribution = False
bad_turn_counter = 0

#Global Constants
INTERRUPT_AFTER_LEN = 2
INTERRUPT_PROB = .25
INTERRUPT_THRESHOLD = 5
BAD_TURN_MAX = 1
NUM_CANDIDATES = 100
NUM_START = 6


def getYesOrNo(prompt):
        print prompt
        yesOrNo = raw_input("Y/N: ")
        if yesOrNo in ["Y", "y", "YES", "Yes", "yes"]: return True
        return False

def setupBot(turnSet):
##        get_demo_info = getYesOrNo("Do we want to give demographic info about ourselves?")
    global turns, transcript_metadata, get_act_tag, use_specific_discourse, use_distribution

    transcript_metadata = getTranscript_Metadata(False)
    get_act_tag = getYesOrNo("Should I ask for act_tags?")
    use_specific_discourse = getYesOrNo("Do you want to use a specific discourse?")
    use_distribution = getYesOrNo("Do you want to use a distribution selection?")
    turns = []
    if use_specific_discourse:
        turns = getSpecificTurns(turnSet)
    else:
        for discourse in turnSet:
            turns.extend(discourse)

def getTranscript_Metadata(get_demo_info):
        if get_demo_info:
                print "First, tell me about yourself."
                from_caller_sex = raw_input("Sex (MALE, FEMALE, etc.): ")
                from_caller_birth_year = raw_input("Birth year: ")
                from_caller_dialect_area = raw_input("Dialect area: ")
                from_caller_education = 1

                print "I can pretend to be different people."
                print "Tell me about myself: "
                to_caller_sex = raw_input("Sex (MALE, FEMALE, etc.): ")
                to_caller_birth_year = raw_input("Birth year: ")
                to_caller_dialect_area = raw_input("Dialect area: ")
                to_caller_education = 1
        else:
                to_caller_birth_year = '1992'
                to_caller_education = 1
                to_caller_sex = 'MALE'
                to_caller_dialect_area = 'TIDEWATER'
                from_caller_birth_year = '1992'
                from_caller_education = 1
                from_caller_sex = 'MALE'
                from_caller_dialect_area = 'TIDEWATER'
        transcript_metadata = {'to_caller_birth_year':to_caller_birth_year, 'to_caller_education':to_caller_education, 'to_caller_sex':to_caller_sex, 'to_caller_dialect_area':to_caller_dialect_area, 'from_caller_birth_year':from_caller_birth_year, 'from_caller_education':from_caller_education, 'from_caller_sex':from_caller_sex, 'from_caller_dialect_area':from_caller_dialect_area}
        return transcript_metadata

def getSpecificTurns(turnSet):
        print "A few of the possible topic choices:"
        print "(0,11) Child care, Household duties"
        print "(1) Drug Testing"
        print "(2,6) Tracking Finances, Pensions"
        print "(4,5,13,14) Elderly care"
        print "(3,7,12) Judicial system"
        print "(8,10) Recycling"
        print "(9) Cars"
        print "(15) Sports"
        print "(20) Music"
        
        discourse_num = raw_input("Enter a discourse number [0-{0}]: ".format(len(turnSet)-1))
        return turnSet[int(discourse_num)]

def getUserUtterance(get_act_tag, transcript_metadata):

    # Constants for construction of utterance
    # Act_tags are currently added manually during the conversation.
    # With the dialogue act tagging algorithm, this would be done by the bot.
    # A parsing/tagging algorithm would also be used for pos and trees.
    swda_filename = 'chatFile'
    ptb_basename = 'chatFileptb'
    conversation_no = '0'
    transcript_index = '0'
    pos = ''
    trees = ''
    ptb_treenumbers = '0'
    utterance_index = '0'
    subutterance_index = '0'
    caller = 'A'
    act_tag = ''
    text = raw_input("YOU: ")
    if text == "":
        return None
    if get_act_tag:
        act_tag = raw_input("Act Tag: ")
    row = [swda_filename, ptb_basename, conversation_no, transcript_index, act_tag, caller, utterance_index, subutterance_index, text, pos, trees, ptb_treenumbers]
    return Utterance(row, transcript_metadata)

#returns candidate interruption
def getInterruptCandidate(turnA, NUM_CANDIDATES, weights, featureExtractor):

        global turns

        candidates = random.sample(turns, min(NUM_CANDIDATES, len(turns)))
        maxCandidate = candidates[0]
        maxScore = 0
        for candidate in candidates:
            score = 0
            if candidate[0].act_tag == 'b':
                score = dotProduct(weights, featureExtractor((turnA, candidate)))
            if candidate[0].act_tag == '^2':
                score = dotProduct(weights, featureExtractor((turnA, candidate)))
            if score > maxScore:
                    maxScore = score
                    maxCandidate = candidate
        return (maxCandidate, maxScore)

def considerInterrupt(turnA, NUM_CANDIDATES, weights, featureExtractor):

    global bad_turn_counter

    if len(turnA) > INTERRUPT_AFTER_LEN:
        candidate, score = getInterruptCandidate(turnA, NUM_CANDIDATES, weights, featureExtractor)
        # if random.random() < score * INTERRUPT_PROB:
        if random.random() < INTERRUPT_PROB and score > INTERRUPT_THRESHOLD:
            for utt in candidate:
                print "ME (" + utt.act_tag + "): " + utt.text
            print "SCORE"
            print score
            if isBadTurn(candidate):
                bad_turn_counter += 1
            else:
                bad_turn_counter = 0
            return True
    return False

def swda_chat(weights, featureExtractor, turnSet, restrict_caller = True, restrict_bad_turns = True):        
        global turns, transcript_metadata, get_act_tag, use_specific_discourse, use_distribution
        global bad_turn_counter

        if not getYesOrNo("Do you want to run the chat bot?"):
                return False

        print "Hello, I am BobBot!"
        print "I am trained in natural conversation."
        print "Let's pretend we are two people talking on the phone."

        setupBot(turnSet)

        print "Great.  You may speak several sentences at a time if you wish."
        print "Enter an empty line (just hit ENTER) to have me respond."
        print "Enter hit ENTER without any other lines to exit."

        if use_specific_discourse:
                print "Here is the first few exchanges in the discourse"
                print "Carry on the conversation with me"
                printTurns(turns, True, 6)
        
        while True:
                # User turn
                turnA = []
                while True:
                        user_utterance = getUserUtterance(get_act_tag, transcript_metadata)
                        if not user_utterance:
                            break
                        turnA.append(user_utterance)
                        interrupted = considerInterrupt(turnA, NUM_CANDIDATES, weights, featureExtractor)
                        if interrupted:
                            turnA = []
                if not turnA:
                        return True
                
                # Bot turn

                #Using set of candidate turns
                candidates = random.sample(turns, min(NUM_CANDIDATES, len(turns)))
                candidates_and_scores = []
                maxCandidate = candidates[0]
                maxScore = dotProduct(weights, featureExtractor((turnA, candidates[0])))
                for candidate in candidates:
                        if restrict_bad_turns and bad_turn_counter >= BAD_TURN_MAX and isBadTurn(candidate):
                                continue
                        if restrict_caller and candidate[0].caller != 'B':
                                continue
                        score = dotProduct(weights, featureExtractor((turnA, candidate)))
                        if score > 0:
                                candidates_and_scores.append((candidate, score))
                        if score > maxScore:
                              maxScore = score
                              maxCandidate = candidate
                if use_distribution:
                    maxCandidate, maxScore = chooseFromDistribution(candidates_and_scores)
                print_candidates_and_scores(candidates_and_scores)
                if isBadTurn(maxCandidate):
                        bad_turn_counter += 1
                else:
                        bad_turn_counter = 0
                for utt in maxCandidate:
                        print "ME (" + utt.act_tag + "): " + utt.text
                # print featureExtractor((turnA, maxCandidate))
                print "SCORE"
                print maxScore
        return True