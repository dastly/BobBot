import swda
from swda import Transcript
from swda import CorpusReader
from swda import Utterance
from util import dotProduct
from features import swda_feature_extractor
from nltk.grammar import PCFG, induce_pcfg, toy_pcfg1, toy_pcfg2
from nltk.parse.generate import generate
import random
from bot_utils import isBadTurn
from bot_utils import printTurns

def getYesOrNo(prompt):
        print prompt
        yesOrNo = raw_input("Y/N: ")
        if yesOrNo in ["Y", "y", "YES", "Yes", "yes"]: return True
        return False

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

def swda_chat(weights, featureExtractor, turnSet, restrict_caller = True, restrict_bad_turns = True, NUM_CANDIDATES = 100, NUM_START = 6):        
        if not getYesOrNo("Do you want to run the chat bot?"):
                return False

        print "Hello, I am BobBot!"
        print "I am trained in natural conversation."
        print "Let's pretend we are two people talking on the phone."

##        get_demo_info = getYesOrNo("Do we want to give demographic info about ourselves?")
        transcript_metadata = getTranscript_Metadata(False)

        get_act_tag = getYesOrNo("Should I ask for act_tags?")

        turns = []
        use_specific_discourse = getYesOrNo("Do you want to use a specific discourse?")
        if use_specific_discourse:
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
                turns = turnSet[int(discourse_num)]
        else:
                for discourse in turnSet:
                        turns.extend(discourse)

        print "Great.  You may speak several sentences at a time if you wish."
        print "Enter an empty line (just hit ENTER) to have me respond."
        print "Enter DONE to finish the conversation, or hit ENTER without any other lines."

        # Junk info to build proper utterance
        # Any info actually used in a feature must be changed
        #   to either be accurate or at least neutral with respect to evaluation
        # Move useful info into the while loop below
        swda_filename = 'chatFile'
        ptb_basename = 'chatFileptb'
        conversation_no = '0'
        transcript_index = '0'
        pos = ''
        trees = ''
        ptb_treenumbers = '0'
        utterance_index = '0'
        subutterance_index = '0'

        if use_specific_discourse:
                print "Here is the first few exchanges in the discourse"
                print "Carry on the conversation with me"
                printTurns(turns, True, 6)
        
        bad_turn_counter = 0
        while True:
                # The only information that needs to be updated here is pos tags and parsings
                # I have act_tag as manual input, but ideally this bot would use the other algorithm already built
                act_tag = ''
                caller = 'A'
                turnA = []
                while True:
                        text = raw_input("YOU: ")
                        if text == "":
                                break
                        if text == "DONE":
                                return True
                        if get_act_tag:
                                act_tag = raw_input("Act Tag: ")
                        row = [swda_filename, ptb_basename, conversation_no, transcript_index, act_tag, caller, utterance_index, subutterance_index, text, pos, trees, ptb_treenumbers]
                        turnA.append(Utterance(row, transcript_metadata))

                if not turnA:
                        return True
                
                BAD_TURN_MAX = 1

                #Using set of candidate turns
                candidates = random.sample(turns, min(NUM_CANDIDATES, len(turns)))
                candidate_and_score = []
                maxCandidate = candidates[0]
                maxScore = dotProduct(weights, featureExtractor((turnA, candidates[0])))
                for candidate in candidates:
                        if restrict_bad_turns and bad_turn_counter >= BAD_TURN_MAX and isBadTurn(candidate):
                                continue
                        if restrict_caller and candidate[0].caller != 'B':
                                continue
                        score = dotProduct(weights, featureExtractor((turnA, candidate)))
                        candidate_and_score.append((candidate, score))
                        if score > maxScore:
                              maxScore = score
                              maxCandidate = candidate
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

"""
OLD JUNK
                ##This will be replaced with a turnB generated from a grammar or given set of responses
                ##Useful for manual testing and error analysis
                act_tag = ''
                caller = 'B'
                turnB = []
                while True:
                        text = raw_input("ME: ")
                        if text == "":
                                break
                        if text == "DONE":
                                return
                        if get_act_tag:
                                act_tag = raw_input("Act Tag: ")
                        row = [swda_filename, ptb_basename, conversation_no, transcript_index, act_tag, caller, utterance_index, subutterance_index, text, pos, trees, ptb_treenumbers]
                        turnB.append(Utterance(row, transcript_metadata))
                if not turnB:
                        return
                
                score = dotProduct(weights, featureExtractor((turnA, turnB)))
                print "SCORE"
                print score
                
            
                

##        row = ['sw00utt/sw_0001_4325.utt', '4/sw4325', '4325', '0', 'o', 'A', '1', '1', 'Okay.  /', 'Okay/UH ./.', '(INTJ (UH Okay) (. .) (-DFL- E_S))', '1']
##        transcript_metadata = {'conversation_no': 4346, 'prompt': 'DITEMS TODAYY', 'from_caller_birth_year': '1963', 'from_caller_education': 1, 'to_caller_birth_year': '1963', 'length': 5, 'from_caller_dialect_area': 'SOUTH MIDLAND', 'from_caller_sex': 'FEMALE', 'to_caller_education': 2, 'to_caller_sex': 'MALE', 'talk_day': '920323', 'to_caller_dialect_area': 'NORTHERN', 'topic_description': 'PUBLIC EDUCATION'}
##        utt = Utterance(row, transcript_metadata)
##        print featureExtractor(([utt], [utt]))
##        print 'swda_filename'
##        print utt.swda_filename
##        print 'ptb_basename'
##        print utt.ptb_basename
##        print 'conversation_no'
##        print utt.conversation_no
##        print 'transcript_index'
##        print utt.transcript_index
##        print 'act_tag'
##        print utt.act_tag
##        print 'caller'
##        print utt.caller
##        print 'uindex'
##        print utt.utterance_index
##        print 'suindex'
##        print utt.subutterance_index
##        print 'text'
##        print utt.text
##        print 'pos'
##        print utt.pos
##        print 'trees'
##        print utt.trees
##        print 'ptbtn'
##        print utt.ptb_treenumbers
##        header = [
##        'swda_filename',      # (str) The filename: directory/basename
##        'ptb_basename',       # (str) The Treebank filename: add ".pos" for POS and ".mrg" for trees
##        'conversation_no',    # (int) The conversation Id, to key into the metadata database.
##        'transcript_index',   # (int) The line number of this item in the transcript (counting only utt lines).
##        'act_tag',            # (list of str) The Dialog Act Tags (separated by ||| in the file).
##        'caller',             # (str) A, B, @A, @B, @@A, @@B
##        'utterance_index',    # (int) The encoded index of the utterance (the number in A.49, B.27, etc.)
##        'subutterance_index', # (int) Utterances can be broken across line. This gives the internal position.
##        'text',               # (str) The text of the utterance
##        'pos',                # (str) The POS tagged version of the utterance, from PtbBasename+.pos
##        'trees',              # (list of nltk.tree.Tree) The tree(s) containing this utterance (separated by ||| in the file).
##        'ptb_treenumbers'     # (list of int) The tree numbers in the PtbBasename+.mrg
##       ]
        
##swda_chat(None, swda_feature_extractor)

# Example of getting random sentence from a grammar
# Note that generate can be used to get all sentences from a grammar
#  from which several candidates can be chosen

grammar = toy_pcfg2
print grammar
sentences = []
for sentence in generate(grammar, n=100):
    sentences.append(' '.join(sentence))
print random.choice(sentences)

Generate can be rewritten to generate according to the probabilities of a pcfg
"""
