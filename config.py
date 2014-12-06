# For actual_chat_bot
INTERRUPT_AFTER_LEN = 2
INTERRUPT_PROB = .25
INTERRUPT_THRESHOLD = 5
BAD_TURN_MAX = 1
NUM_CANDIDATES = 100
NUM_START = 6

#For bot_utils
restrict_bad_utterances = False
restrict_bad_turns = True
PRE_PROCESS_NOISE = ['b', '%', 'x']
BAD_TURN_NOISE = ['b', '%', 'x']

# For bot
WEIGHTS_FILENAME = "weights.json"
TRAIN_SET_SIZE = 1000
TEST_SET_SIZE = 50 # if 100, may infinite loop (one speaker?)
RANDOM_SEED = 5

# For SGD
numIters = 40
stepSize = .05


