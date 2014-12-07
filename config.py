# For actual_chat_bot
INTERRUPT_AFTER_LEN = 2
INTERRUPT_PROB = .25
INTERRUPT_THRESHOLD = 5
BAD_TURN_MAX = 1
NUM_CANDIDATES = 100
NUM_START = 6

#For bot_utils
pre_process_restrict_bad = False
neg_restrict_bad = True
PRE_PROCESS_NOISE = ['b', '%', 'x']
BAD_TURN_NOISE = ['b', '%', 'x']

# For bot
WEIGHTS_FILENAME = "weights.json"
TRAIN_SET_SIZE = 750
TEST_SET_SIZE = 180
FINAL_TEST_SET_SIZE = 180

RANDOM_SEED = 5

# For SGD
numIters = 40
stepSize = .05
dampeningFactor = 1
# CONVERGENCE_MAX = 0
# .01 18 steps converged average
# .01, .95 60 steps converged
