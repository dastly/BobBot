import sys
from nltk.tree import Tree
from sets import Set

def baseline_feature_extractor(x):
    phi = dict()
    return phi

def swda_feature_extractor(x):
    phi = dict()
    turnA = x[0]
    turnB = x[1] # both utterances

    features_to_use = [

    # I tried removing all act tag stuff (minus pairwise, and it went to 26%/21%)
        act_tag, ## Without this, error is back near 50%.  With it, it is around 21%
        act_tag2, ## 
        act_tag3, ##
        act_tag4, ## 2-4 provide an additional 2% decrease
        act_tag5, ## Like 1-4 combined  Changes from 15/13 to 14/14

        act_tag_set, ## Another 4%!
        # pronouns, # Adds 3% error
        # short_turn, 
        # gender,
        turn_length,
        turn_length_words, 

        utt_length1,
        utt_length2,
        utt_length3,
        utt_length4,
        utt_length5,
        utt_length_last_2_first_2,

        # Length is good, but it seems to worsen the actual chatting a bit.
        # This may just be because we tend to type in shorter sentences,
        # but speak in longer ones.  This discrepency causes issues.

        # A_act_tag_list,
        # B_act_tag_list,
        # A_act_tag_pairs,
        # B_act_tag_pairs,
        # pos_tags1, # increases error by a few % points
        # A_pos_pairs,
        # B_pos_pairs,  # having A and B pos_pairs increased error by 13%
 
        ########### CREATE A LOT OF ERROR ###########
        
        # A_contains_yes_no_question,
        # A_contains_declarative_yn_question,
        # B_is_yes_no_response,
        # contains_question,
        # contains_stmt,
        # contains_acknowledge,

        ############################################
        # FOR INTERRUPTIONS #
        interruption_features,

        A_add_subjects,
        B_add_subjects,
    ]
    mod = sys.modules[__name__]
    fn = lambda x : phi.update(x(turnA, turnB))
    map(fn, features_to_use)

    pair_features = [
        # FORMAT: [<string key for phi>, function1, function2]
        ["yn_question_response", A_contains_yes_no_question, B_is_yes_no_response],
        ["yn_question_response", A_contains_declarative_yn_question, B_is_yes_no_response],
        ["yn_question_maybe", A_contains_yes_no_question, B_is_maybe],
        ["yn_question_maybe", A_contains_declarative_yn_question, B_is_maybe],
        ["yn_question_other_answer", A_contains_yes_no_question, B_is_other_answer],
        ["yn_question_other_answer", A_contains_declarative_yn_question, B_is_other_answer],
        ["question_stmt_pair", A_contains_non_yn_question, B_contains_stmt],
        ["open_question_and_opinion", A_contains_open_ended_question, B_contains_opinion_stmt],
        ["wh_question_nonopinion_stmt", A_contains_wh_question, B_contains_nonopinion_stmt],
        ["summarize_acknowledge", A_ends_with_summarize, B_is_acknowledge],
        ["summarize_positive", A_ends_with_summarize, B_is_positive],
        ["collab_completion_acknowledge", A_ends_with_collab_completion, B_is_acknowledge],
        ["collab_completion_maybe", A_ends_with_collab_completion, B_is_other_answer],
        ["collab_completion_reject", A_ends_with_collab_completion, B_is_reject],
        ["apology_downplayer", A_contains_apology, B_contains_downplayer]
        #The above altogether yeild ~.3% reduction
    ]

    fn = lambda x : phi.update(create_pair_feature(x, turnA, turnB))
    map(fn, pair_features)
    
    return phi

def create_pair_feature(fn_names, turnA, turnB):
    key, feature1, feature2 = fn_names
    call_feature = lambda fn : fn(turnA, turnB, True)
    if call_feature(feature1) and call_feature(feature2):
        return {key : 1}
    return {}

######### TREE FEATURES ################

def A_add_subjects(turnA, turnB, return_flag=False):
    return add_subjects(turnA, 'A')

def B_add_subjects(turnA, turnB, return_flag=False):
    return add_subjects(turnB, 'B')

######### A ONLY FEATURES ################

def A_act_tag_list(turnA, turnB):
    return act_tag_list_helper(turnA, 'A')

def A_pos_pairs(turnA, turnB):
    return pos_pairs_helper(turnA, 'A')

def A_act_tag_pairs(turnA, turnB):
    return act_tag_pairs_helper(turnA, 'A')

def A_contains_declarative_yn_question(turnA, turnB, return_flag=False):
    return contains_something(turnA, is_declarative_yn_question, "A_contains_yn_question", return_flag)

def A_contains_yes_no_question(turnA, turnB, return_flag=False):
    return contains_something(turnA, is_yes_no_question, "A_contains_yn_question", return_flag)

def A_contains_non_yn_question(turnA, turnB, return_flag=False):
    return contains_something(turnA, is_non_yn_question, "A_contains_non_yn_question", return_flag)

def A_ends_with_summarize(turnA, turnB, return_flag=False):
    return last_is(turnA, is_summarize, "A_ends_with_summarize", return_flag)

def A_contains_open_ended_question(turnA, turnB, return_flag=False):
    return contains_something(turnA, is_open_ended_question, "A_contains_open_ended_question", return_flag)

def A_contains_wh_question(turnA, turnB, return_flag=False):
    return contains_something(turnA, is_wh_question, "A_contains_wh_question", return_flag)

def A_ends_with_collab_completion(turnA, turnB, return_flag=False):
    return last_is(turnA, is_collab_completion, "A_ends_with_collab_completion", return_flag)

def A_contains_apology(turnA, turnB, return_flag=False):
    return contains_something(turnA, is_apology, "A_contains_apology", return_flag)

######### B ONLY FEATURES ###############

def B_pos_pairs(turnA, turnB):
    return pos_pairs_helper(turnB, 'B')

def B_act_tag_pairs(turnA, turnB):
    return act_tag_pairs_helper(turnB, 'B')

def B_act_tag_list(turnA, turnB):
    return act_tag_list_helper(turnB, 'B')
    
def B_is_yes_no_response(turnA, turnB, return_flag=False):
    fn = lambda x: is_yes(x) or is_no(x)
    return exactly_one(turnB, fn, "B_is_yn_response", return_flag)

def B_contains_opinion_stmt(turnA, turnB, return_flag=False):
    return contains_something(turnB, is_opinion_stmt, "B_contains_opinion_stmt", return_flag)

def B_contains_stmt(turnA, turnB, return_flag=False):
    return contains_something(turnB, is_stmt, "B_contains_stmt", return_flag)

def B_contains_nonopinion_stmt(turnA, turnB, return_flag=False):
    return contains_something(turnB, is_nonopinion_stmt, "B_contains_nonopinion_stmt", return_flag)

def B_is_maybe(turnA, turnB, return_flag=False):
    return first_is(turnB, is_maybe, "B_is_maybe", return_flag)

def B_is_acknowledge(turnA, turnB, return_flag=False):
    return exactly_one(turnB, is_acknowledge, "B_is_acknowledge", return_flag)

def B_is_positive(turnA, turnB, return_flag=False):
    fn = lambda x : is_yes(x) or is_accept(x)
    return first_is(turnB, fn, "B_is_positive", return_flag)

def B_is_other_answer(turnA, turnB, return_flag=False):
    return first_is(turnB, is_other_answer, "B_is_other_answer", return_flag)

def B_is_reject(turnA, turnB, return_flag=False):
    return first_is(turnB, is_reject, "B_is_reject", return_flag)

def B_contains_downplayer(turnA, turnB, return_flag=False):
    return contains_something(turnB, is_downplayer, "B_contains_downplayer", return_flag)
    
########### BOTH #################

# POS tags

def pos_tags1(turnA, turnB):
    return pos_tags_helper(0, turnA, turnB)

def pos_tags_helper(num, turnA, turnB):
    if len(turnA) > num and len(turnB) > num:
        uttA = turnA[-1 * (num+1)].pos_lemmas()
        uttB = turnB[num].pos_lemmas()
        fn = lambda lst, tup : lst + [tup[1]]
        posA = reduce(fn, uttA, [])
        posB = reduce(fn, uttB, [])
        return {"pos1 : {0}, {1}" : 1}
    return {}

# Turn length

def short_turn(turnA, turnB):
    if len(turnA) + len(turnB) < 4:
        return {"Short" : 1}
    return {}

def turn_length(turnA, turnB):
    return {"length: {0}, {1}".format(len(turnA), len(turnB)) : 1}

def turn_length_words(turnA, turnB):
    lengthA = 0
    lengthB = 0
    for utt in turnA:
        lengthA += len(utt.text_words())
    for utt in turnB:
        lengthB += len(utt.text_words())
    return {"length: {0}, {1}".format(lengthA, lengthB) : 1}

# Utterance length
    
def utt_length1(turnA, turnB):
    return get_utt_length(turnA, turnB, 0) # getting the last of A and first of B

def utt_length2(turnA, turnB):
    return get_utt_length(turnA, turnB, 1) # 2nd last of A and 2nd of B

def utt_length3(turnA, turnB):
    return get_utt_length(turnA, turnB, 2)

def utt_length4(turnA, turnB):
    return get_utt_length(turnA, turnB, 3)

def utt_length5(turnA, turnB):
    return get_utt_length(turnA, turnB, 4)

def utt_length_last_2_first_2(turnA, turnB):
    if len(turnA) >= 2 and len(turnB) >= 2:
        second_last_A = turnA[-2]
        last_A = turnA[-1]
        first_B = turnB[0]
        second_B = turnB[1]
        lengths = map(lambda utt : len(utt.text_words()), [second_last_A, last_A, first_B, second_B])
        return {"Utt length: last 2 = {0}, {1} / first 2 = {2}, {3}".format(lengths[0], lengths[1], lengths[2], lengths[3]) : 1}
    return {}

def get_utt_length(turnA, turnB, num):
    if len(turnA) > num and len(turnB) > num:
        lengthA = len(turnA[-1 * (num+1)].text_words())
        lengthB = len(turnB[num].text_words())
        return {"utt_length{2}: {0}, {1}".format(lengthA, lengthB, num) : 1} # TODO Change back
    return {}
    
def act_tag(turnA, turnB): # last of A, first of B
    tagA = turnA[len(turnA)-1].act_tag
    tagB = turnB[0].act_tag
    return {"Tag1: {0}, {1}".format(tagA, tagB) : 1}

def act_tag2(turnA, turnB): # 2nd last of A, 2nd of B
    if len(turnA) > 1 and len(turnB) > 1:
        tagA = turnA[len(turnA)-2].act_tag
        tagB = turnB[1].act_tag
        return {"Tag2: {0}, {1}".format(tagA, tagB) : 1}
    return{}

def act_tag3(turnA, turnB): # 2nd to last of A, first of B
    if len(turnA) > 1:
        tagA = turnA[len(turnA)-2].act_tag
        tagB = turnB[0].act_tag
        return {"Tag3: {0}, {1}".format(tagA, tagB) : 1}
    return{}

def act_tag4(turnA, turnB): # last of A, second of B
    if len(turnB) > 1:
        tagA = turnA[len(turnA)-1].act_tag
        tagB = turnB[1].act_tag
        return {"Tag4: {0}, {1}".format(tagA, tagB) : 1}
    return{}

def act_tag5(turnA, turnB):
    secondToLastA = lastA = firstB = secondB = ''
    if len(turnA) > 1:
        secondToLastA = turnA[len(turnA)-2].act_tag
    lastA = turnA[len(turnA)-1].act_tag
    firstB = turnB[0].act_tag
    if len(turnB) > 1:
        secondB = turnB[1].act_tag
    return{"Last 2 First 2: {0}, {1} ; {2} {3}".format(secondToLastA, lastA, firstB, secondB) : 1}

def act_tag_set(turnA, turnB): # entire set for both turns
    tagSetA = Set([turn.act_tag for turn in turnA])
    tagSetB = Set([turn.act_tag for turn in turnB])
    return{"Tag Set: {0}, {1}".format(tagSetA, tagSetB) : 1}

def gender(turnA, turnB):
    genderA = turnA[len(turnA)-1].caller_sex
    genderB = turnB[0].caller_sex
    return {"{0}, {1}".format(genderA, genderB) : 1}

def contains_question(turnA, turnB):
    return both_contain_something(turnA, turnB, is_question, "both_contain_question")

def contains_acknowledge(turnA, turnB):
    return both_contain_something(turnA, turnB, is_acknowledge, "both_contain_acknowledge")

def contains_stmt(turnA, turnB):
    return both_contain_something(turnA, turnB, is_stmt, "both_contain_stmt")

############ ACT TAG HELPERS #################

def pos_pairs_helper(turn, order):
    result = {}
    for utt in turn:
        tupls = utt.pos_lemmas()
        for i in range(0, len(tupls)-1):
            word1, tag1 = tupls[i]
            word2, tag2 = tupls[i+1]
            result['{2}_pos_pair={0}, {1}'.format(tag1, tag2, order)] = 1
    return result

def act_tag_pairs_helper(turn, order):
    result = {}
    for i in range(0, len(turn)-1):
        tag1 = turn[i].act_tag
        tag2 = turn[i].act_tag
        result['{2}_act_tag_pair={0}, {1}'.format(tag1, tag2, order)] = 1
    return result

def act_tag_list_helper(turn, order):
    tag_list = map(lambda x : x.act_tag, turn)
    return { '{0}_act_tag_list={1}'.format(order, tag_list) : 1}

def first_is(turn, fn, key, return_flag):
    return check_specific(turn, 0, fn, key, return_flag)

def last_is(turn, fn, key, return_flag):
    return check_specific(turn, len(turn)-1, fn, key, return_flag)

def exactly_one(turn, fn, key, return_flag):
    if len(turn) == 1 and fn(turn[0]):
        return return_flag or {key : 1}
    if return_flag:
        return False
    return {}

def check_specific(turn, index, fn, key, return_flag):
    if len(turn) > 0 and fn(turn[index]):
        return return_flag or {key : 1}
    if return_flag:
        return False
    return {}
    
def contains_something(turn, fn, key, return_flag):
    for utt in turn:
        if fn(utt):
            return return_flag or {key : 1}
    if return_flag:
        return False
    return {}

def both_contain_something(turnA, turnB, fn, key):
    a = b = False
    for utt in turnA:
        if fn(utt):
            a = True
            break
    for utt in turnB:
        if fn(utt):
            b = True
            break
    return {"{0} - {1}, {2}".format(key, a, b) : 1}

def is_question(utt):
    return utt.act_tag in {'qy', 'qw', 'qy^d', 'qo', 'qw^d'}

def is_non_yn_question(utt):
    return is_question(utt) and not (is_yes_no_question(utt))

def is_yes_no_question(utt):
    return utt.act_tag == 'qy'

def is_open_ended_question(utt):
    return utt.act_tag == 'qo'

def is_declarative_yn_question(utt):
    return utt.act_tag == 'qy^d'

def is_wh_question(utt):
    return utt.act_tag in {'qw', 'qw^d'}

def is_opinion_stmt(utt):
    return utt.act_tag == 'sv'

def is_nonopinion_stmt(utt):
    return utt.act_tag == 'sd'

def is_accept(utt):
    return utt.act_tag == 'aa'

def is_reject(utt):
    return utt.act_tag == 'ar'
    
def is_yes(utt):
    return utt.act_tag == 'ny'

def is_no(utt):
    return utt.act_tag == 'nn'

def is_negative(utt):
    return utt.act_tag in {'nn', 'ng'}

def is_maybe(utt):
    return utt.act_tag == 'aap_am'

def is_stmt(utt):
    return utt.act_tag in {'sd', 'sv'}

def is_acknowledge(utt):
    return utt.act_tag == 'b'

def is_summarize(utt):
    return utt.act_tag == 'bf'

def is_other_answer(utt):
    return utt.act_tag == 'no'

def is_collab_completion(utt):
    return utt.act_tag == '^2'

def is_apology(utt):
    return utt.act_tag == 'fa'

def is_downplayer(utt):
    return utt.act_tag == 'bd'

########### TREE HELPERS #############

def is_none(label):
    return label == "-NONE-"

def add_subjects(turn, order):
    weights = dict()
    weights_fn = lambda subj: weights.update({"{0}-subject={1}".format(order, subj) : 1})
    get_subjects_fn = lambda utt: map(weights_fn, get_subjects(utt))
    map(get_subjects_fn, turn)
    return weights

def get_subjects(utt):
    if not utt.tree_is_perfect_match():
        return []
    tree = Tree.fromstring(utt.trees[0].pprint())
    subj_phrases = []
    for sub in tree.subtrees():
        if 'SBJ' in sub.label():
            subj_phrases.append(sub)
    subj_nodes = []
    for node in subj_phrases:
        child = None
        for sub in node.subtrees():
            if sub.subtrees().next().height() == 2:
                child = sub
                break
        subj_nodes.append(child)
    
    subj_nodes = [node for node in subj_nodes if not is_none(node.label())]
    subj_labels = [node.leaves()[0] for node in subj_nodes if len(node.leaves()) == 1]
    return subj_labels
    
##### INTERRUPTION FEATURES ######

def interruption_features(turnA, turnB):
    result = {}
    if len(turnB) > 0:
        interruption_type = None
        if is_acknowledge(turnB[0]):
            interruption_type = 'backchannel'
        elif is_collab_completion(turnB[0]):
            interruption_type = 'collab'
        if interruption_type:
            result['turn_length_pre_backchannel_interruption={0}'.format(len(turnA), interruption_type)] = 1
            act_tags = set([utt.act_tag for utt in turnA])
            result['pre_{1}_tags={0}'.format(act_tags, interruption_type)] = 1
    return result

##### PRONOUN FEATURES #####
def pronouns(turnA, turnB):
    AIfound = BIfound = AYouFound = BYouFound = False
    for utt in turnA:
        if utt.text.find("I"):
            AIfound = True
        if utt.text.find("you"):
            AYouFound = True
    for utt in turnB:
        if utt.text.find("I"):
            BIfound = True
        if utt.text.find("you"):
            BYouFound = True
    proA = proB = ""
    if AIfound and AYouFound: proA = "us"
    if BIfound and BYouFound: proB = "us"
    if AIfound and not AYouFound: proA = "I"
    if not AIfound and AYouFound: proA = "You"
    if BIfound and not BYouFound: proB = "I"
    if not BIfound and BYouFound: proB = "You"
    if AIfound or BIfound or AYouFound or BYouFound:
        return {"Pronouns: {0} {1}".format(proA, proB): 1}
    return {}
