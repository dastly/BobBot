import sys
import pdb

def swda_feature_extractor(x):
    phi = dict()
    turnA = x[0]
    turnB = x[1] # both utterances

    # Note: Single features create a lot more error than pairwise features
    features_to_use = [
        act_tag,
        # short, 
        # gender,
        # length, 
        # A_contains_yes_no_question,
        # A_contains_declarative_yn_question,
        # B_is_yes_no_response,
        # contains_question,
        # contains_stmt,
        # contains_acknowledge
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

######### A ONLY FEATURES ################

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

def short(turnA, turnB):
    if len(turnA) + len(turnB) < 4:
        return {"Short" : 1}
    return {}

def length(turnA, turnB):
    return {"length: {0}, {1}".format(len(turnA), len(turnB)) : 1}

def act_tag(turnA, turnB):
    tagA = turnA[len(turnA)-1].act_tag
    tagB = turnB[0].act_tag
    return {"{0}, {1}".format(tagA, tagB) : 1}


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

############ HELPERS #################

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

"""
- contains question
- contains y or n answer
- yes/no question ('qy') with yes/no response ('ny' or 'nn')
- same as above but with declarative y/no qs (qy^d)


- contains statement
- question / statement

- contains open ended question
- contains opinion
- open ended question / opinion

- contains wh-question
- contains statement nonopinion
- wh-question / nonopinion

- maybe
- general question / maybe

- end with right? ***

- uh huh
- summarize/reformulate / uh huh
- summarize/reformulate / positive answer

- question / i don't know

- collaborative completion
- collaborative completion / maybe
- collaborative completion / uh-huh/yes
- collaborative completion / reject

- apology
- apology / that's ok (downplayer)


"""
