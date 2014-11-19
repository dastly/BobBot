import sys

def swda_feature_extractor(x):
    phi = dict()
    turnA, turnB = x # both utterances

    features_to_use = [
        "act_tag",
        "short"
    ]
    fn = lambda x : phi.update(getattr(sys.modules[__name__], x)(turnA, turnB))
    map(fn, features_to_use)
    return phi

# length
def short(turnA, turnB):
    if len(turnA) + len(turnB) < 4:
        return {"Short" : 1}
    return {}

# ACT tags
def act_tag(turnA, turnB):
    tagA = turnA[len(turnA)-1].act_tag
    tagB = turnB[0].act_tag
    return {tagA + ", " + tagB : 1}

"""
- same caller_no
- the actual caller_no

"""
