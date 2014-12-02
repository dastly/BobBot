def subtitles_feature_extractor(x):
    # phi = dict()
    # turnA = x[0]
    # turnB = x[1] # less important since we don't actually have speaker
    # # info

    # features_to_use = [
    #    A_ending_punct
    # ]
    
    # fn = lambda x : phi.update(x(turnA, turnB))
    # map(fn, features_to_use)
    phi = {"test" : 1}
    return phi

###### A only features #################
    
def A_ending_punct(turnA, turnB):
    return {"A_ending_punct" : turnA[-1]}

####### B only features ################
    
def B_ending_punct(turnA, turnB):
    return {"B_ending_punct" : turnB[-1]}
