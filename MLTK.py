import nltk
from nltk.corpus import nps_chat
from nltk.corpus import switchboard
from nltk.corpus import treebank
from nltk.parse import pchartd
from nltk.parse.pchart.BottomUpProbabilisticChartParser

parser=nltk.parse.stanford.StanfordParser(
    model_path="edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz"
    )
parser.raw_parse_sents((
    "the quick brown fox jumps over the lazy dog",
    "the quick grey wolf jumps over the lazy fox"
))

##nltk.chat.chatbots()

##for fileid in nps_chat.fileids():
##    posts = nps_chat.posts(fileid)
##    for post in posts:
##        print(post)

##for fileid in switchboard.fileids():
##    for discourse in switchboard.tagged_discourses():
##        print(discourse)

##start = nltk.grammar.Nonterminal("Root")
##hello = nltk.grammar.Nonterminal("Hello")
##world = nltk.grammar.Nonterminal("World")
##productions = [nltk.grammar.Production(start, [hello, world]), nltk.grammar.Production(start, [world]), nltk.grammar.Production(start, [world])]
##pcfg = nltk.grammar.induce_pcfg(start, productions)
##for production in pcfg.productions():
##    print(production.lhs())
##    print(production.rhs())
##    print(production.prob())
##productions.append(nltk.grammar.Production(start, [hello, world]))
##pcfg = nltk.grammar.induce_pcfg(start, productions)
##for production in pcfg.productions():
##    print(production.lhs())
##    print(production.rhs())
##    print(production.prob())
##
##print("Induce PCFG grammar from treebank data:")
##
##productions = []
##for item in treebank._fileids:
####    item = treebank._fileids[0]
##    for tree in treebank.parsed_sents(item):
##        # perform optional tree transformations, e.g.:
##        tree.collapse_unary(collapsePOS = False)
##        tree.chomsky_normal_form(horzMarkov = 2)
##
##        productions += tree.productions()
##
##S = nltk.grammar.Nonterminal('S')
##grammar = nltk.grammar.induce_pcfg(S, productions)
##print(grammar)
##print()
##
##print("Parse sentence using induced grammar:")
##
##parser = pchart.InsideChartParser(grammar)
##parser.trace(3)
##
### doesn't work as tokens are different:
###sent = treebank.tokenized('wsj_0001.mrg')[0]
##
##sent = treebank.parsed_sents(item)[0].leaves()
##print(sent)
##parse = parser.parse(sent)[0]
##print(parse)

##nltk.grammar.pcfg_demo()
