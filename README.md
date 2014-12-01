BobBot
======

Applies SGD with Features to conversations, basically that yeah.

Requires nltk:
http://www.nltk.org/index.html

Relevant corpuses:
switchboard
nps_chat

#### Getting the SWDA corpus

    python get_swda.py

###Utilities to download and POS tag subtitles from the Open Subtitles corpus (http://opus.lingfil.uu.se/OpenSubtitles.php)

###DOWNLOADING THE CORPUS

    python get_xml.py [-d] [-g] [-x]
    
-d = to download the source gz files that have the names of xml files (step 1)

-g = to extract those gz files (step 2)

-x = to get the actual xml files (step 3)


####Running this for the first time:

    python get_xml.py -dgx

###GETTING TAGGED SUBTITLES

    import parse_xml as parse

    for movie in parse.tag_all_movies():
        print movie

Each movie is a list of sentences. Each sentence is a list of tuples, containing the word and its POS tag (tagged with nltk). 