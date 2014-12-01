import xml.etree.ElementTree as ET
from nltk.tag import pos_tag
import os, glob

KNOWN_MALFORMED_FILES = {"4055_184230_245623_aleksandr_nevskiy.xml"}
XML_PATH = "./en/xml/"

def get_movie_sentences(filename):
    if os.path.basename(filename) in KNOWN_MALFORMED_FILES:
        return []
    tree = ET.parse(filename)
    root = tree.getroot()
    movie = []
    for sentence in root:
        if sentence.tag != 's':
            continue
        sentence_words = []        
        for word in sentence:
            if word.tag != 'w':
                continue
            sentence_words.append(word.text)
        if len(sentence_words) == 0:
            continue
        movie.append(pos_tag(sentence_words))
    return movie

def tag_all_movies():
    files = glob.glob(XML_PATH + "*.xml")
    for filename in files:
        movie = get_movie_sentences(filename)
        if len(movie) == 0:
            continue
        yield movie

