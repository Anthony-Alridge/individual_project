import spacy
from spacy.symbols import *
import numpy as np
import re
#
#
#
model = spacy.load('en_core_web_md')


def find_related(word):
    subject = None
    xcomp_verb = None
    object = None
    for cand in word.children:
        print(cand)
        print(cand.dep_)
        if cand.dep == nsubj:
            print(f'nsubj is {cand}')
        if cand.dep == xcomp:
            print(f'xcomp is {cand}')
        if cand.dep == dobj:
            print(f'nobj is {cand}')


def process(tokens):
    for token in tokens:
        if token.pos == VERB:
            find_related(token)

doc = model("Ian volunteered to eat Dennis's menudo after already having a bowl because Dennis despised eating intestine")

token = doc[1]
process(doc)
