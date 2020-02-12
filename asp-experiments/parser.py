import spacy
from spacy import displacy

nlp = spacy.load('en_core_web_md')
sentence = 'The man could not lift his son because he was so weak.'

class Rule:
    def apply_negative(self, token, negative):
        if negative:
            return 'not_' + token
        return token


class DOBJ(Rule):
    def __init__(self, verb, obj, negative):
        self.verb = verb
        self.obj = obj
        self.negative = negative

    def __repr__(self):
        return f'dobj({self.apply_negative(self.verb, self.negative)}, {self.obj})'

    def to_asp(self):
        return f'dobj({self.verb}, {self.obj})'


class SOBJ(Rule):
    def __init__(self, verb, sobj, negative):
        self.verb = verb
        self.sobj = sobj
        self.negative = negative

    def __repr__(self):
        return f'sobj({self.apply_negative(self.verb, self.negative)}, {self.sobj})'

    def to_asp(self):
        return f'sobj({self.verb}, {self.obj})'


class XCOMP(Rule):
    def __init__(self, root_verb, verb, negative):
        self.root_verb = root_verb
        self.verb = verb
        self.negative = negative

    def __repr__(self):
        return f'xcomp({self.apply_negative(self.root_verb, self.negative)}, {self.apply_negative(self.verb, self.negative)})'

    def to_asp(self):
        return ''


class ACOMP(Rule):
    def __init__(self, root_a, a, negative):
        self.root_a = root_a
        self.a = a
        self.negative = negative

    def __repr__(self):
        return f'acomp({self.apply_negative(self.root_a, self.negative)}, {self.a})'

    def to_asp(self):
        return ''


def extract_relations(tokens):
    relations = []
    for token in tokens:
        relation = extract_relation(token)
        if relation is not None:
            relations.append(relation)
    return relations


def extract_relation(token):
    print(token.text)
    print(token.dep_)
    print('=======================')
    head = token.head
    if token.dep_ == 'dobj':
        return DOBJ(head.lemma_, token.lemma_, is_negated(head))
    if token.dep_ == 'nsubj':
        return SOBJ(head.lemma_, token.lemma_, is_negated(head))
    if token.dep_ == 'xcomp':
        return XCOMP(head.lemma_, token.lemma_, is_negated(head))
    if token.dep_ == 'acomp':
        return ACOMP(head.lemma_, token.lemma_, is_negated(head))


def is_negated(token):
    for child in token.children:
        if child.dep_ == 'neg':
            return True
    return False


doc = nlp(sentence)
relations = extract_relations(doc)

html = displacy.render(doc, style='dep')
with open('parse.html', 'w') as f:
    f.write(html)
for relation in relations:
    print(relation)
