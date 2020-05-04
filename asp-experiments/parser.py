import spacy
from spacy import displacy

nlp = spacy.load('en_core_web_md')
sentence = 'The delivery truck zoomed by the school bus because it is going so fast.'
class Program:

    def __init__(self):
        self.rules = []

    def append_rules(self, rules):
        self.rules.extend([rule.to_asp() for rule in rules])

    def write_to_file(self, filename='test.lp'):
        lines = self.rules + self.axioms()
        with open(filename, 'w') as f:
            f.writelines('\n'.join(lines))
        print(f'Program generated in {filename}.')

    def axioms(self):
        rules = []
        rules.append('-event(Event, Sub, Obj) :- nsubj(Event, Sub), dobj(Event, Obj), neg(Event, Sub), not goal_event(Event).')
        rules.append('event(Event, Sub, Obj) :- nsubj(Event, Sub), dobj(Event, Obj), not neg(Event, Sub), not goal_event(Event).')
        rules.append('property(Name, X) :- nsubj(Name, X).')
        rules.append('property(Name, X) :- nsubj(Z, X), acomp(Z, Name).')
        rules.append(':- not goal.')
        return rules

class LexicalRule:
    def __init__(self, rule_type, root, root_pos, child, is_root_negative):
        self.rule_type = rule_type
        self.root = root
        self.root_pos = root_pos
        self.child = child
        self.is_root_negative = is_root_negative

    def __repr__(self):
        verb = self.format_event(self.root, self.is_root_negative)
        obj = self.format_entity(self.child)
        return f'{self.rule_type}({verb}, {obj})'

    def to_asp(self):
        event = self.root
        entity = self.format_entity(self.child)
        rule = f'{self.rule_type}({event}, {entity}).'
        if self.is_root_negative:
            return rule + '\n' + f'neg({event}, {entity}).'
        return rule

    def format_event(self, token, negative):
        if negative:
            return 'not_' + token
        return token

    def format_entity(self, token):
        if token == '-PRON-':
            return 'pronoun'
        return token

class GoalRule:
    def __init__(self, event, is_event_negative):
        self.event = event
        self.is_event_negative = is_event_negative

    def to_asp(self):
        event_predicate = ('-' if self.is_event_negative else '') + 'event(Event, _, _)'
        goal_axiom =  f'goal :- goal_event(Event), {event_predicate}.'
        goal =  f'goal_event({self.event}).'
        return goal_axiom + '\n' + goal

class EntityRule:
    def __init__(self, entity):
        self.entity = entity

    def to_asp(self):
        return f'entity({self.entity}).'

def extract_relations(tokens):
    relations = []
    for token in tokens:
        relations.extend(extract_relation(token))
    return relations

def extract_relation(token):
    relations = []
    valid_deps = {'dobj', 'nsubj', 'xcomp', 'acomp'}
    head = token.head
    if token.dep_ in valid_deps:
        relations.append(LexicalRule(
            token.dep_,
            head.lemma_,
            head.pos_,
            token.lemma_,
            is_negated(head)))
        if token.lemma_ == 'man': # candidate
            relations.append(GoalRule(head.lemma_, is_negated(head)))
        if token.lemma_ in ['man', 'son']:
            relations.append(EntityRule(token.lemma_))
    return relations

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

program = Program()
program.append_rules(relations)
program.write_to_file()
