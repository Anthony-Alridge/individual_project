import jsonlines
import re
import spacy
from spacy import displacy
import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV

train_filename = '../data/train_xs.jsonl'
test_filename = '../data/dev.jsonl'

model = spacy.load('en_core_web_md')
print('SPACY model loaded')


class Feature:

    def process(problem):
        pass


class LexicalFeature(Feature):
    def __init__(self, all_problems):
        pronoun = 'MASK_PRONOUN'
        mask = re.compile('_')
        unigrams = set()
        bigrams = set()
        trigrams = set()
        # antecedent_pairs = set()
        for problem in all_problems:
            sentence = mask.sub(pronoun, problem.sentence)
            tokens = model(sentence)
            ugrams, bgrams, tgrams = self.get_grams(tokens)
            # c1_pairs, c2_pairs = self.antecedent_pairs(tokens, problem.candidate_1, problem.candidate_2, pronoun)
            unigrams.update(ugrams)
            bigrams.update(bgrams)
            trigrams.update(tgrams)
            # antecedent_pairs.update(c1_pairs, c2_pairs)
        self.vocab = {word: i for i, word in enumerate(unigrams)}
        self.bigram_locs = {bigram: i for i, bigram in enumerate(bigrams)}
        self.trigram_locs = {trigram: i for i, trigram in enumerate(trigrams)}
        # self.antecedent_pair_locs = {pair: i for i, pair in enumerate(pairs)}

    def unigram(self, tokens):
        result = np.zeros(len(self.vocab))
        for token in tokens:
            if token in self.vocab:
                result[self.vocab[token]] = 1
        return result

    def get_grams(self, tokens):
        all = []
        left = []
        right = []
        discourse_conns = 0
        conn = None
        for token in tokens:
            all.append(token.lemma_)
            if 'CON' in token.pos_:
                discourse_conns += 1
                conn = token.lemma_
            else:
                arr_to_fill = right if discourse_conns > 0 else left
                arr_to_fill.append(token.lemma_)
        if discourse_conns > 1:
            # Invalid sentence structure for bgram and tgram features
            bgrams, tgrams = [], []
        else:
            bgrams = [(l, r) for l in left for r in right]
            tgrams = [(l, conn, r) for (l, r) in bgrams]
        return all, bgrams, tgrams

    def antecedent_pairs(self, tokens, c1, c2, p):
        c1_verb = None
        c2_verb = None
        p_verb = None
        for token in tokens.noun_chunks:
            if c1 in token.text:
                c1_verb = token.root.head.lemma_
            elif c2 in token.text:
                c2_verb = token.root.head.lemma_
            elif token.text == p:
                p_verb = token.root.head.lemma_
        c1_pairs = [(c1, c1_verb), (c1, p_verb)]
        c2_pairs = [(c2, c2_verb), (c2, p_verb)]
        return c1_pairs, c2_pairs

    def bigram(self, pairs):
        result = np.zeros(len(self.bigram_locs))
        for pair in pairs:
            if pair in self.bigram_locs:
                result[self.bigram_locs[pair]] = 1
        return result

    def trigram(self, tris):
        result = np.zeros(len(self.trigram_locs))
        for tri in tris:
            if tri in self.trigram_locs:
                result[self.trigram_locs[tri]] = 1
        return result

    def antecedent(self, pairs):
        result = np.zeros(len(self.antecedent_pair_locs))
        for pair in pairs:
            if pair in self.antecedent_pair_locs:
                result[self.antecedent_pair_locs[pair]] = 1
        return result

    def process(self, problem):
        tokens = model(problem.sentence)
        us, bis, tris = self.get_grams(tokens)
        ugram_features = self.unigram(us)
        bgram_features = self.bigram(bis)
        tgram_features = self.trigram(tris)
        return ugram_features
        #return np.append(np.append(ugram_features, bgram_features), tgram_features)

class NarrativeChainsFeature(Feature):
    def __init__(self):
        with open('../data/narrative_chains.txt', 'r') as f:
            self.chains = f.readlines()

    # Returns the role the pronoun plays in matching event chains.
    # Returns None if no chains match, or if the detected role for the pronoun
    # is conflicting
    def get_role(self, event_tuples):
        role = None
        for c_event, p_event in event_tuples:
            for chain in self.chains:
                events = chain.split()
                # This chain matches the event_tuple
                if p_event in events:
                    if c_event + '-o' in events:
                        # pronoun plays object role
                        if role == 's': return None  # Clashing roles
                        role = 'o'
                    if c_event + '-s' in events:
                        if role == 'o': return None  # Clashing roles
                        role = 's'
        return role

    '''
    Given a WSCProblem returns a (1, 1) vector
    which has value (1, -1). The value represents
    a prediction for the answer based on narrative chains.
    '''
    def process(self, problem, debug = False):
        pronoun = 'MASK_PRONOUN'
        mask = re.compile('_')
        sentence = mask.sub(pronoun, problem.sentence)
        tokens = model(sentence)
        c1 = problem.candidate_1
        c2 = problem.candidate_2
        c1_events = []
        c1_role = ''
        c2_events = []
        c2_role = ''
        pronoun_events = []
        pronoun_role = ''
        html = displacy.render(tokens, style='dep')
        with open('parse.html', 'w') as f:
            f.write(html)
        for token in tokens.noun_chunks:
            if token.text == c1:
                c1_events = [token.root.head.lemma_]
                c1_role = token.root.dep_
            elif token.text == c2:
                c2_events = [token.root.head.lemma_]
                c2_role = token.root.dep_
            elif token.text == 'MASK_PRONOUN':
                pronoun_events = [token.root.head.lemma_]
                pronoun_role = token.root.dep_

        for token in tokens:
            if token.dep_ == 'xcomp':
                if token.head.lemma_ in pronoun_events:
                    pronoun_events.append(token.lemma_)
                elif token.head.lemma_ in c1_events:
                    c1_events.append(token.lemma_)
                elif token.head.lemma_ in c2_events:
                    c2_events.append(token.lemma_)

        # we've got the events and roles. now form the event tuples
        event_tuples = []
        if debug:
            print(f'Role of C1: {c1_role}')
            print(f'Role of C2: {c2_role}')
            print(f'Role of Pronoun: {pronoun_role}')
            print(f'Pronoun events: {str(pronoun_events)}')
            print(f'C1 events: {str(c1_events)}')
            print(f'C2 events: {str(c2_events)}')

        def role_to_string(role):
            if role == 'nsubj':
                return 's'
            return 'o'
        for p_event in pronoun_events:
            p = p_event + '-' + role_to_string(pronoun_role)
            for event in set(c1_events + c2_events):
                event_tuples.append((event, p))
        if debug:
            print(event_tuples)
        role = self.get_role(event_tuples)
        if role is None:
            return 0  # Undecided
        if role == role_to_string(c1_role):
            return 1
        if role == role_to_string(c2_role):
            return -1
        return 0  # Undecided (failed to identify role for c1 or c2)


class WSCProblem:
    def __init__(self, sentence, candidate_1, candidate_2, answer):
        self.sentence = sentence
        self.candidate_1 = candidate_1
        self.candidate_2 = candidate_2
        self.answer = int(answer)

    def __repr__(self):
        return f'{self.sentence} \n CANDIDATE_1: {self.candidate_1} \n' \
            + f'CANDIDATE_2: {self.candidate_2} \n ANSWER: {self.answer} \n'

    def max_length(self):
        return len(self.sentence.split()) \
            + max(len(self.candidate_1.split()), len(self.candidate_2.split()))

    def to_svm_rank_feature(self):
        mask = re.compile('_')
        candidate_1_sent = mask.sub(self.candidate_1, self.sentence)
        candidate_2_sent = mask.sub(self.candidate_2, self.sentence)
        c2 = self._word2vecfeature(candidate_2_sent)
        c1 = self._word2vecfeature(candidate_1_sent)
        if self.answer == 1:
            label = 1
        else:
            label = -1
        return (c1 - c2, label)

    def _word2vecfeature(self, sentence):
        vec = np.array(model(sentence).vector)
        return vec


def load_file(filename):
    data = []
    SENTENCE = 'sentence'
    CANDIDATE_1 = 'option1'
    CANDIDATE_2 = 'option2'
    ANSWER = 'answer'
    with jsonlines.open(filename) as reader:
        for line in reader:
            data.append(WSCProblem(
                line[SENTENCE],
                line[CANDIDATE_1],
                line[CANDIDATE_2],
                line[ANSWER]))
    return data


def max_length_sentence(problems):
    max_length = 0
    for datum in problems:
        max_length = max(max_length, datum.max_length())
    return max_length


def apply_word2vec_features(problems):
    # A list of [(sample, label), ... ]
    train_and_labels = \
        [problem.to_svm_rank_feature() for problem in problems]
    # Unpack the tuples into the training set and labels
    train, labels = [np.array(list(l)) for l in zip(*train_and_labels)]
    return train, labels


def apply_nc_features(problems, debug=False):
    _, labels = apply_word2vec_features(problems)
    nf = NarrativeChainsFeature()
    train = []
    for problem in problems:
        features = nf.process(problem, debug)
        train.append(features)
    train = np.array(train)
    return train.reshape(-1, 1), labels

def apply_lexical_features(corpus, problems):
    _, labels = apply_word2vec_features(problems)
    lf = LexicalFeature(corpus)
    train = []
    for problem in problems:
        features = lf.process(problem)
        train.append(features)
    train = np.array(train)
    return train, labels

train_data = load_file(train_filename)
test_data = load_file(test_filename)

train, train_labels = apply_lexical_features(train_data, train_data)
test, test_labels = apply_lexical_features(train_data, test_data)

print(train.shape)
print(train_labels.shape)

svc = svm.SVC()
Cs = [2**k for k in range(-5, 5)]
params = {'kernel': ('linear', 'rbf'), 'C': Cs}
clf = GridSearchCV(svc, params)
model = clf.fit(train, train_labels)

print(model.best_params_)
test_accuracy = model.score(test, test_labels)
#train_accuracy = model.score(train, train_labels)

print('Scores:')
print(f'Accuracy on test set:  {test_accuracy}')
#print(f'Accuracy on train set: {train_accuracy}')
