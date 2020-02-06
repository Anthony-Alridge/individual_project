import numpy as np
from spacy import symbols

from utils import WSCProblem
import spacy


# TODO: Documentation
# TODO: Write tests for basic features (get_grams)
def get_grams(problem):
    tokens = problem.tokens_without_candidates()
    def invalid_pair(l, r):
        return l.pos == symbols.NOUN and r.pos == symbols.ADJ \
            or l.pos == symbols.ADJ and r.pos == symbols.ADJ
    all = []
    left = []
    right = []
    discourse_conns = 0
    conn = None
    for token in tokens:
        if not token.is_alpha:
            continue
        all.append(token.lemma_)
        if 'CON' in token.pos_:
            print(token.text)
            print(token.pos_)
            discourse_conns += 1
            conn = token.lemma_
        else:
            arr_to_fill = right if discourse_conns > 0 else left
            arr_to_fill.append(token)
    if discourse_conns > 1:
        # Invalid sentence structure for bgram and tgram features
        bgrams, tgrams = [], []
    else:
        bgrams = \
            [(l.lemma_, r.lemma_) for l in left for r in right if not invalid_pair(l, r)]
        tgrams = [(l, conn, r) for (l, r) in bgrams]
    return all, bgrams, tgrams

class LexicalFeature():
    def __init__(self, all_problems):
        unigrams = set()
        bigrams = set()
        trigrams = set()
        for problem in all_problems:
            ugrams, bgrams, tgrams = self.get_grams(problem)
            unigrams.update(ugrams)
            bigrams.update(bgrams)
            trigrams.update(tgrams)
        self.vocab = {word: i for i, word in enumerate(unigrams)}
        self.bigram_locs = {bigram: i for i, bigram in enumerate(bigrams)}
        self.trigram_locs = {trigram: i for i, trigram in enumerate(trigrams)}

    def unigram(self, tokens):
        result = np.zeros(len(self.vocab))
        for token in tokens:
            if token in self.vocab:
                result[self.vocab[token]] = 1
        return result

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
        us, bis, tris = get_grams(problem)
        ugram_features = self.unigram(us)
        bgram_features = self.bigram(bis)
        tgram_features = self.trigram(tris)
        return np.append(np.append(ugram_features, bgram_features), tgram_features)

model = spacy.load('en_core_web_sm')
sentence = 'Rice beat Texas, even though they were the best team in the nation'
c1 = 'Rice'
c2 = 'Texas'
problem = WSCProblem(sentence, c1, c2, '1', model)
a, b, c = get_grams(problem)

print('unigrams: \n')
print(a)
print()

print('bgrams: \n')
print(b)
print()

print('tgrams: \n')
print(c)
print()
