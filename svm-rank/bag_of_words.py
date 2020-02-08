from collections import Counter
import numpy as np
import spacy
from utils import WSCProblem

def keep_word(word):
    return word.is_alpha


def unique_words(problems):
    return set([word.lemma_ for problem in problems for word in problem.tokens() if keep_word(word)])


def create_word2idx(vocab):
    return {word: idx for idx, word in enumerate(vocab)}


class BagOfWordsFeature():
    def __init__(self, corpus):
        self.vocab = list(unique_words(corpus))
        # Mapping from words to their index in the feature vector.
        self.word2idx = create_word2idx(self.vocab)

    def process(self, problem):
        features = np.zeros(len(self.vocab))
        words = [word.lemma_ for word in problem.tokens() if keep_word(word)]
        freqs = Counter(words)
        for word in freqs:
            # Skip unknown words.
            if word in self.word2idx:
                features[self.word2idx[word]] = freqs[word]
        return features
