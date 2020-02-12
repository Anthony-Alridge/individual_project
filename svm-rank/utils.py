import re
import numpy as np


class WSCProblem:
    def __init__(self, sentence, candidate_1, candidate_2, answer, model):
        self.sentence = sentence
        self.candidate_1 = candidate_1
        self.candidate_2 = candidate_2
        self.answer = int(answer)
        self.model = model

    def __repr__(self):
        return f'{self.sentence} \n CANDIDATE_1: {self.candidate_1} \n' \
            + f'CANDIDATE_2: {self.candidate_2} \n ANSWER: {self.answer} \n'

    def max_length(self):
        return len(self.sentence) \
            + max(len(self.candidate_1), len(self.candidate_2))

    def label(self):
        return 1 if self.answer == 1 else -1

    def tokens(self):
        return self.model(self.sentence)

    def tokens_without_candidates(self):
        candidate_symbol = 'CANDIDATE'
        mask = re.compile(self.candidate_1 + '|' + self.candidate_2)
        spaces = re.compile(' +')
        # Keep dummy token so that sentence is parsed correctly.
        sentence = mask.sub('CANDIDATE', self.sentence)
        # Previous transform can result in multiple spaces. Remove them.
        sentence = spaces.sub(' ', sentence)
        tokens = self.model(sentence)
        return [token for token in tokens if token.text != candidate_symbol]

    def _word2vecfeature(self, sentence):
        vec = np.array(self.model(sentence).vector)
        return vec
