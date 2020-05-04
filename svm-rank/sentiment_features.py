from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from utils import WSCProblem
import spacy
import numpy as np


class SentimentFeature:
    def __init__(self, max_length):
        self.feature_length = max_length
        self.sentimentAnalyzer = SentimentIntensityAnalyzer()

    def word_level_sentiments(self, tokens):
        word_sentiments = np.zeros(self.feature_length)
        i = 0
        for token in tokens:
            scores = self.sentimentAnalyzer.polarity_scores(token.text)
            word_sentiments[i] = scores['compound']
            i += 1
        return word_sentiments

    def process(self, problem):
        return self.word_level_sentiments(problem.tokens_without_candidates())
