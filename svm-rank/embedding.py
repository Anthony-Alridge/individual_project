import transformers as ppb
import numpy as np


class WordEmbeddingFeature():
    def __init__(self, max_length):
        self.max_length = max_length + 10
        self.tokeniser = ppb.BertTokenizer.from_pretrained(
            'bert-base-cased')

    def process(self, problem):
        return np.array(self.tokeniser.encode(
            problem.sentence,
            truncation_strategy='do_not_truncate',
            pad_to_max_length=True,
            max_length=self.max_length))
