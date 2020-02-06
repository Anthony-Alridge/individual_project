import jsonlines
import spacy
import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from features.lexical_features import LexicalFeature
from utils import WSCProblem

train_filename = '../data/pdp.jsonl'
test_filename = '../data/pdp-test.jsonl'

model = spacy.load('en_core_web_md')
print('SPACY model loaded')


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
                line[ANSWER]),
                model)
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


def apply_features(problems, processors):
    data = []
    labels = []
    for problem in problems:
        labels.append(problem.label)
        features = np.array([])
        for processor in processors:
            np.append(features, processor.process(problem))
        data.append(features)
    return np.array(data), np.array(labels)


train_data = load_file(train_filename)
test_data = load_file(test_filename)

lexical_features = LexicalFeature(train_data)
train, train_labels = apply_features(train_data, [lexical_features])
test, test_labels = apply_features(test_data, [lexical_features])

print(train.shape)
print(train_labels.shape)

svc = svm.SVC()
Cs = [2**k for k in range(-2, 2)]
params = {'C': Cs}
clf = GridSearchCV(svc, params)
model = clf.fit(train, train_labels)

print(model.best_params_)
test_accuracy = model.score(test, test_labels)
train_accuracy = model.score(train, train_labels)

print('Scores:')
print(f'Accuracy on test set:  {test_accuracy}')
print(f'Accuracy on train set: {train_accuracy}')
