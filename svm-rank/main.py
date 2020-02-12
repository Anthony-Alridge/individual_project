import jsonlines
import spacy
import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from embedding import WordEmbeddingFeature
from utils import WSCProblem
import argparse


def load_file(filename, model):
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
                line[ANSWER],
                model)
                )
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
        labels.append(problem.label())
        features = np.array([])
        for processor in processors:
            f = processor.process(problem)
            features = np.append(features, f)
        data.append(features)
    return np.array(data), np.array(labels)


def main(train_filename, test_filename, data_dir):
    model = spacy.load('en_core_web_md')
    print('SPACY model loaded')
    # Prepare data
    train_data = load_file(data_dir + train_filename, model)
    test_data = load_file(data_dir + test_filename, model)
    bag_of_words = WordEmbeddingFeature(max_length_sentence(train_data))
    train, train_labels = apply_features(train_data, [bag_of_words])
    test, test_labels = apply_features(test_data, [bag_of_words])
    print(
        f'Training shape is {train.shape} and labels is {train_labels.shape}')
    print(f'Testing shape is {test.shape} and labels is {test_labels.shape}')
    # Train classifier
    svc = svm.SVC()
    Cs = [2**k for k in range(-2, 2)]
    params = {'C': Cs}
    clf = GridSearchCV(svc, params)
    model = clf.fit(train, train_labels)
    # Evaluate model.
    test_accuracy = model.score(test, test_labels)
    train_accuracy = model.score(train, train_labels)
    print(f'Parameters used are {model.best_params_}')
    print('Scores:')
    print(f'Accuracy on test set:  {test_accuracy}')
    print(f'Accuracy on train set: {train_accuracy}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train an SVM model for WSC.')
    parser.add_argument(
        '--train',
        default='pdp.jsonl',
        help='The name of the input file for training')
    parser.add_argument(
        '--test',
        default='pdp-test.jsonl',
        help='The name of the input file for evaluation data.')
    parser.add_argument(
        '--data_dir',
        default='../data/',
        help='The path to the data directory.')
    args = parser.parse_args()
    main(args.train, args.test, args.data_dir)
