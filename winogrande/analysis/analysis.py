import matplotlib as plt
import jsonlines

predicitions = []
num_correct = 0
total = 0
correct = []
incorrect = []

with open('predictions_dev.lst', 'r') as p:
    txt = p.read()
    predicitions = [pred for pred in txt.split('\n') if pred != '']

with jsonlines.open('dev.jsonl') as reader:
    i = 0
    for obj in reader:
        total += 1
        s = obj['sentence']
        if obj['answer'] == predicitions[i]:
            num_correct += 1
            correct.append(s)
        else:
            incorrect.append(s)

print(float(num_correct) / float(total))


def convert_to_embedding_mean(sentences):
    # Embedding mean using Bert.
    return None

def parse_tree_width(sentences):
    return None

def parse_tree_height(sentences):
    return None

def num_words(sentences):
    return None

def num_verbs(sentences):
    return None

def num_adjectives(sentences):
    return None

def fluency_score(sentences):
    return None

def unknown_words(sentences):
    return None

# Is it possible to identify recognised scenarios

def scatter_plot(correct, incorrect):
    return None
