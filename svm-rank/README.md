# Evaluation Notes
Note that WSC273 and winogrande test is reserved as a test set.
pdp + pdp-test are easier than winogrande dataset.
winogrande dev set is used for evaluation when trained with winogrande data

## SVM

### Lexical Features (based on Rahman and Ng)


### Bag of Words

Simple Bag of Words with frequencies (evaluations):

On Pdp + pdp-test: 0.5798 accuracy.

On train-xs + dev: 0.4775 accuracy.
On train-s + dev:  0.5957 - check....
On train-m + dev: 0.5114

With stop words removed (based on spacy stop words):
On pdp + pdp-test: 0.5390
On train-xs + dev: 0.4941
On train-s + dev: 0.5004

Binary not frequencies:
pdp + pdp-test: 0.5993
train-xs + dev:
train-s + dev: 0.5107

With candidates removed:
pdp + pdp-test: 0.5851
train-xs + dev: 0.5043
train-s + dev: 0.5051


# other tests TODO
-- binary

### Word embeddings

Using mean vector:
