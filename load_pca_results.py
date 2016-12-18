
from __future__ import print_function

from preprocess import load_pickle

all_results = load_pickle('all_results.pkl')

print('num_feats\tnaive_bayes\t')
for num_feats, results in all_results:
    reordered_accuracies = [0. for i in range(3)]
    for name, accuracy in results:
        if name == 'Logistic Regression':
            reordered_accuracies[1] = accuracy
        elif name == 'Linear SVM':
            reordered_accuracies[2] = accuracy
        elif name == 'Gaussian Naive Bayes':
            reordered_accuracies[0] = accuracy
    print('%d\t%f\t%f\t%f' % (num_feats, reordered_accuracies[0], reordered_accuracies[1], reordered_accuracies[2]))
