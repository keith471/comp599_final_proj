'''Common functions pertaining to classification'''
from __future__ import print_function

from time import time
from sklearn import metrics

def benchmark(clf, X_train, y_train, X_test, y_test, verbose=True):
    if verbose:
        print('-' * 40)
        print('Benchmarking %s' % str(clf).split('(')[0])
        print('_' * 40)
        print('Training')
    t0 = time()
    clf.fit(X_train, y_train)
    dur = time() - t0
    if verbose:
        print('Finished training in %fs' % dur)
        print('Making predictions')
    t0 = time()
    pred = clf.predict(X_test)
    dur = time() - t0
    if verbose:
        print('Finished making predictions in %fs' % dur)
    acc = metrics.accuracy_score(y_test, pred)
    if verbose:
        print('Accuracy: %f' % acc)
        print()
    return acc
