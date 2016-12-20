'''Cross validation'''
from __future__ import print_function

from time import time

from data_partitioner import DataPartitioner
from utils import benchmark

class CrossValidate:

    def __init__(self, X, y, clf, cv=3):
        self.X = X
        self.y = y
        self.cv = cv
        self.clf = clf
        self.partitioner = DataPartitioner(cv, X, y)

    def cross_validate(self):
        '''Trains and tests the given classifier on cv folds, and returns the average accuracy'''
        sum_accuracy = 0.0
        for i, (X_train, y_train, X_test, y_test) in enumerate(self.partitioner.getPartitions()):
            print('Cross validation iteration: %d' % i)
            accuracy = benchmark(self.clf, X_train, y_train, X_test, y_test)
            sum_accuracy += accuracy
            print('Accuracy of partition %d: %f' % (i, accuracy))
            print()
        avg_acc = sum_accuracy / self.cv
        print('Average accuracy: %f' % avg_acc)
        print()
        return avg_acc
