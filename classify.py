'''classification'''

from __future__ import print_function

import numpy as np
from argparse import ArgumentParser
import sys
from time import time
import logging

from sklearn.feature_selection import SelectKBest, chi2
#from sklearn.grid_search import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.decomposition import PCA

from preprocess import get_data, get_X_train, get_X_test, get_frac
from utils import benchmark
from cross_validation import CrossValidate
from postprocess import to_pickle

################################################################################
# Constants
################################################################################

NAIVE_BAYES = 'nb'
LOGISTIC_REGRESSION = 'lr'
LINEAR_SVM = 'lsvm'
VALID_LEARNERS = [NAIVE_BAYES, LOGISTIC_REGRESSION, LINEAR_SVM]

################################################################################
# Logging and command line arguments
################################################################################

# display progress logs on stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# parse commandline arguments
parser = ArgumentParser()
parser.add_argument('--all_categories',
                    action='store_true', default=False,
                    help='Whether to use all categories or not.')
parser.add_argument('--filter',
                    action='store_true', default=False,
                    help='Remove newsgroup information that is easily overfit: '
                    'headers, signatures, and quoting.')
parser.add_argument('--lowercase',
                    action='store_true', default=False,
                    help='If set, the documents will be converted to lowercase.')
parser.add_argument('--lemmatize',
                    action='store_true', default=False,
                    help='If set, all words will be lemmatized.')
parser.add_argument('--remove_stop_words',
                    action='store_true', default=False,
                    help='If set, sklearn\'s list of English stop words will be removed.')
parser.add_argument('--tfidf',
                    action='store_true', default=False,
                    help='If set, tf-idf term weighting will be used.')
parser.add_argument('--max_n_gram',
                    action='store', type=int, default=1,
                    help='The maximum n-gram size to be used.')
parser.add_argument('--verbose',
                    action='store_true',
                    help='Print lots of info to sdtout!')
parser.add_argument('--frac',
                    action='store', type=float,
                    help='A float between 0 and 1 indicating the fraction of training '
                    'data to actually train on')
parser.add_argument('--clf',
                    action='store', type=str,
                    help='The classifier to use if performing cross-validation')
parser.add_argument('--wn',
                    action='store_true',
                    help='If set, WordNet will be used to develop the feature vectors')
parser.add_argument('--chi2_select',
                    action='store', type=int,
                    help='Select some number of features using a chi-squared test')
parser.add_argument('--chi2_select_range',
                    action='store', type=int, nargs=3,
                    help='Three positive integers, start, end and range, specifying the '
                    'number of dimensions to select using a chi2 test. Cross-validation will '
                    'be used to select the best number of dimensions within the range')
parser.add_argument('--pca_select',
                    action='store', type=int,
                    help='Select some number of features using principal component analysis')
parser.add_argument('--pca_select_range',
                    action='store', type=int, nargs=3,
                    help='Same as --chi2_select_range except that PCA will be used instead of a chi2 test')
parser.add_argument('--confusion_matrix',
                    action='store_true',
                    help='Print the confusion matrix.')
parser.add_argument('--top10',
                    action='store_true',
                    help='Print ten most discriminative terms per class'
                    ' for every classifier.')
parser.add_argument('--report',
                    action='store_true',
                    help='Print a detailed classification report.')
parser.add_argument('--pca_mass_compute',
                    action='store_true',
                    help='use to get a bunch of PCA results overnight :-)')

args = parser.parse_args()

if args.max_n_gram < 1:
    parser.error('Max n-gram length must be positive')
    sys.exit()

if args.frac:
    if args.frac <= 0 or args.frac > 1:
        parser.error('`frac` flag must be in the range (0, 1]')
        sys.exit(1)

if args.chi2_select_range or args.pca_select_range and not args.clf:
    parser.error('Please specify a classifier - must use a classifier when cross-validating')
    sys.exit(1)

if args.clf:
    if args.clf not in VALID_LEARNERS:
        parser.error('`%s` is not a valid learner' % args.clf)
        sys.exit(1)

print(__doc__)
parser.print_help()
print()

################################################################################
# Helper definitions
################################################################################

def select_chi2(X_train, y_train, k):
    '''Select the k best features accoding to a chi-squared test'''
    print('Extracting %d best features by a chi-squared test' % k)
    t0 = time()
    # the SelectKBest object is essentially a vectorizer that will select only the most influential k features of your input vectors
    ch2 = SelectKBest(chi2, k=k)
    X_train = ch2.fit_transform(X_train, y_train)
    print('done in %fs' % (time() - t0))
    print('n_samples: %d, n_features: %d' % X_train.shape)
    print()
    return X_train, ch2

def select_pca(X_train, y_train, k):
    pca = PCA(n_components=k)
    # fit the PCA to X_train so that we get the same transformation for X_test later on
    pca.fit(X_train)
    # update X_train
    return pca.transform(X_train), pca

'''
def select_chi2_and_revec(X_train, y_train, X_test, k):
    Select k best features according to chi-squared test and revetorize X_test
    X_train, ch2 = select_chi2(X_train, y_train, k)
    X_test = ch2.transform(X_test)
    return X_train, X_test
'''

def select_k_and_revec(X_train, y_train, X_test, k, chi2):
    '''Select k best features according to either a chi-squared test or PCA and revetorize X_test'''
    if chi2:
        X_train, ch2 = select_chi2(X_train, y_train, k)
        X_test = ch2.transform(X_test)
    else:
        X_train, pca = select_pca(X_train, y_train, k)
        X_test = pca.transform(X_test)
    return X_train, X_test

'''
def get_feature_set_scores(X_train, y_train, clf, cv_range):
    Get accuracies for a range of numbers of features. For each number of features,
    we cross validate to determine the accuracy.
    Returns a an array of tuples: (# features used, avg prediction accuracy)
    start, end, step = cv_range
    rng = range(start, end+1, step)
    arr = []
    best_acc = 0.
    best_num_feats = 0
    for num_feats in rng:
        X_t, _ = select_chi2(X_train, y_train, num_feats)
        cross_validator = CrossValidate(X_t, y_train, clf)
        acc = cross_validator.cross_validate()
        if acc > best_acc:
            best_acc = acc
            best_num_feats = num_feats
        arr.append((num_feats, acc))
    return arr, (best_num_feats, best_acc)
'''

def get_feature_set_scores(X_train, y_train, clf, cv_range, chi2):
    '''Get accuracies for a range of numbers of features. For each number of features,
    we cross validate to determine the accuracy.
    Returns a an array of tuples: (# features used, avg prediction accuracy)'''
    start, end, step = cv_range
    rng = range(start, end+1, step)
    arr = []
    best_acc = 0.
    best_num_feats = 0
    for num_feats in rng:
        if chi2:
            X_t, _ = select_chi2(X_train, y_train, num_feats)
        else:
            X_t, _ = select_pca(X_train, y_train, num_feats)
        cross_validator = CrossValidate(X_t, y_train, clf)
        acc = cross_validator.cross_validate()
        if acc > best_acc:
            best_acc = acc
            best_num_feats = num_feats
        arr.append((num_feats, acc))
    return arr, (best_num_feats, best_acc)

def parse_clf(s):
    '''Instantiate and return the appropriate classifier based on s'''
    if s == NAIVE_BAYES:
        clf = GaussianNB()
    elif s == LOGISTIC_REGRESSION:
        clf = LogisticRegression()
    elif s == LINEAR_SVM:
        clf = LinearSVC()
    else:
        # default to LogisticRegression
        clf = LogisticRegression()
    return clf

def print_pairs(pairs, titles):
    '''Convenience function for printing'''
    print('%s\t%s' % titles)
    for (a, b) in pairs:
        print(str(a) + '\t' + str(b))

def get_results(X_train, y_train, X_test, y_test):

    results = []

    if args.clf:
        if args.clf == 'nb':
            learners = [(GaussianNB(), 'Gaussian Naive Bayes')]
        elif args.clf == 'lr':
            learners = [(LogisticRegression(), 'Logistic Regression')]
        else:
            learners = [(LinearSVC(), 'Linear SVM')]
    else:
        learners = [(LogisticRegression(), 'Logistic Regression'), (LinearSVC(), 'Linear SVM'), (GaussianNB(), 'Gaussian Naive Bayes')]
    for clf, name in learners:
        print('-' * 80)
        print(name)
        print('_' * 80)
        accuracy = benchmark(clf, X_train, y_train, X_test, y_test)
        results.append((name, accuracy))
    print_pairs(results, ('classifier', 'accuracy'))
    return results

################################################################################
# The meat and potatoes
################################################################################

if __name__ == '__main__':

    ############################################################################
    # Load and preprocess data
    ############################################################################

    categories, data_train, data_test = get_data(args.all_categories, args.filter, args.verbose)

    all_unproc_X_train, all_unproc_X_test = data_train.data, data_test.data
    all_y_train, y_test = data_train.target, data_test.target

    if args.frac:
        unproc_X_train, y_train = get_frac(args.frac, all_unproc_X_train, all_y_train)
    else:
        unproc_X_train = all_unproc_X_train
        y_train = all_y_train

    print('Final train and test set sizes')
    print('Train set size: %d documents' % len(unproc_X_train))
    print('Test set size: %d documents' % len(all_unproc_X_test))
    print()

    # turn the unprocessed training and testing data into feature vectors
    X_train, vectorizer = get_X_train(unproc_X_train, wn=args.wn, max_n_gram=args.max_n_gram, lowercase=args.lowercase, lemmatize=args.lemmatize, remove_stop_words=args.remove_stop_words, tfidf=args.tfidf)

    # use the same vectorizer to vectorize the test data
    X_test = get_X_test(all_unproc_X_test, vectorizer, wn=args.wn)

    print('Final dataset shapes')
    print('X_train:')
    print(X_train.shape)
    print('y_train')
    print(y_train.shape)
    print('X_test')
    print(X_test.shape)
    print('y_test')
    print(y_test.shape)
    print()

    ############################################################################
    # benchmark classifier(s)
    ############################################################################

    # either cross validate over a range of numbers of features, or determine the
    # performance for all or just some features (no cross validation)
    if args.chi2_select_range or args.pca_select_range:

        clf = parse_clf(args.clf)

        if args.chi2_select_range:
            accuracies, best = get_feature_set_scores(X_train, y_train, clf, args.chi2_select_range, True)
        else:
            accuracies, best = get_feature_set_scores(X_train, y_train, clf, args.pca_select_range, False)

        print('Summary of accuracies:')
        print_pairs(accuracies, ('# features', 'accuracy'))
        print()

        best_num_feats, best_acc = best

        print('Best number of features: %d' % best_num_feats)
        print('Best accuracy: %f' % best_acc)

        # benchmark against the test set for the best number of features
        if args.chi2_select_range:
            X_train, X_test = select_k_and_revec(X_train, y_train, X_test, best_num_feats, True)
        else:
            X_train, X_test = select_k_and_revec(X_train, y_train, X_test, best_num_feats, False)

        print('X_train shape')
        print(X_train.shape)
        print('X_test shape')
        print(X_test.shape)
        print()
        # necessary for GaussianNB to convert X_train and X_test from sparse to dense arrays
        X_train = X_train.toarray()
        X_test = X_test.toarray()
        accuracy = benchmark(clf, X_train, y_train, X_test, y_test)
        print('Accuracy: %0.3f' % accuracy)
        print()
    else:
        # since we are not cross validating, we might as well compare all the classifiers
        if args.pca_mass_compute:
            # select the first 70000 features using a chi-squared test, then subsequently select a varying amount
            # of features by PCA and run all three learners on the results and save their accuracies
            print('Selecting %d features using a chi-squared test' % args.chi2_select)
            X_train_chi2, X_test_chi2 = select_k_and_revec(X_train, y_train, X_test, args.chi2_select, True)
            X_train_chi2 = X_train_chi2.toarray()
            X_test_chi2 = X_test_chi2.toarray()
            print('Shape of X_train after chi-squared selection of features')
            print(X_train_chi2.shape)
            print('Shapte of X_test after chi-squared selection of features')
            print(X_test_chi2.shape)
            # the number of features to select with PCA
            all_num_feats = [300, 400, 500, 750, 1000, 1500, 2000, 5000, 10000, 15000, 20000, 30000, 40000, 50000]
            results = []
            for num_feats in all_num_feats:
                print('Selecting %d features using PCA' % num_feats)
                X_train, X_test = select_k_and_revec(X_train_chi2, y_train, X_test_chi2, num_feats, False)
                print('Shape of X_train')
                print(X_train.shape)
                print('Shapte of X_test')
                print(X_test.shape)
                curr_results = get_results(X_train, y_train, X_test, y_test)
                # pickle the current results for safety
                name = 'accuracies_' + str(num_feats) + '_feats'
                to_pickle(name, curr_results)
                results.append((num_feats, curr_results))

            # pickle all the results
            to_pickle('all_results', results)

            # print all the results
            for num_feats, res in results:
                print('-' * 40)
                print('Number of features: %d' % num_feats)
                print('_' * 40)
                print()
                print_pairs(res, ('classifier', 'accuracy'))
                print()
            print()
            sys.exit(0)
        elif args.chi2_select or args.pca_select:
            if args.chi2_select and not args.pca_select:
                print('Testing model using the top %d features, selected by a chi-squared test' % args.chi2_select)
                X_train, X_test = select_k_and_revec(X_train, y_train, X_test, args.chi2_select, True)
                # necessary for GaussianNB to convert X_train and X_test from sparse to dense arrays
                X_train = X_train.toarray()
                X_test = X_test.toarray()
            elif args.pca_select and not args.chi2_select:
                print('Testing model using the top %d features, selected by PCA' % args.pca_select)
                # PCA does not work on sparse input, so we have to convert X_train and X_test to dense arrays
                # Also works out for GaussianNB
                X_train = X_train.toarray()
                X_test = X_test.toarray()
                X_train, X_test = select_k_and_revec(X_train, y_train, X_test, args.pca_select, False)
            else:
                # First use chi-squared test to narrow down the range, then use PCA from there
                # Have to do it this way because PCA is very space (??) expensive to compute
                print('Testing model using the top %d features' % args.chi2_select)
                print('First, selecting the top %d features by a chi-squared test' % args.chi2_select)
                print('Then, from those, selecting the top %d features by PCA' % args.pca_select)
                X_train, X_test = select_k_and_revec(X_train, y_train, X_test, args.chi2_select, True)
                X_train = X_train.toarray()
                X_test = X_test.toarray()
                print('Shape of X_train after chi-squared selection of features')
                print(X_train.shape)
                print('Shapte of X_test after chi-squared selection of features')
                print(X_test.shape)
                X_train, X_test = select_k_and_revec(X_train, y_train, X_test, args.pca_select, False)
        else:
            print('Using all features')

        get_results(X_train, y_train, X_test, y_test)
