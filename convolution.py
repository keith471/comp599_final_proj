'''Uses a convolution neural network to classify documents using word2vec features'''
from __future__ import print_function
from __future__ import with_statement

from sknn.mlp import Classifier, Convolution, Layer

import sys
import logging
from time import time
import string

from optparse import OptionParser

from gensim.models.word2vec import Word2Vec

import numpy as np

import nltk.data
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics

# my modules
from preprocess import processData
from postprocess import writeResults

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# parse commandline arguments
op = OptionParser()
op.add_option("--lowercase",
              action="store_true",
              help="If set, the documents will be converted to lowercase.")
op.add_option("--lemmatize",
              action="store_true",
              help="If set, all words will be lemmatized.")
op.add_option("--remove_stop_words",
              action="store_true",
              help="If set, sklearn's list of English stop words will be removed.")
op.add_option("--report",
              action="store_true", dest="print_report",
              help="Print a detailed classification report.")
op.add_option("--confusion_matrix",
              action="store_true", dest="print_cm",
              help="Print the confusion matrix.")
op.add_option("--persist",
              action="store_true",
              help="If set, predictions for the test data will be persisted to disk")
op.add_option("--test",
              action="store", type="float", dest="test_fraction",
              help="Run on a fraction of the entire training corpus")

(opts, args) = op.parse_args()
if len(args) > 0:
    op.error("This script takes no arguments.")
    sys.exit(1)

if opts.test_fraction > 1.0 or opts.test_fraction < 0.0:
    op.error("The test fraction must be between 0.0 and 1.0")
    sys.exit(1)

print(__doc__)
op.print_help()
print()

def loadVectors():
    print("loading word2vec vectors...")
    t0 = time()
    model = Word2Vec.load_word2vec_format('/Volumes/Seagate Backup Plus Drive/GoogleNews-vectors-negative300.bin', binary = True)
    loadTime = time() - t0
    print("word2vec vectors loaded in %0.3f seconds" % loadTime)
    print()

    # done "training" the model; we can do the following to trim uneeded memory
    t0 = time()
    print("trimming model memory...")
    model.init_sims(replace=True)
    trimTime = time() - t0
    print("trimmed memory in %0.3f seconds" % trimTime)
    print()

    return model

def processDocument(doc, lowercase, lemmatize):
    ''' takes all text of a document and returns just the words (punctuation removed, other than apostrophes) '''
    # must remove punctuation as we have no word2vec vectors for them
    nopunc = doc.translate(None, string.punctuation)
    tokens =  word_tokenize(nopunc)
    if lowercase:
        tokens = [w.lower() for w in tokens]
    if lemmatize:
        wnl = WordNetLemmatizer()
        tokens = [wnl.lemmatize(w) for w in tokens]
    return tokens

def vectorize(X_train, model):
    '''computes an array of word2vec feature vectors from an array of documents'''
    X = []
    maxDocLength = 0
    for doc in X_train:
        features = []
        tokens = processDocument(doc, opts.lowercase, opts.lemmatize)
        wordCount = 0
        for token in tokens:
            if token in model:
                features.append(model[token])
                wordCount += 1
        X.append(features)
        if len(features) > maxDocLength:
            maxDocLength = len(features)

    return X, maxDocLength

def extend(X, maxDocLength):
    '''Exend any feature vectors shorter than the longest vector by zero vectors'''
    zeros = [0.0 for x in range(0, model.vector_size)]
    for vector in X:
        while len(vector) < maxDocLength:
            vector.append(zeros)

def benchmark(clf, X_train, y_train, X_test, y_test):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)
    print()

    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)
    print()

    # get the accuracy of the predictions against the train data
    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)
    print()

    if opts.print_report:
        print("classification report:")
        print(metrics.classification_report(y_test, pred, target_names=categories))

    # print a confusion matrix
    if opts.print_cm:
        print("confusion matrix:")
        print(metrics.confusion_matrix(y_test, pred))

    print()
    clf_descr = str(clf).split('(')[0]
    return clf_descr, pred, score

def predict(clf, X_train, y_train, X_test):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)
    print()

    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)
    print()

    clf_descr = str(clf).split('(')[0]
    return clf_descr, pred

if __name__ == "__main__":

    model = loadVectors()

    # define the categories
    categories = [
        'stats',
        'math',
        'physics',
        'cs'
    ]

    print("Processing data...")
    abstractsTrain, y_train, abstractsTest = processData()
    if opts.test_fraction:
        percent = (opts.test_fraction * 100.0)
        print("Using only %.f percent of the training data" % percent)
        threshold = int(opts.test_fraction * len(abstractsTrain))
        if threshold == 0:
            print("Fraction too small, please choose a larger fraction")
            print()
            sys.exit(1)
        abstractsTrain = abstractsTrain[:threshold]
        y_train = y_train[:threshold]
    print("Train set size: %d documents" % len(abstractsTrain))
    print("Test set size: %d documents" % len(abstractsTest))
    print("done")
    print()

    print("Extracting development set from training set")
    X_train, X_test, y_train, y_test = train_test_split(abstractsTrain, y_train, test_size=0.3, random_state=0)
    print("Using %d training examples and %d testing examples" % (len(X_train), len(X_test)))
    print("done")
    print()

    # X is our feature vectors
    print("Constructing feature vectors for training and testing data")
    X_train, maxDocLength_train = vectorize(X_train, model)
    X_test, maxDocLength_test = vectorize(X_test, model)
    print("done")
    print()

    maxDocLength = max(maxDocLength_train, maxDocLength_test)
    print("Largest feature vector consists of %d word2vec vectors" % maxDocLength)
    print("Extending other feature vectors to this length for both train and test data")
    extend(X_train, maxDocLength)
    extend(X_test, maxDocLength)
    print("done")
    print()

    clf = Classifier(
        layers=[
            Convolution('Rectifier', channels=12, kernel_shape=(3, 3), border_mode='full'),
            Convolution('Rectifier', channels=8, kernel_shape=(3, 3), border_mode='valid'),
            Layer('Rectifier', units=64),
            Layer('Softmax')],
        learning_rate=0.002,
        valid_size=0.2,
        n_stable=10,
        verbose=True)

    # convert data to ndarrays for training
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    clfDesc, pred, score = benchmark(clf, X_train, y_train, X_test, y_test)
    print('=' * 80)
    print("Summary:")
    print('_' * 80)
    print("Accuracy of %s using word2vec: %f" % (clfDesc, score))
    print()

    # see if we need to retrain and make predictions for the test data
    if opts.persist:
        print("Retraining on all train data and writing predictions for test data to disk")
        X_train, maxDocLength_train = vectorize(abstractsTrain, model)
        X_test, maxDocLength_test = vectorize(abstractsTest, model)
        maxDocLength = max(maxDocLength_train, maxDocLength_test)
        extend(X_train, maxDocLength)
        extend(X_test, maxDocLength)
        X_train = np.array(X_train)
        print("Shape of X_train:")
        print(X_train.shape)
        X_test = np.array(X_test)
        print("Shape of X_test:")
        print(X_test.shape)
        print()
        clfDesc, pred = predict(clf, X_train, y_train, X_test)
        writeResults(clfDesc, pred)
        print()
