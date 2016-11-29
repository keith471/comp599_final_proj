'''preprocessing'''
from __future__ import print_function

import csv
import sys
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from time import time

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

def read_file(fname):
    ''' returns an array containing the data from the file '''
    data = []
    with open(fname, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        # skip the header
        reader.next()
        for row in reader:
            # row[0] contains the id, row [1] contains the data
            data.append(row[1])
    return data

def read_data():
    ''' reads all train and test data and returns as three arrays '''
    abstracts_train = read_file('../datasets/train_in.csv')
    y = read_file('../datasets/train_out.csv')
    abstracts_test = read_file('../datasets/test_in.csv')
    return abstracts_train, y, abstracts_test

def process_data():
    # preprocessing to go here
    return read_data()

def size_mb(docs):
    return sum(len(s.encode('utf-8')) for s in docs) / 1e6

def get_data(all_categories=True, filtered=False, verbose=True):
    '''gets the training and testing data'''
    if not all_categories:
        # use just a selection of categories
        categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']
    else:
        categories = None

    if filtered:
        remove = ('headers', 'footers', 'quotes')
    else:
        remove = ()

    if verbose:
        print('Loading 20 newsgroups dataset for categories:')
        print(categories if categories else 'all')
        print()

    data_train = fetch_20newsgroups(subset='train', categories=categories,
                                    shuffle=True, random_state=42,
                                    remove=remove)

    data_test = fetch_20newsgroups(subset='test', categories=categories,
                                   shuffle=True, random_state=42,
                                   remove=remove)
    if verbose:
        print('data loaded')

    categories = data_train.target_names    # for case categories == None

    if verbose:
        data_train_size_mb = size_mb(data_train.data)
        data_test_size_mb = size_mb(data_test.data)
        print('%d documents - %0.3fMB (training set)' % (len(data_train.data), data_train_size_mb))
        print('%d documents - %0.3fMB (test set)' % (len(data_test.data), data_test_size_mb))
        print('%d categories' % len(categories))
        print()

    return categories, data_train, data_test

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(word) for word in word_tokenize(doc)]

def get_X_train(data, max_n_gram=1, lowercase=True, lemmatize=False, remove_stop_words=True, tfidf=False, verbose=True):

    if verbose:
        print('Using n-grams of up to %d words in length' % max_n_gram)

    if lowercase and verbose:
        print('Converting all text to lowercase')

    if lemmatize:
        tokenizer = LemmaTokenizer()
        if verbose:
            print('Lemmatizing all words')
    else:
        tokenizer = None

    if remove_stop_words:
        stop_words = 'english'
        if verbose:
            print('Removing English stop words')
    else:
        stop_words = None

    t0 = time()
    if tfidf:
        if verbose:
            print()
            print('Extracting features from the test data using a tfidf vectorizer')
        vectorizer = TfidfVectorizer(lowercase=lowercase, tokenizer=tokenizer, stop_words=stop_words, ngram_range=(1, max_n_gram))
        X_train = vectorizer.fit_transform(data)
    else:
        if verbose:
            print('Extracting features from the test data using a count vectorizer')
        vectorizer = CountVectorizer(lowercase=lowercase, tokenizer=tokenizer, stop_words=stop_words, ngram_range=(1, max_n_gram))
        X_train = vectorizer.fit_transform(data)
    duration = time() - t0
    if verbose:
        data_train_size_mb = size_mb(data)
        print('done in %fs at %0.3fMB/s' % (duration, data_train_size_mb / duration))
        print('n_samples: %d, n_features: %d' % X_train.shape)
        print()
    return X_train, vectorizer

def get_X_test(data, vectorizer, verbose=True):
    if verbose:
        print('Extracting features from the test data using the same vectorizer')
    t0 = time()
    X_test = vectorizer.transform(data)
    duration = time() - t0
    if verbose:
        data_test_size_mb = size_mb(data)
        print('done in %fs at %0.3fMB/s' % (duration, data_test_size_mb / duration))
        print('n_samples: %d, n_features: %d' % X_test.shape)
        print()
    return X_test

def get_frac(frac, all_X, all_y):
    percent = (frac * 100.0)
    print('Using only %.f percent of the training data' % percent)
    threshold = int(frac * len(all_X))
    if threshold == 0:
        print('Fraction too small, please choose a larger fraction')
        print()
        sys.exit(1)
    unproc_X_train = all_X[:threshold]
    y_train = all_y[:threshold]
    return unproc_X_train, y_train
