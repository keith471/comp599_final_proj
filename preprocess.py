'''preprocessing'''
from __future__ import print_function

import csv
import sys
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from time import time
import string
import cPickle as pickle

from gensim.models.word2vec import Word2Vec

from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

from wordnet import WordNetVectorizer

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
    def __init__(self, nopunc):
        self.wnl = WordNetLemmatizer()
        self.nopunc = nopunc

    def __call__(self, doc):
        if self.nopunc:
            doc = ''.join([i if i not in string.punctuation else '' for i in doc])
        return [self.wnl.lemmatize(word) for word in word_tokenize(doc)]

class StemTokenizer(object):
    def __init__(self, nopunc):
        self.stemmer = PorterStemmer()
        self.nopunc = nopunc

    def __call__(self, doc):
        if self.nopunc:
            doc = ''.join([i if i not in string.punctuation else '' for i in doc])
        return [self.stemmer.stem(word) for word in word_tokenize(doc)]

def get_X_train(data, wn=False, max_n_gram=1, lowercase=True, nopunc=False, lemmatize=False, stem=False, remove_stop_words=True, tfidf=False, verbose=True):

    if verbose:
        print('Using n-grams of up to %d words in length' % max_n_gram)

    if lowercase and verbose:
        print('Converting all text to lowercase')

    if lemmatize:
        tokenizer = LemmaTokenizer(nopunc)
        if verbose:
            print('Lemmatizing all words')
    elif stem:
        tokenizer = StemTokenizer(nopunc)
        if verbose:
            print('Stemming all words')
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
        if wn:
            print('Learning a vocabulary dictionary with a count vectorizer')
            vectorizer.fit(data)
            print('Done learning vocabulary dictionary')
            vectorizer = WordNetVectorizer(vectorizer)
            print('Getting wordnet based feature vectors...')
            X_train = vectorizer.get_word_net_feature_vecs(data)
            print('Done getting wordnet based feature vectors')
        else:
            X_train = vectorizer.fit_transform(data)
    duration = time() - t0
    if verbose:
        data_train_size_mb = size_mb(data)
        print('done in %fs at %0.3fMB/s' % (duration, data_train_size_mb / duration))
        print('n_samples: %d, n_features: %d' % X_train.shape)
        print()
    return X_train, vectorizer

def get_X_test(data, vectorizer, wn=False, verbose=True):
    if verbose:
        print('Extracting features from the test data using the same vectorizer')
    t0 = time()
    if wn:
        X_test = vectorizer.vec_test_docs(data)
    else:
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

# we need to write a function that will prepare feature vectors to be fed to the conv net
# it will work as follows:
# for each document, parse the words in the document, in order
#   create an empty feature vector
#   for each word in the document,

################################################################################
# word2vec preprocessing
################################################################################

def load_word2vec_vectors():
    print("loading word2vec vectors...")
    t0 = time()
    model = Word2Vec.load_word2vec_format('/Volumes/Seagate Backup Plus Drive/MacFilesThatICantFit/GoogleNews-vectors-negative300.bin', binary = True)
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

    vec = model['hello']

    print('type of vector')
    print(type(vec))
    print('vector')
    print(vec)

    sys.exit(1)

    return model

def process_document(doc, lowercase, remove_stop_words, lemmatize, stem):
    ''' takes all text of a document and returns just the words (punctuation removed, other than apostrophes) '''
    # must remove punctuation as we have no word2vec vectors for them
    nopunc = doc.translate(None, string.punctuation)
    tokens =  word_tokenize(nopunc)
    if lowercase:
        tokens = [w.lower() for w in tokens]
    if remove_stop_words:
        tokens = [w for w in tokens if w not in stopwords.words('english')]
    if lemmatize:
        wnl = WordNetLemmatizer()
        tokens = [wnl.lemmatize(w) for w in tokens]
    if stem:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(w) for w in tokens]
    return tokens

def word2vec_vectorize(docs, model):
    '''computes an array of word2vec-based feature vectors from an array of unprocessed documents'''
    X = []
    max_feat_vec_length = 0
    for doc in docs:
        features = []
        tokens = process_document(doc, opts.lowercase, opts.lemmatize)
        for token in tokens:
            if token in model:
                features += model[token].tolist()
        X.append(features)
        if len(features) > max_feat_vec_length:
            max_feat_vec_length = len(features)
    return X, max_feat_vec_length

def extend(X, max_feat_vec_length):
    '''Exend any feature vectors shorter than the longest vector by zero vectors'''
    # for each feature vector, extend it to max_feat_vec_length
    print('Longest feature vector: %d features' % max_feat_vec_length)
    zeros = [0.0 for x in range(0, model.vector_size)]
    for feat_vec in X:
        print('Old feature vector length: %d' % len(feat_vec))
        while len(feat_vec) < max_feat_vec_length:
            feat_vec.append(zeros)
        print('New feature vector length: %d' % len(feat_vec))
        if len(feat_vec) != max_feat_vec_length:
            print('Error extending feature vectors')
            sys.exit(1)

def load_pickle(name):
    name = 'postprocessed_data/' + name
    with open(name, 'rb') as f:
        p = pickle.load(f)
    return p
