'''Word net extension'''

from __future__ import print_function

import sys

from nltk.corpus import wordnet as wn
import numpy as np

# The idea is to extend a feature set by synonyms of all the words in the set

# The methodology below assumes most frequent word sense, but we saw in assignment 3 that this is pretty accurate.
# It effectively adds meaning of words to the classification process

# another idea is, rather than each feature be the count of a word, we have the feature
# represent the count of the synset for that word. This will reduce dimensionality of the feature vector,
# whilst also making it more likely for two documents of the same class to have the same non-zero features
# A good experiment in fact would be to simply use binary Synset features

# So, we need to
# 1. Get all the words in all the documents (can a count vectorizer give us this?)
#       Yes. vectorizer.get_feature_names()
# 2. Get all the synsets for all the words in all the docs and put them in a map of Synset to
#    integer (0 indexed) where the integer is the index of the synset in the feature vec
#    Note: if a word has no synset, then leave the word itself as a feature! (could be important, like a name)
# 3. For each doc, get a list of its words
#       Use vectorizer.build_analyzer() to do so
# 3. Create a feature vector for each doc:
# -    Create a list feat_vec of length(num features) entries, initialized to zero
# -    For each word in the doc, first see if the word is in the dictionary. If so, get the index.
#      Else, get the word's synset and use the dictionary to determine its corresponding index in the feature vec
# -    Increment feat_vec[index] by 1
# Once you've done this for all docs, you have X_train! Try out X_train without any PCA

class WordNetVectorizer(object):

    def __init__(self, count_vectorizer):
        self.vectorizer = count_vectorizer

    def get_word_net_feature_vecs(self, docs, ignore=False):
        '''docs should be a 1d array of strings, where each string is all the contents of  a document'''
        all_words = self.vectorizer.get_feature_names()
        print('%d different words in total' % len(all_words))
        print('Getting features (wordnet synset or word if no synset) for all words')
        self.feat_to_idx = self.get_features(all_words, ignore)
        print('Done getting features')
        # now, for each doc, we have to get a list of the doc's words
        print('Getting a list of the words in each document')
        self.analyzer = self.vectorizer.build_analyzer()
        docs_as_words = []
        for doc in docs:
            docs_as_words.append(self.analyzer(doc))
        print('Done getting words in each document')
        print('Converting document word lists into features vectors')
        X_train = self.get_feature_vectors(docs_as_words)
        print('Done generating feature vectors')
        return np.array(X_train)

    # need something that will vectorize docs_test in the same way
    def vec_test_docs(self, docs):
        docs_as_words = []
        for doc in docs:
            docs_as_words.append(self.analyzer(doc))
        X_test = self.get_feature_vectors(docs_as_words)
        return np.array(X_test)

    '''
    def get_feature_vec(self, words):
        vec = [0 for i in range(len(self.feat_to_idx))]
        for word in words:
            if word in self.feat_to_idx:
                idx = self.feat_to_idx[word]
            else:
                synset = wn.synsets(word)[0]
                idx = self.feat_to_idx[synset]
            vec[idx] = vec[idx] + 1
        return vec
    '''

    def get_feature_vec(self, words):
        vec = [0 for i in range(len(self.feat_to_idx))]
        for word in words:
            if word in self.feat_to_idx:
                idx = self.feat_to_idx[word]
                vec[idx] = vec[idx] + 1
            else:
                synsets = wn.synsets(word)
                if len(synsets) > 0:
                    synset = synsets[0]
                    if synset in self.feat_to_idx:
                        idx = self.feat_to_idx[synset]
                        vec[idx] = vec[idx] + 1
        return vec

    def get_feature_vectors(self, docs_as_words):
        vecs = []
        for words in docs_as_words:
                vecs.append(self.get_feature_vec(words))
        return vecs

    def get_features(self, words, ignore):
        feat_to_idx = {}
        idx = 0
        for word in words:
            synsets = wn.synsets(word)
            if len(synsets) == 0:
                if ignore:
                    continue
                else:
                    feat = word
            else:
                feat = synsets[0]
            if feat not in feat_to_idx:
                feat_to_idx[feat] = idx
                idx += 1
        return feat_to_idx
