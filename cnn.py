'''classify text documents using a CNN and word embeddings'''

from __future__ import print_function

import sys
import os
import numpy as np
np.random.seed(1337)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model

from preprocess import get_data

BASE_DIR = '.'
GLOVE_DIR = BASE_DIR + '/glove.6B/'
TEXT_DATA_DIR = BASE_DIR + '/20_newsgroup/'
# consider only the first MAX_SEQUENCE_LENGTH words in a document
MAX_SEQUENCE_LENGTH = 1000
# consider only the top MAX_NB_WORDS most frequently occuring words in the dataset
MAX_NB_WORDS = 20000
# the dimension of the word embeddings (gloVe embeddings consist of 100 dimensions)
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

def get_embeddings_index():
    '''returns an index that maps words to their gloVe embeddings'''
    embeddings_index = {}
    with open(os.path.join(GLOVE_DIR, 'glove.6B.' + str(EMBEDDING_DIM) + 'd.txt')) as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

def get_data_locally():
    '''returns '''
    texts = []  # list of text samples
    labels_index = {}  # dictionary mapping label name to numeric id, e.g. atheism to 3
    labels = []  # list of label ids
    for name in sorted(os.listdir(TEXT_DATA_DIR)):
        # name is a newsgroup label
        path = os.path.join(TEXT_DATA_DIR, name)
        if os.path.isdir(path):
            label_id = len(labels_index)
            labels_index[name] = label_id
            for fname in sorted(os.listdir(path)):
                if fname.isdigit():
                    fpath = os.path.join(path, fname)
                    if sys.version_info < (3,):
                        f = open(fpath)
                    else:
                        f = open(fpath, encoding='latin-1')
                    texts.append(f.read())
                    f.close()
                    labels.append(label_id)
    return texts, labels, labels_index

print('Indexing word vectors.')
embeddings_index = get_embeddings_index()
print('Collected %s word vectors.' % len(embeddings_index))
print()

print('Loading data')

'''
categories, data_train, data_test = get_data(True, False, True)
all_unproc_X_train, all_unproc_X_test = data_train.data, data_test.data
all_y_train, y_test = data_train.target, data_test.target
'''

texts, labels, labels_index = get_data_locally()

'''
print(all_unproc_X_train[0])
print()
print(type(all_unproc_X_train[0]))
# unicode
print('-' * 80)
print(texts[0])
print()
print(type(texts[0]))
# str
print()
sys.exit(0)
'''

print('Tokenizing data')
# TODO figure out how to do a bit of your own preprocessing before using this tokenizer
# remove stop words, lemmatize, remove punctuation, make lowercase
# i.e. see if you can do this yourself first, and then pass the resulting tokenized text to the tokenizer
# SOLUTION: pretokenize the text yourself, but then string it back together and pass it to the tokenizer, which will simply split using " "
# Once this is done,
#   use your own data loading (can convert from unicode to str())
#   make sure training, validation, and testing sets are correct
#       take 20% of training data as validation data
#   add final line to test model on test set!
#   run the model!

tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
print()

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
print()

print('Padding sequences')
# pads the start of the sequences with zero, up to a length of 1000
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
print()

# convert labels into one-hot vectors
labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
# some numpy shuffling magic
indices = np.arange(data.shape[0]) # [0, ..., number of documents in the dataset-1]
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]

print('Preparing embedding matrix.')

# prepare embedding matrix
nb_words = min(MAX_NB_WORDS, len(word_index))
# create a matrix with number of words + 1 (why +1??) rows an EMBEDDING_DIM columns
# i.e. each row in the embedding matrix represents an embedding
embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))
# in fact, the ith row in the embedding matrix contains the embedding for the ith most common word in the documents
# this is why we use nb_words + 1 rows - the first row in the embedding matrix is all zeros
for word, i in word_index.items():
    if i <= 10:
        print('word: %s, i: %d' % (word, i))
    if i > MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(nb_words + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

print('Training model.')
# train a 1D convnet with global maxpooling
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(35)(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(len(labels_index), activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

model.fit(x_train, y_train, validation_data=(x_val, y_val),
          nb_epoch=2, batch_size=128)


## END
