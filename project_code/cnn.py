'''classify text documents using a CNN and word embeddings'''

from __future__ import print_function

import sys
import os
import string

from argparse import ArgumentParser

import numpy as np
# set for reproducibility
np.random.seed(1337)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.models import model_from_json
from keras.callbacks import EarlyStopping

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from preprocess import get_data, get_frac
from postprocess import to_pickle

BASE_DIR = '.'
GLOVE_DIR = BASE_DIR + '/glove.6B/'
TEXT_DATA_DIR = BASE_DIR + '/20_newsgroup/'
# consider only the first MAX_SEQUENCE_LENGTH words in a document
#MAX_SEQUENCE_LENGTH = 1000
# consider only the top MAX_NB_WORDS most frequently occuring words in the dataset
#MAX_NB_WORDS = 20000
# the dimension of the word embeddings (gloVe embeddings consist of 100 dimensions)
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

parser = ArgumentParser()
parser.add_argument('--frac',
                    action='store', type=float,
                    help='A float between 0 and 1 indicating the fraction of training '
                    'data to actually train on')
args = parser.parse_args()

print(__doc__)
parser.print_help()
print()

# function to save model to disk
def save_model(model, name='model'):
    model_name = name + '_model.json'
    weights_name = name + '_weights.h5'
    print('saving model json to %s and model weights to %s' % (model_name, weights_name))
    model.save_weights(weights_name)
    with open(model_name, 'wb') as f:
        f.write(model.to_json())

# load existing model from disk
def load_model(name='model'):
    model_name = name + '_model.json'
    weights_name = name + '_weights.h5'
    with open(model_name, 'rb') as f:
        json_data = f.read()
    model = model_from_json(json_data)
    model.load_weights(weights_name)
    return model

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

def preproc(data):
    tokenized_texts = []
    wnl = WordNetLemmatizer()
    for doc in data:
        nopunc = ''.join([i if i not in string.punctuation else '' for i in doc])
        tokens =  word_tokenize(nopunc)
        tokens = [w.lower() for w in tokens]
        tokens = [w for w in tokens if w not in stopwords.words('english')]
        tokens = [wnl.lemmatize(w) for w in tokens]
        tokenized_texts.append(tokens)
    texts = []
    total_length = 0.
    max_length = 0
    for doc in tokenized_texts:
        total_length += len(doc)
        if len(doc) > max_length:
            max_length = len(doc)
        texts.append(' '.join([word.encode('ascii', 'replace') for word in doc]))
    return texts, total_length / len(texts), max_length

def preproc2(data):
    texts = [text.encode('ascii', 'replace') for text in data]
    return texts

def tokenize(texts, max_nb_words, max_sequence_length):
    '''converts preprocessed texts into a list where each entry corresponds to a text and
    each entry is a list where entry i contains the index of ith word in the text as indexed by word_index'''
    tokenizer = Tokenizer(nb_words=max_nb_words)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    print()

    print('Padding sequences')
    # pads the start of the sequences with zero, up to a length of 1000
    data = pad_sequences(sequences, maxlen=max_sequence_length)
    print()
    return data, word_index, tokenizer

def tokenize_test_texts(texts, tokenizer, max_sequence_length):
    sequences = tokenizer.texts_to_sequences(texts)
    data = pad_sequences(sequences, maxlen=max_sequence_length)
    return data

def get_avgs(loss_and_acc):
    loss_total = 0.
    acc_total = 0.
    count = 0
    for loss, acc in loss_and_acc:
        loss_total += loss
        acc_total += acc
        count += 1
    return loss_total / count, acc_total / count

def print_results(results):
    print('max nb words\tmax sequence length\tavg loss\tavg err\tmax sequence length\tavg loss\tavg err')
    for res in results:
        max_nb_words = res[0]
        print(max_nb_words, end='\t')
        rest = res[1:]
        for max_sequence_length, loss, acc in rest:
            print('%d\t%f\t%f' % (max_sequence_length, loss, acc), end='\t')
        print()
    print()

if __name__ == '__main__':

    print('Indexing word vectors.')
    embeddings_index = get_embeddings_index()
    print('Collected %s word vectors.' % len(embeddings_index))
    print()

    print('Loading data')

    categories, data_train, data_test = get_data(True, False, True)
    all_unproc_X_train, all_unproc_X_test = data_train.data, data_test.data
    all_y_train, y_test = data_train.target, data_test.target

    if args.frac:
        unproc_X_train, y_train = get_frac(args.frac, all_unproc_X_train, all_y_train)
        all_unproc_X_test, y_test = get_frac(args.frac, all_unproc_X_test, y_test)
    else:
        unproc_X_train = all_unproc_X_train
        y_train = all_y_train

    #texts, labels, labels_index = get_data_locally()

    print('Preprocessing training text')
    texts_train, avg_length, max_length = preproc(unproc_X_train)

    print('Average text lenght:', avg_length)
    print('Length of longest text:', max_length)

    # convert labels into one-hot vectors
    y_train = to_categorical(np.asarray(y_train))
    print('Shape of label tensor:', y_train.shape)
    print()

    y_test = to_categorical(np.asarray(y_test))
    print('Shape of y_test tensor:', y_test.shape)
    print()

    # now that we have preprocessed the data, we need to begin iterating, once per number of words to consider
    # for each number of words
    #   tokenize the training text
    #   prepare embedding matrix
    #   for three iterations
    #       create and compile the model
    #       split the data into training and validation sets
    #       fit the model

    results = []
    for max_nb_words in [500, 1000, 2000, 5000, 10000, 15000, 20000, 30000, 40000, 50000]:
        curr_results = [max_nb_words]
        for max_sequence_length in [1000]:
            print('-' * 80)
            print('Max number of words: %d, max sequence length: %d' % (max_nb_words, max_sequence_length))
            print('_'*80)
            print()

            print('Tokenizing training text')
            data, word_index, tokenizer = tokenize(texts_train, max_nb_words, max_sequence_length)
            print('Shape of data tensor:', data.shape)
            print()

            # prepare embedding matrix
            print('Preparing embedding matrix')
            nb_words = min(max_nb_words, len(word_index))
            # create a matrix with number of words + 1 (why +1??) rows an EMBEDDING_DIM columns
            # i.e. each row in the embedding matrix represents an embedding
            embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))
            # in fact, the ith row in the embedding matrix contains the embedding for the ith most common word in the documents
            # this is why we use nb_words + 1 rows - the first row in the embedding matrix is all zeros
            for word, i in word_index.items():
                if i > max_nb_words:
                    continue
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None:
                    # words not found in embedding index will be all-zeros.
                    embedding_matrix[i] = embedding_vector

            embedding_layer = Embedding(nb_words + 1,
                                        EMBEDDING_DIM,
                                        weights=[embedding_matrix],
                                        input_length=max_sequence_length,
                                        trainable=False)

            test_loss_and_acc = []
            for i in range(3):

                print('Building model.')
                # train a 1D convnet with global maxpooling
                sequence_input = Input(shape=(max_sequence_length,), dtype='int32')
                embedded_sequences = embedding_layer(sequence_input)
                x = Conv1D(128, 5, activation='relu')(embedded_sequences)
                x = MaxPooling1D(5)(x)
                x = Conv1D(128, 5, activation='relu')(x)
                x = MaxPooling1D(5)(x)
                x = Conv1D(128, 5, activation='relu')(x)
                x = MaxPooling1D(35)(x)
                x = Flatten()(x)
                x = Dense(128, activation='relu')(x)
                preds = Dense(20, activation='softmax')(x) # 20 is the number of newsgroups (classes)

                model = Model(sequence_input, preds)
                model.compile(loss='categorical_crossentropy',
                              optimizer='rmsprop',
                              metrics=['acc'])

                print('Splitting into training and validation sets.')
                # split the data into a training set and a validation set
                # some numpy shuffling magic
                indices = np.arange(data.shape[0]) # [0, ..., number of documents in the dataset-1]
                np.random.shuffle(indices)
                data = data[indices]
                y_train = y_train[indices]
                nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

                X_train = data[:-nb_validation_samples]
                curr_y_train = y_train[:-nb_validation_samples]
                X_val = data[-nb_validation_samples:]
                y_val = y_train[-nb_validation_samples:]
                print()

                print('Training model')
                model.fit(X_train, curr_y_train, validation_data=(X_val, y_val),
                          nb_epoch=20, batch_size=128,
                          callbacks=[EarlyStopping(monitor='val_loss', patience=3, verbose=True)])
                print()

                # convert the test data into the proper form and get test results
                print('Processing test docs')
                texts_test, _, _ = preproc(all_unproc_X_test)
                X_test = tokenize_test_texts(texts_test, tokenizer, max_sequence_length)

                print('Shape of X_test tensor:', X_test.shape)
                print('Shape of y_test tensor:', y_test.shape)
                print()

                loss_and_acc = model.evaluate(X_test, y_test, batch_size=128)

                print('Iteration %d:' % i)
                print('Max number of words: %d, max sequence length: %d, test loss: %f, test acc: %f' % (max_nb_words, max_sequence_length, loss_and_acc[0], loss_and_acc[1]))
                print()

                test_loss_and_acc.append(loss_and_acc)

            # average the loss and error over the three iterations
            avg_loss, avg_err = get_avgs(test_loss_and_acc)

            print('*'*40)
            print('Averages for max number of words %d, max sequence length %d' % (max_nb_words, max_sequence_length))
            print('avg loss\tavg err')
            print('%f\t%f' % (avg_loss, avg_err))
            print('*'*40)
            print()

            curr_results.append((max_sequence_length, avg_loss, avg_err))

        results.append(curr_results)
        to_pickle('cnn_results_%d_%d' % (max_nb_words, max_sequence_length), results, with_time=False)

    # save the results
    to_pickle('cnn_final_results', results, with_time=False)

    # print a summary of results
    print_results(results)

## END
