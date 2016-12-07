'''Postprocess'''

import cPickle as pickle
from time import time

def to_pickle(name, content):
    name = 'postprocessed_data/' + name + '_%.f.pkl' % time()
    print('saving data as %s' % name)
    with open(name, 'wb') as f:
        pickle.dump(content, f)
