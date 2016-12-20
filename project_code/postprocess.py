'''Postprocess'''

import cPickle as pickle
from time import time

def to_pickle(name, content, with_time=True):
    if with_time:
        name = 'postprocessed_data/' + name + '_%.f.pkl' % time()
    else:
        name = 'postprocessed_data/' + name + '.pkl'
    print('saving data as %s' % name)
    with open(name, 'wb') as f:
        pickle.dump(content, f)
