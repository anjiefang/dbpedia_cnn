# -*- coding: utf-8 -*-
import sys
import numpy as np
import itertools


def print_progress(count, total, suffix=''):
    bar_len = 30
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', suffix))
    sys.stdout.flush()
    if count == total:
        print("\n")


def get_WE_vectors(vectorPath, we_length, indices, npy=None):
    print("loading pre-trained vectors...")
    if npy:
        we_vectors = np.load(vectorPath)
        return we_vectors[indices]

    vectors = []
    with open(vectorPath, 'r') as f:
        count = 0
        print_progress(count, we_length, 'WE')
        for line in f:
            vectors.append(line.strip().split(' '))
            count += 1
            print_progress(count, we_length, 'WE')

    ''' convert from string to float '''
    trained_vectors = np.array(vectors).astype(np.float)
    np.save(open("/home/xyang/code/WEs/multi-en-d500.npy", 'w'), trained_vectors)
    return trained_vectors[indices]


def initialise_OOV(vocab_size, we_vectors):
    rows_diff = vocab_size - len(we_vectors)

    ''' 没见过的word填充0 '''
    print "fill 0 for unseen words"
    fill_vector = np.zeros((rows_diff, len(we_vectors[0])))
    return  np.append(we_vectors, fill_vector, axis=0)

def initialise_UNK(we_vectors):
    ''' 没见过的word填充0 '''
    print "fill 0 for unseen words"
    fill_vector = np.zeros((1, len(we_vectors[0])))
    return np.append(fill_vector, we_vectors, axis=0)

def build_folds(folds):
    for i in xrange(len(folds)):
        validation = folds[i]
        try:
            test = folds[i + 1]
            train = list(itertools.chain.from_iterable(folds[:i] + folds[i + 2:]))
        except:
            test = folds[0]
            train = list(itertools.chain.from_iterable(folds[1:i]))
        yield train, validation, test