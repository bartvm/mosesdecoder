import cPickle
import logging
import sys

import numpy
import theano

from theano import ProfileMode

logging.basicConfig(filename='cslm.log', level=logging.DEBUG)
sys.stdout = open('cslm.log', 'a')
sys.stderr = open('cslm.log', 'a')

profmode = theano.ProfileMode(optimizer='fast_run',
                              linker=theano.gof.OpWiseCLinker())

logging.debug("Imported modules")

theano.config.optimizer = 'None'

with open('europarl_cslm.pkl') as f:
    model = cPickle.load(f)

logging.debug("Loaded model")

input = theano.tensor.imatrix()
windows = theano.tensor.ivector()
targets = theano.tensor.ivector()
results = model.fprop(input)
f = theano.function(
    [input, windows, targets],
    theano.tensor.log10(
        results.flatten()[windows * results.shape[1] + targets]
    ), mode=profmode
)

logging.debug("Compiled Theano function")


def filter(ngrams):
    try:
        inputs, targets = numpy.ascontiguousarray(ngrams[:, :-1]), numpy.ascontiguousarray(ngrams[:, -1])
        sorted_indices = numpy.lexsort(inputs.T[::-1])
        reverse_sorted_indices = numpy.argsort(sorted_indices)
        unique_indices = numpy.nonzero(numpy.concatenate((
            [True], # Always take the first row
            numpy.any(inputs[sorted_indices[1:]] != inputs[sorted_indices[:-1]],
                      axis=1) # Only the unique ones
        )))[0]
        final_inputs = inputs[sorted_indices][unique_indices]
    except:
        logging.debug('A')
        logging.debug(sys.exc_info())
        logging.debug(locals())
        return inputs, numpy.arange(len(targets), dtype='int32'), targets, numpy.arange(len(targets), dtype='int32')
    try:
        repeats = numpy.roll(unique_indices, -1) - unique_indices
        repeats[-1] += inputs.shape[0]
     
        target_samples = numpy.repeat(numpy.arange(unique_indices.shape[0],
                                                   dtype='int32'), repeats)
        target_words = targets[sorted_indices]
    	return final_inputs, target_samples, target_words, reverse_sorted_indices
    except:
        logging.debug('B')
        logging.debug(sys.exc_info())
        logging.debug(locals())
        return inputs, numpy.arange(len(targets), dtype='int32'), targets, numpy.arange(len(targets), dtype='int32')


def get(ngrams, scores, batch_size):
    try:
        (final_inputs, target_samples,
            target_words, reverse_sorted_indices) = filter(ngrams[:batch_size])
        results = f(final_inputs, target_samples, target_words)
        scores[:batch_size] = results[reverse_sorted_indices]
        return True
    except:
        logging.debug(sys.exc_info())
        return False


def profile():
    profmode.print_summary()

import atexit
atexit.register(profile)
