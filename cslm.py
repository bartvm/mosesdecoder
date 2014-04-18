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
    inputs, targets = ngrams[:, :-1], ngrams[:, -1]
    inputs_row_view = numpy.ascontiguousarray(inputs).view(
        numpy.dtype((numpy.void, inputs.dtype.itemsize * inputs.shape[1]))
    ).flatten()
    sorted_indices = numpy.argsort(inputs_row_view)
    reverse_sorted_indices = numpy.argsort(sorted_indices)
    _, unique_indices = numpy.unique(inputs_row_view[sorted_indices],
                                     return_index=True)
    final_inputs = inputs[sorted_indices][unique_indices]

    repeats = numpy.roll(unique_indices, -1) - unique_indices
    repeats[-1] += inputs.shape[0]

    target_samples = numpy.repeat(numpy.arange(unique_indices.shape[0],
                                               dtype='int32'), repeats)
    target_words = targets[sorted_indices]
    return final_inputs, target_samples, target_words, reverse_sorted_indices


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
