import cPickle
import logging
import signal

import numpy
import theano
from theano import function
from theano import tensor

logging.basicConfig(level=logging.DEBUG)


def handler(signum, frame):
    """
    We catch the SIGINT that is broadcast when Moses is quit using
    e.g. CTRL + C, otherwise an uncaught KeyboardInterrupt is raised,
    causing the embedded interpreter to segfault.
    """
    pass

signal.signal(signal.SIGINT, handler)

logging.debug("Loading model...")

with open('/u/vanmerb/pylearn2/holger_bow_best.pkl') as f:
    model = cPickle.load(f)

logging.debug("Compiling Theano function...")

input = tensor.imatrix()
windows = tensor.ivector()
targets = tensor.ivector()
if True:
    source = theano.tensor.ivector()
    source_embeddings = model.layers[0].raw_layer.layers[1].get_params()[0]
    assert source_embeddings.name == 'source_projection_W'
    source_projection = tensor.cast(tensor.sum(source_embeddings[source],
                                               axis=0) / source.shape[0],
                                    'float32')
    stacked_source = tensor.extra_ops.repeat(
        source_projection.dimshuffle('x', 0), input.shape[0], axis=0
    )
    ngram_projection = model.layers[0].raw_layer.layers[0].fprop(input)
    state = tensor.concatenate([ngram_projection, stacked_source], axis=1)
    for layer in model.layers[1:]:
        state = layer.fprop(state)
    results = state.flatten()[windows * state.shape[1] + targets]
    assert results.dtype == 'float32'
    f = function(
        [input, windows, targets, source],
        tensor.log(results)
    )
else:
    results = model.fprop(input)
    f = function(
        [input, windows, targets],
        tensor.log(
            results.flatten()[windows * results.shape[1] + targets]
        )
    )

logging.debug("Python module loaded!")


def filter(ngrams):
    """
    This function takes a matrix of ngrams and filters
    out all the duplicate contexts (i.e. n-1 inputs).

    Parameters
    ----------
    ngrams : 2d ndarray
        An array with rows of n-grams

    Returns
    -------
    final_inputs : 2d ndarray
        An array of unique n-1 grams (contexts)
    target_samples : 1d ndarray
        An array of indices to final_inputs, with
        duplicate entries for when multiple targets
        are needed from a single input
    target_words : 1d ndarray
        An array of targets, corresponding to the
        samples given in target_samples
    reverse_sorted_indices : 1d ndarray
        The indices required to reorder the target
        words back to the original input
    """
    inputs, targets = ngrams[:, :-1], ngrams[:, -1]
    sorted_indices = numpy.lexsort(inputs.T[::-1])
    reverse_sorted_indices = numpy.argsort(sorted_indices)
    unique_indices = numpy.nonzero(numpy.concatenate((
        [True],  # Always take the first row
        numpy.any(inputs[sorted_indices[1:]] != inputs[sorted_indices[:-1]],
                  axis=1)  # Only the unique ones
    )))[0]
    final_inputs = inputs[sorted_indices][unique_indices]
    repeats = numpy.roll(unique_indices, -1) - unique_indices
    repeats[-1] += inputs.shape[0]

    target_samples = numpy.repeat(numpy.arange(unique_indices.shape[0],
                                               dtype='int32'), repeats)
    target_words = targets[sorted_indices]
    return final_inputs, target_samples, target_words, reverse_sorted_indices


def get(ngrams, scores, batch_size, source=None):
    """
    Scores a given number of ngrams and writes the scores
    to a given vector.

    Parameters
    ----------
    ngrams : 2d ndarray
        A matrix of ngrams
    scores : 1d ndarray
        A vector of scores
    batch_size : int
        Only the first batch_size rows will
        be scored, and the scores will be written
        to the first batch_size elements of the scores
        vector.
    source : 1d ndarray
        The indices of the source sentence; read up to -1

    Returns
    -------
    True if successful
    """
    (final_inputs, target_samples,
        target_words, reverse_sorted_indices) = filter(ngrams[:batch_size])
    if source is not None:
        results = f(final_inputs, target_samples, target_words, source)
    else:
        results = f(final_inputs, target_samples, target_words)
    scores[:batch_size] = results[reverse_sorted_indices]
    return True
