import cPickle
import numpy as np
import theano
from collections import defaultdict
import line_profiler

profile_ = True

def preprocess(batch):
    data = np.empty((len(batch), 7), dtype='int32')
    for i, pair in enumerate(batch):
        data[i] = map(int, pair.key())
    inputs, targets = data[:, :-1], data[:, -1]
    inputs_row_view = np.ascontiguousarray(inputs).view(
        np.dtype((np.void, inputs.dtype.itemsize * inputs.shape[1]))
    ).flatten()
    sorted_indices = np.argsort(inputs_row_view)
    reverse_sorted_indices = np.argsort(sorted_indices)
    _, unique_indices = np.unique(inputs_row_view[sorted_indices], return_index=True)
    final_inputs = inputs[sorted_indices][unique_indices]

    repeats = np.roll(unique_indices, -1) - unique_indices
    repeats[-1] += inputs.shape[0]

    target_samples = np.repeat(np.arange(unique_indices.shape[0], dtype='int32'), repeats)
    target_words = targets[sorted_indices]
    return final_inputs, target_samples, target_words, reverse_sorted_indices

def run_cslm(batch):
    # This is probably redundant
    if len(batch) > 0:
        final_inputs, target_samples, target_words, reverse_sorted_indices = preprocess(batch)
        results = f(final_inputs, target_samples, target_words)
        results = results[reverse_sorted_indices]
        for i, pair in enumerate(batch):
            batch[pair.key()] = float(results[i])
    return batch

if profile_:
    from theano import ProfileMode
    import sys
    filename = str(np.random.randint(1000000)) + '.log'
    sys.stdout = open(filename, 'w')
    sys.stderr = open(filename, 'w')
    profmode = theano.ProfileMode(optimizer='fast_run', linker=theano.gof.OpWiseCLinker())

with open('europarl_cslm.pkl') as f:
    model = cPickle.load(f)

input = theano.tensor.imatrix()
windows = theano.tensor.ivector()
targets = theano.tensor.ivector()
results = model.fprop(input)
if profile_:
    f = theano.function(
        [input, windows, targets],
        theano.tensor.log10(
            results.flatten()[windows * results.shape[1] + targets]
        ), mode=profmode
    )
    # import cProfile
    # pr = cProfile.Profile()
    # pr.enable()
    lprof = line_profiler.LineProfiler(preprocess, run_cslm)
    lprof.enable()
else:
    f = theano.function(
        [input, windows, targets],
        theano.tensor.log10(
            model.fprop(input)[windows, targets]
        )
    )

def profile():
    if profile_:
        profmode.print_summary()
        # pr.create_stats()
        # pr.dump_stats('python' + filename)
        lprof.disable()
        lprof.dump_stats('python' + filename)
        lprof.print_stats()
