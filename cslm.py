import cPickle
import numpy
import theano
from collections import defaultdict
from itertools import chain

#from theano import ProfileMode
#import sys
#sys.stdout = open('cslm-profile.txt', 'w')
#sys.stderr = open('cslm-profile2.txt', 'w')
#profmode = theano.ProfileMode(optimizer='fast_run', linker=theano.gof.OpWiseCLinker())

with open('europarl_best_sentence.pkl') as f:
    model = cPickle.load(f)
with open('en.vcb.pkl') as f:
    table = cPickle.load(f)

input = theano.tensor.lmatrix()
windows = theano.tensor.lvector()
targets = theano.tensor.lvector()
f = theano.function(
    [input, windows, targets],
    theano.tensor.log10(
        model.fprop(input)[windows, targets]
    )
)

def run_cslm(batch):
    if len(batch) > 0:
        data = defaultdict(list)
        for pair in batch:
            ngram, score = pair.key(), pair.data()
            ngram_indices = []
            for word in ngram:
                index = table.get(word, 1)
                if index < 10000:
                    ngram_indices.append(index)
                else:
                    ngram_indices.append(1)
            data[tuple(ngram_indices[:-1])].append((ngram_indices[-1], ngram))
        input = numpy.array(data.keys(), dtype='int64')
        windows = numpy.array([i for i, window in enumerate(data.iterkeys())
                               for j in range(len(data[window]))], dtype='int64')
        targets = numpy.array([index for targets in data.itervalues()
                               for index, ngram in targets], dtype='int64')
        results = f(input, windows, targets)
        i = 0
        for window in data.iterkeys():
            for target, ngram in data[window]:
                batch[ngram] = float(results[i])
                i += 1
#    sys.stdout = open('cslm-profile.txt', 'w')
    return batch

#def profile():
#    profmode.print_summary()
