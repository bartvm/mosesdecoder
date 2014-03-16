import cPickle
import numpy
#import theano
from collections import defaultdict
from itertools import chain

#with open('europarl_best_sentence.pkl') as f:
#    model = cPickle.load(f)
#with open('en.vcb.pkl') as f:
#    table = cPickle.load(f)
#
#input = theano.tensor.lmatrix()
#windows = theano.tensor.lvector()
#targets = theano.tensor.lvector()
#f = theano.function(
#    [input, windows, targets],
#    theano.tensor.log10(
#        model.fprop(input)[windows, targets]
#    )
#)

def run_cslm(batch):
#    print ngram
#    print dir(ngram)
#    print len(ngram)
#    print ngram[0]
    data = defaultdict(list)
    for pair in batch:
#        pair.data() = 2.0
        ngram, score = pair.key(), pair.data()
#        print str(batch[ngram]) + "->",
        batch[ngram] = 2.0
#        print str(batch[ngram]) + ", ",
#        print type(batch)
#        print type(batch[ngram])
#    print ""
#        print dir(batch)
#        print dir(batch[ngram])
#        break
#        print str(batch[ngram])
#    ngram_indices = []
#        print "Score is " + str(score)
#        print "Word IDs are ",
#        for word in ngram:
#            print word,
#        print ""
#        index = table.get(word, 1)
#        if index < 10000:
#            ngram_indices.append(index)
#        else:
#            ngram_indices.append(1)
#    data[tuple(ngram_indices[:-1])].append(ngram_indices[-1])
#    input = numpy.array(data.keys())
#    windows = numpy.array([i for i, window in enumerate(data.iterkeys()) for j in range(len(data[window]))])
#    targets = numpy.array([index for targets in data.itervalues() for index in targets])
#    f(input, windows, targets)
#    print len(input)
#    if requests:
#      results = list(f(numpy.array(requests, dtype='int64'),
#                       numpy.array([targets], dtype='int64')))
#      requests = []
#      targets = []
#    else:
#        results = []
#    return results
    return batch
