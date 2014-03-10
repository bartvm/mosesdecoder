import cPickle
import numpy
import theano

with open('europarl_best_sentence.pkl') as f:
    model = cPickle.load(f)
with open('en.vcb.pkl') as f:
    table = cPickle.load(f)

input = theano.tensor.lmatrix()
targets = theano.tensor.lrow()
f = theano.function(
    [input, targets],
    theano.tensor.log10(
        model.fprop(input)[theano.tensor.arange(targets.shape[0]),
                           targets.flatten()]
    )
)

requests = []
targets = []
count = 0

def request(phrase):
    global requests
    global targets
    phrase = [table.get(word, 1) for word in phrase]
    phrase = [index if index < 10000 else 1 for index in phrase]
    targets.append(phrase[-1])
    requests.append(phrase[:-1])

def run_cslm():
    global requests
    global targets
    global count
    count += len(targets)
    print count
#    print "Number of duplicate inputs: " + str(len(requests) - len(set(map(tuple, requests)))) + "/" + str(len(requests))
#    ngrams = [request + [target] for request, target in zip(requests, targets)]
#    print "Number of duplicate n-grams: " + str(len(ngrams) - len(set(map(tuple, ngrams)))) + "/" + str(len(ngrams))
#    print requests
    if requests:
      results = list(f(numpy.array(requests, dtype='int64'),
                       numpy.array([targets], dtype='int64')))
      requests = []
      targets = []
    else:
        results = []
    return results