"""
This module creates a number of worker processes, each of
which load the given pylearn2 model. Batches of n-grams can
then be submitted, which get filtered for duplicates before
the model is run and the output returned (asynchronously)
"""
import cPickle
import numpy
import sys

from multiprocessing import Pool


sys.stdout = open('cslm_pool.log', 'w')
sys.stderr = open('cslm_pool.log', 'w')

num_processes = 1
model_file = 'europarl_cslm_cpu.pkl'


def init_cslm():
    global f
    from theano import function, tensor
    with open(model_file) as f:
        model = cPickle.load(f)
    input = tensor.imatrix()
    target_samples = tensor.ivector()
    target_words = tensor.ivector()
    output = model.fprop(input)
    f = function(
        [input, target_samples, target_words],
        tensor.log10(
            output.flatten()[target_samples * output.shape[1] + target_words]
        ), allow_input_downcast=True
    )


def cslm(batch):
    input = numpy.random.randint(0, 10000, (500, 6))
    target_samples = numpy.random.randint(0, 500, 10000)
    target_words = numpy.random.randint(0, 10000, 10000)
    return f(input, target_samples, target_words)


def apply_async(batch, i):
    async_result = pool.apply_async(cslm, [batch[:i]])
    return async_result


def get(async_result):
    return async_result.get()


def close():
    pool.close()

pool = Pool(processes=num_processes, initializer=init_cslm)
print "__main__"
