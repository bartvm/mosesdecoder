import numpy as np
import sys

sys.stdout = open('cslm.log', 'w')
sys.stderr = open('cslm.log', 'w')

print "test"

def get(ngrams, scores):
    scores = np.random.random(scores.shape)
    print ngrams
    print scores
