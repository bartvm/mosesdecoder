#!/usr/bin/env python

from __future__ import print_function
import argparse
import cPickle
import gzip

parser = argparse.ArgumentParser()
parser.add_argument("input",
                    type=argparse.FileType('r'),
                    help="The phrase table to be processed")
parser.add_argument("output",
                    type=argparse.FileType('w'),
                    help="The output phrase table with CSLM IDs added")
parser.add_argument("dictionary",
                    type=argparse.FileType('r'),
                    help="A pickled dictionary with words and IDs as keys and "
                         "values respectively")
parser.add_argument("--labels",
                    type=int, default=0,
                    help="Set the maximum word index")
parser.add_argument("--source", action='store_true',
                    help="Binarize the source phrases instead of the targets")
args = parser.parse_args()

if args.input.name.endswith('.gz'):
    args.input = gzip.GzipFile(args.input.name, args.input.mode,
                               9, args.input)

if args.output.name.endswith('.gz'):
    args.output = gzip.GzipFile(args.output.name, args.output.mode,
                                9, args.output)

table = cPickle.load(args.dictionary)

for line in args.input:
    fields = line.strip().split("|||")
    field_index = 0 if args.source else 1
    words = fields[field_index].strip().split(" ")
    word_indices = [table.get(word, 1) for word in words]
    if args.labels > 0:
        word_indices = [word_index if word_index < args.labels else 1
                        for word_index in word_indices]
    fields[field_index] = " " + " ".join([word + "|" + str(word_index)
                                          for word, word_index
                                          in zip(words, word_indices)]) + " "
    print("|||".join(fields), file=args.output)
