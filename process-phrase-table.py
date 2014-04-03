#!/usr/bin/python

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
    words = fields[1].strip().split(" ")
    fields[1] = " " + " ".join([word + "|" + str(table.get(word, 1))
                                for word in words]) + " "
    print("|||".join(fields), file=args.output)
