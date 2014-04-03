#!/usr/bin/python

from __future__ import print_function
import argparse
import cPickle

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

table = cPickle.load(args.dictionary)

for line in args.input:
    fields = line.strip().split("|||")
    print(fields)
    words = fields[1].strip().split(" ")
    print(words)
    fields[1] = " " + " ".join([word + "|" + str(table.get(word, 1))
                                for word in words]) + " "
    print(fields)
    print("|||".join(fields), file=args.output)
