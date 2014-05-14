#!/usr/bin/env python
import argparse
import cPickle
import gzip

parser = argparse.ArgumentParser()
parser.add_argument("input",
                    type=argparse.FileType('r'),
                    help="The phrase table to be processed")
parser.add_argument("source_output",
                    type=argparse.FileType('w'),
                    help="The source output file")
parser.add_argument("target_output",
                    type=argparse.FileType('w'),
                    help="The target output file")
parser.add_argument("source_dictionary",
                    type=argparse.FileType('r'),
                    help="A pickled dictionary with words and IDs as keys and "
                         "values respectively")
parser.add_argument("target_dictionary",
                    type=argparse.FileType('r'),
                    help="A pickled dictionary with words and IDs as keys and "
                         "values respectively")
parser.add_argument("--labels",
                    type=int, default=0,
                    help="Set the maximum word index")
args = parser.parse_args()

files = [args.source_output, args.target_output, args.input]
for i, f in enumerate(files):
    if f.name.endswith('.gz'):
        files[i] = gzip.GzipFile(f.name, f.mode, 9, f)

source_table = cPickle.load(args.source_dictionary)
target_table = cPickle.load(args.target_dictionary)
tables = [source_table, target_table]

for line in files[2]:
    fields = line.strip().split('|||')
    for field_index in [0, 1]:
        words = fields[field_index].strip().split(' ')
        word_indices = [tables[field_index].get(word, 1) for word in words]
        if args.labels > 0:
            word_indices = [word_index if word_index < args.labels else 1
                            for word_index in word_indices]
        files[field_index].write(' '.join([str(word_index) for word_index
                                           in word_indices]) + '\n')
for f in enumerate(files):
    f.close()
