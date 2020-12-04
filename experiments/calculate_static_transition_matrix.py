###############################################################################
#
#                           COPYRIGHT NOTICE
#                  Mark O. Hatfield Clinical Research Center
#                       National Institutes of Health
#            United States Department of Health and Human Services
#
# This software was developed and is owned by the National Institutes of
# Health Clinical Center (NIHCC), an agency of the United States Department
# of Health and Human Services, which is making the software available to the
# public for any commercial or non-commercial purpose under the following
# open-source BSD license.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# (1) Redistributions of source code must retain this copyright
# notice, this list of conditions and the following disclaimer.
# 
# (2) Redistributions in binary form must reproduce this copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
# 
# (3) Neither the names of the National Institutes of Health Clinical
# Center, the National Institutes of Health, the U.S. Department of
# Health and Human Services, nor the names of any of the software
# developers may be used to endorse or promote products derived from
# this software without specific prior written permission.
# 
# (4) Please acknowledge NIHCC as the source of this software by including
# the phrase "Courtesy of the U.S. National Institutes of Health Clinical
# Center"or "Source: U.S. National Institutes of Health Clinical Center."
# 
# THIS SOFTWARE IS PROVIDED BY THE U.S. GOVERNMENT AND CONTRIBUTORS "AS
# IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
# TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED.
# 
# You are under no obligation whatsoever to provide any bug fixes,
# patches, or upgrades to the features, functionality or performance of
# the source code ("Enhancements") to anyone; however, if you choose to
# make your Enhancements available either publicly, or directly to
# the National Institutes of Health Clinical Center, without imposing a
# separate written license agreement for such Enhancements, then you hereby
# grant the following license: a non-exclusive, royalty-free perpetual license
# to install, use, modify, prepare derivative works, incorporate into
# other computer software, distribute, and sublicense such Enhancements or
# derivative works thereof, in binary and source code form.
#
###############################################################################

'''
Calculate an observation-independent state transition matrix
from observed label transition counts
'''

import numpy as np
from data.dataset import Dataset
from evaluation import transitions_io
from experiments import document_splits
from hedgepig_logger import log

def calculateTransitionMatrix(dataset):
    transition_counts = np.zeros(shape=(2,2), dtype=int)
    for (doc_ID, line_samples) in dataset.indexed.items():
        for (line_ix, indexed_samples) in line_samples.items():
            ordered_tokens = [
                t
                    for (ix, t)
                    in sorted(indexed_samples.items(), key=lambda k:k[0])
            ]
            for i in range(1, len(ordered_tokens)):
                prev_label = ordered_tokens[i-1].label
                this_label = ordered_tokens[i].label
                transition_counts[prev_label,this_label] += 1

    total_to_0 = np.sum(transition_counts[:,0])
    total_to_1 = np.sum(transition_counts[:,1])
    matrix = np.array([
        [ 
            transition_counts[0,0] / total_to_0,
            transition_counts[0,1] / total_to_1
        ],
        [
            transition_counts[1,0] / total_to_0,
            transition_counts[1,1] / total_to_1
        ]
    ])

    return transitions_io.StaticTransitionMatrix(matrix)

if __name__ == '__main__':
    def _cli():
        import optparse
        parser = optparse.OptionParser(usage='Usage: %prog')
        parser.add_option('--training', dest='training_file',
            help='labels file for training data')
        parser.add_option('--splits', dest='splits_base',
            help='file specifying cross-validation splits'
                 ' (if not provided, trains on full dataset)')
        parser.add_option('-o', '--output', dest='outf',
            help='file to write transition matri(x|ces) to'
                 ' (if using --splits, writes one matrix for'
                 ' the training set of each split)')
        parser.add_option('-l', '--logfile', dest='logfile',
                help='name of file to write log contents to (empty for stdout)',
                default=None)
        (options, args) = parser.parse_args()

        if not options.training_file:
            parser.error('Must supply --training <FILE>')
        elif not options.outf:
            parser.error('Must supply --output <FILE>')

        return options

    options = _cli()
    log.start(options.logfile)
    log.writeConfig([
        ('Training file', options.training_file),
        ('Cross-validation splits file', ('N/A' if options.splits_base is None else options.splits_base)),
        ('Output path', options.outf),
    ])

    log.writeln('Loading dataset from %s...' % options.training_file)
    dataset = Dataset(
        labels_file=options.training_file
    )
    log.writeln('Loaded')
    log.writeln('  Positive samples: {0:,}'.format(len(dataset.positive_sample_IDs)))
    log.writeln('  Negative samples: {0:,}\n'.format(len(dataset.negative_sample_IDs)))

    if options.splits_base:
        log.writeln('Reading dataset splits from %s...' % options.splits_base)
        splits = document_splits.readDocumentSplits(options.splits_base)
        log.writeln('Found splits for {0:,} folds.\n'.format(len(splits)))

        for i in range(len(splits)):
            log.writeln('Calculating transition matrix for fold {0:,}/{1:,}...'.format(
                (i+1), len(splits)
            ))
            outf = '%s.fold-%d' % (options.outf, i)
            (train_files, dev_files, test_files) = splits[i]
            train = dataset.filter(train_files)
            matrix = calculateTransitionMatrix(train)
            matrix.toFile(outf)
            log.writeln('Wrote matrix to {0}.\n'.format(outf))
    else:
        log.writeln('Calculating overall transition matrix...')
        matrix = calculateTransitionMatrix(dataset)
        matrix.toFile(options.outf)
        log.writeln('Wrote matrix to {0}.\n'.format(options.outf))

    log.stop()
