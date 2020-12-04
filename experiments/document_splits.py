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
Methods for generating document-level dataset splits
'''

import time
import os
import glob
import numpy as np
from data.dataset import Dataset
from data.util import readDocIDList
from hedgepig_logger import log

def readDocumentSplits(fbase):
    '''Given a base file path, returns a list of
    (train, dev, test) triples, where each is a list
    of document IDs for that subset of that fold.
    '''
    fold_data = []
    (training_template, _, _) = getTrainDevTestFiles(fbase, 0)
    training_template = training_template.replace('fold-0', 'fold-*')
    n_folds = len(glob.glob(training_template))

    for i in range(n_folds):
        (trainf, devf, testf) = getTrainDevTestFiles(fbase, i)
        train = readDocIDList(trainf)
        dev = readDocIDList(devf)
        test = readDocIDList(testf)
        fold_data.append(
            (train, dev, test)
        )
    return fold_data

def getTrainDevTestFiles(fbase, fold):
    train = '%s.fold-%d.training' % (fbase, fold)
    dev = '%s.fold-%d.dev' % (fbase, fold)
    test = '%s.fold-%d.test' % (fbase, fold)
    return (train, dev, test)

def logAndValidateFold(fold, n_folds, training, dev, test, doc_IDs):
    recompiled_set, recompiled_list = set(), []
    for subset in [training, dev, test]:
        recompiled_set = recompiled_set.union(set(subset))
        recompiled_list.extend(list(subset))

    log.writeln('  >> Fold {0}/{1}'.format(fold+1, n_folds))
    log.writeln('     Training: {0:,} samples'.format(len(training)))
    log.writeln('     Dev: {0:,} samples'.format(len(dev)))
    log.writeln('     Test: {0:,} samples'.format(len(test)))
    log.writeln('     Adds up correctly? {0}'.format(len(recompiled_list) == len(doc_IDs)))
    log.writeln('     Covers full set? {0}'.format(len(recompiled_set) == len(doc_IDs)))


if __name__ == '__main__':
    def _cli():
        import optparse
        parser = optparse.OptionParser(usage='Usage: %prog LABELS_FILE OUTF')
        parser.add_option('-n', '--n-folds', dest='n_folds',
                type='int', default=5,
                help='number of folds to split data into for cross-validation')
        parser.add_option('--dev-size', dest='dev_size',
                type='float', default=0.1,
                help='fraction of total dataset (NOT TRAIN) to hold out for dev')
        parser.add_option('--random-seed', dest='random_seed',
                type='int', default=-1,
                help='random seed for reproducibility; defaults to current epoch time')
        parser.add_option('-l', '--logfile', dest='logfile',
                help='name of file to write log contents to (empty for stdout)',
                default=None)
        (options, args) = parser.parse_args()

        def _bail(msg):
            print(msg)
            print('')
            parser.print_help()
            exit()

        if options.dev_size < 0 or options.dev_size >= 1:
            _bail('--dev-size must be in range [0,1)')
        elif len(args) != 2:
            _bail('Must provide LABELS_FILE and OUTF')

        if options.random_seed == -1:
            options.random_seed = int(time.time())

        return args, options

    (labels_f, outf), options = _cli()
    log.start(options.logfile)
    log.writeConfig([
        ('Labels file', labels_f),
        ('Output file base', outf),
        ('Number of folds', options.n_folds),
        ('Dev set size (overall)', options.dev_size),
        ('Random seed', options.random_seed),
    ])

    log.writeln('Reading set of doc IDs from %s...' % labels_f)
    doc_IDs = set(Dataset(labels_f).samples_by_doc_ID.keys())
    log.writeln('Found {0:,} unique doc IDs.\n'.format(len(doc_IDs)))

    log.writeln('Calculating split sizes...')
    doc_IDs = list(doc_IDs)
    dev_size = int(options.dev_size * len(doc_IDs))
    fold_size = len(doc_IDs) // options.n_folds
    log.writeln('  Training set size (approx): {0:,} documents'.format(
        ((options.n_folds - 1) * fold_size) - dev_size
    ))
    log.writeln('  Dev set size (approx): {0:,} documents'.format(
        dev_size
    ))
    log.writeln('  Test set size (approx): {0:,} documents\n'.format(
        fold_size
    ))

    log.writeln('Generating dataset splits...')
    np.random.shuffle(doc_IDs)
    data_folds = []
    for i in range(options.n_folds - 1):
        data_folds.append(doc_IDs[i*fold_size:(i+1)*fold_size])
    data_folds.append(doc_IDs[(options.n_folds-1) * fold_size:])

    for i in range(options.n_folds):
        training, dev, test = [], [], []
        for j in range(options.n_folds):
            if j == i:
                test = data_folds[j]
            else:
                training.extend(data_folds[j])

        dev = training[:dev_size]
        training = training[dev_size:]

        logAndValidateFold(i, options.n_folds, training, dev, test, doc_IDs)

        (trainf, devf, testf) = getTrainDevTestFiles(outf, i)
        with open(trainf, 'w') as training_stream, \
             open(devf, 'w') as dev_stream, \
             open(testf, 'w') as test_stream:
            for (t_src, stream) in [
                (training, training_stream),
                (dev, dev_stream),
                (test, test_stream)
            ]:
                for doc_ID in t_src:
                    stream.write('%s\n' % doc_ID)
        log.writeln('     Wrote output to {0}\n'.format('%s.fold-%d' % (outf, i)))

    log.writeln('\n\n--- Testing split reading ---\n')

    log.writeln('Reading %d-fold document splits from %s...' % (options.n_folds, outf))
    splits = readDocumentSplits(outf)

    for i in range(options.n_folds):
        (training, dev, test) = splits[i]
        logAndValidateFold(i, options.n_folds, training, dev, test, doc_IDs)

    log.stop()
