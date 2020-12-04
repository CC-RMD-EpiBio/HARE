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
Evaluation methods for binary relevance retrieval experiment.
'''

import numpy as np
import sklearn.metrics
from types import SimpleNamespace
from . import labeling
from . import scoring
from . import ranking
from . import predictions_io

def readLabelsFile(f):
    labels = {}
    with open(f, 'r') as stream:
        for line in stream:
            (unique_ID, label) = [s.strip() for s in line.split('\t')]
            labels[unique_ID] = int(label)
    return labels

def runEvaluation(predictions, labels, options):
    scorer = scoring.getScorer(SimpleNamespace(
        model_scorer=scoring.CountSegmentsAndTokens.NAME,
        threshold=options.threshold,
        num_blanks=options.num_blanks,
        use_label_instead=False
    ))

    scores = scorer.score(
        predictions
    )
    ranked = ranking.rankUnlabeledDataByModelScore(
        model_scores=scores
    )

    max_score = ranked[0].model_score

    y_true, y_score = [], []
    for doc in ranked:
        if doc.ID in labels:
            y_true.append(labels[doc.ID])
            y_score.append(doc.model_score / max_score)

    ap = sklearn.metrics.average_precision_score(
        y_true,
        y_score,
        pos_label=1
    )

    return ap

if __name__ == '__main__':
    from hedgepig_logger import log
    from . import predictions_io

    def _cli():
        import optparse
        parser = optparse.OptionParser(usage='Usage: %prog')
        parser.add_option('-i', '--predictions', dest='predictionsf',
            help='(REQUIRED) predictions file to analyze')
        parser.add_option('--labels', dest='labelsf',
            help='(REQUIRED) file mapping doc IDs/types to binary relevance labels')
        parser.add_option('--threshold', dest='threshold',
            type='float', default=0.5,
            help='binarization threshold for scoring (default %default)')
        parser.add_option('--num-blanks', dest='num_blanks',
            type='int', default=0,
            help='number of blanks to collapse for adjacent relevant'
                 ' segments (default %default)')
        parser.add_option('-l', '--logfile', dest='logfile',
            help='logfile path')
        (options, args) = parser.parse_args()

        if (not options.predictionsf):
            parser.error('Must provide --predictions')
        elif (not options.labelsf):
            parser.error('Must provide --labels')

        return options

    options = _cli()
    log.start(options.logfile)
    log.writeConfig([
        ('Predictions file', options.predictionsf),
        ('Labels file', options.labelsf),
        ('Scoring settings', [
            ('Binarization threshold', options.threshold),
            ('Number of blanks to collapse', options.num_blanks),
        ]),
    ], 'Binary ranking evaluation')

    log.writeln('Reading binary relevance labels from %s...' % options.labelsf)
    labels = readLabelsFile(options.labelsf)
    log.writeln('Read labels for {0:,} documents.\n'.format(len(labels)))

    log.writeln('Reading predictions from %s...' % options.predictionsf)
    predictions = predictions_io.readSamplePredictions(options.predictionsf, verbose=True)
    log.writeln('Read {0:,} predictions.\n'.format(len(predictions)))

    log.writeln('Running binary ranking evaluation...')
    ap = runEvaluation(
        predictions,
        labels,
        options
    )
    log.writeln('  AP: {0:.2f}%'.format(100*ap))

    log.stop()
