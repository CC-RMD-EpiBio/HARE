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

import numpy as np
import sklearn.metrics
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
from evaluation import predictions_io
from evaluation import utils
import visualization.utils
from hedgepig_logger import log

COLOR_PRECISION = '#e41a1c'
COLOR_RECALL = '#377eb8'
COLOR_FSCORE = '#984ea3'
COLOR_THRESH = '#aaaaaa'

def precisionRecallF2Curve(predictions):
    labels = [p.label for p in predictions]
    probs = utils.cleanProbabilities([p.positive_probability for p in predictions])

    precisions, recalls, thresholds = sklearn.metrics.precision_recall_curve(
        labels,
        probs,
        pos_label=1
    )

    # trim off the final 1 and 0 values
    precisions = precisions[:-1]
    recalls = recalls[:-1]

    beta = 2

    f2s = [
        (
            (
                (
                    (1+(beta**2)) * precisions[i] * recalls[i]
                ) / (
                    ((beta**2) * precisions[i]) + recalls[i]
                )
            )
            if precisions[i] + recalls[i] > 0
            else 0
        )
            for i in range(len(thresholds))
    ]
    best_ix = np.argmax(f2s)
    best_threshold = thresholds[np.argmax(f2s)]

    return (
        precisions,
        recalls,
        f2s,
        thresholds,
        best_ix
    )

def plotCurve(precisions, recalls, fscores, thresholds, best_ix, outf=None, figsize=(5,5), font_size=18):
    '''
    If outf is None, plots to binary stream and returns contents
    '''
    font = {
        'family'  : 'sans-serif',
        'size'    : font_size
    }
    matplotlib.rc('font', **font)

    (fig, ax) = plt.subplots(figsize=figsize)

    ax.step(
        thresholds,
        precisions,
        label='Precision',
        color=COLOR_PRECISION,
        where='post',
        linestyle='-.',
        linewidth=2
    )
    ax.step(
        thresholds,
        recalls,
        label='Recall',
        color=COLOR_RECALL,
        where='post',
        linestyle='--',
        linewidth=2
    )
    ax.step(
        thresholds,
        fscores,
        label='F-2',
        color=COLOR_FSCORE,
        where='post',
        linestyle='-',
        linewidth=3
    )

    ax.plot(
        [thresholds[best_ix], thresholds[best_ix]],
        [0, 1],
        color=COLOR_THRESH,
        linestyle=':',
        label='Threshold={0:.2f}'.format(thresholds[best_ix])
    )

    plt.xlabel('Threshold')
    plt.ylabel('Metric')
    leg = plt.legend(
        loc='center left',
        bbox_to_anchor=(1.0,0.5)
    )

    plt.savefig(outf, format='png', bbox_inches='tight', extra_artists=[leg])

    plt.close()

if __name__ == '__main__':
    def _cli():
        import optparse
        parser = optparse.OptionParser(usage='Usage: %prog PREDICTIONS')
        parser.add_option('-f', '--file-map', dest='file_mapf',
            help='(REQUIRED) visualization file map')
        parser.add_option('-d', '--dataset', dest='dataset',
            help='(REQUIRED) dataset to visualize output for')
        parser.add_option('-o', '--output', dest='outf',
            help='(REQUIRED) output path for .png file')
        (options, args) = parser.parse_args()

        if not options.outf:
            parser.error('Must supply --output')
        elif not options.file_mapf:
            parser.error('Must supply --file-map')
        elif not options.dataset:
            parser.error('Must supply --dataset')

        return options

    options = _cli()

    log.writeln('Loading file map from %s...' % options.file_mapf)
    file_map = visualization.utils.loadFileMap(config=None, fpath=options.file_mapf)
    log.writeln('Mapped {0:,} annotation sets.\n'.format(len(file_map)))

    log.writeln('Reading predictions from %s...' % file_map[options.dataset])
    predictions = predictions_io.readSamplePredictions(file_map[options.dataset].filepath)
    log.writeln('Read {0:,} predictions.\n'.format(len(predictions)))

    (precisions, recalls, f2s, thresholds, best_ix) = precisionRecallF2Curve(predictions)
    plotCurve(precisions, recalls, f2s, thresholds, best_ix, options.outf, figsize=(6,2))
