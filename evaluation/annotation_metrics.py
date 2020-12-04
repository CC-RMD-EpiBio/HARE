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
Token-level annotation evaluation
'''

import numpy as np
from types import SimpleNamespace
import sklearn.metrics
from . import labeling
from . import utils

def getAnnotationMetricsRaw(probabilities, labels, threshold=0.5, f_score_beta=1):
    data = [
        SimpleNamespace(
            positive_probability=probabilities[i],
            label=labels[i]
        )
            for i in range(len(probabilities))
    ]
    return getAnnotationMetrics(data, threshold=threshold, f_score_beta=f_score_beta)

def getAnnotationMetrics(predictions, threshold=0.5, f_score_beta=None):
    '''
    Only uses .positive_probability and .label fields
    '''
    metrics = SimpleNamespace()

    labels = [int(p.label) for p in predictions]
    probabilities = utils.cleanProbabilities([p.positive_probability for p in predictions])
    preds = labeling.binarizeProbabilities(probabilities, threshold)

    # accuracy eval
    metrics.correct = 0
    metrics.total = len(labels)
    for i in range(len(labels)):
        if preds[i] == labels[i]:
            metrics.correct += 1
    if metrics.total > 0:
        metrics.accuracy = metrics.correct / metrics.total
    else:
        metrics.accuracy = 0
    
    # Pr/Rec/FBeta measures
    (
        metrics.precision,
        metrics.recall,
        metrics.f1,
        _
    ) = sklearn.metrics.precision_recall_fscore_support(
        labels,
        preds,
        pos_label=1,
        average='binary',
        beta=1,
        zero_division=0
    )
    (_, _, metrics.f2, _) = sklearn.metrics.precision_recall_fscore_support(
        labels,
        preds,
        pos_label=1,
        average='binary',
        beta=2,
        zero_division=0
    )
    if (not f_score_beta is None):
        (_, _, metrics.f_beta, _) = sklearn.metrics.precision_recall_fscore_support(
            labels,
            preds,
            pos_label=1,
            average='binary',
            beta=f_score_beta,
            zero_division=0
        )

    # AUC
    metrics.support = sum(labels)
    if metrics.support > 0:
        metrics.auc = sklearn.metrics.roc_auc_score(
            labels,
            probabilities
        )
    else:
        metrics.auc = 0

    return metrics


if __name__ == '__main__':
    from hedgepig_logger import log
    from . import predictions_io
    from . import transitions_io
    from . import decoding
    from experiments import document_splits

    def _cli():
        import optparse
        parser = optparse.OptionParser(usage='Usage: %prog')
        parser.add_option('-i', '--predictions', dest='predictionsf',
            help='(REQUIRED) predictions file to analyze')
        parser.add_option('--splits', dest='splits_base',
            help='file specifying cross-validation splits'
                 ' (if not provided, trains on full dataset)')
        parser.add_option('--viterbi', dest='viterbi_smoothing',
            action='store_true', default=False,
            help='use Viterbi smoothing')
        parser.add_option('--transitions-file', dest='transitionsf',
            help='(REQUIRED if --viterbi) file specifying static transition matrix')
        parser.add_option('--f-score-beta', dest='f_score_beta',
            type='float', default=2,
            help='Beta value for f measure (default %default)')
        parser.add_option('-l', '--logfile', dest='logfile',
            help='logfile path')
        (options, args) = parser.parse_args()

        if (not options.predictionsf):
            parser.error('Must provide --predictions')
        if options.viterbi_smoothing and not options.transitionsf:
            parser.error('--transitions-file must be provided with --viterbi')

        return options

    options = _cli()
    log.start(options.logfile)
    log.writeConfig([
        ('Predictions file', options.predictionsf),
        ('Cross-validation splits file', ('N/A' if options.splits_base is None else options.splits_base)),
        ('F-score beta', options.f_score_beta),
        ('Using Viterbi smoothing', options.viterbi_smoothing),
        ('Viterbi transitions file', ('N/A' if not options.viterbi_smoothing else options.transitionsf)),
    ], 'Token-level annotation evaluation')

    log.writeln('Reading predictions from %s...' % options.predictionsf)
    predictions = predictions_io.readSamplePredictions(options.predictionsf)
    log.writeln('Read {0:,} predictions.\n'.format(len(predictions)))

    if options.splits_base:
        log.writeln('Reading dataset splits from %s...' % options.splits_base)
        splits = document_splits.readDocumentSplits(options.splits_base)
        log.writeln('Found splits for {0:,} folds.\n'.format(len(splits)))

        fold_metrics = []
        for i in range(len(splits)):
            (train, dev, test) = splits[i]
            filtered_predictions = []
            test = set(test)
            for p in predictions:
                if p.doc_ID in test:
                    filtered_predictions.append(p)

            if options.viterbi_smoothing:
                transition_model = transitions_io.StaticTransitionMatrix.fromFile(
                    options.transitionsf
                )
                predictions_by_doc = labeling.indexPredictionsByDocAndLine(filtered_predictions)
                for (_, doc_predictions) in predictions_by_doc.items():
                    for (_, line_predictions) in doc_predictions.items():
                        decoding.viterbiSmooth(
                            line_predictions,
                            transition_model,
                            0.1
                        )

            metrics = getAnnotationMetrics(
                filtered_predictions,
                threshold=0.5,
                f_score_beta=options.f_score_beta
            )

            log.writeln('-- Fold {0}/{1} --'.format(i+1, len(splits)))
            log.indent()
            log.writeln('Num samples: {0:,}'.format(len(filtered_predictions)))
            log.writeln('Precision: {0:.3f}'.format(metrics.precision))
            log.writeln('Recall: {0:.3f}'.format(metrics.recall))
            log.writeln('F-1: {0:.3f}'.format(metrics.f1))
            log.writeln('F-2: {0:.3f}\n'.format(metrics.f2))
            log.unindent()

            fold_metrics.append(metrics)

        log.writeln('\n\n=== MACRO METRICS ===\n')
        log.indent()
        log.writeln('Num samples: {0:,}'.format(len(predictions)))
        log.writeln('Macro precision: {0:.3f}'.format(np.mean([m.precision for m in fold_metrics])))
        log.writeln('Macro recall: {0:.3f}'.format(np.mean([m.recall for m in fold_metrics])))
        log.writeln('Macro F-1: {0:.3f}'.format(np.mean([m.f1 for m in fold_metrics])))
        log.writeln('Macro F-2: {0:.3f}\n'.format(np.mean([m.f2 for m in fold_metrics])))
        log.unindent()

    metrics = getAnnotationMetrics(
        predictions,
        threshold=0.5,
        f_score_beta=options.f_score_beta
    )

    log.writeln('\n\n=== MICRO METRICS ===\n')
    log.indent()
    log.writeln('Num samples: {0:,}'.format(len(predictions)))
    log.writeln('Micro precision: {0:.3f}'.format(metrics.precision))
    log.writeln('Micro recall: {0:.3f}'.format(metrics.recall))
    log.writeln('Micro F-1: {0:.3f}'.format(metrics.f1))
    log.writeln('Micro F-2: {0:.3f}\n'.format(metrics.f2))
    log.unindent()

    log.stop()
