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
Document ranking evaluation
'''

from hedgepig_logger import log
import numpy as np
import scipy.stats
from . import predictions_io
from . import scoring

class RankedItem:
    
    ID = None
    model_rank = None
    gold_rank = None
    model_score = None
    gold_score = None
    
    def __init__(self, model_rank=None, gold_rank=None, model_score=None,
            gold_score=None, ID=None):
        self.ID = ID
        self.model_rank = model_rank
        self.gold_rank = gold_rank
        self.model_score = model_score
        self.gold_score = gold_score

def rankingComparison(model_scores, gold_scores, order_by_gold=False):
    key_set = list(gold_scores.keys())

    # flip sign in order to rank from highest score down to lowest
    model_scores = -1 * np.array([
        model_scores.get(doc_ID, 0)
            for doc_ID in key_set
    ])
    gold_scores = -1 * np.array([
        gold_scores[doc_ID]
            for doc_ID in key_set
    ])

    model_ranking = scipy.stats.rankdata(model_scores)
    gold_ranking = scipy.stats.rankdata(gold_scores)

    model_index_order = np.argsort(model_scores)
    gold_index_order = np.argsort(gold_scores)

    if order_by_gold:
        index_order = gold_index_order
    else:
        index_order = model_index_order

    ranked = []
    for ix in index_order:
        ranked.append(RankedItem(
            ID=key_set[ix],
            model_rank=int(model_ranking[ix]),
            gold_rank=int(gold_ranking[ix]),
            model_score=-model_scores[ix],
            gold_score=-gold_scores[ix]
        ))

    (rho, _) = scipy.stats.spearmanr(model_scores, gold_scores)
    return (ranked, rho)

def rankUnlabeledDataByModelScore(model_scores):
    key_set = list(model_scores.keys())
    
    # flip sign in order to rank from highest score to lowest
    model_scores = -1 * np.array([
        model_scores[doc_ID]
            for doc_ID in key_set
    ])
    
    model_ranking = scipy.stats.rankdata(model_scores)
    model_index_order = np.argsort(model_scores)
    
    ranked = []
    for ix in model_index_order:
        ranked.append(RankedItem(
            ID=key_set[ix],
            model_rank=int(model_ranking[ix]),
            model_score=-model_scores[ix]
        ))
    
    return ranked

if __name__ == '__main__':
    def _cli():
        import optparse
        parser = optparse.OptionParser(usage='Usage: %prog PREDICTIONS_FILE')
        parser.add_option('--model-scorer', dest='model_scorer',
            type='choice', choices=['CountSegmentsAndTokens', 'SumTokenScores'])
        parser.add_option('--min-probability', dest='min_probability',
            type='float', default=0.,
            help='minimum positive probability for scoring (default %default)')
        parser.add_option('--threshold', dest='threshold',
            type='float', default=0.5,
            help='threshold for positive/negative decision function')
        parser.add_option('--num-blanks', dest='num_blanks',
            type='int', default=0,
            help='number of blank tokens to allow before stopping Mobility segment')
        parser.add_option('-l', '--logfile', dest='logfile',
            help='name of file to write log contents to (empty for stdout)',
            default=None)
        (options, args) = parser.parse_args()

        def _bail(msg):
            print(msg)
            print('')
            parser.print_help()
            exit()

        if len(args) != 1:
            parser.error('Must provide PREDICTIONS_FILE')

        return args, options
    (preds_f,), options = _cli()
    log.start(options.logfile)
    log.writeConfig([
        ('Predictions file', preds_f),
        ('Model scorer', options.model_scorer),
        ('SumTokenScores settings', [
            ('Minimum positive probability', options.min_probability),
        ]),
        ('CountSegmentsAndTokens settings', [
            ('Positive score threshold', options.threshold),
            ('Number of blanks', options.num_blanks),
        ]),
    ], 'Document ranking evaluation (strict)')

    log.writeln('Reading set of predictions from %s...' % preds_f)
    sample_predictions = predictions_io.readSamplePredictions(preds_f)
    log.writeln('Read {0:,} predictions.\n'.format(len(sample_predictions)))

    log.writeln('Calculating gold scores...')
    gold_scorer = scoring.CountSegmentsAndTokens(
        threshold=1.0,
        num_blanks=options.num_blanks,
        use_label_instead=True
    )
    gold_scores = gold_scorer.score(sample_predictions)
    log.writeln('Scored {0:,} documents.\n'.format(len(gold_scores)))

    log.writeln('Calculating model scores...')
    model_scorer = scoring.getScorer(options)
    model_scores = model_scorer.score(sample_predictions)
    log.writeln('Scored {0:,} documents.\n'.format(len(model_scores)))

    log.writeln('Calculating rank correlation coefficient...')
    (ranked, rho) = rankingComparison(model_scores, gold_scores, order_by_gold=True)
    log.indent()
    log.writeln('Detailed report')
    for doc in ranked:
        log.writeln('  Gold {0:3d} ({1:10d}) -- Pred {2:3d} ({3:10d})  [{4}]'.format(
            doc.gold_ranking,
            doc.gold_score,
            doc.model_ranking,
            doc.model_score,
            doc.ID
        ))
    log.unindent()
    log.writeln('  Spearman\'s rho: {0:.3f}'.format(rho))

    log.stop()
