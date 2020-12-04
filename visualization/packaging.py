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

from types import SimpleNamespace
from io import BytesIO
import base64

def packageAnnotSets(file_map, annot_set, annot_set_info=None, doc_ID=None):
    annot_sets = []
    for label in file_map.keys():
        if annot_set_info and doc_ID:
            annot_set_files = annot_set_info[label]
            include = (doc_ID in annot_set_files)
        else:
            include = True
        if include:
            annot_sets.append(SimpleNamespace(
                name=label,
                selected=(label==annot_set)
            ))
    return annot_sets

def getPostProcessingConfiguration(getter, config):
    # TODO: pass around order_by_gold and show_gold_classes variables statefully between view and list
    settings = SimpleNamespace()

    settings.threshold = float(getter(
        'threshold',
        config['Predictions']['DefaultHighlightThreshold']
    ))
    settings.num_blanks = int(getter(
        'num_blanks',
        config['Predictions']['DefaultNumBlanks']
    ))
    settings.min_probability = float(getter(
        'min_probability',
        config['Predictions']['DefaultMinProbability']
    ))
    settings.viterbi_smoothing = getter(
        'viterbi_smoothing',
        False
    )

    # correct boolean values as needed
    if settings.viterbi_smoothing == "False":
        settings.viterbi_smoothing = False

    return settings

def passPostProcessingConfiguration(settings):
    return {
        'min_probability': settings.min_probability,
        'threshold': settings.threshold,
        'num_blanks': settings.num_blanks,
        'viterbi_smoothing': settings.viterbi_smoothing
    }

def getRankingConfiguration(getter, config):
    ranking_settings = SimpleNamespace()

    ranking_settings.gold_scorer = getter(
        'gold_scorer',
        config['Ranking']['DefaultGoldScorer']
    )
    ranking_settings.gold_score_description = config['Ranking']['{0} Score Description'.format(ranking_settings.gold_scorer)]
    ranking_settings.model_scorer = getter(
        'model_scorer',
        config['Ranking']['DefaultModelScorer']
    )
    ranking_settings.model_score_description = config['Ranking']['{0} Score Description'.format(ranking_settings.model_scorer)]
    ranking_settings.order_by_gold = getter(
        'order_by_gold',
        False
    )
    ranking_settings.classification_method = getter(
        'classification_method',
        config['Classification']['Method']
    )
    ranking_settings.classification_thresholds = getter(
        'classification_thresholds',
        config['Classification']['BinningThresholds']
    )
    ranking_settings.classification_colors = getter(
        'classification_colors',
        config['Classification']['BinningColors']
    )
    ranking_settings.class_by_gold = getter(
        'class_by_gold',
        False
    )

    # correct boolean values as needed
    if ranking_settings.order_by_gold == "False":
        ranking_settings.order_by_gold = False
    if ranking_settings.class_by_gold == "False":
        ranking_settings.class_by_gold = False

    return ranking_settings

def passRankingConfiguration(ranking_settings):
    return {
        'gold_scorer': ranking_settings.gold_scorer,
        'model_scorer': ranking_settings.model_scorer,
        'order_by_gold': ranking_settings.order_by_gold,
        'class_by_gold': ranking_settings.class_by_gold,
        'classification_method': ranking_settings.classification_method,
        'classification_thresholds': ranking_settings.classification_thresholds,
        'classification_colors': ranking_settings.classification_colors,
    }

def prettifyMetrics(metrics):
    if metrics:
        metrics.precision = '%.2f' % (100*metrics.precision)
        metrics.recall = '%.2f' % (100*metrics.recall)
        metrics.f1 = '%.2f' % (100*metrics.f1)
        metrics.f2 = '%.2f' % (100*metrics.f2)
        metrics.accuracy = '%.2f' % (100*metrics.accuracy)
        metrics.auc = '%.3f' % (metrics.auc)
        metrics.correct = '{0:,}'.format(metrics.correct)
        metrics.total = '{0:,}'.format(metrics.total)
        if metrics.support is None:
            metrics.support = 'N/A'
        else:
            metrics.support = '{0:,}'.format(metrics.support)
    return metrics

def prettifyStatistics(statistics):
    if statistics:
        statistics.num_segments_mean = '%.1f' % statistics.num_segments_mean
        statistics.num_segments_std = '%.1f' % statistics.num_segments_std
        statistics.num_segments_min = '{0:,}'.format(statistics.num_segments_min)
        statistics.num_segments_max = '{0:,}'.format(statistics.num_segments_max)
        statistics.segment_lengths_mean = '%.1f' % statistics.segment_lengths_mean
        statistics.segment_lengths_std = '%1.f' % statistics.segment_lengths_std
        statistics.segment_lengths_min = '{0:,}'.format(statistics.segment_lengths_min)
        statistics.segment_lengths_max = '{0:,}'.format(statistics.segment_lengths_max)
    return statistics

def prettifyLexicalizationStats(lex_stats):
    if lex_stats:
        lex_stats.num_filtered = '{0:,}'.format(lex_stats.num_filtered)
        if not (lex_stats.rho is None):
            lex_stats.rho = '%.2f' % lex_stats.rho
        for lex_item in lex_stats.lex_info:
            lex_item.model_lex_score = '%.1f' % (100*lex_item.model_lex_score)
            if not (lex_item.gold_lex_score is None): 
                lex_item.gold_lex_score = '%.1f' % (100*lex_item.gold_lex_score)
            lex_item.count = '{0:,}'.format(lex_item.count)
    return lex_stats

def renderImage(func, args, kwargs):
    stream = BytesIO()
    func(*args, outf=stream, **kwargs)
    stream.seek(0)
    base64_data = base64.b64encode(stream.getvalue())
    return base64_data.decode('utf8')
