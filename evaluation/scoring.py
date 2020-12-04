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
Classes for scoring model predictions
'''

import os
from . import labeling

CACHE_BASE = 'cached_scoring'

def scorerChoices():
    return [
        SumTokenScores.NAME,
        CountSegmentsAndTokens.NAME,
        DensityScorer.NAME,
    ]

def scorerDefault():
    return 'CountSegmentsAndTokens'

def getScorer(options):
    if options.model_scorer == SumTokenScores.NAME:
        return SumTokenScores(
            min_probability=options.min_probability,
            use_label_instead=options.use_label_instead
        )
    elif options.model_scorer == CountSegmentsAndTokens.NAME:
        return CountSegmentsAndTokens(
            threshold=options.threshold,
            num_blanks=options.num_blanks,
            use_label_instead=options.use_label_instead
        )
    elif options.model_scorer == DensityScorer.NAME:
        return DensityScorer(
            threshold=options.threshold,
            use_label_instead=options.use_label_instead
        )
    else:
        raise KeyError('Unknown scorer type "%s"' % options.model_scorer)

def cleanAnnotSet(annot_set):
    return (
        annot_set.replace(' ', '_')
    )

class SumTokenScores:
    '''Given a list of estimated positive probabilities,
    assigns a ranking by summing together the positive
    probabilities in each document.
    '''
    NAME = 'SumTokenScores'

    def __init__(self, min_probability=0, use_label_instead=False):
        self.min_probability = min_probability
        self.use_label_instead = use_label_instead
        
    def score(self, sample_predictions, pre_labeled_predictions=None,
            annot_set_for_caching=None, additional_settings_for_caching=None):
        if annot_set_for_caching:
            cache = ScorerOutputCache(
                cache_file = '%s.%s.SumTokenScores.cache' % (CACHE_BASE, annot_set_for_caching)
            )
        else:
            cache = None
            
        document_scores = {}
        
        if self.use_label_instead:
            value_getter = lambda s: s.label
        else:
            value_getter = lambda s: s.positive_probability

        for sample in sample_predictions:
            if value_getter(sample) > self.min_probability:
                document_scores[sample.doc_ID] = \
                    document_scores.get(sample.doc_ID, 0) \
                    + value_getter(sample)

        return document_scores

    def renderScore(self, score):
        return '{0:.3f}'.format(score)

class DensityScorer:
    '''Given a list of estimated positive probabilities,
    assigns a ranking by binarizing at a specified threshold
    and measuring fraction of tokens that are relevant.
    '''
    NAME = 'DensityScorer'

    def __init__(self, threshold=0.5, use_label_instead=False):
        self.threshold = threshold
        self.use_label_instead = use_label_instead

    def score(self, sample_predictions, pre_labeled_predictions=None,
            annot_set_for_caching=None, additional_settings_for_caching=None):
        if annot_set_for_caching:
            cache = ScorerOutputCache(
                cache_file = '%s.%s.DensityScorer.cache' % (CACHE_BASE, annot_set_for_caching)
            )
        else:
            cache = None
            
        document_relevant, document_total = {}, {}

        if self.use_label_instead:
            value_getter = lambda s: s.label
        else:
            value_getter = lambda s: s.positive_probability
        
        for sample in sample_predictions:
            document_total[sample.doc_ID] = document_total.get(sample.doc_ID, 0) + 1
            if value_getter(sample) >= self.threshold:
                document_relevant[sample.doc_ID] = document_relevant.get(sample.doc_ID, 0) + 1

        document_scores = {}
        for doc_ID in document_total.keys():
            document_scores[doc_ID] = (
                document_relevant.get(doc_ID, 0)/
                document_total[doc_ID]
            )

        return document_scores

    def renderScore(self, score):
        return '{0:.2f}'.format(100*score)

class CountSegmentsAndTokens:
    '''Given a list of labeled predictions, assigns "gold" relevance
    scores using the following formula:
       10000 * number of contiguous Mobility segments
       + number of Mobility tokens

    Assumes that labeled_predictions are provided in doc_ID order.
    '''
    SEGMENT_VALUE = 1000000
    TOKEN_VALUE = 1
    NAME = 'CountSegmentsAndTokens'

    def __init__(self, threshold=0.5, num_blanks=0, use_label_instead=False):
        self.threshold = threshold
        self.num_blanks = num_blanks
        self.use_label_instead = use_label_instead

    def score(self, sample_predictions, pre_labeled_predictions=None,
            annot_set_for_caching=None, additional_settings_for_caching=None):
        if annot_set_for_caching:
            cache = ScorerOutputCache(
                cache_file = '%s.%s.CountSegmentsAndTokens.cache' % (CACHE_BASE, cleanAnnotSet(annot_set_for_caching))
            )
            cache_key = {
                'threshold': self.threshold,
                'num_blanks': self.num_blanks,
                'use_label_instead': self.use_label_instead
            }
            if additional_settings_for_caching:
                for (key, value) in additional_settings_for_caching.items():
                    cache_key[key] = value
            document_scores = cache[cache_key]
        else:
            cache = None
            document_scores = None
        
        if document_scores is None:
            document_scores = {}

            score_func = lambda num_segments, num_tokens: (CountSegmentsAndTokens.SEGMENT_VALUE * num_segments) + (CountSegmentsAndTokens.TOKEN_VALUE * num_tokens)

            predictions_by_doc_and_line = labeling.indexPredictionsByDocAndLine(sample_predictions)

            if self.use_label_instead:
                value_getter = lambda s: s.label
            else:
                value_getter = None

            for (doc_ID, doc_predictions) in predictions_by_doc_and_line.items():
                if pre_labeled_predictions and doc_ID in pre_labeled_predictions:
                    (_, doc_statistics) = pre_labeled_predictions[doc_ID]
                else:
                    (doc_line_outputs, doc_statistics) = labeling.assignDocumentSegmentsAndLabels(
                        doc_predictions,
                        threshold=self.threshold,
                        num_blanks=self.num_blanks,
                        value_getter=value_getter
                    )
                    if pre_labeled_predictions:
                        pre_labeled_predictions[doc_ID] = (doc_line_outputs, doc_statistics)
                document_scores[doc_ID] = score_func(
                    doc_statistics.num_segments,
                    doc_statistics.num_tokens
                )
            
            if cache:
                cache[cache_key] = document_scores

        return document_scores

    def renderScore(self, score):
        num_segments = score // CountSegmentsAndTokens.SEGMENT_VALUE
        num_tokens = (score % CountSegmentsAndTokens.SEGMENT_VALUE) // CountSegmentsAndTokens.TOKEN_VALUE

        return '{0:,}/{1:,}'.format(int(num_segments), int(num_tokens))


class ScorerOutputCache:
    
    def __init__(self, cache_file):
        self.cache = {}
        self.cache_file = cache_file
        
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r') as stream:
                for line in stream:
                    (settings, doc_scores) = self.deserializeEntry(line.strip())
                    self.cache[settings] = doc_scores
    
    def __getitem__(self, settings):
        settings = self.settingsToKey(settings)
        return self.cache.get(settings, None)
    
    def __setitem__(self, settings, doc_scores):
        settings = self.settingsToKey(settings)
        self.cache[settings] = doc_scores
        with open(self.cache_file, 'a') as stream:
            stream.write(self.serializeEntry(settings, doc_scores))
            stream.write('\n')
    
    def settingsToKey(self, settings):
        # settings are internally keyed in alpha order, to force the
        # settings dict to this order
        if type(settings) is dict:
            settings = list(settings.items())
        settings = sorted(settings, key=lambda key_value: key_value[0])
        # use serialized settings as keys within the cache
        return self.serializeKey(settings)
    
    def deserializeEntry(self, line):
        (key, value) = line.split('\t')
        
        # the key remains serialized, since it never needs to be used
        # outside the cache
        key = key
        
        # parse the value, specifying the scores assigned to each document
        doc_scores = value.split('|')
        doc_scores = [ds.split(':') for ds in doc_scores]
        doc_scores = {
            doc_ID : float(score)
                for (doc_ID, score)
                in doc_scores
        }
        
        return (key, doc_scores)
    
    def serializeKey(self, settings):
        # serialize the settings as:
        # <key_1>:<value_1>|<key_2>:<value_2>|...
        settings_str = '|'.join([
            '{0}:{1}'.format(key, value)
                for (key, value) in settings
        ])
        return settings_str
        
    def serializeEntry(self, key, doc_scores):
        # serialize the document scores as:
        # <doc_ID_1>:<score_1>|<doc_ID_2>:<score_2>|...
        scores_str = '|'.join([
            '{0}:{1}'.format(key, value)
                for (key, value) in doc_scores.items()
        ])
        
        serialized = '{0}\t{1}'.format(key, scores_str)
        return serialized
