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
Methods for turning continuous scores into discrete labels
'''

import numpy as np
from types import SimpleNamespace

def indexPredictionsByDocAndLine(predictions):
    '''Assumes tokens within a line are written to the predictions
    file in linear order.
    '''
    by_doc = {}
    for p in predictions:
        if not p.doc_ID in by_doc:
            by_doc[p.doc_ID] = {}
        if not p.line_index in by_doc[p.doc_ID]:
            by_doc[p.doc_ID][p.line_index] = []
        by_doc[p.doc_ID][p.line_index].append(p)
    return by_doc

def assignCorpusSegmentsAndLabels(indexed_predictions, threshold, num_blanks, value_getter=None):
    '''Given a set of pre-indexed token-level predictions for a full dataset,
    and configuration settings for segmentation, assign segmentation/labels
    for each document/line in turn.
    
    @returns
    indexed_labeled_corpus :: dictionary of the form
    {
        doc ID : (
            {
                line_index : (
                    labeled_tokens,
                    line_statistics
                )
            },
            doc_statistics
        )
    }
    '''
    indexed_labeled_corpus = {}
    for (doc_ID, doc_predictions) in indexed_predictions.items():
        (doc_line_outputs, doc_statistics) = assignDocumentSegmentsAndLabels(
            doc_predictions,
            threshold=threshold,
            num_blanks=num_blanks,
            value_getter=value_getter
        )
        indexed_labeled_corpus[doc_ID] = (
            doc_line_outputs,
            doc_statistics
        )
    return indexed_labeled_corpus


def assignDocumentSegmentsAndLabels(doc_predictions, threshold, num_blanks, value_getter=None):
    doc_statistics = SimpleNamespace(
        num_segments = 0,
        num_tokens = 0,
        segment_lengths = []
    )
    doc_line_outputs = {}
    for (line_ix, line_predictions) in doc_predictions.items():
        (labeled_tokens, line_statistics) = assignLineSegmentsAndLabels(
            line_predictions,
            threshold=threshold,
            num_blanks=num_blanks,
            value_getter=value_getter
        )
        doc_line_outputs[line_ix] = (
            labeled_tokens,
            line_statistics
        )
        doc_statistics.num_segments += line_statistics.num_segments
        doc_statistics.num_tokens += line_statistics.num_tokens
        doc_statistics.segment_lengths.extend(line_statistics.segment_lengths)
    
    return (doc_line_outputs, doc_statistics)
    

def assignLineSegmentsAndLabels(line_predictions, threshold, num_blanks, value_getter=None):
    '''Given a set of token-level predictions for a line and configuration settings
    for segmentation, identifies contiguous segments and assigns BILOU-style labels
    to each token.
    
    @returns
    labeled_tokens :: list of (<str:token>, <str:label>) pairs
    statistics :: namespace with the following statistics:
        .num_segments
        .num_tokens
        .segment_lengths
    '''
    if value_getter is None:
        value_getter = lambda s: s.positive_probability

    statistics = SimpleNamespace(
        num_segments = 0,
        num_tokens = 0,
        segment_lengths = []
    )
    labeled_tokens = []

    in_segment, segment_length, blanks_so_far = False, 0, 0
    blank_buffer = []

    for sample in line_predictions:
        sample_value = value_getter(sample)
        # does it meet the token-level threshold?
        if sample_value >= threshold:
            # first, flush any buffered blanks and reset counter to 0
            blanks_so_far = 0
            for t in blank_buffer:
                labeled_tokens.append((t, 'I'))
                # include blanks that are collapsed into a segment
                # in the token count
                statistics.num_tokens += 1
            blank_buffer = []
            # now deal with this token
            if not in_segment:
                in_segment = True
                segment_length = 1
                tag = 'B'
            else:
                tag = 'I'
                segment_length += 1
            labeled_tokens.append((sample.token, tag))
            statistics.num_tokens += 1
        else:
            if in_segment:
                blanks_so_far += 1
                segment_length += 1
                blank_buffer.append(sample.token)
                if blanks_so_far > num_blanks:
                    in_segment = False
                    statistics.num_segments += 1
                    statistics.segment_lengths.append(
                        segment_length - blanks_so_far
                    )
                    # flush any buffered blanks as Os and reset counter to 0
                    for t in blank_buffer:
                        labeled_tokens.append((t, 'O'))
                    blank_buffer = []
                    blanks_so_far = 0
            else:
                labeled_tokens.append((sample.token, 'O'))

    if in_segment:
        statistics.num_segments += 1
        statistics.segment_lengths.append(
            segment_length - blanks_so_far
        )

    # any buffered blanks at this point are Os
    for t in blank_buffer:
        labeled_tokens.append((t, 'O'))

    # relabel to BILOU style
    relabeled_tokens = []
    for i in range(len(labeled_tokens)):
        (t, lbl) = labeled_tokens[i]
        # replace any Bs followed by an O with a U
        if (
            lbl == 'B'
            and 
            (
                (i == (len(labeled_tokens) - 1))
                or (labeled_tokens[i+1][1] == 'O')
            )
        ):
            lbl = 'U'
        # replace all final Is with an L
        elif (
            lbl == 'I'
            and
            (
                (i == (len(labeled_tokens) - 1))
                or (labeled_tokens[i+1][1] == 'O')
            )
        ):
            lbl = 'L'

        relabeled_tokens.append(
            (t, lbl)
        )

    return (
        relabeled_tokens,
        statistics
    )

def binarizeProbabilities(probabilities, threshold):
    binary_preds = []
    for p in probabilities:
        if p >= threshold:
            binary_preds.append(1)
        else:
            binary_preds.append(0)
    return binary_preds


def summaryAnnotationStatistics(labeled_corpus):
    '''Takes as input a corpus processed by assignCorpusSegmentsAndLabels
    and calculates descriptive statistics about its annotations.
    '''
    corpus_num_segments, corpus_segment_lengths = [], []
    for (_, doc_statistics) in labeled_corpus.values():
        corpus_num_segments.append(doc_statistics.num_segments)
        corpus_segment_lengths.extend(doc_statistics.segment_lengths)
    statistics = SimpleNamespace(
        num_segments_mean = np.mean(corpus_num_segments),
        num_segments_std = np.std(corpus_num_segments),
        num_segments_min = np.min(corpus_num_segments),
        num_segments_max = np.max(corpus_num_segments),
        segment_lengths_mean = (
            np.mean(corpus_segment_lengths) if len(corpus_segment_lengths) > 0 else 0
        ),
        segment_lengths_std = (
            np.std(corpus_segment_lengths) if len(corpus_segment_lengths) > 0 else 0
        ),
        segment_lengths_min = (
            np.min(corpus_segment_lengths) if len(corpus_segment_lengths) > 0 else 0
        ),
        segment_lengths_max = (
            np.max(corpus_segment_lengths) if len(corpus_segment_lengths) > 0 else 0
        ),
    )
    return statistics


if __name__ == '__main__':
    from hedgepig_logger import log
    from . import predictions_io
    from data import exclusion

    def _cli():
        import optparse
        parser = optparse.OptionParser(usage='Usage: %prog')
        parser.add_option('-i', '--predictions', dest='predictionsf',
            help='(REQUIRED) predictions file to analyze')
        parser.add_option('--threshold', dest='threshold',
            type='float', default=0.5,
            help='binarization threshold (default %default)')
        parser.add_option('--num-blanks', dest='num_blanks',
            type='int', default=0,
            help='number of blanks to collapse for sequential segments'
                 ' (default %default)')
        exclusion.addCLIExclusionOptions(parser)
        parser.add_option('-l', '--logfile', dest='logfile',
            help='logfile path')
        (options, args) = parser.parse_args()

        if (not options.predictionsf):
            parser.error('Must provide --predictions')

        return options

    options = _cli()
    log.start(options.logfile)
    log.writeConfig([
        ('Predictions file', options.predictionsf),
        ('Binarization threshold', options.threshold),
        ('Number of blanks to collapse', options.num_blanks),
        exclusion.logCLIExclusionOptions(options),
    ], 'Statistics calculation for segmentation of predictions')

    log.writeln('Reading predictions from %s...' % options.predictionsf)
    predictions = predictions_io.readSamplePredictions(options.predictionsf, verbose=True)
    log.writeln('Read {0:,} predictions.\n'.format(len(predictions)))

    log.writeln('Indexing predictions by document and line...')
    predictions_by_doc = indexPredictionsByDocAndLine(predictions)
    log.writeln('Indexed predictions to {0:,} documents.\n'.format(len(predictions_by_doc)))

    exclusions = exclusion.getExclusionSet(options, verbose=True)

    for doc_ID in exclusions:
        if doc_ID in predictions_by_doc:
            del(predictions_by_doc[doc_ID])

    log.writeln('Number documents after exclusion: {0:,}\n'.format(len(predictions_by_doc)))

    log.writeln('Assigning segmentation labels...')
    segment_labeled_predictions = assignCorpusSegmentsAndLabels(
        predictions_by_doc,
        threshold=options.threshold,
        num_blanks=options.num_blanks
    )
    log.writeln('Calculating statistics...')
    stats = summaryAnnotationStatistics(segment_labeled_predictions)

    log.writeln()
    log.writeln('--- Statistics ---')
    log.writeln('Number segments per document')
    log.indent()
    log.writeln('Mean: {0:.1f}'.format(stats.num_segments_mean))
    log.writeln('Std dev: {0:.1f}'.format(stats.num_segments_std))
    log.writeln('Min: {0:,}'.format(stats.num_segments_min))
    log.writeln('Max: {0:,}'.format(stats.num_segments_max))
    log.unindent()
    log.writeln('Segment length')
    log.indent()
    log.writeln('Mean: {0:.1f}'.format(stats.segment_lengths_mean))
    log.writeln('Std dev: {0:.1f}'.format(stats.segment_lengths_std))
    log.writeln('Min: {0:,}'.format(stats.segment_lengths_min))
    log.writeln('Max: {0:,}'.format(stats.segment_lengths_max))
    log.unindent()

    log.stop()
