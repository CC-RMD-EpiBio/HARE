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
from data import exclusion
from data.dataset import Dataset
from evaluation import predictions_io
from evaluation import labeling
import visualization.utils
from hedgepig_logger import log

if __name__ == '__main__':
    def _cli():
        import optparse
        parser = optparse.OptionParser(usage='Usage: %prog PREDICTIONS')
        parser.add_option('-d', '--dataset', dest='dataset',
            help='(REQUIRED) .labels file for dataset to evaluate')
        exclusion.addCLIExclusionOptions(parser)
        parser.add_option('-l', '--logfile', dest='logfile',
            help='logfile')
        (options, args) = parser.parse_args()

        if (not options.dataset):
            parser.error('Must supply --dataset')

        return options
    options = _cli()
    log.writeConfig([
        ('Dataset file', options.dataset),
        exclusion.logCLIExclusionOptions(options),
    ], 'Dataset-level statistics')

    exclusions = exclusion.getExclusionSet(options, verbose=True)

    log.writeln('Loading dataset from %s...' % options.dataset)
    dataset = Dataset(
        labels_file=options.dataset,
        exclusion_IDs=exclusions
    )
    log.writeln('Loaded {0:,} samples.\n'.format(len(dataset)))

    log.writeln('Analyzing documents...')
    log.track('  >> Processed {0:,}/{1:,}', writeInterval=10)
    num_relevant_documents = 0
    l_num_tokens_total, l_num_segments, l_num_tokens = [], [], []
    for (doc_ID, by_line) in dataset.indexed.items():
        doc_num_tokens_total, doc_num_segments, doc_num_tokens = 0, 0, 0
        for (line_index, by_token) in by_line.items():
            line_tokens = [
                sample for (token_index, sample) in sorted(
                    by_token.items(),
                    key=lambda pair: pair[0]
                )
            ]
            (_, statistics) = labeling.assignLineSegmentsAndLabels(
                line_tokens,
                threshold=0.5,
                num_blanks=0,
                value_getter=lambda s: s.label
            )
            doc_num_tokens_total += len(line_tokens)
            doc_num_segments += statistics.num_segments
            doc_num_tokens += statistics.num_tokens
        l_num_tokens_total.append(doc_num_tokens_total)
        l_num_segments.append(doc_num_segments)
        l_num_tokens.append(doc_num_tokens)

        if doc_num_tokens > 0:
            num_relevant_documents += 1

        log.tick(len(dataset.indexed))
    log.flushTracker(len(dataset.indexed))
    
    log.writeln()
    log.writeln('--- Statistics ---\n')
    log.writeln('Number of documents: {0:,}'.format(len(dataset.indexed)))
    log.writeln('Number of relevant documents: {0:,}'.format(num_relevant_documents))
    log.writeln('Mean number tokens per documents: {0:,}'.format(np.mean(l_num_tokens_total)))
    log.writeln('Mean number relevant tokens per documents: {0:,}'.format(np.mean(l_num_tokens)))
    log.writeln('Mean number relevant segments per documents: {0:,}'.format(np.mean(l_num_segments)))
