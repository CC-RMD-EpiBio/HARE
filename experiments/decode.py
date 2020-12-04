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
'''

from hedgepig_logger import log
import numpy as np
from evaluation import predictions_io
from evaluation import transitions_io
from evaluation import decoding

if __name__ == '__main__':
    def _cli():
        import optparse
        parser = optparse.OptionParser(usage='Usage: %prog')
        parser.add_option('-l', '--logfile', dest='logfile',
            help='name of file to write log contents to (empty for stdout)',
            default=None)
        parser.add_option('-o', '--output', dest='output_f',
            help='(required) path to write decoded predictions to')

        token = optparse.OptionGroup(parser, 'Token scoring settings')
        token.add_option('--token-predictions', dest='token_predictions_f',
            help='file with token level probabilities')

        transition = optparse.OptionGroup(parser, 'Transition socring settings')
        transition.add_option('--transition-model-type', dest='transition_model_type',
            type='choice', choices=['static', 'dynamic'])
        transition.add_option('--transitions-file', dest='transitions_f',
            help='file with transition information')

        parser.add_option_group(token)
        parser.add_option_group(transition)
        (options, args) = parser.parse_args()

        if not options.output_f:
            parser.error('Must provide --output <FILE>')
        elif not options.token_predictions_f:
            parser.error('Must provide --token-predictions <FILE>')
        elif not options.transitions_f:
            parser.error('Must provide --transitions-file')

        return options
    options = _cli()
    log.start(options.logfile)
    log.writeConfig([
        ('Output file', options.output_f),
        ('Token probability settings', [
            ('Token predictions file', options.token_predictions_f),
        ]),
        ('Transition model settings', [
            ('Transition model type', options.transition_model_type),
            ('Transitions file', options.transitions_f),
        ]),
    ], 'Decoding experiment')

    log.writeln('Reading set of token-level predictions from %s...' % options.token_predictions_f)
    token_predictions = predictions_io.readSamplePredictions(options.token_predictions_f)
    # group together token predictions by file
    token_predictions_by_file = {}
    for t_pred in token_predictions:
        if not t_pred.doc_ID in token_predictions_by_file:
            token_predictions_by_file[t_pred.doc_ID] = []
        token_predictions_by_file[t_pred.doc_ID].append(t_pred)
    log.writeln('Read {0:,} predictions for {1:,} documents.\n'.format(
        len(token_predictions), len(token_predictions_by_file)
    ))

    if options.transition_model_type == 'static':
        log.writeln('Loading static transition matrix from %s...' % options.transitions_f)
        transition_model = transitions_io.StaticTransitionMatrix.fromFile(options.transitions_f)
        log.writeln('Done.\n')
    else:
        raise Exception('Not implemented yet!')

    log.writeln('Decoding document-level sequences...')
    log.track('  >> Processed {0}/{1:,} documents'.format(
        '{0:,}', len(token_predictions_by_file)
    ))
    with open(options.output_f, 'w') as stream:
        for (doc_ID, prediction_sequence) in token_predictions_by_file.items():
            binomial_prob_sequence = [
                sample.positive_probability
                    for sample in prediction_sequence
            ]
            _, decoded_sequence, conditional_probs = viterbi(
                binomial_prob_sequence,
                transition_model,
                [0.9, 0.1]
            )
            for i in range(len(prediction_sequence)):
                predictions_io.writeSamplePrediction(
                    stream,
                    prediction_sequence[i],
                    (
                        conditional_probs[i]
                            if decoded_sequence[i] == 1
                            else (1-conditional_probs[i])
                    )
                )
            log.tick()
    log.flushTracker()

    log.stop()
