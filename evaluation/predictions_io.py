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
I/O methods for predictions
'''

from hedgepig_logger import log
from data.dataset import DataPoint

class SamplePrediction:
    def __init__(self, doc_ID, line_index, token, label, positive_probability):
        self.doc_ID = doc_ID
        self.line_index = line_index
        self.token = token
        self.label = label
        self.positive_probability = positive_probability

def writeSamplePrediction(stream, sample, prob):
    stream.write('%s\t%s\t%s\t%s\t%.4f\n' % (
        sample.doc_ID,
        sample.line_index,
        sample.token,
        sample.label,
        prob
    ))

def readSamplePredictions(f, verbose=False):
    predictions = []
    with open(f, 'r', errors='replace') as stream:
        if verbose:
            log.track('  >> Read {0:,} lines', writeInterval=1000)
            tick = lambda: log.tick()
        else:
            tick = lambda: None

        for line in stream:
            (
                doc_ID,
                line_index,
                token,
                label,
                prob
            ) = [s.strip() for s in line.split('\t')]

            predictions.append(SamplePrediction(
                doc_ID=doc_ID,
                line_index=int(line_index),
                token=token,
                label=int(label),
                positive_probability=float(prob)
            ))

            tick()

        if verbose:
            log.flushTracker()
    return predictions
