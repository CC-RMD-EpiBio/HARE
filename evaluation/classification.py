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
from . import scoring

def getClassifier(options):
    if options.classification_method == 'Binning':
        return BinningClassifier(
            thresholds=options.binning_thresholds
        )
    else:
        raise ValueError('Classifier "%s" unconfigured' % options.classification_method)

class BinningClassifier:
    
    @staticmethod
    def parseThresholdsString(thresholds_str):
        thresholds = {}
        for substr in thresholds_str.split(','):
            (key, thresh) = substr.split(':')
            thresholds[key.strip()] = int(thresh)
        return thresholds

    @staticmethod
    def parseColorsString(colors_str):
        colors = {}
        for substr in colors_str.split(','):
            (key, color_pair) = substr.split(':')
            (bg_color, fg_color) = color_pair.split('|')
            colors[key.strip()] = (bg_color.strip(), fg_color.strip())
        return colors

    def __init__(self, thresholds):
        self.thresholds = sorted(
            thresholds.items(),
            key=lambda pair: pair[1],
            reverse=True
        )

    def __call__(self, score):
        for (_class, threshold) in self.thresholds:
            if score >= threshold:
                return _class
        return None

    def getConfiguration(self, colors, scorer):
        return [
            SimpleNamespace(
                name=_class,
                value=scorer.renderScore(threshold),
                bg_color=colors[_class][0],
                fg_color=colors[_class][1]
            )
                for (_class, threshold) in self.thresholds
        ]
