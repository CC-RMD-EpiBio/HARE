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
import visualization.utils
from hedgepig_logger import log

def plotScoreDistribution(probabilities, figsize=(3,3), outf=None, font_size=15):
    font = {
        'family'  : 'sans-serif',
        'size'    : font_size
    }
    matplotlib.rc('font', **font)

    (fig, ax) = plt.subplots(figsize=figsize)

    plotScoreDistributionSubfigure(probabilities, ax)

    plt.xlabel('Relevance score')
    plt.ylabel('# Samples')

    plt.savefig(outf, format='png', bbox_inches='tight')
    plt.close()

def plotScoreDistributionSubfigure(probabilities, ax, color=None):
    (values, _, _) = ax.hist(
        probabilities,
        bins=100,
        range=[0,1],
        density=True,
        color=color
    )

    max_pow = int(np.ceil(np.log10(np.max(values))))
    ticks = [(10**(i+1)) for i in range(max_pow)]
    ax.set_yscale('log')
    ax.set_yticks(ticks)
    ax.set_yticklabels([
        '{0:,}'.format(tick)
            for tick in ticks
    ])

if __name__ == '__main__':
    def _cli():
        import optparse
        parser = optparse.OptionParser(usage='Usage: %prog PREDICTIONS [PREDICTIONS_2 [...]]')
        parser.add_option('-f', '--file-map', dest='file_mapf',
            help='(REQUIRED) visualization file map')
        parser.add_option('-d', '--datasets', dest='datasets',
            help='(REQUIRED) comma-separated list of datasets to visualize output for')
        parser.add_option('-o', '--output', dest='outf',
            help='(REQUIRED) output path for .png file')
        (options, args) = parser.parse_args()

        if not options.outf:
            parser.error('Must supply --output')
        elif not options.file_mapf:
            parser.error('Must supply --file-map')
        elif not options.datasets:
            parser.error('Must supply --datasets')

        options.datasets = [s.strip() for s in options.datasets.split(',')]

        return options

    options = _cli()

    log.writeln('Loading file map from %s...' % options.file_mapf)
    file_map = visualization.utils.loadFileMap(config=None, fpath=options.file_mapf)
    log.writeln('Mapped {0:,} annotation sets.\n'.format(len(file_map)))

    for ds in options.datasets:
        log.writeln('Reading predictions from %s...' % file_map[ds])
        predictions = predictions_io.readSamplePredictions(file_map[ds])
        log.writeln('Read {0:,} predictions.\n'.format(len(predictions)))

        probabilities = [p.positive_probability for p in predictions]
        plotScoreDistribution(probabilities, figsize=(10,6), outf=options.outf)
