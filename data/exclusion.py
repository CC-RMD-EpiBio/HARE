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

from hedgepig_logger import log

def getExclusionSet(options, verbose=False):
    exclusions = set()
    if options.exclude_undecided_fromf:
        if verbose:
            log.writeln('Loading exclusion %s doc IDs from %s...' % (
                options.exclude_undecided_type, options.exclude_undecided_fromf
            ))
        with open(options.exclude_undecided_fromf, 'r') as stream:
            for line in stream:
                (doc_type, doc_ID, lbl) = [s.strip() for s in line.split('\t')]
                if doc_type == options.exclude_undecided_type and lbl == 'Undecided':
                    exclusions.add('{0}.txt'.format(doc_ID))
        if verbose:
            log.writeln('Found {0:,} doc IDs to exclude.\n'.format(len(exclusions)))
    return exclusions

def addCLIExclusionOptions(parser):
    parser.add_option('--exclude-undecided-from', dest='exclude_undecided_fromf',
        help='file listing per-document ternary labels; if provided,'
             ' will exclude any documents labeled "Undecided" from'
             ' analysis.')
    parser.add_option('--exclude-undecided-type', dest='exclude_undecided_type',
        help='if used in conjunction with --exclude-undecided-from,'
             ' will only check for matches with files listed in the'
             ' labels file with this type (e.g., CE)')

def logCLIExclusionOptions(options):
    return ('Undecided exclusion settings', [
        ('Labels file', options.exclude_undecided_fromf),
        ('Doc type to exclude from', options.exclude_undecided_type),
    ])
