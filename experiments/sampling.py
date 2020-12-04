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
Library for sampling training data
'''

import numpy as np

def sampleTrainingData(dataset, positive_fraction, negative_ratio, random_seed=None):
    '''
    @param dataset             A pre-loaded data.dataset.Dataset object
    @param positive_fraction   Fraction (0-1] of positive samples to use
    @param negative_ratio      Maximum ratio of negative:positive samples
                               (None uses all negative samples)
    @param random_seed         For replicability

    @returns (positive_sample_set, negative_sample_set)
    '''

    if random_seed:
        np.random.seed(random_seed)
    
    all_positive_samples = list(dataset.positive_sample_IDs)
    all_negative_samples = list(dataset.negative_sample_IDs)

    num_sampled_positive = int(positive_fraction * len(all_positive_samples))
    if (not negative_ratio is None):
        num_sampled_negative = int(negative_ratio * num_sampled_positive)
    else:
        num_sampled_negative = len(all_negative_samples)

    np.random.shuffle(all_positive_samples)
    np.random.shuffle(all_negative_samples)

    sampled_positive = all_positive_samples[:num_sampled_positive]
    sampled_negative = all_negative_samples[:num_sampled_negative]

    return (
        sampled_positive,
        sampled_negative
    )
