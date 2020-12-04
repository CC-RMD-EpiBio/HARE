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
Methods for decoding state probabilities into a state sequence
(discriminative model approach)
'''

import numpy as np

def viterbi(binomial_prob_sequence, transition_prob_getter, initial_probs):
    '''Eternally indebted to JHU
    https://nbviewer.jupyter.org/gist/BenLangmead/7460513
    '''
    
    probability_matrix = np.zeros(
        shape=(2, len(binomial_prob_sequence)),
        dtype=np.float32
    )
    backtrace_matrix = np.zeros(
        shape=(2, len(binomial_prob_sequence)),
        dtype=np.int
    )

    # fill in first column of probability matrix
    probability_matrix[0,0] = (
        np.log(max(1 - binomial_prob_sequence[0], 1e-5))
        + np.log(initial_probs[0])
    )
    probability_matrix[1,0] = (
        np.log(max(binomial_prob_sequence[0], 1e-5))
        + np.log(initial_probs[1])
    )

    # fill in remained of prob and backtrace matrices
    for token_pos in range(1, len(binomial_prob_sequence)):
        # probability of ending up in state 0 here
        from_0_to_0 = (
            probability_matrix[0,token_pos-1]
            + np.log(transition_prob_getter(token_pos-1, token_pos, 0, 0))
            + np.log(max(1 - binomial_prob_sequence[token_pos], 1e-5))
        )
        from_1_to_0 = (
            probability_matrix[1,token_pos-1]
            + np.log(transition_prob_getter(token_pos-1, token_pos, 1, 0))
            + np.log(max(1 - binomial_prob_sequence[token_pos], 1e-5))
        )
        if from_0_to_0 > from_1_to_0:
            probability_matrix[0, token_pos] = from_0_to_0
            backtrace_matrix[0, token_pos] = 0
        else:
            probability_matrix[0, token_pos] = from_1_to_0
            backtrace_matrix[0, token_pos] = 1

        # probability of ending up in state 1 here
        from_0_to_1 = (
            probability_matrix[0,token_pos-1]
            + np.log(transition_prob_getter(token_pos-1, token_pos, 0, 1))
            + np.log(max(binomial_prob_sequence[token_pos], 1e-5))
        )
        from_1_to_1 = (
            probability_matrix[1,token_pos-1]
            + np.log(transition_prob_getter(token_pos-1, token_pos, 1, 1))
            + np.log(max(binomial_prob_sequence[token_pos], 1e-5))
        )
        if from_0_to_1 > from_1_to_1:
            probability_matrix[1, token_pos] = from_0_to_1
            backtrace_matrix[1, token_pos] = 0
        else:
            probability_matrix[1, token_pos] = from_1_to_1
            backtrace_matrix[1, token_pos] = 1

    # find final state with max log prob
    if probability_matrix[0,-1] > probability_matrix[1,-1]:
        max_final_log_prob = probability_matrix[0,-1]
        final_state = 0
    else:
        max_final_log_prob = probability_matrix[1,-1]
        final_state = 1

    # backtrace to get the most likely path
    # (calculate the adjusted per-state posteriors at each step)
    path = [final_state]
    conditional_probabilities = [
        getConditionalProbability(
            final_state,
            -1,
            probability_matrix,
            backtrace_matrix
        )
    ]
    for j in range(len(binomial_prob_sequence)-1, 0, -1):
        # get the previous state used for this path from the backtrace matrix
        prev_state = backtrace_matrix[path[-1], j]
        path.append(prev_state)
        # and calculate its conditional probability at that step
        conditional_probabilities.append(
            getConditionalProbability(
                prev_state,
                j-1,
                probability_matrix,
                backtrace_matrix
            )
        )
    path.reverse()
    conditional_probabilities.reverse()

    return max_final_log_prob, path, conditional_probabilities

def getConditionalProbability(state, timestep, probability_matrix, backtrace_matrix):
    if backtrace_matrix.shape[1] > 1:
        prev_state = backtrace_matrix[state, timestep-1]
        path_probability_to_prev_state = probability_matrix[prev_state, timestep-1]
    else:
        path_probability_to_prev_state = 0

    adjusted_probabilities_for_current_timestep = np.array([
        probability_matrix[0, timestep] - path_probability_to_prev_state,
        probability_matrix[1, timestep] - path_probability_to_prev_state,
    ])

    conditional_probability = (
        np.exp(adjusted_probabilities_for_current_timestep[state])
        /
        np.sum(np.exp(adjusted_probabilities_for_current_timestep))
    )

    return conditional_probability

def viterbiSmooth(line_predictions, transition_model, initial_relevant_prob):
    initial_state_probs = [
        (1 - initial_relevant_prob),
        initial_relevant_prob
    ]

    binomial_prob_sequence = [
        p.positive_probability
            for p in line_predictions
    ]

    (_, decoded_sequence, conditional_probs) = viterbi(
        binomial_prob_sequence,
        transition_model,
        initial_state_probs
    )

    for i in range(len(conditional_probs)):
        line_predictions[i].positive_probability = (
            conditional_probs[i]
                if decoded_sequence[i] == 1
                else (1-conditional_probs[i])
        )
