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

import h5py
import pyemblib
import optparse
import numpy as np

class EmbeddingMethods:
    Static = 'static'
    ELMo = 'elmo'
    BERT = 'bert'

    @staticmethod
    def asList():
        return [
            EmbeddingMethods.Static,
            EmbeddingMethods.ELMo,
            EmbeddingMethods.BERT
        ]

    @staticmethod
    def isMatch(query, key):
        return query.lower() == key.lower()


class StaticEmbeddings:
    
    def __init__(self, embeddings_file, embeddings_file_format, window_size):
        self.embeddings = pyemblib.read(
            embeddings_file,
            mode=embeddings_file_format
        )
        self.window_size = window_size

    def getFeatures(self, dataset, doc_ID, line_index, token_index):
        line_tokens = dataset.indexed[doc_ID][line_index]
        left_ctx_tokens = sorted(
            [
                (k,v)
                    for (k,v) in line_tokens.items()
                    if k < token_index
            ],
            key=lambda pair: pair[0]
        )
        right_ctx_tokens = sorted(
            [
                (k,v)
                    for (k,v) in line_tokens.items()
                    if k > token_index
            ],
            key=lambda pair: pair[0]
        )

        left_ctx_window_tokens = left_ctx_tokens[-self.window_size:]
        left_ctx_tokens = [
            t.token
                for (ix, t) in left_ctx_window_tokens
        ]
        right_ctx_window_tokens = right_ctx_tokens[:self.window_size]
        right_ctx_tokens = [
            t.token
                for (ix, t) in right_ctx_window_tokens
        ]
        this_token = line_tokens[token_index].token

        mean_embedding = np.zeros(self.embeddings.size, dtype=np.float32)
        num_tokens = 0
        for tokens in [left_ctx_tokens, [this_token], right_ctx_tokens]:
            for token in tokens:
                if token.lower() in self.embeddings:
                    mean_embedding += self.embeddings[token.lower()]
                    num_tokens += 1

        if num_tokens > 0:
            mean_embedding = mean_embedding / num_tokens

        return mean_embedding

class ELMoEmbeddings:

    def __init__(self, embeddings_file):
        self.embeddings = h5py.File(embeddings_file, 'r')

    def getFeatures(self, dataset, doc_ID, line_index, token_index):
        return self.embeddings['{}'.format(line_index)][:,token_index,:]

class BERTEmbeddings:
    
    def __init__(self, embeddings_file):
        self.embeddings = h5py.File(embeddings_file, 'r')

    def getFeatures(self, dataset, doc_ID, line_index, token_index):
        return self.embeddings['{}'.format(line_index)][:,token_index,:]

def addCLIOptions(parser):
    parser.add_option('--embedding-method', dest='embedding_method',
        type='choice', choices=EmbeddingMethods.asList(),
        help='Embedding method to use for feature generation (choices: %s)' % (
            ', '.join(EmbeddingMethods.asList())
        ))
    parser.add_option('--embeddings-file', dest='embeddings_file',
        help='(REQUIRED) file containing formatted embeddings')
    parser.add_option('--embeddings-file-format', dest='embeddings_file_format',
        type='choice', choices=[pyemblib.Mode.Text, pyemblib.Mode.Binary],
        default=pyemblib.Mode.Binary,
        help='[static embeddings only] indicate binary or text format of'
             ' --embeddings-file (options: {0}; default %default)'.format(
                ', '.join([pyemblib.Mode.Text, pyemblib.Mode.Binary])
             ))
    parser.add_option('--embedding-dimensionality', dest='embedding_dim',
        type='int', default=300,
        help='dimensionality of input embeddings (default %default)')
    parser.add_option('--window-size', dest='window_size',
        type='int', default=10,
        help='[static embeddings only] context window size (default %default)')

def validateCLIOptions(options, parser):
    if not options.embeddings_file:
        parser.error('Must provide --embeddings-file')

def logCLIOptions(options):
    is_static = EmbeddingMethods.isMatch( options.embedding_method, EmbeddingMethods.Static)
    return [
        ('Embedding method', options.embedding_method),
        ('Embeddings file', options.embeddings_file),
        ('Embeddings file format', options.embeddings_file_format if is_static else 'N/A'),
        ('Embedding dimensionality', options.embedding_dim),
        ('Context window size', options.window_size if is_static else 'N/A'),
    ]

def instantiateFromCLIOptions(options):
    if EmbeddingMethods.isMatch(
        options.embedding_method,
        EmbeddingMethods.Static
    ):
        return StaticEmbeddings(
            embeddings_file=options.embeddings_file,
            embeddings_file_format=options.embeddings_file_format,
            window_size=options.window_size
        )

    elif EmbeddingMethods.isMatch(
        options.embedding_method,
        EmbeddingMethods.ELMo
    ): 
        return ELMoEmbeddings(
            embeddings_file=options.embeddings_file
        )

    elif EmbeddingMethods.isMatch(
        options.embedding_method,
        EmbeddingMethods.BERT
    ): 
        return BERTEmbeddings(
            embeddings_file=options.embeddings_file
        )

    else:
        raise ValueError('Unknown embedding method "%s"' % options.embedding_method)
