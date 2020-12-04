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

from data.embeddings import EmbeddingMethods

def getDNNTokenClassifierParameters(embedding_method, **kwargs):
    if EmbeddingMethods.isMatch(
        embedding_method,
        EmbeddingMethods.Static
    ):
        return DNNTokenClassifierStaticParameters(
            **kwargs
        )

    elif EmbeddingMethods.isMatch(
        embedding_method,
        EmbeddingMethods.ELMo
    ):
        return DNNTokenClassifierELMoParameters(
            **kwargs
        )

    elif EmbeddingMethods.isMatch(
        embedding_method,
        EmbeddingMethods.BERT
    ):
        return DNNTokenClassifierBERTParameters(
            **kwargs
        )

    else:
        raise ValueError("No DNNTokenClassifierParameters subclass configured for embedding method '%s'" % embedding_method)


class DNNTokenClassifierParameters:
    
    def __init__(self,
        embedding_dim = 1024,
        layer_dims = None,
        activation_function = 'relu',
        dropout_keep_prob = 0.,
        use_bias = True,
        class_weights = None,
        learning_rate = 0.001,
        random_seed = -1,
        **kwargs
    ):
        self.embedding_dim = embedding_dim
        self.layer_dims = layer_dims
        self.activation_function = activation_function
        self.dropout_keep_prob = dropout_keep_prob
        self.use_bias = use_bias
        self.class_weights = class_weights
        self.learning_rate = learning_rate
        self.random_seed = random_seed


class DNNTokenClassifierStaticParameters(DNNTokenClassifierParameters):
    pass


class DNNTokenClassifierELMoParameters(DNNTokenClassifierParameters):
    pass


class DNNTokenClassifierBERTParameters(DNNTokenClassifierParameters):
    
    num_bert_layers = 1
    
    def __init__(self,
        num_bert_layers = 1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_bert_layers = num_bert_layers
