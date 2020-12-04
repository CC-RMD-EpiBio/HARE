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
Tensorflow implementation of independent token-level DNN model
'''

import tensorflow as tf
from data.embeddings import EmbeddingMethods
from .model_base import ELMoModelBase, StaticModelBase, BERTModelBase


def getDNNTokenClassifier(embedding_method, session, params, checkpoint_path, debug=False):
    if EmbeddingMethods.isMatch(
        embedding_method,
        EmbeddingMethods.Static
    ):
        return DNNTokenClassifierStatic(
            session,
            params,
            checkpoint_path,
            debug=debug
        )

    elif EmbeddingMethods.isMatch(
        embedding_method,
        EmbeddingMethods.ELMo
    ):
        return DNNTokenClassifierELMo(
            session,
            params,
            checkpoint_path,
            debug=debug
        )

    elif EmbeddingMethods.isMatch(
        embedding_method,
        EmbeddingMethods.BERT
    ):
        return DNNTokenClassifierBERT(
            session,
            params,
            checkpoint_path,
            debug=debug
        )

    else:
        raise ValueError("No DNNTokenClassifier model configured for embedding method '%s'" % embedding_method)



class DNNTokenClassifierStatic(StaticModelBase):
    
    def _build(self):
        super()._build()

        # embedding input
        self.embedding_ph = self._buildEmbeddingInput()

        # pass through the DNN
        (current_state, current_state_size) = self._buildDNN(
            self.embedding_ph,
            self.p.embedding_dim
        )

        # binary predictor
        output_logits = self._buildOutputLayer(
            current_state,
            current_state_size,
            2
        )

        # loss function and training
        self._buildLossFunction(output_logits)


    def train(self, batch_features, batch_labels):
        feed_dict = {
            self.embedding_ph: batch_features,
            self.labels_ph: batch_labels,
            self.dropout_ph: self.p.dropout_keep_prob
        }

        (_, batch_loss) = self.session.run(
            [self.train_step, self.batch_loss],
            feed_dict=feed_dict
        )

        if self.debug:
            input('Press [Enter] to continue to next batch')

        return batch_loss

    def predict(self, batch_features):
        feed_dict = {
            self.embedding_ph: batch_features,
            self.dropout_ph: 1.0
        }

        (output_probs, output_preds) = self.session.run(
            [self.output_probabilities, self.output_predictions],
            feed_dict=feed_dict
        )
        
        return (output_probs, output_preds)


class DNNTokenClassifierELMo(ELMoModelBase):

    def _build(self):
        super()._build()

        # ELMo embedding inputs
        (
            self.elmo_layer_0_ph,
            self.elmo_layer_1_ph,
            self.elmo_layer_2_ph
        ) = self._buildEmbeddingInputs()

        self.input_embeddings = self._buildEmbeddingCombinationModel(
            self.elmo_layer_0_ph,
            self.elmo_layer_1_ph,
            self.elmo_layer_2_ph
        )

        # pass through the DNN
        (current_state, current_state_size) = self._buildDNN(
            self.input_embeddings,
            self.p.embedding_dim
        )

        # binary predictor
        output_logits = self._buildOutputLayer(
            current_state,
            current_state_size,
            2
        )

        # loss function and training
        self._buildLossFunction(output_logits)


    def train(self, batch_features, batch_labels):
        elmo_layer_0_features = batch_features[:,0,:]
        elmo_layer_1_features = batch_features[:,1,:]
        elmo_layer_2_features = batch_features[:,2,:]

        feed_dict = {
            self.elmo_layer_0_ph: elmo_layer_0_features,
            self.elmo_layer_1_ph: elmo_layer_1_features,
            self.elmo_layer_2_ph: elmo_layer_2_features,
            self.labels_ph: batch_labels,
            self.dropout_ph: self.p.dropout_keep_prob
        }

        (_, batch_loss) = self.session.run(
            [self.train_step, self.batch_loss],
            feed_dict=feed_dict
        )

        return batch_loss

    def predict(self, batch_features):
        elmo_layer_0_features = batch_features[:,0,:]
        elmo_layer_1_features = batch_features[:,1,:]
        elmo_layer_2_features = batch_features[:,2,:]

        feed_dict = {
            self.elmo_layer_0_ph: elmo_layer_0_features,
            self.elmo_layer_1_ph: elmo_layer_1_features,
            self.elmo_layer_2_ph: elmo_layer_2_features,
            self.dropout_ph: 1.0
        }

        (output_probs, output_preds) = self.session.run(
            [self.output_probabilities, self.output_predictions],
            feed_dict=feed_dict
        )
        
        return (output_probs, output_preds)


class DNNTokenClassifierBERT(BERTModelBase):

    def _build(self):
        super()._build()

        # ELMo embedding inputs
        self.bert_layer_phs = self._buildEmbeddingInputs()

        self.input_embeddings = self._buildEmbeddingCombinationModel(
            self.bert_layer_phs
        )

        # pass through the DNN
        (current_state, current_state_size) = self._buildDNN(
            self.input_embeddings,
            self.p.embedding_dim
        )

        # binary predictor
        output_logits = self._buildOutputLayer(
            current_state,
            current_state_size,
            2
        )

        # loss function and training
        self._buildLossFunction(output_logits)


    def train(self, batch_features, batch_labels):
        feed_dict = {
            self.labels_ph: batch_labels,
            self.dropout_ph: self.p.dropout_keep_prob
        }
        for i in range(self.p.num_bert_layers):
            feed_dict[self.bert_layer_phs[i]] = batch_features[:,i,:]

        (_, batch_loss) = self.session.run(
            [self.train_step, self.batch_loss],
            feed_dict=feed_dict
        )

        if self.debug:
            input('Press [Enter] to continue to next batch')

        return batch_loss

    def predict(self, batch_features):
        feed_dict = {
            self.dropout_ph: 1.0
        }
        for i in range(self.p.num_bert_layers):
            feed_dict[self.bert_layer_phs[i]] = batch_features[:,i,:]

        (output_probs, output_preds) = self.session.run(
            [self.output_probabilities, self.output_predictions],
            feed_dict=feed_dict
        )

        if self.debug:
            input('Press [Enter] to continue to next batch')
        
        return (output_probs, output_preds)
