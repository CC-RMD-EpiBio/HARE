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


import tensorflow as tf

def getTransitionLabel(prev, nxt):
    if prev == 0 and nxt == 0:
        return 0
    elif prev == 0 and nxt == 1:
        return 1
    elif prev == 1 and nxt == 0:
        return 2
    elif prev == 1 and nxt == 1:
        return 3

class ModelBase:
    
    def __init__(self, session, params, checkpoint_path, debug=False):
        self.session = session
        self.debug = debug
        self.p = params
        self._build()

        self.saver = tf.train.Saver()
        self.checkpoint_path = checkpoint_path

        self.session.run(tf.global_variables_initializer())
    
    def save(self, fold=None):
        if not fold is None:
            pth = '%s.fold%d' % (self.checkpoint_path, fold)
        else:
            pth = self.checkpoint_path
        self.saver.save(self.session, pth)
    def restore(self, fold=None):
        if not fold is None:
            pth = '%s.fold%d' % (self.checkpoint_path, fold)
        else:
            pth = self.checkpoint_path
        self.saver.restore(self.session, pth)

    def _build(self):
        self.labels_ph = tf.placeholder(
            dtype=tf.int32,
            shape=[None],
            name='labels_placeholder'
        )
        self.dropout_ph = tf.placeholder_with_default(
            1.0,
            shape=[],
            name='dropout_prob_placeholder'
        )

        if self.debug:
            self.dropout = tf.Print(
                self.dropout_ph,
                [self.dropout_ph],
                message='Dropout keep probability'
            )
        else:
            self.dropout = self.dropout_ph

    def _buildDNN(self, current_state, current_state_size):
        if self.debug:
            current_state = tf.Print(
                current_state,
                [current_state],
                summarize=30,
                message='DNN input'
            )

        for i in range(len(self.p.layer_dims)):
            W = tf.get_variable(
                dtype=tf.float32,
                name='dnn_layer_%d_weights' % i,
                shape=[current_state_size, self.p.layer_dims[i]],
                initializer=tf.glorot_uniform_initializer(
                    seed=self.p.random_seed + 1042 + (2*i)
                ),
            )
            b = tf.get_variable(
                dtype=tf.float32,
                name='dnn_layer_%d_bias' % i,
                shape=[self.p.layer_dims[i]],
                initializer=tf.glorot_uniform_initializer(
                    seed=self.p.random_seed + 1042 + (2*i) + 1
                ),
            )
            current_state = (
                tf.matmul(current_state, W)
                + b
            )

            if self.p.activation_function == 'relu':
                current_state = tf.nn.relu(current_state)

            current_state = tf.nn.dropout(
                current_state,
                keep_prob=self.dropout
            )

            if self.debug:
                current_state = tf.Print(
                    current_state,
                    [current_state],
                    summarize=30,
                    message='DNN layer output'
                )

            current_state_size = self.p.layer_dims[i]

        return (current_state, current_state_size)

    def _buildOutputLayer(self, current_state, current_state_size, n_classes):
        # binary predictor
        W_out = tf.get_variable(
            dtype=tf.float32,
            name='output_layer_weights',
            shape=[current_state_size, n_classes],
            initializer=tf.glorot_uniform_initializer(
                seed=self.p.random_seed + 1042 + (3*len(self.p.layer_dims))
            )
        )
        b_out = tf.get_variable(
            dtype=tf.float32,
            name='output_layer_bias',
            shape=[n_classes],
            initializer=tf.glorot_uniform_initializer(
                seed=self.p.random_seed + 1042 + (3*len(self.p.layer_dims)) + 1
            )
        )
        output_logits = (
            tf.matmul(current_state, W_out)
            + b_out
        )

        # softmax output
        self.output_probabilities = tf.nn.softmax(
            output_logits,
            axis=1
        )
        self.output_predictions = tf.argmax(
            self.output_probabilities,
            axis=1
        )

        return output_logits

    def _buildLossFunction(self, output_logits):
        # set up the class weights
        class_weights = tf.constant(
            self.p.class_weights,
            dtype=tf.float32
        )
        batch_sample_weights = tf.gather(
            class_weights,
            self.labels_ph
        )

        if self.debug:
            batch_sample_weights = tf.Print(
                batch_sample_weights,
                [self.labels_ph],
                summarize=30,
                message='Batch labels'
            )
            batch_sample_weights = tf.Print(
                batch_sample_weights,
                [batch_sample_weights],
                summarize=30,
                message='Batch sample weights'
            )

        # weighted cross entropy as loss function
        sample_wise_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=output_logits,
            labels=self.labels_ph
        )
        weighted_sample_wise_loss = (
            sample_wise_loss
            * batch_sample_weights
        )

        if self.debug:
            weighted_sample_wise_loss = tf.Print(
                weighted_sample_wise_loss,
                [sample_wise_loss],
                summarize=30,
                message='Unweighted sample loss'
            )
            weighted_sample_wise_loss = tf.Print(
                weighted_sample_wise_loss,
                [weighted_sample_wise_loss],
                summarize=30,
                message='Weighted sample loss'
            )

        optimizer = tf.train.AdamOptimizer(
            learning_rate=self.p.learning_rate
        )
        self.train_step = optimizer.minimize(
            weighted_sample_wise_loss
        )

        # batch-level loss
        self.batch_loss = tf.reduce_sum(
            weighted_sample_wise_loss
        )


class ELMoModelBase(ModelBase):
    
    def _buildEmbeddingInputs(self, spec=None):
        if not spec is None:
            name = lambda n: '%s_%s' % (spec, n)
        else:
            name = lambda n: n

        elmo_layer_0_ph = tf.placeholder(
            dtype=tf.float32,
            shape=[None, self.p.embedding_dim],
            name=name('elmo_layer_0_ph')
        )
        elmo_layer_1_ph = tf.placeholder(
            dtype=tf.float32,
            shape=[None, self.p.embedding_dim],
            name=name('elmo_layer_1_ph')
        )
        elmo_layer_2_ph = tf.placeholder(
            dtype=tf.float32,
            shape=[None, self.p.embedding_dim],
            name=name('elmo_layer_2_ph')
        )

        return (
            elmo_layer_0_ph,
            elmo_layer_1_ph,
            elmo_layer_2_ph
        )
    
    def _buildEmbeddingCombinationModel(self, layer_0_ph, layer_1_ph, layer_2_ph):
        '''Assumes method is being called from within an
        appropriate variable scope.
        '''
        # ELMo embedding transformations
        elmo_layer_0_weights = tf.get_variable(
            name='elmo_layer_0_weights',
            shape=[1, self.p.embedding_dim],
            dtype=tf.float32,
            initializer=tf.glorot_uniform_initializer(
                seed=self.p.random_seed + 137
            ),
        )
        elmo_layer_1_weights = tf.get_variable(
            name='elmo_layer_1_weights',
            shape=[1, self.p.embedding_dim],
            dtype=tf.float32,
            initializer=tf.glorot_uniform_initializer(
                seed=self.p.random_seed + 138
            ),
        )
        elmo_layer_2_weights = tf.get_variable(
            name='elmo_layer_2_weights',
            shape=[1, self.p.embedding_dim],
            dtype=tf.float32,
            initializer=tf.glorot_uniform_initializer(
                seed=self.p.random_seed + 139
            ),
        )

        # combined embeddings
        weighted_layer_0 = (
            layer_0_ph
            * elmo_layer_0_weights
        )
        weighted_layer_1 = (
            layer_1_ph
            * elmo_layer_1_weights
        )
        weighted_layer_2 = (
            layer_2_ph
            * elmo_layer_2_weights
        )
        combined_embeddings = (
            weighted_layer_0
            + weighted_layer_1
            + weighted_layer_2
        )

        return combined_embeddings

class StaticModelBase(ModelBase):
    
    def _buildEmbeddingInput(self, spec=None):
        if not spec is None:
            name = lambda n: '%s_%s' % (spec, n)
        else:
            name = lambda n: n

        embedding_ph = tf.placeholder(
            dtype=tf.float32,
            shape=[None, self.p.embedding_dim]
        )

        return embedding_ph

class BERTModelBase(ModelBase):
    
    def _buildEmbeddingInputs(self, spec=None):
        if not spec is None:
            name = lambda n: '%s_%s' % (spec, n)
        else:
            name = lambda n: n

        bert_layer_phs = []
        for i in range(self.p.num_bert_layers):
            ph = tf.placeholder(
                dtype=tf.float32,
                shape=[None, self.p.embedding_dim],
                name=name('bert_layer_%d_ph' % i)
            )
            bert_layer_phs.append(ph)

        return bert_layer_phs
    
    def _buildEmbeddingCombinationModel(self, phs):
        '''Assumes method is being called from within an
        appropriate variable scope.
        '''
        # per-layer embedding transformations
        bert_layer_weights = []
        for i in range(self.p.num_bert_layers):
            weights = tf.get_variable(
                name='bert_layer_%d_weights' % i,
                shape=[1, self.p.embedding_dim],
                dtype=tf.float32,
                initializer=tf.glorot_uniform_initializer(
                    seed=self.p.random_seed + 1066
                ),
            )
            bert_layer_weights.append(weights)

        # combine the embeddings
        weighted_layers = []
        for i in range(self.p.num_bert_layers):
            weighted_layer = (
                phs[i]
                * bert_layer_weights[i]
            )
            weighted_layers.append(weighted_layer)
        combined_embeddings = tf.reduce_sum(
            weighted_layers,
            axis=0
        )

        return combined_embeddings
