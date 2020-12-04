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
Model training script
'''

import os
import time
import tensorflow as tf
from datetime import datetime
import numpy as np
import sklearn.metrics
from types import SimpleNamespace
from hedgepig_logger import log
from data.dataset import Dataset
from data import embeddings
from evaluation import predictions_io
from . import sampling
from . import document_splits
from model import *

# shut TF up
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KMP_WARNINGS'] = '0'

def extractFeatures(dataset, embeds, next_batch):
    features = []
    for s in next_batch:
        embedding_features = embeds.getFeatures(
            dataset,
            s.doc_ID,
            s.line_index,
            s.token_index
        )
        features.append(embedding_features)

    labels = [
        s.label for s in next_batch
    ]

    return (np.array(features), labels)

def trainModel(model, dataset, embeds, train, dev, options, fold=0):
    dev_f_score_by_epoch = []
    best_dev_f_score = 0.
    best_epoch = -1

    log.writeln('Training...')
    still_training = True
    epoch = 0
    while still_training:
        log.writeln()
        log.writeln('= Epoch {0:,} ='.format(epoch))
        (train_pos, train_neg) = sampling.sampleTrainingData(
            train,
            options.positive_training_fraction,
            options.negative_training_ratio,
            random_seed=(
                options.random_seed
                + ( (options.max_epochs+1) * fold)
                + epoch
            )
        )
        log.writeln('Samples for epoch training:')
        log.writeln('  Positive: {0:,}'.format(len(train_pos)))
        log.writeln('  Negative: {0:,}\n'.format(len(train_neg)))

        sampled_train = [
            train.samples_by_ID[s_ID]
                for s_ID in [*train_pos, *train_neg]
        ]
        np.random.shuffle(sampled_train)

        n_batches = int(np.ceil(len(sampled_train) / options.batch_size))
        log.track('  >> Processed {0}/{1:,} batches'.format('{0:,}', n_batches), writeInterval=100)

        batch_start = 0
        while batch_start < len(sampled_train):
            next_batch = sampled_train[batch_start:batch_start+options.batch_size]
            (features, labels) = extractFeatures(dataset, embeds, next_batch)

            batch_loss = model.train(
                features,
                labels
            )

            batch_start += options.batch_size
            log.tick()
        log.flushTracker()

        # evaluate on dev data
        log.writeln('Evaluating on dev...')
        dev_metrics = testModel(
            model,
            dataset,
            embeds,
            dev.samples,
            options
        )
        log.writeln('  F-measure: {0:.4f}  [Accuracy: {1:.4f} -- {2:,}/{3:,}]'.format(
            dev_metrics.f_score,
            dev_metrics.accuracy,
            dev_metrics.correct,
            dev_metrics.total
        ))

        dev_f_score_by_epoch.append(dev_metrics.f_score)

        # patience/early stopping handling
        if dev_metrics.f_score > (best_dev_f_score + options.early_stopping):
            log.writeln('    >>> Improvement! Saving model state. <<<')
            model.save(fold)
            best_dev_f_score = dev_metrics.f_score
            best_epoch = epoch
            patience_so_far = 0
        else:
            patience_so_far += 1
            log.writeln('    >>> Impatience building... (%d/%d) <<<' % (patience_so_far, options.patience))
            if patience_so_far >= options.patience:
                log.writeln("    >>> Ran out of patience! <<<")
                log.writeln("           (╯'-')╯︵ ┻━┻ ")
                still_training = False

        if still_training and (options.max_epochs > 0) and epoch >= options.max_epochs:
            log.writeln("    >>> Hit maximum epoch threshold! <<<")
            log.writeln("                 ¯\(°_o)/¯")
            still_training = False

        epoch += 1

    log.writeln()
    model.restore(fold)
    log.writeln('Reverted to best model state.')

def testModel(model, dataset, embeds, samples, options, predictions_stream=None, labeled=True):
    n_batches = int(np.ceil(len(samples) / options.batch_size))
    log.track('  >> Processed {0}/{1:,} batches'.format('{0:,}', n_batches), writeInterval=100)

    # if we're writing out, ensure that the samples are in line and token order
    if predictions_stream:
        samples = sorted(
            samples,
            key=lambda s: (100000*s.line_index) + s.token_index
        )

    batch_start = 0
    correct, total = 0, 0
    all_labels, all_pos_probabilities, all_predictions = [], [], []
    while batch_start < len(samples):
        next_batch = samples[batch_start:batch_start+options.batch_size]
        (features, labels) = extractFeatures(dataset, embeds, next_batch)

        (probabilities, predictions) = model.predict(
            features
        )

        for i in range(len(labels)):
            if labels[i] == predictions[i]:
                correct += 1
            total += 1

        all_labels.extend(labels)
        all_pos_probabilities.extend(
            [p[1] for p in probabilities]
        )
        all_predictions.extend(predictions)

        batch_start += options.batch_size
        log.tick()
    log.flushTracker()

    if labeled:
        metrics = SimpleNamespace()
        metrics.accuracy = correct/total
        metrics.correct = correct
        metrics.total = total

        (
            metrics.precision,
            metrics.recall,
            metrics.f_score,
            metrics.support
        ) = sklearn.metrics.precision_recall_fscore_support(
            all_labels,
            all_predictions,
            pos_label=1,
            average='binary',
            beta=options.f_score_beta
        )

        metrics.auc = sklearn.metrics.roc_auc_score(
            all_labels,
            all_pos_probabilities
        )
    else:
        metrics = None

    if predictions_stream:
        for i in range(len(all_pos_probabilities)):
            predictions_io.writeSamplePrediction(
                predictions_stream,
                samples[i],
                all_pos_probabilities[i]
            )

    return metrics

if __name__ == '__main__':
    def _cli():
        import optparse
        parser = optparse.OptionParser(usage='Usage: %prog')
        parser.add_option('-c', '--checkpoint', dest='checkpoint_path',
            help='(REQUIRED) pretrained checkpoint file')
        parser.add_option('-l', '--logfile',
            help='logfile')

        testing = optparse.OptionGroup(parser, 'testing settings')
        testing.add_option('--testing', dest='testing_file',
            help='labels file for testing data')
        testing.add_option('--splits', dest='splits_base',
            help='file specifying cross-validation splits'
                 ' (if not provided, test on full dataset)')
        testing.add_option('--fold', dest='fold',
            help='fold to test on (if --splits)',
            type='int', default=0)
        testing.add_option('--batch-size', dest='batch_size',
            type='int', default=5,
            help='minibatch size for training (default %default)')
        testing.add_option('--predictions', dest='predictions_file',
            help='(REQUIRED) file to write predictions to')
        testing.add_option('--labeled', dest='test_dataset_labeled',
            action='store_true', default=False,
            help='flag if the testing file is labeled (default off)')

        evaluation = optparse.OptionGroup(parser, 'Evaluation settings')
        evaluation.add_option('--f-score-beta', dest='f_score_beta',
            type='float', default=2,
            help='Beta value for f measure (default %default)')

        embeddings_opts = optparse.OptionGroup(parser, 'Embeddings settings')
        embeddings.addCLIOptions(embeddings_opts)

        params = optparse.OptionGroup(parser, 'Hyperparameters')
        params.add_option('--layer-dims', dest='layer_dims',
            default='100',
            help='comma-separated list of integer dimensionalities for hidden layers')
        params.add_option('--activation-function', dest='activation_function',
            type='choice', choices=['relu'], default='relu',
            help='NN activation function')
        params.add_option('--no-bias', dest='use_bias',
            action='store_false', default=True,
            help='disable bias term in NN (default: bias enabled)')
        params.add_option('--class-weights', dest='class_weights',
            default='1,1',
            help='comma-separated pair of real weights for positive,negative class (default: %default)')
        params.add_option('--learning-rate', dest='learning_rate',
            type='float', default=0.001,
            help='learning rate for training (default %default)')
        params.add_option('--dropout-keep-prob', dest='dropout_keep_prob',
            type='float', default=0.5,
            help='dropout keep probability')

        model_params = optparse.OptionGroup(parser, 'Model-specific architecture settings')
        model_params.add_option('--num-bert-layers', dest='num_bert_layers',
            type='int', default=1,
            help='(ONLY WITH --embedding-method BERT) number of BERT layers'
                 ' providing input for')

        parser.add_option_group(testing)
        parser.add_option_group(embeddings_opts)
        parser.add_option_group(evaluation)
        parser.add_option_group(params)
        parser.add_option_group(model_params)
        (options, args) = parser.parse_args()

        options.layer_dims = [
            int(d) for d in options.layer_dims.split(',')
        ]
        options.class_weights = [
            float(w) for w in options.class_weights.split(',')
        ]

        if (not options.checkpoint_path):
            parser.error('Must provide --checkpoint')
        elif (not options.testing_file):
            parser.error('Must provide --testing')
        elif (not options.predictions_file):
            parser.error('Must provide --predictions')

        if not options.splits_base:
            options.fold = None
        
        embeddings.validateCLIOptions(options, parser)

        return options

    options = _cli()
    log.start(options.logfile)
    log.writeConfig([
        ('Testing settings', [
            ('Data file', options.testing_file),
            ('Cross-validation splits file', ('N/A' if options.splits_base is None else options.splits_base)),
            ('Cross-validation fold to test on', ('N/A' if options.splits_base is None else options.fold)),
            ('Minibatch size', options.batch_size),
            ('Model checkpoint path', options.checkpoint_path),
            ('Predictions file', options.predictions_file),
            ('Dataset is labeled', options.test_dataset_labeled),
        ]),
        ('Embedding features settings', embeddings.logCLIOptions(options)),
        ('Evaluation settings', [
            ('F-score beta', options.f_score_beta),
        ]),
        ('Model hyperparameters', [
            ('Hidden layer sizes', options.layer_dims),
            ('Activation function', options.activation_function),
            ('Using bias terms', options.use_bias),
            ('Dropout keep probability', options.dropout_keep_prob),
        ]),
        ('Embedding model-specific hyperparameters', [
            ('Number of BERT layers', options.num_bert_layers),
        ]),
    ], 'Pretrained MOHAIR model testing')

    log.writeln('Loading dataset from %s...' % options.testing_file)
    dataset = Dataset(
        labels_file=options.testing_file,
        labeled=options.test_dataset_labeled
    )
    log.writeln('Loaded')
    if options.test_dataset_labeled:
        log.writeln('  Positive samples: {0:,}'.format(len(dataset.positive_sample_IDs)))
        log.writeln('  Negative samples: {0:,}\n'.format(len(dataset.negative_sample_IDs)))
    else:
        log.writeln('  Total number of samples: {0:,}'.format(len(dataset)))

    if options.splits_base:
        log.writeln('Reading dataset splits from %s...' % options.splits_base)
        splits = document_splits.readDocumentSplits(options.splits_base)
        log.writeln('Found splits for {0:,} folds.\n'.format(len(splits)))

        log.writeln('Filtering down to test data for fold %d...' % options.fold)
        (train_files, dev_files, test_files) = splits[options.fold]
        dataset = dataset.filter(test_files)
        log.writeln('Filtered down to {0:,} samples.\n'.format(len(dataset)))

    log.writeln('Instantiating {0} embedding model...'.format(options.embedding_method))
    embeds = embeddings.instantiateFromCLIOptions(options)
    log.writeln('Loaded.\n')

    if options.predictions_file:
        test_predictions_stream = open(options.predictions_file, 'w')
    else:
        test_predictions_stream = None

    params = getDNNTokenClassifierParameters(
        options.embedding_method,

        # common hyperparameters
        embedding_dim = options.embedding_dim,
        layer_dims = options.layer_dims,
        activation_function = options.activation_function,
        dropout_keep_prob = options.dropout_keep_prob,
        use_bias = options.use_bias,
        class_weights = options.class_weights,
        learning_rate = options.learning_rate,

        # model-specific hyperparameters
        num_bert_layers = options.num_bert_layers,
    )

    with tf.Session() as session:
        model = getDNNTokenClassifier(
            options.embedding_method,
            session,
            params,
            options.checkpoint_path
        )
        model.restore(options.fold)

        log.writeln()
        log.writeln('Evaluating on fold test...')
        test_metrics = testModel(
            model,
            dataset,
            embeds,
            dataset.samples,
            options,
            predictions_stream=test_predictions_stream,
            labeled=options.test_dataset_labeled
        )
        if options.test_dataset_labeled:
            log.writeln('  F-measure: {0:.4f}'.format(test_metrics.f_score))
            log.writeln('  AUC: {0:.4f}'.format(test_metrics.auc))

        del(model)

    if options.predictions_file:
        test_predictions_stream.close()

    log.stop()
