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
from evaluation import annotation_metrics
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
    patience_so_far = 0

    # always save the model up front so there's at least one saved version of it
    model.save(fold)

    log.writeln('Training...')
    still_training = True
    epoch = 0
    while still_training:
        log.writeln()
        log.writeln('= Epoch {0:,} ='.format(epoch))

        if fold is None:
            epoch_random_seed = options.random_seed + 1066 + epoch
        else:
            epoch_random_seed = (
                options.random_seed
                + ( (options.max_epochs+1) * fold)
                + epoch
            )

        (train_pos, train_neg) = sampling.sampleTrainingData(
            train,
            options.positive_training_fraction,
            options.negative_training_ratio,
            random_seed=epoch_random_seed
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
            if options.debug:
                print('BATCH IDs: {0}'.format([s.ID for s in next_batch]))

            batch_loss = model.train(
                features,
                labels
            )

            batch_start += options.batch_size
            log.tick()
            if options.debug:
                break
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
            dev_metrics.f_beta,
            dev_metrics.accuracy,
            dev_metrics.correct,
            dev_metrics.total
        ))

        dev_f_score_by_epoch.append(dev_metrics.f_beta)

        # patience/early stopping handling
        if dev_metrics.f_beta > (best_dev_f_score + options.early_stopping):
            log.writeln('    >>> Improvement! Saving model state. <<<')
            model.save(fold)
            best_dev_f_score = dev_metrics.f_beta
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

def testModel(model, dataset, embeds, samples, options, predictions_stream=None):
    n_batches = int(np.ceil(len(samples) / options.batch_size))
    log.track('  >> Processed {0}/{1:,} batches'.format('{0:,}', n_batches), writeInterval=100)

    # if we're writing out, ensure that the samples are in line and token order
    if predictions_stream:
        samples = sorted(
            samples,
            key=lambda s: (100000*s.line_index) + s.token_index
        )

    batch_start = 0
    all_labels, all_pos_probabilities = [], []
    while batch_start < len(samples):
        next_batch = samples[batch_start:batch_start+options.batch_size]
        (features, labels) = extractFeatures(dataset, embeds, next_batch)

        (probabilities, predictions) = model.predict(
            features
        )

        all_labels.extend(labels)
        all_pos_probabilities.extend(
            [p[1] for p in probabilities]
        )

        batch_start += options.batch_size
        log.tick()
    log.flushTracker()

    metrics = annotation_metrics.getAnnotationMetricsRaw(
        all_pos_probabilities,
        all_labels,
        threshold=0.5,
        f_score_beta=options.f_score_beta
    )

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
        parser.add_option('-m', '--model', dest='model',
            help='(REQUIRED) name of directory to write training log/outputs to')
        parser.add_option('--debug', dest='debug',
            action='store_true', default=False,
            help='run in debug mode (batch by batch, print everything)')

        training = optparse.OptionGroup(parser, 'Training settings')
        training.add_option('--training', dest='training_file',
            help='labels file for training data')
        training.add_option('--splits', dest='splits_base',
            help='file specifying cross-validation splits'
                 ' (if not provided, trains on full dataset)')
        training.add_option('--positive-training-fraction', dest='positive_training_fraction',
            type='float', default=0.5,
            help='fraction (0-1] of the positive samples to include in each epoch'
                 ' (default %default)')
        training.add_option('--negative-training-ratio', dest='negative_training_ratio',
            type='float', default=1.0,
            help='ratio of negative to positive samples in each epoch\'s training'
                 ' (default %default)')
        training.add_option('--batch-size', dest='batch_size',
            type='int', default=5,
            help='minibatch size for training (default %default)')
        training.add_option('--max-epochs', dest='max_epochs',
            type='int', default=50,
            help='maximum number of epochs to train for (default %default)')
        training.add_option('--patience', dest='patience',
            type='int', default=5,
            help='maximum number of epochs to wait for improvement on'
                 ' dev set (default %default)')
        training.add_option('--early-stopping', dest='early_stopping',
            type='float', default=1e-5,
            help='early stopping threshold for dev set error'
                 ' (default %default)')
        training.add_option('--full-train-num-dev-docs', dest='full_train_num_dev_docs',
            type='int', default=5,
            help='number of documents to use as dev data when training'
                 ' on full dataset (default %default)')
        training.add_option('--random-seed', dest='random_seed',
            type='int', default=-1,
            help='random seed for reproducibility; if not given, uses current epoch time')

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

        parser.add_option_group(training)
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
        # flip class weights around, as given in <positive>,<negative>
        # order, but need to apply for 0=negative, 1=positive
        options.class_weights.reverse()

        if options.random_seed <= 0:
            options.random_seed = int(time.time())

        if (not options.model):
            parser.error('Must provide --model')
        elif (not options.training_file):
            parser.error('Must provide --training')
        elif options.positive_training_fraction <= 0 or options.positive_training_fraction > 1:
            parser.error('--positive-training-fraction must be in range (0,1]')
        
        embeddings.validateCLIOptions(options, parser)

        options.predictions_file = None

        # set up output files
        now_stamp = datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M-%S')
        options.model = '%s.%s' % (options.model, now_stamp)
        if not os.path.exists(options.model):
            os.mkdir(options.model)
        options.logfile = os.path.join(options.model, 'training.log')
        options.checkpoint_path = os.path.join(options.model, 'model')
        if options.splits_base:
            options.predictions_file = os.path.join(options.model, 'cross_validation.predictions')

        return options

    options = _cli()
    log.start(options.logfile)
    log.writeConfig([
        ('Model directory', options.model),
        ('Running in debug', options.debug),
        ('Training settings', [
            ('Labels file', options.training_file),
            ('Cross-validation splits file', ('N/A' if options.splits_base is None else options.splits_base)),
            ('Max number of epochs', options.max_epochs),
            ('Patience epochs', options.patience),
            ('Early stopping threshold', options.early_stopping),
            ('Per-epoch positive fraction for training', options.positive_training_fraction),
            ('Per-epoch ratio of negative:positive samples', options.negative_training_ratio),
            ('Minibatch size', options.batch_size),
            ('Model checkpoint path', options.checkpoint_path),
            ('Predictions file', ('N/A' if options.splits_base is None else options.predictions_file)),
            ('Number of docs held out for dev (full-set training only)', ('N/A' if options.splits_base else options.full_train_num_dev_docs)),
            ('Random seed', options.random_seed),
        ]),
        ('Embedding features settings', embeddings.logCLIOptions(options)),
        ('Evaluation settings', [
            ('F-score beta', options.f_score_beta),
        ]),
        ('Model hyperparameters', [
            ('Hidden layer sizes', options.layer_dims),
            ('Activation function', options.activation_function),
            ('Using bias terms', options.use_bias),
            ('Class weights', options.class_weights),
            ('Learning rate', options.learning_rate),
            ('Dropout keep probability', options.dropout_keep_prob),
        ]),
        ('Embedding model-specific hyperparameters', [
            ('Number of BERT layers', options.num_bert_layers),
        ]),
    ], 'MOHAIR model training')

    log.writeln('Loading dataset from %s...' % options.training_file)
    dataset = Dataset(
        labels_file=options.training_file
    )
    log.writeln('Loaded')
    log.writeln('  Positive samples: {0:,}'.format(len(dataset.positive_sample_IDs)))
    log.writeln('  Negative samples: {0:,}\n'.format(len(dataset.negative_sample_IDs)))

    if options.splits_base:
        log.writeln('Reading dataset splits from %s...' % options.splits_base)
        splits = document_splits.readDocumentSplits(options.splits_base)
        log.writeln('Found splits for {0:,} folds.\n'.format(len(splits)))

        training_settings = [
            (i, splits[i])
                for i in range(len(splits))
        ]

    else:
        log.writeln('Training on full dataset!')
        all_docs = list(dataset.samples_by_doc_ID.keys())

        np.random.seed(options.random_seed)
        np.random.shuffle(all_docs)

        train_files = set(all_docs[options.full_train_num_dev_docs:])
        dev_files = set(all_docs[:options.full_train_num_dev_docs])
        test_files = set()

        log.writeln('  Using {0:,} training documents'.format(len(train_files)))
        log.writeln('  Using {0:,} dev documents:'.format(len(dev_files)))
        nice_dev_files = sorted(dev_files)
        for i in range(len(nice_dev_files)):
            log.writeln('    %d) %s' % (i, nice_dev_files[i]))
        log.writeln()

        training_settings = [
            (None, (train_files, dev_files, test_files))
        ]

    log.writeln('Instantiating {0} embedding model...'.format(options.embedding_method))
    embeds = embeddings.instantiateFromCLIOptions(options)
    log.writeln('Loaded.\n')

    if options.predictions_file and (options.splits_base or len(test_files) > 0):
        test_predictions_stream = open(options.predictions_file, 'w')
    else:
        test_predictions_stream = None

    fold_train_metrics = []
    fold_dev_metrics = []
    fold_test_metrics = []

    for (fold, (train_files, dev_files, test_files)) in training_settings:
        train = dataset.filter(train_files)
        dev = dataset.filter(dev_files)
        if len(test_files) > 0:
            test = dataset.filter(test_files)
        else:
            test = []

        if options.predictions_file:
            train_predictions_stream = open('%s.%s.train' % (
                options.predictions_file,
                '' if fold is None else ('fold%d' % fold)
            ), 'w')
            dev_predictions_stream = open('%s.%s.dev' % (
                options.predictions_file,
                '' if fold is None else ('fold%d' % fold)
            ), 'w')
        else:
            train_predictions_stream = None
            dev_predictions_stream = None

        if fold is None:
            log.writeln('\n\n--- Full-set training ---\n')
        else:
            log.writeln('\n\n--- Fold {0:,}/{1:,} ---\n'.format(fold+1, len(splits)))
        log.indent()

        log.writeln('Training size: {0:,} samples'.format(len(train)))
        log.writeln('Dev size: {0:,} samples'.format(len(dev)))
        log.writeln('Test size: {0:,} samples\n'.format(len(test)))

        if fold is None:
            random_seed_offset = 1337
        else:
            random_seed_offset = fold

        np.random.seed(options.random_seed + random_seed_offset)
        tf.set_random_seed(options.random_seed + random_seed_offset)
        tf.reset_default_graph()

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
            random_seed = options.random_seed + random_seed_offset,

            # model-specific hyperparameters
            num_bert_layers = options.num_bert_layers,
        )

        with tf.Session() as session:
            model = getDNNTokenClassifier(
                options.embedding_method,
                session,
                params,
                options.checkpoint_path,
                debug=options.debug
            )

            trainModel(
                model,
                dataset,
                embeds,
                train,
                dev,
                options,
                fold=fold
            )

            log.writeln()
            log.writeln('Evaluating best model on fold train...')
            train_metrics = testModel(
                model,
                dataset,
                embeds,
                train.samples,
                options,
                predictions_stream=train_predictions_stream
            )
            fold_train_metrics.append(train_metrics)
            log.writeln('  F-measure: {0:.4f}'.format(train_metrics.f_beta))
            log.writeln('  AUC: {0:.4f}'.format(train_metrics.auc))

            log.writeln()
            log.writeln('Re-evaluating best model on fold dev...')
            dev_metrics = testModel(
                model,
                dataset,
                embeds,
                dev.samples,
                options,
                predictions_stream=dev_predictions_stream
            )
            fold_dev_metrics.append(dev_metrics)
            log.writeln('  F-measure: {0:.4f}'.format(dev_metrics.f_beta))
            log.writeln('  AUC: {0:.4f}'.format(dev_metrics.auc))

            log.writeln()
            if len(test) > 0:
                log.writeln('Evaluating on fold test...')
                test_metrics = testModel(
                    model,
                    dataset,
                    embeds,
                    test.samples,
                    options,
                    predictions_stream=test_predictions_stream
                )
                fold_test_metrics.append(test_metrics)
                log.writeln('  F-measure: {0:.4f}'.format(test_metrics.f_beta))
                log.writeln('  AUC: {0:.4f}'.format(test_metrics.auc))
            else:
                log.writeln('--- No test data provided ---')

            del(model)

        if options.predictions_file:
            train_predictions_stream.close()
            dev_predictions_stream.close()

        log.unindent()

    log.writeln('\n\n')
    if options.splits_base:
        log.writeln('--- Cross-validation report ---')
        for (lbl, subset_metrics) in [
            ('Train', fold_train_metrics),
            ('Dev', fold_dev_metrics),
            ('Test', fold_test_metrics)
        ]:
            log.indent()
            log.writeln()
            log.writeln('>>> {0} <<<\n'.format(lbl))

            log.writeln('Per-fold results')
            log.indent()
            for i in range(len(subset_metrics)):
                log.writeln('Fold {0} -- F-measure: {1:.4f}  AUC: {2:.4f}'.format(
                    i, subset_metrics[i].f_beta, subset_metrics[i].auc
                ))
            log.unindent()

            log.writeln()
            log.writeln('Macro results')
            log.writeln('  F-measure: {0:.4f}'.format(
                np.mean([m.f_beta for m in subset_metrics])
            ))
            log.writeln('  AUC: {0:.4f}'.format(
                np.mean([m.auc for m in subset_metrics])
            ))

            log.unindent()

    if test_predictions_stream:
        test_predictions_stream.close()

    log.stop()
