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

from flask import Flask
from flask import render_template
from flask import request
app = Flask(__name__)

import os
import configparser
import sklearn.metrics
import numpy as np
from types import SimpleNamespace
import packaging
import utils
from evaluation import predictions_io
from evaluation import transitions_io
from evaluation import labeling
from evaluation import scoring
from evaluation import ranking
from evaluation import classification
from evaluation import decoding
from evaluation import annotation_metrics
from analysis import lexicalization
from analysis import score_distribution
from analysis import thresholding_analysis

config = configparser.ConfigParser()
config.read('visualization/viz_config.ini')

if config['Decoding']['TransitionModelType'] == 'static':
    viterbi_transition_model = transitions_io.StaticTransitionMatrix.fromFile(
        config['Decoding']['StaticTransitionMatrixFile']
    )
else:
    raise ValueError('TransitionModelType "%s" not known' % config['Decoding']['TransitionModelType'])
viterbi_initial_relevant_prob = float(config['Decoding']['InitialStateMobilityProb'])

def readAnnotationSetInfoCache():
    set_info = {}
    # read the existing cached annot set info
    if os.path.exists('cached_annot_set_info'):
        with open('cached_annot_set_info', 'r') as stream:
            for line in stream:
                (set_name, file_IDs) = [s.strip() for s in line.split('\t')]
                set_info[set_name] = set([s.strip() for s in file_IDs.split(',')])
    return set_info
def completeAnnotationSetInfoCache(set_info, file_map):
    # fill in any missing annot sets
    for annot_set in file_map:
        if not annot_set in set_info:
            predictions = predictions_io.readSamplePredictions(file_map[annot_set].filepath)
            predictions_by_doc = labeling.indexPredictionsByDocAndLine(predictions)
            set_info[annot_set] = set(predictions_by_doc.keys())
def writeAnnotationSetInfoCache(set_info):
    # write the cache back to disk
    with open('cached_annot_set_info', 'w') as stream:
        for (annot_set, file_IDs) in set_info.items():
            stream.write('%s\t%s\n' % (annot_set, ','.join(file_IDs)))
def getAnnotationSetInfo(file_map):
    set_info = readAnnotationSetInfoCache()
    completeAnnotationSetInfoCache(set_info, file_map)
    writeAnnotationSetInfoCache(set_info)
    return set_info

@app.route('/')
def listAnnotSets():
    file_map = utils.loadFileMap(config)
    annot_set_info = getAnnotationSetInfo(file_map)

    annot_sets = []
    for (set_name, set_obj) in file_map.items():
        annot_sets.append(SimpleNamespace(
            name=set_name,
            labeled=('Labeled' if file_map[set_name].labeled else 'Unlabeled'),
            num_files=('{0:,}'.format(len(annot_set_info[set_name]))),
        ))
    return render_template(
        'annot_set_list.html',
        annot_sets=annot_sets
    )

@app.route('/list', methods=['POST'])
@app.route('/list/<annot_set>', methods=['GET', 'POST'])
def listDocuments(annot_set=None):
    file_map = utils.loadFileMap(config)

    if request.method == 'GET':
        getter = request.args.get
    else:
        getter = request.form.get
    
    if annot_set is None:
        annot_set = getter('annot_set', None)

    annotation_settings = packaging.getPostProcessingConfiguration(getter, config)
    ranking_settings = packaging.getRankingConfiguration(getter, config)

    # resolve to actual scorer objects
    gold_scorer = scoring.getScorer(
        SimpleNamespace(
            model_scorer=ranking_settings.gold_scorer,
            min_probability=float(annotation_settings.min_probability),
            threshold=float(annotation_settings.threshold),
            num_blanks=int(annotation_settings.num_blanks),
            use_label_instead=True
        )
    )
    model_scorer = scoring.getScorer(
        SimpleNamespace(
            model_scorer=ranking_settings.model_scorer,
            min_probability=float(annotation_settings.min_probability),
            threshold=float(annotation_settings.threshold),
            num_blanks=int(annotation_settings.num_blanks),
            use_label_instead=False
        )
    )

    # resolve to classifier object
    classification_thresholds = classification.BinningClassifier.parseThresholdsString(ranking_settings.classification_thresholds) 
    classification_colors = classification.BinningClassifier.parseColorsString(ranking_settings.classification_colors)
    classifier = classification.getClassifier(
        SimpleNamespace(
            classification_method=ranking_settings.classification_method,
            binning_thresholds=classification_thresholds
        )
    )
    
    if not annot_set in file_map:
        return 'Annotation set %s is unknown!' % annot_set
    else:
        predictions = predictions_io.readSamplePredictions(file_map[annot_set].filepath)
        predictions_by_doc = labeling.indexPredictionsByDocAndLine(predictions)

        # run Viterbi decoder if desired
        if annotation_settings.viterbi_smoothing:
            for (_, doc_predictions) in predictions_by_doc.items():
                for (_, line_predictions) in doc_predictions.items():
                    decoding.viterbiSmooth(
                        line_predictions,
                        viterbi_transition_model,
                        viterbi_initial_relevant_prob
                    )

        # get model ranking
        model_segment_labeled_predictions = labeling.assignCorpusSegmentsAndLabels(
            predictions_by_doc,
            threshold=float(annotation_settings.threshold),
            num_blanks=int(annotation_settings.num_blanks)
        )
        model_scores = model_scorer.score(
            predictions,
            pre_labeled_predictions=model_segment_labeled_predictions,
            annot_set_for_caching=annot_set,
            additional_settings_for_caching={ 'viterbi_smoothing': annotation_settings.viterbi_smoothing }
        )
        model_statistics = labeling.summaryAnnotationStatistics(model_segment_labeled_predictions)
        
        # if using a labeled dataset, get "gold" ranking (with current configuration) and run
        # (a) ranking comparison evaluation and
        # (b) token-level evaluations
        if file_map[annot_set].labeled:
            gold_segment_labeled_predictions = labeling.assignCorpusSegmentsAndLabels(
                predictions_by_doc,
                threshold=0.5,
                num_blanks=int(annotation_settings.num_blanks),
                value_getter=lambda x:x.label
            )
            gold_scores = gold_scorer.score(
                predictions,
                pre_labeled_predictions=gold_segment_labeled_predictions,
                annot_set_for_caching=annot_set,
                additional_settings_for_caching={ 'viterbi_smoothing': annotation_settings.viterbi_smoothing }
            )
            gold_statistics = labeling.summaryAnnotationStatistics(gold_segment_labeled_predictions)
            
            # and compare the rankings
            (ranked, rho) = ranking.rankingComparison(
                model_scores=model_scores,
                gold_scores=gold_scores,
                order_by_gold=ranking_settings.order_by_gold
            )

            docs = []
            for doc in ranked:
                if ranking_settings.class_by_gold:
                    doc_class = classifier(doc.gold_score)
                else:
                    doc_class = classifier(doc.model_score)
                (bg_color, fg_color) = classification_colors[doc_class]

                docs.append({
                    'ID': doc.ID,
                    'model_rank': doc.model_rank,
                    'gold_rank': doc.gold_rank,
                    'model_score': model_scorer.renderScore(doc.model_score),
                    'gold_score': gold_scorer.renderScore(doc.gold_score),
                    'class': doc_class,
                    'bg_color': bg_color,
                    'fg_color': fg_color,
                })

            # get overall evaluation info
            token_metrics = annotation_metrics.getAnnotationMetrics(
                predictions,
                threshold=annotation_settings.threshold
            )
            
        # if it's an unlabeled dataset, just get the rankings by the model score
        else:
            ranked = ranking.rankUnlabeledDataByModelScore(
                model_scores=model_scores
            )
            
            docs = []
            for doc in ranked:
                doc_class = classifier(doc.model_score)
                (bg_color, fg_color) = classification_colors[doc_class]
                
                docs.append({
                    'ID': doc.ID,
                    'model_rank': doc.model_rank,
                    'model_score': model_scorer.renderScore(doc.model_score),
                    'class': doc_class,
                    'bg_color': bg_color,
                    'fg_color': fg_color
                })
            
            rho = None
            token_metrics = None
            gold_statistics = None

        classifier_info = classifier.getConfiguration(
            colors=classification_colors,
            scorer=(
                gold_scorer 
                    if (file_map[annot_set].labeled and ranking_settings.class_by_gold)
                    else model_scorer
            )
        )
        
        # prettify of labeled evaluation statistics
        if file_map[annot_set].labeled:
            rho = '%.1f' % (100*rho)
            token_metrics = packaging.prettifyMetrics(token_metrics)
        
        return render_template(
            'document_list.html',
            docs=docs,
            labeled=file_map[annot_set].labeled,
            
            rho=rho,
            num_documents=len(docs),
            annot_set=annot_set,

            scorer_options=scoring.scorerChoices(),
            ranking_settings=ranking_settings,
            annotation_settings=annotation_settings,

            classifier_info=classifier_info,

            annotation_metrics=token_metrics,
            model_statistics=packaging.prettifyStatistics(model_statistics),
            gold_statistics=packaging.prettifyStatistics(gold_statistics),
            
            annot_sets=packaging.packageAnnotSets(file_map, annot_set)
        )

@app.route('/view', methods=['GET', 'POST'])
@app.route('/view/<annot_set>/<doc_ID>', methods=['GET', 'POST'])
def view(annot_set=None, doc_ID=None):
    if request.method == 'GET':
        getter = request.args.get
    else:
        getter = request.form.get
    
    if annot_set is None:
        annot_set = getter('annot_set', None)
    if doc_ID is None:
        doc_ID = getter('doc_ID', None)
    
    annotation_settings = packaging.getPostProcessingConfiguration(getter, config)
    ranking_settings = packaging.getRankingConfiguration(getter, config)
    
    file_map = utils.loadFileMap(config)
    annot_set_info = getAnnotationSetInfo(file_map)

    if not annot_set in file_map:
        return 'Annotation set "%s" not known!' % annot_set
    else:
        predictions = predictions_io.readSamplePredictions(file_map[annot_set].filepath)
        doc_predictions = [
            p for p in predictions if p.doc_ID == doc_ID
        ]
        line_predictions = labeling.indexPredictionsByDocAndLine(doc_predictions)[doc_ID]

        line_predictions = sorted(
            line_predictions.items(),
            key=lambda pair: pair[0]
        )

        cur_token_ID = 0

        lines = []
        model_statistics = SimpleNamespace(
            num_segments=0,
            num_tokens=0
        )
        for i in range(len(line_predictions)):
            line_ix, ordered_tokens = line_predictions[i]
            if i > 0:
                # add in any blank lines as needed here
                while cur_line_ix < (line_ix-1):
                    lines.append({
                        'blank': True
                    })
                    cur_line_ix += 1
            cur_line_ix = line_ix

            # run Viterbi decoder if desired
            if annotation_settings.viterbi_smoothing:
                decoding.viterbiSmooth(
                    ordered_tokens,
                    viterbi_transition_model,
                    viterbi_initial_relevant_prob
                )

            # get token-level information
            (labeled_tokens, statistics) = labeling.assignLineSegmentsAndLabels(
                ordered_tokens,
                threshold=annotation_settings.threshold,
                num_blanks=annotation_settings.num_blanks
            )
            model_statistics.num_segments += statistics.num_segments
            model_statistics.num_tokens += statistics.num_tokens

            tokens = []
            for j in range(len(labeled_tokens)):
                sample = ordered_tokens[j]
                (t, tag) = labeled_tokens[j]
                assert sample.token == t

                tokens.append({
                    'ID': cur_token_ID,
                    'class': (
                        'true_mobility' 
                        if (file_map[annot_set].labeled and sample.label == 1)
                        else 'normal'
                    ),
                    'score': str(sample.positive_probability),
                    'text': sample.token,
                    'tag': tag,
                    'is_EOL': False
                })

                cur_token_ID += 1

            tokens[-1]['is_EOL'] = True

            lines.append({
                'blank': False,
                'tokens': tokens
            })
        
        if file_map[annot_set].labeled:
            metrics = annotation_metrics.getAnnotationMetrics(
                doc_predictions,
                threshold=annotation_settings.threshold
            )

            #  gold segmentation
            gold_statistics = SimpleNamespace(
                num_segments=0,
                num_tokens=0
            )
            for (_, ordered_tokens) in line_predictions:
                (_, true_statistics) = labeling.assignLineSegmentsAndLabels(
                    ordered_tokens,
                    threshold=annotation_settings.threshold,
                    num_blanks=annotation_settings.num_blanks,
                    value_getter=lambda s:s.label
                )
                gold_statistics.num_segments += true_statistics.num_segments
                gold_statistics.num_tokens += true_statistics.num_tokens
        
        else:
            metrics = None

        # render qualitative analysis plots for this document
        score_distrib_base64 = packaging.renderImage(
            score_distribution.plotScoreDistribution,
            args=([p.positive_probability for p in doc_predictions],),
            kwargs={'figsize': (6,3), 'font_size': 18}
        )
        
        if file_map[annot_set].labeled:
            thresholding_distrib_base64 = packaging.renderImage(
                thresholding_analysis.plotCurve,
                args=thresholding_analysis.precisionRecallF2Curve(doc_predictions),
                kwargs={'figsize': (6,3)}
            )
        else:
            thresholding_distrib_base64 = None
            gold_statistics = None
        
        # prettify stuff
        if file_map[annot_set].labeled:
            metrics = packaging.prettifyMetrics(metrics)

        return render_template(
            'document_annotations.html',
            doc_title=doc_ID,
            lines=lines,
            annot_set=annot_set,
            doc_ID=doc_ID,
            labeled=file_map[annot_set].labeled,

            model_statistics=model_statistics,
            gold_statistics=gold_statistics,
            annotation_metrics=metrics,
            score_distrib_base64=score_distrib_base64,
            thresholding_distrib_base64=thresholding_distrib_base64,
            
            annot_sets=packaging.packageAnnotSets(file_map, annot_set, annot_set_info, doc_ID),
            annotation_settings=annotation_settings,
            ranking_settings=ranking_settings,
        )

@app.route('/qualitative', methods=['POST'])
@app.route('/qualitative/<annot_set>', methods=['GET', 'POST'])
def qualitativeAnalysis(annot_set=None):
    file_map = utils.loadFileMap(config)

    if request.method == 'GET':
        getter = request.args.get
    else:
        getter = request.form.get
    
    if annot_set is None:
        annot_set = getter('annot_set', None)

    annotation_settings = packaging.getPostProcessingConfiguration(getter, config)
    ranking_settings = packaging.getRankingConfiguration(getter, config)

    lexicalization_settings = SimpleNamespace(
        min_freq = int(getter(
            'min_lexicalization_frequency',
            config['Lexicalization']['MinimumFrequency']
        ))
    )

    if not annot_set in file_map:
        return 'Annotation set %s is unknown!' % annot_set
    else:
        predictions = predictions_io.readSamplePredictions(file_map[annot_set].filepath)

        if annotation_settings.viterbi_smoothing:
            decoding.viterbiSmooth(
                predictions,
                viterbi_transition_model,
                viterbi_initial_relevant_prob
            )

        lexicalization_stats = SimpleNamespace()

        pred_lex_scores, token_counts, lexicalization_stats.num_filtered = \
            lexicalization.calculateLexicalization(
                predictions, 
                lexicalization_settings.min_freq
            )
        if file_map[annot_set].labeled:
            gold_lex_scores, _, _ = lexicalization.calculateLexicalization(
                predictions,
                lexicalization_settings.min_freq,
                use_labels=True
            )
            (ranked_tokens, lexicalization_stats.rho) = ranking.rankingComparison(
                pred_lex_scores,
                gold_lex_scores,
                order_by_gold=False
            )
        else:
            (ranked_tokens, _) = ranking.rankingComparison(
                pred_lex_scores,
                pred_lex_scores
            )
            lexicalization_stats.rho = None

        lexicalization_stats.lex_info = []
        for token in ranked_tokens:
            lex_item = SimpleNamespace(
                token=token.ID,
                model_rank=token.model_rank,
                model_lex_score=token.model_score,
                count=token_counts[token.ID]
            )
            if file_map[annot_set].labeled:
                lex_item.gold_rank=token.gold_rank
                lex_item.gold_lex_score=token.gold_score
            else:
                lex_item.gold_rank = None
                lex_item.gold_lex_score = None
            lexicalization_stats.lex_info.append(lex_item)

        score_distrib_base64 = packaging.renderImage(
            score_distribution.plotScoreDistribution,
            args=([p.positive_probability for p in predictions],),
            kwargs={'figsize': (6,3), 'font_size': 18}
        )

        if file_map[annot_set].labeled:
            (precisions, recalls, f2s, thresholds, best_ix) = PRFC = thresholding_analysis.precisionRecallF2Curve(predictions)
            thresholding_distrib_base64 = packaging.renderImage(
                thresholding_analysis.plotCurve,
                args=PRFC,
                kwargs={'figsize': (6,3)}
            )
            best_threshold = thresholds[best_ix]
        else:
            thresholding_distrib_base64 = None
            best_threshold = None

        if file_map[annot_set].labeled:
            best_metrics = annotation_metrics.getAnnotationMetrics(
                predictions,
                threshold=best_threshold
            )
        else:
            best_metrics = None

        # get performance info for the current settings
        predictions_by_doc = labeling.indexPredictionsByDocAndLine(predictions)
        model_segment_labeled_predictions = labeling.assignCorpusSegmentsAndLabels(
            predictions_by_doc,
            threshold=float(annotation_settings.threshold),
            num_blanks=int(annotation_settings.num_blanks)
        )
        model_statistics = labeling.summaryAnnotationStatistics(model_segment_labeled_predictions)
        if file_map[annot_set].labeled:
            gold_segment_labeled_predictions = labeling.assignCorpusSegmentsAndLabels(
                predictions_by_doc,
                threshold=0.5,
                num_blanks=int(annotation_settings.num_blanks),
                value_getter=lambda x:x.label
            )
            gold_statistics = labeling.summaryAnnotationStatistics(gold_segment_labeled_predictions)
            current_metrics = annotation_metrics.getAnnotationMetrics(
                predictions,
                threshold=annotation_settings.threshold
            )
        else:
            current_metrics = None
            gold_statistics = None

        return render_template(
            'qualitative_analysis.html',
            annot_set=annot_set,
            labeled=file_map[annot_set].labeled,

            lexicalization_stats=packaging.prettifyLexicalizationStats(lexicalization_stats),

            score_distrib_base64=score_distrib_base64,
            thresholding_distrib_base64=thresholding_distrib_base64,
            
            best_threshold=best_threshold,
            best_metrics=packaging.prettifyMetrics(best_metrics),

            annot_sets=packaging.packageAnnotSets(file_map, annot_set),
            annotation_settings=annotation_settings,
            ranking_settings=ranking_settings,
            lexicalization_settings=lexicalization_settings,

            annotation_metrics=packaging.prettifyMetrics(current_metrics),
            model_statistics=packaging.prettifyStatistics(model_statistics),
            gold_statistics=packaging.prettifyStatistics(gold_statistics),
        )
