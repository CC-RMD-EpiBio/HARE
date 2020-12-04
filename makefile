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

SHELL=/bin/bash
## Path to Python 3 binary to use (with HARE requirements pre-installed)
PY=python3

## Initial preprocessing, using SpaCy for tokenization; generates 2 files:
##  (1) .tokens -- tokenized version of dataset, maintaining line breaks
##  (2) .labels -- tokenized version with one token and corresponding label per line
preprocess_with_SpaCy:
	@if [ -z "${DATASET}" ]; then \
		echo "Must supply DATASET (using a section header in data/config.ini)"; \
		exit; \
	else \
		DATASET=${DATASET}; \
	fi; \
	DATADIR=$$(${PY} -m cli_configparser.read_setting -c data/config.ini Experiments DataDirectory); \
	if [ ! -d "$${DATADIR}/$${DATASET}" ]; then \
		mkdir $${DATADIR}/$${DATASET}; \
	fi; \
	${PY} -m data.extract_data_files \
		-c data/config.ini \
		--dataset $${DATASET} \
		-t SpaCy \
		$${DATADIR}/$${DATASET}/preprocessed.SpaCy \
		-l $${DATADIR}/$${DATASET}/preprocessed.SpaCy.log

## Initial preprocessing, using WordPiece (via BERT) for tokenization; generates 2 files:
##  (1) .tokens -- tokenized version of dataset, maintaining line breaks
##  (2) .labels -- tokenized version with one token and corresponding label per line
preprocess_with_WordPiece:
	@export PYTHONPATH=/home/griffisd/toolkits/bert:$${PYTHONPATH}; \
	if [ -z "${MODEL}" ]; then \
		echo "Must supply MODEL (using a BERT model reference from data/config.ini)"; \
		exit; \
	else \
		MODEL=${MODEL}; \
	fi; \
	if [ -z "${DATASET}" ]; then \
		echo "Must supply DATASET (using a section header in data/config.ini)"; \
		exit; \
	else \
		DATASET=${DATASET}; \
	fi; \
	DATADIR=$$(${PY} -m cli_configparser.read_setting -c data/config.ini Experiments DataDirectory); \
	if [ ! -d "$${DATADIR}/$${DATASET}" ]; then \
		mkdir $${DATADIR}/$${DATASET}; \
	fi; \
	${PY} -m data.extract_data_files \
		-c data/config.ini \
		--dataset $${DATASET} \
		-t BERT \
		--BERT-vocab-file $$(${PY} -m cli_configparser.read_setting -c data/config.ini BERT "$${MODEL} Vocabfile") \
		$${DATADIR}/$${DATASET}/preprocessed.BERT.$${MODEL} \
		-l $${DATADIR}/$${DATASET}/preprocessed.BERT.$${MODEL}.log

## Generate splits at document-level for cross-fold validation
## Split generation is linked to sample-level IDs, so will not
## necessarily map across different preprocessed versions of the
## same dataset.
## TODO: Add script to automatically remap these IDs
generate_cross_validation_splits:
	@if [ -z "${DATASET}" ]; then \
		echo "Must supply DATASET (using a section header in data/config.ini)"; \
		exit; \
	else \
		DATASET=${DATASET}; \
	fi; \
	if [ -z "${SPEC}" ]; then \
		echo "Must supply SPEC (specifier for preprocessed file, e.g. SPEC=SpaCy or SPEC=BERT.clinicalBERT)"; \
		exit; \
	else \
		SPEC=${SPEC}; \
	fi; \
	if [ -z "${K}" ]; then \
		K=10; \
	fi; \
	DATADIR=$$(${PY} -m cli_configparser.read_setting -c data/config.ini Experiments DataDirectory); \
	if [ ! -d "$${DATADIR}/$${DATASET}/splits" ]; then \
		mkdir $${DATADIR}/$${DATASET}/splits; \
	fi; \
	${PY} -m experiments.document_splits \
		$${DATADIR}/$${DATASET}/preprocessed.$${SPEC}.labels \
		$${DATADIR}/$${DATASET}/splits/preprocessed.$${SPEC}.splits \
		--dev-size 0.2 \
		--n-folds $${K} \
		-l $${DATADIR}/$${DATASET}/splits/preprocessed.$${SPEC}.splits.log

## Feature generation with ELMo
## Requires that allennlp be in the PATH
generate_ELMo_features:
	@if [ -z "${MODEL}" ]; then \
		echo "Must supply MODEL (using an ELMo model reference from data/config.ini)"; \
		exit; \
	else \
		MODEL=${MODEL}; \
	fi; \
	if [ -z "${DATASET}" ]; then \
		echo "Must supply DATASET (using a section header in data/config.ini)"; \
		exit; \
	else \
		DATASET=${DATASET}; \
	fi; \
	DATADIR=$$(${PY} -m cli_configparser.read_setting -c data/config.ini Experiments DataDirectory); \
	allennlp elmo \
		--options-file $$(${PY} -m cli_configparser.read_setting -c data/config.ini ELMo "$${MODEL} OptionsFile") \
		--weight-file $$(${PY} -m cli_configparser.read_setting -c data/config.ini ELMo "$${MODEL} WeightsFile") \
		--all \
		$${DATADIR}/$${DATASET}/preprocessed.SpaCy.tokens \
		$${DATADIR}/$${DATASET}/preprocessed.SpaCy.ELMo-embedded.$${MODEL}.hdf5

## Feature generation with BERT
## Requires that BERT code be cloned from GitHuab; see utils/get_bert_hdf5_features.sh for details
generate_BERT_features:
	@bash utils/get_bert_hdf5_features.sh ${DATASET} ${MODEL} ${GPU}


## Master training maketarget
## Supports:
##  - Choice of dataset to train on (DATASET flag)
##  - Cross-validation or training on full set (FULL_TRAIN flag)
##  - Hyperparameter tuning (CLASS_WEIGHTS, DROPOUT, NEGRATIO, POSFRAC, LAYERS flags)
##  - Choice of embedding method (METHOD flag)
##  - Choice of embedding model (MODEL flag)
train:
	@export CUDA_VISIBLE_DEVICES=${GPU}; \
	if [ -z "${MODEL}" ]; then \
		echo "MODEL must be supplied"; \
		exit; \
	else \
		MODEL=${MODEL}; \
	fi; \
	if [ -z "${DATASET}" ]; then \
		echo "Must supply DATASET (using a section header in data/config.ini)"; \
		exit; \
	else \
		DATASET=${DATASET}; \
	fi; \
	if [ -z "${CLASS_WEIGHTS}" ]; then \
		CLASSFLAG=; \
		LOGPREFIX=$${LOGPREFIX}; \
	else \
		CLASSFLAG="--class-weights ${CLASS_WEIGHTS}"; \
		LOGPREFIX=hptune.class-weights.${CLASS_WEIGHTS}; \
	fi; \
	if [ -z "${DROPOUT}" ]; then \
		DROPOUTFLAG=; \
		LOGPREFIX=$${LOGPREFIX}; \
	else \
		DROPOUTFLAG="--dropout-keep-prob ${DROPOUT}"; \
		LOGPREFIX=hptune.dropout-keep-prob.${DROPOUT}; \
	fi; \
	if [ -z "${NEGRATIO}" ]; then \
		NEGRATIOFLAG=; \
		LOGPREFIX=$${LOGPREFIX}; \
	else \
		NEGRATIOFLAG="--negative-training-ratio ${NEGRATIO}"; \
		LOGPREFIX=hptune.neg-ratio.${NEGRATIO}; \
	fi; \
	if [ -z "${POSFRAC}" ]; then \
		POSFRACFLAG=; \
		LOGPREFIX=$${LOGPREFIX}; \
	else \
		POSFRACFLAG="--positive-training-fraction ${POSFRAC}"; \
		LOGPREFIX=hptune.pos-fraction.${POSFRAC}; \
	fi; \
	if [ -z "${LAYERS}" ]; then \
		LAYERSFLAG=; \
		LOGPREFIX=$${LOGPREFIX}; \
	else \
		LAYERSFLAG="--layer-dims ${LAYERS}"; \
		LOGPREFIX=hptune.layers.${LAYERS}; \
	fi; \
	if [ -z "${DEBUG}" ]; then \
		DEBUGFLAG=; \
		LOGPREFIX=$${LOGPREFIX}; \
	else \
		DEBUGFLAG="--debug"; \
		LOGPREFIX=debug; \
	fi; \
	DATADIR=$$(${PY} -m cli_configparser.read_setting -c data/config.ini Experiments DataDirectory); \
	case "${METHOD}" in \
		BERT) \
			METHODSFLAGS="--embedding-method bert --embedding-dim 768 --num-bert-layers 3"; \
			EMBEDDINGSFILE=$${DATADIR}/$${DATASET}/preprocessed.BERT.$${MODEL}.hdf5; \
			TRAININGFILE=$${DATADIR}/$${DATASET}/preprocessed.BERT.$${MODEL}.labels; \
			SPLITSFILE=$${DATADIR}/$${DATASET}/splits/preprocessed.BERT.$${MODEL}.splits; \
			LOGSUFFIX=BERT.$${MODEL}; \
			;; \
		ELMo) \
			METHODSFLAGS="--embedding-method elmo --embedding-dim 1024"; \
			EMBEDDINGSFILE=$${DATADIR}/$${DATASET}/preprocessed.SpaCy.ELMo-embedded.$${MODEL}.hdf5; \
			TRAININGFILE=$${DATADIR}/$${DATASET}/preprocessed.SpaCy.labels; \
			SPLITSFILE=$${DATADIR}/$${DATASET}/splits/preprocessed.SpaCy.splits; \
			LOGSUFFIX=ELMo.$${MODEL}; \
			;; \
		Static) \
			METHODSFLAGS="--embedding-method static --embedding-dim"; \
			METHODSFLAGS="$${METHODSFLAGS} $$(${PY} -m cli_configparser.read_setting -c data/config.ini Static "$${MODEL} Dimensionality")"; \
			EMBEDDINGSFILE=$$(${PY} -m cli_configparser.read_setting -c data/config.ini Static "$${MODEL} File"); \
			TRAININGFILE=$${DATADIR}/$${DATASET}/preprocessed.SpaCy.labels; \
			SPLITSFILE=$${DATADIR}/$${DATASET}/splits/preprocessed.SpaCy.splits; \
			LOGSUFFIX=Static.$${MODEL}; \
			;; \
		*) \
			echo "METHOD must be supplied as one of [Static, BERT, ELMo]"; \
			exit; \
			;; \
    esac; \
	if [ -z "${FULL_TRAIN}" ]; then \
		SPLITSFLAG="--splits $${SPLITSFILE}"; \
		LOGPREFIX="xval"; \
	else \
		SPLITSFLAG=; \
		LOGPREFIX="full"; \
	fi; \
	if [ ! -d "$${DATADIR}/$${DATASET}/experiments" ]; then \
		mkdir $${DATADIR}/$${DATASET}/experiments; \
	fi; \
	${PY} -m experiments.train \
		--training $${TRAININGFILE} \
        $${METHODSFLAGS} \
        --embeddings-file $${EMBEDDINGSFILE} \
		$${SPLITSFLAG} \
		$${CLASSFLAG} \
		$${DROPOUTFLAG} \
		$${NEGRATIOFLAG} \
		$${POSFRACFLAG} \
		$${LAYERSFLAG} \
		$${DEBUGFLAG} \
		--batch-size 25 \
		-m $${DATADIR}/$${DATASET}/experiments/$${LOGPREFIX}.$${LOGSUFFIX}


## Master test maketarget
## Supports
##  - Specification of pretrained model (EXP flag)
##  - Dataset to run on (DATASET)
##  - Specification of per-fold pretrained model (FOLD flag)
##  - Model structure (LAYERS flag; must match what was used for the pretrained model)
##  - Running on labeled or unlabeled data (UNLABELED flag)
##  - Embedding method and model (METHOD and MODEL; must match pretrained model)
test:
	@export CUDA_VISIBLE_DEVICES=${GPU}; \
	if [ -z "${EXP}" ]; then \
		echo "EXP must be specified"; \
		exit; \
	fi; \
	if [ -z "${DATASET}" ]; then \
		echo "Must supply DATASET (using a section header in data/config.ini)"; \
		exit; \
	else \
		DATASET=${DATASET}; \
	fi; \
	if [ -z "${TRAIN_DATASET}" ]; then \
		echo "Must supply TRAIN_DATASET (using a section header in data/config.ini)"; \
		exit; \
	else \
		TRAIN_DATASET=${TRAIN_DATASET}; \
	fi; \
	if [ -z "${MODEL}" ]; then \
		echo "MODEL must be supplied"; \
		exit; \
	else \
		MODEL=${MODEL}; \
	fi; \
	if [ -z "${FOLD}" ]; then \
		FOLD=0; \
	else \
		FOLD=${FOLD}; \
	fi; \
	if [ -z "${LAYERS}" ]; then \
		LAYERSFLAG=; \
	else \
		LAYERSFLAG="--layer-dims ${LAYERS}"; \
	fi; \
	if [ -z "${UNLABELED}" ]; then \
		LABELEDFLAG="--labeled"; \
	else \
		LABELEDFLAG=; \
	fi; \
	if [ -z "${METHOD}" ]; then \
		echo "METHOD must be supplied as one of [Static, BERT, ELMo]"; \
		exit; \
	fi; \
	if [ -z "${MODEL}" ]; then \
		echo "MODEL must be supplied"; \
		exit; \
	else \
		MODEL=${MODEL}; \
	fi; \
	DATADIR=$$(${PY} -m cli_configparser.read_setting -c data/config.ini Experiments DataDirectory); \
	case "${METHOD}" in \
		BERT) \
			METHODSFLAGS="--embedding-method bert --embedding-dim 768 --num-bert-layers 3"; \
			EMBEDDINGSFILE=$${DATADIR}/$${DATASET}/preprocessed.BERT.$${MODEL}.hdf5; \
			TESTFILE=$${DATADIR}/$${DATASET}/preprocessed.BERT.$${MODEL}.labels; \
			;; \
		ELMo) \
			METHODSFLAGS="--embedding-method elmo --embedding-dim 1024"; \
			EMBEDDINGSFILE=$${DATADIR}/$${DATASET}/preprocessed.SpaCy.ELMo-embedded.$${MODEL}.hdf5; \
			TESTFILE=$${DATADIR}/$${DATASET}/preprocessed.SpaCy.labels; \
			;; \
		Static) \
			METHODSFLAGS="--embedding-method static --embedding-dim"; \
			METHODSFLAGS="$${METHODSFLAGS} $$(${PY} -m cli_configparser.read_setting -c data/config.ini Static "$${MODEL} Dimensionality")"; \
			EMBEDDINGSFILE=$$(${PY} -m cli_configparser.read_setting -c data/config.ini Static "$${MODEL} File"); \
			TESTFILE=$${DATADIR}/$${DATASET}/preprocessed.SpaCy.labels; \
			;; \
		*) \
			echo "METHOD must be supplied as one of [Static, BERT, ELMo]"; \
			exit; \
			;; \
    esac; \
	${PY} -m experiments.run_pretrained \
		--testing $${TESTFILE} \
		$${METHODSFLAGS} \
		--embeddings-file $${EMBEDDINGSFILE} \
		$${LAYERSFLAG} \
		--batch-size 25 \
		$${LABELEDFLAG} \
		--dropout-keep-prob 1.0 \
		-c $${DATADIR}/$${TRAIN_DATASET}/experiments/${EXP}/model.fold${FOLD} \
		-l $${DATADIR}/$${TRAIN_DATASET}/experiments/${EXP}/$${DATASET}.model.fold${FOLD}.log \
		--predictions $${DATADIR}/$${TRAIN_DATASET}/experiments/${EXP}/$${DATASET}.model.fold${FOLD}.predictions


## Start the web-based visualization interface
## Reads settings from visualization/viz_config.ini
## Requires browser access to local webserver
start_web_interface:
	@export FLASK_APP=visualization/app.py; \
	${PY} -m flask run
