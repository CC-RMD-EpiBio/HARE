#!/bin/bash
#
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
#
# Script to get HDF5-format embeddings from BERT, with aligned
# tokenization.
#
# General flow of script:
#   == Input ==
#      (1) Pre-extracted data from running data/extract_data_files.py with a BERT tokenizer
#
#   == Output ==
#      (1) HDF5 file, with rows mapped one-to-one in input file, and columns mapped to each
#          token in each line.
#
#   == Procedure ==
#      (1) Split pre-tokenized lines into max-length chunks, for BERT max_seq_length
#          compatibility (creates subsequence files)
#      (2) Run BERT on subsequence files (creates JSON files)
#      (3) Convert JSON files to HDF5 and squish subsequences together (creates HDF5 file)
#
# Script arguments
# (1) DATASET -- dataset name (section header) in data/config.ini
# (2) MODEL -- BERT model to use (field names configured in data/config.ini)
# (3) GPU -- GPU card to use for BERT

set -e

if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
    echo "Usage: $0 DATASET MODEL GPU"
    exit
fi

PYTHON=python              # Python 3 binary with BERT-compatible libraries
DATADIR=$(pwd)/demo_data    # Root directory to find data files in
MODEL=$2
GPU=$3
DATASET=$1

## Location to look for BERT GitHub repo (default: in local directory);
## if not found at this location, will clone from GitHub
BERT=$(pwd)/bert_wrapper
## Location to look for BERT->HDF5 conversion code (default: in local directory);
## if not found at this location, will clone from GitHub
BERT_TO_HDF5=$(pwd)/bert_to_hdf5

## Check for BERT; if not found in the path, download it
STATUS=$(${PY} -c "import bert" 1>/dev/null 2>&1; echo $?)
if [ ! "${STATUS}" -eq 0 ]; then
    echo "##################################################################"
    echo "BERT not found in PYTHONPATH"
    echo "Using local copy in ${BERT}"
    if [ ! -d ${BERT}/bert ]; then
        echo "Cloning BERT from GitHub to ${BERT}/bert..."
        THISDIR=$(pwd)
        mkdir -p ${BERT}
        cd ${BERT}
        git clone https://github.com/google-research/bert.git
        cd ${THISDIR}
    fi
    export PYTHONPATH=${PYTHONPATH}:${BERT}
    echo "##################################################################"
    echo
fi
## Download BERT->HDF5 conversion package
if [ ! -d ${BERT_TO_HDF5} ]; then
    echo "##################################################################"
    echo "BERT->HDF5 conversion utility not found"
    echo "Cloning from GitHub to ${BERT_TO_HDF5}"
    git clone https://github.com/drgriffis/bert_to_hdf5.git
    mv bert_to_hdf5 ${BERT_TO_HDF5}
    echo "##################################################################"
    echo
fi

## Fetch BERT files from per-model specifications in config.ini
VOCABFILE=$(${PYTHON} -m cli_configparser.read_setting -c data/config.ini BERT "${MODEL} VocabFile")
CONFIGFILE=$(${PYTHON} -m cli_configparser.read_setting -c data/config.ini BERT "${MODEL} ConfigFile")
CKPTFILE=$(${PYTHON} -m cli_configparser.read_setting -c data/config.ini BERT "${MODEL} CkptFile")

if [[ "${VOCABFILE:0:1}" != '/' ]]; then VOCABFILE=$(pwd)/${VOCABFILE}; fi
if [[ "${CONFIGFILE:0:1}" != '/' ]]; then CONFIGFILE=$(pwd)/${CONFIGFILE}; fi
if [[ "${CKPTFILE:0:1}" != '/' ]]; then CKPTFILE=$(pwd)/${CKPTFILE}; fi

INPUT=${DATADIR}/${DATASET}/preprocessed.BERT.${MODEL}.tokens
PRE_TOKENIZED=${DATADIR}/${DATASET}/preprocessed.BERT.${MODEL}.pre_tokenized_for_BERT
JSON=${DATADIR}/${DATASET}/preprocessed.BERT.${MODEL}.json1
HDF5=${DATADIR}/${DATASET}/preprocessed.BERT.${MODEL}.hdf5
SEQ_LENGTH=200

cd ${BERT_TO_HDF5}

if [ ! -e "${PRE_TOKENIZED}.tokens" ]; then
    ${PYTHON} -m pre_tokenize_for_BERT \
        -i ${INPUT} \
        -o ${PRE_TOKENIZED} \
        -s ${SEQ_LENGTH} \
        --overlap 0.5 \
        -v ${VOCABFILE}
fi

if [ ! -e "${JSON}" ]; then
    export CUDA_VISIBLE_DEVICES=${GPU}
    ${PYTHON} -m extract_features_pretokenized \
        --input_file ${PRE_TOKENIZED}.subsequences \
        --output_file ${JSON} \
        --vocab_file ${VOCABFILE} \
        --bert_config_file ${CONFIGFILE} \
        --init_checkpoint ${CKPTFILE} \
        --layers -1,-2,-3 \
        --max_seq_length ${SEQ_LENGTH} \
        --batch_size 8
fi

if [ ! -e "${HDF5}" ]; then
    ${PYTHON} -m recombine_BERT_embeddings \
        --bert-output ${JSON} \
        --overlaps ${PRE_TOKENIZED}.overlaps \
        -o ${HDF5} \
        --tokenized ${HDF5}.aligned_tokens.txt \
        -l ${HDF5}.log
fi
