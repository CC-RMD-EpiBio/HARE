#!/bin/bash
#
#############################################################################
#
#                          PUBLIC DOMAIN NOTICE                         
#                Mark O. Hatfield Clinical Research Center
#                      National Institues of Health
#           United States Department of Health and Human Services
#                                                                         
#  This software is a United States Government Work under the terms of the
#  United States Copyright Act. It was written as part of the authors'
#  official duties as United States Government employees and contractors
#  and thus cannot be copyrighted. This software is freely available
#  to the public for use. The National Institutes of Health Clinical Center
#  and the #  United States Government have not placed any restriction on
#  its use or reproduction.
#                                                                        
#  Although all reasonable efforts have been taken to ensure the accuracy 
#  and reliability of the software and data, the National Institutes of
#  Health Clinical Center and the United States Government do not and cannot
#  warrant the performance or results that may be obtained by using this 
#  software or data. The National Institutes of Health Clinical Center and
#  the U.S. Government disclaim all warranties, expressed or implied,
#  including warranties of performance, merchantability or fitness for any
#  particular purpose.
#                                                                         
#  For full details, please see the licensing guidelines in the LICENSE file.
#
#############################################################################

## Path to Python binary with HARE requirements pre-installed
## (Should match the Python binary used in makefile)
PY=python

## Download pretrained BERT model
if [ ! -d demo_data/BERT_models/BERT_Base_uncased ]; then
    echo "##################################################################"
    echo "Downloading BERT-Base (uncased)..."
    mkdir -p demo_data/BERT_models
    cd demo_data/BERT_models
    wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
    unzip uncased_L-12_H-768_A-12.zip
    mv uncased_L-12_H-768_A-12 BERT_Base_uncased
    cd ../../
    echo "##################################################################"
    echo
fi
## Download pretrained ELMo model
if [ ! -d demo_data/ELMo_models/Original ]; then
    echo "##################################################################"
    echo "Downloading ELMo Original..."
    mkdir -p demo_data/ELMo_models/Original
    cd demo_data/ELMo_models/Original
    wget https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5
    wget https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json
    cd ../../../
    echo "##################################################################"
    echo
fi
## Download pretrained static model
if [ ! -d demo_data/static_embeddings/FastText ]; then
    echo "##################################################################"
    echo "Downloading FastText (WikiNews with subword)..."
    mkdir -p demo_data/static_embeddings/FastText
    cd demo_data/static_embeddings/FastText
    wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M-subword.vec.zip
    unzip wiki-news-300d-1M-subword.vec.zip
    ${PY} -m pyemblib.convert --from word2vec-text --to word2vec-binary wiki-news-300d-1M-subword.vec wiki-news-300d-1M-subword.vec.bin
    cd ../../../
    echo "##################################################################"
    echo
fi

## Run both versions of the preprocessing on labeled data...
echo "##################################################################"
echo "Preprocessing DemoLabeledDataset with SpaCy"
echo
make preprocess_with_SpaCy DATASET=DemoLabeledDataset
echo "##################################################################"
echo

echo "##################################################################"
echo "Preprocessing DemoLabeledDataset with WordPiece"
echo
make preprocess_with_WordPiece DATASET=DemoLabeledDataset MODEL=BERT-Base
echo "##################################################################"
echo


## ...and on unlabeled data
echo "##################################################################"
echo "Preprocessing DemoUnlabeledDataset with SpaCy"
echo
make preprocess_with_SpaCy DATASET=DemoUnlabeledDataset
echo "##################################################################"
echo

echo "##################################################################"
echo "Preprocessing DemoUnlabeledDataset with WordPiece"
echo
make preprocess_with_WordPiece DATASET=DemoUnlabeledDataset MODEL=BERT-Base
echo "##################################################################"
echo


## Generate cross-validation splits for each version
echo "##################################################################"
echo "Generating cross-validation splits for SpaCy-tokenized data"
echo
make generate_cross_validation_splits DATASET=DemoLabeledDataset SPEC=SpaCy K=5
echo "##################################################################"
echo

echo "##################################################################"
echo "Generating cross-validation splits for WordPiece-tokenized data"
echo
make generate_cross_validation_splits DATASET=DemoLabeledDataset SPEC=BERT.BERT-Base K=5
echo "##################################################################"
echo


## Generate ELMo and BERT features for labeled data...
echo "##################################################################"
echo "Generating ELMo features for DemoLabeledDataset"
echo
make generate_ELMo_features MODEL=Original DATASET=DemoLabeledDataset
echo "##################################################################"
echo

echo "##################################################################"
echo "Generating BERT features for DemoLabeledDataset"
echo
make generate_BERT_features DATASET=DemoLabeledDataset MODEL=BERT-Base GPU=0
echo "##################################################################"
echo


## ...and unlabeled data
echo "##################################################################"
echo "Generating ELMo features for DemoUnlabeledDataset"
echo
make generate_ELMo_features MODEL=Original DATASET=DemoUnlabeledDataset
echo "##################################################################"
echo

echo "##################################################################"
echo "Generating BERT features for DemoUnlabeledDataset"
echo
make generate_BERT_features DATASET=DemoUnlabeledDataset MODEL=BERT-Base GPU=0
echo "##################################################################"
echo


## Train with static features
echo "##################################################################"
echo "Training 2-layer DNN model with static embedding features"
echo
make train \
    DATASET=DemoLabeledDataset \
    METHOD=Static \
    MODEL=FastTextWikiNews \
    LAYERS=300,300 \
    GPU=0
EXP=$(ls demo_data/DemoLabeledDataset/experiments | grep xval.Static.FastTextWikiNews | sort -r | head -n 1)
echo -e "Demo Labeled - FastText\tLabeled\t$(pwd)/demo_data/DemoLabeledDataset/experiments/${EXP}/cross_validation.predictions" > demo_data/visualization_file_map
echo "##################################################################"
echo

echo "##################################################################"
echo "Training 2-layer DNN model with ELMo embedding features"
echo
make train \
    DATASET=DemoLabeledDataset \
    METHOD=ELMo \
    MODEL=Original \
    LAYERS=300,300 \
    GPU=0
EXP=$(ls demo_data/DemoLabeledDataset/experiments | grep xval.ELMo.Original | sort -r | head -n 1)
echo -e "Demo Labeled - ELMo\tLabeled\t$(pwd)/demo_data/DemoLabeledDataset/experiments/${EXP}/cross_validation.predictions" >> demo_data/visualization_file_map
echo "##################################################################"
echo

echo "##################################################################"
echo "Training 2-layer DNN model with BERT embedding features"
echo
make train \
    DATASET=DemoLabeledDataset \
    METHOD=BERT \
    MODEL=BERT-Base \
    LAYERS=300,300 \
    GPU=0
EXP=$(ls demo_data/DemoLabeledDataset/experiments | grep xval.BERT.BERT-Base | sort -r | head -n 1)
echo -e "Demo Labeled - BERT\tLabeled\t$(pwd)/demo_data/DemoLabeledDataset/experiments/${EXP}/cross_validation.predictions" >> demo_data/visualization_file_map
echo "##################################################################"
echo


## And test on the unlabeled dataset
echo "##################################################################"
echo "Test 2-layer DNN model with static embedding features on DemoUnlabeledDataset"
echo
EXP=$(ls demo_data/DemoLabeledDataset/experiments | grep xval.Static.FastTextWikiNews | sort -r | head -n 1)
make test \
    DATASET=DemoUnlabeledDataset \
    TRAIN_DATASET=DemoLabeledDataset \
    EXP=${EXP} \
    METHOD=Static \
    MODEL=FastTextWikiNews \
    LAYERS=300,300 \
    GPU=0 \
    FOLD=0 \
    UNLABELED=1
echo -e "Demo Unlabeled - FastText\tUnlabeled\t$(pwd)/demo_data/DemoLabeledDataset/experiments/${EXP}/DemoUnlabeledDataset.model.fold0.predictions" >> demo_data/visualization_file_map
echo "##################################################################"
echo

echo "##################################################################"
echo "Test 2-layer DNN model with ELMo embedding features on DemoUnlabeledDataset"
echo
EXP=$(ls demo_data/DemoLabeledDataset/experiments | grep xval.ELMo.Original | sort -r | head -n 1)
make test \
    DATASET=DemoUnlabeledDataset \
    TRAIN_DATASET=DemoLabeledDataset \
    EXP=${EXP} \
    METHOD=ELMo \
    MODEL=Original \
    LAYERS=300,300 \
    GPU=0 \
    FOLD=0 \
    UNLABELED=1
echo -e "Demo Unlabeled - ELMo\tUnlabeled\t$(pwd)/demo_data/DemoLabeledDataset/experiments/${EXP}/DemoUnlabeledDataset.model.fold0.predictions" >> demo_data/visualization_file_map
echo "##################################################################"
echo

echo "##################################################################"
echo "Test 2-layer DNN model with BERT embedding features on DemoUnlabeledDataset"
echo
EXP=$(ls demo_data/DemoLabeledDataset/experiments | grep xval.BERT.BERT-Base | sort -r | head -n 1)
make test \
    DATASET=DemoUnlabeledDataset \
    TRAIN_DATASET=DemoLabeledDataset \
    EXP=${EXP} \
    METHOD=BERT \
    MODEL=BERT-Base \
    LAYERS=300,300 \
    GPU=0 \
    FOLD=0 \
    UNLABELED=1
echo -e "Demo Unlabeled - BERT\tUnlabeled\t$(pwd)/demo_data/DemoLabeledDataset/experiments/${EXP}/DemoUnlabeledDataset.model.fold0.predictions" >> demo_data/visualization_file_map
echo "##################################################################"
echo

echo
echo
echo "Demo script complete!"
echo "Run the following to start the web-based interface to look at the output predictions:"
echo "   make start_web_interface"
echo
echo
