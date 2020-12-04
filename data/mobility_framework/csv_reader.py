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
Given a set of CSV annotations for Mobility, extract all annotations.
'''

import codecs
import csv
import os
import re
import glob
import json
from hedgepig_logger import log
from .models import *

def matchAnnotationAndTextFiles(data_directories, text_directory, csv_id_pattern, txt_sub_pattern, log=log):
    csv_files = {}

    csv_id_getter = re.compile(csv_id_pattern)
    for csvdir in data_directories:
        for f in os.listdir(csvdir):
            match = re.match(csv_id_getter, f)
            if match:
                _id = match.groups(1)[0]
                fpath = os.path.join(csvdir, f)
                csv_files[_id] = fpath

    paired_files = {}
    for (_id, csv_path) in csv_files.items():
        txt_path = os.path.join(
            text_directory,
            txt_sub_pattern.format(_id)
        )
        if os.path.isfile(txt_path):
            paired_files[_id] = (
                csv_path,
                txt_path
            )
        else:
            log.writeln('[WARNING] Could not find plaintext file for ID {0}'.format(_id))

    return paired_files

def extractAnnotationsFromFile(csvf, txtf, as_document=False):
    mobilities = []
    actions = []
    assistances = []
    quantifications = []

    with codecs.open(txtf, 'r', 'utf-8') as stream:
        full_text = stream.read()

    with codecs.open(csvf, 'r', 'utf-8-sig') as stream:
        reader = csv.reader(stream, delimiter='|')
        for record in reader:
            if record[0].lower() == 'mobility':
                mobilities.append(parseMobility(record, txtf))
            elif record[0].lower() == 'action':
                actions.append(parseAction(record, txtf))
            elif record[0].lower() == 'assistance':
                assistances.append(parseAssistance(record, txtf))
            elif record[0].lower() == 'quantification':
                quantifications.append(parseQuantification(record, txtf))

    if as_document:
        return Document(
            mobilities=mobilities,
            actions=actions,
            assistances=assistances,
            quantifications=quantifications,
            full_text=full_text
        )
    else:
        return (
            mobilities,
            actions,
            assistances,
            quantifications
        )

def baseParse(record, txtf):
    start_pos = int(record[1])
    end_pos = int(record[2])

    expected_text = record[3]
    with open(txtf, 'r') as stream:
        doc_text = stream.read()
    actual_text = doc_text[start_pos:end_pos]

    if expected_text != actual_text:
        log.writeln('[WARNING] Mis-alignment on {0} mention -- Expected "{1}"  Found "{2}"'.format(
            record[0], expected_text, actual_text
        ))

    return (start_pos, end_pos, expected_text)

def parseAction(record, txtf):
    (start_pos, end_pos, text) = baseParse(record, txtf)

    code = None
    if len(record[4]) > 0:
        code = record[4]
    polarity = None
    if len(record[5]) > 0:
        polarity = int(record[5])

    return Action(
        start = start_pos,
        end = end_pos,
        text = text,
        code = code,
        polarity = polarity
    )

def parseMobility(record, txtf):
    (start_pos, end_pos, text) = baseParse(record, txtf)

    _type = None
    if len(record[4]) > 0:
        _type = record[4]
    subject = None
    if len(record[5]) > 0:
        subject = record[5]
    history = None
    if len(record[6]) > 0:
        history = int(record[6])

    return Mobility(
        start = start_pos,
        end = end_pos,
        text = text,
        type = _type,
        history = history
    )

def parseQuantification(record, txtf):
    (start_pos, end_pos, text) = baseParse(record, txtf)

    _type = None

    if len(record[4]) > 0:
        _type = record[4]

    return Quantification(
        start = start_pos,
        end = end_pos,
        text = text,
        type = _type
    )

def parseAssistance(record, txtf):
    (start_pos, end_pos, text) = baseParse(record, txtf)

    polarity = None
    if len(record[4]) > 0:
        polarity = record[4]
    source = None
    if len(record[5]) > 0:
        source = record[5]

    return Assistance(
        start = start_pos,
        end = end_pos,
        text = text,
        polarity = polarity,
        source = source
    )

def extractAllEntities(data_directories, text_directory, csv_id_pattern, txt_sub_pattern, log=log,
        with_full_text=False, by_document=True):
    '''
    Extract all Mobility, Action, Assistance, and Quantification entities from
    CSV-formatted annotation files.

    @parameters
      data_directories :: list of directories containing .csv annotation files
      text_directory   :: directory containing reference .txt files
      csv_id_pattern   :: Python regex pattern for extracting file ID from a CSV file name
                          (as first group); e.g. 'myfile_([0-9]*).csv' will extract
                          '12345' as file ID from file myfile_12345.csv
      txt_sub_pattern  :: Python string formatting pattern for matching to reference
                          text files (ID substituted for {0}); e.g., 'mytext_{0}.txt'
                          will look for mytext_12345.txt for file ID 12345
      log              :: logging object to write to (defaults to dng_logger.log)

    @returns
      mobilities      :: list of Mobility objects
      actions         :: list of Action objects
      assistances     :: list of Assistance objects
      quantifications :: list of Quantification objects
    '''
    documents = []

    mobilities = []
    actions = []
    assistances = []
    quantifications = []

    paired_files = matchAnnotationAndTextFiles(data_directories, text_directory, csv_id_pattern, txt_sub_pattern, log=log)

    log.track(message='  >> Extracted entities from {0:,}/{1:,} files ({2:,} entities)', writeInterval=1)
    for (_id, (csvf, txtf)) in paired_files.items():
        doc = extractAnnotationsFromFile(
            csvf, 
            txtf, 
            as_document=True
        )

        doc.file_path = txtf
        doc.ID = _id

        for m in doc.mobilities:
            m.file_ID = _id
            mobilities.append(m)
        for m in doc.actions:
            m.file_ID = _id
            actions.append(m)
        for m in doc.assistances:
            m.file_ID = _id
            assistances.append(m)
        for m in doc.quantifications:
            m.file_ID = _id
            quantifications.append(m)

        documents.append(doc)

        log.tick(len(paired_files), len(mobilities) + len(actions) + len(assistances) + len(quantifications))
    log.flushTracker(len(paired_files), len(mobilities) + len(actions) + len(assistances) + len(quantifications))

    if by_document:
        return documents
    else:
        return (
            mobilities,
            actions,
            assistances,
            quantifications
        )
