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
'''

import os
import codecs
from bs4 import BeautifulSoup
import html.parser
from .models import *
from hedgepig_logger import log

class XMLEntityExtractor:
    
    def extractMentions(self, f, with_full_text=False, errors='strict', as_document=False, polarity_type=int):
        self._with_full_text = with_full_text
        self._polarity_type = polarity_type

        with codecs.open(f, 'r', 'utf-8', errors=errors) as stream:
            contents = stream.read()

        soup = BeautifulSoup(contents, 'lxml')

        text_with_nodes = soup.find_all('textwithnodes')
        if len(text_with_nodes) == 0:
            raise Exception("Failed to find <TextWithNodes> elements in %s" % f)
        text_with_nodes = text_with_nodes[0]
        str_text_with_nodes = str(text_with_nodes) \
            .replace('<textwithnodes>', '') \
            .replace('</textwithnodes>', '')

        # since we don't need to actually parse this as HTML, can unencode
        # HTML encodings to make string indices match up between
        # str(text_with_nodes) and text_with_nodes.text
        html_parser = html.parser.HTMLParser()
        str_text_with_nodes = html_parser.unescape(str_text_with_nodes)

        self._node_positions = {}

        cur_text_position = 0
        try: ix = str_text_with_nodes.index('<node')
        except: ix = -1

        while ix > -1:
            node_end_ix = str_text_with_nodes[ix:].index('</node>') + len('</node>')
            node_txt = str_text_with_nodes[ix:ix+node_end_ix]

            str_text_with_nodes = str_text_with_nodes[ix+node_end_ix:]
            cur_text_position += ix

            node_id = int(node_txt.split('"')[1])
            self._node_positions[node_id] = cur_text_position

            try: ix = str_text_with_nodes.index('<node')
            except: ix = -1

        self._doc_text = text_with_nodes.text

        annot_sets = soup.find_all('annotationset')
        for annot_set in annot_sets:
            if 'name' in annot_set and annot_set['name'] == 'ICF':
                break

        annot_set = annot_set.find_all('annotation')
        mobilities = []
        actions = []
        assistances = []
        quantifications = []

        for annot in annot_set:
            if annot['type'].lower() == 'mobility':
                mobilities.append(
                    self.parseMobility(annot)
                )
            elif annot['type'].lower() == 'action':
                actions.append(
                    self.parseAction(annot)
                )
            elif annot['type'].lower() == 'assistance':
                assistances.append(
                    self.parseAssistance(annot)
                )
            elif annot['type'].lower() == 'quantification':
                quantifications.append(
                    self.parseQuantification(annot)
                )

        if as_document:
            return Document(
                mobilities=mobilities,
                actions=actions,
                assistances=assistances,
                quantifications=quantifications,
                full_text=self._doc_text
            )
        else:
            return (
                mobilities,
                actions,
                assistances,
                quantifications
            )

    def baseParse(self, xml):
        start_node = int(xml['startnode'])
        end_node = int(xml['endnode'])

        start_pos = self._node_positions[start_node]
        end_pos = self._node_positions[end_node]

        text = self._doc_text[start_pos:end_pos]

        base_args = {
            'start': start_pos,
            'end': end_pos,
            'text': text
        }
        if self._with_full_text:
            base_args['full_text'] = self._doc_text

        return base_args

    def parseAction(self, action_xml):
        args = self.baseParse(action_xml)

        code = None
        polarity = None

        features = action_xml.find_all('feature')
        for feature in features:
            name = feature.find_all('name')[0].text
            if name == 'Subdomain Code':
                code = feature.find_all('value')[0].text
            elif name == 'Polarity':
                polarity = self._polarity_type(feature.find_all('value')[0].text)

        args['code'] = code,
        args['polarity'] = polarity
        return Action(**args)
    
    def parseMobility(self, mobility_xml):
        args = self.baseParse(mobility_xml)

        history = None
        _type = None
        subject = None

        features = mobility_xml.find_all('feature')
        for feature in features:
            name = feature.find_all('name')[0].text
            if name == 'History':
                history = int(feature.find_all('value')[0].text)
            elif name == 'Type':
                _type = feature.find_all('value')[0].text
            elif name == 'Subject':
                subject = feature.find_all('value')[0].text

        args['type'] = _type
        args['history'] = history,
        args['subject'] = subject

        return Mobility(**args)

    def parseQuantification(self, quant_xml):
        args = self.baseParse(quant_xml)

        _type = None

        features = quant_xml.find_all('feature')
        for feature in features:
            name = feature.find_all('name')[0].text
            if name == 'Type':
                _type = feature.find_all('value')[0].text

        args['type'] = _type

        return Quantification(**args)
    
    def parseAssistance(self, asst_xml):
        args = self.baseParse(asst_xml)

        polarity = None
        source = None

        features = asst_xml.find_all('feature')
        for feature in features:
            name = feature.find_all('name')[0].text
            if name == 'Polarity':
                polarity = feature.find_all('value')[0].text
            elif name == 'Source':
                source = feature.find_all('value')[0].text

        args['polarity'] = polarity,
        args['source'] = source

        return Assistance(**args)

def extractAllEntities(data_directories, log=log, with_full_text=False,
        errors='strict', by_document=False, polarity_type=int):
    '''
    Extract all Mobility, Action, Assistance, and Quantification entities from
    XML-formatted annotation files.

    @parameters
      data_directories :: list of directories containing .xml annotation files
      with_full_text   :: includes full document text in "full_text" field of each object
      log              :: logging object to write to (defaults to dng_logger.log)

    @returns
      mobilities      :: list of Mobility objects
      actions         :: list of Action objects
      assistances     :: list of Assistance objects
      quantifications :: list of Quantification objects
    '''
    mobilities = []
    actions = []
    assistances = []
    quantifications = []

    documents = []

    extractor = XMLEntityExtractor()

    for dir_path in data_directories:
        files = os.listdir(dir_path)

        log.writeln('Extracting data from %s...' % dir_path)
        log.track(message='  >> Extracted entities from {0:,}/{1:,} files ({2:,} entities)', writeInterval=1)

        for f in files:
            fpath = os.path.join(dir_path, f)
            doc = extractor.extractMentions(
                fpath,
                with_full_text=with_full_text,
                errors=errors,
                polarity_type=polarity_type,
                as_document=True
            )

            doc.file_path = fpath
            doc.ID = f

            for m in doc.mobilities:
                m.file_ID = f
                mobilities.append(m)
            for m in doc.actions:
                m.file_ID = f
                actions.append(m)
            for m in doc.assistances:
                m.file_ID = f
                assistances.append(m)
            for m in doc.quantifications:
                m.file_ID = f
                quantifications.append(m)

            documents.append(doc)

            log.tick(len(files), len(mobilities) + len(actions) + len(assistances) + len(quantifications))
        log.flushTracker(len(files), len(mobilities) + len(actions) + len(assistances) + len(quantifications))

    if by_document:
        return documents
    else:
        return (
            mobilities,
            actions,
            assistances,
            quantifications
        )
