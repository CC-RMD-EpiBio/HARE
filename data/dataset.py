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
Data model for dataset samples
'''

def parseLabelsFileLine(line):
    (doc_ID, line_index, token, lbl) = parsed = [s.strip() for s in line.split('\t')]
    parsed[1] = int(line_index)
    return parsed

class DataPoint:
    
    def __init__(self, ID, doc_ID, token, line_index, token_index, label):
        self.ID = ID
        self.doc_ID = doc_ID
        self.token = token
        self.line_index = line_index
        self.token_index = token_index
        self.label = label

    def __repr__(self):
        return '<DataPoint {0} -- {1}:{2}:{3}; "{4}" {5}>'.format(
            self.ID, self.doc_ID, self.line_index, self.token_index,
            self.token, self.label
        )

class Dataset:
    
    def __init__(self, labels_file=None, labeled=True, exclusion_IDs=None):
        self.samples = []
        self.positive_sample_IDs = set()
        self.negative_sample_IDs = set()
        self.samples_by_doc_ID = {}
        self.labeled = labeled

        if exclusion_IDs:
            self.exclusion_IDs = exclusion_IDs
        else:
            self.exclusion_IDs = set()

        if labels_file:
            self.readLabelsFile(labels_file)

    def filter(self, doc_ID_set):
        doc_ID_set = set(doc_ID_set)
        filtered = Dataset()

        for doc_ID in doc_ID_set:
            sample_IDs = self.samples_by_doc_ID.get(doc_ID, set())
            filtered.samples_by_doc_ID[doc_ID] = sample_IDs

            for s_ID in sample_IDs:
                sample = self.samples_by_ID[s_ID]
                filtered.samples.append(sample)

                if sample.label == 1:
                    filtered.positive_sample_IDs.add(sample.ID)
                else:
                    filtered.negative_sample_IDs.add(sample.ID)

        filtered.samples_by_ID = {
            s.ID : s
                for s in self.samples
        }

        filtered.index()

        return filtered

    def readLabelsFile(self, labels_file, doc_ID_set=None):
        cur_ID = 0
        prev_line_index = None
        cur_token_index = 0
        with open(labels_file, 'r') as stream:
            for line in stream:
                (doc_ID, line_index, token, lbl) = parseLabelsFileLine(line)

                # if rolling over to a new line, reset the token index
                if line_index != prev_line_index:
                    cur_token_index = 0
                else:
                    cur_token_index += 1

                # if viewing control tokens <BR> or <EOF>, skip them
                if token == '<BR>' or token == '<EOF>':
                    continue

                sample = DataPoint(
                    ID=cur_ID,
                    doc_ID=doc_ID,
                    token=token,
                    line_index=line_index,
                    token_index=cur_token_index,
                    label=(
                        1 if lbl == 'M' else 0
                    )
                )

                # only check for the exclusion condition for adding it
                # to the sample list; allows maintenance of consistent
                # IDs with pre-generated files using this dataset
                if not doc_ID in self.exclusion_IDs:
                    self.samples.append(sample)
                    if self.labeled:
                        if sample.label == 1:
                            self.positive_sample_IDs.add(sample.ID)
                        else:
                            self.negative_sample_IDs.add(sample.ID)

                    if not doc_ID in self.samples_by_doc_ID:
                        self.samples_by_doc_ID[doc_ID] = set()
                    self.samples_by_doc_ID[doc_ID].add(sample.ID)

                cur_ID += 1
                prev_line_index = line_index

        self.samples_by_ID = {
            s.ID : s
                for s in self.samples
        }

        self.index()

    def __len__(self):
        return len(self.samples)

    def index(self):
        self.indexed = {}
        for (doc_ID, sample_IDs) in self.samples_by_doc_ID.items():
            self.indexed[doc_ID] = {}
            for s_ID in sample_IDs:
                s = self.samples_by_ID[s_ID]
                if not s.line_index in self.indexed[doc_ID]:
                    self.indexed[doc_ID][s.line_index] = {}
                self.indexed[doc_ID][s.line_index][s.token_index] = s
