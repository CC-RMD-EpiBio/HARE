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

class Document:
    
    ID = None
    mobilities = None
    actions = None
    assistances = None
    quantification = None
    file_path = None
    full_text = None

    def __init__(self, mobilities=None, actions=None, quantifications=None,
            assistances=None, full_text=None, file_path=None, ID=None):
        self.mobilities = [] if mobilities is None else mobilities
        self.actions = [] if actions is None else actions
        self.assistances = [] if assistances is None else assistances
        self.quantifications = [] if quantifications is None else quantifications

        self.full_text = full_text
        self.file_path = file_path
        self.ID = ID

class BaseMention:
    
    ID = -1
    file_ID = None
    start = -1
    end = -1
    text = None
    full_text = None

    def __init__(self, **kwargs):
        for (k, v) in kwargs.items():
            self.__dict__[k] = v

class Mobility(BaseMention):
    history = None
    type = None
    subject = None

    action = None
    assistance = None
    quantification = None

    def __init__(self, **kwargs):
        self.action = []
        self.assistance = []
        self.quantification = []
        super().__init__(**kwargs)

    def __repr__(self):
        return '{Mobility: "%s" (%d-%d) History: %s Type: %s Subject: %s}' % (
            self.text, self.start, self.end,
            str(self.history), self.type, self.subject
        )

class Action(BaseMention):
    code = None
    polarity = None
    mobility = None

    def __repr__(self):
        return '{Action: "%s" (%d-%d) Code: %s Polarity: %s }' % (
            self.text, self.start, self.end, self.code, str(self.polarity)
        )

class Quantification(BaseMention):
    type = None
    mobility = None

    def __repr__(self):
        return '{Quantification: "%s" (%d-%d) Type: %s}' % (
            self.text, self.start, self.end, self.type
        )

class Assistance(BaseMention):
    polarity = None
    source = None
    mobility = None

    def __repr__(self):
        return '{Assistance: "%s" (%d-%d) Polarity: %s Source: %s}' % (
            self.text, self.start, self.end,
            self.polarity, self.source
        )
