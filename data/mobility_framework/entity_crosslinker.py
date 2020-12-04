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

from hedgepig_logger import log

def crosslinkEntities(mobilities, actions, assistances, quantifications, log=log):
    mobility_by_file_ID = {}
    action_by_file_ID = {}
    assistance_by_file_ID = {}
    quantification_by_file_ID = {}

    file_IDs = set()

    for mob in mobilities:
        if mob.file_ID not in mobility_by_file_ID:
            mobility_by_file_ID[mob.file_ID] = []
        mobility_by_file_ID[mob.file_ID].append(mob)
        file_IDs.add(mob.file_ID)
    for act in actions:
        if act.file_ID not in action_by_file_ID:
            action_by_file_ID[act.file_ID] = []
        action_by_file_ID[act.file_ID].append(act)
        file_IDs.add(act.file_ID)
    for ast in assistances:
        if ast.file_ID not in assistance_by_file_ID:
            assistance_by_file_ID[ast.file_ID] = []
        assistance_by_file_ID[ast.file_ID].append(ast)
        file_IDs.add(ast.file_ID)
    for qnt in quantifications:
        if qnt.file_ID not in quantification_by_file_ID:
            quantification_by_file_ID[qnt.file_ID] = []
        quantification_by_file_ID[qnt.file_ID].append(qnt)
        file_IDs.add(qnt.file_ID)

    for file_ID in file_IDs:
        crosslinkFileEntities(
            mobility_by_file_ID.get(file_ID, []),
            action_by_file_ID.get(file_ID, []),
            assistance_by_file_ID.get(file_ID, []),
            quantification_by_file_ID.get(file_ID, []),
            log=log
        )

def crosslinkFileEntities(mobilities, actions, assistances, quantifications, log=log):
    crosslinkSubentityType(
        mobilities,
        actions,
        lambda mob, act: mob.action.append(act),
        'Action',
        log=log
    )
    crosslinkSubentityType(
        mobilities,
        assistances,
        lambda mob, ast: mob.assistance.append(ast),
        'Assistance',
        log=log
    )
    crosslinkSubentityType(
        mobilities,
        quantifications,
        lambda mob, qnt: mob.quantification.append(qnt),
        'Quantification',
        log=log
    )

def crosslinkSubentityType(mobilities, subentities, child_op, _type, log=log):
    # sort by starting, then ending positions
    sorted_mobilities = sorted(
        mobilities,
        key = lambda mob: (100 * mob.start) + (0.00001 * mob.end)
    )
    sorted_subentities = sorted(
        subentities,
        key = lambda act: (100 * act.start) + (0.00001 * act.end)
    )

    # for each action, find its containing mobility annotation
    for subent in sorted_subentities:
        i = 0
        while i < len(sorted_mobilities) and sorted_mobilities[i].start < subent.start:
            i += 1
        if i >= len(sorted_mobilities) or sorted_mobilities[i].start > subent.start:
            i -= 1

        mob = sorted_mobilities[i]
        if (mob.start > subent.start) or (mob.end < subent.end):
            log.writeln('[WARNING] Failed to map {0} to Mobility, skipping'.format(_type))
        elif (mob.text[subent.start-mob.start:subent.end-mob.start] != subent.text):
            log.writeln('[WARNING] Text mismatch in entity crosslinking: Mobility has text "{0}", {1} has text "{2}"; skipping'.format(mob.text[subent.start-mob.start:subent.end-mob.start], _type, subent.text))
        else:
            subent.mobility = mob
            child_op(mob, subent)
