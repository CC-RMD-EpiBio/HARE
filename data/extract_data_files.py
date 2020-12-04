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
Given a configuration file, extract document text and Mobility mentions,
tokenize (if needed), and generate:
  (1) Tokenized file for ELMo input
  (2) Label file (one per token)
'''

import os
import configparser
import spacy
from . import mobility_framework
import multiprocessing as mp
from hedgepig_logger import log
from .tokenizer import Tokenizer

class _SIGNALS:
    HALT = -1

def getUnannotatedDocuments(config):
    docs = []
    data_dirs = config['DataDirectories'].split(',')
    for data_dir in data_dirs:
        for f in os.listdir(data_dir):
            fpath = os.path.join(data_dir, f)
            if os.path.isfile(fpath):
                docs.append(fpath)
    return docs

def getUnlabeledTokens(config, options, tokenized_lines_q):
    documents = getUnannotatedDocuments(config)

    log.writeln('Tokenizing data from all files...')
    processed_file_IDs = set()

    if not options.verbose:
        log.track('  >> Tokenized {0}/{1:,} files'.format('{0:,}', len(documents)))

    tokenizer = Tokenizer(
        setting=options.tokenizer,
        bert_vocab_file=options.bert_vocab_file
    )

    for i in range(len(documents)):
        doc = documents[i]
        if options.verbose:
            log.writeln('\nTokenizing file {0:,}/{1:,}'.format(i+1, len(documents)))
            log.track('  >> Tokenized {0:,}/{1:,} lines', writeInterval=5)
        doc_ID = os.path.basename(doc)
        tokenized_lines = []
        if options.verbose:
            num_lines = 0
            with open(doc, 'r', errors='replace') as stream:
                for line in stream:
                    num_lines += 1
        with open(doc, 'r', errors='replace') as stream:
            for line in stream:
                line_tokens = tokenizer.tokenize(line.strip())
                tokenized_lines.append(line_tokens)
                if options.verbose:
                    log.tick(num_lines)
        if options.verbose:
            log.flushTracker(num_lines)
        else:
            log.tick()
        tokenized_lines_q.put( (doc_ID, tokenized_lines) )
        processed_file_IDs.add(doc_ID)
    tokenized_lines_q.put(_SIGNALS.HALT)

    if not options.verbose:
        log.flushTracker()

    return processed_file_IDs

def getAnnotatedDocuments(config):
    if config['ExtractionMode'] == 'csv':
        documents = mobility_framework.csv_reader.extractAllEntities(
            config['DataDirectories'].split(','),
            config['PlaintextDirectory'],
            config['CSVIdentifierPattern'],
            config['PlaintextIdentifierPattern'],
            with_full_text=True,
            log=log,
            by_document=True
        )
    elif config['ExtractionMode'] == 'xml':
        documents = mobility_framework.xml_reader.extractAllEntities(
            config['DataDirectories'].split(','),
            with_full_text=True,
            log=log,
            errors='replace',
            by_document=True,
            polarity_type=(int if (config['PolarityType'] == 'int') else str)
        )
    return documents

def labeledTokenize(line, tokenizer, lbl):
    tokens = tokenizer.tokenize(line)
    return [
        (t, lbl)
            for t in tokens
            if len(t.strip()) > 0
    ]

def getLabeledTokens(config, options, tokenized_lines_q):
    tokenizer = Tokenizer(
        setting=options.tokenizer,
        bert_vocab_file=options.bert_vocab_file
    )
    documents = getAnnotatedDocuments(config)

    log.writeln('Tokenizing data from all files...')
    log.track('  >> Tokenized {0}/{1:,} files'.format('{0:,}', len(documents)))
    file_IDs = set()
    for doc in documents:
        sorted_mobilities = sorted(
            doc.mobilities,
            key=lambda m: m.start
        )

        if len(sorted_mobilities) > 0:
            mobility_bounds = []
            cur_mobility = sorted_mobilities[0]
            cur_bounds = [cur_mobility.start]

            for i in range(1, len(sorted_mobilities)):
                if sorted_mobilities[i].start > cur_mobility.end:
                    cur_bounds.append(cur_mobility.end)
                    mobility_bounds.append(cur_bounds)

                    cur_mobility = sorted_mobilities[i]
                    cur_bounds = [cur_mobility.start]
                elif sorted_mobilities[i].end > cur_mobility.end:
                    cur_mobility = sorted_mobilities[i]
            cur_bounds.append(cur_mobility.end)
            mobility_bounds.append(cur_bounds)

            labeled_tokenized_lines = []
            cur_line = []
            for i in range(len(mobility_bounds)):
                (start, end) = mobility_bounds[i]
                if i == 0:
                    prefix = doc.full_text[:start]
                else:
                    prefix = doc.full_text[mobility_bounds[i-1][1]:start]

                prefix_lines = [s.strip() for s in prefix.split('\n')]
                for j in range(len(prefix_lines)):
                    cur_line.extend(labeledTokenize(
                        prefix_lines[j], tokenizer, None
                    ))
                    if j < len(prefix_lines)-1:
                        labeled_tokenized_lines.append(list(cur_line))
                        cur_line = []

                mobility_text = doc.full_text[start:end]
                mobility_lines = [s.strip() for s in mobility_text.split('\n')]
                for j in range(len(mobility_lines)):
                    cur_line.extend(labeledTokenize(
                        mobility_lines[j], tokenizer, 'M'
                    ))
                    if i < len(mobility_lines)-1:
                        labeled_tokenized_lines.append(list(cur_line))
                        cur_line = []

                if i == len(mobility_bounds) - 1:
                    suffix = doc.full_text[end:]
                    suffix_lines = [s.strip() for s in suffix.split('\n')]
                    for j in range(len(suffix_lines)):
                        cur_line.extend(labeledTokenize(
                            suffix_lines[j], tokenizer, None
                        ))
                        labeled_tokenized_lines.append(list(cur_line))
                        cur_line = []
        else:
            all_lines = doc.full_text.split('\n')
            labeled_tokenized_lines = [
                labeledTokenize(line, tokenizer, None)
                    for line in all_lines
            ]

        tokenized_lines_q.put( (doc.ID, labeled_tokenized_lines) )
        file_IDs.add(doc.ID)
        log.tick()
    log.flushTracker()
    tokenized_lines_q.put(_SIGNALS.HALT)

    return file_IDs

def _threadedWriteOutputFiles(tokenized_lines_q, outf, labeled=True):
    line_index = 0
    num_files = 0

    tokenf = '%s.tokens' % outf
    labelf = '%s.labels' % outf
    with open(tokenf, 'w') as token_stream, \
         open(labelf, 'w') as label_stream:
        result = tokenized_lines_q.get()
        while result != _SIGNALS.HALT:
            num_files += 1
            (file_ID, tokenized_lines) = result
            for tokens in tokenized_lines:
                for i in range(len(tokens)):
                    if labeled:
                        (t, lbl) = tokens[i]
                    else:
                        (t, lbl) = (tokens[i], None)
                    # skip all empty tokens
                    if len(t.strip()) == 0:
                        continue

                    token_stream.write('%s' % t)
                    if i < len(tokens) - 1:
                        token_stream.write(' ')

                    label_stream.write('%s\t%d\t%s\t%s\n' % (
                        str(file_ID),
                        line_index,
                        t,
                        '' if lbl is None else lbl
                    ))

                if len(tokens) > 0:
                    token_stream.write('\n')
                else:
                    token_stream.write('<BR>\n')
                    label_stream.write('%s\t%d\t<BR>\t\n' % (str(file_ID), line_index))

                line_index += 1

            token_stream.write('<EOF>\n')
            label_stream.write('%s\t%d\t<EOF>\t\n' % (str(file_ID), line_index))

            line_index += 1
            result = tokenized_lines_q.get()
    log.writeln('Wrote output for {0:,} files to'.format(num_files))
    log.writeln('  Tokens: {0}'.format(tokenf))
    log.writeln('  Labels: {0}'.format(labelf))

if __name__ == '__main__':
    def _cli():
        import optparse
        parser = optparse.OptionParser(usage='Usage: %prog OUTF')
        parser.add_option('-c', '--config', dest='configf',
                help='configuration file')
        parser.add_option('--dataset', dest='dataset',
                help='dataset to extract features for (matches section name in config file)')
        parser.add_option('-t', '--tokenizer', dest='tokenizer',
                type='choice', default=Tokenizer.default(), choices=Tokenizer.choices())
        parser.add_option('--BERT-vocab-file', dest='bert_vocab_file',
                help='vocabulary file for BERT tokenization'
                     ' (REQUIRED IFF --tokenizer=BERT, otherwise ignored)')
        parser.add_option('-v', '--verbse', dest='verbose',
                action='store_true', default=False,
                help='verbose logging for file tokenization')
        parser.add_option('-l', '--logfile', dest='logfile',
                help='name of file to write log contents to (empty for stdout)',
                default=None)
        (options, args) = parser.parse_args()

        def _bail(msg):
            print(msg)
            print('')
            parser.print_help()
            exit()

        if (not options.configf) or (not os.path.exists(options.configf)):
            _bail('Must supply valid value for --config')
        elif len(args) != 1:
            _bail('Must supply OUTF')

        return args, options
    (outf,), options = _cli()

    log.start(options.logfile)
    log.writeConfig([
        ('Config file', options.configf),
        ('Dataset', options.dataset),
        ('Tokenizer', options.tokenizer),
        ('BERT vocab file', 'N/A' if options.tokenizer != Tokenizer.BERT else options.bert_vocab_file),
        ('Output file', outf),
        ('Verbose logging', options.verbose),
    ], 'Mobility token/label extraction')

    log.writeln('Reading configuration file %s...' % options.configf)
    config = configparser.ConfigParser()
    config.read(options.configf)
    config = config[options.dataset]
    log.writeln('Done.\n')

    if config['ExtractionMode'] == 'unannotated':
        labeled = False
    else:
        labeled = True

    tokenized_lines_q = mp.Queue()
    t_writer = mp.Process(
        target=_threadedWriteOutputFiles,
        args=(tokenized_lines_q, outf, labeled)
    )
    t_writer.start()

    if config['ExtractionMode'] == 'unannotated':
        log.writeln('Extracting unlabeled tokens...')
        processed_file_IDs = getUnlabeledTokens(config, options, tokenized_lines_q)
        log.writeln('Extracted tokens from {0:,} documents.\n'.format(len(processed_file_IDs)))
        labeled = False
    else:
        log.writeln('Extracting labeled tokens...')
        processed_file_IDs = getLabeledTokens(config, options, tokenized_lines_q)
        log.writeln('Extracted tokens from {0:,} documents.\n'.format(len(processed_file_IDs)))
        labeled = True

    t_writer.join()

    log.stop()
