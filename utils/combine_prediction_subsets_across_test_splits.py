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

import os
from evaluation import predictions_io
from hedgepig_logger import log

def readFoldMembership(f):
    membership, docs = {}, set()
    with open(f, 'r') as stream:
        for line in stream:
            (file_ID, fold_num) = line.split(',')
            fold_num = int(fold_num)
            if not fold_num in membership:
                membership[fold_num] = set()
            membership[fold_num].add(file_ID)
            docs.add(file_ID)
    return membership, len(docs)

def getPredictionSubset(predictions, file_IDs):
    subset = []
    for p in predictions:
        if os.path.splitext(p.doc_ID)[0] in file_IDs:
            subset.append(p)
    return subset

if __name__ == '__main__':
    def _cli():
        import optparse
        parser = optparse.OptionParser(usage='Usage: %prog')
        parser.add_option('-e', '--experiment', dest='experiment_d',
            help='experiment output directory')
        parser.add_option('-m', '--membership', dest='membership_f',
            help='file mapping doc IDs to fold membership')
        parser.add_option('-k', '--num-folds', dest='num_folds',
            type='int', default=10,
            help='number of folds to pull predictions from (default %default)')
        parser.add_option('-p', '--predictions-pattern', dest='predictions_pattern',
            help='filename pattern (using {0} for fold number) for per-fold'
                 ' predictions in --experiment')
        parser.add_option('-o', '--output', dest='output_f',
            help='file to write combined predictions to')
        parser.add_option('-l', '--logfile', dest='logfile',
            help='name of file to write log contents to (empty for stdout)',
            default=None)
        (options, args) = parser.parse_args()
        if not options.experiment_d:
            parser.print_help()
            parser.error('Must provide --experiment')
        elif not options.membership_f:
            parser.print_help()
            parser.error('Must provide --membership')
        elif not options.output_f:
            parser.print_help()
            parser.error('Must provide --output')
        elif not options.predictions_pattern:
            parser.print_help()
            parser.error('Must provide --predictions-pattern')
        return options
    options = _cli()
    log.start(options.logfile)
    log.writeConfig([
        ('Experimental output directory (per-fold)', options.experiment_d),
        ('Doc membership mapping file', options.membership_f),
        ('Number of folds', options.num_folds),
        ('Predictions file pattern', options.predictions_pattern),
        ('Output file', options.output_f),
    ], 'Combining prediction subsets across test splits')

    log.writeln('Reading fold membership from %s...' % options.membership_f)
    membership, num_docs = readFoldMembership(options.membership_f)
    log.writeln('Found membership of {0} documents in {1} folds.\n'.format(
        num_docs, len(membership)
    ))

    output_predictions = []
    for i in range(options.num_folds):
        log.writeln('Processing fold {0}/{1}'.format(i+1, options.num_folds))
        log.indent()

        preds_f = os.path.join(
            options.experiment_d,
            options.predictions_pattern.format(i)
        )
        log.writeln('Reading predictions from %s...' % preds_f)
        preds = predictions_io.readSamplePredictions(preds_f)
        log.writeln('Read {0:,} predictions.\n'.format(len(preds)))

        log.writeln('Filtering to specified doc IDs for this fold...')
        doc_IDs = membership.get(i, set())
        sub_preds = getPredictionSubset(preds, doc_IDs)
        log.writeln('Filtered to {0:,} predictions from {1} doc IDs.\n'.format(
            len(sub_preds), len(doc_IDs)
        ))

        log.unindent()
        output_predictions.extend(sub_preds)

    log.writeln('Writing combined predictions to %s...' % options.output_f)
    with open(options.output_f, 'w') as stream:
        for p in output_predictions:
            predictions_io.writeSamplePrediction(
                stream,
                p,
                p.positive_probability
            )
    log.writeln('Wrote {0:,} predictions.\n'.format(len(output_predictions)))

    log.stop()
