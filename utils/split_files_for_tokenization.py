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

import os
import shutil
from hedgepig_logger import log

if __name__ == '__main__':
    def _cli():
        import optparse
        parser = optparse.OptionParser(usage='Usage: %prog FULL CHUNKED')
        parser.add_option('--chunk-size', dest='chunk_size',
            type='int', default=100,
            help='number of files to put in each chunk (default %default)')
        parser.add_option('-l', '--logfile', dest='logfile',
            help='name of file to write log contents to (empty for stdout)',
            default=None)

        (options, args) = parser.parse_args()

        if len(args) != 2:
            parser.error('Must provide FULL and CHUNKED')

        return args, options
    (full_dir, chunked_dir), options = _cli()

    if not os.path.exists(chunked_dir):
        os.mkdir(chunked_dir)

    log.start(options.logfile)
    log.writeConfig([
        ('Full directory', full_dir),
        ('Chunked directory', chunked_dir),
        ('Chunk size', options.chunk_size)
    ])

    log.track('  >> Processed {0:,} files ({1:,} chunks)')
    all_files = os.listdir(full_dir)
    cur_chunk, cur_chunk_dir, cur_chunk_size = -1, None, options.chunk_size
    for i in range(len(all_files)):
        if cur_chunk_size >= options.chunk_size:
            cur_chunk += 1
            cur_chunk_dir = os.path.join(chunked_dir, str(cur_chunk))
            if not os.path.exists(cur_chunk_dir):
                os.mkdir(cur_chunk_dir)
            cur_chunk_size = 0
        next_f = all_files[i]
        src_path = os.path.join(full_dir, next_f)
        dst_path = os.path.join(cur_chunk_dir, next_f)
        shutil.copy(src_path, dst_path)
        cur_chunk_size += 1

        log.tick(cur_chunk+1)
    log.flushTracker(cur_chunk+1)

    log.stop()
