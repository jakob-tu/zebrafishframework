#!/usr/bin/env python

import argparse
from pyprind import prog_percent
import shutil
import os
import sys

from zebrafishframework import util

ALIGNED_SUFFIX = '_aligned.h5'
SHIFTS_SUFFIX = '_shifts.npy'
SUFFIXES = [ALIGNED_SUFFIX, SHIFTS_SUFFIX]

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def run(args):
    if args.files:
        files = args.files
    elif args.listfile:
        with open(args.listfile) as f:
            files = list(map(str.strip, f.readlines()))
    else:
        parser.print_usage()
        sys.exit(0)

    failed = False
    for f in files:
        ext = os.path.splitext(f)
        if not os.path.exists(f):
            eprint('Error: %s does not exist.' % f)
            failed = True
        elif ext[1] != '.lif':
            eprint('Error: %s is not a lif file.' % f)
            failed = True
        elif not os.path.isfile(f):
            eprint('Error: %s is not a file.' % f)
            failed = True
    if failed:
        sys.exit(1)

    if args.copy:
        base = os.path.dirname(args.copy)
        if base != '' and not os.path.isdir(base):
            eprint('Error: "%s" does not exist (-c/--copy file destination).' % base)
            sys.exit(1)

    sub = None
    if args.substitute:
        split = args.substitute.split(':')
        if len(split) != 2:
            eprint('Error: invalid subsitution syntax "%s". Syntax is "replace:with".' % args.substitute)
            failed = True
        sub = split
    if args.destdir:
        if not os.path.isdir(args.destdir):
            eprint('Error: destination dir "%s" does not exist.' % args.destdir)
            failed = True
    if failed:
        sys.exit(1)

    bases = []
    for f in files:
        base_name = os.path.splitext(f)[0]
        if args.destdir:
            base_name = os.path.join(args.destdir, os.path.basename(base_name))
            bases.append(base_name)
        elif sub:
            if base_name.find(sub[0]) == -1:
                eprint('Error: filename "%s" does not contain "%s" for substitution.' % (f, sub[0]))
                failed = True
            base_name = base_name.replace(*sub)
            bases.append(base_name)
        else:
            bases.append(base_name)
    if failed:
        sys.exit(1)

    necessary = []

    if not args.no_verbose:
        print('Arguments look good. This will be processed:')
    for f, b in zip(files, bases):
        this_necessary = not all([os.path.isfile(b + s) for s in SUFFIXES]) or args.overwrite
        necessary.append(this_necessary)

        if not args.no_verbose:
            print(('' if this_necessary else '[SKIP] ') + f)
            for suffix in SUFFIXES:
                print((' -> ' if this_necessary else '[ALREADY EXISTS] ') + '%s%s' % (b, suffix))
            print()

    necessary_files = [(f, b) for f, b, n in zip(files, bases, necessary) if n]

    if len(necessary_files) == 0:
        print('Nothing to process.')
        sys.exit(0)

    for f, b in prog_percent(necessary_files):
        print(f)
        print('='*len(f))

        try:
            if args.copy:
                print('Copying to %s...' % args.copy)
                util.print_time(lambda: shutil.copyfile(f, args.copy))
                src = args.copy
            else:
                src = f

            print('Running alignment...')
            
            # dirty dirty hack
            # use ipython because it can handle javabridge in conda somehow. i don't really know what's going on
            cmd = 'ipython -m zebrafishframework.pyfish "%s" "%s"' % (src, b)
            os.system(cmd)

            if args.copy:
                print('Removing temporary copy')
                os.remove(src)

        except Exception as e:
            print('An exception occured:')
            print(e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Align multiple .lif files with plane-wise translation registration on a CUDA-enabled GPU and save them as .h5.')
    parser.add_argument('files', nargs='*', help='List of source files (*.lif).')
    parser.add_argument('-l', '--listfile', help='File with newline separated filenames.')
    group_dest = parser.add_mutually_exclusive_group()
    group_dest.add_argument('-d', '--destdir', help='Destination directory. If none is provided, destination defaults to the source directory.')
    group_dest.add_argument('-s', '--substitute', help='Substitute a part in the destination path. Syntax is replace:with. For instance: -s lif_files:aligned_files.')
    parser.add_argument('--overwrite', default=False, action='store_true', help='Don\'t skip existing files.')
    parser.add_argument('--copy', metavar='tempfile', help='Copy each file first to specified location. This improves performance if files are on a remote location.', required=False)
    parser.add_argument('--no-verbose', default=False, action='store_true', help='Omit the listing of what will be processed.')
    args = parser.parse_args()

    run(args)
