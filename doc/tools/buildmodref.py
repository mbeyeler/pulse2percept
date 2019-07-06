#!/usr/bin/env python
"""Script to auto-generate our API docs.
"""
from __future__ import print_function, division

# stdlib imports
import sys
import re
from os.path import join as pjoin

# local imports
from apigen import ApiDocWriter

# version comparison
from distutils.version import LooseVersion as V

# *****************************************************************************


def abort(error):
    print('*WARNING* API documentation not generated: %s' % error)
    exit()


if __name__ == '__main__':
    package = sys.argv[1]
    outdir = sys.argv[2]
    try:
        other_defines = sys.argv[3]
    except IndexError:
        other_defines = True
    else:
        other_defines = other_defines in ('True', 'true', '1')

    # Check that the package is available. If not, the API documentation is not
    # (re)generated and existing API documentation sources will be used.

    try:
        __import__(package)
    except ImportError as e:
        abort("Can not import " + package)

    module = sys.modules[package]

    installed_version = V(module.__version__)

    docwriter = ApiDocWriter(package, rst_extension='.rst',
                             other_defines=other_defines)
    docwriter.package_skip_patterns += [r'\.fixes$',
                                        r'\.externals$',
                                        r'\.tracking\.interfaces.*$',
                                        r'\.tracking\.gui_tools.*$',
                                        r'.*test.*$',
                                        r'^\.utils.*',
                                        r'\.boots\.resampling.*$',
                                        r'\.fixes.*$',
                                        r'\.info.*$',
                                        r'\.pkg_info.*$',
                                        ]
    docwriter.write_api_docs(outdir)
    docwriter.write_index(outdir, 'index', relative_to=outdir)
    print('%d files written' % len(docwriter.written_modules))