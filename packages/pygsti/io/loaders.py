""" Functions for loading GST objects from text files."""
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

import os as _os

from . import stdinput as _stdinput
from .. import objects as _objs

def load_dataset(filename, cache=False, collisionAction="aggregate",
                 verbosity=1):
    """
    Load a DataSet from a file.  First tries to load file as a
    saved DataSet object, then as a standard text-formatted DataSet.

    Parameters
    ----------
    filename : string
        The name of the file

    cache : bool, optional
        When set to True, a pickle file with the name filename + ".cache"
        is searched for and loaded instead of filename if it exists
        and is newer than filename.  If no cache file exists or one
        exists but it is older than filename, a cache file will be
        written after loading from filename.

    collisionAction : {"aggregate", "keepseparate"}
        Specifies how duplicate operation sequences should be handled.  "aggregate"
        adds duplicate-sequence counts, whereas "keepseparate" tags duplicate-
        sequence data with by appending a final "#<number>" operation label to the
        duplicated gate sequence.

    verbosity : int, optional
        If zero, no output is shown.  If greater than zero,
        loading progress is shown.

    Returns
    -------
    DataSet
    """

    printer = _objs.VerbosityPrinter.build_printer(verbosity)
    try:
        # a saved Dataset object is ok
        ds = _objs.DataSet(fileToLoadFrom=filename)
    except:

        #Parser functions don't take a VerbosityPrinter yet, and so
        # always output to stdout (TODO)
        bToStdout = (printer.verbosity > 0 and printer.filename is None)

        if cache:
            #bReadCache = False
            cache_filename = filename + ".cache"
            if _os.path.exists( cache_filename ) and \
               _os.path.getmtime(filename) < _os.path.getmtime(cache_filename):
                try:
                    printer.log("Loading from cache file: %s" % cache_filename)
                    ds = _objs.DataSet(fileToLoadFrom=cache_filename)
                    return ds
                except: print("WARNING: Failed to load from cache file") # pragma: no cover
            else:
                printer.log("Cache file not found or is tool old -- one will"
                            + "be created after loading is completed")

            # otherwise must use standard dataset file format
            parser = _stdinput.StdInputParser()
            ds = parser.parse_datafile(filename, bToStdout,
                                       collisionAction=collisionAction)

            printer.log("Writing cache file (to speed future loads): %s"
                        % cache_filename)
            ds.save(cache_filename)
        else:
            # otherwise must use standard dataset file format
            parser = _stdinput.StdInputParser()
            ds = parser.parse_datafile(filename, bToStdout,
                                       collisionAction=collisionAction)
        return ds


def load_multidataset(filename, cache=False, collisionAction="aggregate",
                      verbosity=1):
    """
    Load a MultiDataSet from a file.  First tries to load file as a
    saved MultiDataSet object, then as a standard text-formatted MultiDataSet.

    Parameters
    ----------
    filename : string
        The name of the file

    cache : bool, optional
        When set to True, a pickle file with the name filename + ".cache"
        is searched for and loaded instead of filename if it exists
        and is newer than filename.  If no cache file exists or one
        exists but it is older than filename, a cache file will be
        written after loading from filename.

    collisionAction : {"aggregate", "keepseparate"}
        Specifies how duplicate operation sequences should be handled.  "aggregate"
        adds duplicate-sequence counts, whereas "keepseparate" tags duplicate-
        sequence data with by appending a final "#<number>" operation label to the
        duplicated gate sequence.

    verbosity : int, optional
        If zero, no output is shown.  If greater than zero,
        loading progress is shown.


    Returns
    -------
    MultiDataSet
    """

    printer = _objs.VerbosityPrinter.build_printer(verbosity)
    try:
        # a saved MultiDataset object is ok
        mds = _objs.MultiDataSet(fileToLoadFrom=filename)
    except:

        #Parser functions don't take a VerbosityPrinter yet, and so
        # always output to stdout (TODO)
        bToStdout = (printer.verbosity > 0 and printer.filename is None)
        
        if cache:
            # bReadCache = False
            cache_filename = filename + ".cache"
            if _os.path.exists( cache_filename ) and \
               _os.path.getmtime(filename) < _os.path.getmtime(cache_filename):
                try:
                    printer.log("Loading from cache file: %s" % cache_filename)
                    mds = _objs.MultiDataSet(fileToLoadFrom=cache_filename)
                    return mds
                except: print("WARNING: Failed to load from cache file") # pragma: no cover
            else:
                printer.log("Cache file not found or is too old -- one will be"
                            + "created after loading is completed")

            # otherwise must use standard dataset file format
            parser = _stdinput.StdInputParser()
            mds = parser.parse_multidatafile(filename, bToStdout,
                                             collisionAction=collisionAction)

            printer.log("Writing cache file (to speed future loads): %s" 
                        % cache_filename)
            mds.save(cache_filename)

        else:
            # otherwise must use standard dataset file format
            parser = _stdinput.StdInputParser()
            mds = parser.parse_multidatafile(filename, bToStdout,
                                             collisionAction=collisionAction)
    return mds


def load_tddataset(filename, cache=False):
    """
    Load a TDDataSet (time-dependent data set) from a file.
    """
    parser = _stdinput.StdInputParser()
    tdds = parser.parse_tddatafile(filename)
    return tdds


def load_model(filename):
    """
    Load a Model from a file, formatted using the
    standard text-format for models.

    Parameters
    ----------
    filename : string
        The name of the file

    Returns
    -------
    Model
    """
    return _stdinput.read_model(filename)

def load_circuit_dict(filename):
    """
    Load a operation sequence dictionary from a file, formatted
    using the standard text-format.

    Parameters
    ----------
    filename : string
        The name of the file.

    Returns
    -------
    Dictionary with keys = operation sequence labels and
      values = Circuit objects.
    """
    std = _stdinput.StdInputParser()
    return std.parse_dictfile(filename)

def load_circuit_list(filename, readRawStrings=False, line_labels='auto', num_lines=None):
    """
    Load a operation sequence list from a file, formatted
    using the standard text-format.

    Parameters
    ----------
    filename : string
        The name of the file

    readRawStrings : boolean
        If True, operation sequences are not converted
        to tuples of operation labels.

    line_labels : iterable, optional
        The (string valued) line labels used to initialize :class:`Circuit`
        objects when line label information is absent from the one-line text 
        representation contained in `filename`.  If `'auto'`, then line labels
        are taken to be the list of all state-space labels present in the 
        circuit's layers.  If there are no such labels then the special value
        `'*'` is used as a single line label.

    num_lines : int, optional
        Specify this instead of `line_labels` to set the latter to the
        integers between 0 and `num_lines-1`.

    Returns
    -------
    list of Circuit objects
    """
    if readRawStrings:
        rawList = []
        with open(filename, 'r') as circuitlist:
            for line in circuitlist:
                if len(line.strip()) == 0: continue
                if len(line) == 0 or line[0] == '#': continue
                rawList.append( line.strip() )
        return rawList
    else:
        std = _stdinput.StdInputParser()
        return std.parse_stringfile(filename, line_labels, num_lines)
