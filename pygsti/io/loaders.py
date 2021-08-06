"""
Functions for loading GST objects from text files.
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import os as _os
import pathlib as _pathlib
import warnings as _warnings

from pygsti.io import metadir as _metadir
from pygsti.io import stdinput as _stdinput
from pygsti import baseobjs as _baseobjs
from pygsti import circuits as _circuits
from pygsti import data as _data


def load_dataset(filename, cache=False, collision_action="aggregate",
                 record_zero_counts=True, ignore_zero_count_lines=True,
                 with_times="auto", circuit_parse_cache=None, verbosity=1):
    """
    Load a DataSet from a file.

    This function first tries to load file as a saved DataSet object,
    then as a standard text-formatted DataSet.

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

    collision_action : {"aggregate", "keepseparate"}
        Specifies how duplicate circuits should be handled.  "aggregate"
        adds duplicate-circuit counts, whereas "keepseparate" tags duplicate
        circuits by setting their `.occurrence` IDs to sequential positive integers.

    record_zero_counts : bool, optional
        Whether zero-counts are actually recorded (stored) in the returned
        DataSet.  If False, then zero counts are ignored, except for potentially
        registering new outcome labels.  When reading from a cache file
        (using `cache==True`) this argument is ignored: the presence of zero-
        counts is dictated by the value of `record_zero_counts` when the cache file
        was created.

    ignore_zero_count_lines : bool, optional
        Whether circuits for which there are no counts should be ignored
        (i.e. omitted from the DataSet) or not.

    with_times : bool or "auto", optional
        Whether to the time-stamped data format should be read in.  If
        "auto", then the time-stamped format is allowed but not required on a
        per-circuit basis (so the dataset can contain both formats).  Typically
        you only need to set this to False when reading in a template file.

    circuit_parse_cache : dict, optional
        A dictionary mapping qubit string representations into created
        :class:`Circuit` objects, which can improve performance by reducing
        or eliminating the need to parse circuit strings.

    verbosity : int, optional
        If zero, no output is shown.  If greater than zero,
        loading progress is shown.

    Returns
    -------
    DataSet
    """

    printer = _baseobjs.VerbosityPrinter.create_printer(verbosity)
    try:
        # a saved Dataset object is ok
        ds = _data.DataSet(file_to_load_from=filename)
    except:

        #Parser functions don't take a VerbosityPrinter yet, and so
        # always output to stdout (TODO)
        bToStdout = (printer.verbosity > 0 and printer.filename is None)

        if cache:
            #bReadCache = False
            cache_filename = filename + ".cache"
            if _os.path.exists(cache_filename) and \
               _os.path.getmtime(filename) < _os.path.getmtime(cache_filename):
                try:
                    printer.log("Loading from cache file: %s" % cache_filename)
                    ds = _data.DataSet(file_to_load_from=cache_filename)
                    return ds
                except: print("WARNING: Failed to load from cache file")  # pragma: no cover
            else:
                printer.log("Cache file not found or is tool old -- one will"
                            + "be created after loading is completed")

            # otherwise must use standard dataset file format
            parser = _stdinput.StdInputParser()
            ds = parser.parse_datafile(filename, bToStdout,
                                       collision_action=collision_action,
                                       record_zero_counts=record_zero_counts,
                                       ignore_zero_count_lines=ignore_zero_count_lines,
                                       with_times=with_times)

            printer.log("Writing cache file (to speed future loads): %s"
                        % cache_filename)
            ds.save(cache_filename)
        else:
            # otherwise must use standard dataset file format
            parser = _stdinput.StdInputParser()
            ds = parser.parse_datafile(filename, bToStdout,
                                       collision_action=collision_action,
                                       record_zero_counts=record_zero_counts,
                                       ignore_zero_count_lines=ignore_zero_count_lines,
                                       with_times=with_times)
        return ds


def load_multidataset(filename, cache=False, collision_action="aggregate",
                      record_zero_counts=True, verbosity=1):
    """
    Load a MultiDataSet from a file.

    This function first tries to load file as a saved MultiDataSet object,
    then as a standard text-formatted MultiDataSet.

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

    collision_action : {"aggregate", "keepseparate"}
        Specifies how duplicate circuits should be handled.  "aggregate"
        adds duplicate-circuit counts, whereas "keepseparate" tags duplicate
        circuits by setting their `.occurrence` IDs to sequential positive integers.

    record_zero_counts : bool, optional
        Whether zero-counts are actually recorded (stored) in the returned
        MultiDataSet.  If False, then zero counts are ignored, except for
        potentially registering new outcome labels.  When reading from a cache
        file (using `cache==True`) this argument is ignored: the presence of
        zero-counts is dictated by the value of `record_zero_counts` when the cache
        file was created.

    verbosity : int, optional
        If zero, no output is shown.  If greater than zero,
        loading progress is shown.

    Returns
    -------
    MultiDataSet
    """

    printer = _baseobjs.VerbosityPrinter.create_printer(verbosity)
    try:
        # a saved MultiDataset object is ok
        mds = _data.MultiDataSet(file_to_load_from=filename)
    except:

        #Parser functions don't take a VerbosityPrinter yet, and so
        # always output to stdout (TODO)
        bToStdout = (printer.verbosity > 0 and printer.filename is None)

        if cache:
            # bReadCache = False
            cache_filename = filename + ".cache"
            if _os.path.exists(cache_filename) and \
               _os.path.getmtime(filename) < _os.path.getmtime(cache_filename):
                try:
                    printer.log("Loading from cache file: %s" % cache_filename)
                    mds = _data.MultiDataSet(file_to_load_from=cache_filename)
                    return mds
                except: print("WARNING: Failed to load from cache file")  # pragma: no cover
            else:
                printer.log("Cache file not found or is too old -- one will be"
                            + "created after loading is completed")

            # otherwise must use standard dataset file format
            parser = _stdinput.StdInputParser()
            mds = parser.parse_multidatafile(filename, bToStdout,
                                             collision_action=collision_action,
                                             record_zero_counts=record_zero_counts)

            printer.log("Writing cache file (to speed future loads): %s"
                        % cache_filename)
            mds.save(cache_filename)

        else:
            # otherwise must use standard dataset file format
            parser = _stdinput.StdInputParser()
            mds = parser.parse_multidatafile(filename, bToStdout,
                                             collision_action=collision_action,
                                             record_zero_counts=record_zero_counts)
    return mds


def load_time_dependent_dataset(filename, cache=False, record_zero_counts=True):
    """
    Load time-dependent (time-stamped) data as a DataSet.

    Parameters
    ----------
    filename : string
        The name of the file

    cache : bool, optional
        Reserved to perform caching similar to `load_dataset`.  Currently
        this argument doesn't do anything.

    record_zero_counts : bool, optional
        Whether zero-counts are actually recorded (stored) in the returned
        DataSet.  If False, then zero counts are ignored, except for
        potentially registering new outcome labels.

    Returns
    -------
    DataSet
    """
    parser = _stdinput.StdInputParser()
    create_subcircuits = not _circuits.Circuit.default_expand_subcircuits
    tdds = parser.parse_tddatafile(filename, record_zero_counts=record_zero_counts,
                                   create_subcircuits=create_subcircuits)
    return tdds


def load_model(filename):
    """
    Load a Model from a file, formatted using the standard text-format for models.

    Parameters
    ----------
    filename : string
        The name of the file

    Returns
    -------
    Model
    """
    return _stdinput.parse_model(filename)


def load_circuit_dict(filename):
    """
    Load a circuit dictionary from a file, formatted using the standard text-format.

    Parameters
    ----------
    filename : string
        The name of the file.

    Returns
    -------
    Dictionary with keys = circuit labels and values = :class:`Circuit` objects.
    """
    std = _stdinput.StdInputParser()
    return std.parse_dictfile(filename)


def load_circuit_list(filename, read_raw_strings=False, line_labels='auto', num_lines=None):
    """
    Load a circuit list from a file, formatted using the standard text-format.

    Parameters
    ----------
    filename : string
        The name of the file

    read_raw_strings : boolean
        If True, circuits are not converted to :class:`Circuit` objects.

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
    if read_raw_strings:
        rawList = []
        with open(str(filename), 'r') as circuitlist:
            for line in circuitlist:
                if len(line.strip()) == 0: continue
                if len(line) == 0 or line[0] == '#': continue
                rawList.append(line.strip())
        return rawList
    else:
        create_subcircuits = not _circuits.Circuit.default_expand_subcircuits
        std = _stdinput.StdInputParser()
        return std.parse_stringfile(filename, line_labels, num_lines, create_subcircuits)


def load_protocol_from_dir(dirname, quick_load=False, comm=None):
    """
    Load a :class:`Protocol` from a directory on disk.

    Parameters
    ----------
    dirname : string
        Directory name.

    quick_load : bool, optional
        Setting this to True skips the loading of components that may take
        a long time to load. This can be useful when this information isn't
        needed and loading takes a long time.

    comm : mpi4py.MPI.Comm, optional
        When not ``None``, an MPI communicator used to synchronize file access.

    Returns
    -------
    Protocol
    """
    dirname = _pathlib.Path(dirname)
    return _metadir._cls_from_meta_json(dirname).from_dir(dirname, quick_load=quick_load)


def load_edesign_from_dir(dirname, quick_load=False, comm=None):
    """
    Load a :class:`ExperimentDesign` from a directory on disk.

    Parameters
    ----------
    dirname : string
        Directory name.

    quick_load : bool, optional
        Setting this to True skips the loading of components that may take
        a long time to load. This can be useful when this information isn't
        needed and loading takes a long time.

    comm : mpi4py.MPI.Comm, optional
        When not ``None``, an MPI communicator used to synchronize file access.

    Returns
    -------
    ExperimentDesign
    """
    dirname = _pathlib.Path(dirname)
    return _metadir._cls_from_meta_json(dirname / 'edesign').from_dir(dirname, quick_load=quick_load)


def create_edesign_from_dir(dirname):
    from .. import protocols as _proto
    topdir = _pathlib.Path(dirname)
    edesign_dir = topdir / 'edesign'
    circuit_lists = []; circuit_list_names = []

    if edesign_dir.is_dir():
        if (edesign_dir / 'meta.json').exists():  # load existing edesign
            return _metadir._cls_from_meta_json(dirname / 'edesign').from_dir(dirname, quick_load=False)

        # Find any circuit list files in the edesign directory
        for child in sorted(edesign_dir.iterdir()):
            if child.is_file():
                try:
                    lst = load_circuit_list(child, read_raw_strings=False, line_labels='auto')
                    circuit_lists.append(lst); circuit_list_names.append(child.name)
                except Exception:
                    pass

    #Otherwise see if we should recurse or not
    subdirs = []
    for child in topdir.iterdir():
        if child == edesign_dir: continue  # special case, shouldn't be strictly needed
        if child.is_dir() and (child / 'edesign').is_dir():
            subdirs.append(child)

    sub_edesigns = [create_edesign_from_dir(subdir) for subdir in subdirs]
    if len(sub_edesigns) > 0:
        if len(circuit_lists) > 0:
            _warnings.warn("Ignoring %d circuit-list files [%s] in %d because sub-designs were detected." %
                           (len(circuit_lists), ", ".join(circuit_list_names), edesign_dir.name))
        return _proto.CombinedExperimentDesign({subdir.name: sub_edesign
                                                for subdir, sub_edesign in zip(subdirs, sub_edesigns)})
    elif len(circuit_lists) > 1:
        return _proto.CircuitListsDesign(circuit_lists)
    elif len(circuit_lists) == 1:
        return _proto.ExperimentDesign(circuit_lists[0])
    else:
        raise ValueError("Could not create an experiment design from the files in this directory!")


def load_data_from_dir(dirname, quick_load=False, comm=None):
    """
    Load a :class:`ProtocolData` from a directory on disk.

    Parameters
    ----------
    dirname : string
        Directory name.

    quick_load : bool, optional
        Setting this to True skips the loading of components that may take
        a long time to load. This can be useful when this information isn't
        needed and loading takes a long time.

    comm : mpi4py.MPI.Comm, optional
        When not ``None``, an MPI communicator used to synchronize file access.

    Returns
    -------
    ProtocolData
    """
    dirname = _pathlib.Path(dirname)
    try:
        protocol_data = _metadir._cls_from_meta_json(dirname / 'data')
    except FileNotFoundError:
        from ..protocols import ProtocolData as _ProtocolData
        protocol_data = _ProtocolData  # use ProtocolData as default class
    return protocol_data.from_dir(dirname, quick_load=quick_load)


def load_results_from_dir(dirname, name=None, preloaded_data=None, quick_load=False, comm=None):
    """
    Load a :class:`ProtocolResults` or :class:`ProtocolsResultsDir` from a directory on disk.

    Which object type is loaded depends on whether `name` is given: if it is, then
    a :class:`ProtocolResults` object is loaded.  If not, a :class:`ProtocolsResultsDir`
    results.

    Parameters
    ----------
    dirname : string
        Directory name.  This should be a "base" directory, containing
        subdirectories like "edesign", "data", and "results"

    name : string or None
        The 'name' of a particular :class:`ProtocolResults` object, which
        is a sub-directory beneath `dirname/results/`.  If None, then *all*
        the results (all names) at the given base-directory are loaded and
        returned as a :class:`ProtocolResultsDir` object.

    preloaded_data : ProtocolData, optional
        The data object belonging to the to-be-loaded results, in cases
        when this has been loaded already (only use this if you know what
        you're doing).

    quick_load : bool, optional
        Setting this to True skips the loading of data and experiment-design
        components that may take a long time to load. This can be useful
        all the information of interest lies only within the results objects.

    comm : mpi4py.MPI.Comm, optional
        When not ``None``, an MPI communicator used to synchronize file access.

    Returns
    -------
    ProtocolResults or ProtocolResultsDir
    """
    from ..protocols import ProtocolResultsDir as _ProtocolResultsDir
    dirname = _pathlib.Path(dirname)
    results_dir = dirname / 'results'
    if name is None:  # then it's a directory object
        cls = _metadir._cls_from_meta_json(results_dir) if (results_dir / 'meta.json').exists() \
            else _ProtocolResultsDir  # default if no meta.json (if only a results obj has been written inside dir)
        return cls.from_dir(dirname, preloaded_data=preloaded_data, quick_load=quick_load)
    else:  # it's a ProtocolResults object
        return _metadir._cls_from_meta_json(results_dir / name).from_dir(dirname, name, preloaded_data, quick_load)
