"""
Functions for writing GST objects to text files.
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import pathlib as _pathlib
import warnings as _warnings
import json as _json

import numpy as _np
import json

from pygsti.io import readers as _loaders
from pygsti import circuits as _circuits
from pygsti.models import gaugegroup as _gaugegroup

# from . import stdinput as _stdinput
from pygsti import tools as _tools
from pygsti.tools.legacytools import deprecate as _deprecated_fn
from pygsti.modelmembers import instruments as _instrument
from pygsti.modelmembers import operations as _op
from pygsti.modelmembers import povms as _povm
from pygsti.modelmembers import states as _state


def write_empty_dataset(filename, circuits,
                        header_string='## Columns = 1 frequency, count total', num_zero_cols=None,
                        append_weights_column=False):
    """
    Write an empty dataset file to be used as a template.

    Parameters
    ----------
    filename : string
        The filename to write.

    circuits : list of Circuits
        List of circuits to write, each to be followed by num_zero_cols zeros.

    header_string : string, optional
        Header string for the file; should start with a pound (#) or double-pound (##)
        so it is treated as a commend or directive, respectively.

    num_zero_cols : int, optional
        The number of zero columns to place after each circuit.  If None,
        then header_string must begin with "## Columns = " and number of zero
        columns will be inferred.

    append_weights_column : bool, optional
        Add an additional 'weights' column.

    Returns
    -------
    None
    """

    if len(circuits) > 0 and not isinstance(circuits[0], _circuits.Circuit):
        raise ValueError("Argument circuits must be a list of Circuit objects!")

    if num_zero_cols is None:  # TODO: cleaner way to extract number of columns from header_string?
        if header_string.startswith('## Columns = '):
            num_zero_cols = len(header_string.split(','))
        else:
            raise ValueError("Must specify num_zero_cols since I can't figure it out from the header string")

    with open(str(filename), 'w') as output:
        zeroCols = "  ".join(['0'] * num_zero_cols)
        output.write(header_string + '\n')
        for circuit in circuits:  # circuit should be a Circuit object here
            output.write(circuit.str + "  " + zeroCols + (("  %f" %
                                                           circuit.weight) if append_weights_column else "") + '\n')


def _outcome_to_str(x):
    if isinstance(x, str): return x
    else: return ":".join([str(i) for i in x])


def write_dataset(filename, dataset, circuits=None,
                  outcome_label_order=None, fixed_column_mode='auto', with_times="auto"):
    """
    Write a text-formatted dataset file.

    Parameters
    ----------
    filename : string
        The filename to write.

    dataset : DataSet
        The data set from which counts are obtained.

    circuits : list of Circuits, optional
        The list of circuits to include in the written dataset.
        If None, all circuits are output.

    outcome_label_order : list, optional
        A list of the outcome labels in dataset which specifies
        the column order in the output file.

    fixed_column_mode : bool or 'auto', optional
        When `True`, a file is written with column headers indicating which
        outcome each column of counts corresponds to.  If a row doesn't have
        any counts for an outcome, `'--'` is used in its place.  When `False`,
        each row's counts are written in an expanded form that includes the
        outcome labels (each "count" has the format <outcomeLabel>:<count>).

    with_times : bool or "auto", optional
        Whether to include (save) time-stamp information in output.  This
        can only be True when `fixed_column_mode=False`.  `"auto"` will set
        this to True if `fixed_column_mode=False` and `dataset` has data at
        non-trivial (non-zero) times.

    Returns
    -------
    None
    """
    if circuits is not None:
        if len(circuits) > 0 and not isinstance(circuits[0], _circuits.Circuit):
            raise ValueError("Argument circuits must be a list of Circuit objects!")
    else:
        circuits = list(dataset.keys())

    if outcome_label_order is not None:  # convert to tuples if needed
        outcome_label_order = [(ol,) if isinstance(ol, str) else ol
                               for ol in outcome_label_order]

    outcomeLabels = dataset.outcome_labels
    if outcome_label_order is not None:
        assert(len(outcome_label_order) == len(outcomeLabels))
        assert(all([ol in outcomeLabels for ol in outcome_label_order]))
        assert(all([ol in outcome_label_order for ol in outcomeLabels]))
        outcomeLabels = outcome_label_order

    headerString = ""
    if hasattr(dataset, 'comment') and dataset.comment is not None:
        for commentLine in dataset.comment.split('\n'):
            if commentLine.startswith('#'):
                headerString += commentLine + '\n'
            else:
                headerString += "# " + commentLine + '\n'

    if fixed_column_mode == "auto":
        if with_times == "auto":
            with_times = not dataset.has_trivial_timedependence
        fixed_column_mode = bool(len(outcomeLabels) <= 8 and not with_times)

    if fixed_column_mode is True:
        headerString += '## Columns = ' + ", ".join(["%s count" % _outcome_to_str(ol)
                                                     for ol in outcomeLabels]) + '\n'
        assert(not (with_times is True)), "Cannot set `witTimes=True` when `fixed_column_mode=True`"
    else:
        headerString += '## Outcomes = ' + ", ".join([_outcome_to_str(ol) for ol in outcomeLabels]) + '\n'

        if with_times == "auto":
            trivial_times = dataset.has_trivial_timedependence
        else:
            trivial_times = not with_times

    with open(str(filename), 'w') as output:
        output.write(headerString)
        for circuit in circuits:  # circuit should be a Circuit object here
            dataRow = dataset[circuit]
            counts = dataRow.counts

            if fixed_column_mode:
                #output '--' for outcome labels that aren't present in this row
                output.write(circuit.str + "  "
                             + "  ".join([(("%g" % counts[ol]) if (ol in counts) else '--')
                                          for ol in outcomeLabels]))
                if dataRow.aux: output.write(" # %s" % str(repr(dataRow.aux)))  # write aux info
                output.write('\n')  # finish the line

            elif trivial_times:  # use expanded label:count format
                output.write(circuit.str + "  "
                             + "  ".join([("%s:%g" % (_outcome_to_str(ol), cnt)) for ol, cnt in counts.items()]))
                if dataRow.aux: output.write(" # %s" % str(repr(dataRow.aux)))  # write aux info
                output.write('\n')  # finish the line

            else:
                output.write(circuit.str + "\n"
                             + "times: " + "  ".join(["%g" % tm for tm in dataRow.time]) + "\n"
                             + "outcomes: " + "  ".join([_outcome_to_str(ol) for ol in dataRow.outcomes]) + "\n")
                if dataRow.reps is not None:
                    fmt = "%d" if _np.all(_np.mod(dataRow.reps, 1) == 0) else "%g"
                    output.write("repetitions: " + "  ".join([fmt % rep for rep in dataRow.reps]) + "\n")
                if dataRow.aux:
                    output.write("aux: " + str(repr(dataRow.aux)) + "\n")
                output.write('\n')  # blank line between circuits


def write_multidataset(filename, multidataset, circuits=None, outcome_label_order=None):
    """
    Write a text-formatted multi-dataset file.

    Parameters
    ----------
    filename : string
        The filename to write.

    multidataset : MultiDataSet
        The multi data set from which counts are obtained.

    circuits : list of Circuits
        The list of circuits to include in the written dataset.
        If None, all circuits are output.

    outcome_label_order : list, optional
        A list of the SPAM labels in multidataset which specifies
        the column order in the output file.

    Returns
    -------
    None
    """

    if circuits is not None:
        if len(circuits) > 0 and not isinstance(circuits[0], _circuits.Circuit):
            raise ValueError("Argument circuits must be a list of Circuit objects!")
    else:
        circuits = list(multidataset.cirIndex.keys())  # TODO: make access function for circuits?

    if outcome_label_order is not None:  # convert to tuples if needed
        outcome_label_order = [(ol,) if isinstance(ol, str) else ol
                               for ol in outcome_label_order]

    outcomeLabels = multidataset.outcome_labels
    if outcome_label_order is not None:
        assert(len(outcome_label_order) == len(outcomeLabels))
        assert(all([ol in outcomeLabels for ol in outcome_label_order]))
        assert(all([ol in outcome_label_order for ol in outcomeLabels]))
        outcomeLabels = outcome_label_order

    dsLabels = list(multidataset.keys())

    headerString = ""
    if hasattr(multidataset, 'comment') and multidataset.comment is not None:
        for commentLine in multidataset.comment.split('\n'):
            if commentLine.startswith('#'):
                headerString += commentLine + '\n'
            else:
                headerString += "# " + commentLine + '\n'
    headerString += '## Columns = ' + ", ".join(["%s %s count" % (dsl, _outcome_to_str(ol))
                                                 for dsl in dsLabels
                                                 for ol in outcomeLabels])

    datasets = [multidataset[dsl] for dsl in dsLabels]
    with open(str(filename), 'w') as output:
        output.write(headerString + '\n')
        for circuit in circuits:  # circuit should be a Circuit object here
            cnts = [ds[circuit].counts.get(ol, '--') for ds in datasets for ol in outcomeLabels]
            output.write(circuit.str + "  " + "  ".join([(("%g" % cnt) if (cnt != '--') else cnt)
                                                         for cnt in cnts]) + '\n')
            #write aux info
            if multidataset.auxInfo[circuit]:
                output.write(" # %s" % str(repr(multidataset.auxInfo[circuit])))
            output.write('\n')  # finish the line


def write_circuit_list(filename, circuits, header=None):
    """
    Write a text-formatted circuit list file.

    Parameters
    ----------
    filename : string
        The filename to write.

    circuits : list of Circuits
        The list of circuits to include in the written dataset.

    header : string, optional
        Header line (first line of file).  Prepended with a pound sign (#), so no
        need to include one.

    Returns
    -------
    None
    """
    if len(circuits) > 0 and not isinstance(circuits[0], _circuits.Circuit):
        raise ValueError("Argument circuits must be a list of Circuit objects!")

    with open(str(filename), 'w') as output:
        if header is not None:
            output.write("# %s" % header + '\n')

        for circuit in circuits:
            output.write(circuit.str + '\n')


@_deprecated_fn('pygsti.models.Model.write(...)')
def write_model(model, filename, title=None):
    """
    Write a text-formatted model file.

    Parameters
    ----------
    model : Model
        The model to write to file.

    filename : string
        The filename to write.

    title : string, optional
        Header line (first line of file).  Prepended with a pound sign (#), so no
        need to include one.

    Returns
    -------
    None
    """
    _warnings.warn("write_model(...) is unable to write all types of pyGSTi models, and really should NOT be used!")

    def writeprop(f, lbl, val):
        """ Write (label,val) property to output file """
        if isinstance(val, _np.ndarray):  # then write as rows
            f.write("%s\n" % lbl)
            if val.ndim == 1:
                f.write(" ".join("%.8g" % el for el in val) + '\n')
            elif val.ndim == 2:
                f.write(_tools.mx_to_string(val, width=16, prec=8))
            else:
                raise ValueError("Cannot write an ndarray with %d dimensions!" % val.ndim)
            f.write("\n")
        else:
            f.write("%s = %s\n" % (lbl, repr(val)))

    with open(str(filename), 'w') as output:

        if title is not None:
            output.write("# %s" % title + '\n')
        output.write('\n')

        for prepLabel, rhoVec in model.preps.items():
            props = None
            if isinstance(rhoVec, _state.FullState): typ = "PREP"
            elif isinstance(rhoVec, _state.TPState): typ = "TP-PREP"
            elif isinstance(rhoVec, _state.StaticState): typ = "STATIC-PREP"
            #elif isinstance(rhoVec, _state.LindbladSPAMVec):  # TODO - change to ComposedState?
            #    typ = "CPTP-PREP"
            #    props = [("PureVec", rhoVec.state_vec.to_dense(on_space='HilbertSchmidt')),
            #             ("ErrgenMx", rhoVec.error_map.to_dense(on_space='HilbertSchmidt'))]
            else:
                _warnings.warn(
                    ("Non-standard prep of type {typ} cannot be described by"
                     "text format model files.  It will be read in as a"
                     "fully parameterized spam vector").format(typ=str(type(rhoVec))))
                typ = "PREP"

            if props is None: props = [("LiouvilleVec", rhoVec.to_dense(on_space='HilbertSchmidt'))]
            output.write("%s: %s\n" % (typ, prepLabel))
            for lbl, val in props:
                writeprop(output, lbl, val)

        for povmLabel, povm in model.povms.items():
            props = None; povm_to_write = povm
            if isinstance(povm, _povm.UnconstrainedPOVM): povmType = "POVM"
            elif isinstance(povm, _povm.TPPOVM): povmType = "TP-POVM"
            #elif isinstance(povm, _povm.LindbladPOVM):  # TODO - change to ComposedPOVM?
            #    povmType = "CPTP-POVM"
            #    props = [("ErrgenMx", povm.error_map.to_dense(on_space='HilbertSchmidt'))]
            #    povm_to_write = povm.base_povm
            else:
                _warnings.warn(
                    ("Non-standard POVM of type {typ} cannot be described by"
                     "text format model files.  It will be read in as a"
                     "standard POVM").format(typ=str(type(povm))))
                povmType = "POVM"

            output.write("%s: %s\n\n" % (povmType, povmLabel))
            if props is not None:
                for lbl, val in props:
                    writeprop(output, lbl, val)

            for ELabel, EVec in povm_to_write.items():
                if isinstance(EVec, _povm.FullPOVMEffect): typ = "EFFECT"
                elif isinstance(EVec, _povm.ComplementPOVMEffect): typ = "EFFECT"  # ok
                elif isinstance(EVec, _povm.StaticPOVMEffect): typ = "STATIC-EFFECT"
                else:
                    _warnings.warn(
                        ("Non-standard effect of type {typ} cannot be described by"
                         "text format model files.  It will be read in as a"
                         "fully parameterized spam vector").format(typ=str(type(EVec))))
                    typ = "EFFECT"
                output.write("%s: %s\n" % (typ, ELabel))
                writeprop(output, "LiouvilleVec", EVec.to_dense(on_space='HilbertSchmidt'))

            output.write("END POVM\n\n")

        for label, gate in model.operations.items():
            props = None
            if isinstance(gate, _op.FullArbitraryOp): typ = "GATE"
            elif isinstance(gate, _op.FullTPOp): typ = "TP-GATE"
            elif isinstance(gate, _op.StaticArbitraryOp): typ = "STATIC-GATE"
            elif isinstance(gate, _op.ComposedOp):
                typ = "COMPOSED-GATE"
                props = [("%dLiouvilleMx" % i, factor.to_dense(on_space='HilbertSchmidt'))
                         for i, factor in enumerate(gate.factorops)]
            else:
                _warnings.warn(
                    ("Non-standard gate of type {typ} cannot be described by"
                     "text format model files.  It will be read in as a"
                     "fully parameterized gate").format(typ=str(type(gate))))
                typ = "GATE"

            if props is None: props = [("LiouvilleMx", gate.to_dense(on_space='HilbertSchmidt'))]
            output.write(typ + ": " + str(label) + '\n')
            for lbl, val in props:
                writeprop(output, lbl, val)

        for instLabel, inst in model.instruments.items():
            if isinstance(inst, _instrument.Instrument): typ = "Instrument"
            elif isinstance(inst, _instrument.TPInstrument): typ = "TP-Instrument"
            else:
                _warnings.warn(
                    ("Non-standard Instrument of type {typ} cannot be described by"
                     "text format model files.  It will be read in as a"
                     "standard Instrument").format(typ=str(type(inst))))
                typ = "Instrument"
            output.write(typ + ": " + str(instLabel) + '\n\n')

            for label, gate in inst.items():
                if isinstance(gate, _op.FullArbitraryOp): typ = "IGATE"
                elif isinstance(gate, _instrument.TPInstrumentOp): typ = "IGATE"  # ok b/c instrument is marked as TP
                elif isinstance(gate, _op.StaticArbitraryOp): typ = "STATIC-IGATE"
                else:
                    _warnings.warn(
                        ("Non-standard gate of type {typ} cannot be described by"
                         "text format model files.  It will be read in as a"
                         "fully parameterized gate").format(typ=str(type(gate))))
                    typ = "IGATE"
                output.write(typ + ": " + str(label) + '\n')
                writeprop(output, "LiouvilleMx", gate.to_dense(on_space='HilbertSchmidt'))
            output.write("END Instrument\n\n")

        if model.state_space is not None:
            output.write("STATESPACE: " + str(model.state_space) + "\n")
            # StateSpaceLabels.__str__ formats the output properly

        basisdim = model.basis.dim

        if basisdim is None:
            output.write("BASIS: %s\n" % model.basis.name)
        else:
            if model.basis.name not in ('std', 'pp', 'gm', 'qt'):  # a "fancy" basis
                assert(model.state_space is not None), \
                    "Must set a Model's state space labels when using fancy a basis!"
                # don't write the dim - the state space labels will cover this.
                output.write("BASIS: %s\n" % model.basis.name)
            else:
                output.write("BASIS: %s %d\n" % (model.basis.name, basisdim))

        if isinstance(model.default_gauge_group, _gaugegroup.FullGaugeGroup):
            output.write("GAUGEGROUP: Full\n")
        elif isinstance(model.default_gauge_group, _gaugegroup.TPGaugeGroup):
            output.write("GAUGEGROUP: TP\n")
        elif isinstance(model.default_gauge_group, _gaugegroup.UnitaryGaugeGroup):
            output.write("GAUGEGROUP: Unitary\n")


def write_empty_protocol_data(dirname, edesign, sparse="auto", clobber_ok=False):
    """
    Write to disk an empty :class:`ProtocolData` object.

    Write to a directory an experimental design (`edesign`) and the dataset
    template files needed to load in a :class:`ProtocolData` object, e.g.
    using the :function:`read_data_from_dir` function, after the template
    files are filled in.

    Parameters
    ----------
    dirname : str
        The *root* directory to write into.  This directory will have 'edesign'
        and 'data' subdirectories created beneath it.

    edesign : ExperimentDesign
        The experiment design defining the circuits that need to be performed.

    sparse : bool or "auto", optional
        If True, then the template data set(s) are written in a sparse-data
        format, i.e. in a format where not all the outcomes need to be given.
        If False, then a dense data format is used, where counts for *all*
        possible bit strings are given.  `"auto"` causes the sparse format
        to be used when the number of qubits is > 2.

    clobber_ok : bool, optional
        If True, then a template dataset file will be written even if a file
        of the same name already exists (this may overwrite existing data
        with an empty template file, so be careful!).

    Returns
    -------
    None
    """
    if isinstance(edesign, str):
        _warnings.warn(("This function has recently changed its signature - it looks like you need to swap"
                        " the first two arguments.  Continuing using the old signature..."))
        edesign, dirname = dirname, edesign

    dirname = _pathlib.Path(dirname)
    data_dir = dirname / 'data'
    circuits = edesign.all_circuits_needing_data
    nQubits = "multiple" if edesign.qubit_labels == "multiple" else len(edesign.qubit_labels)
    if sparse == "auto":
        sparse = bool(nQubits == "multiple" or nQubits > 3)  # HARDCODED

    if sparse:
        header_str = "# Note: on each line, put comma-separated <outcome:count> items, i.e. 00110:23"
        nZeroCols = 0
    else:
        fstr = '{0:0%db} count' % nQubits
        nZeroCols = 2**nQubits
        header_str = "## Columns = " + ", ".join([fstr.format(i) for i in range(nZeroCols)])

    pth = data_dir / 'dataset.txt'
    if pth.exists() and clobber_ok is False:
        raise ValueError(("Template data file would clobber %s, which already exists!  Set `clobber_ok=True`"
                          " to allow overwriting." % pth))
    data_dir.mkdir(parents=True, exist_ok=True)

    from ..protocols import ProtocolData as _ProtocolData
    data = _ProtocolData(edesign, None)
    data.write(dirname)
    write_empty_dataset(pth, circuits, header_str, nZeroCols)


def fill_in_empty_dataset_with_fake_data(dataset_filename, model, num_samples, sample_error="multinomial", seed=None,
                                         rand_state=None, alias_dict=None, collision_action="aggregate",
                                         record_zero_counts=True, comm=None, mem_limit=None, times=None,
                                         fixed_column_mode="auto"):
    """
    Fills in the text-format data set file `dataset_fileame` with simulated data counts using `model`.

    Parameters
    ----------
    dataset_filename : str
        the path to the text-formatted data set file.

    model : Model
        the model to use to simulate the data.

    num_samples : int or list of ints or None
        The simulated number of samples for each circuit.  This only has
        effect when  ``sample_error == "binomial"`` or ``"multinomial"``.  If an
        integer, all circuits have this number of total samples. If a list,
        integer elements specify the number of samples for the corresponding
        circuit.  If ``None``, then `model_or_dataset` must be a
        :class:`~pygsti.objects.DataSet`, and total counts are taken from it
        (on a per-circuit basis).

    sample_error : string, optional
        What type of sample error is included in the counts.  Can be:

        - "none"  - no sample error: counts are floating point numbers such
          that the exact probabilty can be found by the ratio of count / total.
        - "clip" - no sample error, but clip probabilities to [0,1] so, e.g.,
          counts are always positive.
        - "round" - same as "clip", except counts are rounded to the nearest
          integer.
        - "binomial" - the number of counts is taken from a binomial
          distribution.  Distribution has parameters p = (clipped) probability
          of the circuit and n = number of samples.  This can only be used
          when there are exactly two SPAM labels in model_or_dataset.
        - "multinomial" - counts are taken from a multinomial distribution.
          Distribution has parameters p_k = (clipped) probability of the gate
          string using the k-th SPAM label and n = number of samples.

    seed : int, optional
        If not ``None``, a seed for numpy's random number generator, which
        is used to sample from the binomial or multinomial distribution.

    rand_state : numpy.random.RandomState
        A RandomState object to generate samples from. Can be useful to set
        instead of `seed` if you want reproducible distribution samples across
        multiple random function calls but you don't want to bother with
        manually incrementing seeds between those calls.

    alias_dict : dict, optional
        A dictionary mapping single operation labels into tuples of one or more
        other operation labels which translate the given circuits before values
        are computed using `model_or_dataset`.  The resulting Dataset, however,
        contains the *un-translated* circuits as keys.

    collision_action : {"aggregate", "keepseparate"}
        Determines how duplicate circuits are handled by the resulting
        `DataSet`.  Please see the constructor documentation for `DataSet`.

    record_zero_counts : bool, optional
        Whether zero-counts are actually recorded (stored) in the returned
        DataSet.  If False, then zero counts are ignored, except for
        potentially registering new outcome labels.

    comm : mpi4py.MPI.Comm, optional
        When not ``None``, an MPI communicator for distributing the computation
        across multiple processors and ensuring that the *same* dataset is
        generated on each processor.

    mem_limit : int, optional
        A rough memory limit in bytes which is used to determine job allocation
        when there are multiple processors.

    times : iterable, optional
        When not None, a list of time-stamps at which data should be sampled.
        `num_samples` samples will be simulated at each time value, meaning that
        each circuit in `circuits` will be evaluated with the given time
        value as its *start time*.

    fixed_column_mode : bool or 'auto', optional
        How the underlying data set file is written - see :function:`write_dataset`.

    Returns
    -------
    DataSet
        The generated data set (also written in place of the template file).
    """
    if isinstance(model, str):
        _warnings.warn(("This function has recently changed its signature - it looks like you need to swap"
                        " the first two arguments.  Continuing using the old signature..."))
        model, dataset_filename = dataset_filename, model

    from pygsti.data.datasetconstruction import simulate_data as _simulate_data
    ds_template = _loaders.read_dataset(dataset_filename, ignore_zero_count_lines=False, with_times=False, verbosity=0)
    ds = _simulate_data(model, list(ds_template.keys()), num_samples,
                        sample_error, seed, rand_state, alias_dict,
                        collision_action, record_zero_counts, comm,
                        mem_limit, times)
    if fixed_column_mode == "auto":
        fixed_column_mode = bool(len(ds_template.outcome_labels) <= 8 and times is None)
    write_dataset(dataset_filename, ds, fixed_column_mode=fixed_column_mode)
    return ds


def write_circuit_strings(filename, obj):
    """ TODO: docstring - write various Circuit-containing standard objects with circuits
        replaced by their string reps """
    from pygsti.circuits import Circuit as _Circuit

    def _replace_circuits_with_strs(x):
        if isinstance(x, (list, tuple)):
            return [_replace_circuits_with_strs(el) for el in x]
        if isinstance(x, dict):
            return {_replace_circuits_with_strs(k): _replace_circuits_with_strs(v) for k, v in x.items}
        return x.str if isinstance(x, _Circuit) else x

    json_dict = _replace_circuits_with_strs(obj)
    if str(filename).endswith('.json'):
        with open(filename, 'w') as f:
            _json.dump(json_dict, f, indent=4)
    else:
        raise ValueError("Cannot determine format from extension of filename: %s" % str(filename))
