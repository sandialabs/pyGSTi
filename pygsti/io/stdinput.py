"""
Text-parsing classes and functions to read input files.
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import re as _re
import os as _os
import sys as _sys
import time as _time
import numpy as _np
import ast as _ast
import warnings as _warnings
from scipy.linalg import expm as _expm
from collections import OrderedDict as _OrderedDict

from ..modelmembers import operations as _op
from ..modelmembers import states as _state
from ..modelmembers import povms as _povm
from ..modelmembers import instruments as _instrument
from ..models import statespace as _statespace

from .. import objects as _objs
from .. import tools as _tools

from . import CircuitParser as _CircuitParser

# A dictionary mapping qubit string representations into created
# :class:`Circuit` objects, which can improve performance by reducing
# or eliminating the need to parse circuit strings we've already parsed.
_global_parse_cache = {False: {}, True: {}}  # key == create_subcircuits


def _create_display_progress_fn(show_progress):
    """
    Create and return a progress-displaying function.

    Only return a function that does somethign if `show_progress == True`
    and the current environment is interactive. Otherwise, return a
    do-nothing function.

    Parameters
    ----------
    show_progress : bool
        Whether or not to even try to get a real progress-displaying function.

    Returns
    -------
    function
    """

    def _is_interactive():
        import __main__ as main
        return not hasattr(main, '__file__')

    if _is_interactive() and show_progress:
        try:
            from IPython.display import clear_output

            def _display_progress(i, n, filename):
                _time.sleep(0.001); clear_output()
                print("Loading %s: %.0f%%" % (filename, 100.0 * float(i) / float(n)))
                _sys.stdout.flush()
        except:
            def _display_progress(i, n, f): pass
    else:
        def _display_progress(i, n, f): pass

    return _display_progress


class StdInputParser(object):
    """
    Encapsulates a text parser for reading GST input files.
    """

    #  Using a single parser. This speeds up parsing, however, it means the parser is NOT reentrant
    _circuit_parser = _CircuitParser()
    use_global_parse_cache = True

    def __init__(self):
        """ Create a new standard-input parser object """
        pass

    def parse_circuit(self, s, lookup={}, create_subcircuits=True):
        """
        Parse a circuit from a string.

        Parameters
        ----------
        s : string
            The string to parse.

        lookup : dict, optional
            A dictionary with keys == reflbls and values == tuples of operation labels
            which can be used for substitutions using the S<reflbl> syntax.

        create_subcircuits : bool, optional
            Whether to create sub-circuit-labels when parsing
            string representations or to just expand these into non-subcircuit
            labels.

        Returns
        -------
        Circuit
        """
        circuit = None
        if self.use_global_parse_cache:
            circuit = _global_parse_cache[create_subcircuits].get(s, None)
        if circuit is None:  # wasn't in cache
            layer_tuple, line_lbls, occurrence_id = self.parse_circuit_raw(s, lookup, create_subcircuits)
            if line_lbls is None:  # if there are no line labels then we need to use "auto" and do a full init
                circuit = _objs.Circuit(layer_tuple, stringrep=s, line_labels="auto",
                                        expand_subcircuits=False, check=False, occurrence=occurrence_id)
                #Note: never expand subcircuits since parse_circuit_raw already does this w/create_subcircuits arg
            else:
                circuit = _objs.Circuit._fastinit(layer_tuple, line_lbls, editable=False,
                                                  name='', stringrep=s, occurrence=occurrence_id)

            if self.use_global_parse_cache:
                _global_parse_cache[create_subcircuits][s] = circuit
        return circuit

    def parse_circuit_raw(self, s, lookup={}, create_subcircuits=True):
        """
        Parse a circuit's constituent pieces from a string.

        This doesn't actually create a circuit object, which may be desirable
        in some scenarios.

        Parameters
        ----------
        s : string
            The string to parse.

        lookup : dict, optional
            A dictionary with keys == reflbls and values == tuples of operation labels
            which can be used for substitutions using the S<reflbl> syntax.

        create_subcircuits : bool, optional
            Whether to create sub-circuit-labels when parsing
            string representations or to just expand these into non-subcircuit
            labels.

        Returns
        -------
        tuple of operation labels
            Representing the circuit.
        """
        self._circuit_parser.lookup = lookup
        circuit_tuple, circuit_labels, occurrence_id = self._circuit_parser.parse(s, create_subcircuits)
        # print "DB: result = ",result
        # print "DB: stack = ",self.exprStack
        return circuit_tuple, circuit_labels, occurrence_id

    def parse_dataline(self, s, lookup={}, expected_counts=-1, create_subcircuits=True,
                       line_labels=None):
        """
        Parse a data line (dataline in grammar)

        Parameters
        ----------
        s : string
            The string to parse.

        lookup : dict, optional
            A dictionary with keys == reflbls and values == tuples of operation labels
            which can be used for substitutions using the S<reflbl> syntax.

        expected_counts : int, optional
            The expected number of counts to accompany the circuit on this
            data line.  If < 0, no check is performed; otherwise raises ValueError
            if the number of counts does not equal expected_counts.

        create_subcircuits : bool, optional
            Whether to create sub-circuit-labels when parsing string representations
            or to just expand these into non-subcircuit labels.

        Returns
        -------
        circuit : Circuit
            The circuit.
        counts : list
            List of counts following the circuit.
        """

        # get counts from end of s
        parts = s.split()
        circuitStr = parts[0]

        counts = []
        if expected_counts == -1:  # then we expect to be given <outcomeLabel>:<count> items
            if len(parts) == 1:  # only a circuit, no counts on line
                pass  # just leave counts empty
            elif parts[1] == "BAD":
                counts.append("BAD")
            else:
                for p in parts[1:]:
                    t = p.split(':')
                    counts.append((tuple(t[0:-1]), float(t[-1])))

        else:  # data is in columns as given by header
            for p in parts[1:]:
                if p in ('--', 'BAD'):
                    counts.append(p)
                else:
                    counts.append(float(p))

            if len(counts) > expected_counts >= 0:
                counts = counts[0:expected_counts]

            nCounts = len(counts)
            if nCounts != expected_counts:
                raise ValueError("Found %d count columns when %d were expected" % (nCounts, expected_counts))
            if nCounts == len(parts):
                raise ValueError("No circuit column found -- all columns look like data")

        circuit = self.parse_circuit(circuitStr, lookup, create_subcircuits)
        return circuit, counts

    def parse_dictline(self, s):
        """
        Parse a circuit dictionary line (dictline in grammar)

        Parameters
        ----------
        s : string
            The string to parse.

        Returns
        -------
        circuitLabel : string
            The user-defined label to represent this circuit.
        circuitTuple : tuple
            The circuit as a tuple of operation labels.
        circuitStr : string
            The circuit as represented as a string in the dictline.
        circuitLineLabels : tuple
            The line labels of the cirucit.
        occurrence : object
            Circuit's occurrence id, or `None` if there is none.
        """
        label = r'\s*([a-zA-Z0-9_]+)\s+'
        match = _re.match(label, s)
        if not match:
            raise ValueError("'{}' is not a valid dictline".format(s))
        circuitLabel = match.group(1)
        circuitStr = s[match.end():]
        circuitTuple, circuitLineLabels, occurrence_id = self._circuit_parser.parse(circuitStr)
        return circuitLabel, circuitTuple, circuitStr, circuitLineLabels, occurrence_id

    def parse_stringfile(self, filename, line_labels="auto", num_lines=None, create_subcircuits=True):
        """
        Parse a circuit list file.

        Parameters
        ----------
        filename : string
            The file to parse.

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

        create_subcircuits : bool, optional
            Whether to create sub-circuit-labels when parsing
            string representations or to just expand these into non-subcircuit
            labels.

        Returns
        -------
        list of Circuits
            The circuits read from the file.
        """
        circuit_list = []
        with open(filename, 'r') as stringfile:
            for line in stringfile:
                line = line.strip()
                if len(line) == 0 or line[0] == '#': continue
                if line_labels == "auto":
                    # can be cached, and cache assumes "auto" behavior
                    circuit = self.parse_circuit(line, {}, create_subcircuits)
                else:
                    layer_lbls, parsed_line_lbls, occurrence_id = self.parse_circuit_raw(line, {}, create_subcircuits)
                    if parsed_line_lbls is None:
                        parsed_line_lbls = line_labels  # default to the passed-in argument
                        #nlines = num_lines
                    #else: nlines = None  # b/c we've got a valid line_lbls
                    circuit = _objs.Circuit._fastinit(layer_lbls, parsed_line_lbls, editable=False,
                                                      name='', stringrep=line.strip(), occurrence=occurrence_id)
                    #circuit = _objs.Circuit(layer_lbls, stringrep=line.strip(),
                    #                        line_labels=parsed_line_lbls, num_lines=nlines,
                    #                        expand_subcircuits=False, check=False, occurrence=occurrence_id)
                    ##Note: never expand subcircuits since parse_circuit_raw already does this w/create_subcircuits arg
                circuit_list.append(circuit)
        return circuit_list

    def parse_dictfile(self, filename):
        """
        Parse a circuit dictionary file.

        Parameters
        ----------
        filename : string
            The file to parse.

        Returns
        -------
        dict
            Dictionary with keys == circuit labels and values == Circuits.
        """
        lookupDict = {}
        with open(filename, 'r') as dictfile:
            for line in dictfile:
                line = line.strip()
                if len(line) == 0 or line[0] == '#': continue
                label, tup, s, lineLbls, occurrence_id = self.parse_dictline(line)
                if lineLbls is None: lineLbls = "auto"
                lookupDict[label] = _objs.Circuit(tup, stringrep=s, line_labels=lineLbls,
                                                  check=False, occurrence=occurrence_id)
        return lookupDict

    def parse_datafile(self, filename, show_progress=True,
                       collision_action="aggregate", record_zero_counts=True,
                       ignore_zero_count_lines=True, with_times="auto"):
        """
        Parse a data set file into a DataSet object.

        Parameters
        ----------
        filename : string
            The file to parse.

        show_progress : bool, optional
            Whether or not progress should be displayed

        collision_action : {"aggregate", "keepseparate"}
            Specifies how duplicate circuits should be handled.  "aggregate"
            adds duplicate-circuit counts, whereas "keepseparate" tags duplicate
            circuits by setting their `.occurrence` IDs to sequential positive integers.

        record_zero_counts : bool, optional
            Whether zero-counts are actually recorded (stored) in the returned
            DataSet.  If False, then zero counts are ignored, except for potentially
            registering new outcome labels.

        ignore_zero_count_lines : bool, optional
            Whether circuits for which there are no counts should be ignored
            (i.e. omitted from the DataSet) or not.

        with_times : bool or "auto", optional
            Whether to the time-stamped data format should be read in.  If
            "auto", then this format is allowed but not required.  Typically
            you only need to set this to False when reading in a template file.

        Returns
        -------
        DataSet
            A static DataSet object.
        """

        #Parse preamble -- lines beginning with # or ## until first non-# line
        preamble_directives = {}
        preamble_comments = []
        with open(filename, 'r') as datafile:
            for line in datafile:
                line = line.strip()
                if len(line) == 0 or line[0] != '#': break
                if line.startswith("## "):
                    parts = line[len("## "):].split("=")
                    if len(parts) == 2:  # key = value
                        preamble_directives[parts[0].strip()] = parts[1].strip()
                elif line.startswith("#"):
                    preamble_comments.append(line[1:].strip())

        def str_to_outcome(x):  # always return a tuple as the "outcome label" (even if length 1)
            return tuple(x.strip().split(":"))

        #Process premble
        orig_cwd = _os.getcwd()
        outcomeLabels = None
        outcome_labels_specified_in_preamble = False
        if len(_os.path.dirname(filename)) > 0: _os.chdir(
            _os.path.dirname(filename))  # allow paths relative to datafile path
        try:
            if 'Lookup' in preamble_directives:
                lookupDict = self.parse_dictfile(preamble_directives['Lookup'])
            else: lookupDict = {}
            if 'Columns' in preamble_directives:
                colLabels = [l.strip() for l in preamble_directives['Columns'].split(",")]
                #OLD: outcomeLabels, fillInfo = self._extract_labels_from_col_labels(colLabels)
                fixed_column_outcome_labels = []
                for i, colLabel in enumerate(colLabels):
                    assert(colLabel.endswith(' count')), \
                        "Invalid count column name `%s`! (Only *count* columns are supported now)" % colLabel
                    outcomeLabel = str_to_outcome(colLabel[:-len(' count')])
                    if outcomeLabel not in fixed_column_outcome_labels:
                        fixed_column_outcome_labels.append(outcomeLabel)

                nDataCols = len(colLabels)
            else:
                fixed_column_outcome_labels = None
                nDataCols = -1  # no column count check
            if 'Outcomes' in preamble_directives:
                outcomeLabels = [l.strip().split(':') for l in preamble_directives['Outcomes'].split(",")]
                outcome_labels_specified_in_preamble = True
            if 'StdOutcomeQubits' in preamble_directives:
                outcomeLabels = int(preamble_directives['Outcomes'])
                outcome_labels_specified_in_preamble = True
        finally:
            _os.chdir(orig_cwd)

        #Read data lines of data file
        dataset = _objs.DataSet(outcome_labels=outcomeLabels, collision_action=collision_action,
                                comment="\n".join(preamble_comments))

        if outcome_labels_specified_in_preamble and (fixed_column_outcome_labels is not None):
            fixed_column_outcome_indices = [dataset.olIndex[ol] for ol in fixed_column_outcome_labels]
        else:
            fixed_column_outcome_indices = None

        nLines = 0
        with open(filename, 'r') as datafile:
            nLines = sum(1 for line in datafile)
        nSkip = int(nLines / 100.0)
        if nSkip == 0: nSkip = 1

        display_progress = _create_display_progress_fn(show_progress)
        warnings = []  # to display *after* display progress
        looking_for = "circuit_line"; current_item = {}

        def parse_comment(comment, filename, i_line):
            commentDict = {}
            comment = comment.strip()
            if len(comment) == 0: return {}
            try:
                if comment.startswith("{") and comment.endswith("}"):
                    commentDict = _ast.literal_eval(comment)
                else:  # put brackets around it
                    commentDict = _ast.literal_eval("{ " + comment + " }")
                #commentDict = _json.loads("{ " + comment + " }")
                #Alt: safer(?) & faster, but need quotes around all keys & vals
            except:
                commentDict = {}
                warnings.append("%s Line %d: Could not parse comment '%s'"
                                % (filename, i_line, comment))
            return commentDict

        last_circuit = last_commentDict = None

        with open(filename, 'r') as inputfile:
            for (iLine, line) in enumerate(inputfile):
                if iLine % nSkip == 0 or iLine + 1 == nLines: display_progress(iLine + 1, nLines, filename)

                line = line.strip()
                if '#' in line:
                    i = line.index('#')
                    dataline, comment = line[:i], line[i + 1:]
                else:
                    dataline, comment = line, ""

                if looking_for == "circuit_data_or_line":
                    # Special confusing case:  lines that just have a circuit could be either the beginning of a
                    # long-format (with times, reps, etc, lines) block OR could just be a circuit that doesn't have
                    # any count data.  This case figures out which one based on the line that follows.
                    if len(dataline) == 0 or dataline.split()[0] in ('times:', 'outcomes:', 'repetitions:', 'aux:'):
                        looking_for = "circuit_data"  # blank lines shoudl process acumulated data
                    else:
                        # previous blank line was just a circuit without any data (*not* the beginning of a timestamped
                        # section), so add it with zero counts (if we don't ignore it), and look for next circuit.
                        looking_for = "circuit_line"
                        if ignore_zero_count_lines is False and last_circuit is not None:
                            dataset.add_count_list(last_circuit, [], [], aux=last_commentDict,
                                                   record_zero_counts=record_zero_counts, update_ol=False, unsafe=True)

                if looking_for == "circuit_line":
                    if len(dataline) == 0: continue
                    try:
                        circuit, valueList = \
                            self.parse_dataline(dataline, lookupDict, nDataCols,
                                                create_subcircuits=not _objs.Circuit.default_expand_subcircuits)

                        commentDict = parse_comment(comment, filename, iLine)

                    except ValueError as e:
                        raise ValueError("%s Line %d: %s" % (filename, iLine, str(e)))

                    if with_times is True and len(valueList) > 0:
                        raise ValueError(("%s Line %d: Circuit line cannot contain count information when "
                                          "'with_times=True'") % (filename, iLine))

                    if with_times is False or len(valueList) > 0:
                        if 'BAD' in valueList:  # entire line is known to be BAD => no data for this circuit
                            oliArray = _np.zeros(0, dataset.oliType)
                            countArray = _np.zeros(0, dataset.repType)
                        else:
                            if fixed_column_outcome_labels is not None:
                                if outcome_labels_specified_in_preamble:
                                    outcome_indices, count_values = \
                                        zip(*[(oli, v) for (oli, v) in zip(fixed_column_outcome_indices, valueList)
                                              if v != '--'])  # drop "empty" sentinels
                                else:
                                    outcome_labels, count_values = \
                                        zip(*[(nm, v) for (nm, v) in zip(fixed_column_outcome_labels, valueList)
                                              if v != '--'])  # drop "empty" sentinels
                                    dataset.add_outcome_labels(outcome_labels, update_ol=False)
                                    outcome_indices = [dataset.olIndex[ol] for ol in outcome_labels]
                            else:  # assume valueList is a list of (outcomeLabel, count) tuples -- see parse_dataline
                                outcome_labels, count_values = zip(*valueList)
                                if not outcome_labels_specified_in_preamble:
                                    dataset.add_outcome_labels(outcome_labels, update_ol=False)
                                outcome_indices = [dataset.olIndex[ol] for ol in outcome_labels]
                            oliArray = _np.array(outcome_indices, dataset.oliType)
                            countArray = _np.array(count_values, dataset.repType)

                        if all([(abs(v) < 1e-9) for v in count_values]):
                            if ignore_zero_count_lines is True:
                                if not ('BAD' in valueList):  # supress "no data" warning for known-bad circuits
                                    s = circuit.str if len(circuit.str) < 40 else circuit.str[0:37] + "..."
                                    warnings.append("Dataline for circuit '%s' has zero counts and will be ignored" % s)
                                continue  # skip lines in dataset file with zero counts (no experiments done)
                            else:
                                #if not bBad:
                                #    s = circuitStr if len(circuitStr) < 40 else circuitStr[0:37] + "..."
                                #    warnings.append("Dataline for circuit '%s' has zero counts." % s)
                                # don't make a fuss if we don't ignore the lines (needed for
                                # fill_in_empty_dataset_with_fake_data).
                                pass

                        #Call this low-level function for performance, so need to construct outcome *index* arrays above
                        dataset.add_count_arrays(circuit, oliArray, countArray,
                                                 record_zero_counts=record_zero_counts, aux=commentDict)
                    else:
                        current_item['circuit'] = circuit
                        looking_for = "circuit_data" if (with_times is True) else "circuit_data_or_line"
                        last_circuit, last_commentDict = circuit, commentDict  # for circuit_data_or_line processing

                elif looking_for == "circuit_data":
                    if len(line) == 0:
                        #add current item & look for next one
                        dataset.add_raw_series_data(current_item['circuit'], current_item['outcomes'],
                                                    current_item['times'], current_item.get('repetitions', None),
                                                    record_zero_counts=record_zero_counts,
                                                    aux=current_item.get('aux', None),
                                                    update_ol=False)  # for performance - to this once at the end.
                        current_item.clear()
                        looking_for = "circuit_line"
                    else:
                        parts = dataline.split()
                        if parts[0] == 'times:':
                            current_item['times'] = [float(x) for x in parts[1:]]
                        elif parts[0] == 'outcomes:':
                            current_item['outcomes'] = parts[1:]  # no conversion needed
                        elif parts[0] == 'repetitions:':
                            try:
                                current_item['repetitions'] = [int(x) for x in parts[1:]]
                            except ValueError:  # raised if int(x) fails b/c reps are floats
                                current_item['repetitions'] = [float(x) for x in parts[1:]]
                        elif parts[0] == 'aux:':
                            current_item['aux'] = parse_comment(" ".join(parts[1:]), filename, iLine)
                        else:
                            raise ValueError("Invalid circuit data-line prefix: '%s'" % parts[0])

        if looking_for == "circuit_data" and current_item:
            #add final circuit info (no blank line at end of file)
            dataset.add_raw_series_data(current_item['circuit'], current_item['outcomes'],
                                        current_item['times'], current_item.get('repetitions', None),
                                        record_zero_counts=record_zero_counts, aux=current_item.get('aux', None),
                                        update_ol=False)  # for performance - to this once at the end.

        dataset.update_ol()  # because we set update_ol=False above, we need to do this
        if warnings:
            _warnings.warn('\n'.join(warnings))  # to be displayed at end, after potential progress updates

        dataset.done_adding_data()
        return dataset

    #TODO: REMOVE - unless we want to support frequency/total count columns again and take performance hit.
    #def _extract_labels_from_col_labels(self, col_labels):
    #    outcomeLabels = []; countCols = []; freqCols = []; impliedCountTotCol1Q = (-1, -1)
    #
    #    def str_to_outcome(x):  # always return a tuple as the "outcome label" (even if length 1)
    #        return tuple(x.strip().split(":"))
    #
    #    for i, colLabel in enumerate(col_labels):
    #        if colLabel.endswith(' count'):
    #            outcomeLabel = str_to_outcome(colLabel[:-len(' count')])
    #            if outcomeLabel not in outcomeLabels: outcomeLabels.append(outcomeLabel)
    #            countCols.append((outcomeLabel, i))
    #
    #        elif colLabel.endswith(' frequency'):
    #            if 'count total' not in col_labels:
    #                raise ValueError("Frequency columns specified without count total")
    #            else: iTotal = col_labels.index('count total')
    #            outcomeLabel = str_to_outcome(colLabel[:-len(' frequency')])
    #            if outcomeLabel not in outcomeLabels: outcomeLabels.append(outcomeLabel)
    #            freqCols.append((outcomeLabel, i, iTotal))
    #
    #    if 'count total' in col_labels:
    #        if ('1',) in outcomeLabels and ('0',) not in outcomeLabels:
    #            outcomeLabels.append(('0',))
    #            impliedCountTotCol1Q = ('0',), col_labels.index('count total')
    #        elif ('0',) in outcomeLabels and ('1',) not in outcomeLabels:
    #            outcomeLabels.append(('1',))
    #            impliedCountTotCol1Q = '1', col_labels.index('count total')
    #        #TODO - add standard count completion for 2Qubit case?
    #
    #    fillInfo = (countCols, freqCols, impliedCountTotCol1Q)
    #    return outcomeLabels, fillInfo
    #
    #def _fill_data_count_dict(self, count_dict, fill_info, col_values):
    #    if 'BAD' in col_values:
    #        return  # indicates entire row is known to be bad (no counts)
    #
    #    #Note: can use setitem_unsafe here because count_dict is a OutcomeLabelDict and
    #    # by construction (see str_to_outcome in _extract_labels_from_col_labels) the
    #    # outcome labels in fill_info are *always* tuples.
    #    if fill_info is not None:
    #        countCols, freqCols, impliedCountTotCol1Q = fill_info
    #
    #        for outcomeLabel, iCol in countCols:
    #            if col_values[iCol] == '--': continue  # skip blank sentinels
    #            if col_values[iCol] > 0 and col_values[iCol] < 1:
    #                _warnings.warn("Count column (%d) contains value(s) between 0 and 1 - "
    #                               "could this be a frequency?" % iCol)
    #            assert(not isinstance(col_values[iCol], tuple)), \
    #                "Expanded-format count not allowed with column-key header"
    #            count_dict.setitem_unsafe(outcomeLabel, col_values[iCol])
    #
    #        for outcomeLabel, iCol, iTotCol in freqCols:
    #            if col_values[iCol] == '--' or col_values[iTotCol] == '--': continue  # skip blank sentinels
    #            if col_values[iCol] < 0 or col_values[iCol] > 1.0:
    #                _warnings.warn("Frequency column (%d) contains value(s) outside of [0,1.0] interval - "
    #                               "could this be a count?" % iCol)
    #            assert(not isinstance(col_values[iTotCol], tuple)), \
    #                "Expanded-format count not allowed with column-key header"
    #            count_dict.setitem_unsafe(outcomeLabel, col_values[iCol] * col_values[iTotCol])
    #
    #        if impliedCountTotCol1Q[1] >= 0:
    #            impliedOutcomeLabel, impliedCountTotCol = impliedCountTotCol1Q
    #            if impliedOutcomeLabel == ('0',):
    #                count_dict.setitem_unsafe(('0',), col_values[impliedCountTotCol] - count_dict[('1',)])
    #            else:
    #                count_dict.setitem_unsafe(('1',), col_values[impliedCountTotCol] - count_dict[('0',)])
    #
    #    else:  # assume col_values is a list of (outcomeLabel, count) tuples
    #        for tup in col_values:
    #            assert(isinstance(tup, tuple)), \
    #                ("Outcome labels must be specified with"
    #                 "count data when there's no column-key header")
    #            assert(len(tup) == 2), "Invalid count! (parsed to %s)" % str(tup)
    #            count_dict.setitem_unsafe(tup[0], tup[1])
    #    return count_dict

    def parse_multidatafile(self, filename, show_progress=True,
                            collision_action="aggregate", record_zero_counts=True, ignore_zero_count_lines=True):
        """
        Parse a multiple data set file into a MultiDataSet object.

        Parameters
        ----------
        filename : string
            The file to parse.

        show_progress : bool, optional
            Whether or not progress should be displayed

        collision_action : {"aggregate", "keepseparate"}
            Specifies how duplicate circuits should be handled.  "aggregate"
            adds duplicate-circuit counts, whereas "keepseparate" tags duplicate
            circuits by setting their `.occurrence` IDs to sequential positive integers.

        record_zero_counts : bool, optional
            Whether zero-counts are actually recorded (stored) in the returned
            MultiDataSet.  If False, then zero counts are ignored, except for
            potentially registering new outcome labels.

        ignore_zero_count_lines : bool, optional
            Whether circuits for which there are no counts should be ignored
            (i.e. omitted from the MultiDataSet) or not.

        Returns
        -------
        MultiDataSet
            A MultiDataSet object.
        """

        #Parse preamble -- lines beginning with # or ## until first non-# line
        preamble_directives = {}
        preamble_comments = []
        with open(filename, 'r') as multidatafile:
            for line in multidatafile:
                line = line.strip()
                if len(line) == 0 or line[0] != '#': break
                if line.startswith("## "):
                    parts = line[len("## "):].split("=")
                    if len(parts) == 2:  # key = value
                        preamble_directives[parts[0].strip()] = parts[1].strip()
                elif line.startswith("#"):
                    preamble_comments.append(line[1:].strip())

        #Process premble
        orig_cwd = _os.getcwd()
        if len(_os.path.dirname(filename)) > 0:
            _os.chdir(_os.path.dirname(filename))  # allow paths relative to datafile path
        try:
            if 'Lookup' in preamble_directives:
                lookupDict = self.parse_dictfile(preamble_directives['Lookup'])
            else: lookupDict = {}
            if 'Columns' in preamble_directives:
                colLabels = [l.strip() for l in preamble_directives['Columns'].split(",")]
            else: colLabels = ['dataset1 1 count', 'dataset1 count total']
            dsOutcomeLabels, fillInfo = self._extract_labels_from_multi_data_col_labels(colLabels)
            nDataCols = len(colLabels)
        finally:
            _os.chdir(orig_cwd)

        #Read data lines of data file
        datasets = _OrderedDict()
        for dsLabel, outcomeLabels in dsOutcomeLabels.items():
            datasets[dsLabel] = _objs.DataSet(outcome_labels=outcomeLabels,
                                              collision_action=collision_action)

        dsCountDicts = _OrderedDict()
        for dsLabel in dsOutcomeLabels: dsCountDicts[dsLabel] = {}

        nLines = 0
        with open(filename, 'r') as datafile:
            nLines = sum(1 for line in datafile)
        nSkip = max(int(nLines / 100.0), 1)

        display_progress = _create_display_progress_fn(show_progress)
        warnings = []  # to display *after* display progress
        mds = _objs.MultiDataSet(comment="\n".join(preamble_comments))

        with open(filename, 'r') as inputfile:
            for (iLine, line) in enumerate(inputfile):
                if iLine % nSkip == 0 or iLine + 1 == nLines: display_progress(iLine + 1, nLines, filename)

                line = line.strip()
                if '#' in line:
                    i = line.index('#')
                    dataline, comment = line[:i], line[i + 1:]
                else:
                    dataline, comment = line, ""
                if len(dataline) == 0: continue

                try:
                    circuit, valueList = \
                        self.parse_dataline(dataline, lookupDict, nDataCols,
                                            create_subcircuits=not _objs.Circuit.default_expand_subcircuits)

                    commentDict = {}
                    comment = comment.strip()
                    if len(comment) > 0:
                        try:
                            if comment.startswith("{") and comment.endswith("}"):
                                commentDict = _ast.literal_eval(comment)
                            else:  # put brackets around it
                                commentDict = _ast.literal_eval("{ " + comment + " }")
                        except:
                            warnings.append("%s Line %d: Could not parse comment '%s'"
                                            % (filename, iLine, comment))

                except ValueError as e:
                    raise ValueError("%s Line %d: %s" % (filename, iLine, str(e)))

                bBad = ('BAD' in valueList)  # supresses warnings
                for count_dict in dsCountDicts.values(): count_dict.clear()  # reset before filling
                self._fill_multi_data_count_dicts(dsCountDicts, fillInfo, valueList)

                bSkip = False
                if all([(abs(v) < 1e-9) for cDict in dsCountDicts.values() for v in cDict.values()]):
                    if ignore_zero_count_lines:
                        if not bBad:
                            s = circuit.str if len(circuit.str) < 40 else circuit.str[0:37] + "..."
                            warnings.append("Dataline for circuit '%s' has zero counts and will be ignored" % s)
                        bSkip = True  # skip lines in dataset file with zero counts (no experiments done)
                    else:
                        if not bBad:
                            s = circuit.str if len(circuit.str) < 40 else circuit.str[0:37] + "..."
                            warnings.append("Dataline for circuit '%s' has zero counts." % s)

                if not bSkip:
                    for dsLabel, countDict in dsCountDicts.items():
                        datasets[dsLabel].add_count_dict(
                            circuit, countDict, record_zero_counts=record_zero_counts, update_ol=False)
                        mds.add_auxiliary_info(circuit, commentDict)

        for dsLabel, ds in datasets.items():
            ds.update_ol()  # because we set update_ol=False above, we need to do this
            ds.done_adding_data()
            # auxinfo already added, and ds shouldn't have any anyway
            mds.add_dataset(dsLabel, ds, update_auxinfo=False)
        return mds

    #Note: outcome labels must not contain spaces since we use spaces to separate
    # the outcome label from the dataset label

    def _extract_labels_from_multi_data_col_labels(self, col_labels):

        def str_to_outcome(x):  # always return a tuple as the "outcome label" (even if length 1)
            return tuple(x.strip().split(":"))

        dsOutcomeLabels = _OrderedDict()
        countCols = []; freqCols = []; impliedCounts1Q = []
        for i, colLabel in enumerate(col_labels):
            wordsInColLabel = colLabel.split()  # split on whitespace into words
            if len(wordsInColLabel) < 3: continue  # allow other columns we don't recognize

            if wordsInColLabel[-1] == 'count':
                if len(wordsInColLabel) > 3:
                    _warnings.warn("Column label '%s' has more words than were expected (3)" % colLabel)
                outcomeLabel = str_to_outcome(wordsInColLabel[-2])
                dsLabel = wordsInColLabel[-3]
                if dsLabel not in dsOutcomeLabels:
                    dsOutcomeLabels[dsLabel] = [outcomeLabel]
                else: dsOutcomeLabels[dsLabel].append(outcomeLabel)
                countCols.append((dsLabel, outcomeLabel, i))

            elif wordsInColLabel[-1] == 'frequency':
                if len(wordsInColLabel) > 3:
                    _warnings.warn("Column label '%s' has more words than were expected (3)" % colLabel)
                outcomeLabel = str_to_outcome(wordsInColLabel[-2])
                dsLabel = wordsInColLabel[-3]
                if '%s count total' % dsLabel not in col_labels:
                    raise ValueError("Frequency columns specified without"
                                     "count total for dataset '%s'" % dsLabel)
                else: iTotal = col_labels.index('%s count total' % dsLabel)

                if dsLabel not in dsOutcomeLabels:
                    dsOutcomeLabels[dsLabel] = [outcomeLabel]
                else: dsOutcomeLabels[dsLabel].append(outcomeLabel)
                freqCols.append((dsLabel, outcomeLabel, i, iTotal))

        for dsLabel, outcomeLabels in dsOutcomeLabels.items():
            if '%s count total' % dsLabel in col_labels:
                if ('1',) in outcomeLabels and ('0',) not in outcomeLabels:
                    dsOutcomeLabels[dsLabel].append(('0',))
                    iTotal = col_labels.index('%s count total' % dsLabel)
                    impliedCounts1Q.append((dsLabel, ('0',), iTotal))
                if ('0',) in outcomeLabels and ('1',) not in outcomeLabels:
                    dsOutcomeLabels[dsLabel].append(('1',))
                    iTotal = col_labels.index('%s count total' % dsLabel)
                    impliedCounts1Q.append((dsLabel, ('1',), iTotal))

            #TODO - add standard count completion for 2Qubit case?

        fillInfo = (countCols, freqCols, impliedCounts1Q)
        return dsOutcomeLabels, fillInfo

    def _fill_multi_data_count_dicts(self, count_dicts, fill_info, col_values):
        countCols, freqCols, impliedCounts1Q = fill_info

        for dsLabel, outcomeLabel, iCol in countCols:
            if col_values[iCol] == '--':
                continue
            if col_values[iCol] > 0 and col_values[iCol] < 1:
                raise ValueError("Count column (%d) contains value(s) between 0 and 1 - "
                                 "could this be a frequency?" % iCol)
            count_dicts[dsLabel][outcomeLabel] = col_values[iCol]

        for dsLabel, outcomeLabel, iCol, iTotCol in freqCols:
            if col_values[iCol] == '--':
                continue
            if col_values[iCol] < 0 or col_values[iCol] > 1.0:
                raise ValueError("Frequency column (%d) contains value(s) outside of [0,1.0] interval - "
                                 "could this be a count?" % iCol)
            count_dicts[dsLabel][outcomeLabel] = col_values[iCol] * col_values[iTotCol]

        for dsLabel, outcomeLabel, iTotCol in impliedCounts1Q:
            if col_values[iTotCol] == '--': raise ValueError("Mising total (== '--')!")
            if outcomeLabel == '0':
                count_dicts[dsLabel]['0'] = col_values[iTotCol] - count_dicts[dsLabel]['1']
            elif outcomeLabel == '1':
                count_dicts[dsLabel]['1'] = col_values[iTotCol] - count_dicts[dsLabel]['0']

        #TODO - add standard count completion for 2Qubit case?
        return count_dicts

    def parse_tddatafile(self, filename, show_progress=True, record_zero_counts=True,
                         create_subcircuits=True):
        """
        Parse a timstamped data set file into a DataSet object.

        Parameters
        ----------
        filename : string
            The file to parse.

        show_progress : bool, optional
            Whether or not progress should be displayed

        record_zero_counts : bool, optional
            Whether zero-counts are actually recorded (stored) in the returned
            DataSet.  If False, then zero counts are ignored, except for
            potentially registering new outcome labels.

        create_subcircuits : bool, optional
            Whether to create sub-circuit-labels when parsing
            string representations or to just expand these into non-subcircuit
            labels.

        Returns
        -------
        DataSet
            A static DataSet object.
        """

        #Parse preamble -- lines beginning with # or ## until first non-# line
        preamble_directives = _OrderedDict()
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if len(line) == 0 or line[0] != '#': break
                if line.startswith("## "):
                    parts = line[len("## "):].split("=")
                    if len(parts) == 2:  # key = value
                        preamble_directives[parts[0].strip()] = parts[1].strip()

        #Process premble
        orig_cwd = _os.getcwd()
        if len(_os.path.dirname(filename)) > 0: _os.chdir(
            _os.path.dirname(filename))  # allow paths relative to datafile path
        try:
            if 'Lookup' in preamble_directives:
                lookupDict = self.parse_dictfile(preamble_directives['Lookup'])
            else: lookupDict = {}
        finally:
            _os.chdir(orig_cwd)

        outcomeLabelAbbrevs = _OrderedDict()
        for key, val in preamble_directives.items():
            if key == "Lookup": continue
            outcomeLabelAbbrevs[key] = val
        outcomeLabels = outcomeLabelAbbrevs.values()

        #Read data lines of data file
        dataset = _objs.DataSet(outcome_labels=outcomeLabels)
        with open(filename, 'r') as f:
            nLines = sum(1 for line in f)
        nSkip = int(nLines / 100.0)
        if nSkip == 0: nSkip = 1

        display_progress = _create_display_progress_fn(show_progress)

        with open(filename, 'r') as f:
            for (iLine, line) in enumerate(f):
                if iLine % nSkip == 0 or iLine + 1 == nLines: display_progress(iLine + 1, nLines, filename)

                line = line.strip()
                if len(line) == 0 or line[0] == '#': continue
                try:
                    parts = line.split()
                    lastpart = parts[-1]
                    circuitStr = line[:-len(lastpart)].strip()
                    circuit = self.parse_circuit(circuitStr, lookupDict, create_subcircuits)
                    timeSeriesStr = lastpart.strip()
                except ValueError as e:
                    raise ValueError("%s Line %d: %s" % (filename, iLine, str(e)))

                seriesList = [outcomeLabelAbbrevs[abbrev] for abbrev in timeSeriesStr]  # iter over characters in str
                timesList = list(range(len(seriesList)))  # FUTURE: specify an offset and step??
                dataset.add_raw_series_data(circuit, seriesList, timesList,
                                            record_zero_counts=record_zero_counts)

        dataset.done_adding_data()
        return dataset


def _eval_element(el, b_complex):
    myLocal = {'pi': _np.pi, 'sqrt': _np.sqrt}
    exec("element = %s" % el, {"__builtins__": None}, myLocal)
    return complex(myLocal['element']) if b_complex else float(myLocal['element'])


def _eval_row_list(rows, b_complex):
    return _np.array([[_eval_element(x, b_complex) for x in r] for r in rows],
                     'complex' if b_complex else 'd')


def parse_model(filename):
    """
    Parse a model file into a Model object.

    Parameters
    ----------
    filename : string
        The file to parse.

    Returns
    -------
    Model
    """
    from ..models import ExplicitOpModel as _ExplicitOpModel
    basis = 'pp'  # default basis to load as

    basis_abbrev = "pp"  # default assumed basis
    basis_dim = None
    gaugegroup_name = None
    state_space = None

    #First try to find basis:
    with open(filename) as inputfile:
        for line in inputfile:
            line = line.strip()

            if line.startswith("BASIS:"):
                parts = line[len("BASIS:"):].split()
                basis_abbrev = parts[0]
                if len(parts) > 1:
                    basis_dims = list(map(int, "".join(parts[1:]).split(",")))
                    assert(len(basis_dims) == 1), "Multiple basis dims is no longer supported!"
                    basis_dim = basis_dims[0]
                else:
                    basis_dim = None
            elif line.startswith("GAUGEGROUP:"):
                gaugegroup_name = line[len("GAUGEGROUP:"):].strip()
                if gaugegroup_name not in ("Full", "TP", "Unitary"):
                    _warnings.warn(("Unknown GAUGEGROUP name %s.  Default gauge"
                                    "group will be set to None") % gaugegroup_name)
            elif line.startswith("STATESPACE:"):
                tpbs_lbls = []; tpbs_udims = []
                tensor_prod_blk_strs = line[len("STATESPACE:"):].split("+")
                for tpb_str in tensor_prod_blk_strs:
                    tpb_lbls = []; tpb_udims = []
                    for lbl_and_dim in tpb_str.split("*"):
                        start = lbl_and_dim.index('(')
                        end = lbl_and_dim.rindex(')')
                        lbl, dim = lbl_and_dim[:start], lbl_and_dim[start + 1:end]
                        tpb_lbls.append(lbl.strip())
                        tpb_udims.append(int(_np.sqrt(int(dim.strip()))))
                    tpbs_lbls.append(tuple(tpb_lbls))
                    tpbs_udims.append(tuple(tpb_udims))
                state_space = _statespace.ExplicitStateSpace(tpbs_lbls, tpbs_udims)

    if basis_dim is not None:
        # then specfy a dimensionful basis at the outset
        # basis_dims should be just a single int now that the *vector-space* dimension
        basis = _objs.BuiltinBasis(basis_abbrev, basis_dim)
    else:
        # otherwise we'll try to infer one from state space labels
        if state_space is not None:
            basis = _objs.Basis.cast(basis_abbrev, state_space.dim)
        else:
            raise ValueError("Cannot infer basis dimension!")

    if state_space is None:
        assert(basis_dim is not None)  # b/c of logic above
        state_space = _statespace.ExplicitStateSpace(['*'], [basis_dim])
        # special '*' state space label w/entire dimension inferred from BASIS line

    mdl = _ExplicitOpModel(state_space, basis)

    state = "look for label or property"
    cur_obj = None
    cur_group_obj = None
    cur_property = ""; cur_rows = []
    top_level_objs = []

    def to_int(x):  # tries to convert state space labels to integers, but if fails OK
        try: return int(x)
        except Exception: return x

    with open(filename) as inputfile:
        for line in inputfile:
            line = line.strip()

            if len(line) == 0 or line.startswith("END"):
                #Blank lines or "END..." statements trigger the end of properties
                state = "look for label or property"
                if len(cur_property) > 0:
                    assert((cur_obj is not None) or (cur_group_obj is not None)), \
                        "No object to add %s property to!" % cur_property
                    obj = cur_obj if (cur_obj is not None) else cur_group_obj
                    obj['properties'][cur_property] = cur_rows
                    cur_property = ""; cur_rows = []

                #END... ends the current group
                if line.startswith("END"):
                    assert(cur_group_obj is not None), "%s does not correspond to any object group!" % line
                    if cur_obj is not None:
                        cur_group_obj['objects'].append(cur_obj); cur_obj = None
                    top_level_objs.append(cur_group_obj); cur_group_obj = None

            elif line[0] == "#":
                pass  # skip comments

            elif state == "look for label or property":
                assert(cur_property == ""), "Logic error!"

                parts = line.split(':')
                if any([line.startswith(pre) for pre in ("BASIS", "GAUGEGROUP", "STATESPACE")]):
                    pass  # handled above

                elif len(parts) >= 2:  # then this is a '<type>: <label>' line => new cur_obj
                    typ = parts[0].strip()
                    label = _objs.Label(name=parts[1].strip() if parts[1].strip() != "[]" else (),
                                        state_space_labels=tuple(map(to_int, parts[2:])) if len(parts) > 2 else None)

                    # place any existing cur_obj
                    if cur_obj is not None:
                        if cur_group_obj is not None:
                            cur_group_obj['objects'].append(cur_obj)
                        else:
                            top_level_objs.append(cur_obj)
                        cur_obj = None

                    if typ in ("POVM", "TP-POVM", "CPTP-POVM", "Instrument", "TP-Instrument"):
                        # a group type - so create a new *group* object
                        assert(cur_group_obj is None), "Group label encountered before ENDing prior group:\n%s" % line
                        cur_group_obj = {'label': label, 'type': typ, 'properties': {}, 'objects': []}
                    else:
                        #All other "types" are object labels
                        cur_obj = {'label': label, 'type': typ, 'properties': {}}

                elif len(parts) == 1:
                    # a "property" line - either just <prop_name> (for a
                    # multiline format) or <prop_name> = <value>
                    assert((cur_obj is not None) or (cur_group_obj is not None)), \
                        "Property: %s\nencountered without a containing object!" % line
                    eqparts = line.split('=')

                    if len(eqparts) == 2:
                        lhs = eqparts[0].strip()
                        rhs = eqparts[1].strip()
                        obj = cur_obj if (cur_obj is not None) else cur_group_obj
                        obj['properties'][lhs] = _ast.literal_eval(rhs)
                    elif len(eqparts) == 1:
                        cur_property = eqparts[0].strip()
                        state = "read array"
                    else:
                        raise ValueError("Invalid property definition: %s" % line)
                else:
                    raise ValueError("Line: %s\nDoes not look like an object label or property!" % line)

            elif state == "read array":
                cur_rows.append(line.split())

    #Deal with any lingering properties or objects
    if len(cur_property) > 0:
        assert((cur_obj is not None) or (cur_group_obj is not None)), \
            "No object to add %s property to!" % cur_property
        obj = cur_obj if (cur_obj is not None) else cur_group_obj
        obj['properties'][cur_property] = cur_rows

    if cur_obj is not None:
        if cur_group_obj is not None:
            cur_group_obj['objects'].append(cur_obj)
        else:
            top_level_objs.append(cur_obj)

    if cur_group_obj is not None:
        top_level_objs.append(cur_group_obj)

    def get_liouville_mx(obj, prefix=""):
        """ Process properties of `obj` to extract a single liouville representation """
        props = obj['properties']; lmx = None
        if prefix + "StateVec" in props:
            ar = _eval_row_list(props[prefix + "StateVec"], b_complex=True)
            if ar.shape == (1, 2):
                stdmx = _tools.state_to_stdmx(ar[0, :])
                lmx = _tools.stdmx_to_vec(stdmx, basis)
            else: raise ValueError("Invalid state vector shape for %s: %s" % (cur_label, ar.shape))

        elif prefix + "DensityMx" in props:
            ar = _eval_row_list(props[prefix + "DensityMx"], b_complex=True)
            if ar.shape == (2, 2) or ar.shape == (4, 4):
                lmx = _tools.stdmx_to_vec(ar, basis)
            else: raise ValueError("Invalid density matrix shape for %s: %s" % (cur_label, ar.shape))

        elif prefix + "LiouvilleVec" in props:
            lmx = _np.transpose(_eval_row_list(props[prefix + "LiouvilleVec"], b_complex=False))

        elif prefix + "UnitaryMx" in props:
            ar = _eval_row_list(props[prefix + "UnitaryMx"], b_complex=True)
            lmx = _tools.change_basis(_tools.unitary_to_process_mx(ar), 'std', basis)

        elif prefix + "UnitaryMxExp" in props:
            ar = _eval_row_list(props[prefix + "UnitaryMxExp"], b_complex=True)
            lmx = _tools.change_basis(_tools.unitary_to_process_mx(_expm(-1j * ar)), 'std', basis)

        elif prefix + "LiouvilleMx" in props:
            lmx = _eval_row_list(props[prefix + "LiouvilleMx"], b_complex=False)

        if lmx is None:
            raise ValueError("No valid format found in %s" % str(list(props.keys())))

        return lmx

    #Now process top_level_objs to create a Model
    for obj in top_level_objs:  # `obj` is a dict of object info
        cur_typ = obj['type']
        cur_label = obj['label']

        #Preps
        if cur_typ == "PREP":
            mdl.preps[cur_label] = _state.FullState(
                get_liouville_mx(obj))
        elif cur_typ == "TP-PREP":
            mdl.preps[cur_label] = _state.TPState(
                get_liouville_mx(obj))
        elif cur_typ == "CPTP-PREP":
            props = obj['properties']
            assert("PureVec" in props and "ErrgenMx" in props)  # must always be Liouville reps!
            qty = _eval_row_list(props["ErrgenMx"], b_complex=False)
            nQubits = _np.log2(qty.size) / 2.0
            bQubits = bool(abs(nQubits - round(nQubits)) < 1e-10)  # integer # of qubits?
            proj_basis = "pp" if (basis == "pp" or bQubits) else basis
            errorgen = _op.LinbladErrorgen.from_operation_matrix(
                qty, proj_basis, proj_basis, truncate=False, mx_basis=basis)
            errorMap = _op.ExpErrorgenOp(errorgen)
            pureVec = _state.StaticState(_np.transpose(_eval_row_list(props["PureVec"], b_complex=False)))
            mdl.preps[cur_label] = _state.ComposedState(pureVec, errorMap)
        elif cur_typ == "STATIC-PREP":
            mdl.preps[cur_label] = _state.StaticState(get_liouville_mx(obj))

        #POVMs
        elif cur_typ in ("POVM", "TP-POVM", "CPTP-POVM"):
            effects = []
            for sub_obj in obj['objects']:
                sub_typ = sub_obj['type']
                if sub_typ == "EFFECT":
                    Evec = _povm.FullPOVMEffect(get_liouville_mx(sub_obj))
                elif sub_typ == "STATIC-EFFECT":
                    Evec = _povm.StaticPOVMEffect(get_liouville_mx(sub_obj))
                #elif sub_typ == "CPTP-EFFECT":
                #    Evec = _objs.LindbladSPAMVec.from_spam_vector(qty,qty,"effect")
                effects.append((sub_obj['label'], Evec))

            if cur_typ == "POVM":
                mdl.povms[cur_label] = _povm.UnconstrainedPOVM(effects)
            elif cur_typ == "TP-POVM":
                assert(len(effects) > 1), "TP-POVMs must have at least 2 elements!"
                mdl.povms[cur_label] = _povm.TPPOVM(effects)
            elif cur_typ == "CPTP-POVM":
                props = obj['properties']
                assert("ErrgenMx" in props)  # and it must always be a Liouville rep!
                qty = _eval_row_list(props["ErrgenMx"], b_complex=False)
                nQubits = _np.log2(qty.size) / 2.0
                bQubits = bool(abs(nQubits - round(nQubits)) < 1e-10)  # integer # of qubits?
                proj_basis = "pp" if (basis == "pp" or bQubits) else basis
                errorgen = _op.LinbladErrorgen.from_operation_matrix(
                    qty, proj_basis, proj_basis, truncate=False, mx_basis=basis)
                errorMap = _op.ExpErrorgenOp(errorgen)
                base_povm = _povm.UnconstrainedPOVM(effects)  # could try to detect a ComputationalBasisPOVM in FUTURE
                mdl.povms[cur_label] = _povm.ComposedPOVM(errorMap, base_povm)
            else: assert(False), "Logic error!"

        elif cur_typ == "GATE":
            mdl.operations[cur_label] = _op.FullDenseOp(
                get_liouville_mx(obj))
        elif cur_typ == "TP-GATE":
            mdl.operations[cur_label] = _op.TPDenseOp(
                get_liouville_mx(obj))
        elif cur_typ == "CPTP-GATE":
            qty = get_liouville_mx(obj)
            try:
                unitary_post = get_liouville_mx(obj, "Ref")
            except ValueError:
                unitary_post = None
            nQubits = _np.log2(qty.shape[0]) / 2.0
            bQubits = bool(abs(nQubits - round(nQubits)) < 1e-10)  # integer # of qubits?
            proj_basis = "pp" if (basis == "pp" or bQubits) else basis
            mdl.operations[cur_label] = _op.ComposedOp(
                (_op.StaticOp(unitary_post),
                 _op.ExpErrogenOp(_op.LinbladErrorgen.from_operation_matrix(
                     qty, proj_basis, proj_basis, truncate=False, mx_basis=basis))))

        elif cur_typ == "STATIC-GATE":
            mdl.operations[cur_label] = _op.StaticDenseOp(get_liouville_mx(obj))

        elif cur_typ in ("Instrument", "TP-Instrument"):
            matrices = []
            for sub_obj in obj['objects']:
                sub_typ = sub_obj['type']
                qty = get_liouville_mx(sub_obj)
                mxOrOp = _op.StaticDenseOp(qty) if cur_typ == "STATIC-IGATE" \
                    else qty  # just add numpy array `qty` to matrices list
                # and it will be made into a fully-param gate.
                matrices.append((sub_obj['label'], mxOrOp))

            if cur_typ == "Instrument":
                mdl.instruments[cur_label] = _instrument.Instrument(matrices)
            elif cur_typ == "TP-Instrument":
                mdl.instruments[cur_label] = _instrument.TPInstrument(matrices)
            else: assert(False), "Logic error!"
        else:
            raise ValueError("Unknown type: %s!" % cur_typ)

    #Add default gauge group -- the full group because
    # we add FullyParameterizedGates above.
    if gaugegroup_name == "Full":
        mdl.default_gauge_group = _objs.FullGaugeGroup(mdl.state_space, mdl.evotype)
    elif gaugegroup_name == "TP":
        mdl.default_gauge_group = _objs.TPGaugeGroup(mdl.state_space, mdl.evotype)
    elif gaugegroup_name == "Unitary":
        mdl.default_gauge_group = _objs.UnitaryGaugeGroup(mdl.state_space, mdl.basis, mdl.evotype)
    else:
        mdl.default_gauge_group = None

    return mdl
