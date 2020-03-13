""" Text-parsing classes and functions to read input files."""
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

from .. import objects as _objs
from .. import tools as _tools

from . import CircuitParser as _CircuitParser


def get_display_progress_fn(show_progress):
    """
    Create and return a progress-displaying function if `show_progress == True`
    and it's run within an interactive environment.
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

    def __init__(self):
        """ Create a new standard-input parser object """
        pass

    def parse_circuit(self, s, lookup={}, create_subcircuits=True):
        """
        Parse a operation sequence (string in grammar)

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
            Representing the operation sequence.
        """
        self._circuit_parser.lookup = lookup
        circuit_tuple, circuit_labels = self._circuit_parser.parse(s, create_subcircuits)
        # print "DB: result = ",result
        # print "DB: stack = ",self.exprStack
        return circuit_tuple, circuit_labels

    def parse_dataline(self, s, lookup={}, expected_counts=-1, create_subcircuits=True):
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
            The expected number of counts to accompany the operation sequence on this
            data line.  If < 0, no check is performed; otherwise raises ValueError
            if the number of counts does not equal expected_counts.

        Returns
        -------
        circuitTuple : tuple
            The circuit as a tuple of layer-operation labels.
        circuitStr : string
            The circuit as represented as a string in the dataline (minus any line labels)
        circuitLabels : tuple
            A tuple of the circuit's line labels (given after '@' symbol on line)
        counts : list
            List of counts following the operation sequence.
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

        circuitTuple, circuitLabels = self.parse_circuit(circuitStr, lookup, create_subcircuits)
        return circuitTuple, circuitStr, circuitLabels, counts

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
            The user-defined label to represent this operation sequence.
        circuitTuple : tuple
            The operation sequence as a tuple of operation labels.
        circuitStr : string
            The operation sequence as represented as a string in the dictline.
        """
        label = r'\s*([a-zA-Z0-9_]+)\s+'
        match = _re.match(label, s)
        if not match:
            raise ValueError("'{}' is not a valid dictline".format(s))
        circuitLabel = match.group(1)
        circuitStr = s[match.end():]
        circuitTuple, circuitLineLabels = self._circuit_parser.parse(circuitStr)
        return circuitLabel, circuitTuple, circuitStr, circuitLineLabels

    def parse_stringfile(self, filename, line_labels="auto", num_lines=None):
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
                layer_lbls, line_lbls = self.parse_circuit(line)
                if line_lbls is None:
                    line_lbls = line_labels  # default to the passed-in argument
                    nlines = num_lines
                else: nlines = None  # b/c we've got a valid line_lbls

                circuit_list.append(_objs.Circuit(layer_lbls, stringrep=line.strip(),
                                                  line_labels=line_lbls, num_lines=nlines, check=False))
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
           Dictionary with keys == operation sequence labels and values == Circuits.
        """
        lookupDict = {}
        with open(filename, 'r') as dictfile:
            for line in dictfile:
                line = line.strip()
                if len(line) == 0 or line[0] == '#': continue
                label, tup, s, lineLbls = self.parse_dictline(line)
                if lineLbls is None: lineLbls = "auto"
                lookupDict[label] = _objs.Circuit(tup, stringrep=s, line_labels=lineLbls, check=False)
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
            Specifies how duplicate operation sequences should be handled.  "aggregate"
            adds duplicate-sequence counts, whereas "keepseparate" tags duplicate-
            sequence data with by appending a final "#<number>" operation label to the
            duplicated gate sequence.

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

        #Process premble
        orig_cwd = _os.getcwd()
        outcomeLabels = None
        if len(_os.path.dirname(filename)) > 0: _os.chdir(
            _os.path.dirname(filename))  # allow paths relative to datafile path
        try:
            if 'Lookup' in preamble_directives:
                lookupDict = self.parse_dictfile(preamble_directives['Lookup'])
            else: lookupDict = {}
            if 'Columns' in preamble_directives:
                colLabels = [l.strip() for l in preamble_directives['Columns'].split(",")]
                outcomeLabels, fillInfo = self._extract_labels_from_col_labels(colLabels)
                nDataCols = len(colLabels)
            else:
                fillInfo = None
                nDataCols = -1  # no column count check
            if 'Outcomes' in preamble_directives:
                outcomeLabels = [l.strip().split(':') for l in preamble_directives['Outcomes'].split(",")]
            if 'StdOutcomeQubits' in preamble_directives:
                outcomeLabels = int(preamble_directives['Outcomes'])
        finally:
            _os.chdir(orig_cwd)

        #Read data lines of data file
        dataset = _objs.DataSet(outcome_labels=outcomeLabels, collision_action=collision_action,
                                comment="\n".join(preamble_comments))
        nLines = 0
        with open(filename, 'r') as datafile:
            nLines = sum(1 for line in datafile)
        nSkip = int(nLines / 100.0)
        if nSkip == 0: nSkip = 1

        display_progress = get_display_progress_fn(show_progress)
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

        with open(filename, 'r') as inputfile:
            for (iLine, line) in enumerate(inputfile):
                if iLine % nSkip == 0 or iLine + 1 == nLines: display_progress(iLine + 1, nLines, filename)

                line = line.strip()
                if '#' in line:
                    i = line.index('#')
                    dataline, comment = line[:i], line[i + 1:]
                else:
                    dataline, comment = line, ""

                if looking_for == "circuit_line":
                    if len(dataline) == 0: continue
                    try:
                        circuitTuple, circuitStr, circuitLbls, valueList = \
                            self.parse_dataline(dataline, lookupDict, nDataCols,
                                                create_subcircuits=not _objs.Circuit.default_expand_subcircuits)

                        commentDict = parse_comment(comment, filename, iLine)

                    except ValueError as e:
                        raise ValueError("%s Line %d: %s" % (filename, iLine, str(e)))

                    if circuitLbls is None: circuitLbls = "auto"  # if line labels weren't given just use defaults
                    circuit = _objs.Circuit(circuitTuple, stringrep=circuitStr,
                                            line_labels=circuitLbls, expand_subcircuits=False, check=False)
                    #Note: don't expand subcircuits because we've already directed parse_dataline to expand if needed

                    if with_times is True and len(valueList) > 0:
                        raise ValueError(("%s Line %d: Circuit line cannot contain count information when "
                                          "'with_times=True'") % (filename, iLine))

                    if with_times is False or len(valueList) > 0:
                        bBad = ('BAD' in valueList)  # supresses warnings
                        countDict = _objs.labeldicts.OutcomeLabelDict()
                        self._fill_data_count_dict(countDict, fillInfo, valueList)
                        if all([(abs(v) < 1e-9) for v in list(countDict.values())]):
                            if ignore_zero_count_lines is True:
                                if not bBad:
                                    s = circuitStr if len(circuitStr) < 40 else circuitStr[0:37] + "..."
                                    warnings.append("Dataline for circuit '%s' has zero counts and will be ignored" % s)
                                continue  # skip lines in dataset file with zero counts (no experiments done)
                            else:
                                #if not bBad:
                                #    s = circuitStr if len(circuitStr) < 40 else circuitStr[0:37] + "..."
                                #    warnings.append("Dataline for circuit '%s' has zero counts." % s)
                                # don't make a fuss if we don't ignore the lines (needed for
                                # fill_in_empty_dataset_with_fake_data).
                                pass

                        dataset.add_count_dict(circuit, countDict, aux=commentDict, record_zero_counts=record_zero_counts,
                                               update_ol=False)  # for performance - to this once at the end.
                    else:
                        current_item['circuit'] = circuit
                        looking_for = "circuit_data"

                elif looking_for == "circuit_data":
                    if len(line) == 0:
                        #add current item & look for next one
                        dataset.add_raw_series_data(current_item['circuit'], current_item['outcomes'],
                                                    current_item['times'], current_item.get('repetitions', None),
                                                    record_zero_counts=record_zero_counts, aux=current_item.get('aux', None),
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
                            current_item['repetitions'] = [int(x) for x in parts[1:]]
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

    def _extract_labels_from_col_labels(self, col_labels):
        outcomeLabels = []; countCols = []; freqCols = []; impliedCountTotCol1Q = (-1, -1)

        def str_to_outcome(x):  # always return a tuple as the "outcome label" (even if length 1)
            return tuple(x.strip().split(":"))

        for i, colLabel in enumerate(col_labels):
            if colLabel.endswith(' count'):
                outcomeLabel = str_to_outcome(colLabel[:-len(' count')])
                if outcomeLabel not in outcomeLabels: outcomeLabels.append(outcomeLabel)
                countCols.append((outcomeLabel, i))

            elif colLabel.endswith(' frequency'):
                if 'count total' not in col_labels:
                    raise ValueError("Frequency columns specified without count total")
                else: iTotal = col_labels.index('count total')
                outcomeLabel = str_to_outcome(colLabel[:-len(' frequency')])
                if outcomeLabel not in outcomeLabels: outcomeLabels.append(outcomeLabel)
                freqCols.append((outcomeLabel, i, iTotal))

        if 'count total' in col_labels:
            if ('1',) in outcomeLabels and ('0',) not in outcomeLabels:
                outcomeLabels.append(('0',))
                impliedCountTotCol1Q = ('0',), col_labels.index('count total')
            elif ('0',) in outcomeLabels and ('1',) not in outcomeLabels:
                outcomeLabels.append(('1',))
                impliedCountTotCol1Q = '1', col_labels.index('count total')
            #TODO - add standard count completion for 2Qubit case?

        fillInfo = (countCols, freqCols, impliedCountTotCol1Q)
        return outcomeLabels, fillInfo

    def _fill_data_count_dict(self, count_dict, fill_info, col_values):
        if 'BAD' in col_values:
            return  # indicates entire row is known to be bad (no counts)

        #Note: can use set_unsafe here because count_dict is a OutcomeLabelDict and
        # by construction (see str_to_outcome in _extract_labels_from_col_labels) the
        # outcome labels in fill_info are *always* tuples.
        if fill_info is not None:
            countCols, freqCols, impliedCountTotCol1Q = fill_info

            for outcomeLabel, iCol in countCols:
                if col_values[iCol] == '--': continue  # skip blank sentinels
                if col_values[iCol] > 0 and col_values[iCol] < 1:
                    _warnings.warn("Count column (%d) contains value(s) between 0 and 1 - "
                                   "could this be a frequency?" % iCol)
                assert(not isinstance(col_values[iCol], tuple)), \
                    "Expanded-format count not allowed with column-key header"
                count_dict.set_unsafe(outcomeLabel, col_values[iCol])

            for outcomeLabel, iCol, iTotCol in freqCols:
                if col_values[iCol] == '--' or col_values[iTotCol] == '--': continue  # skip blank sentinels
                if col_values[iCol] < 0 or col_values[iCol] > 1.0:
                    _warnings.warn("Frequency column (%d) contains value(s) outside of [0,1.0] interval - "
                                   "could this be a count?" % iCol)
                assert(not isinstance(col_values[iTotCol], tuple)), \
                    "Expanded-format count not allowed with column-key header"
                count_dict.set_unsafe(outcomeLabel, col_values[iCol] * col_values[iTotCol])

            if impliedCountTotCol1Q[1] >= 0:
                impliedOutcomeLabel, impliedCountTotCol = impliedCountTotCol1Q
                if impliedOutcomeLabel == ('0',):
                    count_dict.set_unsafe(('0',), col_values[impliedCountTotCol] - count_dict[('1',)])
                else:
                    count_dict.set_unsafe(('1',), col_values[impliedCountTotCol] - count_dict[('0',)])

        else:  # assume col_values is a list of (outcomeLabel, count) tuples
            for tup in col_values:
                assert(isinstance(tup, tuple)), \
                    ("Outcome labels must be specified with"
                     "count data when there's no column-key header")
                assert(len(tup) == 2), "Invalid count! (parsed to %s)" % str(tup)
                count_dict.set_unsafe(tup[0], tup[1])
        return count_dict

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
            Specifies how duplicate operation sequences should be handled.  "aggregate"
            adds duplicate-sequence counts, whereas "keepseparate" tags duplicate-
            sequence data with by appending a final "#<number>" operation label to the
            duplicated gate sequence.

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

        display_progress = get_display_progress_fn(show_progress)
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
                    circuitTuple, circuitStr, circuitLbls, valueList = \
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

                if circuitLbls is None: circuitLbls = "auto"  # if line labels aren't given find them automatically
                opStr = _objs.Circuit(circuitTuple, stringrep=circuitStr, line_labels=circuitLbls,
                                      check=False, expand_subcircuits=False)  # , lookup=lookupDict)
                #Note: don't expand subcircuits because we've already directed parse_dataline to expand if needed
                bBad = ('BAD' in valueList)  # supresses warnings
                self._fill_multi_data_count_dicts(dsCountDicts, fillInfo, valueList)

                bSkip = False
                if all([(abs(v) < 1e-9) for cDict in dsCountDicts.values() for v in cDict.values()]):
                    if ignore_zero_count_lines:
                        if not bBad:
                            s = circuitStr if len(circuitStr) < 40 else circuitStr[0:37] + "..."
                            warnings.append("Dataline for circuit '%s' has zero counts and will be ignored" % s)
                        bSkip = True  # skip lines in dataset file with zero counts (no experiments done)
                    else:
                        if not bBad:
                            s = circuitStr if len(circuitStr) < 40 else circuitStr[0:37] + "..."
                            warnings.append("Dataline for circuit '%s' has zero counts." % s)

                if not bSkip:
                    for dsLabel, countDict in dsCountDicts.items():
                        datasets[dsLabel].add_count_dict(
                            opStr, countDict, record_zero_counts=record_zero_counts, update_ol=False)
                        mds.add_auxiliary_info(opStr, commentDict)

        for dsLabel, ds in datasets.items():
            ds.update_ol()  # because we set update_ol=False above, we need to do this
            ds.done_adding_data()
            # auxinfo already added, and ds shouldn't have any anyway
            mds.add_dataset(dsLabel, ds, update_auxinfo=False)
        return mds

    #Note: outcome labels must not contain spaces since we use spaces to separate
    # the outcome label from the dataset label

    def _extract_labels_from_multi_data_col_labels(self, col_labels):
        dsOutcomeLabels = _OrderedDict()
        countCols = []; freqCols = []; impliedCounts1Q = []
        for i, colLabel in enumerate(col_labels):
            wordsInColLabel = colLabel.split()  # split on whitespace into words
            if len(wordsInColLabel) < 3: continue  # allow other columns we don't recognize

            if wordsInColLabel[-1] == 'count':
                if len(wordsInColLabel) > 3:
                    _warnings.warn("Column label '%s' has more words than were expected (3)" % colLabel)
                outcomeLabel = wordsInColLabel[-2]
                dsLabel = wordsInColLabel[-3]
                if dsLabel not in dsOutcomeLabels:
                    dsOutcomeLabels[dsLabel] = [outcomeLabel]
                else: dsOutcomeLabels[dsLabel].append(outcomeLabel)
                countCols.append((dsLabel, outcomeLabel, i))

            elif wordsInColLabel[-1] == 'frequency':
                if len(wordsInColLabel) > 3:
                    _warnings.warn("Column label '%s' has more words than were expected (3)" % colLabel)
                outcomeLabel = wordsInColLabel[-2]
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
                if '1' in outcomeLabels and '0' not in outcomeLabels:
                    dsOutcomeLabels[dsLabel].append('0')
                    iTotal = col_labels.index('%s count total' % dsLabel)
                    impliedCounts1Q.append((dsLabel, '0', iTotal))
                if '0' in outcomeLabels and '1' not in outcomeLabels:
                    dsOutcomeLabels[dsLabel].append('1')
                    iTotal = col_labels.index('%s count total' % dsLabel)
                    impliedCounts1Q.append((dsLabel, '1', iTotal))

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

    def parse_tddatafile(self, filename, show_progress=True, record_zero_counts=True):
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

        display_progress = get_display_progress_fn(show_progress)

        with open(filename, 'r') as f:
            for (iLine, line) in enumerate(f):
                if iLine % nSkip == 0 or iLine + 1 == nLines: display_progress(iLine + 1, nLines, filename)

                line = line.strip()
                if len(line) == 0 or line[0] == '#': continue
                try:
                    parts = line.split()
                    lastpart = parts[-1]
                    circuitStr = line[:-len(lastpart)].strip()
                    circuitTuple, circuitLbls = self.parse_circuit(circuitStr, lookupDict)
                    # maybe allow a default line_labels to be passed in later?
                    if circuitLbls is None: circuitLbls = "auto"
                    circuit = _objs.Circuit(circuitTuple, stringrep=circuitStr, line_labels=circuitLbls, check=False)
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


def read_model(filename):
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
    basis = 'pp'  # default basis to load as

    basis_abbrev = "pp"  # default assumed basis
    basis_dim = None
    gaugegroup_name = None
    state_space_labels = None

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
                tpbs_lbls = []; tpbs_dims = []
                tensor_prod_blk_strs = line[len("STATESPACE:"):].split("+")
                for tpb_str in tensor_prod_blk_strs:
                    tpb_lbls = []; tpb_dims = []
                    for lbl_and_dim in tpb_str.split("*"):
                        start = lbl_and_dim.index('(')
                        end = lbl_and_dim.rindex(')')
                        lbl, dim = lbl_and_dim[:start], lbl_and_dim[start + 1:end]
                        tpb_lbls.append(lbl.strip())
                        tpb_dims.append(int(dim.strip()))
                    tpbs_lbls.append(tuple(tpb_lbls))
                    tpbs_dims.append(tuple(tpb_dims))
                state_space_labels = _objs.StateSpaceLabels(tpbs_lbls, tpbs_dims)

    if basis_dim is not None:
        # then specfy a dimensionful basis at the outset
        # basis_dims should be just a single int now that the *vector-space* dimension
        basis = _objs.BuiltinBasis(basis_abbrev, basis_dim)
    else:
        # otherwise we'll try to infer one from state space labels
        if state_space_labels is not None:
            basis = _objs.Basis.cast(basis_abbrev, state_space_labels.dim)
        else:
            raise ValueError("Cannot infer basis dimension!")

    if state_space_labels is None:
        assert(basis_dim is not None)  # b/c of logic above
        state_space_labels = _objs.StateSpaceLabels(['*'], [basis_dim])
        # special '*' state space label w/entire dimension inferred from BASIS line

    mdl = _objs.ExplicitOpModel(state_space_labels, basis)

    state = "look for label or property"
    cur_obj = None
    cur_group_obj = None
    cur_property = ""; cur_rows = []
    top_level_objs = []

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

                elif len(parts) == 2:  # then this is a '<type>: <label>' line => new cur_obj
                    typ = parts[0].strip()
                    label = parts[1].strip()

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
            mdl.preps[cur_label] = _objs.FullSPAMVec(
                get_liouville_mx(obj), typ="prep")
        elif cur_typ == "TP-PREP":
            mdl.preps[cur_label] = _objs.TPSPAMVec(
                get_liouville_mx(obj))
        elif cur_typ == "CPTP-PREP":
            props = obj['properties']
            assert("PureVec" in props and "ErrgenMx" in props)  # must always be Liouville reps!
            qty = _eval_row_list(props["ErrgenMx"], b_complex=False)
            nQubits = _np.log2(qty.size) / 2.0
            bQubits = bool(abs(nQubits - round(nQubits)) < 1e-10)  # integer # of qubits?
            proj_basis = "pp" if (basis == "pp" or bQubits) else basis
            errorMap = _objs.LindbladDenseOp.from_operation_matrix(
                qty, None, proj_basis, proj_basis, truncate=False, mx_basis=basis)  # unitary postfactor = Id
            pureVec = _objs.StaticSPAMVec(_np.transpose(_eval_row_list(props["PureVec"], b_complex=False)), typ="prep")
            mdl.preps[cur_label] = _objs.LindbladSPAMVec(pureVec, errorMap, "prep")
        elif cur_typ == "STATIC-PREP":
            mdl.preps[cur_label] = _objs.StaticSPAMVec(get_liouville_mx(obj), typ="prep")

        #POVMs
        elif cur_typ in ("POVM", "TP-POVM", "CPTP-POVM"):
            effects = []
            for sub_obj in obj['objects']:
                sub_typ = sub_obj['type']
                if sub_typ == "EFFECT":
                    Evec = _objs.FullSPAMVec(get_liouville_mx(sub_obj), typ="effect")
                elif sub_typ == "STATIC-EFFECT":
                    Evec = _objs.StaticSPAMVec(get_liouville_mx(sub_obj), typ="effect")
                #elif sub_typ == "CPTP-EFFECT":
                #    Evec = _objs.LindbladSPAMVec.from_spam_vector(qty,qty,"effect")
                effects.append((sub_obj['label'], Evec))

            if cur_typ == "POVM":
                mdl.povms[cur_label] = _objs.UnconstrainedPOVM(effects)
            elif cur_typ == "TP-POVM":
                assert(len(effects) > 1), "TP-POVMs must have at least 2 elements!"
                mdl.povms[cur_label] = _objs.TPPOVM(effects)
            elif cur_typ == "CPTP-POVM":
                props = obj['properties']
                assert("ErrgenMx" in props)  # and it must always be a Liouville rep!
                qty = _eval_row_list(props["ErrgenMx"], b_complex=False)
                nQubits = _np.log2(qty.size) / 2.0
                bQubits = bool(abs(nQubits - round(nQubits)) < 1e-10)  # integer # of qubits?
                proj_basis = "pp" if (basis == "pp" or bQubits) else basis
                errorMap = _objs.LindbladDenseOp.from_operation_matrix(
                    qty, None, proj_basis, proj_basis, truncate=False, mx_basis=basis)  # unitary postfactor = Id
                base_povm = _objs.UnconstrainedPOVM(effects)  # could try to detect a ComputationalBasisPOVM in FUTURE
                mdl.povms[cur_label] = _objs.LindbladPOVM(errorMap, base_povm)
            else: assert(False), "Logic error!"

        elif cur_typ == "GATE":
            mdl.operations[cur_label] = _objs.FullDenseOp(
                get_liouville_mx(obj))
        elif cur_typ == "TP-GATE":
            mdl.operations[cur_label] = _objs.TPDenseOp(
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
            mdl.operations[cur_label] = _objs.LindbladDenseOp.from_operation_matrix(
                qty, unitary_post, proj_basis, proj_basis, truncate=False, mx_basis=basis)

        elif cur_typ == "STATIC-GATE":
            mdl.operations[cur_label] = _objs.StaticDenseOp(get_liouville_mx(obj))

        elif cur_typ in ("Instrument", "TP-Instrument"):
            matrices = []
            for sub_obj in obj['objects']:
                sub_typ = sub_obj['type']
                qty = get_liouville_mx(sub_obj)
                mxOrOp = _objs.StaticDenseOp(qty) if cur_typ == "STATIC-IGATE" \
                    else qty  # just add numpy array `qty` to matrices list
                # and it will be made into a fully-param gate.
                matrices.append((sub_obj['label'], mxOrOp))

            if cur_typ == "Instrument":
                mdl.instruments[cur_label] = _objs.Instrument(matrices)
            elif cur_typ == "TP-Instrument":
                mdl.instruments[cur_label] = _objs.TPInstrument(matrices)
            else: assert(False), "Logic error!"
        else:
            raise ValueError("Unknown type: %s!" % cur_typ)

    #Add default gauge group -- the full group because
    # we add FullyParameterizedGates above.
    if gaugegroup_name == "Full":
        mdl.default_gauge_group = _objs.FullGaugeGroup(mdl.dim)
    elif gaugegroup_name == "TP":
        mdl.default_gauge_group = _objs.TPGaugeGroup(mdl.dim)
    elif gaugegroup_name == "Unitary":
        mdl.default_gauge_group = _objs.UnitaryGaugeGroup(mdl.dim, mdl.basis)
    else:
        mdl.default_gauge_group = None

    return mdl
