""" Text-parsing classes and functions to read input files."""
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

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

from ..baseobjs import CircuitParser as _CircuitParser


def get_display_progress_fn(showProgress):
    """
    Create and return a progress-displaying function if `showProgress == True`
    and it's run within an interactive environment.
    """
    
    def _is_interactive():
        import __main__ as main
        return not hasattr(main, '__file__')

    if _is_interactive() and showProgress:
        try:
            from IPython.display import clear_output
            def _display_progress(i,N,filename):
                _time.sleep(0.001); clear_output()
                print("Loading %s: %.0f%%" % (filename, 100.0*float(i)/float(N)))
                _sys.stdout.flush()
        except:
            def _display_progress(i,N,f): pass
    else:
        def _display_progress(i,N,f): pass
        
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

    def parse_circuit(self, s, lookup={}):
        """
        Parse a operation sequence (string in grammar)

        Parameters
        ----------
        s : string
            The string to parse.

        lookup : dict, optional
            A dictionary with keys == reflbls and values == tuples of operation labels
            which can be used for substitutions using the S<reflbl> syntax.

        Returns
        -------
        tuple of operation labels
            Representing the operation sequence.
        """
        self._circuit_parser.lookup = lookup
        circuit_tuple, circuit_labels = self._circuit_parser.parse(s)
        # print "DB: result = ",result
        # print "DB: stack = ",self.exprStack
        return circuit_tuple, circuit_labels

    def parse_dataline(self, s, lookup={}, expectedCounts=-1):
        """
        Parse a data line (dataline in grammar)

        Parameters
        ----------
        s : string
            The string to parse.

        lookup : dict, optional
            A dictionary with keys == reflbls and values == tuples of operation labels
            which can be used for substitutions using the S<reflbl> syntax.

        expectedCounts : int, optional
            The expected number of counts to accompany the operation sequence on this
            data line.  If < 0, no check is performed; otherwise raises ValueError
            if the number of counts does not equal expectedCounts.

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
        parts = s.split();
        counts = []
        for p in reversed(parts):
            if p == '--':
                counts.append('--') #special blank symbol
                continue
            try: #single float/int format
                f = float(p)
                counts.append(f)
            except:
                if 'G' in p: break # somewhat a hack - if there's a 'G' in it, then it's not a count column
                try: # "expanded" ColonContainingLabels:count
                    t = p.split(':')
                    assert(len(t) > 1)
                    f = float(t[-1])
                    counts.append( (tuple(t[0:-1]),f) )
                except:
                    break

        counts.reverse()  # because we appended them in reversed order
        totalCounts = len(counts)  # in case expectedCounts is less
        if len(counts) > expectedCounts >= 0:
            counts = counts[0:expectedCounts]

        nCounts = len(counts)
        if expectedCounts >= 0 and nCounts != expectedCounts:
            raise ValueError("Found %d count columns when %d were expected" % (nCounts, expectedCounts))
        if nCounts == len(parts):
            raise ValueError("No circuit column found -- all columns look like data")

        circuitStr = " ".join(parts[0:len(parts)-totalCounts])
        circuitTuple, circuitLabels = self.parse_circuit(circuitStr, lookup)
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
        circuitTuple,circuitLineLabels = self._circuit_parser.parse(circuitStr)
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
        circuit_list = [ ]
        with open(filename, 'r') as stringfile:
            for line in stringfile:
                line = line.strip()
                if len(line) == 0 or line[0] =='#': continue
                layer_lbls, line_lbls = self.parse_circuit(line)
                if line_lbls is None:
                    line_lbls = line_labels # default to the passed-in argument
                    nlines = num_lines
                else: nlines = None # b/c we've got a valid line_lbls
                
                circuit_list.append( _objs.Circuit(layer_lbls, stringrep=line.strip(),
                                                   line_labels=line_lbls, num_lines=nlines, check=False) )
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
        lookupDict = { }
        with open(filename, 'r') as dictfile:
            for line in dictfile:
                line = line.strip()
                if len(line) == 0 or line[0] =='#': continue
                label, tup, s, lineLbls = self.parse_dictline(line)
                if lineLbls is None: lineLbls = "auto"
                lookupDict[ label ] = _objs.Circuit(tup, stringrep=s, line_labels=lineLbls, check=False)
        return lookupDict

    def parse_datafile(self, filename, showProgress=True,
                       collisionAction="aggregate"):
        """
        Parse a data set file into a DataSet object.

        Parameters
        ----------
        filename : string
            The file to parse.

        showProgress : bool, optional
            Whether or not progress should be displayed

        collisionAction : {"aggregate", "keepseparate"}
            Specifies how duplicate operation sequences should be handled.  "aggregate"
            adds duplicate-sequence counts, whereas "keepseparate" tags duplicate-
            sequence data with by appending a final "#<number>" operation label to the
            duplicated gate sequence.

        Returns
        -------
        DataSet
            A static DataSet object.
        """

        #Parse preamble -- lines beginning with # or ## until first non-# line
        preamble_directives = { }
        preamble_comments = []
        with open(filename, 'r') as datafile:
            for line in datafile:
                line = line.strip()
                if len(line) == 0 or line[0] != '#': break
                if line.startswith("## "):
                    parts = line[len("## "):].split("=")
                    if len(parts) == 2: # key = value
                        preamble_directives[ parts[0].strip() ] = parts[1].strip()
                elif line.startswith("#"):
                    preamble_comments.append(line[1:].strip())

        #Process premble
        orig_cwd = _os.getcwd()
        if len(_os.path.dirname(filename)) > 0: _os.chdir( _os.path.dirname(filename) ) #allow paths relative to datafile path
        try:
            if 'Lookup' in preamble_directives:
                lookupDict = self.parse_dictfile( preamble_directives['Lookup'] )
            else: lookupDict = { }
            if 'Columns' in preamble_directives:
                colLabels = [ l.strip() for l in preamble_directives['Columns'].split(",") ]
                outcomeLabels,fillInfo = self._extractLabelsFromColLabels(colLabels)
                nDataCols = len(colLabels)
            else:
                outcomeLabels = fillInfo = None
                nDataCols = -1 # no column count check

            # "default" case when we have no columns and no "expanded-form" counts
            default_colLabels = [ '1 count', 'count total' ] #  outcomeLabel (' frequency' | ' count') | 'count total'
            _,default_fillInfo = self._extractLabelsFromColLabels(default_colLabels)


        finally:
            _os.chdir(orig_cwd)

        #Read data lines of data file
        dataset = _objs.DataSet(outcomeLabels=outcomeLabels,collisionAction=collisionAction,
                                comment="\n".join(preamble_comments))
        nLines  = 0
        with open(filename, 'r') as datafile:
            nLines = sum(1 for line in datafile)
        nSkip = int(nLines / 100.0)
        if nSkip == 0: nSkip = 1

        display_progress = get_display_progress_fn(showProgress)
        warnings = [] # to display *after* display progress

        with open(filename, 'r') as inputfile:
            for (iLine,line) in enumerate(inputfile):
                if iLine % nSkip == 0 or iLine+1 == nLines: display_progress(iLine+1, nLines, filename)

                line = line.strip()
                if '#' in line:
                    i = line.index('#')
                    dataline,comment = line[:i], line[i+1:]
                else:
                    dataline,comment = line, ""

                if len(dataline) == 0: continue
                try:
                    circuitTuple, circuitStr, circuitLbls, valueList = \
                            self.parse_dataline(dataline, lookupDict, nDataCols)

                    commentDict = {}
                    comment = comment.strip()
                    if len(comment) > 0:
                        try:
                            if comment.startswith("{") and comment.endswith("}"):
                                commentDict = _ast.literal_eval(comment)
                            else: # put brackets around it
                                commentDict = _ast.literal_eval("{ " + comment + " }")
                            #commentDict = _json.loads("{ " + comment + " }")
                              #Alt: safer(?) & faster, but need quotes around all keys & vals
                        except:
                            warnings.append("%s Line %d: Could not parse comment '%s'"
                                            % (filename, iLine, comment))

                except ValueError as e:
                    raise ValueError("%s Line %d: %s" % (filename, iLine, str(e)))

                if (len(dataset) == 0) and (fillInfo is None) and \
                   (len(valueList) > 0) and (not isinstance(valueList[0],tuple)):
                    #In order to preserve backward compatibility, if the first
                    # data-line is not in expanded form and there was no column
                    # header, then use "default" column label info.
                    fillInfo = default_fillInfo

                countDict = _OrderedDict()
                self._fillDataCountDict( countDict, fillInfo, valueList )
                if all([ (abs(v) < 1e-9) for v in list(countDict.values())]):
                    warnings.append("Dataline for circuit '%s' has zero counts and will be ignored" % circuitStr)
                    continue #skip lines in dataset file with zero counts (no experiments done)
                if circuitLbls is None: circuitLbls = "auto" # if line labels weren't given just use defaults
                circuit = _objs.Circuit(circuitTuple, stringrep=circuitStr, line_labels=circuitLbls, check=False) #, lookup=lookupDict)
                dataset.add_count_dict(circuit, countDict, aux=commentDict)

        if warnings:
            _warnings.warn('\n'.join(warnings)) # to be displayed at end, after potential progress updates

        dataset.done_adding_data()
        return dataset

    def _extractLabelsFromColLabels(self, colLabels ):
        outcomeLabels = []; countCols = []; freqCols = []; impliedCountTotCol1Q = (-1,-1)

        def str_to_outcome(x): #always return a tuple as the "outcome label" (even if length 1)
            return tuple(x.strip().split(":"))
        
        for i,colLabel in enumerate(colLabels):
            if colLabel.endswith(' count'):
                outcomeLabel = str_to_outcome(colLabel[:-len(' count')])
                if outcomeLabel not in outcomeLabels: outcomeLabels.append( outcomeLabel )
                countCols.append( (outcomeLabel,i) )

            elif colLabel.endswith(' frequency'):
                if 'count total' not in colLabels:
                    raise ValueError("Frequency columns specified without count total")
                else: iTotal = colLabels.index( 'count total' )
                outcomeLabel = str_to_outcome(colLabel[:-len(' frequency')])
                if outcomeLabel not in outcomeLabels: outcomeLabels.append( outcomeLabel )
                freqCols.append( (outcomeLabel,i,iTotal) )

        if 'count total' in colLabels:
            if ('1',) in outcomeLabels and ('0',) not in outcomeLabels:
                outcomeLabels.append( ('0',) )
                impliedCountTotCol1Q = ('0',), colLabels.index( 'count total' )
            elif ('0',) in outcomeLabels and ('1',) not in outcomeLabels:
                outcomeLabels.append( ('1',) )
                impliedCountTotCol1Q = '1', colLabels.index( 'count total' )
            #TODO - add standard count completion for 2Qubit case?

        fillInfo = (countCols, freqCols, impliedCountTotCol1Q)        
        return outcomeLabels, fillInfo


    def _fillDataCountDict(self, countDict, fillInfo, colValues):
        if fillInfo is not None:
            countCols, freqCols, impliedCountTotCol1Q = fillInfo

            for outcomeLabel,iCol in countCols:
                if colValues[iCol] == '--': continue #skip blank sentinels
                if colValues[iCol] > 0 and colValues[iCol] < 1:
                    _warnings.warn("Count column (%d) contains value(s) " % iCol +
                                     "between 0 and 1 - could this be a frequency?")
                assert(not isinstance(colValues[iCol],tuple)), \
                    "Expanded-format count not allowed with column-key header"
                countDict[outcomeLabel] = colValues[iCol]
    
            for outcomeLabel,iCol,iTotCol in freqCols:
                if colValues[iCol] == '--' or colValues[iTotCol] == '--': continue #skip blank sentinels
                if colValues[iCol] < 0 or colValues[iCol] > 1.0:
                    _warnings.warn("Frequency column (%d) contains value(s) " % iCol +
                                     "outside of [0,1.0] interval - could this be a count?")
                assert(not isinstance(colValues[iTotCol],tuple)), \
                    "Expanded-format count not allowed with column-key header"
                countDict[outcomeLabel] = colValues[iCol] * colValues[iTotCol]
    
            if impliedCountTotCol1Q[1] >= 0:
                impliedOutcomeLabel, impliedCountTotCol = impliedCountTotCol1Q
                if impliedOutcomeLabel == ('0',):
                    countDict[('0',)] = colValues[impliedCountTotCol] - countDict[('1',)]
                else:
                    countDict[('1',)] = colValues[impliedCountTotCol] - countDict[('0',)]

        else: #assume colValues is a list of (outcomeLabel, count) tuples
            for tup in colValues:
                assert(isinstance(tup,tuple)), \
                    ("Outcome labels must be specified with"
                     "count data when there's no column-key header")
                assert(len(tup) == 2),"Invalid count! (parsed to %s)" % str(tup)
                countDict[ tup[0] ] = tup[1]
        return countDict


    def parse_multidatafile(self, filename, showProgress=True,
                            collisionAction="aggregate"):
        """
        Parse a multiple data set file into a MultiDataSet object.

        Parameters
        ----------
        filename : string
            The file to parse.

        showProgress : bool, optional
            Whether or not progress should be displayed

        collisionAction : {"aggregate", "keepseparate"}
            Specifies how duplicate operation sequences should be handled.  "aggregate"
            adds duplicate-sequence counts, whereas "keepseparate" tags duplicate-
            sequence data with by appending a final "#<number>" operation label to the
            duplicated gate sequence.

        Returns
        -------
        MultiDataSet
            A MultiDataSet object.
        """

        #Parse preamble -- lines beginning with # or ## until first non-# line
        preamble_directives = { }
        preamble_comments = []
        with open(filename, 'r') as multidatafile:
            for line in multidatafile:
                line = line.strip()
                if len(line) == 0 or line[0] != '#': break
                if line.startswith("## "):
                    parts = line[len("## "):].split("=")
                    if len(parts) == 2: # key = value
                        preamble_directives[ parts[0].strip() ] = parts[1].strip()
                elif line.startswith("#"):
                    preamble_comments.append(line[1:].strip())


        #Process premble
        orig_cwd = _os.getcwd()
        if len(_os.path.dirname(filename)) > 0:
            _os.chdir( _os.path.dirname(filename) ) #allow paths relative to datafile path
        try:
            if 'Lookup' in preamble_directives:
                lookupDict = self.parse_dictfile( preamble_directives['Lookup'] )
            else: lookupDict = { }
            if 'Columns' in preamble_directives:
                colLabels = [ l.strip() for l in preamble_directives['Columns'].split(",") ]
            else: colLabels = [ 'dataset1 1 count', 'dataset1 count total' ]
            dsOutcomeLabels, fillInfo = self._extractLabelsFromMultiDataColLabels(colLabels)
            nDataCols = len(colLabels)
        finally:
            _os.chdir(orig_cwd)

        #Read data lines of data file
        datasets = _OrderedDict()
        for dsLabel,outcomeLabels in dsOutcomeLabels.items():
            datasets[dsLabel] = _objs.DataSet(outcomeLabels=outcomeLabels,
                                              collisionAction=collisionAction)

        dsCountDicts = _OrderedDict()
        for dsLabel in dsOutcomeLabels: dsCountDicts[dsLabel] = {}

        nLines = 0
        with open(filename, 'r') as datafile:
            nLines = sum(1 for line in datafile)
        nSkip = max(int(nLines / 100.0),1)

        display_progress = get_display_progress_fn(showProgress)

        with open(filename, 'r') as inputfile:
            for (iLine,line) in enumerate(inputfile):
                if iLine % nSkip == 0 or iLine+1 == nLines: display_progress(iLine+1, nLines, filename)

                line = line.strip()
                if len(line) == 0 or line[0] == '#': continue
                try:
                    circuitTuple, circuitStr, circuitLbls, valueList = \
                        self.parse_dataline(line, lookupDict, nDataCols)
                except ValueError as e:
                    raise ValueError("%s Line %d: %s" % (filename, iLine, str(e)))

                if circuitLbls is None: circuitLbls = "auto" # if line labels aren't given find them automatically
                opStr = _objs.Circuit(circuitTuple, stringrep=circuitStr, line_labels=circuitLbls, check=False) #, lookup=lookupDict)
                self._fillMultiDataCountDicts(dsCountDicts, fillInfo, valueList)
                for dsLabel, countDict in dsCountDicts.items():                    
                    datasets[dsLabel].add_count_dict(opStr, countDict)

        mds = _objs.MultiDataSet(comment="\n".join(preamble_comments))
        for dsLabel,ds in datasets.items():
            ds.done_adding_data()
            mds.add_dataset(dsLabel, ds)
        return mds


    #Note: outcome labels must not contain spaces since we use spaces to separate
    # the outcome label from the dataset label
    def _extractLabelsFromMultiDataColLabels(self, colLabels):
        dsOutcomeLabels = _OrderedDict()
        countCols = []; freqCols = []; impliedCounts1Q = []
        for i,colLabel in enumerate(colLabels):
            wordsInColLabel = colLabel.split() #split on whitespace into words
            if len(wordsInColLabel) < 3: continue #allow other columns we don't recognize

            if wordsInColLabel[-1] == 'count':
                outcomeLabel = wordsInColLabel[-2]
                dsLabel = wordsInColLabel[-3]
                if dsLabel not in dsOutcomeLabels:
                    dsOutcomeLabels[dsLabel] = [ outcomeLabel ]
                else: dsOutcomeLabels[dsLabel].append( outcomeLabel )
                countCols.append( (dsLabel,outcomeLabel,i) )

            elif wordsInColLabel[-1] == 'frequency':
                outcomeLabel = wordsInColLabel[-2]
                dsLabel = wordsInColLabel[-3]
                if '%s count total' % dsLabel not in colLabels:
                    raise ValueError("Frequency columns specified without" +
                                     "count total for dataset '%s'" % dsLabel)
                else: iTotal = colLabels.index( '%s count total' % dsLabel )

                if dsLabel not in dsOutcomeLabels:
                    dsOutcomeLabels[dsLabel] = [ outcomeLabel ]
                else: dsOutcomeLabels[dsLabel].append( outcomeLabel )
                freqCols.append( (dsLabel,outcomeLabel,i,iTotal) )

        for dsLabel,outcomeLabels in dsOutcomeLabels.items():
            if '%s count total' % dsLabel in colLabels:
                if '1' in outcomeLabels and '0' not in outcomeLabels:
                    dsOutcomeLabels[dsLabel].append('0')
                    iTotal = colLabels.index( '%s count total' % dsLabel )
                    impliedCounts1Q.append( (dsLabel, '0', iTotal) )
                if '0' in outcomeLabels and '1' not in outcomeLabels:
                    dsOutcomeLabels[dsLabel].append('1')
                    iTotal = colLabels.index( '%s count total' % dsLabel )
                    impliedCounts1Q.append( (dsLabel, '1', iTotal) )

            #TODO - add standard count completion for 2Qubit case?

        fillInfo = (countCols, freqCols, impliedCounts1Q)
        return dsOutcomeLabels, fillInfo


    def _fillMultiDataCountDicts(self, countDicts, fillInfo, colValues):
        countCols, freqCols, impliedCounts1Q = fillInfo

        for dsLabel,outcomeLabel,iCol in countCols:
            if colValues[iCol] == '--': continue
            if colValues[iCol] > 0 and colValues[iCol] < 1:
                raise ValueError("Count column (%d) contains value(s) " % iCol +
                                 "between 0 and 1 - could this be a frequency?")
            countDicts[dsLabel][outcomeLabel] = colValues[iCol]

        for dsLabel,outcomeLabel,iCol,iTotCol in freqCols:
            if colValues[iCol] == '--': continue
            if colValues[iCol] < 0 or colValues[iCol] > 1.0:
                raise ValueError("Frequency column (%d) contains value(s) " % iCol +
                                 "outside of [0,1.0] interval - could this be a count?")
            countDicts[dsLabel][outcomeLabel] = colValues[iCol] * colValues[iTotCol]

        for dsLabel,outcomeLabel,iTotCol in impliedCounts1Q:
            if colValues[iTotCol] == '--': raise ValueError("Mising total (== '--')!")
            if outcomeLabel == '0':
                countDicts[dsLabel]['0'] = colValues[iTotCol] - countDicts[dsLabel]['1']
            elif outcomeLabel == '1':
                countDicts[dsLabel]['1'] = colValues[iTotCol] - countDicts[dsLabel]['0']

        #TODO - add standard count completion for 2Qubit case?
        return countDicts


    def parse_tddatafile(self, filename, showProgress=True):
        """ 
        Parse a data set file into a TDDataSet object.

        Parameters
        ----------
        filename : string
            The file to parse.

        showProgress : bool, optional
            Whether or not progress should be displayed

        Returns
        -------
        TDDataSet
            A static TDDataSet object.
        """

        #Parse preamble -- lines beginning with # or ## until first non-# line
        preamble_directives = _OrderedDict()
        with open(filename,'r') as f:
            for line in f:
                line = line.strip()
                if len(line) == 0 or line[0] != '#': break
                if line.startswith("## "):
                    parts = line[len("## "):].split("=")
                    if len(parts) == 2: # key = value
                        preamble_directives[ parts[0].strip() ] = parts[1].strip()
        
        #Process premble
        orig_cwd = _os.getcwd()
        if len(_os.path.dirname(filename)) > 0: _os.chdir( _os.path.dirname(filename) ) #allow paths relative to datafile path
        try:
            if 'Lookup' in preamble_directives: 
                lookupDict = self.parse_dictfile( preamble_directives['Lookup'] )
            else: lookupDict = { }
        finally:
            _os.chdir(orig_cwd)

        outcomeLabelAbbrevs = _OrderedDict()
        for key,val in preamble_directives.items():
            if key == "Lookup": continue 
            outcomeLabelAbbrevs[key] = val
        outcomeLabels = outcomeLabelAbbrevs.values()

        #Read data lines of data file
        dataset = _objs.DataSet(outcomeLabels=outcomeLabels)
        with open(filename,'r') as f:
            nLines = sum(1 for line in f)
        nSkip = int(nLines / 100.0)
        if nSkip == 0: nSkip = 1

        display_progress = get_display_progress_fn(showProgress)

        with open(filename,'r') as f:
            for (iLine,line) in enumerate(f):
                if iLine % nSkip == 0 or iLine+1 == nLines: display_progress(iLine+1, nLines, filename)
    
                line = line.strip()
                if len(line) == 0 or line[0] == '#': continue
                try:
                    parts = line.split()
                    lastpart = parts[-1]
                    circuitStr = line[:-len(lastpart)].strip()
                    circuitTuple, circuitLbls = self.parse_circuit(circuitStr, lookupDict)
                    if circuitLbls is None: circuitLbls = "auto" # maybe allow a default line_labels to be passed in later?
                    circuit = _objs.Circuit(circuitTuple, stringrep=circuitStr, line_labels=circuitLbls, check=False)
                    timeSeriesStr = lastpart.strip()
                except ValueError as e:
                    raise ValueError("%s Line %d: %s" % (filename, iLine, str(e)))
    
                seriesList = [ outcomeLabelAbbrevs[abbrev] for abbrev in timeSeriesStr ] #iter over characters in str
                timesList = list(range(len(seriesList))) #FUTURE: specify an offset and step??
                dataset.add_raw_series_data(circuit, seriesList, timesList)
                
        dataset.done_adding_data()
        return dataset



def _evalElement(el, bComplex):
    myLocal = { 'pi': _np.pi, 'sqrt': _np.sqrt }
    exec( "element = %s" % el, {"__builtins__": None}, myLocal )
    return complex( myLocal['element'] ) if bComplex else float( myLocal['element'] )

def _evalRowList(rows, bComplex):
    return _np.array( [ [ _evalElement(x,bComplex) for x in r ] for r in rows ],
                     'complex' if bComplex else 'd' )

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
    basis = 'pp' #default basis to load as

    spam_vecs = _OrderedDict();
    spam_labels = _OrderedDict(); remainder_spam_label = ""
    identity_vec = _np.transpose( _np.array( [ _np.sqrt(2.0), 0,0,0] ) )  #default = 1-QUBIT identity vector

    basis_abbrev = "pp" #default assumed basis
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
                if gaugegroup_name not in ("Full","TP","Unitary"):
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
                        lbl,dim = lbl_and_dim[:start], lbl_and_dim[start+1:end]
                        tpb_lbls.append(lbl.strip())
                        tpb_dims.append(int(dim.strip()))
                    tpbs_lbls.append( tuple(tpb_lbls) )
                    tpbs_dims.append( tuple(tpb_dims) )
                state_space_labels = _objs.StateSpaceLabels(tpbs_lbls, tpbs_dims)

    if basis_dim is not None:
        # then specfy a dimensionful basis at the outset
        basis = _objs.BuiltinBasis(basis_abbrev, basis_dim) # basis_dims should be just a single int now that the *vector-space* dimension
    else:
        # otherwise we'll try to infer one from state space labels
        if state_space_labels is not None:
            basis = _objs.Basis.cast(basis_abbrev, state_space_labels.dim)
        else:
            raise ValueError("Cannot infer basis dimension!")

    if state_space_labels is None:
        assert(basis_dim is not None) # b/c of logic above
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
                        cur_group_obj['objects'].append( cur_obj ); cur_obj = None
                    top_level_objs.append(cur_group_obj); cur_group_obj = None

            elif line[0] == "#":
                pass # skip comments

            elif state == "look for label or property":
                assert(cur_property == ""), "Logic error!"
                
                parts = line.split(':')
                if any([line.startswith(pre) for pre in ("BASIS","GAUGEGROUP","STATESPACE")]):
                    pass #handled above

                elif len(parts) == 2: # then this is a '<type>: <label>' line => new cur_obj
                    typ = parts[0].strip()
                    label = parts[1].strip()

                    # place any existing cur_obj
                    if cur_obj is not None:
                        if cur_group_obj is not None:
                            cur_group_obj['objects'].append( cur_obj )
                        else:
                            top_level_objs.append(cur_obj)
                        cur_obj = None
                    
                    if typ in ("POVM","TP-POVM","CPTP-POVM","Instrument","TP-Instrument"):
                        # a group type - so create a new *group* object
                        assert(cur_group_obj is None), "Group label encountered before ENDing prior group:\n%s" % line
                        cur_group_obj = {'label': label, 'type': typ, 'properties': {}, 'objects': [] }
                    else:
                        #All other "types" are object labels
                        cur_obj = {'label': label, 'type': typ, 'properties': {} }
                        
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
                cur_rows.append( line.split() )


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


    def get_liouville_mx(obj,prefix=""):
        """ Process properties of `obj` to extract a single liouville representation """
        props = obj['properties']; lmx = None
        if prefix+"StateVec" in props:
            ar = _evalRowList( props[prefix+"StateVec"], bComplex=True )
            if ar.shape == (1,2):
                stdmx = _tools.state_to_stdmx(ar[0,:])
                lmx = _tools.stdmx_to_vec(stdmx, basis)
            else: raise ValueError("Invalid state vector shape for %s: %s" % (cur_label,ar.shape))

        elif prefix+"DensityMx" in props:
            ar = _evalRowList( props[prefix+"DensityMx"], bComplex=True )
            if ar.shape == (2,2) or ar.shape == (4,4):
                lmx = _tools.stdmx_to_vec(ar, basis)
            else: raise ValueError("Invalid density matrix shape for %s: %s" % (cur_label,ar.shape))

        elif prefix+"LiouvilleVec" in props:
            lmx = _np.transpose( _evalRowList( props[prefix+"LiouvilleVec"], bComplex=False ) )

        elif prefix+"UnitaryMx" in props:
            ar = _evalRowList( props[prefix+"UnitaryMx"], bComplex=True )
            lmx = _tools.change_basis(_tools.unitary_to_process_mx(ar), 'std', basis)

        elif prefix+"UnitaryMxExp" in props:
            ar = _evalRowList( props[prefix+"UnitaryMxExp"], bComplex=True )
            lmx = _tools.change_basis(_tools.unitary_to_process_mx(_expm(-1j*ar)), 'std', basis)

        elif prefix+"LiouvilleMx" in props:
            lmx = _evalRowList( props[prefix+"LiouvilleMx"], bComplex=False )
            
        if lmx is None:
            raise ValueError("No valid format found in %s" % str(list(props.keys())))
        
        return lmx

    
    #Now process top_level_objs to create a Model
    for obj in top_level_objs: # `obj` is a dict of object info
        cur_typ = obj['type']
        cur_label = obj['label']

        #Preps
        if cur_typ == "PREP":
            mdl.preps[cur_label] = _objs.FullSPAMVec(
                get_liouville_mx(obj))
        elif cur_typ == "TP-PREP":
            mdl.preps[cur_label] = _objs.TPSPAMVec(
                get_liouville_mx(obj))
        elif cur_typ == "CPTP-PREP":
            props = obj['properties']
            assert("PureVec" in props and "ErrgenMx" in props) # must always be Liouville reps!
            qty = _evalRowList( props["ErrgenMx"], bComplex=False )
            nQubits = _np.log2(qty.size)/2.0
            bQubits = bool(abs(nQubits-round(nQubits)) < 1e-10) #integer # of qubits?
            proj_basis = "pp" if (basis == "pp" or bQubits) else basis
            errorMap = _objs.LindbladDenseOp.from_operation_matrix(
                qty, None, proj_basis, proj_basis, truncate=False, mxBasis=basis) #unitary postfactor = Id
            pureVec = _objs.StaticSPAMVec( _np.transpose(_evalRowList( props["PureVec"], bComplex=False )))
            mdl.preps[cur_label] = _objs.LindbladSPAMVec(pureVec,errorMap,"prep")
        elif cur_typ == "STATIC-PREP":
            mdl.preps[cur_label] = _objs.StaticSPAMVec(get_liouville_mx(obj))

        #POVMs
        elif cur_typ in ("POVM","TP-POVM","CPTP-POVM"):
            effects = []
            for sub_obj in obj['objects']:
                sub_typ = sub_obj['type']
                if sub_typ == "EFFECT":
                    Evec = _objs.FullSPAMVec(get_liouville_mx(sub_obj))
                elif sub_typ == "TP-EFFECT":
                    Evec = _objs.TPSPAMVec(get_liouville_mx(sub_obj))
                elif sub_typ == "STATIC-EFFECT":
                    Evec = _objs.StaticSPAMVec(get_liouville_mx(sub_obj))
                #elif sub_typ == "CPTP-EFFECT":
                #    Evec = _objs.LindbladSPAMVec.from_spam_vector(qty,qty,"effect")
                effects.append( (sub_obj['label'],Evec) )

            if cur_typ == "POVM":
                mdl.povms[cur_label] = _objs.UnconstrainedPOVM( effects )
            elif cur_typ == "TP-POVM":
                assert(len(effects) > 1), "TP-POVMs must have at least 2 elements!"
                mdl.povms[cur_label] = _objs.TPPOVM( effects )
            elif cur_typ == "CPTP-POVM":
                props = obj['properties']
                assert("ErrgenMx" in props) # and it must always be a Liouville rep!
                qty = _evalRowList( props["ErrgenMx"], bComplex=False )
                nQubits = _np.log2(qty.size)/2.0
                bQubits = bool(abs(nQubits-round(nQubits)) < 1e-10) #integer # of qubits?
                proj_basis = "pp" if (basis == "pp" or bQubits) else basis
                errorMap = _objs.LindbladDenseOp.from_operation_matrix(
                    qty, None, proj_basis, proj_basis, truncate=False, mxBasis=basis) #unitary postfactor = Id
                base_povm = _objs.UnconstrainedPOVM(effects) # could try to detect a ComputationalBasisPOVM in FUTURE
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
                unitary_post = get_liouville_mx(obj,"Ref")
            except ValueError:
                unitary_post = None
            nQubits = _np.log2(qty.shape[0])/2.0
            bQubits = bool(abs(nQubits-round(nQubits)) < 1e-10) #integer # of qubits?
            proj_basis = "pp" if (basis == "pp" or bQubits) else basis
            mdl.operations[cur_label] = _objs.LindbladDenseOp.from_operation_matrix(
                qty, unitary_post, proj_basis, proj_basis, truncate=False, mxBasis=basis)

        elif cur_typ == "STATIC-GATE":
            mdl.operations[cur_label] = _objs.StaticDenseOp(get_liouville_mx(obj))


        elif cur_typ in ("Instrument","TP-Instrument"):
            matrices = []
            for sub_obj in obj['objects']:
                sub_typ = sub_obj['type']
                qty = get_liouville_mx(sub_obj)
                mxOrOp = _objs.StaticDenseOp(qty) if cur_typ == "STATIC-IGATE" \
                       else qty #just add numpy array `qty` to matrices list
                                # and it will be made into a fully-param gate.
                matrices.append( (sub_obj['label'],mxOrOp) )

            if cur_typ == "Instrument":
                mdl.instruments[cur_label] = _objs.Instrument( matrices )
            elif cur_typ == "TP-Instrument":
                mdl.instruments[cur_label] = _objs.TPInstrument( matrices )
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
