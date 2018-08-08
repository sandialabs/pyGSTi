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

from ..baseobjs import GateStringParser as _GateStringParser


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
    _string_parser = _GateStringParser()

    def __init__(self):
        """ Create a new standard-input parser object """
        pass

    def parse_gatestring(self, s, lookup={}):
        """
        Parse a gate string (string in grammar)

        Parameters
        ----------
        s : string
            The string to parse.

        lookup : dict, optional
            A dictionary with keys == reflbls and values == tuples of gate labels
            which can be used for substitutions using the S<reflbl> syntax.

        Returns
        -------
        tuple of gate labels
            Representing the gate string.
        """
        self._string_parser.lookup = lookup
        gate_tuple = self._string_parser.parse(s)
        # print "DB: result = ",result
        # print "DB: stack = ",self.exprStack
        return gate_tuple

    def parse_dataline(self, s, lookup={}, expectedCounts=-1):
        """
        Parse a data line (dataline in grammar)

        Parameters
        ----------
        s : string
            The string to parse.

        lookup : dict, optional
            A dictionary with keys == reflbls and values == tuples of gate labels
            which can be used for substitutions using the S<reflbl> syntax.

        expectedCounts : int, optional
            The expected number of counts to accompany the gate string on this
            data line.  If < 0, no check is performed; otherwise raises ValueError
            if the number of counts does not equal expectedCounts.

        Returns
        -------
        gateStringTuple : tuple
            The gate string as a tuple of gate labels.
        gateStringStr : string
            The gate string as represented as a string in the dataline
        counts : list
            List of counts following the gate string.
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
            raise ValueError("No gatestring column found -- all columns look like data")

        gateStringStr = " ".join(parts[0:len(parts)-totalCounts])
        gateStringTuple = self.parse_gatestring(gateStringStr, lookup)
        return gateStringTuple, gateStringStr, counts

    def parse_dictline(self, s):
        """
        Parse a gatestring dictionary line (dictline in grammar)

        Parameters
        ----------
        s : string
            The string to parse.

        Returns
        -------
        gateStringLabel : string
            The user-defined label to represent this gate string.
        gateStringTuple : tuple
            The gate string as a tuple of gate labels.
        gateStringStr : string
            The gate string as represented as a string in the dictline.
        """
        label = r'\s*([a-zA-Z0-9_]+)\s+'
        match = _re.match(label, s)
        if not match:
            raise ValueError("'{}' is not a valid dictline".format(s))
        gateStringLabel = match.group(1)
        gateStringStr = s[match.end():]
        gateStringTuple = self._string_parser.parse(gateStringStr)
        return gateStringLabel, gateStringTuple, gateStringStr

    def parse_stringfile(self, filename):
        """
        Parse a gatestring list file.

        Parameters
        ----------
        filename : string
            The file to parse.

        Returns
        -------
        list of GateStrings
            The gatestrings read from the file.
        """
        gatestring_list = [ ]
        with open(filename, 'r') as stringfile:
            for line in stringfile:
                line = line.strip()
                if len(line) == 0 or line[0] =='#': continue
                gatestring_list.append( _objs.GateString(self.parse_gatestring(line), line) )
        return gatestring_list

    def parse_dictfile(self, filename):
        """
        Parse a gatestring dictionary file.

        Parameters
        ----------
        filename : string
            The file to parse.

        Returns
        -------
        dict
           Dictionary with keys == gate string labels and values == GateStrings.
        """
        lookupDict = { }
        with open(filename, 'r') as dictfile:
            for line in dictfile:
                line = line.strip()
                if len(line) == 0 or line[0] =='#': continue
                label, tup, s = self.parse_dictline(line)
                lookupDict[ label ] = _objs.GateString(tup, s)
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
            Specifies how duplicate gate sequences should be handled.  "aggregate"
            adds duplicate-sequence counts, whereas "keepseparate" tags duplicate-
            sequence data with by appending a final "#<number>" gate label to the
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
                    gateStringTuple, gateStringStr, valueList = \
                            self.parse_dataline(dataline, lookupDict, nDataCols)

                    commentDict = {}
                    if len(comment) > 0:
                        try:
                            commentDict = _ast.literal_eval("{ " + comment + " }")
                            #commentDict = _json.loads("{ " + comment + " }")
                              #Alt: safer(?) & faster, but need quotes around all keys & vals
                        except:
                            _warnings.warn("%s Line %d: Could not parse comment '%s'"
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
                    _warnings.warn( "Dataline for gateString '%s' has zero counts and will be ignored" % gateStringStr)
                    continue #skip lines in dataset file with zero counts (no experiments done)
                gateStr = _objs.GateString(gateStringTuple, gateStringStr, lookup=lookupDict)
                dataset.add_count_dict(gateStr, countDict, aux=commentDict)

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
            Specifies how duplicate gate sequences should be handled.  "aggregate"
            adds duplicate-sequence counts, whereas "keepseparate" tags duplicate-
            sequence data with by appending a final "#<number>" gate label to the
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
                    gateStringTuple, gateStringStr, valueList = self.parse_dataline(line, lookupDict, nDataCols)
                except ValueError as e:
                    raise ValueError("%s Line %d: %s" % (filename, iLine, str(e)))

                gateStr = _objs.GateString(gateStringTuple, gateStringStr, lookup=lookupDict)
                self._fillMultiDataCountDicts(dsCountDicts, fillInfo, valueList)
                for dsLabel, countDict in dsCountDicts.items():                    
                    datasets[dsLabel].add_count_dict(gateStr, countDict)

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
                    gateStringStr = line[:-len(lastpart)].strip()
                    gateStringTuple = self.parse_gatestring(gateStringStr, lookupDict)
                    gateString = _objs.GateString(gateStringTuple, gateStringStr)
                    timeSeriesStr = lastpart.strip()
                except ValueError as e:
                    raise ValueError("%s Line %d: %s" % (filename, iLine, str(e)))
    
                seriesList = [ outcomeLabelAbbrevs[abbrev] for abbrev in timeSeriesStr ] #iter over characters in str
                timesList = list(range(len(seriesList))) #FUTURE: specify an offset and step??
                dataset.add_raw_series_data(gateString, seriesList, timesList)
                
        dataset.done_adding_data()
        return dataset



def _evalElement(el, bComplex):
    myLocal = { 'pi': _np.pi, 'sqrt': _np.sqrt }
    exec( "element = %s" % el, {"__builtins__": None}, myLocal )
    return complex( myLocal['element'] ) if bComplex else float( myLocal['element'] )

def _evalRowList(rows, bComplex):
    return _np.array( [ [ _evalElement(x,bComplex) for x in r ] for r in rows ],
                     'complex' if bComplex else 'd' )

def read_gateset(filename):
    """
    Parse a gateset file into a GateSet object.

    Parameters
    ----------
    filename : string
        The file to parse.

    Returns
    -------
    GateSet
    """
    basis = 'pp' #default basis to load as
    
    def add_current():
        """ Adds the current object, described by lots of cur_* variables """

        qty = None
        if cur_format == "StateVec":
            ar = _evalRowList( cur_rows, bComplex=True )
            if ar.shape == (1,2):
                stdmx = _tools.state_to_stdmx(ar[0,:])
                qty = _tools.stdmx_to_vec(stdmx, basis)
            else: raise ValueError("Invalid state vector shape for %s: %s" % (cur_label,ar.shape))

        elif cur_format == "DensityMx":
            ar = _evalRowList( cur_rows, bComplex=True )
            if ar.shape == (2,2) or ar.shape == (4,4):
                qty = _tools.stdmx_to_vec(ar, basis)
            else: raise ValueError("Invalid density matrix shape for %s: %s" % (cur_label,ar.shape))

        elif cur_format == "LiouvilleVec":
            qty = _np.transpose( _evalRowList( cur_rows, bComplex=False ) )

        elif cur_format == "UnitaryMx":
            ar = _evalRowList( cur_rows, bComplex=True )
            qty = _tools.change_basis(_tools.unitary_to_process_mx(ar), 'std', basis)

        elif cur_format == "UnitaryMxExp":
            ar = _evalRowList( cur_rows, bComplex=True )
            qty = _tools.change_basis(_tools.unitary_to_process_mx(_expm(-1j*ar)), 'std', basis)

        elif cur_format == "LiouvilleMx":
            qty = _evalRowList( cur_rows, bComplex=False )

        assert(qty is not None), "Invalid format: %s" % cur_format

        if cur_typ == "PREP":
            gs.preps[cur_label] = _objs.FullyParameterizedSPAMVec(qty)
        elif cur_typ == "TP-PREP":
            gs.preps[cur_label] = _objs.TPParameterizedSPAMVec(qty)
        elif cur_typ == "STATIC-PREP":
            gs.preps[cur_label] = _objs.StaticSPAMVec(qty)

        elif cur_typ in ("EFFECT","TP-EFFECT","STATIC-EFFECT"):
            if cur_typ == "EFFECT": qty = _objs.FullyParameterizedSPAMVec(qty)
            elif cur_typ == "TP-EFFECT": qty = _objs.TPParameterizedSPAMVec(qty)
            elif cur_typ == "STATIC-EFFECT": qty = _objs.StaticSPAMVec(qty)
            if "effects" in cur_group_info:
                cur_group_info['effects'].append( (cur_label,qty) )
            else:  cur_group_info['effects'] = [ (cur_label,qty) ]            
                
        elif cur_typ == "GATE":
            gs.gates[cur_label] = _objs.FullyParameterizedGate(qty)
        elif cur_typ == "TP-GATE":
            gs.gates[cur_label] = _objs.TPParameterizedGate(qty)
        elif cur_typ == "CPTP-GATE":
            #Similar to gate.convert(...) method
            J = _tools.fast_jamiolkowski_iso_std(qty, basis) #Choi mx basis doesn't matter
            RANK_TOL = 1e-6
            if _np.linalg.matrix_rank(J, RANK_TOL) == 1: 
                unitary_post = qty # when 'gate' is unitary
            else: unitary_post = None

            nQubits = _np.log2(qty.shape[0])/2.0
            bQubits = bool(abs(nQubits-round(nQubits)) < 1e-10) #integer # of qubits?

            proj_basis = "pp" if (basis == "pp" or bQubits) else basis
            ham_basis = proj_basis
            nonham_basis = proj_basis
            nonham_diagonal_only = False; cptp = True; truncate=False
            gs.gates[cur_label] = _objs.LindbladParameterizedGate.from_gate_matrix(
                qty, unitary_post, ham_basis, nonham_basis,
                cptp, nonham_diagonal_only, truncate, basis)
        elif cur_typ == "STATIC-GATE":
            gs.gates[cur_label] = _objs.StaticGate(qty)

        elif cur_typ in ("IGATE","STATIC-IGATE"):
            mxOrGate = _objs.StaticGate(qty) if cur_typ == "STATIC-IGATE" \
                       else qty #just add numpy array `qty` to matrices list
                                # and it will be made into a fully-param gate.
            if "matrices" in cur_group_info:
                cur_group_info['matrices'].append( (cur_label,mxOrGate) )
            else:  cur_group_info['matrices'] = [ (cur_label,mxOrGate) ]
        else:
            raise ValueError("Unknown type: %s!" % cur_typ)


    def add_current_group():
        """ 
        Adds the current "group" - either a POVM or Instrument - which contains
        multiple objects.
        """
        if cur_group_typ == "POVM":
            gs.povms[cur_group] = _objs.UnconstrainedPOVM( cur_group_info['effects'] )
        elif cur_group_typ == "TP-POVM":
            assert(len(cur_group_info['effects']) > 1), "TP-POVMs must have at least 2 elements!"
            gs.povms[cur_group] = _objs.TPPOVM( cur_group_info['effects'] )
        elif cur_group_typ == "Instrument":
            gs.instruments[cur_group] = _objs.Instrument( cur_group_info['matrices'] )
        elif cur_group_typ == "TP-Instrument":
            gs.instruments[cur_group] = _objs.TPInstrument( cur_group_info['matrices'] )
        else:
            raise ValueError("Unknown group type: %s!" % cur_group_typ ) # pragma: no cover
            # should be unreachable given group-name test below


    gs = _objs.GateSet()
    spam_vecs = _OrderedDict();
    spam_labels = _OrderedDict(); remainder_spam_label = ""
    identity_vec = _np.transpose( _np.array( [ _np.sqrt(2.0), 0,0,0] ) )  #default = 1-QUBIT identity vector

    basis_abbrev = "pp" #default assumed basis
    basis_dims = None
    gaugegroup_name = None

    #First try to find basis:
    with open(filename) as inputfile:
        for line in inputfile:
            line = line.strip()

            if line.startswith("BASIS:"):
                parts = line[len("BASIS:"):].split()
                basis_abbrev = parts[0]
                if len(parts) > 1:
                    basis_dims = list(map(int, "".join(parts[1:]).split(",")))
                    if len(basis_dims) == 1: basis_dims = basis_dims[0]
                else:
                    basis_dims = None
            elif line.startswith("GAUGEGROUP:"):
                gaugegroup_name = line[len("GAUGEGROUP:"):].strip()
                if gaugegroup_name not in ("Full","TP","Unitary"):
                    _warnings.warn(("Unknown GAUGEGROUP name %s.  Default gauge"
                                    "group will be set to None") % gaugegroup_name)

    if basis_dims is not None:
        # then specfy a dimensionful basis at the outset
        basis = _objs.Basis(basis_abbrev, basis_dims)
    else:
        # otherwise we'll try to infer one at the end (and add_current routine
        # uses basis in a way that can infer a dimension)
        basis = basis_abbrev

    state = "look for label"
    cur_label = ""; cur_typ = ""
    cur_group = ""; cur_group_typ = ""
    cur_format = ""; cur_rows = []; cur_group_info = {}
    with open(filename) as inputfile:
        for line in inputfile:
            line = line.strip()

            if len(line) == 0 or line.startswith("END"):
                #Blank lines or "END..." statements trigger the end of objects
                state = "look for label"
                if len(cur_label) > 0:
                    add_current()
                    cur_label = ""; cur_rows = []

                #END... ends the current group
                if line.startswith("END"):
                    if len(cur_group) > 0:
                        add_current_group()
                        cur_group = ""; cur_group_info = {}

            elif line[0] == "#":
                pass # skip comments

            elif state == "look for label":
                parts = line.split(':')
                assert(len(parts) == 2), "Invalid '<type>: <label>' line: %s" % line
                typ = parts[0].strip()
                label = parts[1].strip()

                if typ in ("BASIS","GAUGEGROUP"):
                    pass #handled above
                
                elif typ in ("POVM","TP-POVM","Instrument","TP-Instrument"):
                    # if this is a group type, just record this and continue looking
                    #  for the next labeled object
                    cur_group = label
                    cur_group_typ = typ
                else:
                    #All other "types" should be objects with formatted data
                    # associated with them: set cur_label and cur_typ to hold 
                    # the object label and type - next read it in.
                    cur_label = label
                    cur_typ = typ
                    state = "expect format" # the default next action
                    
            elif state == "expect format":
                cur_format = line
                if cur_format not in ["StateVec", "DensityMx", "UnitaryMx", "UnitaryMxExp", "LiouvilleVec", "LiouvilleMx"]:
                    raise ValueError("Expected object format for label %s and got line: %s -- must specify a valid object format" % (cur_label,line))
                state = "read object"

            elif state == "read object":
                cur_rows.append( line.split() )

    if len(cur_label) > 0:
        add_current()
    if len(cur_group) > 0:
        add_current_group()

    #Try to infer basis dimension if none is given
    if basis_dims is None:
        if gs.get_dimension() is not None:
            basis_dims = int(round(_np.sqrt(gs.get_dimension())))
        elif len(spam_vecs) > 0:
            basis_dims = int(round(_np.sqrt(list(spam_vecs.values())[0].size)))
        else:
            raise ValueError("Cannot infer basis dimension!")

        #Set basis (only needed if we didn't set it above)
        gs.basis = _objs.Basis(basis_abbrev, basis_dims)
    else:
        gs.basis = basis # already created a Basis obj above

    #Add default gauge group -- the full group because
    # we add FullyParameterizedGates above.
    if gaugegroup_name == "Full":
        gs.default_gauge_group = _objs.FullGaugeGroup(gs.dim)
    elif gaugegroup_name == "TP":
        gs.default_gauge_group = _objs.TPGaugeGroup(gs.dim)
    elif gaugegroup_name == "Unitary":
        gs.default_gauge_group = _objs.UnitaryGaugeGroup(gs.dim, gs.basis)
    else:
        gs.default_gauge_group = None
        
    return gs
