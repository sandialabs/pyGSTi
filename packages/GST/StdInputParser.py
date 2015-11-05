""" Text-parsering classes and functions to read input files."""
import os as _os
import sys as _sys
import numpy as _np
import warnings as _warnings
from scipy.linalg import expm as _expm
from collections import OrderedDict as _OrderedDict

import Gate as _Gate
import gatestring as _gatestring
import BasisTools as _BT
import dataset as _dataset
import multidataset as _multidataset
import gateset as _gateset

import pyparsing as _pp
import string as _string

_pp.ParserElement.enablePackrat()
_sys.setrecursionlimit(10000)

class StdInputParser:
    """
    Encapsulates a text parser for reading GST input files.
    
    ** Grammar **

    expop   :: '^'
    multop  :: '*'
    integer :: '0'..'9'+
    real    :: ['+'|'-'] integer [ '.' integer [ 'e' ['+'|'-'] integer ] ]
    reflbl  :: (alpha | digit | '_')+
    
    nop     :: '{}'
    gate    :: 'G' [ lowercase | digit | '_' ]+
    strref  :: 'S' '[' reflbl ']'
    slcref  :: strref [ '[' integer ':' integer ']' ]
    expable :: gate | slcref | '(' string ')' | nop
    expdstr :: expable [ expop integer ]*
    string  :: expdstr [ [ multop ] expdstr ]*
    
    dataline :: string [ real ]+
    dictline :: reflbl string
    """

    def __init__(self):
        """ Creates a new StdInputParser object """

        def pushFirst( strg, loc, toks ):
            self.exprStack.append( toks[0] )
        def pushMult( strg, loc, toks ):
            self.exprStack.append( '*' )
        def pushSlice( strg, loc, toks ):
            self.exprStack.append( 'SLICE' )
        def pushCount( strg, loc, toks ):
            self.exprStack.append( toks[0] )
            self.exprStack.append( 'COUNT' )
    
        #caps = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        lowers = _string.lowercase #caps.lower()
        digits = _pp.nums #"0123456789"  #same as "nums"
        #point = _pp.Literal( "." )
        #e     = _pp.CaselessLiteral( "E" )
        #real  = _pp.Combine( _pp.Word( "+-"+_pp.nums, _pp.nums ) + 
        #                  _pp.Optional( point + _pp.Optional( _pp.Word( _pp.nums ) ) ) +
        #                  _pp.Optional( e + _pp.Word( "+-"+_pp.nums, _pp.nums ) ) ).setParseAction(pushFirst)
        real = _pp.Regex(r'[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?') #faster than above
        nop   = _pp.Literal("{}").setParseAction(pushFirst)

        expop = _pp.Literal( "^" )
        lpar  = _pp.Literal( "(" ).suppress()
        rpar  = _pp.Literal( ")" ).suppress()
        lbrk  = _pp.Literal( "[" ).suppress()
        rbrk  = _pp.Literal( "]" ).suppress()
    
        integer = _pp.Word( digits ).setParseAction(pushFirst)
        reflbl  = _pp.Word(_pp.alphas+_pp.nums+"_").setParseAction(pushFirst)
        #gate   = _pp.Word( "G", lowers + digits + "_" ).setParseAction(pushFirst)
        gate    = _pp.Regex(r'G[a-z0-9_]+').setParseAction(pushFirst) #faster than above
        strref  = (_pp.Literal("S") + "[" + reflbl + "]" ).setParseAction(pushFirst)
        slcref  = (strref + _pp.Optional( ("[" + integer + ":" + integer + "]").setParseAction(pushSlice)) )

        bSimple = False #experimenting with possible parser speedups
        if bSimple:
            string  = _pp.Forward()
            gateSeq = _pp.OneOrMore( gate )
            expable = (nop | gateSeq | lpar + gateSeq + rpar)
            expdstr = expable + _pp.Optional( (expop + integer).setParseAction(pushFirst) )
            string << expdstr + _pp.ZeroOrMore( (_pp.Optional("*") + expdstr).setParseAction(pushMult))
        else:
            string  = _pp.Forward()
            expable = (gate | slcref | lpar + string + rpar | nop)
            expdstr = expable + _pp.ZeroOrMore( (expop + integer).setParseAction(pushFirst) )
            string << expdstr + _pp.ZeroOrMore( (_pp.Optional("*") + expdstr).setParseAction(pushMult))
        
        count = real.copy().setParseAction(pushCount)
        dataline = string + _pp.OneOrMore( count )
        dictline = reflbl + string

        self.string_parser = string
        self.dataline_parser = dataline
        self.dictline_parser = dictline


    def _evaluateStack(self, s):
        op = s.pop()
        if op == "*":
            op2 = self._evaluateStack( s )
            op1 = self._evaluateStack( s )
            return op1 + op2 #tuple addition
    
        elif op == "^":
            exp = self._evaluateStack( s )
            op  = self._evaluateStack( s )
            return op*exp #tuple mulitplication = repeat op exp times

        elif op == "SLICE":
            upper = self._evaluateStack( s )
            lower = self._evaluateStack( s )
            op = self._evaluateStack( s )
            return op[lower:upper]

        elif op == 'COUNT':
            cnt = float(s.pop())          # next item on stack is a count
            self.countList.insert(0, cnt) # so add it to countList and eval the rest of the stack
            return self._evaluateStack( s )
    
        elif op[0] == 'G':
            return (op,) #as tuple

        elif op[0] == 'S':
            reflabel = s.pop() # next item on stack is a reference label (keep as a str)
            return tuple(self.lookup[reflabel]) #lookup dict typically holds GateString objs...

        elif op == '{}': # no-op returns empty tuple
            return ()

        else:
            return int(op)

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
        self.lookup = lookup
        self.exprStack = []
        result = self.string_parser.parseString(s)
        #print "DB: result = ",result
        #print "DB: stack = ",self.exprStack
        return self._evaluateStack(self.exprStack)

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
        self.lookup = lookup
        self.exprStack = []
        self.countList = []

        #get counts from end of s
        parts = s.split(); counts = []
        for p in reversed(parts):
            try: f = float(p)
            except: break
            counts.append( f )
        counts.reverse() #because we appended them in reversed order
        if len(counts) > expectedCounts >= 0:
            counts = counts[0:expectedCounts]

        nCounts = len(counts)
        if expectedCounts >= 0 and nCounts != expectedCounts:
            raise ValueError("Found %d count columns when %d were expected" % (nCounts, expectedCounts))
        if nCounts == len(parts):
            raise ValueError("No gatestring column found -- all columns look like data")

        gateStringStr = " ".join(parts[0:len(parts)-nCounts])
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
        self.exprStack = []
        result = self.dictline_parser.parseString(s)
        #print "DB: result = ",result
        #print "DB: stack = ",self.exprStack
        gateStringLabel = result[0]
        gateStringTuple = self._evaluateStack(self.exprStack)
        gateStringStr = s[ s.index(gateStringLabel) + len(gateStringLabel) : ].strip()
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
        gateStringList = [ ]
        for line in open(filename,'r'):
            line = line.strip()
            if len(line) == 0 or line[0] =='#': continue
            gateStringList.append( _gatestring.GateString(self.parse_gatestring(line), line) )
        return gateStringList

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
        for line in open(filename,'r'):
            line = line.strip()
            if len(line) == 0 or line[0] =='#': continue
            label, tup, s = self.parse_dictline(line)
            lookupDict[ label ] = _gatestring.GateString(tup, s)
        return lookupDict
            
    def parse_datafile(self, filename):
        """ 
        Parse a data set file into a DataSet object.

        Parameters
        ----------
        filename : string
            The file to parse.

        Returns
        -------
        DataSet
            A static DataSet object.
        """

        #Parse preamble -- lines beginning with # or ## until first non-# line
        preamble_directives = { }
        for line in open(filename,'r'):
            line = line.strip()
            if len(line) == 0 or line[0] != '#': break
            if line.startswith("## "):
                parts = line[len("## "):].split("=")
                if len(parts) == 2: # key = value
                    preamble_directives[ parts[0].strip() ] = parts[1].strip()
        
        #Process premble
        orig_cwd = _os.getcwd()
        if len(_os.path.dirname(filename)) > 0: _os.chdir( _os.path.dirname(filename) ) #allow paths relative to datafile path
        if preamble_directives.has_key('Lookup'): 
            lookupDict = self.parse_dictfile( preamble_directives['Lookup'] )
        else: lookupDict = { }
        if preamble_directives.has_key('Columns'):
            colLabels = [ l.strip() for l in preamble_directives['Columns'].split(",") ]
        else: colLabels = [ 'plus count', 'count total' ] #  spamLabel (' frequency' | ' count') | 'count total' |  ?? 'T0' | 'Tf' ??
        spamLabels,fillInfo = self._extractLabelsFromColLabels(colLabels)
        nDataCols = len(colLabels)
        _os.chdir(orig_cwd)

        #Read data lines of data file
        dataset = _dataset.DataSet(spamLabels=spamLabels)
        nLines = sum(1 for line in open(filename,'r'))
        nSkip = int(nLines / 100.0)
	if nSkip == 0: nSkip = 1

        def is_interactive():
            import __main__ as main
            return not hasattr(main, '__file__')

        if is_interactive():
            try:
                import time
                from IPython.display import clear_output
                def displayProgress(i,N): 
                    time.sleep(0.001)
                    clear_output()
                    print "Loading %s: %.0f%%" % (filename, 100.0*float(i)/float(N))
                    _sys.stdout.flush()
            except:
                def displayProgress(i,N): pass
        else:
            def displayProgress(i,N): pass

        countDict = {}
        for (iLine,line) in enumerate(open(filename,'r')):
            if iLine % nSkip == 0 or iLine+1 == nLines: displayProgress(iLine+1, nLines)

            line = line.strip()
            if len(line) == 0 or line[0] == '#': continue
            try:
                gateStringTuple, gateStringStr, valueList = self.parse_dataline(line, lookupDict, nDataCols)
            except _pp.ParseException as e:
                raise _pp.ParseException("Parse error in file %s in line:\n%s\nError = %s" % \
                    (filename, line, str(e)))
            self._fillDataCountDict( countDict, fillInfo, valueList )
            if all([ (abs(v) < 1e-9) for v in countDict.values()]):
                _warnings.warn( "Dataline for gateString '%s' has zero counts and will be ignored" % gateStringStr)
                continue #skip lines in dataset file with zero counts (no experiments done)
            dataset.addCountDict(gateStringTuple, countDict) #Note: don't use gateStringStr since DataSet currently doesn't hold GateString objs (just tuples)
                
        dataset.doneAddingData()
        return dataset

    def _extractLabelsFromColLabels(self, colLabels ):
        spamLabels = []; countCols = []; freqCols = []; impliedCountTotCol1Q = -1
        for i,colLabel in enumerate(colLabels):
            if colLabel.endswith(' count'):
                spamLabel = colLabel[:-len(' count')]
                if spamLabel not in spamLabels: spamLabels.append( spamLabel )
                countCols.append( (spamLabel,i) )

            elif colLabel.endswith(' frequency'):
                if 'count total' not in colLabels:
                    raise ValueError("Frequency columns specified without count total")
                else: iTotal = colLabels.index( 'count total' )
                spamLabel = colLabel[:-len(' frequency')]
                if spamLabel not in spamLabels: spamLabels.append( spamLabel )
                freqCols.append( (spamLabel,i,iTotal) )
        
        if 'count total' in colLabels:
            if 'plus' in spamLabels and 'minus' not in spamLabels:
                spamLabels.append('minus')
                impliedCountTotCol1Q = colLabels.index( 'count total' )
            #TODO - add standard count completion for 2Qubit case?
        
        fillInfo = (countCols, freqCols, impliedCountTotCol1Q)
        return spamLabels, fillInfo


    def _fillDataCountDict(self, countDict, fillInfo, colValues):
        countCols, freqCols, impliedCountTotCol1Q = fillInfo

        for spamLabel,iCol in countCols:
            if colValues[iCol] > 0 and colValues[iCol] < 1:
                raise ValueError("Count column (%d) contains value(s) " % iCol +
                                 "between 0 and 1 - could this be a frequency?")
            countDict[spamLabel] = colValues[iCol]

        for spamLabel,iCol,iTotCol in freqCols:
            if colValues[iCol] < 0 or colValues[iCol] > 1.0:
                    raise ValueError("Frequency column (%d) contains value(s) " % iCol +
                                     "outside of [0,1.0] interval - could this be a count?")
            countDict[spamLabel] = colValues[iCol] * colValues[iTotCol]

        if impliedCountTotCol1Q >= 0:
            countDict['minus'] = colValues[impliedCountTotCol1Q] - countDict['plus']
        #TODO - add standard count completion for 2Qubit case?
        return countDict


    def parse_multidatafile(self, filename):
        """ 
        Parse a multiple data set file into a MultiDataSet object.

        Parameters
        ----------
        filename : string
            The file to parse.

        Returns
        -------
        MultiDataSet
            A MultiDataSet object.
        """

        #Parse preamble -- lines beginning with # or ## until first non-# line
        preamble_directives = { }
        for line in open(filename,'r'):
            line = line.strip()
            if len(line) == 0 or line[0] != '#': break
            if line.startswith("## "):
                parts = line[len("## "):].split("=")
                if len(parts) == 2: # key = value
                    preamble_directives[ parts[0].strip() ] = parts[1].strip()
        
        #Process premble
        orig_cwd = _os.getcwd()
        if len(_os.path.dirname(filename)) > 0: 
            _os.chdir( _os.path.dirname(filename) ) #allow paths relative to datafile path
        if preamble_directives.has_key('Lookup'): 
            lookupDict = self.parse_dictfile( preamble_directives['Lookup'] )
        else: lookupDict = { }
        if preamble_directives.has_key('Columns'):
            colLabels = [ l.strip() for l in preamble_directives['Columns'].split(",") ]
        else: colLabels = [ 'dataset1 plus count', 'dataset1 count total' ]
        dsSpamLabels, fillInfo = self._extractLabelsFromMultiDataColLabels(colLabels)
        nDataCols = len(colLabels)
        _os.chdir(orig_cwd)

        #Read data lines of data file
        datasets = _OrderedDict()
        for dsLabel,spamLabels in dsSpamLabels.iteritems():
            datasets[dsLabel] = _dataset.DataSet(spamLabels=spamLabels)

        dsCountDicts = _OrderedDict()
        for dsLabel in dsSpamLabels: dsCountDicts[dsLabel] = {}

        nLines = sum(1 for line in open(filename,'r'))
        nSkip = max(int(nLines / 100.0),1)

        def is_interactive():
            import __main__ as main
            return not hasattr(main, '__file__')

        if is_interactive():
            try:
                import time
                from IPython.display import clear_output
                def displayProgress(i,N): 
                    time.sleep(0.001)
                    clear_output()
                    print "Loading %s: %.0f%%" % (filename, 100.0*float(i)/float(N))
                    _sys.stdout.flush()
            except:
                def displayProgress(i,N): pass
        else:
            def displayProgress(i,N): pass

        for (iLine,line) in enumerate(open(filename,'r')):
            if iLine % nSkip == 0 or iLine+1 == nLines: displayProgress(iLine+1, nLines)

            line = line.strip()
            if len(line) == 0 or line[0] == '#': continue
            try:
                gateStringTuple, gateStringStr, valueList = self.parse_dataline(line, lookupDict, nDataCols)
            except _pp.ParseException as e:
                raise _pp.ParseException("Parse error in file %s in line:\n%s\nError = %s" % \
                    (filename, line, str(e)))
            self._fillMultiDataCountDicts(dsCountDicts, fillInfo, valueList)
            for dsLabel, countDict in dsCountDicts.iteritems():
                datasets[dsLabel].addCountDict(gateStringTuple, countDict)
              
        mds = _multidataset.MultiDataSet()
        for dsLabel,ds in datasets.iteritems():
            ds.doneAddingData()
            mds.addDataset(dsLabel, ds)
        return mds


    #Note: spam labels must not contain spaces since we use spaces to separate 
    # the spam label from the dataset label
    def _extractLabelsFromMultiDataColLabels(self, colLabels):
        dsSpamLabels = _OrderedDict()
        countCols = []; freqCols = []; impliedCounts1Q = []
        for i,colLabel in enumerate(colLabels):
            wordsInColLabel = colLabel.split() #split on whitespace into words
            if len(wordsInColLabel) < 3: continue #allow other columns we don't recognize

            if wordsInColLabel[-1] == 'count':
                spamLabel = wordsInColLabel[-2]
                dsLabel = wordsInColLabel[-3]
                if dsLabel not in dsSpamLabels: 
                    dsSpamLabels[dsLabel] = [ spamLabel ]
                else: dsSpamLabels[dsLabel].append( spamLabel )
                countCols.append( (dsLabel,spamLabel,i) )

            elif wordsInColLabel[-1] == 'frequency':
                spamLabel = wordsInColLabel[-2]
                dsLabel = wordsInColLabel[-3]
                if '%s count total' % dsLabel not in colLabels:
                    raise ValueError("Frequency columns specified without" +
                                     "count total for dataset '%s'" % dslabel)
                else: iTotal = colLabels.index( '%s count total' % dsLabel )
                
                if dsLabel not in dsSpamLabels: 
                    dsSpamLabels[dsLabel] = [ spamLabel ]
                else: dsSpamLabels[dsLabel].append( spamLabel )
                freqCols.append( (dsLabel,spamLabel,i,iTotal) )
        
        for dsLabel,spamLabels in dsSpamLabels.iteritems():
            if '%s count total' % dsLabel in colLabels:
                if 'plus' in spamLabels and 'minus' not in spamLabels:
                    dsSpamLabels[dsLabel].append('minus')
                    iTotal = colLabels.index( '%s count total' % dsLabel )
                    impliedCounts1Q.append( (dsLabel, iTotal) )
            #TODO - add standard count completion for 2Qubit case?
        
        fillInfo = (countCols, freqCols, impliedCounts1Q)
        return dsSpamLabels, fillInfo


    def _fillMultiDataCountDicts(self, countDicts, fillInfo, colValues):
        countCols, freqCols, impliedCounts1Q = fillInfo

        for dsLabel,spamLabel,iCol in countCols:
            if colValues[iCol] > 0 and colValues[iCol] < 1:
                raise ValueError("Count column (%d) contains value(s) " % iCol +
                                 "between 0 and 1 - could this be a frequency?")
            countDicts[dsLabel][spamLabel] = colValues[iCol]

        for dsLabel,spamLabel,iCol,iTotCol in freqCols:
            if colValues[iCol] < 0 or colValues[iCol] > 1.0:
                    raise ValueError("Frequency column (%d) contains value(s) " % iCol +
                                     "outside of [0,1.0] interval - could this be a count?")
            countDicts[dsLabel][spamLabel] = colValues[iCol] * colValues[iTotCol]

        for dsLabel,iTotCol in impliedCounts1Q:
            countDicts[dsLabel]['minus'] = colValues[iTotCol] - countDicts[dsLabel]['plus']
        #TODO - add standard count completion for 2Qubit case?
        return countDicts



def _evalElement(el, bComplex):
    myLocal = { 'pi': _np.pi, 'sqrt': _np.sqrt }
    exec( "element = %s" % el, {"__builtins__": None}, myLocal )
    return complex( myLocal['element'] ) if bComplex else float( myLocal['element'] )

def _evalRowList(rows, bComplex):
    return _np.array( [ [ _evalElement(x,bComplex) for x in r ] for r in rows ], 
                     'complex' if bComplex else 'd' )

def readGateset(filename):
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

    def add_current_label():
        if cur_format == "StateVec":
            ar = _evalRowList( cur_rows, bComplex=True )
            if ar.shape == (1,2):
                spam_vecs[cur_label] = _BT.stateToPauliDensityVec(ar[0,:])
            else: raise ValueError("Invalid state vector shape for %s: %s" % (cur_label,ar.shape))
            
        elif cur_format == "DensityMx":
            ar = _evalRowList( cur_rows, bComplex=True )
            if ar.shape == (2,2):
                spam_vecs[cur_label] = _BT.matrixInStdBasisToPauliProdVector(ar)
            else: raise ValueError("Invalid density matrix shape for %s: %s" % (cur_label,ar.shape))

        elif cur_format == "PauliVec":
            spam_vecs[cur_label] = _np.transpose( _evalRowList( cur_rows, bComplex=False ) )
            
        elif cur_format == "UnitaryMx":
            ar = _evalRowList( cur_rows, bComplex=True )
            if ar.shape == (2,2):
                gs.set_gate(cur_label, _Gate.FullyParameterizedGate(_BT.stateUnitaryToPauliDensityMxOp(ar)))
            elif ar.shape == (4,4):
                gs.set_gate(cur_label, _Gate.FullyParameterizedGate(_BT.stateUnitaryToPauliDensityMxOp_2Q(ar)))
            else: raise ValueError("Invalid unitary matrix shape for %s: %s" % (cur_label,ar.shape))
            
        elif cur_format == "UnitaryMxExp":
            ar = _evalRowList( cur_rows, bComplex=True )
            if ar.shape == (2,2):
                gs.set_gate(cur_label, _Gate.FullyParameterizedGate(_BT.stateUnitaryToPauliDensityMxOp( _expm(-1j * ar) )))
            elif ar.shape == (4,4):
                gs.set_gate(cur_label, _Gate.FullyParameterizedGate( _BT.stateUnitaryToPauliDensityMxOp_2Q( _expm(-1j * ar) )))
            else: raise ValueError("Invalid unitary matrix exponent shape for %s: %s" % (cur_label,ar.shape))
            
        elif cur_format == "PauliMx":
            gs.set_gate(cur_label, _Gate.FullyParameterizedGate( _evalRowList( cur_rows, bComplex=False ) ) )


    gs = _gateset.GateSet()
    spam_vecs = { }; spam_labels = { }; remainder_spam_label = ""
    identity_vec = _np.transpose( _np.array( [ _np.sqrt(2.0), 0,0,0] ) )  #default = 1-QUBIT identity vector

    state = "look for label"
    cur_label = ""; cur_format = ""; cur_rows = []
    for line in open(filename):
        line = line.strip()

        if len(line) == 0:
            state = "look for label"
            if len(cur_label) > 0:
                add_current_label()
                cur_label = ""; cur_rows = []
            continue

        if line[0] == "#": 
            continue
        
        if state == "look for label":
            if line.startswith("SPAMLABEL "):
                eqParts = line[len("SPAMLABEL "):].split('=')
                if len(eqParts) != 2: raise ValueError("Invalid spam label line: ", line)
                if eqParts[1].strip() == "remainder":
                    remainder_spam_label = eqParts[0].strip()
                else:
                    spam_labels[ eqParts[0].strip() ] = [ s.strip() for s in eqParts[1].split() ]

            elif line.startswith("IDENTITYVEC "):  #Vectorized form of identity density matrix in whatever basis is used
                if line != "IDENTITYVEC None":  #special case for designating no identity vector, so default is not used
                    identity_vec  = _np.transpose( _evalRowList( [ line[len("IDENTITYVEC "):].split() ], bComplex=False ) )
            else:
                cur_label = line
                state = "expect format"

        elif state == "expect format":
            cur_format = line
            if cur_format not in ["StateVec", "DensityMx", "UnitaryMx", "UnitaryMxExp", "PauliVec", "PauliMx"]:
                raise ValueError("Expected object format for label %s and got line: %s -- must specify a valid object format" % (cur_label,line))
            state = "read object"

        elif state == "read object":
            cur_rows.append( line.split() )

    if len(cur_label) > 0:
        add_current_label()

    #Default SPAMLABEL directive if none are give and rho and E vectors are:
    if len(spam_labels) == 0 and spam_vecs.has_key("rho") and spam_vecs.has_key("E"):
        spam_labels['plus'] = [ 'rho', 'E' ]
        remainder_spam_label = 'minus'
    if len(spam_labels) == 0: raise ValueError("Must specify rho and E or spam labels directly.")

    #Make SPAMs
     #get unique rho and E names
    rho_names = list(_OrderedDict.fromkeys( [ rho for (rho,E) in spam_labels.values() ] ) ) #if this fails, may be due to malformatted
    E_names   = list(_OrderedDict.fromkeys( [ E   for (rho,E) in spam_labels.values() ] ) ) #  SPAMLABEL line (not 2 items to right of = sign)
    if "remainder" in E_names:
        del E_names[ E_names.index("remainder") ]

     #add vectors to gateset
    for (i,rho_nm) in enumerate(rho_names): gs.set_rhoVec(spam_vecs[rho_nm],i)
    for (i,E_nm)   in enumerate(E_names):   gs.set_EVec(spam_vecs[E_nm],i)

    gs.set_identityVec(identity_vec)

     #add spam labels to gateset
    for spam_label in spam_labels:
        (rho_nm,E_nm) = spam_labels[spam_label]
        if E_nm == "remainder":
            gs.add_SPAM_label( rho_names.index(rho_nm) , -1, spam_label)
        else:
            gs.add_SPAM_label( rho_names.index(rho_nm) , E_names.index(E_nm), spam_label)        
    if len(remainder_spam_label) > 0:
        gs.add_SPAM_label( -1, -1, remainder_spam_label)

    return gs
