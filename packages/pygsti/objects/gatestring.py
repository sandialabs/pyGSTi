from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
""" Defines the GateString class and derived classes which represent gate strings."""

import numpy as _np

def _gateSeqToStr(seq):
    if len(seq) == 0: return "{}" #special case of empty gate string
    return ''.join(seq)

class GateString(object):
    """
    Encapsulates a gate string as a tuple of gate labels associated
    with a string representation for that tuple.

    Typically there are multiple string representations for the same tuple (for
    example "GxGx" and "Gx^2" both correspond to the tuple ("Gx","Gx") ) and it
    is convenient to store a specific string represntation along with the tuple.

    A GateString objects behaves very similarly to a tuple and most operations
    supported by a tuple are supported by a GateString (e.g. adding, hashing,
    testing for equality, indexing,  slicing, multiplying).
    """

    def __init__(self, tupleOfGateLabels, stringRepresentation=None, bCheck=True):
        """
        Create a new GateString object

        Parameters
        ----------
        tupleOfGateLabels : tuple or GateString (or None)
            A tuple of gate labels specifying the gate sequence, or None if the
            sequence should be obtained by evaluating stringRepresentation as
            a standard-text-format gate string (e.g. "GxGy", "Gx(Gy)^2, or "{}").

        stringRepresentation : string, optional
            A string representation of this GateString.

        bCheck : bool, optional
            If true, raise ValueEror if stringRepresentation does not evaluate
            to tupleOfGateLabels.
        """

        if tupleOfGateLabels is None and stringRepresentation is None:
            raise ValueError("tupleOfGateLabels and stringRepresentation cannot both be None");

        if tupleOfGateLabels is None or (bCheck and stringRepresentation is not None):
            from ..io import stdinput as _stdinput
            parser = _stdinput.StdInputParser()
            chkTuple = parser.parse_gatestring( stringRepresentation )
            if tupleOfGateLabels is None: tupleOfGateLabels = chkTuple
            elif tuple(tupleOfGateLabels) != chkTuple:
                raise ValueError("Error intializing GateString: " +
                            " tuple and string do not match: %s != %s"
                             % (tuple(tupleOfGateLabels),stringRepresentation))

        # if tupleOfGateLabels is a GateString, then copy it
        if isinstance(tupleOfGateLabels, GateString):
            self.tup = tupleOfGateLabels.tup
            if stringRepresentation is None:
                self.str = tupleOfGateLabels.str
            else:
                self.str = stringRepresentation

        else:
            if stringRepresentation is None:
                stringRepresentation = _gateSeqToStr( tupleOfGateLabels )

            self.tup = tuple(tupleOfGateLabels)
            self.str = str(stringRepresentation)

    #Conversion routines for evalTree usage -- TODO: make these member functions
    def to_pythonstr(self,gateLabels):
        """
        Convert this gate string into a python string, where each gate label is
        represented as a **single** character, starting with 'A' and contining
        down the alphabet.  This can be useful for processing gate strings
        using python's string tools (regex in particular).

        Parameters
        ----------
        gateLabels : tuple
           tuple containing all the gate labels that will be mapped to alphabet
           characters, beginning with 'A'.

        Returns
        -------
        string
            The converted gate string.

        Examples
        --------
            ('Gx','Gx','Gy','Gx') => "AABA"
        """
        assert(len(gateLabels) < 26) #Complain if we go beyond 'Z'
        translateDict = {}; c = 'A'
        for gateLabel in gateLabels:
            translateDict[gateLabel] = c
            c = chr(ord(c) + 1)
        return "".join([ translateDict[gateLabel] for gateLabel in self.tup ])

    @classmethod
    def from_pythonstr(cls,pythonString,gateLabels):
        """
        Create a GateString from a python string where each gate label is
        represented as a **single** character, starting with 'A' and contining
        down the alphabet.  This performs the inverse of to_pythonstr(...).

        Parameters
        ----------
        pythonString : string
            string whose individual characters correspond to the gate labels of a
            gate string.

        gateLabels : tuple
           tuple containing all the gate labels that will be mapped to alphabet
           characters, beginning with 'A'.

        Returns
        -------
        GateString

        Examples
        --------
            "AABA" => ('Gx','Gx','Gy','Gx')
        """
        assert(len(gateLabels) < 26) #Complain if we go beyond 'Z'
        translateDict = {}; c = 'A'
        for gateLabel in gateLabels:
            translateDict[c] = gateLabel
            c = chr(ord(c) + 1)
        return cls( tuple([ translateDict[c] for c in pythonString ]) )


    def __str__(self):
        return self.str

    def __len__(self):
        return len(self.tup)

    def __repr__(self):
        return "GateString(%s)" % self.str

    def __iter__(self):
        return self.tup.__iter__()

    def __add__(self,x):
        if not isinstance(x, GateString):
            raise ValueError("Can only add GateStrings objects to other GateString objects")
        if self.str != "{}":
            s = (self.str + x.str) if x.str != "{}" else self.str
        else: s = x.str
        return GateString(self.tup + x.tup, s, bCheck=False)

    def __mul__(self,x):
        assert( (isinstance(x,int) or _np.issubdtype(x,int)) and x >= 0)
        if x > 1: s = "(%s)^%d" % (self.str,x)
        elif x == 1: s = "(%s)" % self.str
        else: s = "{}"
        return GateString(self.tup * x, s, bCheck=False)

    def __pow__(self,x): #same as __mul__()
        return self.__mul__(x)

    def __eq__(self,x):
        if x is None: return False
        return self.tup == tuple(x) #better than x.tup since x can be a tuple

    def __lt__(self,x):
        return self.tup.__lt__(x)

    def __gt__(self,x):
        return self.tup.__gt__(x)

    def __hash__(self):
        return self.tup.__hash__()

    def __copy__(self):
        return GateString( self.tup, self.str, bCheck=False)

    #def __deepcopy__(self, memo):
    #    return GateString( self.tup, self.str, bCheck=False)

    def __getitem__(self, key):
        if isinstance( key, slice ):
            return GateString( self.tup.__getitem__(key) )
        return self.tup.__getitem__(key)

    def __setitem__(self, key, value):
        raise ValueError("Cannot set elements of GateString tuple (they're read-only)")



class WeightedGateString(GateString):
    """
    A GateString that contains an additional "weight" member used for
    building up weighted lists of gate strings.

    When two WeightedGateString objects are added together their weights
    add, and when a WeightedGateString object is multiplied by an integer
    (equivalent to being raised to a power) the weight is unchanged. When
    added to plain GateString objects, the plain GateString object is
    treated as having zero weight and the result is another WeightedGateString.
    """

    def __init__(self,tupleOfGateLabels, stringRepresentation=None, weight=1.0, bCheck=True):
        """
        Create a new WeightedGateString object

        Parameters
        ----------
        tupleOfGateLabels : tuple (or None)
            A tuple of gate labels specifying the gate sequence, or None if the
            sequence should be obtained by evaluating stringRepresentation as
            a standard-text-format gate string (e.g. "GxGy", "Gx(Gy)^2, or "{}").

        stringRepresentation : string, optional
            A string representation of this WeightedGateString.

        weight : float, optional
            the weight to assign this gate string.

        bCheck : bool, optional
            If true, raise ValueEror if stringRepresentation does not evaluate
            to tupleOfGateLabels.
        """
        self.weight = weight
        super(WeightedGateString,self).__init__(tupleOfGateLabels, stringRepresentation, bCheck)

    def __repr__(self):
        return "WeightedGateString(%s,%g)" % (self.str,self.weight)

    def __add__(self,x):
        tmp = super(WeightedGateString,self).__add__(x)
        x_weight = x.weight if type(x) == WeightedGateString else 0.0
        return WeightedGateString( tmp.tup, tmp.str, self.weight + x_weight, bCheck=False ) #add weights

    def __radd__(self,x):
        if isinstance(x, GateString):
            tmp = x.__add__(self)
            x_weight = x.weight if type(x) == WeightedGateString else 0.0
            return WeightedGateString( tmp.tup, tmp.str, x_weight + self.weight, bCheck=False )
        raise ValueError("Can only add GateStrings objects to other GateString objects")

    def __mul__(self,x):
        tmp = super(WeightedGateString,self).__mul__(x)
        return WeightedGateString(tmp.tup, tmp.str, self.weight, bCheck=False) #keep weight

    def __copy__(self):
        return WeightedGateString( self.tup, self.str, self.weight, bCheck=False )

#    def __deepcopy__(self, memo):
#        return WeightedGateString( self.tup, self.str, self.weight, bCheck=False )

    def __getitem__(self, key):
        if isinstance( key, slice ):
            return WeightedGateString( self.tup.__getitem__(key), None, self.weight, bCheck=False )
        return self.tup.__getitem__(key)


class CompressedGateString(object):
    """
    A "compressed" GateString class which reduces the memory or disk space
    required to hold the tuple part of a GateString by compressing it.

    One place where CompressedGateString objects can be useful is when saving
    large lists of long gate sequences in some non-human-readable format (e.g.
    pickle).  CompressedGateString objects *cannot* be used in place of
    GateString objects within pyGSTi, and so are *not* useful when manipulating
    and running algorithms which use gate sequences.
    """

    def __init__(self, gatestring, minLenToCompress=20, maxPeriodToLookFor=20):
        """
        Create a new CompressedGateString object

        Parameters
        ----------
        gatestring : GateString
            The gate string object which is compressed to create
            a new CompressedGateString object.

        minLenToCompress : int, optional
            The minimum length string to compress.  If len(gatestring)
            is less than this amount its tuple is returned.

        maxPeriodToLookFor : int, optional
            The maximum period length to use when searching for periodic
            structure within gatestring.  Larger values mean the method
            takes more time but could result in better compressing.
        """
        if not isinstance(gatestring, GateString):
            raise ValueError("CompressedGateStrings can only be created from existing GateString objects")
        self.tup = CompressedGateString.compress_gate_label_tuple(
            gatestring.tup, minLenToCompress, maxPeriodToLookFor)
        self.str = gatestring.str

    def expand(self):
        """
        Expands this compressed gate string into a GateString object.

        Returns
        -------
        GateString
        """
        tup = CompressedGateString.expand_gate_label_tuple(self.tup)
        return GateString(tup, self.str, bCheck=False)

    @staticmethod
    def _getNumPeriods(gateString, periodLen):
        n = 0
        if len(gateString) < periodLen: return 0
        while gateString[0:periodLen] == gateString[n*periodLen:(n+1)*periodLen]:
            n += 1
        return n


    @staticmethod
    def compress_gate_label_tuple(gateString, minLenToCompress=20, maxPeriodToLookFor=20):
        """
        Compress a gate string.  The result is tuple with a special compressed-
        gate-string form form that is not useable by other GST methods but is
        typically shorter (especially for long gate strings with a repetative
        structure) than the original gate string tuple.

        Parameters
        ----------
        gateString : tuple of gate labels or GateString
            The gate string to compress.

        minLenToCompress : int, optional
            The minimum length string to compress.  If len(gateString)
            is less than this amount its tuple is returned.

        maxPeriodToLookFor : int, optional
            The maximum period length to use when searching for periodic
            structure within gateString.  Larger values mean the method
            takes more time but could result in better compressing.

        Returns
        -------
        tuple
            The compressed (or raw) gate string tuple.
        """
        gateString = tuple(gateString) # converts from GateString or list to tuple if needed
        L = len(gateString)
        if L < minLenToCompress: return tuple(gateString)
        compressed = ["CCC"] #list for appending, then make into tuple at the end
        start = 0
        while start < L:
            #print "Start = ",start
            score = _np.zeros( maxPeriodToLookFor+1, 'd' )
            numperiods = _np.zeros( maxPeriodToLookFor+1, 'i' )
            for periodLen in range(1,maxPeriodToLookFor+1):
                n = CompressedGateString._getNumPeriods( gateString[start:], periodLen )
                if n == 0: score[periodLen] = 0
                elif n == 1: score[periodLen] = 4.1/periodLen
                else: score[periodLen] = _np.sqrt(periodLen)*n
                numperiods[periodLen] = n
            bestPeriodLen = _np.argmax(score)
            n = numperiods[bestPeriodLen]
            bestPeriod = gateString[start:start+bestPeriodLen]
            #print "Scores = ",score
            #print "num per = ",numperiods
            #print "best = %s ^ %d" % (str(bestPeriod), n)
            assert(n > 0 and bestPeriodLen > 0)
            if start > 0 and n == 1 and compressed[-1][1] == 1:
                compressed[-1] = (compressed[-1][0]+bestPeriod, 1)
            else:
                compressed.append( (bestPeriod, n) )
            start = start+bestPeriodLen*n

        return tuple(compressed)

    @staticmethod
    def expand_gate_label_tuple(compressedGateString):
        """
        Expand a compressed tuple created with compress_gate_label_tuple(...)
        into a tuple of gate labels.

        Parameters
        ----------
        compressedGateString : tuple
            a tuple in the compressed form created by
            compress_gate_label_tuple(...).

        Returns
        -------
        tuple
            A tuple of gate labels specifying the uncompressed gate string.
        """

        if len(compressedGateString) == 0: return ()
        if compressedGateString[0] != "CCC": return compressedGateString
        expandedString = []
        for (period,n) in compressedGateString[1:]:
            expandedString += period*n
        return tuple(expandedString)



#Now tested in unit tests
#if __name__ == "__main__":
#    wgs = WeightedGateString(('Gx',), weight=0.5)
#    gs = GateString(('Gx',) )
#    print ((gs + wgs)*2).weight
#    print (wgs + gs).weight
