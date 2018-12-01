""" Defines the OpString class and derived classes which represent operation sequences."""
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

import numpy as _np
import itertools as _itertools
#import uuid  as _uuid
from ..tools import compattools as _compat
from ..baseobjs import CircuitParser as _CircuitParser
from ..baseobjs import Label as _Label

import os,inspect
debug_record = {}

def _opSeqToStr(seq):
    if len(seq) == 0: return "{}" #special case of empty operation sequence
    return ''.join(map(str,seq))

class OpString(object):
    """
    Encapsulates a operation sequence as a tuple of operation labels associated
    with a string representation for that tuple.

    Typically there are multiple string representations for the same tuple (for
    example "GxGx" and "Gx^2" both correspond to the tuple ("Gx","Gx") ) and it
    is convenient to store a specific string represntation along with the tuple.

    A OpString objects behaves very similarly to a tuple and most operations
    supported by a tuple are supported by a OpString (e.g. adding, hashing,
    testing for equality, indexing,  slicing, multiplying).
    """

    def __init__(self, tupleOfOpLabels, stringRepresentation=None, bCheck=True, lookup=None):
        """
        Create a new OpString object

        Parameters
        ----------
        tupleOfOpLabels : tuple or OpString (or None)
            A tuple of operation labels specifying the gate sequence, or None if the
            sequence should be obtained by evaluating stringRepresentation as
            a standard-text-format operation sequence (e.g. "GxGy", "Gx(Gy)^2, or "{}").

        stringRepresentation : string, optional
            A string representation of this OpString.

        bCheck : bool, optional
            If true, raise ValueEror if stringRepresentation does not evaluate
            to tupleOfOpLabels.

        lookup : dict, optional
            A dictionary with keys == labels and values == tuples of operation labels
            which can be used for substitutions using the S<label> syntax.
        """
        #self.uuid = _uuid.uuid4()
        
        #caller = inspect.getframeinfo(inspect.currentframe().f_back)
        #ky = "%s:%s:%d" % (caller[2],os.path.basename(caller[0]),caller[1])
        #debug_record[ky] = debug_record.get(ky, 0) + 1

        def convert_to_label(l):
            if isinstance(l, _Label): return l
            else: return _Label(l) # takes care of all other cases

        if tupleOfOpLabels is None and stringRepresentation is None:
            raise ValueError("tupleOfOpLabels and stringRepresentation cannot both be None");

        if tupleOfOpLabels is None or (bCheck and stringRepresentation is not None):
            gsparser = _CircuitParser()
            gsparser.lookup = lookup
            chk = gsparser.parse(stringRepresentation) # tuple of Labels
            if tupleOfOpLabels is None: tupleOfOpLabels = chk
            elif tuple(map(convert_to_label,tupleOfOpLabels)) != chk:
                raise ValueError("Error intializing OpString: " +
                            " tuple and string do not match: %s != %s"
                             % (tuple(tupleOfOpLabels),stringRepresentation))

        # if tupleOfOpLabels is a OpString, then copy it
        if isinstance(tupleOfOpLabels, OpString):
            self._tup = tupleOfOpLabels.tup
            if stringRepresentation is None:
                self._str = tupleOfOpLabels.str
            else:
                self._str = stringRepresentation

        else:
            #If we weren't given a OpString, convert all the elements of the tuple
            # to Labels.  Note that this post-processer parser output too, since the
            # parser returns a *tuple* not a OpString
            tupleOfOpLabels = tuple(map(convert_to_label,tupleOfOpLabels))

            #Note: now it's OK to have _str == None, as str is build on demand
            # In the past we did: if stringRepresentation is None:
            #    stringRepresentation = _opSeqToStr( tupleOfOpLabels )

            self._tup = tuple(tupleOfOpLabels)
            self._str = str(stringRepresentation) \
                        if (stringRepresentation is not None) else None

    @property
    def tup(self):
        """ This OpString as a standard Python tuple of Labels."""
        return self._tup

    @tup.setter
    def tup(self, value):
        """ This OpString as a standard Python tuple of Labels."""
        self._tup = value

    @property
    def str(self):
        """ The Python string representation of this OpString."""
        if self._str is None:
            self._str = _opSeqToStr(self.tup)
        return self._str

    @str.setter
    def str(self, value):
        """ The Python string representation of this OpString."""
        self._str = value

    def map_state_space_labels(self, mapper):
        """
        Return a copy of this operation sequence with all of the state-space-labels
        (often just qubit labels) updated according to a mapping function.

        For example, calling this function with `mapper = {0: 1, 1: 3}`
        on the string "Gx:0Gy:1Gx:1" would return "Gx:1Gy:3Gx:3".

        Parameters
        ----------
        mapper : dict or function
            A dictionary whose keys are the existing state-space-label values
            and whose value are the new labels, or a function which takes a
            single (existing label) argument and returns a new label.

        Returns
        -------
        OpString
        """
        return OpString( [l.map_state_space_labels(mapper) for l in self.tup] )

    def serialize(self):
        """
        Construct a operation sequence whereby all compound labels (containing multiple
        gates in parallel) in this string are converted to separate labels,
        effectively putting each elementary gate operation into its own "layer".
        Ordering is dictated by the ordering of the compound label.

        Returns
        -------
        OpString
        """
        serial_lbls = []
        for lbl in self:
            for c in lbl.components:
                serial_lbls.append(c)
        return OpString(serial_lbls)

    def parallelize(self, can_break_labels=True, adjacent_only=False):
        """
        Construct a circuit with the same underlying labels as this one,
        but with as many gates performed in parallel as possible (with
        some restrictions - see the Parameters sectin below).  Generally,
        gates are moved as far left (toward the start) in the sequence as
        possible.

        Parameters
        ----------
        can_break_labels : bool, optional
            Whether compound (parallel-gate) labels in this OpString can be
            separated during the parallelization process.  For example, if
            `can_break_labels=True` then `"Gx:0[Gy:0Gy:1]" => "[Gx:0Gy:1]Gy:0"`
            whereas if `can_break_labels=False` the result would remain 
            `"Gx:0[Gy:0Gy:1]"` because `[Gy:0Gy:1]` cannot be separated.

        adjacent_only : bool, optional
            It `True`, then operation labels are only allowed to move into an
            adjacent label, that is, they cannot move "through" other 
            operation labels.  For example, if `adjacent_only=True` then
            `"Gx:0Gy:0Gy:1" => "Gx:0[Gy:0Gy:1]"` whereas if 
            `adjacent_only=False` the result would be `"[Gx:0Gy:1]Gy:0`.
            Setting this to `True` is sometimes useful if you want to 
            parallelize a serial string in such a way that subsequently 
            calling `.serialize()` will give you back the original string.
            
        Returns
        -------
        OpString
        """
        parallel_lbls = []
        cur_components = []
        first_free = {'*':0}
        for lbl in self:
            if can_break_labels: # then process label components individually
                for c in lbl.components:
                    if c.sslbls is None: # ~= acts on *all* sslbls
                        pos = max(list(first_free.values())) 
                          #first position where all sslbls are free
                    else:
                        inds = [v for k,v in first_free.items() if k in c.sslbls]
                        pos = max(inds) if len(inds) > 0 else first_free['*']
                          #first position where all c.sslbls are free (uses special
                          # '*' "base" key if we haven't seen any of the sslbls yet)

                    if len(parallel_lbls) < pos+1: parallel_lbls.append([])
                    assert(pos < len(parallel_lbls))
                    parallel_lbls[pos].append(c) #add component in proper place

                    #update first_free
                    if adjacent_only: # all labels/components following this one must at least be at 'pos'
                        for k in first_free: first_free[k] = pos
                    if c.sslbls is None:
                        for k in first_free: first_free[k] = pos+1 #includes '*'
                    else:
                        for k in c.sslbls: first_free[k] = pos+1

            else: # can't break labels - treat as a whole
                if lbl.sslbls is None: # ~= acts on *all* sslbls
                    pos = max(list(first_free.values())) 
                      #first position where all sslbls are free
                else:
                    inds = [v for k,v in first_free.items() if k in lbl.sslbls]
                    pos = max(inds) if len(inds) > 0 else first_free['*']
                      #first position where all c.sslbls are free (uses special
                      # '*' "base" key if we haven't seen any of the sslbls yet)

                if len(parallel_lbls) < pos+1: parallel_lbls.append([])
                assert(pos < len(parallel_lbls))
                for c in lbl.components:  # add *all* components of lbl in proper place
                    parallel_lbls[pos].append(c)

                #update first_free
                if adjacent_only: # all labels/components following this one must at least be at 'pos'
                    for k in first_free: first_free[k] = pos
                if lbl.sslbls is None:
                    for k in first_free: first_free[k] = pos+1 #includes '*'
                else:
                    for k in lbl.sslbls: first_free[k] = pos+1
                    
        return OpString(parallel_lbls)
                    
            
    def to_pythonstr(self,opLabels):
        """
        Convert this operation sequence into a python string, where each operation label is
        represented as a **single** character, starting with 'A' and contining
        down the alphabet.  This can be useful for processing operation sequences
        using python's string tools (regex in particular).

        Parameters
        ----------
        opLabels : tuple
           tuple containing all the operation labels that will be mapped to alphabet
           characters, beginning with 'A'.

        Returns
        -------
        string
            The converted operation sequence.

        Examples
        --------
            ('Gx','Gx','Gy','Gx') => "AABA"
        """
        assert(len(opLabels) < 26) #Complain if we go beyond 'Z'
        translateDict = {}; c = 'A'
        for opLabel in opLabels:
            translateDict[opLabel] = c
            c = chr(ord(c) + 1)
        return "".join([ translateDict[opLabel] for opLabel in self.tup ])

    @classmethod
    def from_pythonstr(cls,pythonString,opLabels):
        """
        Create a OpString from a python string where each operation label is
        represented as a **single** character, starting with 'A' and contining
        down the alphabet.  This performs the inverse of to_pythonstr(...).

        Parameters
        ----------
        pythonString : string
            string whose individual characters correspond to the operation labels of a
            operation sequence.

        opLabels : tuple
           tuple containing all the operation labels that will be mapped to alphabet
           characters, beginning with 'A'.

        Returns
        -------
        OpString

        Examples
        --------
            "AABA" => ('Gx','Gx','Gy','Gx')
        """
        assert(len(opLabels) < 26) #Complain if we go beyond 'Z'
        translateDict = {}; c = 'A'
        for opLabel in opLabels:
            translateDict[c] = opLabel
            c = chr(ord(c) + 1)
        return cls( tuple([ translateDict[c] for c in pythonString ]) )

    def __str__(self):
        return self.str

    def __len__(self):
        return len(self.tup)

    def __repr__(self):
        return "OpString(%s)" % self.str

    def __iter__(self):
        return self.tup.__iter__()

    def __add__(self,x):
        if not isinstance(x, OpString):
            raise ValueError("Can only add Circuits objects to other OpString objects")
        if self.str != "{}":
            s = (self.str + x._str) if x.str != "{}" else self.str
        else: s = x.str
        return OpString(self.tup + x.tup, s, bCheck=False)

    def __mul__(self,x):
        assert( (_compat.isint(x) or _np.issubdtype(x,int)) and x >= 0)
        if x > 1: s = "(%s)^%d" % (self.str,x)
        elif x == 1: s = "(%s)" % self.str
        else: s = "{}"
        return OpString(self.tup * x, s, bCheck=False)

    def __pow__(self,x): #same as __mul__()
        return self.__mul__(x)

    def __eq__(self,x):
        if x is None: return False
        return self.tup == tuple(x) #better than x._tup since x can be a tuple

    def __lt__(self,x):
        return self.tup.__lt__(tuple(x))

    def __gt__(self,x):
        return self.tup.__gt__(tuple(x))

    def __hash__(self):
        return hash(self.tup)
        #return hash(self.uuid)

    def __copy__(self):
        return OpString( self.tup, self.str, bCheck=False)

    #def __deepcopy__(self, memo):
    #    return OpString( self._tup, self._str, bCheck=False)

    def __getitem__(self, key):
        if isinstance( key, slice ):
            return OpString( self.tup.__getitem__(key) )
        return self.tup.__getitem__(key)

    def __setitem__(self, key, value):
        raise ValueError("Cannot set elements of OpString tuple (they're read-only)")

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state_dict):
        self.__dict__.setdefault('_str', None)  # backwards compatibility
        for k, v in state_dict.items():
            if k == 'tup':                    # backwards compatibility
                self.tup = state_dict['tup'] # backwards compatibility
            else:
                self.__dict__[k] = v
    

class WeightedOpString(OpString):
    """
    A OpString that contains an additional "weight" member used for
    building up weighted lists of operation sequences.

    When two WeightedOpString objects are added together their weights
    add, and when a WeightedOpString object is multiplied by an integer
    (equivalent to being raised to a power) the weight is unchanged. When
    added to plain OpString objects, the plain OpString object is
    treated as having zero weight and the result is another WeightedOpString.
    """

    def __init__(self, tupleOfOpLabels, stringRepresentation=None, weight=1.0, bCheck=True):
        """
        Create a new WeightedOpString object

        Parameters
        ----------
        tupleOfOpLabels : tuple (or None)
            A tuple of operation labels specifying the gate sequence, or None if the
            sequence should be obtained by evaluating stringRepresentation as
            a standard-text-format operation sequence (e.g. "GxGy", "Gx(Gy)^2, or "{}").

        stringRepresentation : string, optional
            A string representation of this WeightedOpString.

        weight : float, optional
            the weight to assign this operation sequence.

        bCheck : bool, optional
            If true, raise ValueEror if stringRepresentation does not evaluate
            to tupleOfOpLabels.
        """
        self.weight = weight
        super(WeightedOpString,self).__init__(tupleOfOpLabels, stringRepresentation, bCheck)

    def __repr__(self):
        return "WeightedOpString(%s,%g)" % (self._str,self.weight)

    def __add__(self,x):
        tmp = super(WeightedOpString,self).__add__(x)
        x_weight = x.weight if type(x) == WeightedOpString else 0.0
        return WeightedOpString( tmp._tup, tmp._str, self.weight + x_weight, bCheck=False ) #add weights

    def __radd__(self,x):
        if isinstance(x, OpString):
            tmp = x.__add__(self)
            x_weight = x.weight if type(x) == WeightedOpString else 0.0
            return WeightedOpString( tmp._tup, tmp._str, x_weight + self.weight, bCheck=False )
        raise ValueError("Can only add Circuits objects to other OpString objects")

    def __mul__(self,x):
        tmp = super(WeightedOpString,self).__mul__(x)
        return WeightedOpString(tmp._tup, tmp._str, self.weight, bCheck=False) #keep weight

    def __copy__(self):
        return WeightedOpString( self._tup, self._str, self.weight, bCheck=False )

#    def __deepcopy__(self, memo):
#        return WeightedOpString( self._tup, self._str, self.weight, bCheck=False )

    def __getitem__(self, key):
        if isinstance( key, slice ):
            return WeightedOpString( self._tup.__getitem__(key), None, self.weight, bCheck=False )
        return self._tup.__getitem__(key)


class CompressedOpString(object):
    """
    A "compressed" OpString class which reduces the memory or disk space
    required to hold the tuple part of a OpString by compressing it.

    One place where CompressedOpString objects can be useful is when saving
    large lists of long operation sequences in some non-human-readable format (e.g.
    pickle).  CompressedOpString objects *cannot* be used in place of
    OpString objects within pyGSTi, and so are *not* useful when manipulating
    and running algorithms which use operation sequences.
    """

    def __init__(self, circuit, minLenToCompress=20, maxPeriodToLookFor=20):
        """
        Create a new CompressedOpString object

        Parameters
        ----------
        circuit : OpString
            The operation sequence object which is compressed to create
            a new CompressedOpString object.

        minLenToCompress : int, optional
            The minimum length string to compress.  If len(circuit)
            is less than this amount its tuple is returned.

        maxPeriodToLookFor : int, optional
            The maximum period length to use when searching for periodic
            structure within circuit.  Larger values mean the method
            takes more time but could result in better compressing.
        """
        if not isinstance(circuit, OpString):
            raise ValueError("CompressedGateStrings can only be created from existing OpString objects")
        self._tup = CompressedOpString.compress_op_label_tuple(
            circuit._tup, minLenToCompress, maxPeriodToLookFor)
        self.str = circuit._str

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state_dict):
        for k, v in state_dict.items():
            if k == 'tup':                    # backwards compatibility
                self._tup = state_dict['tup'] # backwards compatibility
            else:
                self.__dict__[k] = v

    def expand(self):
        """
        Expands this compressed operation sequence into a OpString object.

        Returns
        -------
        OpString
        """
        tup = CompressedOpString.expand_op_label_tuple(self._tup)
        return OpString(tup, self.str, bCheck=False)

    @staticmethod
    def _getNumPeriods(circuit, periodLen):
        n = 0
        if len(circuit) < periodLen: return 0
        while circuit[0:periodLen] == circuit[n*periodLen:(n+1)*periodLen]:
            n += 1
        return n


    @staticmethod
    def compress_op_label_tuple(circuit, minLenToCompress=20, maxPeriodToLookFor=20):
        """
        Compress a operation sequence.  The result is tuple with a special compressed-
        gate-string form form that is not useable by other GST methods but is
        typically shorter (especially for long operation sequences with a repetative
        structure) than the original operation sequence tuple.

        Parameters
        ----------
        circuit : tuple of operation labels or OpString
            The operation sequence to compress.

        minLenToCompress : int, optional
            The minimum length string to compress.  If len(circuit)
            is less than this amount its tuple is returned.

        maxPeriodToLookFor : int, optional
            The maximum period length to use when searching for periodic
            structure within circuit.  Larger values mean the method
            takes more time but could result in better compressing.

        Returns
        -------
        tuple
            The compressed (or raw) operation sequence tuple.
        """
        circuit = tuple(circuit) # converts from OpString or list to tuple if needed
        L = len(circuit)
        if L < minLenToCompress: return tuple(circuit)
        compressed = ["CCC"] #list for appending, then make into tuple at the end
        start = 0
        while start < L:
            #print "Start = ",start
            score = _np.zeros( maxPeriodToLookFor+1, 'd' )
            numperiods = _np.zeros( maxPeriodToLookFor+1, _np.int64 )
            for periodLen in range(1,maxPeriodToLookFor+1):
                n = CompressedOpString._getNumPeriods( circuit[start:], periodLen )
                if n == 0: score[periodLen] = 0
                elif n == 1: score[periodLen] = 4.1/periodLen
                else: score[periodLen] = _np.sqrt(periodLen)*n
                numperiods[periodLen] = n
            bestPeriodLen = _np.argmax(score)
            n = numperiods[bestPeriodLen]
            bestPeriod = circuit[start:start+bestPeriodLen]
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
    def expand_op_label_tuple(compressedCircuit):
        """
        Expand a compressed tuple created with compress_op_label_tuple(...)
        into a tuple of operation labels.

        Parameters
        ----------
        compressedCircuit : tuple
            a tuple in the compressed form created by
            compress_op_label_tuple(...).

        Returns
        -------
        tuple
            A tuple of operation labels specifying the uncompressed operation sequence.
        """

        if len(compressedCircuit) == 0: return ()
        if compressedCircuit[0] != "CCC": return compressedCircuit
        expandedString = []
        for (period,n) in compressedCircuit[1:]:
            expandedString += period*n
        return tuple(expandedString)



#Now tested in unit tests
#if __name__ == "__main__":
#    wgstr = WeightedOpString(('Gx',), weight=0.5)
#    opstr = OpString(('Gx',) )
#    print ((opstr + wgstr)*2).weight
#    print (wgstr + opstr).weight

