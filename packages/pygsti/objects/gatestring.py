""" Defines the GateString class and derived classes which represent gate strings."""


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

        if tupleOfGateLabels is None or bCheck:
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
        return GateString(self.tup + x.tup, s)

    def __mul__(self,x):
        assert(isinstance(x,int) and x >= 0)
        if x > 1: s = "(%s)^%d" % (self.str,x)
        elif x == 1: s = "(%s)" % self.str
        else: s = "{}"
        return GateString(self.tup * x, s)

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
        return GateString( self.tup, self.str)

    def __deepcopy__(self):
        return GateString( self.tup, self.str)

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

    def __init__(self,tupleOfGateLabels, stringRepresentation=None, weight=1.0, bCheck=False):
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
        return WeightedGateString( tmp.tup, tmp.str, self.weight + x_weight ) #add weights

    def __radd__(self,x):
        if isinstance(x, GateString):
            tmp = x.__add__(self)
            x_weight = x.weight if type(x) == WeightedGateString else 0.0
            return WeightedGateString( tmp.tup, tmp.str, x_weight + self.weight )
        raise ValueError("Can only add GateStrings objects to other GateString objects")

    def __mul__(self,x):
        tmp = super(WeightedGateString,self).__mul__(x)
        return WeightedGateString(tmp.tup, tmp.str, self.weight) #keep weight

    def __copy__(self):
        return WeightedGateString( self.tup, self.str, self.weight )

    def __deepcopy__(self):
        return WeightedGateString( self.tup, self.str, self.weight )

    def __getitem__(self, key):
        if isinstance( key, slice ):
            return WeightedGateString( self.tup.__getitem__(key), None, self.weight )
        return self.tup.__getitem__(key)


#Conversion routines for evalTree usage -- TODO: make these member functions
def gateStringToPythonString(gateString,gateLabels):
    """
    Convert a gate string into a python string, where each gate label is
    represented as a **single** character, starting with 'A' and contining
    down the alphabet.  This can be useful for processing gate strings 
    using python's string tools (regex in particular).

    Parameters
    ----------
    gateString : tuple or GateString
        the gate label sequence that is converted to a string.
    
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
    return "".join([ translateDict[gateLabel] for gateLabel in gateString ])

def pythonStringToGateString(pythonString,gateLabels):
    """
    Convert a python string into a gate string, where each gate label is
    represented as a **single** character, starting with 'A' and contining
    down the alphabet.  This performs the inverse of gateStringToPythonString(...).

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
        The decoded python string as a GateString object (essentially 
        a tuple of gate labels).

    Examples
    --------
        "AABA" => ('Gx','Gx','Gy','Gx')
    """
    assert(len(gateLabels) < 26) #Complain if we go beyond 'Z'
    translateDict = {}; c = 'A'
    for gateLabel in gateLabels:
        translateDict[c] = gateLabel
        c = chr(ord(c) + 1)
    return _gs.GateString( tuple([ translateDict[c] for c in pythonString ]) )


if __name__ == "__main__":
    wgs = WeightedGateString(('Gx',), weight=0.5)
    gs = GateString(('Gx',) )
    print ((gs + wgs)*2).weight
    print (wgs + gs).weight

