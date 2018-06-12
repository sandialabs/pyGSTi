""" Defines the Label class """
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

import numbers as _numbers

import os,inspect
debug_record = {}

try:  basestring
except NameError: basestring = str

def isstr(x): #Duplicates isstr from compattools! (b/c can't import!)
    """ Return whether `x` has a string type """
    return isinstance(x, basestring)


class Label(object):
    """ 
    A label consisting of a string along with a tuple of 
    integers or sector-names specifying which qubits, or
    more generally, parts of the Hilbert space that is
    acted upon by an object so-labeled.
    """

    # this is just an abstract base class for isinstance checking.
    # actual labels will either be LabelTup or LabelStr instances,
    # depending on whether the tuple of sector names exists or not.
    # (the reason for separate classes is for hashing speed)

    def __new__(cls,name,stateSpaceLabels=None):
        """
        Creates a new GateSet-item label, which is divided into a simple string
        label and a tuple specifying the part of the Hilbert space upon which the
        item acts (ofted just qubit indices).
        
        Parameters
        ----------
        name : str
            The item name. E.g., 'CNOT' or 'H'.
            
        stateSpaceLabels : list or tuple, optional
            A list or tuple that identifies which sectors/parts of the Hilbert
            space is acted upon.  In many cases, this is a list of integers
            specifying the qubits on which a gate acts, when the ordering in the
            list defines the 'direction' of the gate.  If something other than 
            a list or tuple is passed, a single-element tuple is created
            containing the passed object.
        """

        #Case when stateSpaceLabels are given after name in a single tuple
        if not isstr(name) and stateSpaceLabels is None \
           and isinstance(name, (tuple,list)):
            stateSpaceLabels = tuple(name[1:])
            name = name[0]

        if stateSpaceLabels is None:
            return LabelStr(name)
        else:
            return LabelTup(name, stateSpaceLabels)


class LabelTup(Label,tuple):
    """ 
    A label consisting of a string along with a tuple of 
    integers or sector-names specifying which qubits, or
    more generally, parts of the Hilbert space that is
    acted upon by an object so-labeled.
    """
  
    def __new__(cls,name,stateSpaceLabels):
        """
        Creates a new GateSet-item label, which is divided into a simple string
        label and a tuple specifying the part of the Hilbert space upon which the
        item acts (ofted just qubit indices).
        
        Parameters
        ----------
        name : str
            The item name. E.g., 'CNOT' or 'H'.
            
        stateSpaceLabels : list or tuple
            A list or tuple that identifies which sectors/parts of the Hilbert
            space is acted upon.  In many cases, this is a list of integers
            specifying the qubits on which a gate acts, when the ordering in the
            list defines the 'direction' of the gate.  If something other than 
            a list or tuple is passed, a single-element tuple is created
            containing the passed object.
        """
        
        #Type checking
        assert(isstr(name)), "`name` must be a string, but it's '%s'" % str(name)
        assert(stateSpaceLabels is not None), "LabelTup must be initialized with non-None state-space labels"
        if not isinstance(stateSpaceLabels, (tuple,list)):
            stateSpaceLabels = (stateSpaceLabels,)
        for ssl in stateSpaceLabels:
            assert(isstr(ssl) or isinstance(ssl, _numbers.Integral)), \
                "State space label '%s' must be a string or integer!" % str(ssl)

        #Try to convert integer-strings to ints (for parsing from files...)
        integerized_sslbls = []
        for ssl in stateSpaceLabels:
            try: integerized_sslbls.append( int(ssl) )
            except: integerized_sslbls.append( ssl )
            
        # Regardless of whether the input is a list, tuple, or int, the state space labels
        # (qubits) that the item/gate acts on are stored as a tuple (because tuples are immutable).
        sslbls = tuple(integerized_sslbls)
        tup = (name,) + sslbls

        return tuple.__new__(cls, tup) # creates a LabelTup object using tuple's __new__


    @property
    def name(self):
        return self[0]

    @property
    def sslbls(self):
        if len(self) > 1:
            return self[1:]
        else: return None
        
    @property
    def qubits(self): #Used in Circuit
        """An alias for sslbls, since commonly these are just qubit indices"""
        return self.sslbls

    @property
    def number_of_qubits(self): #Used in Circuit
        return len(self.sslbls) if (self.sslbls is not None) else None

    #OLD
    #def __iter__(self):
    #    return self.tup.__iter__()

    #OLD
    #def __iter__(self):
    #    """ Iterate over the name + state space labels """
    #    # Note: tuple(.) uses __iter__ to construct tuple rep.
    #    yield self.name
    #    if self.sslbls is not None:
    #        for ssl in self.sslbls:
    #            yield ssl
    
    def __str__(self):
        """
        Defines how a Label is printed out, e.g. Gx:0 or Gcnot:1:2
        """
        #caller = inspect.getframeinfo(inspect.currentframe().f_back)
        #ky = "%s:%s:%d" % (caller[2],os.path.basename(caller[0]),caller[1])
        #debug_record[ky] = debug_record.get(ky, 0) + 1
        s = str(self.name)
        if self.sslbls: #test for None and len == 0
            s += ":" + ":".join(map(str,self.sslbls))
        return s

    def __repr__(self):
        return "Label[" + str(self) + "]"
    
    def __add__(self, s):
        if isstr(s):
            return LabelTup(self.name + s, self.sslbls)
        else:
            raise NotImplementedError("Cannot add %s to a Label" % str(type(s)))
    
    def __eq__(self,other):
        """
        Defines equality between gates, so that they are equal if their values
        are equal.
        """
        #Unnecessary now that we have a separate LabelStr
        #if isstr(other):
        #    if self.sslbls: return False # tests for None and len > 0
        #    return self.name == other

        return tuple.__eq__(self,other)
        #OLD return self.name == other.name and self.sslbls == other.sslbls # ok to compare None

    def __lt__(self,x):
        return tuple.__lt__(self,tuple(x))

    def __gt__(self,x):
        return tuple.__gt__(self,tuple(x))

    def __pygsti_reduce__(self):
        # Need to tell serialization logic how to create a new Label since it's derived
        # from the immutable tuple type (so cannot have its state set after creation)
        return (LabelTup, (self[0],self[1:]), None)

    __hash__ = tuple.__hash__ # this is why we derive from tuple - using the
                              # native tuple.__hash__ directly == speed boost



class LabelStr(Label,str):
    """ 
    A Label for the special case when only a name is present (no
    state-space-labels).  We create this as a separate class
    so that we can use the string hash function in a 
    "hardcoded" way - if we put switching logic in __hash__
    the hashing gets *much* slower.
    """
  
    def __new__(cls,name):
        """
        Creates a new GateSet-item label, which is just a simple string label.
        
        Parameters
        ----------
        name : str
            The item name. E.g., 'CNOT' or 'H'.
        """

        #Type checking
        assert(isstr(name)), "`name` must be a string, but it's '%s'" % str(name)
        return str.__new__(cls, name)

    @property
    def name(self):
        return str(self)

    @property
    def sslbls(self):
        return None
        
    @property
    def qubits(self): #Used in Circuit
        """An alias for sslbls, since commonly these are just qubit indices"""
        return ()

    @property
    def number_of_qubits(self): #Used in Circuit
        return None
    
    def __str__(self):
        return self[:] # converts to a normal str

    def __repr__(self):
        return "Label{" + str(self) + "}"
    
    def __add__(self, s):
        if isstr(s):
            return LabelStr(self.name + str(s))
        else:
            raise NotImplementedError("Cannot add %s to a Label" % str(type(s)))
    
    def __eq__(self,other):
        """
        Defines equality between gates, so that they are equal if their values
        are equal.
        """
        return str.__eq__(self,other)

    def __lt__(self,x):
        return str.__lt__(self,str(x))

    def __gt__(self,x):
        return str.__gt__(self,str(x))
    
    def __pygsti_reduce__(self):
        # Need to tell serialization logic how to create a new Label since it's derived
        # from the immutable tuple type (so cannot have its state set after creation)
        return (LabelStr, (str(self),), None)

    __hash__ = str.__hash__ # this is why we derive from tuple - using the
                              # native tuple.__hash__ directly == speed boost

                              
#OLD
#    def __hash__(self):
#        #caller = inspect.getframeinfo(inspect.currentframe().f_back)
#        #ky = "%s:%s:%d" % (caller[2],os.path.basename(caller[0]),caller[1])
#        #debug_record[ky] = debug_record.get(ky, 0) + 1
#        assert(False)
#        return hash(self.tup)
