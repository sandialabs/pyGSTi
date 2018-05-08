""" Defines the Label class """
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

import numbers as _numbers

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
  
    def __init__(self,name,stateSpaceLabels=None):
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
        
        #Type checking
        assert(isstr(name)), "`name` must be a string, but it's '%s'" % str(name)
        if stateSpaceLabels is not None:
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
            self.sslbls = tuple(integerized_sslbls)
        else:
            self.sslbls = None
        self.name = name

    @property
    def qubits(self): #Used in Circuit
        """An alias for sslbls, since commonly these are just qubit indices"""
        return self.sslbls

    @property
    def number_of_qubits(self): #Used in Circuit
        return len(self.sslbls) if (self.sslbls is not None) else None

    def __iter__(self):
        """ Iterate over the name + state space labels """
        # Note: tuple(.) uses __iter__ to construct tuple rep.
        yield self.name
        if self.sslbls is not None:
            for ssl in self.sslbls:
                yield ssl
    
    def __str__(self):
        """
        Defines how a Gate is printed out, e.g. Gx:0 or Gcnot:1:2
        """
        s = str(self.name)
        if self.sslbls: #test for None and len == 0
            s += ":" + ":".join(map(str,self.sslbls))
        return s

    def __repr__(self):
        """
        Defines how a Gate is printed out, e.g. Gx:0 or Gcnot:1:2
        """
        return "Label" + str(tuple(self))
    
    def __add__(self, s):
        if isstr(s):
            return Label(self.name + s, self.sslbls)
        else:
            raise NotImplementedError("Cannot add %s to a Label" % str(type(s)))
    
    def __eq__(self,other):
        """
        Defines equality between gates, so that they are equal if their values
        are equal.
        """
        if isstr(other):
            if self.sslbls: return False # tests for None and len > 0
            return self.name == other
        return self.name == other.name and self.sslbls == other.sslbls # ok to compare None

    def __lt__(self,x):
        return tuple(self) < tuple(x)

    def __gt__(self,x):
        return tuple(self) > tuple(x)

    def __hash__(self):        
        return hash(tuple(self))
