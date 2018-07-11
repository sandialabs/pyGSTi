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
        item acts (often just qubit indices).
        
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
        
        if isinstance(name, Label) and stateSpaceLabels is None:
            return name #Note: Labels are immutable, so no need to copy
        
        if not isstr(name) and stateSpaceLabels is None \
           and isinstance(name, (tuple,list)):

            #We're being asked to initialize from a non-string with no
            # stateSpaceLabels explicitly given.  `name` could either be:
            # 1) a (name, ssl0, ssl1, ...) tuple -> LabelTup
            # 2) a (subLabel1_tup, subLabel2_tup, ...) tuple -> LabelTupTup if
            #     length > 1 otherwise just initialize from subLabel1_tup.
            # Note: subLabelX_tup could also be identified as a Label object
            #       (even a LabelStr)
            
            if isinstance(name[0], (tuple,list,Label)): 
                if len(name) > 1: return LabelTupTup(name)
                else: return Label(name[0])
            else:
                #Case when stateSpaceLabels are given after name in a single tuple
                stateSpaceLabels = tuple(name[1:])
                name = name[0]

        if stateSpaceLabels is None or stateSpaceLabels in ( (), (None,) ):
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
        item acts (often just qubit indices).
        
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
    def components(self):
        return (self,) # just a single "sub-label" component
        
    @property
    def qubits(self): #Used in Circuit
        """An alias for sslbls, since commonly these are just qubit indices"""
        return self.sslbls

    @property
    def number_of_qubits(self): #Used in Circuit
        return len(self.sslbls) if (self.sslbls is not None) else None

    def has_prefix(self, prefix, typ="all"):
        """
        Whether this label has the given `prefix`.  Usually used to test whether
        the label names a given type.

        Parameters
        ----------
        prefix : str
            The prefix to check for.

        typ : {"any","all"}
            Whether, when there are multiple parts to the label, the prefix
            must occur in any or all of the parts.

        Returns
        -------
        bool
        """
        return self.name.startswith(prefix)
        
    def map_state_space_labels(self, mapper):
        """
        Return a copy of this Label with all of the state-space-labels
        (often just qubit labels) updated according to a mapping function.

        For example, calling this function with `mapper = {0: 1, 1: 3}`
        on the Label "Gcnot:0:1" would return "Gcnot:1:3".

        Parameters
        ----------
        mapper : dict or function
            A dictionary whose keys are the existing state-space-label values
            and whose value are the new labels, or a function which takes a
            single (existing label) argument and returns a new label.

        Returns
        -------
        Label
        """
        if isinstance(mapper, dict):
            mapped_sslbls = [ mapper[sslbl] for sslbl in self.sslbls ]
        else: # assume mapper is callable
            mapped_sslbls = [ mapper(sslbl) for sslbl in self.sslbls ]
        return Label(self.name, mapped_sslbls)


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
        return self.__reduce__()

    def __reduce__(self):
        # Need to tell serialization logic how to create a new Label since it's derived
        # from the immutable tuple type (so cannot have its state set after creation)
        return (LabelTup, (self[0],self[1:]), None)

    def tonative(self):
        """ Returns this label as native python types.  Useful for 
            faster serialization.
        """
        return tuple(self)

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
    def components(self):
        return (self,) # just a single "sub-label" component
        
    @property
    def qubits(self): #Used in Circuit
        """An alias for sslbls, since commonly these are just qubit indices"""
        return None

    @property
    def number_of_qubits(self): #Used in Circuit
        return None

    def has_prefix(self, prefix, typ="all"):
        """
        Whether this label has the given `prefix`.  Usually used to test whether
        the label names a given type.

        Parameters
        ----------
        prefix : str
            The prefix to check for.

        typ : {"any","all"}
            Whether, when there are multiple parts to the label, the prefix
            must occur in any or all of the parts.

        Returns
        -------
        bool
        """
        return self.startswith(prefix)

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
        return self.__reduce__()

    def __reduce__(self):
        # Need to tell serialization logic how to create a new Label since it's derived
        # from the immutable tuple type (so cannot have its state set after creation)
        return (LabelStr, (str(self),), None)

    def tonative(self):
        """ Returns this label as native python types.  Useful for 
            faster serialization.
        """
        return str(self)

    __hash__ = str.__hash__ # this is why we derive from tuple - using the
                              # native tuple.__hash__ directly == speed boost

class LabelTupTup(Label,tuple):
    """ 
    A label consisting of a *tuple* of (string, state-space-labels) tuples
    which labels a parallel layer/level of a circuit.
    """
  
    def __new__(cls,tupOfTups):
        """
        Creates a new GateSet-item label, which is a tuple of tuples of simple
        string labels and tuples specifying the part of the Hilbert space upon
        which that item acts (often just qubit indices).
        
        Parameters
        ----------
        tupOfTups : tuple
            The item data - a tuple of (string, state-space-labels) tuples
            which labels a parallel layer/level of a circuit.
        """
        tupOfLabels = tuple((Label(tup) for tup in tupOfTups)) # Note: tup can also be a Label obj
        return tuple.__new__(cls, tupOfLabels) # creates a LabelTupTup object using tuple's __new__


    @property
    def name(self):
        assert(False),"TODO - something intelligent here..." # no real "name" for a compound label...?
        # return self[0]

    @property
    def sslbls(self):
        # Note: if any component has sslbls == None, which signifies operating
        # on *all* qubits, then this label is on *all* qubites
        s = set()
        for lbl in self:
            if lbl.sslbls is None: return None 
            s.update(lbl.sslbls)
        return tuple(sorted(list(s)))

    @property
    def components(self):
        return self # self is a tuple of "sub-label" components
        
    @property
    def qubits(self): #Used in Circuit
        """An alias for sslbls, since commonly these are just qubit indices"""
        return self.sslbls

    @property
    def number_of_qubits(self): #Used in Circuit
        return len(self.sslbls) if (self.sslbls is not None) else None

    def has_prefix(self, prefix, typ="all"):
        """
        Whether this label has the given `prefix`.  Usually used to test whether
        the label names a given type.

        Parameters
        ----------
        prefix : str
            The prefix to check for.

        typ : {"any","all"}
            Whether, when there are multiple parts to the label, the prefix
            must occur in any or all of the parts.

        Returns
        -------
        bool
        """
        if typ == "all":
            return all([lbl.startswith(prefix) for lbl in self])
        elif typ == "any":
            return any([lbl.startswith(prefix) for lbl in self])
        else: raise ValueError("Invalid `typ` arg: %s" % str(typ))

    
    def map_state_space_labels(self, mapper):
        """
        Return a copy of this Label with all of the state-space-labels
        (often just qubit labels) updated according to a mapping function.

        For example, calling this function with `mapper = {0: 1, 1: 3}`
        on the Label "Gcnot:0:1" would return "Gcnot:1:3".

        Parameters
        ----------
        mapper : dict or function
            A dictionary whose keys are the existing state-space-label values
            and whose value are the new labels, or a function which takes a
            single (existing label) argument and returns a new label.

        Returns
        -------
        Label
        """
        return LabelTupTup( tuple((lbl.map_state_space_labels(mapper) for lbl in self)) )

    def __str__(self):
        """
        Defines how a Label is printed out, e.g. Gx:0 or Gcnot:1:2
        """
        return "[" + "".join([str(lbl) for lbl in self]) + "]"

    def __repr__(self):
        return "Label[" + str(self) + "]"
    
    def __add__(self, s):
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
        return self.__reduce__()

    def __reduce__(self):
        # Need to tell serialization logic how to create a new Label since it's derived
        # from the immutable tuple type (so cannot have its state set after creation)
        return (LabelTupTup, (self[:],), None)

    def tonative(self):
        """ Returns this label as native python types.  Useful for 
            faster serialization.
        """
        return tuple((x.tonative() for x in self))


    __hash__ = tuple.__hash__ # this is why we derive from tuple - using the
                              # native tuple.__hash__ directly == speed boost
