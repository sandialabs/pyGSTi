""" Defines the Label class """
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

import numbers as _numbers
import sys as _sys
import itertools as _itertools

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

    def __new__(cls,name,stateSpaceLabels=None,timestamp=None):
        """
        Creates a new Model-item label, which is divided into a simple string
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

        timestamp : float
            The time at which this label occurs (can be relative or absolute)
        """
        if isinstance(name, Label) and stateSpaceLabels is None:
            return name #Note: Labels are immutable, so no need to copy
        
        if not isstr(name) and stateSpaceLabels is None \
           and isinstance(name, (tuple,list)):

            #We're being asked to initialize from a non-string with no
            # stateSpaceLabels explicitly given.  `name` could either be:
            # 0) an empty tuple: () -> LabelTupTup with *no* subLabels.
            # 1) a (name, ssl0, ssl1, ...) tuple -> LabelTup
            # 2) a (subLabel1_tup, subLabel2_tup, ...) tuple -> LabelTupTup if
            #     length > 1 otherwise just initialize from subLabel1_tup.
            # Note: subLabelX_tup could also be identified as a Label object
            #       (even a LabelStr)
            
            if len(name) == 0:
                if timestamp is None: return LabelTupTup( () )
                else: return TimestampedLabelTupTup( (), timestamp )
            elif isinstance(name[0], (tuple,list,Label)): 
                if len(name) > 1:
                    if timestamp is None: return LabelTupTup(name)
                    else: return TimestampedLabelTupTup(name, timestamp)
                else:
                    return Label(name[0], timestamp=timestamp)
            else:
                #Case when stateSpaceLabels are given after name in a single tuple
                stateSpaceLabels = tuple(name[1:])
                name = name[0]
                timestamp = None # no way to specify timestamped labels this way (yet)

        if stateSpaceLabels is None or stateSpaceLabels in ( (), (None,) ):
            if timestamp is None: return LabelStr(name)
            else: return TimestampedLabelTup(name,(),timestamp) # just use empty sslbls
        else:
            if timestamp is None: return LabelTup(name, stateSpaceLabels)
            else: return TimestampedLabelTup(name, stateSpaceLabels, timestamp)

    def depth(self):
        return 1 #most labels are depth=1

    @property
    def reps(self):
        return 1 # most labels have only reps==1

    def expand_subcircuits(self):
        """TODO: docstring - returns a list/tuple of labels """
        return (self,) # most labels just expand to themselves



class LabelTup(Label,tuple):
    """ 
    A label consisting of a string along with a tuple of 
    integers or sector-names specifying which qubits, or
    more generally, parts of the Hilbert space that is
    acted upon by an object so-labeled.
    """
  
    def __new__(cls,name,stateSpaceLabels):
        """
        Creates a new Model-item label, which is divided into a simple string
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
    def time(self):
        return None

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

    def replacename(self,oldname,newname):
        """ Returns a label with `oldname` replaced by `newname`."""
        return LabelTup(newname,self.sslbls) if (self.name == oldname) else self

    def issimple(self):
        """ Whether this is a "simple" (opaque w/a true name, from a
            circuit perspective) label or not """
        return True


    __hash__ = tuple.__hash__ # this is why we derive from tuple - using the
                              # native tuple.__hash__ directly == speed boost


# We want LabelStr to act like the string literal type (not
# 'str' when we import unicode_literals above)
strlittype = str if _sys.version_info >= (3, 0) else unicode # (a *native* python type)

class LabelStr(Label,strlittype):
    """ 
    A Label for the special case when only a name is present (no
    state-space-labels).  We create this as a separate class
    so that we can use the string hash function in a 
    "hardcoded" way - if we put switching logic in __hash__
    the hashing gets *much* slower.
    """
  
    def __new__(cls,name):
        """
        Creates a new Model-item label, which is just a simple string label.
        
        Parameters
        ----------
        name : str
            The item name. E.g., 'CNOT' or 'H'.
        """

        #Type checking
        assert(isstr(name)), "`name` must be a string, but it's '%s'" % str(name)
        return strlittype.__new__(cls, name)

    @property
    def name(self):
        return strlittype(self)

    @property
    def sslbls(self):
        return None

    @property
    def time(self):
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
        return "Label{" + strlittype(self) + "}"
    
    def __add__(self, s):
        if isstr(s):
            return LabelStr(self.name + strlittype(s))
        else:
            raise NotImplementedError("Cannot add %s to a Label" % str(type(s)))
    
    def __eq__(self,other):
        """
        Defines equality between gates, so that they are equal if their values
        are equal.
        """
        return strlittype.__eq__(self,other)

    def __lt__(self,x):
        return strlittype.__lt__(self,strlittype(x))

    def __gt__(self,x):
        return strlittype.__gt__(self,strlittype(x))
    
    def __pygsti_reduce__(self):
        return self.__reduce__()

    def __reduce__(self):
        # Need to tell serialization logic how to create a new Label since it's derived
        # from the immutable tuple type (so cannot have its state set after creation)
        return (LabelStr, (strlittype(self),), None)

    def tonative(self):
        """ Returns this label as native python types.  Useful for 
            faster serialization.
        """
        return strlittype(self)

    def replacename(self,oldname,newname):
        """ Returns a label with `oldname` replaced by `newname`."""
        return LabelStr(newname) if (self.name == oldname) else self

    def issimple(self):
        """ Whether this is a "simple" (opaque w/a true name, from a
            circuit perspective) label or not """
        return True
    

    __hash__ = strlittype.__hash__ # this is why we derive from tuple - using the
                              # native tuple.__hash__ directly == speed boost

class LabelTupTup(Label,tuple):
    """ 
    A label consisting of a *tuple* of (string, state-space-labels) tuples
    which labels a parallel layer/level of a circuit.
    """
  
    def __new__(cls,tupOfTups):
        """
        Creates a new Model-item label, which is a tuple of tuples of simple
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
        # TODO - something intelligent here?
        # no real "name" for a compound label... but want it to be a string so
        # users can use .startswith, etc.
        return "COMPOUND"

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
    def time(self):
        t = None
        for lbl in self:
            if lbl.time is not None: t = lbl.time
            else: assert(lbl.time == t), "Components occur at different times!"
        #FUTURE: could cache this value since it shouldn't change?
        return t

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
            return all([lbl.has_prefix(prefix) for lbl in self])
        elif typ == "any":
            return any([lbl.has_prefix(prefix) for lbl in self])
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

    def replacename(self,oldname,newname):
        """ Returns a label with `oldname` replaced by `newname`."""
        return LabelTupTup( tuple((x.replacename(oldname,newname) for x in self)) )

    def issimple(self):
        """ Whether this is a "simple" (opaque w/a true name, from a
            circuit perspective) label or not """
        return False

    def depth(self):
        if len(self.components) == 0: return 1 # still depth 1 even if empty
        return max([x.depth() for x in self.components])

    def expand_subcircuits(self):
        """TODO: docstring - returns a list/tuple of labels """
        ret = []
        expanded_comps = [ x.expand_subcircuits() for x in self.components ]
        
        #DEBUG TODO REMOVE
        #print("DB: expaned comps:") 
        #for i,x in enumerate(expanded_comps):
        #    print(i,": ",x)
        
        for i in range(self.depth()): # depth == # of layers when expanded
            ec = []
            for expanded_comp in expanded_comps:
                if i < len(expanded_comp):
                    ec.extend( expanded_comp[i].components ) # .components = vertical expansion
            #assert(len(ec) > 0), "Logic error!" #this is ok (e.g. an idle subcircuit)
            ret.append( LabelTupTup(ec) )
        return tuple(ret)



    __hash__ = tuple.__hash__ # this is why we derive from tuple - using the
                              # native tuple.__hash__ directly == speed boost

class CircuitLabel(Label,tuple):
    def __new__(cls,name,tupOfTups,stateSpaceLabels,reps=1): # timestamp??
        """
        Creates a new Model-item label, which is a tuple of tuples of simple
        string labels and tuples specifying the part of the Hilbert space upon
        which that item acts (often just qubit indices).

        TODO: docstring!
        
        Parameters
        ----------
        tupOfTups : tuple
            The item data - a tuple of (string, state-space-labels) tuples
            which labels a parallel layer/level of a circuit.
        """
        #if name is None: name = '' # backward compatibility (temporary - TODO REMOVE)
        assert(isinstance(reps, _numbers.Integral) and isstr(name)), "Invalid name or reps: %s %s" % (str(name),str(reps))
        tupOfLabels = tuple((Label(tup) for tup in tupOfTups)) # Note: tup can also be a Label obj
        return tuple.__new__(cls, (name,stateSpaceLabels,reps) + tupOfLabels) # creates a CircuitLabel object using tuple's __new__


    @property
    def name(self):
        return self[0]

    @property
    def sslbls(self):
        return self[1]

    @property
    def reps(self):
        return self[2]

    @property
    def time(self):
        raise NotImplementedError("TODO!")
    
    @property
    def components(self):
        return self[3:]
        
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
        return CircuitLabel(self.name, tuple((lbl.map_state_space_labels(mapper) for lbl in self.components)), mapped_sslbls, self[2])

    def __str__(self):
        """
        Defines how a Label is printed out, e.g. Gx:0 or Gcnot:1:2
        """
        if len(self.name) > 0:
            s = self.name
        else:
            s = "".join([str(lbl) for lbl in self.components])
            if len(self.components) > 1:
                s = "(" + s + ")" # add parenthesis
        if self[2] != 1: s += "^%d" % self[2]
        return s

    def __repr__(self):
        return "CircuitLabel[" + str(self) + "]"
    
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
        return (CircuitLabel, (self[0],self[3:],self[1],self[2]), None)

    def tonative(self):
        """ Returns this label as native python types.  Useful for 
            faster serialization.
        """
        return self[0:3] + tuple((x.tonative() for x in self.components))

    def replacename(self,oldname,newname):
        """ Returns a label with `oldname` replaced by `newname`."""
        return CircuitLabel(self.name, tuple((x.replacename(oldname,newname) for x in self.components)),self.sslbls, self[2])

    def issimple(self):
        """ Whether this is a "simple" (opaque w/a true name, from a
            circuit perspective) label or not """
        return True # still true - even though can have components!

    def depth(self):
        return sum([x.depth() for x in self.components])*self.reps

    def expand_subcircuits(self):
        """TODO: docstring - returns a list/tuple of labels """
        #REMOVE print("Expanding subcircuit components: ",self.components)
        #REMOVE print(" --> ",[ x.expand_subcircuits() for x in self.components ])
        return tuple(_itertools.chain(*[x.expand_subcircuits() for x in self.components]))*self.reps

    __hash__ = tuple.__hash__ # this is why we derive from tuple - using the
                              # native tuple.__hash__ directly == speed boost



#class NamedLabelTupTup(Label,tuple):
#    def __new__(cls,name,tupOfTups):
#        pass


class TimestampedLabelTup(Label,tuple):
    """ 
    A label consisting of a string along with a tuple of 
    integers or sector-names specifying which qubits, or
    more generally, parts of the Hilbert space that is
    acted upon by an object so-labeled.
    """
  
    def __new__(cls,name,stateSpaceLabels,timestamp):
        """
        Creates a new Model-item label, which is divided into a simple string
        label, a tuple specifying the part of the Hilbert space upon which the
        item acts (often just qubit indices), and a timestamp.
        
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

        timestamp : float
            The time at which this label occurs (can be relative or absolute)
        """
        
        #Type checking
        assert(isstr(name)), "`name` must be a string, but it's '%s'" % str(name)
        assert(stateSpaceLabels is not None), "LabelTup must be initialized with non-None state-space labels"
        if not isinstance(stateSpaceLabels, (tuple,list)):
            stateSpaceLabels = (stateSpaceLabels,)
        for ssl in stateSpaceLabels:
            assert(isstr(ssl) or isinstance(ssl, _numbers.Integral)), \
                "State space label '%s' must be a string or integer!" % str(ssl)
        assert(isinstance(timestamp,float)), "`timestamp` must be a floating point value"

        #Try to convert integer-strings to ints (for parsing from files...)
        integerized_sslbls = []
        for ssl in stateSpaceLabels:
            try: integerized_sslbls.append( int(ssl) )
            except: integerized_sslbls.append( ssl )
            
        # Regardless of whether the input is a list, tuple, or int, the state space labels
        # (qubits) that the item/gate acts on are stored as a tuple (because tuples are immutable).
        sslbls = tuple(integerized_sslbls)
        tup = (timestamp,name) + sslbls

        return tuple.__new__(cls, tup) # creates a LabelTup object using tuple's __new__


    @property
    def name(self):
        return self[1]

    @property
    def sslbls(self):
        if len(self) > 2:
            return self[2:]
        else: return None

    @property
    def time(self):
        return self[0]


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
        return Label(self.name, mapped_sslbls, self.time)
    
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
        s += "!%f" % self.time 
        return s

    def __repr__(self):
        return "Label[" + str(self) + "]"
    
    def __add__(self, s):
        if isstr(s):
            return LabelTup(self.name + s, self.sslbls, self.time)
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
        return (TimestampedLabelTup, (self[1],self[2:],self[0]), None)

    def tonative(self):
        """ Returns this label as native python types.  Useful for 
            faster serialization.
        """
        return tuple(self)

    def replacename(self,oldname,newname):
        """ Returns a label with `oldname` replaced by `newname`."""
        return TimestampledLabelTup(newname,self.sslbls,self[0]) if (self.name == oldname) else self

    def issimple(self):
        """ Whether this is a "simple" (opaque w/a true name, from a
            circuit perspective) label or not """
        return True



    __hash__ = tuple.__hash__ # this is why we derive from tuple - using the
                              # native tuple.__hash__ directly == speed boost



class TimestampedLabelTupTup(Label,tuple):
    """ 
    A label consisting of a *tuple* of (string, state-space-labels) tuples
    which labels a parallel layer/level of a circuit at a single timestamp.
    """
  
    def __new__(cls,tupOfTups,timestamp):
        """
        Creates a new Model-item label, which is a tuple of tuples of simple
        string labels and tuples specifying the part of the Hilbert space upon
        which that item acts (often just qubit indices).
        
        Parameters
        ----------
        tupOfTups : tuple
            The item data - a tuple of (string, state-space-labels) tuples
            which labels a parallel layer/level of a circuit.

        timestamp : float
            The time at which this label occurs (can be relative or absolute)
        """
        
        tupOfLabels = (timestamp,) + tuple((Label(tup) for tup in tupOfTups)) # Note: tup can also be a Label obj
        assert(all([(timestamp == l.time or l.time is None) for l in tupOfLabels[1:]])), \
            "Component times do not match compound label time!"
        return tuple.__new__(cls, tupOfLabels) # creates a LabelTupTup object using tuple's __new__


    @property
    def name(self):
        # TODO - something intelligent here?
        # no real "name" for a compound label... but want it to be a string so
        # users can use .startswith, etc.
        return "COMPOUND"

    @property
    def sslbls(self):
        # Note: if any component has sslbls == None, which signifies operating
        # on *all* qubits, then this label is on *all* qubites
        s = set()
        for lbl in self[1:]:
            if lbl.sslbls is None: return None 
            s.update(lbl.sslbls)
        return tuple(sorted(list(s)))

    @property
    def time(self):
        return self[0]

    @property
    def components(self):
        return self[1:] # a tuple of "sub-label" components
        
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
            return all([lbl.has_prefix(prefix) for lbl in self[1:]])
        elif typ == "any":
            return any([lbl.has_prefix(prefix) for lbl in self[1:]])
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
        return LabelTupTup( tuple((lbl.map_state_space_labels(mapper) for lbl in self[1:])), self[0] )

    def __str__(self):
        """
        Defines how a Label is printed out, e.g. Gx:0 or Gcnot:1:2
        """
        return "[" + "".join([str(lbl) for lbl in self]) + "!%f" % self.time +  "]"

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
        return (TimestampedLabelTupTup, (self[1:],self[0]), None)

    def tonative(self):
        """ Returns this label as native python types.  Useful for 
            faster serialization.
        """
        return (self[0],) + tuple((x.tonative() for x in self[1:]))

    def replacename(self,oldname,newname):
        """ Returns a label with `oldname` replaced by `newname`."""
        return TimestampedLabelTupTup( tuple((x.replacename(oldname,newname) for x in self[1:])), self[0] )

    def issimple(self):
        """ Whether this is a "simple" (opaque w/a true name, from a
            circuit perspective) label or not """
        return False

    def depth(self):
        if len(self.components) == 0: return 1 # still depth 1 even if empty
        return max([x.depth() for x in self.components])


    __hash__ = tuple.__hash__ # this is why we derive from tuple - using the
                              # native tuple.__hash__ directly == speed boost
