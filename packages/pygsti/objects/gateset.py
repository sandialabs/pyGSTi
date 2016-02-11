#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation              
#    This Software is released under the GPL license detailed    
#    in the file "license.txt" in the top-level pyGSTi directory 
#*****************************************************************
""" Defines the GateSet class and supporting functionality."""

import itertools as _itertools
import warnings as _warnings
import numpy as _np
import numpy.linalg as _nla
import numpy.random as _rndm
import scipy as _scipy
import collections as _collections

from ..tools import matrixtools as _mt
from ..tools import basistools as _bt
from ..tools import gatetools as _gt
from ..tools import likelihoodfns as _lf
from ..tools import jamiolkowski as _jt

import evaltree as _evaltree
import gate as _gate


# Tolerace for matrix_rank when finding rank of a *normalized* projection matrix.
#  This is a legitimate tolerace since a normalized projection matrix should have
#  just 0 and 1 eigenvalues, and thus a tolerace << 1.0 will work well.
P_RANK_TOL = 1e-7 

# Smallness tolerances, used internally for conditional scaling required
# to control bulk products, their gradients, and their Hessians.
PSMALL = 1e-100
DSMALL = 1e-100
HSMALL = 1e-100

class GateSet(_collections.OrderedDict):
    """
    Encapsulates a set of gate, state preparation, and POVM effect operations.

    A GateSet stores a set of labeled Gate objects and provides dictionary-like
    access to their matrices.  State preparation and POVM effect operations are 
    represented as column vectors.
    """
    
    def __init__(self,items=[]):
        """ 
        Initialize a gate set, possibly from a list of items (used primarily by pickle) 
        """
        self.gate_dim = None  # dimension of gate matrices (each Gate object element should encapsulate a gateDim x gateDim matrix)
        self.rhoVecs = [ ]  # each el == a numpy array with dimension gateDim x 1
        self.EVecs = [ ]    # each el == a numpy array with dimension gateDim x 1
        self.gates = _collections.OrderedDict() #for Gate objects.  GateSet stores just matrices for speed and clean interface
        self.identityVec = None #the identity vector in whatever basis is being used (needed only if "-1" EVec is used)
        self.SPAM_labels = { }  #key = label, value = (rhoIndex, eIndex)
        self.SPAMs = { }
        self.history = [ ]  # a list of strings that track the journey of this gateset

        self.assumeSumToOne = False  #If True, compute probabilities for "-1"-index evec by 1-sum(other probs)
                                     #If False, compute probabilities for "-1"-index evec evaluating
                                     #   with Evec_{-1} = Identity - sum(other Evecs)

        super(GateSet, self).__init__(items)

    #def append_identity(self, gateDimension, nGates=1):
    #    for i in range(nGates):
    #        self.append( _np.identity(gateDimension) ) #note: no 'complex' here - real gate matrix

    #def set_random(self, sd, radius):
    #    rndm = _np.random.RandomState(sd)
    #    for (i,gate) in enumerate(self):
    #        self[i] = rndm.randn(gate.shape) * radius #Need to mult by identity if we want 'complex' matrices

    def get_dimension(self):
        """ 
        Get the dimension of the gateset, which equals d when the gate
        matrices have shape d x d and spam vectors have shape d x 1.  

        Returns
        -------
        int
            gateset dimension
        """
        return self.gate_dim

    def set_identity_vec(self, identityVec):
        """
        Set a the identity vector.  Calls
        make_spams automatically.

        Parameters
        ----------
        identityVec : numpy array
            a column vector containing the identity vector.
        """
        if self.gate_dim is None:
            self.gate_dim = len(identityVec)
        elif self.gate_dim != len(identityVec):
            raise ValueError("Cannot add identity vector with dimension %d to gateset of dimension %d" \
                                 % (len(identityVec),self.gate_dim))
        self.identityVec = identityVec
        self.make_spams()

    def get_identity_vec(self):
        """
        Get the identity vector in the basis being used by this gateset.

        Returns
        -------
        numpy array
            The identity column vector.  Note this is a reference to
            the GateSet's internal object, so callers should copy the
            vector before changing it.
        """
        return self.identityVec
    
    def set_rhovec(self, rhoVec, index=0):
        """
        Set a state prepartion vector by index.  Calls make_spams automatically.

        Parameters
        ----------
        rhoVec : numpy array
            a column vector containing the state preparation vector.

        index : int, optional
            the index of the state preparation vector to set.  Must
            be <= num_rhovecs(), where equality adds a new vector.
        """
        if self.gate_dim is None:
            self.gate_dim = len(rhoVec)
        elif self.gate_dim != len(rhoVec):
            raise ValueError("Cannot add rhoVec with dimension %d to gateset of dimension %d" % (len(rhoVec),self.gate_dim))

        if index < len(self.rhoVecs):
            self.rhoVecs[index] = rhoVec.copy()
        elif index == len(self.rhoVecs):
            self.rhoVecs.append(rhoVec.copy())
        else: raise ValueError("Cannot set rhoVec %d -- index must be <= %d" % (index, len(self.rhoVecs)))
        self.make_spams()


    def get_rhovec(self, index=0):
        """
        Get a state prepartion vector by index.

        Parameters
        ----------
        index : int, optional
            the index of the vector to return.

        Returns
        -------
        numpy array
            a state preparation vector of shape (gate_dim, 1).
        """
        return self.rhoVecs[index]


    def get_rhovecs(self):
        """
        Get an list of all the state prepartion vectors.  These
        vectors are copies of internally stored data and thus
        can be modified without altering the gateset.

        Returns
        -------
        list of numpy arrays
            list of state preparation vectors of shape (gate_dim, 1).
        """
        return [ self.get_rhovec(i).copy() for i in self.get_rhovec_indices() ]

                               
    def set_evec(self, EVec, index=0):
        """
        Set a POVM effect vector by index.  Calls make_spams automatically.

        Parameters
        ----------
        rhoVec : numpy array
            a column vector containing the effect vector.

        index : int, optional
            the index of the effect vector to set.  Must
            be <= num_evecs(), where equality adds a new vector.
        """
        if self.gate_dim is None:
            self.gate_dim = len(EVec)
        elif self.gate_dim != len(EVec):
            raise ValueError("Cannot add EVec with dimension %d to gateset of dimension %d" % (len(EVec),self.gate_dim))

        if index < len(self.EVecs):
            self.EVecs[index] = EVec.copy()
        elif index == len(self.EVecs):
            self.EVecs.append(EVec.copy())
        else: raise ValueError("Cannot set EVec %d -- index must be <= %d" % (index, len(self.EVecs)))
        self.make_spams()

    def get_evec(self, index=0):
        """
        Get a POVM effect vector by index.

        Parameters
        ----------
        index : int, optional
            the index of the vector to return.

        Returns
        -------
        numpy array
            an effect vector of shape (gate_dim, 1).
        """
        if index == -1:
            return self.identityVec - sum(self.EVecs)
        else:
            return self.EVecs[index]

    def get_evecs(self):
        """
        Get an list of all the POVM effect vectors.  This list will include
        the "compliment" effect vector at the end of the list if one has been
        specified.  Also, the returned vectors are copies of internally stored
        data and thus can be modified without altering the gateset.

        Returns
        -------
        list of numpy arrays
            list of POVM effect vectors of shape (gate_dim, 1).
        """
        return [ self.get_evec(i).copy() for i in self.get_evec_indices() ]


    def num_rhovecs(self):
        """
        Get the number of state preparation vectors

        Returns
        -------
        int
        """
        return len(self.rhoVecs)

    def num_evecs(self):
        """
        Get the number of effect vectors, including a "complement" effect vector,
          equal to Identity - sum(other effect vectors)

        Returns
        -------
        int
        """
        bHaveComplementEvec = any( [ (EIndx == -1 and rhoIndx != -1) for (rhoIndx,EIndx) in self.SPAM_labels.values() ] )
        return len(self.EVecs) + ( 1 if bHaveComplementEvec else 0 )

    def get_rhovec_indices(self):
        """
        Get the indices of state preparation vectors.

        Returns
        -------
        list of ints
        """
        return range(len(self.rhoVecs))

    def get_evec_indices(self):
        """
        Get the indices of effect vectors, possibly including -1 as
          the index of a "complement" effect vector,
          equal to Identity - sum(other effect vectors)

        Returns
        -------
        list of ints
        """
        inds = range(len(self.EVecs))
        if any( [ (EIndx == -1 and rhoIndx != -1) for (rhoIndx,EIndx) in self.SPAM_labels.values() ]):
            inds.append( -1 )
        return inds

    def add_spam_label(self, rhoIndex, eIndex, label):
        """
        Adds a new spam label.  That is, associates the SPAM
          pair (rhoIndex,eIndex) with the given label.  Calls
          make_spams automatically.

        Parameters
        ----------
        rhoIndex : int
            state preparation vector index.

        eIndex : int
            POVM effect vector index.

        label : string
            the "spam label" to associate with (rhoIndex,eIndex).
        """
        if rhoIndex == -1:
            if eIndex != -1: raise ValueError("EVec index must always == -1 when rhoVec index does")

            # This spam label generates probabilities which = 1 - sum(other spam label probs)
            self.assumeSumToOne = True
            _warnings.warn( "You have choosen to use a gate set in a sum-to-one mode in which " + 
                           "  the spam label %s will generate probabilities equal to 1.0 minus the sum " % label + 
                           "  of all the probabilities of the other spam labels.  If this wasn't intended " + 
                           "  it's probably because you assigned a spam label (-1,-1) instead of (0,-1)" )
        
        if (rhoIndex, eIndex) in self.SPAM_labels.values():
            raise ValueError("Cannot add label '%s' for (%d,%d): %s already refers to this pair" \
                                 % (label, rhoIndex, eIndex, self.get_spam_label_dict()[(rhoIndex,eIndex)]))
        self.SPAM_labels[label] = (rhoIndex, eIndex)
        assert(rhoIndex >= -1) # -1 allowed as rhoVec index but only when eIndex == -1 also
        assert(eIndex >= -1) # -1 allowed as evec index: meaning depends on self.assumeSumToOne
        self.make_spams()

    def get_spam_label_dict(self):
        """ 
        Get a reverse-lookup dictionary for spam labels.  

        Returns
        -------
        dict
            a dictionary with keys == (rhoIndex,eIndex) tuples and
            values == SPAM labels.
        """
        d = { }
        for label in self.SPAM_labels:
            d[  self.SPAM_labels[label] ] = label
        return d

    def get_spam_labels(self):
        """ 
        Get a list of all the spam labels.

        Returns
        -------
        list of strings
        """
        return self.SPAM_labels.keys()


    def __setitem__(self, key, value):
        if key not in self:
            raise KeyError("You cannot set a new gate of a GateSet using bracket assignment.  Use set_gate(...) instead.");
        raise ValueError("You must set gateset elements using set_gate now, and cannot directly assign a gate via array indexing")

        #NOTE: FUTURE - we could set the value of an existing gate, but this seems somewhat confusing and I don't think we'll need it
        ##Set the *value* of an existing gate
        #self.get_gate(key).set_value(value)
        #super(GateSet, self).__setitem__(key, value)


    def __reduce__(self): 
        #Used by pickle; needed because OrderedDict uses __reduce__ and we don't want 
        #  that one to be called, so we override...
        return (GateSet, (), self.__getstate__())  
            #Construct a GateSet class, passing init no parameters, and state given by __getstate__

    def __getstate__(self):
        #Return the state (for pickling)
        mystate = { 'gate_dim': self.gate_dim,
                    'rhoVecs' : self.rhoVecs,
                    'EVecs': self.EVecs,
                    'gates': self.gates,
                    'identityVec': self.identityVec,
                    'SPAM_labels': self.SPAM_labels,
                    'SPAMs' : self.SPAMs,
                    'history' : self.history,
                    'assumeSumToOne' : self.assumeSumToOne
                    }
        return mystate

    def __setstate__(self, stateDict):
        #Initialize a GateSet from a state dictionary (for un-pickling)
        self.gate_dim = stateDict['gate_dim']
        self.rhoVecs = stateDict['rhoVecs']
        self.EVecs = stateDict['EVecs']
        self.gates = stateDict['gates']
        self.identityVec = stateDict['identityVec']
        self.SPAM_labels = stateDict['SPAM_labels']
        self.SPAMs = stateDict['SPAMs']
        self.history = stateDict['history']
        self.assumeSumToOne = stateDict['assumeSumToOne']

        #Don't serialize dictionary elements (gate matrices) of self since they
        # must be set specially and are contained within self.gates
        for gateLabel,gateObj in self.gates.iteritems():
            super(GateSet, self).__setitem__(gateLabel, gateObj.matrix)
        

    def get_gate(self,label):
        """
        Get the Gate object associated with a given label.

        Parameters
        ----------
        label : string
            the gate label.

        Returns
        -------
        Gate
        """
        #return super(GateSet, self).__getitem__(key) #REMOVE
        return self.gates[label]

    def set_gate(self,label,gate):
        """
        Set the Gate object associated with a given label.

        Parameters
        ----------
        label : string
            the gate label.

        gate : Gate
            the gate object, which must have the dimension of the GateSet.
        """
        if self.gate_dim is None:
            self.gate_dim = gate.dim
        elif self.gate_dim != gate.dim:
            raise ValueError("Cannot add gate with dimension %d to gateset of dimension %d" % (gate.dim,self.gate_dim))
        self.gates[label] = gate
        super(GateSet, self).__setitem__(label, gate.matrix)

    def update(self, *args, **kwargs): #So that our __setitem__ is always called
        """ Updates the Gateset as a dictionary """
        #raise ValueError("Update on gatesets is not implemented")
        if args:
            if len(args) > 1:
                raise TypeError("update expected at most 1 arguments, got %d" % len(args))
            other = dict(args[0])
            for key in other:
                self[key] = other[key]
        for key in kwargs:
            self[key] = kwargs[key]

    def setdefault(self, key, value=None): #So that our __setitem__ is always called
        raise ValueError("setdefault on gatesets is not implemented")
        #if key not in self:
        #    self[key] = value
        #return self[key]

    def log(self, strDescription, extra=None):
        """
        Append a message to the log of this gateset.

        Parameters
        ----------
        strDescription : string
            a string description
            
        extra : anything, optional
            any additional variable to log along with strDescription.
        """
        self.history.append( (strDescription,extra) )


    def num_params(self,gates=True,G0=True,SPAM=True,SP0=True):
        """
        Return the number of free parameters when vectorizing
        this gateset according to the optional parameters.

        Parameters
        ----------
        gates : bool or list, optional
          Whether/which gate matrices should be vectorized (i.e. included
          as gateset parameters).

          - True = all gates
          - False = no gates
          - list of gate labels = those particular gates.

        G0 : bool, optional
          Whether the first row of gate matrices should be vectorized
          (i.e. included as gateset parameters).

        SPAM : bool, optional
          Whether the rhoVecs and EVecs should be vectorized
          (i.e. included as gateset parameters).

        SP0 : bool, optional
          Whether the first element of the state preparation (rho) vectors
          should be vectorized (i.e. included as gateset parameters).

        Returns
        -------
        int
            the number of gateset parameters.
        """
        gates; bSPAM = SPAM; bG0 = G0; bSP0 = SP0

        m = 0 if bSP0 else 1
        rhoSize = [ len(rhoVec)-m for rhoVec in self.rhoVecs ]
        eSize   = [ len(EVec) for EVec in self.EVecs ]

        L = 0
        if gates == True:     gates = self.keys()
        elif gates == False:  gates = []
        for gateLabelToInclude in gates:
            L += self.get_gate(gateLabelToInclude).num_params(bG0)
        if bSPAM: L += sum(rhoSize) + sum(eSize)

        assert(L == len(self.to_vector(gates,G0,SPAM,SP0)) ) #Sanity check
        return L


    def num_elements(self):
        """
        Return the number of total gate matrix and spam vector
        elements in this gateset.  This is in general different
        from the number of *parameters* in the gateset, which
        are the number of free variables used to generate all of
        the matrix and vector *elements*.

        Returns
        -------
        int
            the number of gateset elements.
        """
        rhoSize = [ len(rhoVec) for rhoVec in self.rhoVecs ]
        eSize   = [ len(EVec) for EVec in self.EVecs ]
        gateSize = [ gateMx.size for (label,gateMx) in self.iteritems() ]
        return sum(rhoSize) + sum(eSize) + sum(gateSize)


    def num_nongauge_params(self,gates=True,G0=True,SPAM=True,SP0=True):
        """
        Return the number of non-gauge parameters when vectorizing
        this gateset according to the optional parameters.

        Parameters
        ----------
        gates : bool or list, optional
          Whether/which gate matrices should be vectorized (i.e. included
          as gateset parameters).

          - True = all gates
          - False = no gates
          - list of gate labels = those particular gates.

        G0 : bool, optional
          Whether the first row of gate matrices should be vectorized
          (i.e. included as gateset parameters).

        SPAM : bool, optional
          Whether the rhoVecs and EVecs should be vectorized
          (i.e. included as gateset parameters).

        SP0 : bool, optional
          Whether the first element of the state preparation (rho) vectors
          should be vectorized (i.e. included as gateset parameters).

        Returns
        -------
        int
            the number of non-gauge gateset parameters.
        """
        P = self.get_nongauge_projector(gates,G0,SPAM,SP0)
        return _np.linalg.matrix_rank(P, P_RANK_TOL)


    def num_gauge_params(self,gates=True,G0=True,SPAM=True,SP0=True):
        """
        Return the number of gauge parameters when vectorizing
        this gateset according to the optional parameters.

        Parameters
        ----------
        gates : bool or list, optional
          Whether/which gate matrices should be vectorized (i.e. included
          as gateset parameters).

          - True = all gates
          - False = no gates
          - list of gate labels = those particular gates.

        G0 : bool, optional
          Whether the first row of gate matrices should be vectorized
          (i.e. included as gateset parameters).

        SPAM : bool, optional
          Whether the rhoVecs and EVecs should be vectorized
          (i.e. included as gateset parameters).

        SP0 : bool, optional
          Whether the first element of the state preparation (rho) vectors
          should be vectorized (i.e. included as gateset parameters).

        Returns
        -------
        int
            the number of gauge gateset parameters.
        """
        return self.num_params(gates,G0,SPAM,SP0) - self.num_nongauge_params(gates,G0,SPAM,SP0)


    #TODO: update for multiple rhoVecs / EVecs when I have time
    #def add_to_param(self, iParam, amount):
    #    if iParam < len(self.rhoVec):
    #        self.rhoVec[iParam,0] += amount
    #    elif iParam < len(self.rhoVec) + len(self.EVec):
    #        self.EVec[iParam - len(self.rhoVec),0] += amount
    #    else:
    #        s = len(self.rhoVec) + len(self.EVec); gate_label = ""
    #        for (l,g) in self.iteritems():
    #            gateSize = _np.prod(g.shape)
    #            if s+gateSize > iParam:
    #                gate_label = l; break
    #
    #        leftover = iParam - s
    #        irow = leftover // self[gate_label].shape[0]
    #        icol = leftover - irow*self[gate_label].shape[0]
    #        #print "DEBUG: %d --> %d,%d,%d" % (iParam, gind,irow,icol)
    #        self[gate_label][irow,icol] += amount

    def to_vector(self, gates=True,G0=True,SPAM=True,SP0=True):
        """
        Returns the gateset vectorized according to the optional parameters.

        Parameters
        ----------
        gates : bool or list, optional
          Whether/which gate matrices should be vectorized.

          - True = all gates
          - False = no gates
          - list of gate labels = those particular gates.

        G0 : bool, optional
          Whether the first row of gate matrices should be vectorized.

        SPAM : bool, optional
          Whether the rhoVecs and EVecs should be vectorized.

        SP0 : bool, optional
          Whether the first element of the state preparation (rho) vectors
          should be vectorized.

        Returns
        -------
        numpy array
            The vectorized gateset parameters.
        """
        bSPAM = SPAM; bG0 = G0; bSP0 = SP0
        if len(self) == 0: return _np.array([])

        m = 0 if bSP0 else 1
        gsize = dict( [ (l,g.num_params(bG0)) for (l,g) in self.gates.iteritems() ])
        rhoSize = [ len(rhoVec)-m for rhoVec in self.rhoVecs ]
        eSize   = [ len(EVec) for EVec in self.EVecs ]

        L = 0
        if gates == True:     gates = self.keys()
        elif gates == False:  gates = []
        for gateLabelToInclude in gates:
            L += gsize[gateLabelToInclude]

        if bSPAM: L += sum(rhoSize) + sum(eSize)

        v = _np.empty( L ); off = 0

        if bSPAM:
            for (i,rhoVec) in enumerate(self.rhoVecs):
                v[off:off+rhoSize[i]] = rhoVec[m:,0]          
                off += rhoSize[i]
            for (i,EVec) in enumerate(self.EVecs):
                v[off:off+eSize[i]] = EVec[:,0]          
                off += eSize[i]

        for l in gates:
            v[off:off+gsize[l]] = self.get_gate(l).to_vector(bG0)
            off += gsize[l]

        return v

    def from_vector(self, v, gates=True,G0=True,SPAM=True,SP0=True):
        """
        The inverse of to_vector.  Loads values of gates and/or rho and E vecs from
        from a vector v according to the optional parameters. Note that neither v 
        nor the optional parameters specify what number of gates and their labels,
        and that this information must be contained in the gateset prior to calling
        from_vector.  In practice, this just means you should call the from_vector method
        of the gateset that was used to generate the vector v in the first place.

        Parameters
        ----------
        v : numpy array
           the vectorized gateset vector whose values are loaded into the present gateset.

        gates : bool or list, optional
          Whether/which gate matrices should be un-vectorized.

          - True = all gates
          - False = no gates
          - list of gate labels = those particular gates.

        G0 : bool, optional
          Whether the first row of gate matrices should be un-vectorized.

        SPAM : bool, optional
          Whether the rhoVecs and EVecs should be un-vectorized.

        SP0 : bool, optional
          Whether the first element of the state preparation (rho) vectors
          should be un-vectorized.
        """

        bSPAM = SPAM; bG0 = G0; bSP0 = SP0

        m = 0 if bSP0 else 1
        gsize = dict( [ (l,g.num_params(bG0)) for (l,g) in self.gates.iteritems() ])
        rhoSize = [ len(rhoVec)-m for rhoVec in self.rhoVecs ]
        eSize   = [ len(EVec) for EVec in self.EVecs ]

        L = 0
        if gates == True:     gates = self.keys()
        elif gates == False:  gates = []
        for gateLabelToInclude in gates:
            L += gsize[ gateLabelToInclude ]

        if bSPAM: L += sum(rhoSize) + sum(eSize)

        assert( len(v) == L ); off = 0

        if bSPAM:
            for i in range(len(self.rhoVecs)):
                self.rhoVecs[i][m:,0] = v[off:off+rhoSize[i]]
                off += rhoSize[i]
            for i in range(len(self.EVecs)):
                self.EVecs[i][:,0] = v[off:off+eSize[i]]
                off += eSize[i]
        self.make_spams()

        for l in gates:
            gateObj = self.gates[l]
            gateObj.from_vector( v[off:off+gsize[l]], bG0)
            super(GateSet, self).__setitem__(l, gateObj.matrix)
            off += gsize[l]


    def get_vector_offsets(self, gates=True,G0=True,SPAM=True,SP0=True):
        """
        Returns the offsets of individual components in the vectorized 
        gateset according to the optional parameters.
    
        Parameters
        ----------
        gates : bool or list, optional
          Whether/which gate matrices should be vectorized.
    
          - True = all gates
          - False = no gates
          - list of gate labels = those particular gates.
    
        G0 : bool, optional
          Whether the first row of gate matrices should be vectorized.
    
        SPAM : bool, optional
          Whether the rhoVecs and EVecs should be vectorized.
    
        SP0 : bool, optional
          Whether the first element of the state preparation (rho) vectors
          should be vectorized.
    
        Returns
        -------
        dict
            A dictionary whose keys are either "rho<number>", "E<number>",
            or a gate label and whose values are (start,next_start) tuples of
            integers indicating the start and end+1 indices of the component.
        """
        bSPAM = SPAM; bG0 = G0; bSP0 = SP0
    
        m = 0 if bSP0 else 1
        gsize = dict( [ (l,g.num_params(bG0)) for (l,g) in self.gates.iteritems() ])
        rhoSize = [ len(rhoVec)-m for rhoVec in self.rhoVecs ]
        eSize   = [ len(EVec) for EVec in self.EVecs ]
    
        if gates == True:     gates = self.keys()
        elif gates == False:  gates = []
    
        off = 0
        offsets = {}
    
        if bSPAM:
            for (i,rhoVec) in enumerate(self.rhoVecs):
                offsets["rho%d" % i] = (off,off+rhoSize[i])
                off += rhoSize[i]
    
            for (i,EVec) in enumerate(self.EVecs):
                offsets["E%d" % i] = (off,off+eSize[i])
                off += eSize[i]
    
        for l in gates:
            offsets[l] = (off,off+gsize[l])
            off += gsize[l]
    
        return offsets


    def deriv_wrt_params(self, gates=True,G0=True,SPAM=True,SP0=True):
        """
        Construct a matrix whose columns are the vectorized
        derivatives of the gateset when all gates are treated as
        fully parameterized with respect to a single parameter
        of the true vectorized gateset.

        Each column is the length of a vectorized gateset of
        fully parameterized gates and there are num_params(...) columns.
        If the gateset is fully parameterized (i.e. contains only
        fully parameterized gates) then the resulting matrix will be
        the (square) identity matrix.

        Parameters
        ----------
        gates : bool or list, optional
          Whether/which gate matrices should be vectorized.

          - True = all gates
          - False = no gates
          - list of gate labels = those particular gates.

        G0 : bool, optional
          Whether the first row of gate matrices should be vectorized.

        SPAM : bool, optional
          Whether the rhoVecs and EVecs should be vectorized.

        SP0 : bool, optional
          Whether the first element of the state preparation (rho) vectors
          should be vectorized.

        Returns
        -------
        numpy array
        """

        bSPAM = SPAM; bG0 = G0; bSP0 = SP0
        if len(self) == 0: return _np.array([])

        m = 0 if bSP0 else 1
        gsize = dict( [ (l,g.num_params(bG0)) for (l,g) in self.gates.iteritems() ])
        rhoSize = [ len(rhoVec)-m for rhoVec in self.rhoVecs ]
        eSize   = [ len(EVec) for EVec in self.EVecs ]
        full_vsize = self.gate_dim
        full_gsize = self.gate_dim**2 # (flattened) size of a gate matrix

        if gates == True:     gates = self.keys()
        elif gates == False:  gates = []

        nParams = self.num_params(gates,G0,SPAM,SP0)
        nElements = self.num_elements() #total number of gate mx and spam vec elements
        deriv = _np.zeros( (nElements, nParams), 'd' )

        k = 0; foff= 0; off = 0 #independently track full-offset and (parameterized)-offset

        if bSPAM:
            for (i,rhoVec) in enumerate(self.rhoVecs):
                deriv[foff+m:foff+m+rhoSize[i],off:off+rhoSize[i]] = _np.identity( rhoSize[i], 'd' )
                off += rhoSize[i]; foff += full_vsize

            for (i,EVec) in enumerate(self.EVecs):
                deriv[foff:foff+eSize[i],off:off+eSize[i]] = _np.identity( eSize[i], 'd' )
                off += eSize[i]; foff += full_vsize
        else:
            foff += full_vsize * (len(self.rhoVecs) + len(self.EVecs))

        for l in gates:
            deriv[foff:foff+full_gsize,off:off+gsize[l]] = self.get_gate(l).deriv_wrt_params(bG0)
            off += gsize[l]; foff += full_gsize

        return deriv


    def transform(self, S, Si=None):
        """
        Update each of the gate matrices G in this gateset with inv(S) * G * S,
        each rhoVec with inv(S) * rhoVec, and each EVec with EVec * S

        Parameters
        ----------
        S : numpy array
            Matrix to perform similarity transform.  
            Should be shape (gate_dim, gate_dim).
            
        Si : numpy array, optional
            Inverse of S.  If None, inverse of S is computed.  
            Should be shape (gate_dim, gate_dim).
        """
        if Si is None: Si = _nla.inv(S) 
        for (i,rhoVec) in enumerate(self.rhoVecs): 
            self.rhoVecs[i] = _np.dot(Si,rhoVec)
        for (i,EVec) in enumerate(self.EVecs): 
            self.EVecs[i] = _np.dot(_np.transpose(S),EVec)  # same as ( Evec^T * S )^T

        if self.identityVec is not None:
            self.identityVec = _np.dot(_np.transpose(S),self.identityVec) #same as for EVecs

        self.make_spams()

        for (l,gate) in self.gates.iteritems():
            gate.transform(Si,S)
            super(GateSet, self).__setitem__(l, gate.matrix)


    def make_spams(self):
        """ 
        Updates cached spam gates using rhoVecs and EVecs.  This method needs
          to be called after modifying rhoVecs and/or EVecs directly.
        """
        for label in self.SPAM_labels:
            irhoVec,iEVec = self.SPAM_labels[label]
            if irhoVec >=0 and irhoVec < len(self.rhoVecs):
                if iEVec >= 0 and iEVec < len(self.EVecs):
                    self.SPAMs[label] = _np.kron(self.rhoVecs[irhoVec], _np.conjugate(_np.transpose(self.EVecs[iEVec])))
                elif iEVec == -1:
                    self.SPAMs[label] = _np.kron(self.rhoVecs[irhoVec], _np.conjugate(_np.transpose(self.get_evec(-1))))
                else: 
                    del self.SPAM_labels[label]
                    raise ValueError("Bad E-vector index: %d" % iEVec)
            elif irhoVec == -1:
                assert(iEVec == -1 and self.assumeSumToOne)
                self.SPAMs[label] = None    
            else:
                del self.SPAM_labels[label]
                raise ValueError("Bad rho-vector index: %d" % irhoVec)

    def product(self, gatestring, bScale=False):
        """ 
        Compute the product of a specified sequence of gate labels.

        Note: Gate matrices are multiplied in the reversed order of the tuple. That is,
        the first element of gatestring can be thought of as the first gate operation 
        performed, which is on the far right of the product of matrices.

        Parameters
        ----------
        gatestring : GateString or tuple of gate labels
            The sequence of gate labels.

        bScale : bool, optional
            When True, return a scaling factor (see below).

        Returns
        -------
        product : numpy array
            The product or scaled product of the gate matrices.

        scale : float
            Only returned when bScale == True, in which case the 
            actual product == product * scale.  The purpose of this
            is to allow a trace or other linear operation to be done
            prior to the scaling.
        """
        if bScale:
            scaledGatesAndExps = {}; 
            for (gateLabel,gatemx) in self.iteritems():
                ng = max(_nla.norm(gatemx),1.0)
                scaledGatesAndExps[gateLabel] = (gatemx / ng, _np.log(ng))

            scale_exp = 0
            G = _np.identity( self.gate_dim )
            for lGate in gatestring:
                gate, ex = scaledGatesAndExps[lGate]
                H = _np.dot(gate,G)   # product of gates, starting with identity
                scale_exp += ex   # scale and keep track of exponent
                if H.max() < PSMALL and H.min() > -PSMALL:
                    nG = max(_nla.norm(G), _np.exp(-scale_exp))
                    G = _np.dot(gate,G/nG); scale_exp += _np.log(nG) 
                    #OLD: _np.dot(G/nG,gate); scale_exp += _np.log(nG) LEXICOGRAPHICAL VS MATRIX ORDER
                else: G = H

            old_err = _np.seterr(over='ignore')
            scale = _np.exp(scale_exp)
            _np.seterr(**old_err)

            return G, scale
        
        else:
            G = _np.identity( self.gate_dim )
            for lGate in gatestring:
                G = _np.dot(self[lGate],G)  # product of gates
                #OLD: G = _np.dot(G,self[lGate]) LEXICOGRAPHICAL VS MATRIX ORDER
            return G


    #Vectorizing Identities. (Vectorization)
    # Note when vectorizing op uses numpy.flatten rows are kept contiguous, so the first identity below is valid.
    # Below we use E(i,j) to denote the elementary matrix where all entries are zero except the (i,j) entry == 1
    
    # if vec(.) concatenates rows (which numpy.flatten does)
    # vec( A * E(0,1) * B ) = vec( mx w/ row_i = A[i,0] * B[row1] ) = A tensor B^T * vec( E(0,1) )
    # In general: vec( A * X * B ) = A tensor B^T * vec( X )
    
    # if vec(.) stacks columns
    # vec( A * E(0,1) * B ) = vec( mx w/ col_i = A[col0] * B[0,1] ) = B^T tensor A * vec( E(0,1) )
    # In general: vec( A * X * B ) = B^T tensor A * vec( X )

    def dproduct(self, gatestring, gates=True, G0=True, flat=False):
        """
        Compute the derivative of a specified sequence of gate labels.  

        Parameters
        ----------
        gatestring : GateString or tuple of gate labels
          The sequence of gate labels.

        gates : bool or list, optional
          Whether/which gates should be included as gateset parameters.

          - True = all gates
          - False = no gates
          - list of gate labels = those particular gates.

        G0 : bool, optional
          Whether the first row of the included gate matrices
          should be included as gateset parameters.

        flat : bool, optional
          Affects the shape of the returned derivative array (see below).

        Returns
        -------
        deriv : numpy array
            * if flat == False, a M x G x G array, where:

              - M == length of the vectorized gateset (number of gateset parameters)
              - G == the linear dimension of a gate matrix (G x G gate matrices).

              and deriv[i,j,k] holds the derivative of the (j,k)-th entry of the product
              with respect to the i-th gateset parameter.

            * if flat == True, a N x M array, where:

              - N == the number of entries in a single flattened gate (ordering as numpy.flatten)
              - M == length of the vectorized gateset (number of gateset parameters)

              and deriv[i,j] holds the derivative of the i-th entry of the flattened
              product with respect to the j-th gateset parameter.                
        """

        if gates == True:     gates = self.keys()
        elif gates == False:  gates = []
        gatesToVectorize = gates #which differentiation w.r.t. gates should be done (which is all the differentiation done here)
        bG0 = G0                 #whether differentiation w.r.t. the first row of gate matrices should be done

        # LEXICOGRAPHICAL VS MATRIX ORDER
        revGateLabelList = tuple(reversed(tuple(gatestring))) # we do matrix multiplication in this order (easier to think about)

        #  prod = G1 * G2 * .... * GN , a matrix
        #  dprod/d(gateLabel)_ij   = sum_{L s.t. G(L) == gatelabel} [ G1 ... G(L-1) dG(L)/dij G(L+1) ... GN ] , a matrix for each given (i,j)
        #  vec( dprod/d(gateLabel)_ij ) = sum_{L s.t. G(L) == gatelabel} [ (G1 ... G(L-1)) tensor (G(L+1) ... GN)^T vec( dG(L)/dij ) ]
        #                               = [ sum_{L s.t. G(L) == gatelabel} [ (G1 ... G(L-1)) tensor (G(L+1) ... GN)^T ]] * vec( dG(L)/dij) )
        #  if dG(L)/dij = E(i,j) 
        #                               = vec(i,j)-col of [ sum_{L s.t. G(L) == gatelabel} [ (G1 ... G(L-1)) tensor (G(L+1) ... GN)^T ]]
        # So for each gateLabel the matrix [ sum_{L s.t. GL == gatelabel} [ (G1 ... G(L-1)) tensor (G(L+1) ... GN)^T ]] has columns which 
        #  correspond to the vectorized derivatives of each of the product components (i.e. prod_kl) with respect to a given gateLabel_ij
        # This function returns a concatenated form of the above matrices, so that each column corresponds to a (gateLabel,i,j) tuple and
        #  each row corresponds to an element of the product (els of prod.flatten()).
        #
        # Note: if gate G(L) is just a matrix of parameters, then dG(L)/dij = E(i,j), an elementary matrix

        gate_dim = self.gate_dim
        if len(gates) == 0:  #shortcut
            if flat: return _np.empty( [gate_dim**2,0] )
            else:    return _np.empty( [0,gate_dim,gate_dim] )

        #Cache partial products
        leftProds = [ ]
        G = _np.identity( gate_dim ); leftProds.append(G)
        for gateLabel in revGateLabelList:
            G = _np.dot(G,self[gateLabel]); leftProds.append(G)

        rightProdsT = [ ]
        G = _np.identity( gate_dim ); rightProdsT.append( _np.transpose(G) )
        for gateLabel in reversed(revGateLabelList):
            G = _np.dot(self[gateLabel],G); rightProdsT.append( _np.transpose(G) )

        # Initialize storage
        dprod_dgateLabel = { }; dgate_dgateLabel = {}
        for gateLabel in gatesToVectorize:
            dprod_dgateLabel[gateLabel] = _np.zeros( (gate_dim**2, self.get_gate(gateLabel).num_params(bG0) ) )
            dgate_dgateLabel[gateLabel] = self.get_gate(gateLabel).deriv_wrt_params(bG0) # (gate_dim**2, nParams[gateLabel])

        #Add contributions for each gate in list
        N = len(revGateLabelList)
        range_gate_dim = range(gate_dim)
        for (i,gateLabel) in enumerate(revGateLabelList):
            if gateLabel in dprod_dgateLabel: # same as "in gatesToVectorize" but faster dict lookup
                dprod_dgate = _np.kron( leftProds[i], rightProdsT[N-1-i] )  # (gate_dim**2, gate_dim**2)
                dprod_dgateLabel[gateLabel] += _np.dot( dprod_dgate, dgate_dgateLabel[gateLabel] ) # (gate_dim**2, nParams[gateLabel])
            
        #Concatenate per-gateLabel results to get final result
        to_concat = [ dprod_dgateLabel[gateLabel] for gateLabel in gatesToVectorize ]            
        flattened_dprod = _np.concatenate( to_concat, axis=1 ) # axes = (vectorized_gate_el_index,gateset_parameter)

        if flat:
            return flattened_dprod
        else:
            vec_gs_size = flattened_dprod.shape[1]
            return _np.swapaxes( flattened_dprod, 0, 1 ).reshape( (vec_gs_size, gate_dim, gate_dim) ) # axes = (gate_ij, prod_row, prod_col)

    def hproduct(self, gatestring, gates=True, G0=True, flat=False):
        """
        Compute the hessian of a specified sequence of gate labels.  

        Parameters
        ----------
        gatestring : GateString or tuple of gate labels
          The sequence of gate labels.

        gates : bool or list, optional
          Whether/which gates should be included as gateset parameters.

          - True = all gates
          - False = no gates
          - list of gate labels = those particular gates.

        G0 : bool, optional
          Whether the first row of the included gate matrices
          should be included as gateset parameters.

        flat : bool, optional
          Affects the shape of the returned derivative array (see below).

        Returns
        -------
        hessian : numpy array
            * if flat == False, a  M x M x G x G numpy array, where:

              - M == length of the vectorized gateset (number of gateset parameters)
              - G == the linear dimension of a gate matrix (G x G gate matrices).

              and hessian[i,j,k,l] holds the derivative of the (k,l)-th entry of the product
              with respect to the j-th then i-th gateset parameters.

            * if flat == True, a  N x M x M numpy array, where:

              - N == the number of entries in a single flattened gate (ordered as numpy.flatten)
              - M == length of the vectorized gateset (number of gateset parameters)

              and hessian[i,j,k] holds the derivative of the i-th entry of the flattened
              product with respect to the k-th then k-th gateset parameters.
        """

        if gates == True:     gates = self.keys()
        elif gates == False:  gates = []
        gatesToVectorize1 = gates #which differentiation w.r.t. gates should be done (which is all the differentiation done here)
        gatesToVectorize2 = gates # (possibility to later specify different sets of gates to differentiate firstly and secondly with)
        bG0 = G0                  #whether differentiation w.r.t. the first row of gate matrices should be done

        # LEXICOGRAPHICAL VS MATRIX ORDER
        revGateLabelList = tuple(reversed(tuple(gatestring))) # we do matrix multiplication in this order (easier to think about)

        #  prod = G1 * G2 * .... * GN , a matrix
        #  dprod/d(gateLabel)_ij   = sum_{L s.t. GL == gatelabel} [ G1 ... G(L-1) dG(L)/dij G(L+1) ... GN ] , a matrix for each given (i,j)
        #  d2prod/d(gateLabel1)_kl*d(gateLabel2)_ij = sum_{M s.t. GM == gatelabel1} sum_{L s.t. GL == gatelabel2, M < L} 
        #                                                 [ G1 ... G(M-1) dG(M)/dkl G(M+1) ... G(L-1) dG(L)/dij G(L+1) ... GN ] + {similar with L < M} (if L == M ignore)
        #                                                 a matrix for each given (i,j,k,l)
        #  vec( d2prod/d(gateLabel1)_kl*d(gateLabel2)_ij ) = sum{...} [ G1 ...  G(M-1) dG(M)/dkl G(M+1) ... G(L-1) tensor (G(L+1) ... GN)^T vec( dG(L)/dij ) ]
        #                                                  = sum{...} [ unvec( G1 ...  G(M-1) tensor (G(M+1) ... G(L-1))^T vec( dG(M)/dkl ) )
        #                                                                tensor (G(L+1) ... GN)^T vec( dG(L)/dij ) ]
        #                                                  + sum{ L < M} [ G1 ...  G(L-1) tensor
        #                                                       ( unvec( G(L+1) ... G(M-1) tensor (G(M+1) ... GN)^T vec( dG(M)/dkl ) ) )^T vec( dG(L)/dij ) ]
        #
        #  Note: ignoring L == M terms assumes that d^2 G/(dij)^2 == 0, which is true IF each gate matrix element is at most 
        #        *linear* in each of the gate parameters.  If this is not the case, need Gate objects to have a 2nd-deriv method in addition of deriv_wrt_params
        #
        #  Note: unvec( X ) can be done efficiently by actually computing X^T ( note (A tensor B)^T = A^T tensor B^T ) and using numpy's reshape

        gate_dim = self.gate_dim

        if len(gates) == 0:  #shortcut
            if flat:  return _np.empty( [gate_dim**2,0,0] )
            else:     return _np.empty( [0,0,gate_dim,gate_dim] )

        #Cache partial products
        prods = {}
        ident = _np.identity( gate_dim )
        for (i,gateLabel1) in enumerate(revGateLabelList): #loop over "starting" gate
            prods[ (i,i-1) ] = ident #product of no gates
            G = ident
            for (j,gateLabel2) in enumerate(revGateLabelList[i:],start=i): #loop over "ending" gate (>= starting gate)
                G = _np.dot(G,self[gateLabel2])
                prods[ (i,j) ] = G
        prods[ (len(revGateLabelList),len(revGateLabelList)-1) ] = ident #product of no gates

        # Initialize storage
        dgate_dgateLabel = {}; nParams = {}
        for gateLabel in set(gatesToVectorize1).union(gatesToVectorize2):
            dgate_dgateLabel[gateLabel] = self.get_gate(gateLabel).deriv_wrt_params(bG0) # (gate_dim**2, nParams[gateLabel])
            nParams[gateLabel] = dgate_dgateLabel[gateLabel].shape[1]

        d2prod_dgateLabels = { }; 
        for gateLabel1 in gatesToVectorize1:
            for gateLabel2 in gatesToVectorize2:
                d2prod_dgateLabels[(gateLabel1,gateLabel2)] = _np.zeros( (gate_dim**2, nParams[gateLabel1], nParams[gateLabel2]), 'd')

        #Add contributions for each gate in list
        N = len(revGateLabelList)
        range_gate_dim = range(gate_dim)
        for m,gateLabel1 in enumerate(revGateLabelList):
            if gateLabel1 in gatesToVectorize1:
                for l,gateLabel2 in enumerate(revGateLabelList):
                    if gateLabel2 in gatesToVectorize2:
                        # FUTURE: we could add logic that accounts for the symmetry of the Hessian, so that 
                        # if gl1 and gl2 are both in gatesToVectorize1 and gatesToVectorize2 we only compute d2(prod)/d(gl1)d(gl2)
                        # and not d2(prod)/d(gl2)d(gl1) ...
                        if m < l:
                            x0 = _np.kron(_np.transpose(prods[(0,m-1)]),prods[(m+1,l-1)])  # (gate_dim**2, gate_dim**2)
                            x  = _np.dot( _np.transpose(dgate_dgateLabel[gateLabel1]), x0); xv = x.view() # (nParams[gateLabel1],gate_dim**2)
                            xv.shape = (nParams[gateLabel1], gate_dim, gate_dim) # (reshape without copying - throws error if copy is needed)
                            y = _np.dot( _np.kron(xv, _np.transpose(prods[(l+1,N-1)])), dgate_dgateLabel[gateLabel2] ) 
                            # above: (nParams1,gate_dim**2,gate_dim**2) * (gate_dim**2,nParams[gateLabel2]) = (nParams1,gate_dim**2,nParams2)
                            d2prod_dgateLabels[(gateLabel1,gateLabel2)] += _np.swapaxes(y,0,1)
                            # above: dim = (gate_dim2, nParams1, nParams2); swapaxes takes (kl,vec_prod_indx,ij) => (vec_prod_indx,kl,ij)
                        elif l < m:
                            x0 = _np.kron(_np.transpose(prods[(l+1,m-1)]),prods[(m+1,N-1)]) # (gate_dim**2, gate_dim**2)
                            x  = _np.dot( _np.transpose(dgate_dgateLabel[gateLabel1]), x0); xv = x.view() # (nParams[gateLabel1],gate_dim**2)
                            xv.shape = (nParams[gateLabel1], gate_dim, gate_dim) # (reshape without copying - throws error if copy is needed)
                            xv = _np.swapaxes(xv,1,2) # transposes each of the now un-vectorized gate_dim x gate_dim mxs corresponding to a single kl
                            y = _np.dot( _np.kron(prods[(0,l-1)], xv), dgate_dgateLabel[gateLabel2] )
                            # above: (nParams1,gate_dim**2,gate_dim**2) * (gate_dim**2,nParams[gateLabel2]) = (nParams1,gate_dim**2,nParams2)
                            d2prod_dgateLabels[(gateLabel1,gateLabel2)] += _np.swapaxes(y,0,1)
                            # above: dim = (gate_dim2, nParams1, nParams2); swapaxes takes (kl,vec_prod_indx,ij) => (vec_prod_indx,kl,ij)
                        #else l==m, in which case there's no contribution since we assume all gate elements are at most linear in the parameters


        #Concatenate per-gateLabel results to get final result
        to_concat = []
        for gateLabel1 in gatesToVectorize1:
            to_concat.append( _np.concatenate( [ d2prod_dgateLabels[(gateLabel1,gateLabel2)] for gateLabel2 in gatesToVectorize2 ], axis=2 ) ) #concat along ij (nParams2)
        flattened_d2prod = _np.concatenate( to_concat, axis=1 ) # concat along kl (nParams1)

        if flat:
            return flattened_d2prod # axes = (vectorized_gate_el_index, gateset_parameter1, gateset_parameter2)
        else:
            vec_kl_size, vec_ij_size = flattened_d2prod.shape[1:3]
            return _np.rollaxis( flattened_d2prod, 0, 3 ).reshape( (vec_kl_size, vec_ij_size, gate_dim, gate_dim) )
            # axes = (gateset_parameter1, gateset_parameter2, gateset_element_row, gateset_element_col)


    def pr(self, spamLabel, gatestring, clipTo=None, bUseScaling=True):
        """ 
        Compute the probability of the given gate sequence, where initialization
        & measurement operations are together specified by spamLabel.

        Parameters
        ----------
        spamLabel : string
           the label specifying the state prep and measure operations
           
        gatestring : GateString or tuple of gate labels
          The sequence of gate labels specifying the gate string.

        clipTo : 2-tuple, optional
          (min,max) to clip return value if not None.

        bUseScaling : bool, optional
          Whether to use a post-scaled product internally.  If False, this
          routine will run slightly faster, but with a chance that the 
          product will overflow and the subsequent trace operation will
          yield nan as the returned probability.
           
        Returns
        -------
        float
        """

        if self.SPAMs[spamLabel] is None: #then compute 1.0 - (all other spam label probabilities)
            otherSpamLabels = self.get_spam_labels(); del otherSpamLabels[ otherSpamLabels.index(spamLabel) ]
            assert( all([ (self.SPAMs[sl] is not None) for sl in otherSpamLabels]) )
            return 1.0 - sum( [self.pr(sl, gatestring, clipTo, bUseScaling) for sl in otherSpamLabels] )

        if bUseScaling:
            old_err = _np.seterr(over='ignore')
            G,scale = self.product(gatestring, True)
            p = _mt.trace( _np.dot(self.SPAMs[spamLabel],G) ) * scale # probability, with scaling applied (may generate overflow, but OK)

            #DEBUG: catch warnings to make sure correct (inf if value is large) evaluation occurs when there's a warning
            #bPrint = False
            #with _warnings.catch_warnings():
            #    _warnings.filterwarnings('error')
            #    try:
            #        test = _mt.trace( _np.dot(self.SPAMs[spamLabel],G) ) * scale
            #    except Warning: bPrint = True
            #if bPrint:  print 'Warning in Gateset.pr : scale=%g, trace=%g, p=%g' % (scale,_np.dot(self.SPAMs[spamLabel],G) ), p)
            _np.seterr(**old_err)

        else: #no scaling -- faster but susceptible to overflow
            G = self.product(gatestring, False)
            p = _mt.trace( _np.dot(self.SPAMs[spamLabel],G) )

        if _np.isnan(p): 
            if len(gatestring) < 10:
                strToPrint = str(gatestring)
            else: strToPrint = str(gatestring[0:10]) + " ... (len %d)" % len(gatestring)
            _warnings.warn("pr(%s) == nan" % strToPrint)
            #DEBUG: print "backtrace" of product leading up to nan

            #G = _np.identity( self.gate_dim ); total_exp = 0.0
            #for i,lGate in enumerate(gateLabelList):
            #    G = _np.dot(G,self[lGate])  # product of gates, starting with G0
            #    nG = norm(G); G /= nG; total_exp += log(nG) # scale and keep track of exponent
            #
            #    p = _mt.trace( _np.dot(self.SPAMs[spamLabel],G) ) * exp(total_exp) # probability
            #    print "%d: p = %g, norm %g, exp %g\n%s" % (i,p,norm(G),total_exp,str(G))
            #    if _np.isnan(p): raise ValueError("STOP")

        if clipTo is not None: 
            return _np.clip(p,clipTo[0],clipTo[1])
        else: return p


    def dpr(self, spamLabel, gatestring,
            gates=True,G0=True,SPAM=True,SP0=True,
            returnPr=False,clipTo=None):
        """
        Compute the derivative of a probability generated by a gate string and
        spam label as a 1 x M numpy array, where M is the number of gateset
        parameters.

        Parameters
        ----------
        spamLabel : string
           the label specifying the state prep and measure operations
           
        gatestring : GateString or tuple of gate labels
          The sequence of gate labels specifying the gate string.

        gates : bool or list, optional
          Whether/which gates should be included as gateset parameters.

          - True = all gates
          - False = no gates
          - list of gate labels = those particular gates.

        G0 : bool, optional
          Whether the first row of the included gate matrices
          should be included as gateset parameters.

        SPAM : bool, optional
          Whether rhoVecs and EVecs should be included as gateset
          parameters.

        SP0 : bool, optional
          Whether the first element of the state preparation (rho) vectors
          should be included as gateset parameters.

        returnPr : bool, optional
          when set to True, additionally return the probability itself.

        clipTo : 2-tuple, optional
           (min,max) to clip returned probability to if not None.
           Only relevant when returnPr == True.

        Returns
        -------
        derivative : numpy array
            a 1 x M numpy array of derivatives of the probability w.r.t.
            each gateset parameter (M is the length of the vectorized gateset).

        probability : float
            only returned if returnPr == True.
        """

        bSPAM = SPAM; bG0 = G0; bSP0 = SP0

        if self.SPAMs[spamLabel] is None: #then compute Deriv[ 1.0 - (all other spam label probabilities) ]
            otherSpamLabels = self.get_spam_labels(); del otherSpamLabels[ otherSpamLabels.index(spamLabel) ]
            assert( all([ (self.SPAMs[sl] is not None) for sl in otherSpamLabels]) )
            otherResults = [self.dpr(sl, gatestring, gates, G0, SPAM, SP0, returnPr, clipTo) for sl in otherSpamLabels]
            if returnPr: 
                return -1.0 * sum([dpr for dpr,p in otherResults]), 1.0 - sum([p for dpr,p in otherResults])
            else:
                return -1.0 * sum(otherResults)

        #  pr = Tr( |rho><E| * prod ) = sum E_k prod_kl rho_l
        #  dpr/d(gateLabel)_ij = sum E_k [dprod/d(gateLabel)_ij]_kl rho_l
        #  dpr/d(rho)_i = sum E_k prod_ki
        #  dpr/d(E)_i   = sum prod_il rho_l

        gate_dim = self.gate_dim
        (rhoIndex,eIndex) = self.SPAM_labels[spamLabel]
        rho = self.rhoVecs[rhoIndex]
        E   = _np.conjugate(_np.transpose(self.get_evec(eIndex)))

        old_err = _np.seterr(over='ignore')
        prod,scale = self.product(gatestring,True)
        dprod_dGates = self.dproduct(gatestring, gates=gates, G0=bG0); vec_gs_size = dprod_dGates.shape[0]
        dpr_dGates = _np.empty( (1, vec_gs_size) )
        for i in xrange(vec_gs_size):
            dpr_dGates[0,i] = float(_np.dot(E, _np.dot( dprod_dGates[i], rho)))
        
        if returnPr:
            p = _mt.trace(_np.dot(self.SPAMs[spamLabel],prod)) * scale  #may generate overflow, but OK
            if clipTo is not None:  p = _np.clip( p, clipTo[0], clipTo[1] )

        if bSPAM:
            m = 0 if bSP0 else 1
            dpr_drhos = _np.zeros( (1, (gate_dim-m) * len(self.rhoVecs)) )
            dpr_drhos[0, (gate_dim-m)*rhoIndex:(gate_dim-m)*(rhoIndex+1)] = scale * _np.dot(E,prod)[0,m:]  #may overflow, but OK

            dpr_dEs = _np.zeros( (1, gate_dim * len(self.EVecs)) )
            derivWrtAnyEvec = scale * _np.transpose(_np.dot(prod,rho)) # may overflow, but OK (** doesn't depend on eIndex **)
            if eIndex == -1:
                for ei,EVec in enumerate(self.EVecs):  #compute Deriv w.r.t. [ 1 - sum_of_other_EVecs ]
                    dpr_dEs[0, gate_dim*ei:gate_dim*(ei+1)] = -1.0 * derivWrtAnyEvec
            else:
                dpr_dEs[0, gate_dim*eIndex:gate_dim*(eIndex+1)] = derivWrtAnyEvec
            _np.seterr(**old_err)

            if returnPr:
                  return _np.concatenate( (dpr_drhos,dpr_dEs,dpr_dGates), axis=1 ), p
            else: return _np.concatenate( (dpr_drhos,dpr_dEs,dpr_dGates), axis=1 )
        else:
            _np.seterr(**old_err)
            if returnPr: return dpr_dGates, p
            else:        return dpr_dGates


    def hpr(self, spamLabel, gatestring,
            gates=True, G0=True, SPAM=True, SP0=True,
            returnPr=False,returnDeriv=False,clipTo=None):
        """
        Compute the Hessian of a probability generated by a gate string and
        spam label as a 1 x M x M array, where M is the number of gateset
        parameters.

        Parameters
        ----------
        spamLabel : string
           the label specifying the state prep and measure operations
           
        gatestring : GateString or tuple of gate labels
          The sequence of gate labels specifying the gate string.

        gates : bool or list, optional
          Whether/which gates should be included as gateset parameters.

          - True = all gates
          - False = no gates
          - list of gate labels = those particular gates.

        G0 : bool, optional
          Whether the first row of the included gate matrices
          should be included as gateset parameters.

        SPAM : bool, optional
          Whether rhoVecs and EVecs should be included as gateset
          parameters.

        SP0 : bool, optional
          Whether the first element of the state preparation (rho) vectors
          should be included as gateset parameters.

        returnPr : bool, optional
          when set to True, additionally return the probability itself.

        returnDeriv : bool, optional
          when set to True, additionally return the derivative of the 
          probability.

        clipTo : 2-tuple, optional
           (min,max) to clip returned probability to if not None.
           Only relevant when returnPr == True.

        Returns
        -------
        hessian : numpy array
            a 1 x M x M array, where M is the number of gateset parameters.
            hessian[0,j,k] is the derivative of the probability w.r.t. the
            k-th then the j-th gateset parameter.

        derivative : numpy array
            only returned if returnDeriv == True. A 1 x M numpy array of 
            derivatives of the probability w.r.t. each gateset parameter.

        probability : float
            only returned if returnPr == True.
        """

        bSPAM = SPAM; bG0 = G0; bSP0 = SP0

        if self.SPAMs[spamLabel] is None: #then compute Hessian[ 1.0 - (all other spam label probabilities) ]
            otherSpamLabels = self.get_spam_labels(); del otherSpamLabels[ otherSpamLabels.index(spamLabel) ]
            assert( all([ (self.SPAMs[sl] is not None) for sl in otherSpamLabels]) )
            otherResults = [self.hpr(sl, gatestring, gates, G0, SPAM, SP0, returnPr, returnDeriv, clipTo) for sl in otherSpamLabels]
            if returnDeriv: 
                if returnPr: return ( -1.0 * sum([hpr for hpr,dpr,p in otherResults]),
                                      -1.0 * sum([dpr for hpr,dpr,p in otherResults]), 
                                       1.0 - sum([p   for hpr,dpr,p in otherResults])   )
                else:        return ( -1.0 * sum([hpr for hpr,dpr in otherResults]),
                                      -1.0 * sum([dpr for hpr,dpr in otherResults])     )
            else:
                if returnPr: return ( -1.0 * sum([hpr for hpr,p in otherResults]),
                                       1.0 - sum([p   for hpr,p in otherResults])   )
                else:        return   -1.0 * sum(otherResults)


        #  pr = Tr( |rho><E| * prod ) = sum E_k prod_kl rho_l
        #  d2pr/d(gateLabel1)_mn d(gateLabel2)_ij = sum E_k [dprod/d(gateLabel1)_mn d(gateLabel2)_ij]_kl rho_l
        #  d2pr/d(rho)_i d(gateLabel)_mn = sum E_k [dprod/d(gateLabel)_mn]_ki     (and same for other diff order)
        #  d2pr/d(E)_i d(gateLabel)_mn   = sum [dprod/d(gateLabel)_mn]_il rho_l   (and same for other diff order)
        #  d2pr/d(E)_i d(rho)_j          = prod_ij                                (and same for other diff order)
        #  d2pr/d(E)_i d(E)_j            = 0
        #  d2pr/d(rho)_i d(rho)_j        = 0

        gate_dim = self.gate_dim
        (rhoIndex,eIndex) = self.SPAM_labels[spamLabel]
        rho = self.rhoVecs[rhoIndex]
        E   = _np.conjugate(_np.transpose(self.get_evec(eIndex)))

        d2prod_dGates = self.hproduct(gatestring, gates=gates, G0=bG0)
        vec_gs_size = d2prod_dGates.shape[0]
        assert( d2prod_dGates.shape[0] == d2prod_dGates.shape[1] )

        d2pr_dGates2 = _np.empty( (1, vec_gs_size, vec_gs_size) )
        for i in xrange(vec_gs_size):
            for j in xrange(vec_gs_size):
                d2pr_dGates2[0,i,j] = float(_np.dot(E, _np.dot( d2prod_dGates[i,j], rho)))

        old_err = _np.seterr(over='ignore')
        if returnPr or bSPAM:
            prod,scale = self.product(gatestring,True)
            if returnPr:
                p = _mt.trace(_np.dot(self.SPAMs[spamLabel],prod)) * scale  #may generate overflow, but OK
                if clipTo is not None:  p = _np.clip( p, clipTo[0], clipTo[1] )

        if returnDeriv or bSPAM:
            dprod_dGates  = self.dproduct(gatestring, gates=gates, G0=bG0)
            assert( dprod_dGates.shape[0] == vec_gs_size )
            if returnDeriv: # same as in dpr(...)
                dpr_dGates = _np.empty( (1, vec_gs_size) )
                for i in xrange(vec_gs_size):
                    dpr_dGates[0,i] = float(_np.dot(E, _np.dot( dprod_dGates[i], rho)))


        if bSPAM:
            m = 0 if bSP0 else 1

            if returnDeriv:  #same as in dpr(...)
                dpr_drhos = _np.zeros( (1, (gate_dim-m) * len(self.rhoVecs)) )
                dpr_drhos[0, (gate_dim-m)*rhoIndex:(gate_dim-m)*(rhoIndex+1)] = scale * _np.dot(E,prod)[0,m:]  #may overflow, but OK
                dpr_dEs = _np.zeros( (1, gate_dim * len(self.EVecs)) )
                derivWrtAnyEvec = scale * _np.transpose(_np.dot(prod,rho)) # may overflow, but OK
                if eIndex == -1:
                    for ei,EVec in enumerate(self.EVecs):  #compute Deriv w.r.t. [ 1 - sum_of_other_EVecs ]
                        dpr_dEs[0, gate_dim*ei:gate_dim*(ei+1)] = -1.0 * derivWrtAnyEvec
                else:
                    dpr_dEs[0, gate_dim*eIndex:gate_dim*(eIndex+1)] = derivWrtAnyEvec
                dpr = _np.concatenate( (dpr_drhos,dpr_dEs,dpr_dGates), axis=1 )

            d2pr_drhos = _np.zeros( (1, vec_gs_size, (gate_dim-m) * len(self.rhoVecs)) )
            d2pr_drhos[0, :, (gate_dim-m)*rhoIndex:(gate_dim-m)*(rhoIndex+1)] = _np.dot(E,dprod_dGates)[0,:,m:]

            d2pr_dEs = _np.zeros( (1, vec_gs_size, gate_dim * len(self.EVecs)) )
            derivWrtAnyEvec = _np.squeeze(_np.dot(dprod_dGates,rho), axis=(2,))
            if eIndex == -1:
                for ei,EVec in enumerate(self.EVecs):  #similar to above, but now after also a deriv w.r.t gates
                    d2pr_dEs[0, :, gate_dim*ei:gate_dim*(ei+1)] = -1.0 * derivWrtAnyEvec
            else:
                d2pr_dEs[0, :, gate_dim*eIndex:gate_dim*(eIndex+1)] = derivWrtAnyEvec
            
            d2pr_dErhos = _np.zeros( (1, gate_dim * len(self.EVecs), (gate_dim-m) * len(self.rhoVecs)) )
            derivWrtAnyEvec = scale * prod[:,m:] #may generate overflow, but OK
            if eIndex == -1:
                for ei,EVec in enumerate(self.EVecs):  #similar to above, but now after also a deriv w.r.t rhos
                    d2pr_dErhos[0, gate_dim*ei:gate_dim*(ei+1), (gate_dim-m)*rhoIndex:(gate_dim-m)*(rhoIndex+1)] = -1.0 * derivWrtAnyEvec
            else:
                d2pr_dErhos[0, gate_dim*eIndex:gate_dim*(eIndex+1), (gate_dim-m)*rhoIndex:(gate_dim-m)*(rhoIndex+1)] = derivWrtAnyEvec

            d2pr_d2rhos = _np.zeros( (1, (gate_dim-m) * len(self.rhoVecs), (gate_dim-m) * len(self.rhoVecs)) )
            d2pr_d2Es   = _np.zeros( (1,  gate_dim * len(self.EVecs), gate_dim * len(self.EVecs)) )

            ret_row1 = _np.concatenate( ( d2pr_d2rhos, _np.transpose(d2pr_dErhos,(0,2,1)), _np.transpose(d2pr_drhos,(0,2,1)) ), axis=2) # wrt rho
            ret_row2 = _np.concatenate( ( d2pr_dErhos, d2pr_d2Es, _np.transpose(d2pr_dEs,(0,2,1)) ), axis=2 ) # wrt E
            ret_row3 = _np.concatenate( ( d2pr_drhos,d2pr_dEs,d2pr_dGates2), axis=2 ) #wrt gates
            ret = _np.concatenate( (ret_row1, ret_row2, ret_row3), axis=1 )
        else:
            if returnDeriv: dpr = dpr_dGates
            ret = d2pr_dGates2

        _np.seterr(**old_err)
        if returnDeriv: 
            if returnPr: return ret, dpr, p
            else:        return ret, dpr
        else:
            if returnPr: return ret, p
            else:        return ret



    def probs(self, gatestring, clipTo=None):
        """
        Construct a dictionary containing the probabilities of every spam label
        given a gate string.

        Parameters
        ----------
        gatestring : GateString or tuple of gate labels
          The sequence of gate labels specifying the gate string.

        clipTo : 2-tuple, optional
           (min,max) to clip probabilities to if not None.
           
        Returns
        -------
        probs : dictionary
            A dictionary such that 
            probs[SL] = pr(SL,gatestring,clipTo)
            for each spam label (string) SL.
        """
        probs = { }
        if not self.assumeSumToOne:
            for spamLabel in self.SPAMs:
                probs[spamLabel] = self.pr(spamLabel, gatestring, clipTo)
        else:
            spam_labels_to_loop = self.SPAMs.keys()
            s = 0; lastLabel = None
            for spamLabel in spam_labels_to_loop:
                if self.SPAMs[spamLabel] is None:
                    assert(lastLabel is None) # ensure there is at most one dummy spam label
                    lastLabel = spamLabel; continue
                probs[spamLabel] = self.pr(spamLabel, gatestring, clipTo)
                s += probs[spamLabel]
            if lastLabel is not None: 
                probs[lastLabel] = 1.0 - s  #last spam label is computed so sum == 1
        return probs

    def dprobs(self, gatestring,
               gates=True,G0=True,SPAM=True,SP0=True,
               returnPr=False,clipTo=None):
        """
        Construct a dictionary containing the probability derivatives of every
        spam label for a given gate string.

        Parameters
        ----------
        gatestring : GateString or tuple of gate labels
          The sequence of gate labels specifying the gate string.

        gates : bool or list, optional
          Whether/which gates should be included as gateset parameters.

          - True = all gates
          - False = no gates
          - list of gate labels = those particular gates.

        G0 : bool, optional
          Whether the first row of the included gate matrices
          should be included as gateset parameters.

        SPAM : bool, optional
          Whether rhoVecs and EVecs should be included as gateset
          parameters.

        SP0 : bool, optional
          Whether the first element of the state preparation (rho) vectors
          should be included as gateset parameters.

        returnPr : bool, optional
          when set to True, additionally return the probabilities.

        clipTo : 2-tuple, optional
           (min,max) to clip returned probability to if not None.
           Only relevant when returnPr == True.

        Returns
        -------
        dprobs : dictionary
            A dictionary such that 
            dprobs[SL] = dpr(SL,gatestring,gates,G0,SPAM,SP0,returnPr,clipTo)
            for each spam label (string) SL.
        """
        dprobs = { }
        if not self.assumeSumToOne:
            for spamLabel in self.SPAMs:
                dprobs[spamLabel] = self.dpr(spamLabel, gatestring,
                                             gates,G0,SPAM,SP0,returnPr,clipTo)
        else:
            spam_labels_to_loop = self.SPAMs.keys()
            ds = None; s=0; lastLabel = None
            for spamLabel in spam_labels_to_loop:
                if self.SPAMs[spamLabel] is None:
                    assert(lastLabel is None) # ensure there is at most one dummy spam label
                    lastLabel = spamLabel; continue
                dprobs[spamLabel] = self.dpr(spamLabel, gatestring, 
                                             gates,G0,SPAM,SP0,returnPr,clipTo)
                if returnPr:
                    ds = dprobs[spamLabel][0] if ds is None else ds + dprobs[spamLabel][0]
                    s += dprobs[spamLabel][1]
                else:
                    ds = dprobs[spamLabel] if ds is None else ds + dprobs[spamLabel]
            if lastLabel is not None: 
                dprobs[lastLabel] = (-ds,1.0-s) if returnPr else -ds
        return dprobs



    def hprobs(self, gatestring, 
               gates=True,G0=True,SPAM=True,SP0=True,
               returnPr=False, returnDeriv=False, clipTo=None):
        """
        Construct a dictionary containing the probability derivatives of every
        spam label for a given gate string.

        Parameters
        ----------
        gatestring : GateString or tuple of gate labels
          The sequence of gate labels specifying the gate string.

        gates : bool or list, optional
          Whether/which gates should be included as gateset parameters.

          - True = all gates
          - False = no gates
          - list of gate labels = those particular gates.

        G0 : bool, optional
          Whether the first row of the included gate matrices
          should be included as gateset parameters.

        SPAM : bool, optional
          Whether rhoVecs and EVecs should be included as gateset
          parameters.

        SP0 : bool, optional
          Whether the first element of the state preparation (rho) vectors
          should be included as gateset parameters.

        returnPr : bool, optional
          when set to True, additionally return the probabilities.

        returnDeriv : bool, optional
          when set to True, additionally return the derivatives of the 
          probabilities.

        clipTo : 2-tuple, optional
           (min,max) to clip returned probability to if not None.
           Only relevant when returnPr == True.

        Returns
        -------
        hprobs : dictionary
            A dictionary such that 
            hprobs[SL] = hpr(SL,gatestring,gates,G0,SPAM,SP0,returnPr,returnDeriv,clipTo)
            for each spam label (string) SL.
        """
        hprobs = { }
        if not self.assumeSumToOne:
            for spamLabel in self.SPAMs:
                hprobs[spamLabel] = self.hpr(spamLabel, gatestring,
                                             gates,G0,SPAM,SP0,returnPr,
                                             returnDeriv,clipTo)
        else:
            spam_labels_to_loop = self.SPAMs.keys()
            hs = None; ds=None; s=0; lastLabel = None
            for spamLabel in spam_labels_to_loop:
                if self.SPAMs[spamLabel] is None:
                    assert(lastLabel is None) # ensure there is at most one dummy spam label
                    lastLabel = spamLabel; continue
                hprobs[spamLabel] = self.hpr(spamLabel, gatestring, 
                                             gates,G0,SPAM,SP0,returnPr,
                                             returnDeriv,clipTo)
                if returnPr:
                    if returnDeriv:
                        hs = hprobs[spamLabel][0] if hs is None else hs + hprobs[spamLabel][0]
                        ds = hprobs[spamLabel][1] if ds is None else ds + hprobs[spamLabel][1]
                        s += hprobs[spamLabel][2]
                    else:
                        hs = hprobs[spamLabel][0] if hs is None else hs + hprobs[spamLabel][0]
                        s += hprobs[spamLabel][1]
                else:
                    if returnDeriv:
                        hs = hprobs[spamLabel][0] if hs is None else hs + hprobs[spamLabel][0]
                        ds = hprobs[spamLabel][1] if ds is None else ds + hprobs[spamLabel][1]
                    else:
                        hs = hprobs[spamLabel] if hs is None else hs + hprobs[spamLabel]

            if lastLabel is not None: 
                if returnPr:
                    hprobs[lastLabel] = (-hs,-ds,1.0-s) if returnDeriv else (-hs,1.0-s)
                else:
                    hprobs[lastLabel] = (-hs,-ds) if returnDeriv else -hs
                    
        return hprobs


#    def bulk_evaltree_beta(self, gatestring_list):
#        """
#          Returns an evaluation tree for all the gate 
#          strings in gatestring_list. Used by bulk_pr and
#          bulk_dpr, this is it's own function so that 
#          if many calls to bulk_pr and/or bulk_dpr are
#          made with the same gatestring_list, only a single
#          call to bulk_evaltree is needed.
#        """
#        evalTree = _evaltree.EvalTree()
#        evalTree.initialize_beta([""] + self.keys(), gatestring_list)
#        return evalTree


    def bulk_evaltree(self, gatestring_list):
        """
        Create an evaluation tree for all the gate strings in gatestring_list.

        This tree can be used by other Bulk_* functions, and is it's own 
        function so that for many calls to Bulk_* made with the same 
        gatestring_list, only a single call to bulk_evaltree is needed.
        
        Parameters
        ----------
        gatestring_list : list of (tuples or GateStrings)
            Each element specifies a gate string to include in the evaluation tree.

        Returns
        -------
        EvalTree
            An evaluation tree object.
        """
        evalTree = _evaltree.EvalTree()
        evalTree.initialize([""] + self.keys(), gatestring_list)
        return evalTree


    def bulk_product(self, evalTree, bScale=False):
        """
        Compute the products of many gate strings at once.

        Parameters
        ----------
        evalTree : EvalTree
           given by a prior call to bulk_evaltree.  Specifies the gate strings
           to compute the bulk operation on.

        bScale : bool, optional
           When True, return a scaling factor (see below).
              
        Returns
        -------
        prods : numpy array
            Array of shape S x G x G, where:

            - S == the number of gate strings
            - G == the linear dimension of a gate matrix (G x G gate matrices).

        scaleValues : numpy array
            Only returned when bScale == True. A length-S array specifying 
            the scaling that needs to be applied to the resulting products
            (final_product[i] = scaleValues[i] * prods[i]).
        """

        gate_dim = self.gate_dim
        assert(not evalTree.is_split()) #product functions can't use split trees (as there's really no point)

        cacheSize = len(evalTree)
        prodCache = _np.zeros( (cacheSize, gate_dim, gate_dim) )
        scaleCache = _np.zeros( cacheSize, 'd' )

        #First element of cache are given by evalTree's initial single- or zero-gate labels
        for i,gateLabel in enumerate(evalTree.get_init_labels()):
            if gateLabel == "": #special case of empty label == no gate
                prodCache[i] = _np.identity( gate_dim )
            else:
                gate = self[gateLabel]
                nG = max(_nla.norm(gate), 1.0)
                prodCache[i] = gate / nG
                scaleCache[i] = _np.log(nG)

        nZeroAndSingleStrs = len(evalTree.get_init_labels())

        #evaluate gate strings using tree (skip over the zero and single-gate-strings)
        #cnt = 0
        for (i,tup) in enumerate(evalTree[nZeroAndSingleStrs:],start=nZeroAndSingleStrs):

            # combine iLeft + iRight => i
            # LEXICOGRAPHICAL VS MATRIX ORDER Note: we reverse iLeft <=> iRight from evalTree because
            # (iRight,iLeft,iFinal) = tup implies gatestring[i] = gatestring[iLeft] + gatestring[iRight], but we want:
            (iRight,iLeft,iFinal) = tup   # since then matrixOf(gatestring[i]) = matrixOf(gatestring[iLeft]) * matrixOf(gatestring[iRight])
            L,R = prodCache[iLeft], prodCache[iRight]
            prodCache[i] = _np.dot(L,R)
            scaleCache[i] = scaleCache[iLeft] + scaleCache[iRight]

            if prodCache[i].max() < PSMALL and prodCache[i].min() > -PSMALL:
                nL,nR = max(_nla.norm(L), _np.exp(-scaleCache[iLeft]),1e-300), max(_nla.norm(R), _np.exp(-scaleCache[iRight]),1e-300)
                sL, sR = L/nL, R/nR
                prodCache[i] = _np.dot(sL,sR); scaleCache[i] += _np.log(nL) + _np.log(nR)
               
        #print "bulk_product DEBUG: %d rescalings out of %d products" % (cnt, len(evalTree)) 

        nanOrInfCacheIndices = (~_np.isfinite(prodCache)).nonzero()[0]  #may be duplicates (a list, not a set)
        assert( len(nanOrInfCacheIndices) == 0 ) # since all scaled gates start with norm <= 1, products should all have norm <= 1

        #use cached data to construct return values
        finalIndxList = evalTree.get_list_of_final_value_tree_indices()
        Gs = prodCache.take(  finalIndxList, axis=0 ) #shape == ( len(gatestring_list), gate_dim, gate_dim ), Gs[i] is product for i-th gate string
        scaleExps = scaleCache.take( finalIndxList )

        old_err = _np.seterr(over='ignore')
        scaleVals = _np.exp(scaleExps) #may overflow, but OK if infs occur here
        _np.seterr(**old_err)

        if bScale:
            return Gs, scaleVals
        else:
            old_err = _np.seterr(over='ignore')
            Gs = _np.swapaxes( _np.swapaxes(Gs,0,2) * scaleVals, 0,2)  #may overflow, but ok
            _np.seterr(**old_err)
            return Gs


    def bulk_dproduct(self, evalTree, gates=True, G0=True, flat=False, bReturnProds=False, bScale=False, memLimit=None):
        """
        Compute the derivative of a many gate strings at once.

        Parameters
        ----------
        evalTree : EvalTree
           given by a prior call to bulk_evaltree.  Specifies the gate strings
           to compute the bulk operation on.

        gates : bool or list, optional
          Whether/which gates should be included as gateset parameters.

          - True = all gates
          - False = no gates
          - list of gate labels = those particular gates.

        G0 : bool, optional
          Whether the first row of the included gate matrices
          should be included as gateset parameters.

        flat : bool, optional
          Affects the shape of the returned derivative array (see below).

        bReturnProds : bool, optional
          when set to True, additionally return the probabilities.

        bScale : bool, optional
          When True, return a scaling factor (see below).

        memLimit : int, optional
          A rough memory limit in bytes which restricts the amount of
          intermediate values that are computed and stored.
        
           
        Returns
        -------
        derivs : numpy array
          
          * if flat == False, an array of shape S x M x G x G, where:

            - S == len(gatestring_list)
            - M == the length of the vectorized gateset
            - G == the linear dimension of a gate matrix (G x G gate matrices)
            
            and derivs[i,j,k,l] holds the derivative of the (k,l)-th entry
            of the i-th gate string product with respect to the j-th gateset
            parameter.

          * if flat == True, an array of shape S*N x M where:

            - N == the number of entries in a single flattened gate (ordering same as numpy.flatten),
            - S,M == as above,
              
            and deriv[i,j] holds the derivative of the (i % G^2)-th entry of
            the (i / G^2)-th flattened gate string product  with respect to 
            the j-th gateset parameter.

        products : numpy array
          Only returned when bReturnProds == True.  An array of shape  
          S x G x G; products[i] is the i-th gate string product.

        scaleVals : numpy array
          Only returned when bScale == True.  An array of shape S such that
          scaleVals[i] contains the multiplicative scaling needed for
          the derivatives and/or products for the i-th gate string.
        """

        bG0 = G0
        gate_dim = self.gate_dim
        assert(not evalTree.is_split()) #product functions can't use split trees (as there's really no point)

        nGateStrings = evalTree.num_final_strings() #len(gatestring_list)
        nGateDerivCols = self.num_params(gates=gates, G0=bG0, SPAM=False) 
        deriv_shape = (nGateDerivCols, gate_dim, gate_dim)
        cacheSize = len(evalTree)

        ##DEBUG
        #nc = cacheSize; gd = gate_dim; nd = nGateDerivCols; C = 8.0/1024.0**3
        #print "Memory estimate for bulk_dproduct: %d eval tree size, %d gate dim, %d gateset params" % (nc,gd,nd)
        #print "    ==> %g GB (p) + %g GB (dp) + %g GB (scale) = %g GB (total)" % \
        #    (nc*gd*gd*C, nc*nd*gd*gd*C,nc*C, (nc*gd*gd + nc*nd*gd*gd + nc)*C)

        memEstimate = 8*cacheSize*(1 + gate_dim**2 * (1 + nGateDerivCols)) # in bytes (8* = 64-bit floats)
        if memLimit is not None and memEstimate > memLimit:
            C = 1.0/(1024.0**3) #conversion bytes => GB (memLimit assumed to be in bytes)
            raise MemoryError("Memory estimate of %dGB  exceeds limit of %dGB" % (memEstimate*C,memLimit*C))    

        prodCache = _np.zeros( (cacheSize, gate_dim, gate_dim) )
        dProdCache = _np.zeros( (cacheSize,) + deriv_shape )
        scaleCache = _np.zeros( cacheSize, 'd' )



        #print "DEBUG: cacheSize = ",cacheSize, " gate dim = ",gate_dim, " deriv_shape = ",deriv_shape
        #print "  pc MEM estimate = ", cacheSize*gate_dim*gate_dim*8.0/(1024.0**2), "MB"
        #print "  dpc MEM estimate = ", cacheSize*_np.prod(deriv_shape)*8.0/(1024.0**2), "MB"
        #print "  sc MEM estimate = ", cacheSize*8.0/(1024.0**2), "MB"
        #import time
        #time.sleep(10)
        #print "Continuing..."        

        # This iteration **must** match that in bulk_evaltree
        #   in order to associate the right single-gate-strings w/indices
        for i,gateLabel in enumerate(evalTree.get_init_labels()):
            if gateLabel == "": #special case of empty label == no gate
                prodCache[i] = _np.identity( gate_dim ); dProdCache[0] = _np.zeros( deriv_shape )
            else:
                dgate = self.dproduct( (gateLabel,) ,gates=gates, G0=bG0)
                nG = max(_nla.norm(self[gateLabel]),1.0)
                prodCache[i]  = self[gateLabel] / nG; scaleCache[i] = _np.log(nG)
                dProdCache[i] = dgate / nG 
            
        nZeroAndSingleStrs = len(evalTree.get_init_labels())

        #evaluate gate strings using tree (skip over the zero and single-gate-strings)
        for (i,tup) in enumerate(evalTree[nZeroAndSingleStrs:],start=nZeroAndSingleStrs):

            # combine iLeft + iRight => i
            # LEXICOGRAPHICAL VS MATRIX ORDER Note: we reverse iLeft <=> iRight from evalTree because
            # (iRight,iLeft,iFinal) = tup implies gatestring[i] = gatestring[iLeft] + gatestring[iRight], but we want:
            (iRight,iLeft,iFinal) = tup   # since then matrixOf(gatestring[i]) = matrixOf(gatestring[iLeft]) * matrixOf(gatestring[iRight])
            L,R = prodCache[iLeft], prodCache[iRight]
            prodCache[i] = _np.dot(L,R)

            #if not prodCache[i].any(): #same as norm(prodCache[i]) == 0 but faster
            if prodCache[i].max() < PSMALL and prodCache[i].min() > -PSMALL:
                nL,nR = max(_nla.norm(L), _np.exp(-scaleCache[iLeft]),1e-300), max(_nla.norm(R), _np.exp(-scaleCache[iRight]),1e-300)
                sL, sR, sdL, sdR = L/nL, R/nR, dProdCache[iLeft]/nL, dProdCache[iRight]/nR
                prodCache[i] = _np.dot(sL,sR); dProdCache[i] = _np.dot(sdL, sR) + _np.swapaxes(_np.dot(sL, sdR),0,1)
                scaleCache[i] = scaleCache[iLeft] + scaleCache[iRight] + _np.log(nL) + _np.log(nR)
                if dProdCache[i].max() < DSMALL and dProdCache[i].min() > -DSMALL:
                    _warnings.warn("Scaled dProd small in order to keep prod managable.")
            else:
                dL,dR = dProdCache[iLeft], dProdCache[iRight]
                dProdCache[i] = _np.dot(dL, R) + _np.swapaxes(_np.dot(L, dR),0,1) #dot(dS, T) + dot(S, dT)
                scaleCache[i] = scaleCache[iLeft] + scaleCache[iRight]
                
                if dProdCache[i].max() < DSMALL and dProdCache[i].min() > -DSMALL:
                    nL,nR = max(_nla.norm(dL), _np.exp(-scaleCache[iLeft]),1e-300), max(_nla.norm(dR), _np.exp(-scaleCache[iRight]),1e-300)
                    sL, sR, sdL, sdR = L/nL, R/nR, dL/nL, dR/nR
                    prodCache[i] = _np.dot(sL,sR); dProdCache[i] = _np.dot(sdL, sR) + _np.swapaxes(_np.dot(sL, sdR),0,1)
                    scaleCache[i] = scaleCache[iLeft] + scaleCache[iRight] + _np.log(nL) + _np.log(nR)
                    if prodCache[i].max() < PSMALL and prodCache[i].min() > -PSMALL:
                        _warnings.warn("Scaled prod small in order to keep dProd managable.")
                
        nanOrInfCacheIndices = (~_np.isfinite(prodCache)).nonzero()[0] 
        assert( len(nanOrInfCacheIndices) == 0 ) # since all scaled gates start with norm <= 1, products should all have norm <= 1
        
#        #Possibly re-evaluate tree using slower method if there nan's or infs using the fast method
#        if len(nanOrInfCacheIndices) > 0:
#            iBeginScaled = min( evalTree[ min(nanOrInfCacheIndices) ][0:2] ) # first index in tree that *resulted* in a nan or inf
#            _warnings.warn("Nans and/or Infs triggered re-evaluation at indx %d of %d products" % (iBeginScaled,len(evalTree)))
#            for (i,tup) in enumerate(evalTree[iBeginScaled:],start=iBeginScaled):
#                (iLeft,iRight,iFinal) = tup
#                L,R = prodCache[iLeft], prodCache[iRight],
#                G = dot(L,R); nG = norm(G)
#                prodCache[i] = G / nG
#                dProdCache[i] = dot(dProdCache[iLeft], R) + swapaxes(dot(L, dProdCache[iRight]),0,1) / nG
#                scaleCache[i] = scaleCache[iLeft] + scaleCache[iRight] + log(nG)

        #use cached data to construct return values

        finalIndxList = evalTree.get_list_of_final_value_tree_indices()

        old_err = _np.seterr(over='ignore')
        scaleExps = scaleCache.take( finalIndxList )
        scaleVals = _np.exp(scaleExps) #may overflow, but OK if infs occur here
        _np.seterr(**old_err)

        if bReturnProds:
            Gs  = prodCache.take(  finalIndxList, axis=0 ) #shape == ( len(gatestring_list), gate_dim, gate_dim ), Gs[i] is product for i-th gate string
            dGs = dProdCache.take( finalIndxList, axis=0 ) #shape == ( len(gatestring_list), nGateDerivCols, gate_dim, gate_dim ), dGs[i] is dprod_dGates for ith string

            if not bScale:
                old_err = _np.seterr(over='ignore', invalid='ignore')
                Gs  = _np.swapaxes( _np.swapaxes(Gs,0,2) * scaleVals, 0,2)  #may overflow, but ok
                dGs = _np.swapaxes( _np.swapaxes(dGs,0,3) * scaleVals, 0,3) #may overflow or get nans (invalid), but ok
                dGs[_np.isnan(dGs)] = 0  #convert nans to zero, as these occur b/c an inf scaleVal is mult by a zero deriv value (see below)
                _np.seterr(**old_err)

            if flat: dGs =  _np.swapaxes( _np.swapaxes(dGs,0,1).reshape( (nGateDerivCols, nGateStrings*gate_dim**2) ), 0,1 ) # cols = deriv cols, rows = flattened everything else
            return (dGs, Gs, scaleVals) if bScale else (dGs, Gs)

        else:
            dGs = dProdCache.take( finalIndxList, axis=0 ) #shape == ( len(gatestring_list), nGateDerivCols, gate_dim, gate_dim ), dGs[i] is dprod_dGates for ith string

            if not bScale:
                old_err = _np.seterr(over='ignore', invalid='ignore')
                dGs = _np.swapaxes( _np.swapaxes(dGs,0,3) * scaleVals, 0,3) #may overflow or get nans (invalid), but ok
                dGs[_np.isnan(dGs)] =  0 #convert nans to zero, as these occur b/c an inf scaleVal is mult by a zero deriv value, and we 
                                        # assume the zero deriv value trumps since we've renormed to keep all the products within decent bounds
                #assert( len( (_np.isnan(dGs)).nonzero()[0] ) == 0 ) 
                #assert( len( (_np.isinf(dGs)).nonzero()[0] ) == 0 ) 
                #dGs = clip(dGs,-1e300,1e300)
                _np.seterr(**old_err)

            if flat: dGs =  _np.swapaxes( _np.swapaxes(dGs,0,1).reshape( (nGateDerivCols, nGateStrings*gate_dim**2) ), 0,1 ) # cols = deriv cols, rows = flattened everything else
            return (dGs, scaleVals) if bScale else dGs


    def bulk_hproduct(self, evalTree, gates=True, G0=True, flat=False,
                      bReturnDProdsAndProds=False, bScale=False):
        """
        Return the Hessian of a many gate strings at once.

        Parameters
        ----------
        evalTree : EvalTree
           given by a prior call to bulk_evaltree.  Specifies the gate strings
           to compute the bulk operation on.

        gates : bool or list, optional
          Whether/which gates should be included as gateset parameters.

          - True = all gates
          - False = no gates
          - list of gate labels = those particular gates.

        G0 : bool, optional
          Whether the first row of the included gate matrices
          should be included as gateset parameters.

        flat : bool, optional
          Affects the shape of the returned derivative array (see below).

        bReturnDProdsAndProds : bool, optional
          when set to True, additionally return the probabilities and
          their derivatives.
           
        Returns
        -------
        hessians : numpy array
            * if flat == False, an  array of shape S x M x M x G x G, where 

              - S == len(gatestring_list)
              - M == the length of the vectorized gateset
              - G == the linear dimension of a gate matrix (G x G gate matrices)

              and hessians[i,j,k,l,m] holds the derivative of the (l,m)-th entry
              of the i-th gate string product with respect to the k-th then j-th
              gateset parameters.

            * if flat == True, an array of shape S*N x M x M where

              - N == the number of entries in a single flattened gate (ordering as numpy.flatten),
              - S,M == as above,

              and hessians[i,j,k] holds the derivative of the (i % G^2)-th entry
              of the (i / G^2)-th flattened gate string product with respect to 
              the k-th then j-th gateset parameters.

        derivs : numpy array
          Only returned if bReturnDProdsAndProds == True.

          * if flat == False, an array of shape S x M x G x G, where 

            - S == len(gatestring_list)
            - M == the length of the vectorized gateset
            - G == the linear dimension of a gate matrix (G x G gate matrices)
  
            and derivs[i,j,k,l] holds the derivative of the (k,l)-th entry
            of the i-th gate string product with respect to the j-th gateset
            parameter.

          * if flat == True, an array of shape S*N x M where

            - N == the number of entries in a single flattened gate (ordering is
                   the same as that used by numpy.flatten),
            - S,M == as above,
  
            and deriv[i,j] holds the derivative of the (i % G^2)-th entry of
            the (i / G^2)-th flattened gate string product  with respect to 
            the j-th gateset parameter.

        products : numpy array
          Only returned when bReturnDProdsAndProds == True.  An array of shape
          S x G x G; products[i] is the i-th gate string product.

        scaleVals : numpy array
          Only returned when bScale == True.  An array of shape S such that
          scaleVals[i] contains the multiplicative scaling needed for
          the hessians, derivatives, and/or products for the i-th gate string.
        """

        bG0 = G0
        gate_dim = self.gate_dim
        assert(not evalTree.is_split()) #product functions can't use split trees (as there's really no point)

        nGateStrings = evalTree.num_final_strings() #len(gatestring_list)
        nGateDerivCols = self.num_params(gates=gates, G0=bG0, SPAM=False) 
        deriv_shape = (nGateDerivCols, gate_dim, gate_dim)
        hessn_shape = (nGateDerivCols, nGateDerivCols, gate_dim, gate_dim)

        cacheSize = len(evalTree)
        prodCache = _np.zeros( (cacheSize, gate_dim, gate_dim) )
        dProdCache = _np.zeros( (cacheSize,) + deriv_shape )
        hProdCache = _np.zeros( (cacheSize,) + hessn_shape )
        scaleCache = _np.zeros( cacheSize, 'd' )

        #print "DEBUG: cacheSize = ",cacheSize, " gate dim = ",gate_dim, " deriv_shape = ",deriv_shape," hessn_shape = ",hessn_shape
        #print "  pc MEM estimate = ", cacheSize*gate_dim*gate_dim*8.0/(1024.0**2), "MB"
        #print "  dpc MEM estimate = ", cacheSize*_np.prod(deriv_shape)*8.0/(1024.0**2), "MB"
        #print "  hpc MEM estimate = ", cacheSize*_np.prod(hessn_shape)*8.0/(1024.0**2), "MB"
        #print "  sc MEM estimate = ", cacheSize*8.0/(1024.0**2), "MB"
        #import time
        #time.sleep(10)
        #print "Continuing..."        

        #First element of cache are given by evalTree's initial single- or zero-gate labels
        for i,gateLabel in enumerate(evalTree.get_init_labels()):
            if gateLabel == "": #special case of empty label == no gate
                prodCache[i]  = _np.identity( gate_dim )
                dProdCache[i] = _np.zeros( deriv_shape )
                hProdCache[i] = _np.zeros( hessn_shape )
            else:
                hgate = self.hproduct( (gateLabel,) ,gates=gates, G0=bG0)
                dgate = self.dproduct( (gateLabel,) ,gates=gates, G0=bG0)
                nG = max(_nla.norm(self[gateLabel]),1.0)
                prodCache[i]  = self[gateLabel] / nG; scaleCache[i] = _np.log(nG)
                dProdCache[i] = dgate / nG 
                hProdCache[i] = hgate / nG 
            
        nZeroAndSingleStrs = len(evalTree.get_init_labels())

        #evaluate gate strings using tree (skip over the zero and single-gate-strings)
        for (i,tup) in enumerate(evalTree[nZeroAndSingleStrs:],start=nZeroAndSingleStrs):

            # combine iLeft + iRight => i
            # LEXICOGRAPHICAL VS MATRIX ORDER Note: we reverse iLeft <=> iRight from evalTree because
            # (iRight,iLeft,iFinal) = tup implies gatestring[i] = gatestring[iLeft] + gatestring[iRight], but we want:
            (iRight,iLeft,iFinal) = tup   # since then matrixOf(gatestring[i]) = matrixOf(gatestring[iLeft]) * matrixOf(gatestring[iRight])
            L,R = prodCache[iLeft], prodCache[iRight]
            prodCache[i] = _np.dot(L,R)

            if prodCache[i].max() < PSMALL and prodCache[i].min() > -PSMALL:
                nL,nR = max(_nla.norm(L), _np.exp(-scaleCache[iLeft]),1e-300), max(_nla.norm(R), _np.exp(-scaleCache[iRight]),1e-300)
                sL, sR, sdL, sdR = L/nL, R/nR, dProdCache[iLeft]/nL, dProdCache[iRight]/nR
                shL, shR, sdLdR = hProdCache[iLeft]/nL, hProdCache[iRight]/nR, _np.swapaxes(_np.dot(sdL,sdR),1,2) #_np.einsum('ikm,jml->ijkl',sdL,sdR)
                prodCache[i] = _np.dot(sL,sR); dProdCache[i] = _np.dot(sdL, sR) + _np.swapaxes(_np.dot(sL, sdR),0,1)
                hProdCache[i] = _np.dot(shL, sR) + sdLdR + _np.swapaxes(sdLdR,0,1) + _np.swapaxes(_np.dot(sL,shR),0,2)
                scaleCache[i] = scaleCache[iLeft] + scaleCache[iRight] + _np.log(nL) + _np.log(nR)
                if dProdCache[i].max() < DSMALL and dProdCache[i].min() > -DSMALL:
                    _warnings.warn("Scaled dProd small in order to keep prod managable.")
                if hProdCache[i].max() < HSMALL and hProdCache[i].min() > -HSMALL:
                    _warnings.warn("Scaled hProd small in order to keep prod managable.")
            else:
                dL,dR = dProdCache[iLeft], dProdCache[iRight]
                dProdCache[i] = _np.dot(dL, R) + _np.swapaxes(_np.dot(L, dR),0,1) #dot(dS, T) + dot(S, dT)
                scaleCache[i] = scaleCache[iLeft] + scaleCache[iRight]

                hL,hR = hProdCache[iLeft], hProdCache[iRight]   
                dLdR = _np.swapaxes(_np.dot(dL,dR),1,2) #_np.einsum('ikm,jml->ijkl',dL,dR) # Note: L, R = GxG ; dL,dR = vgs x GxG ; hL,hR = vgs x vgs x GxG
                hProdCache[i] = _np.dot(hL, R) + dLdR + _np.swapaxes(dLdR,0,1) + _np.swapaxes(_np.dot(L,hR),0,2)

                if dProdCache[i].max() < DSMALL and dProdCache[i].min() > -DSMALL:
                    nL,nR = max(_nla.norm(dL), _np.exp(-scaleCache[iLeft]),1e-300), max(_nla.norm(dR), _np.exp(-scaleCache[iRight]),1e-300)
                    sL, sR, sdL, sdR = L/nL, R/nR, dL/nL, dR/nR
                    shL, shR, sdLdR = hL/nL, hR/nR, _np.swapaxes(_np.dot(sdL,sdR),1,2) #_np.einsum('ikm,jml->ijkl',sdL,sdR)
                    prodCache[i] = _np.dot(sL,sR); dProdCache[i] = _np.dot(sdL, sR) + _np.swapaxes(_np.dot(sL, sdR),0,1)
                    hProdCache[i] = _np.dot(shL, sR) + sdLdR + _np.swapaxes(sdLdR,0,1) + _np.swapaxes(_np.dot(sL,shR),0,2)
                    scaleCache[i] = scaleCache[iLeft] + scaleCache[iRight] + _np.log(nL) + _np.log(nR)
                    if prodCache[i].max() < PSMALL and prodCache[i].min() > -PSMALL:
                        _warnings.warn("Scaled prod small in order to keep dProd managable.")
                    if hProdCache[i].max() < HSMALL and hProdCache[i].min() > -HSMALL:
                        _warnings.warn("Scaled hProd small in order to keep dProd managable.")
                
        nanOrInfCacheIndices = (~_np.isfinite(prodCache)).nonzero()[0] 
        assert( len(nanOrInfCacheIndices) == 0 ) # since all scaled gates start with norm <= 1, products should all have norm <= 1
        


        #use cached data to construct return values
        finalIndxList = evalTree.get_list_of_final_value_tree_indices()
        old_err = _np.seterr(over='ignore')
        scaleExps = scaleCache.take( finalIndxList )
        scaleVals = _np.exp(scaleExps) #may overflow, but OK if infs occur here
        _np.seterr(**old_err)

        if bReturnDProdsAndProds:
            Gs  = prodCache.take(  finalIndxList, axis=0 ) #shape == ( len(gatestring_list), gate_dim, gate_dim ), Gs[i] is product for i-th gate string
            dGs = dProdCache.take( finalIndxList, axis=0 ) #shape == ( len(gatestring_list), nGateDerivCols, gate_dim, gate_dim ), dGs[i] is dprod_dGates for ith string
            hGs = hProdCache.take( finalIndxList, axis=0 ) #shape == ( len(gatestring_list), nGateDerivCols, nGateDerivCols, gate_dim, gate_dim ), hGs[i] 
                                                           # is hprod_dGates for ith string

            if not bScale:
                old_err = _np.seterr(over='ignore', invalid='ignore')
                Gs  = _np.swapaxes( _np.swapaxes(Gs,0,2) * scaleVals, 0,2)  #may overflow, but ok
                dGs = _np.swapaxes( _np.swapaxes(dGs,0,3) * scaleVals, 0,3) #may overflow or get nans (invalid), but ok
                hGs = _np.swapaxes( _np.swapaxes(hGs,0,4) * scaleVals, 0,4) #may overflow or get nans (invalid), but ok
                dGs[_np.isnan(dGs)] = 0  #convert nans to zero, as these occur b/c an inf scaleVal is mult by a zero deriv value (see below)
                hGs[_np.isnan(hGs)] = 0  #convert nans to zero, as these occur b/c an inf scaleVal is mult by a zero hessian value (see below)
                _np.seterr(**old_err)

            if flat: 
                dGs = _np.swapaxes( _np.swapaxes(dGs,0,1).reshape( (nGateDerivCols, nGateStrings*gate_dim**2) ), 0,1 ) # cols = deriv cols, rows = flattened all else
                hGs = _np.rollaxis( _np.rollaxis(hGs,0,3).reshape( (nGateDerivCols, nGateDerivCols, nGateStrings*gate_dim**2) ), 2) # cols = deriv cols, rows = all else
                
            return (hGs, dGs, Gs, scaleVals) if bScale else (hGs, dGs, Gs)

        else:
            hGs = hProdCache.take( finalIndxList, axis=0 ) #shape == ( len(gatestring_list), nGateDerivCols, nGateDerivCols, gate_dim, gate_dim )

            if not bScale:
                old_err = _np.seterr(over='ignore', invalid='ignore')
                hGs = _np.swapaxes( _np.swapaxes(hGs,0,4) * scaleVals, 0,4) #may overflow or get nans (invalid), but ok
                hGs[_np.isnan(hGs)] =  0 #convert nans to zero, as these occur b/c an inf scaleVal is mult by a zero hessian value, and we 
                                         # assume the zero hessian value trumps since we've renormed to keep all the products within decent bounds
                #assert( len( (_np.isnan(hGs)).nonzero()[0] ) == 0 ) 
                #assert( len( (_np.isinf(hGs)).nonzero()[0] ) == 0 ) 
                #hGs = clip(hGs,-1e300,1e300)
                _np.seterr(**old_err)

            if flat: hGs = _np.rollaxis( _np.rollaxis(hGs,0,3).reshape( (nGateDerivCols, nGateDerivCols, nGateStrings*gate_dim**2) ), 2) # as above

            return (hGs, scaleVals) if bScale else hGs

    

    def bulk_pr(self, spamLabel, evalTree, clipTo=None, check=False):
        """ 
        Compute the probabilities of the gate sequences given by evalTree,
        where initialization & measurement operations are always the same
        and are together specified by spamLabel.

        Parameters
        ----------
        spamLabel : string
           the label specifying the state prep and measure operations
        
        evalTree : EvalTree
           given by a prior call to bulk_evaltree.  Specifies the gate strings
           to compute the bulk operation on.

        clipTo : 2-tuple, optional
           (min,max) to clip return value if not None.

        check : boolean, optional
          If True, perform extra checks within code to verify correctness,
          generating warnings when checks fail.  Used for testing, and runs
          much slower when True.
           
        Returns
        -------
        numpy array
          An array of length equal to the number of gate strings containing
          the (float) probabilities.
        """

        nGateStrings = evalTree.num_final_strings() #len(gatestring_list)
        if evalTree.is_split():
            vp = _np.empty( nGateStrings, 'd' )

        (rhoIndex,eIndex) = self.SPAM_labels[spamLabel]
        rho = self.rhoVecs[rhoIndex]
        E   = _np.conjugate(_np.transpose(self.get_evec(eIndex)))

        for evalSubTree in evalTree.get_sub_trees():
            Gs, scaleVals = self.bulk_product(evalSubTree, bScale=True)

            #Compute probability and save in return array
            # want vp[iFinal] = float(dot(E, dot(G, rho)))  ##OLD, slightly slower version: p = trace(dot(self.SPAMs[spamLabel], G))
            #  vp[i] = sum_k,l E[0,k] Gs[i,k,l] rho[l,0] * scaleVals[i]
            #  vp[i] = sum_k E[0,k] dot(Gs, rho)[i,k,0]  * scaleVals[i]
            #  vp[i] = dot( E, dot(Gs, rho))[0,i,0]      * scaleVals[i]
            #  vp    = squeeze( dot( E, dot(Gs, rho)), axis=(0,2) ) * scaleVals
            old_err = _np.seterr(over='ignore')
            sub_vp = _np.squeeze( _np.dot(E, _np.dot(Gs, rho)), axis=(0,2) ) * scaleVals  # shape == (len(gatestring_list),) ; may overflow but OK
            _np.seterr(**old_err)
        
            if evalTree.is_split():
                vp[ evalSubTree.myFinalToParentFinalMap ] = sub_vp
            else: vp = sub_vp

        #DEBUG: catch warnings to make sure correct (inf if value is large) evaluation occurs when there's a warning
        #bPrint = False
        #with _warnings.catch_warnings():
        #    _warnings.filterwarnings('error')
        #    try:
        #        vp = squeeze( dot(E, dot(Gs, rho)), axis=(0,2) ) * scaleVals
        #    except Warning: bPrint = True
        #if bPrint:  print 'Warning in Gateset.bulk_pr : scaleVals=',scaleVals,'\n vp=',vp
            
        if clipTo is not None:  
            vp = _np.clip( vp, clipTo[0], clipTo[1])
            #nClipped = len((_np.logical_or(vp < clipTo[0], vp > clipTo[1])).nonzero()[0])
            #if nClipped > 0: print "DEBUG: bulk_pr nClipped = ",nClipped

        if check: 
            # compare with older slower version that should do the same thing (for debugging)
            gatestring_list = evalTree.generate_gatestring_list()
            check_vp = _np.array( [ self.pr(spamLabel, gateString, clipTo) for gateString in gatestring_list ] )
            if _nla.norm(vp - check_vp) > 1e-6:
                _warnings.warn( "norm(vp-check_vp) = %g - %g = %g" % (_nla.norm(vp), _nla.norm(check_vp), _nla.norm(vp - check_vp)) )
                #for i,gs in enumerate(gatestring_list):
                #    if abs(vp[i] - check_vp[i]) > 1e-6: 
                #        check = self.pr(spamLabel, gs, clipTo, bDebug=True)
                #        print "Check = ",check
                #        print "Bulk scaled gates:"
                #        print " prodcache = \n",prodCache[i] 
                #        print " scaleCache = ",scaleCache[i]
                #        print " trace = ", squeeze( dot(E, dot(Gs, rho)), axis=(0,2) )[i]
                #        print " scaleVals = ",scaleVals
                #        #for k in range(1+len(self)):
                #        print "   %s => p=%g, check_p=%g, diff=%g" % (str(gs),vp[i],check_vp[i],abs(vp[i]-check_vp[i]))
                #        raise ValueError("STOP")

        return vp


    def bulk_dpr(self, spamLabel, evalTree, 
                 gates=True,G0=True,SPAM=True,SP0=True,
                 returnPr=False,clipTo=None,check=False,memLimit=None):

        """
        Compute the derivatives of the probabilities generated by a each gate 
        sequence given by evalTree, where initialization
        & measurement operations are always the same and are
        together specified by spamLabel.

        Parameters
        ----------
        spamLabel : string
           the label specifying the state prep and measure operations

        evalTree : EvalTree
           given by a prior call to bulk_evaltree.  Specifies the gate strings
           to compute the bulk operation on.
           
        gates : bool or list, optional
          Whether/which gates should be included as gateset parameters.

          - True = all gates
          - False = no gates
          - list of gate labels = those particular gates.

        G0 : bool, optional
          Whether the first row of the included gate matrices
          should be included as gateset parameters.

        SPAM : bool, optional
          Whether rhoVecs and EVecs should be included as gateset
          parameters.

        SP0 : bool, optional
          Whether the first element of the state preparation (rho) vectors
          should be included as gateset parameters.

        returnPr : bool, optional
          when set to True, additionally return the probabilities.

        clipTo : 2-tuple, optional
           (min,max) to clip returned probability to if not None.
           Only relevant when returnPr == True.
           
        check : boolean, optional
          If True, perform extra checks within code to verify correctness,
          generating warnings when checks fail.  Used for testing, and runs
          much slower when True.

        Returns
        -------
        dprobs : numpy array
            An array of shape S x M, where

            - S == the number of gate strings
            - M == the length of the vectorized gateset

            and dprobs[i,j] holds the derivative of the i-th probability w.r.t.
            the j-th gateset parameter.

        probs : numpy array
            Only returned when returnPr == True. An array of shape S containing
            the probabilities of each gate string.
        """

        bSPAM = SPAM; bG0 = G0; bSP0 = SP0

        if self.SPAMs[spamLabel] is None: #then compute Deriv[ 1.0 - (all other spam label probabilities) ]
            otherSpamLabels = self.get_spam_labels(); del otherSpamLabels[ otherSpamLabels.index(spamLabel) ]
            assert( all([ (self.SPAMs[sl] is not None) for sl in otherSpamLabels]) )
            otherResults = [self.bulk_dpr(sl, evalTree, gates, G0, SPAM, SP0, returnPr, clipTo) for sl in otherSpamLabels]
            if returnPr:
                return ( -1.0 * _np.sum([dpr for dpr,p in otherResults],axis=0),
                          1.0 - _np.sum([p   for dpr,p in otherResults],axis=0) )
            else:
                return -1.0 * _np.sum(otherResults, axis=0)

        (rhoIndex,eIndex) = self.SPAM_labels[spamLabel]
        rho = self.rhoVecs[rhoIndex]
        E   = _np.conjugate(_np.transpose(self.get_evec(eIndex)))
        gate_dim = self.gate_dim
        nGateStrings = evalTree.num_final_strings()
        nDerivCols = self.num_params(gates, bG0, bSPAM, bSP0) 

        if evalTree.is_split():
            vp = _np.empty( nGateStrings, 'd' )
            vdp = _np.empty( (nGateStrings, nDerivCols), 'd' )  

        for evalSubTree in evalTree.get_sub_trees():
            sub_nGateStrings = evalSubTree.num_final_strings()
            dGs, Gs, scaleVals = self.bulk_dproduct(evalSubTree, gates, G0, bScale=True, bReturnProds=True, memLimit=memLimit)

            old_err = _np.seterr(over='ignore')
    
            #Compute probability and save in return array
            # want vp[iFinal] = float(dot(E, dot(G, rho)))  ##OLD, slightly slower version: p = trace(dot(self.SPAMs[spamLabel], G))
            #  vp[i] = sum_k,l E[0,k] Gs[i,k,l] rho[l,0]
            #  vp[i] = sum_k E[0,k] dot(Gs, rho)[i,k,0]
            #  vp[i] = dot( E, dot(Gs, rho))[0,i,0]
            #  vp    = squeeze( dot( E, dot(Gs, rho)), axis=(0,2) )
            if returnPr: 
                sub_vp = _np.squeeze( _np.dot(E, _np.dot(Gs, rho)), axis=(0,2) ) * scaleVals  # shape == (len(gatestring_list),) ; may overflow, but OK
    
            #Compute d(probability)/dGates and save in return list (now have G,dG => product, dprod_dGates)
            #  prod, dprod_dGates = G,dG
            # dp_dGates[i,j] = sum_k,l E[0,k] dGs[i,j,k,l] rho[l,0] 
            # dp_dGates[i,j] = sum_k E[0,k] dot( dGs, rho )[i,j,k,0]
            # dp_dGates[i,j] = dot( E, dot( dGs, rho ) )[0,i,j,0]
            # dp_dGates      = squeeze( dot( E, dot( dGs, rho ) ), axis=(0,3))
            old_err2 = _np.seterr(invalid='ignore', over='ignore')
            dp_dGates = _np.squeeze( _np.dot( E, _np.dot( dGs, rho ) ), axis=(0,3) ) * scaleVals[:,None] 
            _np.seterr(**old_err2)
               # may overflow, but OK ; shape == (len(gatestring_list), nGateDerivCols)
               # may also give invalid value due to scaleVals being inf and dot-prod being 0. In
               #  this case set to zero since we can't tell whether it's + or - inf anyway...
            dp_dGates[ _np.isnan(dp_dGates) ] = 0

            #DEBUG
            #assert( len( (_np.isnan(scaleVals)).nonzero()[0] ) == 0 ) 
            #xxx = _np.dot( E, _np.dot( dGs, rho ) )
            #assert( len( (_np.isnan(xxx)).nonzero()[0] ) == 0 )
            #if len( (_np.isnan(dp_dGates)).nonzero()[0] ) != 0:
            #    print "scaleVals = ",_np.min(scaleVals),", ",_np.max(scaleVals)
            #    print "xxx = ",_np.min(xxx),", ",_np.max(xxx)
            #    print len( (_np.isinf(xxx)).nonzero()[0] )
            #    print len( (_np.isinf(scaleVals)).nonzero()[0] )
            #    assert( len( (_np.isnan(dp_dGates)).nonzero()[0] ) == 0 )
    
            if bSPAM:                
                m = 0 if bSP0 else 1
                gate_dim_minus_m = gate_dim-m

                # Get: dp_drhos[i, gate_dim_minus_m*rhoIndex:gate_dim_minus_m*(rhoIndex+1)] = dot(E,Gs[i])[0,m:]
                # dp_drhos[i,J0+J] = sum_k E[0,k] Gs[i,k,J]
                # dp_drhos[i,J0+J] = dot(E, Gs)[0,i,J]
                # dp_drhos[:,J0+J] = squeeze(dot(E, Gs),axis=(0,))[:,J]            
                dp_drhos = _np.zeros( (sub_nGateStrings, gate_dim_minus_m * len(self.rhoVecs)) )
                dp_drhos[: , gate_dim_minus_m*rhoIndex:gate_dim_minus_m*(rhoIndex+1) ] = _np.squeeze(_np.dot(E, Gs),axis=(0,))[:,m:] * scaleVals[:,None] # may overflow, but OK
                
                # Get: dp_dEs[i, gate_dim*eIndex:gate_dim*(eIndex+1)] = transpose(dot(Gs[i],rho))
                # dp_dEs[i,J0+j] = sum_l Gs[i,j,l] rho[l,0]
                # dp_dEs[i,J0+j] = dot(Gs, rho)[i,j,0]
                # dp_dEs[:,J0+j] = squeeze(dot(Gs, rho),axis=(2,))
                dp_dEs = _np.zeros( (sub_nGateStrings, gate_dim * len(self.EVecs)) )
                dp_dAnyE = _np.squeeze(_np.dot(Gs, rho),axis=(2,)) * scaleVals[:,None] #may overflow, but OK (deriv w.r.t any of self.EVecs - independent of which)
                if eIndex == -1:
                    for ei,EVec in enumerate(self.EVecs):  #compute Deriv w.r.t. [ 1 - sum_of_other_EVecs ]
                        dp_dEs[:,gate_dim*ei:gate_dim*(ei+1)] = -1.0 * dp_dAnyE
                else:
                    dp_dEs[:,gate_dim*eIndex:gate_dim*(eIndex+1)] = dp_dAnyE                
                sub_vdp = _np.concatenate( (dp_drhos,dp_dEs,dp_dGates), axis=1 )
            else:
                sub_vdp = dp_dGates
    
            _np.seterr(**old_err)

            if evalTree.is_split():
                if returnPr: vp[ evalSubTree.myFinalToParentFinalMap ] = sub_vp
                vdp[ evalSubTree.myFinalToParentFinalMap, : ] = sub_vdp
            else: 
                if returnPr: vp = sub_vp
                vdp = sub_vdp

        if returnPr and clipTo is not None: #do this before check...
            vp = _np.clip( vp, clipTo[0], clipTo[1] )

        if check: 
            # compare with older slower version that should do the same thing (for debugging)
            gatestring_list = evalTree.generate_gatestring_list()
            check_vdp = _np.concatenate( [ self.dpr(spamLabel, gateString, gates,G0,SPAM,SP0,False,clipTo) for gateString in gatestring_list ], axis=0 )
            check_vp = _np.array( [ self.pr(spamLabel, gateString, clipTo) for gateString in gatestring_list ] )

            if returnPr and _nla.norm(vp - check_vp) > 1e-6:
                _warnings.warn("norm(vp-check_vp) = %g - %g = %g" % (_nla.norm(vp), _nla.norm(check_vp), _nla.norm(vp - check_vp)))
                #for i,gs in enumerate(gatestring_list):
                #    if abs(vp[i] - check_vp[i]) > 1e-6: 
                #        print "   %s => p=%g, check_p=%g, diff=%g" % (str(gs),vp[i],check_vp[i],abs(vp[i]-check_vp[i]))
            if _nla.norm(vdp - check_vdp) > 1e-6:
                _warnings.warn("Norm(vdp-check_vdp) = %g - %g = %g" % (_nla.norm(vdp), _nla.norm(check_vdp), _nla.norm(vdp - check_vdp)))

        if returnPr: return vdp, vp
        else:        return vdp



    def bulk_hpr(self, spamLabel, evalTree, 
                 gates=True,G0=True,SPAM=True,SP0=True,
                 returnPr=False,returnDeriv=False,
                 clipTo=None,check=False):

        """
        Compute the derivatives of the probabilities generated by a each gate 
        sequence given by evalTree, where initialization & measurement 
        operations are always the same and are together specified by spamLabel.

        Parameters
        ----------
        spamLabel : string
          the label specifying the state prep and measure operations
                   
        evalTree : EvalTree
          given by a prior call to bulk_evaltree.  Specifies the gate strings
          to compute the bulk operation on.

        gates : bool or list, optional
          Whether/which gates should be included as gateset parameters.

          - True = all gates
          - False = no gates
          - list of gate labels = those particular gates.

        G0 : bool, optional
          Whether the first row of the included gate matrices
          should be included as gateset parameters.

        SPAM : bool, optional
          Whether rhoVecs and EVecs should be included as gateset
          parameters.

        SP0 : bool, optional
          Whether the first element of the state preparation (rho) vectors
          should be included as gateset parameters.

        returnPr : bool, optional
          when set to True, additionally return the probabilities.

        returnDeriv : bool, optional
          when set to True, additionally return the probability derivatives.

        clipTo : 2-tuple, optional
          (min,max) to clip returned probability to if not None.
          Only relevant when returnPr == True.
           
        check : boolean, optional
          If True, perform extra checks within code to verify correctness,
          generating warnings when checks fail.  Used for testing, and runs
          much slower when True.

        Returns
        -------
        hessians : numpy array
            a S x M x M array, where 

            - S == the number of gate strings
            - M == the length of the vectorized gateset

            and hessians[i,j,k] is the derivative of the i-th probability
            w.r.t. the k-th then the j-th gateset parameter.

        derivs : numpy array
            only returned if returnDeriv == True. A S x M array where
            derivs[i,j] holds the derivative of the i-th probability
            w.r.t. the j-th gateset parameter.

        probabilities : numpy array
            only returned if returnPr == True.  A length-S array 
            containing the probabilities for each gate string.
        """

        bSPAM = SPAM; bG0 = G0; bSP0 = SP0

        if self.SPAMs[spamLabel] is None: #then compute Deriv[ 1.0 - (all other spam label probabilities) ]
            otherSpamLabels = self.get_spam_labels(); del otherSpamLabels[ otherSpamLabels.index(spamLabel) ]
            assert( all([ (self.SPAMs[sl] is not None) for sl in otherSpamLabels]) )
            otherResults = [self.bulk_hpr(sl, evalTree, gates, G0, SPAM, SP0, returnPr, returnDeriv, clipTo) for sl in otherSpamLabels]
            if returnDeriv: 
                if returnPr: return ( -1.0 * _np.sum([hpr for hpr,dpr,p in otherResults],axis=0),
                                      -1.0 * _np.sum([dpr for hpr,dpr,p in otherResults],axis=0), 
                                       1.0 - _np.sum([p   for hpr,dpr,p in otherResults],axis=0) )
                else:        return ( -1.0 * _np.sum([hpr for hpr,dpr in otherResults],axis=0),
                                      -1.0 * _np.sum([dpr for hpr,dpr in otherResults],axis=0)   )
            else:
                if returnPr: return ( -1.0 * _np.sum([hpr for hpr,p in otherResults],axis=0),
                                       1.0 - _np.sum([p   for hpr,p in otherResults],axis=0)  )
                else:        return   -1.0 * _np.sum(otherResults,axis=0)

        (rhoIndex,eIndex) = self.SPAM_labels[spamLabel]
        rho = self.rhoVecs[rhoIndex]
        E   = _np.conjugate(_np.transpose(self.get_evec(eIndex)))
        gate_dim = self.gate_dim
        nGateStrings = evalTree.num_final_strings()
        nDerivCols = self.num_params(gates, bG0, bSPAM, bSP0) 

        if evalTree.is_split():
            vp = _np.empty( nGateStrings, 'd' )
            vdp = _np.empty( (nGateStrings, nDerivCols), 'd' )  
            vhp = _np.empty( (nGateStrings, nDerivCols, nDerivCols), 'd' )

        for evalSubTree in evalTree.get_sub_trees():
            sub_nGateStrings = evalSubTree.num_final_strings()
            hGs, dGs, Gs, scaleVals = self.bulk_hproduct(evalSubTree, gates, G0, bScale=True, bReturnDProdsAndProds=True)
            old_err = _np.seterr(over='ignore')
    
            #Compute probability and save in return array
            # want vp[iFinal] = float(dot(E, dot(G, rho)))  ##OLD, slightly slower version: p = trace(dot(self.SPAMs[spamLabel], G))
            #  vp[i] = sum_k,l E[0,k] Gs[i,k,l] rho[l,0]
            #  vp[i] = sum_k E[0,k] dot(Gs, rho)[i,k,0]
            #  vp[i] = dot( E, dot(Gs, rho))[0,i,0]
            #  vp    = squeeze( dot( E, dot(Gs, rho)), axis=(0,2) )
            if returnPr: 
                sub_vp = _np.squeeze( _np.dot(E, _np.dot(Gs, rho)), axis=(0,2) ) * scaleVals  # shape == (len(gatestring_list),) ; may overflow, but OK
    
            #Compute d(probability)/dGates and save in return list (now have G,dG => product, dprod_dGates)
            #  prod, dprod_dGates = G,dG
            # dp_dGates[i,j] = sum_k,l E[0,k] dGs[i,j,k,l] rho[l,0] 
            # dp_dGates[i,j] = sum_k E[0,k] dot( dGs, rho )[i,j,k,0]
            # dp_dGates[i,j] = dot( E, dot( dGs, rho ) )[0,i,j,0]
            # dp_dGates      = squeeze( dot( E, dot( dGs, rho ) ), axis=(0,3))
            if returnDeriv:
                old_err2 = _np.seterr(invalid='ignore', over='ignore')
                dp_dGates = _np.squeeze( _np.dot( E, _np.dot( dGs, rho ) ), axis=(0,3) ) * scaleVals[:,None] 
                _np.seterr(**old_err2)
                # may overflow, but OK ; shape == (len(gatestring_list), nGateDerivCols)
                # may also give invalid value due to scaleVals being inf and dot-prod being 0. In
                #  this case set to zero since we can't tell whether it's + or - inf anyway...
                dp_dGates[ _np.isnan(dp_dGates) ] = 0
    
    
            #Compute d2(probability)/dGates2 and save in return list
            # d2pr_dGates2[i,j,k] = sum_l,m E[0,l] hGs[i,j,k,l,m] rho[m,0] 
            # d2pr_dGates2[i,j,k] = sum_l E[0,l] dot( dGs, rho )[i,j,k,l,0]
            # d2pr_dGates2[i,j,k] = dot( E, dot( dGs, rho ) )[0,i,j,k,0]
            # d2pr_dGates2        = squeeze( dot( E, dot( dGs, rho ) ), axis=(0,4))
            old_err2 = _np.seterr(invalid='ignore', over='ignore')
            d2pr_dGates2 = _np.squeeze( _np.dot( E, _np.dot( hGs, rho ) ), axis=(0,4) ) * scaleVals[:,None,None] 
            _np.seterr(**old_err2)
            # may overflow, but OK ; shape == (len(gatestring_list), nGateDerivCols, nGateDerivCols)
            # may also give invalid value due to scaleVals being inf and dot-prod being 0. In
            #  this case set to zero since we can't tell whether it's + or - inf anyway...
            d2pr_dGates2[ _np.isnan(d2pr_dGates2) ] = 0
    
    
            if bSPAM: 
                m = 0 if bSP0 else 1
                gate_dim_minus_m = gate_dim-m
                
                if returnDeriv: #same as in bulk_dpr - see comments there for details
                    dp_drhos = _np.zeros( (sub_nGateStrings, gate_dim_minus_m * len(self.rhoVecs)) )
                    dp_drhos[: , gate_dim_minus_m*rhoIndex:gate_dim_minus_m*(rhoIndex+1) ] = _np.squeeze(_np.dot(E, Gs),axis=(0,))[:,m:] * scaleVals[:,None]
                    dp_dEs = _np.zeros( (sub_nGateStrings, gate_dim * len(self.EVecs)) )
                    dp_dAnyE = _np.squeeze(_np.dot(Gs, rho),axis=(2,)) * scaleVals[:,None] #may overflow, but OK (deriv w.r.t any of self.EVecs)
                    if eIndex == -1:
                        for ei,EVec in enumerate(self.EVecs):  #compute Deriv w.r.t. [ 1 - sum_of_other_EVecs ]
                            dp_dEs[:,gate_dim*ei:gate_dim*(ei+1)] = -1.0 * dp_dAnyE
                    else:
                        dp_dEs[:,gate_dim*eIndex:gate_dim*(eIndex+1)] = dp_dAnyE
                    vdp = _np.concatenate( (dp_drhos,dp_dEs,dp_dGates), axis=1 )
                    sub_vdp = vdp
    
                vec_gs_size = dGs.shape[1]
    
                # Get: d2pr_drhos[i, j, gate_dim_minus_m*rhoIndex:gate_dim_minus_m*(rhoIndex+1)] = dot(E,dGs[i,j])[0,m:]
                # d2pr_drhos[i,j,J0+J] = sum_k E[0,k] dGs[i,j,k,J]
                # d2pr_drhos[i,j,J0+J] = dot(E, dGs)[0,i,j,J]
                # d2pr_drhos[:,:,J0+J] = squeeze(dot(E, dGs),axis=(0,))[:,:,J]            
                d2pr_drhos = _np.zeros( (sub_nGateStrings, vec_gs_size, (gate_dim-m) * len(self.rhoVecs)) )
                d2pr_drhos[:, :, (gate_dim-m)*rhoIndex:(gate_dim-m)*(rhoIndex+1)] = _np.squeeze( _np.dot(E,dGs), axis=(0,))[:,:,m:] * scaleVals[:,None,None] #overflow OK
    
                # Get: d2pr_dEs[i, j, gate_dim*eIndex:gate_dim*(eIndex+1)] = dot(dGs[i,j],rho)
                # d2pr_dEs[i,j,J0+J] = sum_l dGs[i,j,k,l] rho[l,0]
                # d2pr_dEs[i,j,J0+J] = dot(Gs, rho)[i,j,k,0]
                # d2pr_dEs[:,:,J0+J] = squeeze(dot(Gs, rho),axis=(3,))
                d2pr_dEs = _np.zeros( (sub_nGateStrings, vec_gs_size, gate_dim * len(self.EVecs)) )
                dp_dAnyE = _np.squeeze(_np.dot(dGs,rho), axis=(3,)) * scaleVals[:,None,None] #overflow OK
                if eIndex == -1:
                    for ei,EVec in enumerate(self.EVecs):
                        d2pr_dEs[:, :, gate_dim*ei:gate_dim*(ei+1)] = -1.0 * dp_dAnyE
                else:
                    d2pr_dEs[:, :, gate_dim*eIndex:gate_dim*(eIndex+1)] = dp_dAnyE

    
                # Get: d2pr_dErhos[i, gate_dim*eIndex:gate_dim*(eIndex+1), gate_dim_minus_m*rhoIndex:gate_dim_minus_m*(rhoIndex+1)] = prod[i,:,m:]
                # d2pr_dErhos[i,J0+J,K0+K] = prod[i,:,m:]
                # d2pr_dErhos[:,J0+J,K0+K] = prod[:,:,m:]
                d2pr_dErhos = _np.zeros( (sub_nGateStrings, gate_dim * len(self.EVecs), (gate_dim-m) * len(self.rhoVecs)) )
                dp_dAnyE = Gs[:,:,m:] * scaleVals[:,None,None] #overflow OK
                if eIndex == -1:
                    for ei,EVec in enumerate(self.EVecs):
                        d2pr_dErhos[:, gate_dim*ei:gate_dim*(ei+1), (gate_dim-m)*rhoIndex:(gate_dim-m)*(rhoIndex+1)] = -1.0 * dp_dAnyE
                else:
                    d2pr_dErhos[:, gate_dim*eIndex:gate_dim*(eIndex+1), (gate_dim-m)*rhoIndex:(gate_dim-m)*(rhoIndex+1)] = dp_dAnyE
    
                d2pr_d2rhos = _np.zeros( (sub_nGateStrings, (gate_dim-m) * len(self.rhoVecs), (gate_dim-m) * len(self.rhoVecs)) )
                d2pr_d2Es   = _np.zeros( (sub_nGateStrings,  gate_dim * len(self.EVecs), gate_dim * len(self.EVecs)) )
    
                ret_row1 = _np.concatenate( ( d2pr_d2rhos, _np.transpose(d2pr_dErhos,(0,2,1)), _np.transpose(d2pr_drhos,(0,2,1)) ), axis=2) # wrt rho
                ret_row2 = _np.concatenate( ( d2pr_dErhos, d2pr_d2Es, _np.transpose(d2pr_dEs,(0,2,1)) ), axis=2 ) # wrt E
                ret_row3 = _np.concatenate( ( d2pr_drhos,d2pr_dEs,d2pr_dGates2), axis=2 ) #wrt gates
                sub_vhp = _np.concatenate( (ret_row1, ret_row2, ret_row3), axis=1 )
    
            else:
                if returnDeriv: sub_vdp = dp_dGates
                sub_vhp = d2pr_dGates2
    
            _np.seterr(**old_err)

            if evalTree.is_split():
                if returnPr: vp[ evalSubTree.myFinalToParentFinalMap ] = sub_vp
                if returnDeriv: vdp[ evalSubTree.myFinalToParentFinalMap, : ] = sub_vdp
                vhp[ evalSubTree.myFinalToParentFinalMap, :, : ] = sub_vhp
            else: 
                if returnPr: vp = sub_vp
                if returnDeriv: vdp = sub_vdp
                vhp = sub_vhp
        

        if returnPr and clipTo is not None:  # do this before check...
            vp = _np.clip( vp, clipTo[0], clipTo[1] )

        if check: 
            # compare with older slower version that should do the same thing (for debugging)
            gatestring_list = evalTree.generate_gatestring_list()
            check_vhp = _np.concatenate( [ self.hpr(spamLabel, gateString, gates,G0,SPAM,SP0,False,False,clipTo) for gateString in gatestring_list ], axis=0 )
            check_vdp = _np.concatenate( [ self.dpr(spamLabel, gateString, gates,G0,SPAM,SP0,False,clipTo) for gateString in gatestring_list ], axis=0 )
            check_vp = _np.array( [ self.pr(spamLabel, gateString, clipTo) for gateString in gatestring_list ] )

            if returnPr and _nla.norm(vp - check_vp) > 1e-6:
                _warnings.warn("norm(vp-check_vp) = %g - %g = %g" % (_nla.norm(vp), _nla.norm(check_vp), _nla.norm(vp - check_vp)))
                #for i,gs in enumerate(gatestring_list):
                #    if abs(vp[i] - check_vp[i]) > 1e-6: 
                #        print "   %s => p=%g, check_p=%g, diff=%g" % (str(gs),vp[i],check_vp[i],abs(vp[i]-check_vp[i]))
            if returnDeriv and _nla.norm(vdp - check_vdp) > 1e-6:
                _warnings.warn("norm(vdp-check_vdp) = %g - %g = %g" % (_nla.norm(vdp), _nla.norm(check_vdp), _nla.norm(vdp - check_vdp)))
            if _nla.norm(vhp - check_vhp) > 1e-6:
                _warnings.warn("norm(vhp-check_vhp) = %g - %g = %g" % (_nla.norm(vhp), _nla.norm(check_vhp), _nla.norm(vhp - check_vhp)))

        if returnDeriv: 
            if returnPr: return vhp, vdp, vp
            else:        return vhp, vdp
        else:
            if returnPr: return vhp, vp
            else:        return vhp

    def bulk_probs(self, evalTree, clipTo=None, check=False):
        """ 
        Construct a dictionary containing the bulk-probabilities
        for every spam label (each possible initialization &
        measurement pair) for each gate sequence given by 
        evalTree.

        Parameters
        ----------
        evalTree : EvalTree
           given by a prior call to bulk_evaltree.  Specifies the gate strings
           to compute the bulk operation on.

        clipTo : 2-tuple, optional
           (min,max) to clip return value if not None.

        check : boolean, optional
          If True, perform extra checks within code to verify correctness,
          generating warnings when checks fail.  Used for testing, and runs
          much slower when True.
           
        Returns
        -------
        probs : dictionary
            A dictionary such that 
            probs[SL] = bulk_pr(SL,evalTree,clipTo,check)
            for each spam label (string) SL.
        """
        probs = { }
        if not self.assumeSumToOne:
            for spamLabel in self.SPAMs:
                probs[spamLabel] = self.bulk_pr(spamLabel, evalTree, clipTo, check)
        else:
            spam_labels_to_loop = self.SPAMs.keys()
            s = _np.zeros( evalTree.num_final_strings(), 'd'); lastLabel = None
            for spamLabel in spam_labels_to_loop:
                if self.SPAMs[spamLabel] is None:
                    assert(lastLabel is None) # ensure there is at most one dummy spam label
                    lastLabel = spamLabel; continue
                probs[spamLabel] = self.bulk_pr(spamLabel, evalTree, clipTo, check)
                s += probs[spamLabel]
            if lastLabel is not None: probs[lastLabel] = 1.0 - s  #last spam label is computed so sum == 1
        return probs


    def bulk_dprobs(self, evalTree, 
                    gates=True,G0=True,SPAM=True,SP0=True,
                    returnPr=False,clipTo=None,
                    check=False,memLimit=None):

        """
        Construct a dictionary containing the bulk-probability-
        derivatives for every spam label (each possible 
        initialization & measurement pair) for each gate
        sequence given by evalTree.

        Parameters
        ----------
        evalTree : EvalTree
           given by a prior call to bulk_evaltree.  Specifies the gate strings
           to compute the bulk operation on.
           
        gates : bool or list, optional
          Whether/which gates should be included as gateset parameters.

          - True = all gates
          - False = no gates
          - list of gate labels = those particular gates.

        G0 : bool, optional
          Whether the first row of the included gate matrices
          should be included as gateset parameters.

        SPAM : bool, optional
          Whether rhoVecs and EVecs should be included as gateset
          parameters.

        SP0 : bool, optional
          Whether the first element of the state preparation (rho) vectors
          should be included as gateset parameters.

        returnPr : bool, optional
          when set to True, additionally return the probabilities.

        clipTo : 2-tuple, optional
           (min,max) to clip returned probability to if not None.
           Only relevant when returnPr == True.
           
        check : boolean, optional
          If True, perform extra checks within code to verify correctness,
          generating warnings when checks fail.  Used for testing, and runs
          much slower when True.

        memLimit : int, optional
          A rough memory limit in bytes which restricts the amount of
          intermediate values that are computed and stored.

        Returns
        -------
        dprobs : dictionary
            A dictionary such that 
            ``dprobs[SL] = bulk_dpr(SL,evalTree,gates,G0,SPAM,SP0,returnPr,clipTo,check,memLimit)``
            for each spam label (string) SL.
        """
        dprobs = { }
        if not self.assumeSumToOne:
            for spamLabel in self.SPAMs:
                dprobs[spamLabel] = self.bulk_dpr(spamLabel, evalTree,
                                             gates,G0,SPAM,SP0,returnPr,clipTo,check,memLimit)
        else:
            spam_labels_to_loop = self.SPAMs.keys()
            ds = None; lastLabel = None
            s = _np.zeros( evalTree.num_final_strings(), 'd')
            for spamLabel in spam_labels_to_loop:
                if self.SPAMs[spamLabel] is None:
                    assert(lastLabel is None) # ensure there is at most one dummy spam label
                    lastLabel = spamLabel; continue
                dprobs[spamLabel] = self.bulk_dpr(spamLabel, evalTree,
                                             gates,G0,SPAM,SP0,returnPr,clipTo,check,memLimit)
                if returnPr:
                    ds = dprobs[spamLabel][0] if ds is None else ds + dprobs[spamLabel][0]
                    s += dprobs[spamLabel][1]
                else:
                    ds = dprobs[spamLabel] if ds is None else ds + dprobs[spamLabel]                    
            if lastLabel is not None:
                dprobs[lastLabel] = (-ds,1.0-s) if returnPr else -ds
        return dprobs


    def bulk_hprobs(self, evalTree, 
                    gates=True,G0=True,SPAM=True,SP0=True,
                    returnPr=False,returnDeriv=False,clipTo=None,
                    check=False):

        """
        Construct a dictionary containing the bulk-probability-
        Hessians for every spam label (each possible 
        initialization & measurement pair) for each gate
        sequence given by evalTree.

        Parameters
        ----------
        evalTree : EvalTree
           given by a prior call to bulk_evaltree.  Specifies the gate strings
           to compute the bulk operation on.
           
        gates : bool or list, optional
          Whether/which gates should be included as gateset parameters.

          - True = all gates
          - False = no gates
          - list of gate labels = those particular gates.

        G0 : bool, optional
          Whether the first row of the included gate matrices
          should be included as gateset parameters.

        SPAM : bool, optional
          Whether rhoVecs and EVecs should be included as gateset
          parameters.

        SP0 : bool, optional
          Whether the first element of the state preparation (rho) vectors
          should be included as gateset parameters.

        returnPr : bool, optional
          when set to True, additionally return the probabilities.

        returnDeriv : bool, optional
          when set to True, additionally return the probability derivatives.

        clipTo : 2-tuple, optional
           (min,max) to clip returned probability to if not None.
           Only relevant when returnPr == True.
           
        check : boolean, optional
          If True, perform extra checks within code to verify correctness,
          generating warnings when checks fail.  Used for testing, and runs
          much slower when True.

        Returns
        -------
        hprobs : dictionary
            A dictionary such that 
            ``hprobs[SL] = bulk_hpr(SL,evalTree,gates,G0,SPAM,SP0,returnPr,returnDeriv,clipTo,check)``
            for each spam label (string) SL.
        """
        hprobs = { }
        if not self.assumeSumToOne:
            for spamLabel in self.SPAMs:
                hprobs[spamLabel] = self.bulk_hpr(spamLabel, evalTree,
                                                  gates,G0,SPAM,SP0,returnPr,returnDeriv,
                                                  clipTo,check)
        else:
            spam_labels_to_loop = self.SPAMs.keys()
            hs = None; ds = None; lastLabel = None
            s = _np.zeros( evalTree.num_final_strings(), 'd')
            for spamLabel in spam_labels_to_loop:
                if self.SPAMs[spamLabel] is None:
                    assert(lastLabel is None) # ensure there is at most one dummy spam label
                    lastLabel = spamLabel; continue
                hprobs[spamLabel] = self.bulk_hpr(spamLabel, evalTree,
                                                  gates,G0,SPAM,SP0,returnPr,returnDeriv,
                                                  clipTo,check)

                if returnPr:
                    if returnDeriv:
                        hs = hprobs[spamLabel][0] if hs is None else hs + hprobs[spamLabel][0]
                        ds = hprobs[spamLabel][1] if ds is None else ds + hprobs[spamLabel][1]
                        s += hprobs[spamLabel][2]
                    else:
                        hs = hprobs[spamLabel][0] if hs is None else hs + hprobs[spamLabel][0]
                        s += hprobs[spamLabel][1]
                else:
                    if returnDeriv:
                        hs = hprobs[spamLabel][0] if hs is None else hs + hprobs[spamLabel][0]
                        ds = hprobs[spamLabel][1] if ds is None else ds + hprobs[spamLabel][1]
                    else:
                        hs = hprobs[spamLabel] if hs is None else hs + hprobs[spamLabel]

            if lastLabel is not None: 
                if returnPr:
                    hprobs[lastLabel] = (-hs,-ds,1.0-s) if returnDeriv else (-hs,1.0-s)
                else:
                    hprobs[lastLabel] = (-hs,-ds) if returnDeriv else -hs

        return hprobs



    def bulk_fill_probs(self, mxToFill, spam_label_rows, 
                       evalTree, clipTo=None, check=False):
        """ 
        Identical to bulk_probs(...) except results are 
        placed into rows of a pre-allocated array instead
        of being returned in a dictionary.

        Specifically, the probabilities for all gate strings
        and a given SPAM label are placed into the row of 
        mxToFill specified by spam_label_rows[spamLabel].

        Parameters
        ----------
        mxToFill : numpy ndarray
          an already-allocated KxS numpy array, where K is larger
          than the maximum value in spam_label_rows and S is equal
          to the number of gate strings (i.e. evalTree.num_final_strings())

        spam_label_rows : dictionary
          a dictionary with keys == spam labels and values which 
          are integer row indices into mxToFill, specifying the
          correspondence between rows of mxToFill and spam labels.

        evalTree : EvalTree
           given by a prior call to bulk_evaltree.  Specifies the gate strings
           to compute the bulk operation on.

        clipTo : 2-tuple, optional
           (min,max) to clip return value if not None.

        check : boolean, optional
          If True, perform extra checks within code to verify correctness,
          generating warnings when checks fail.  Used for testing, and runs
          much slower when True.
           
        Returns
        -------
        None
        """
        if not self.assumeSumToOne:
            for spamLabel,rowIndex in spam_label_rows.iteritems():
                mxToFill[rowIndex] = self.bulk_pr(spamLabel, evalTree, clipTo, check)
        else:
            spam_labels_to_loop = self.SPAMs.keys()
            s = _np.zeros( evalTree.num_final_strings(), 'd'); lastLabel = None
            for spamLabel in spam_labels_to_loop: #Note: must loop through all spam labels, even if not requested
                if self.SPAMs[spamLabel] is None:
                    assert(lastLabel is None) # ensure there is at most one dummy spam label
                    lastLabel = spamLabel; continue
                probs = self.bulk_pr(spamLabel, evalTree, clipTo, check)
                s += probs

                if spam_label_rows.has_key(spamLabel):
                    mxToFill[ spam_label_rows[spamLabel] ] = probs

            if lastLabel is not None and spam_label_rows.has_key(lastLabel):
                mxToFill[ spam_label_rows[lastLabel] ] = 1.0 - s  #last spam label is computed so sum == 1


    def bulk_fill_dprobs(self, mxToFill, spam_label_rows,
                        evalTree, gates=True,G0=True,SPAM=True,SP0=True,
                        prMxToFill=None,clipTo=None,
                        check=False,memLimit=None):

        """
        Identical to bulk_dprobs(...) except results are 
        placed into rows of a pre-allocated array instead
        of being returned in a dictionary.

        Specifically, the probability derivatives for all gate
        strings and a given SPAM label are placed into 
        mxToFill[ spam_label_rows[spamLabel] ].  
        Optionally, probabilities can be placed into 
        prMxToFill[ spam_label_rows[spamLabel] ]

        Parameters
        ----------
        mxToFill : numpy array
          an already-allocated KxSxM numpy array, where K is larger
          than the maximum value in spam_label_rows, S is equal
          to the number of gate strings (i.e. evalTree.num_final_strings()),
          and M is the length of the vectorized gateset.

        spam_label_rows : dictionary
          a dictionary with keys == spam labels and values which 
          are integer row indices into mxToFill, specifying the
          correspondence between rows of mxToFill and spam labels.

        evalTree : EvalTree
           given by a prior call to bulk_evaltree.  Specifies the gate strings
           to compute the bulk operation on.

        gates : bool or list, optional
          Whether/which gates should be included as gateset parameters.

          - True = all gates
          - False = no gates
          - list of gate labels = those particular gates.

        G0 : bool, optional
          Whether the first row of the included gate matrices
          should be included as gateset parameters.

        SPAM : bool, optional
          Whether rhoVecs and EVecs should be included as gateset
          parameters.

        SP0 : bool, optional
          Whether the first element of the state preparation (rho) vectors
          should be included as gateset parameters.

        prMxToFill : numpy array, optional
          when not None, an already-allocated KxS numpy array that is filled
          with the probabilities as per spam_label_rows, similar to
          bulk_fill_probs(...).

        clipTo : 2-tuple, optional
          (min,max) to clip returned probability to if not None.
          Only relevant when prMxToFill is not None.
           
        check : boolean, optional
          If True, perform extra checks within code to verify correctness,
          generating warnings when checks fail.  Used for testing, and runs
          much slower when True.

        Returns
        -------
        None
        """
        if not self.assumeSumToOne:
            if prMxToFill is not None:
                for spamLabel,rowIndex in spam_label_rows.iteritems():
                    mxToFill[rowIndex], prMxToFill[rowIndex] = self.bulk_dpr(spamLabel, evalTree,
                                                                             gates,G0,SPAM,SP0,True,clipTo,check,memLimit)
            else:
                for spamLabel,rowIndex in spam_label_rows.iteritems():
                    mxToFill[rowIndex] = self.bulk_dpr(spamLabel, evalTree,
                                                       gates,G0,SPAM,SP0,False,clipTo,check,memLimit)

        else:
            spam_labels_to_loop = self.SPAMs.keys()
            ds = None; lastLabel = None
            s = _np.zeros( evalTree.num_final_strings(), 'd')

            if prMxToFill is not None: #then compute & fill probabilities too
                for spamLabel in spam_labels_to_loop: #Note: must loop through all spam labels, even if not requested, in case prMxToFill is not None
                    if self.SPAMs[spamLabel] is None:
                        assert(lastLabel is None) # ensure there is at most one dummy spam label
                        lastLabel = spamLabel; continue
                    dprobs, probs = self.bulk_dpr(spamLabel, evalTree,
                                                  gates,G0,SPAM,SP0,True,clipTo,check,memLimit)
                    ds = dprobs if ds is None else ds + dprobs
                    s += probs
                    if spam_label_rows.has_key(spamLabel):
                        mxToFill[ spam_label_rows[spamLabel] ] = dprobs
                        prMxToFill[ spam_label_rows[spamLabel] ] = probs

                if lastLabel is not None and spam_label_rows.has_key(lastLabel):
                    mxToFill[ spam_label_rows[lastLabel] ] = -ds
                    prMxToFill[ spam_label_rows[lastLabel] ] = 1.0-s

            else: #just compute derivatives of probabilities
                for spamLabel in spam_labels_to_loop: #Note: must loop through all spam labels, even if not requested, in case prMxToFill is not None
                    if self.SPAMs[spamLabel] is None:
                        assert(lastLabel is None) # ensure there is at most one dummy spam label
                        lastLabel = spamLabel; continue
                    dprobs = self.bulk_dpr(spamLabel, evalTree,
                                           gates,G0,SPAM,SP0,False,clipTo,check,memLimit)
                    ds = dprobs if ds is None else ds + dprobs
                    if spam_label_rows.has_key(spamLabel):
                        mxToFill[ spam_label_rows[spamLabel] ] = dprobs

                if lastLabel is not None and spam_label_rows.has_key(lastLabel):
                    mxToFill[ spam_label_rows[lastLabel] ] = -ds


    def bulk_fill_hprobs(self, mxToFill, spam_label_rows,
                        evalTree=None, gates=True,G0=True,SPAM=True,SP0=True,
                        prMxToFill=None, derivMxToFill=None, clipTo=None,
                        check=False):

        """
        Identical to bulk_hprobs(...) except results are 
        placed into rows of a pre-allocated array instead
        of being returned in a dictionary.

        Specifically, the probability hessians for all gate
        strings and a given SPAM label are placed into 
        mxToFill[ spam_label_rows[spamLabel] ].  
        Optionally, probabilities and/or derivatives can be placed into 
        prMxToFill[ spam_label_rows[spamLabel] ] and
        derivMxToFill[ spam_label_rows[spamLabel] ] respectively.

        Parameters
        ----------
        mxToFill : numpy array
          an already-allocated KxSxMxM numpy array, where K is larger
          than the maximum value in spam_label_rows, S is equal
          to the number of gate strings (i.e. evalTree.num_final_strings()),
          and M is the length of the vectorized gateset.

        spam_label_rows : dictionary
          a dictionary with keys == spam labels and values which 
          are integer row indices into mxToFill, specifying the
          correspondence between rows of mxToFill and spam labels.

        evalTree : EvalTree
           given by a prior call to bulk_evaltree.  Specifies the gate strings
           to compute the bulk operation on.

        gates : bool or list, optional
          Whether/which gates should be included as gateset parameters.

          - True = all gates
          - False = no gates
          - list of gate labels = those particular gates.

        G0 : bool, optional
          Whether the first row of the included gate matrices
          should be included as gateset parameters.

        SPAM : bool, optional
          Whether rhoVecs and EVecs should be included as gateset
          parameters.

        SP0 : bool, optional
          Whether the first element of the state preparation (rho) vectors
          should be included as gateset parameters.

        prMxToFill : numpy array, optional
          when not None, an already-allocated KxS numpy array that is filled
          with the probabilities as per spam_label_rows, similar to
          bulk_fill_probs(...).

        derivMxToFill : numpy array, optional
          when not None, an already-allocated KxSxM numpy array that is filled
          with the probability derivatives as per spam_label_rows, similar to
          bulk_fill_dprobs(...).

        clipTo : 2-tuple
          (min,max) to clip returned probability to if not None.
          Only relevant when prMxToFill is not None.
           
        check : boolean, optional
          If True, perform extra checks within code to verify correctness,
          generating warnings when checks fail.  Used for testing, and runs
          much slower when True.

        Returns
        -------
        None
        """
        if not self.assumeSumToOne:
            if prMxToFill is not None:
                if derivMxToFill is not None:
                    for spamLabel,rowIndex in spam_label_rows.iteritems():
                        mxToFill[rowIndex], derivMxToFill[rowIndex], prMxToFill[rowIndex] = self.bulk_hpr(spamLabel, evalTree,
                                                                                                          gates,G0,SPAM,SP0,True,True,clipTo,check)
                else:
                    for spamLabel,rowIndex in spam_label_rows.iteritems():
                        mxToFill[rowIndex], prMxToFill[rowIndex] = self.bulk_hpr(spamLabel, evalTree,
                                                                                 gates,G0,SPAM,SP0,True,False,clipTo,check)

            else:
                if derivMxToFill is not None:
                    for spamLabel,rowIndex in spam_label_rows.iteritems():
                        mxToFill[rowIndex], derivMxToFill[rowIndex] = self.bulk_hpr(spamLabel, evalTree,
                                                                                 gates,G0,SPAM,SP0,False,True,clipTo,check)
                else:
                    for spamLabel,rowIndex in spam_label_rows.iteritems():
                        mxToFill[rowIndex] = self.bulk_hpr(spamLabel, evalTree,
                                                           gates,G0,SPAM,SP0,False,False,clipTo,check)

        else:  # assumeSumToOne == True

            spam_labels_to_loop = self.SPAMs.keys()
            hs = None; ds = None; lastLabel = None
            s = _np.zeros( evalTree.num_final_strings(), 'd')

            if prMxToFill is not None: #then compute & fill probabilities too
                if derivMxToFill is not None: #then compute & fill derivatives too
                    for spamLabel in spam_labels_to_loop: #Note: must loop through all spam labels, even if not requested, in case prMxToFill is not None
                        if self.SPAMs[spamLabel] is None:
                            assert(lastLabel is None) # ensure there is at most one dummy spam label
                            lastLabel = spamLabel; continue
                        hprobs, dprobs, probs = self.bulk_hpr(spamLabel, evalTree,
                                                      gates,G0,SPAM,SP0,True,True,clipTo,check)
                        hs = hprobs if hs is None else hs + hprobs
                        ds = dprobs if ds is None else ds + dprobs
                        s += probs
                        if spam_label_rows.has_key(spamLabel):
                            mxToFill[ spam_label_rows[spamLabel] ] = hprobs
                            derivMxToFill[ spam_label_rows[spamLabel] ] = dprobs
                            prMxToFill[ spam_label_rows[spamLabel] ] = probs
    
                    if lastLabel is not None and spam_label_rows.has_key(lastLabel):
                        mxToFill[ spam_label_rows[lastLabel] ] = -hs
                        derivMxToFill[ spam_label_rows[lastLabel] ] = -ds
                        prMxToFill[ spam_label_rows[lastLabel] ] = 1.0-s

                else: #compute hessian & probs (no derivs)

                    for spamLabel in spam_labels_to_loop: #Note: must loop through all spam labels, even if not requested, in case prMxToFill is not None
                        if self.SPAMs[spamLabel] is None:
                            assert(lastLabel is None) # ensure there is at most one dummy spam label
                            lastLabel = spamLabel; continue
                        hprobs, probs = self.bulk_hpr(spamLabel, evalTree,
                                                      gates,G0,SPAM,SP0,True,False,clipTo,check)
                        hs = hprobs if hs is None else hs + hprobs
                        s += probs
                        if spam_label_rows.has_key(spamLabel):
                            mxToFill[ spam_label_rows[spamLabel] ] = hprobs
                            prMxToFill[ spam_label_rows[spamLabel] ] = probs
    
                    if lastLabel is not None and spam_label_rows.has_key(lastLabel):
                        mxToFill[ spam_label_rows[lastLabel] ] = -hs
                        prMxToFill[ spam_label_rows[lastLabel] ] = 1.0-s


            else: 
                if derivMxToFill is not None: #compute hessians and derivatives (no probs)

                    for spamLabel in spam_labels_to_loop: #Note: must loop through all spam labels, even if not requested, in case prMxToFill is not None
                        if self.SPAMs[spamLabel] is None:
                            assert(lastLabel is None) # ensure there is at most one dummy spam label
                            lastLabel = spamLabel; continue
                        hprobs, dprobs = self.bulk_hpr(spamLabel, evalTree,
                                                       gates,G0,SPAM,SP0,False,True,clipTo,check)
                        hs = hprobs if hs is None else hs + hprobs
                        ds = dprobs if ds is None else ds + dprobs
                        if spam_label_rows.has_key(spamLabel):
                            mxToFill[ spam_label_rows[spamLabel] ] = hprobs
                            derivMxToFill[ spam_label_rows[spamLabel] ] = dprobs
    
                    if lastLabel is not None and spam_label_rows.has_key(lastLabel):
                        mxToFill[ spam_label_rows[lastLabel] ] = -hs
                        derivMxToFill[ spam_label_rows[lastLabel] ] = -ds

                else: #just compute derivatives of probabilities

                    for spamLabel in spam_labels_to_loop: #Note: must loop through all spam labels, even if not requested, in case prMxToFill is not None
                        if self.SPAMs[spamLabel] is None:
                            assert(lastLabel is None) # ensure there is at most one dummy spam label
                            lastLabel = spamLabel; continue
                        hprobs = self.bulk_hpr(spamLabel, evalTree,
                                               gates,G0,SPAM,SP0,False,False,clipTo,check)
                        hs = hprobs if hs is None else hs + hprobs
                        if spam_label_rows.has_key(spamLabel):
                            mxToFill[ spam_label_rows[spamLabel] ] = hprobs
    
                    if lastLabel is not None and spam_label_rows.has_key(lastLabel):
                        mxToFill[ spam_label_rows[lastLabel] ] = -hs
    


    def frobeniusdist(self, otherGateSet, transformMx=None, gateWeight=1.0, spamWeight=1.0, normalize=True):
        """
        Compute the weighted frobenius norm of the difference between this
        gateset and otherGateSet.  Differences in each corresponding gate
        matrix and spam vector element are squared, weighted (using gateWeight
        or spamWeight as applicable), then summed.  The value returned is the
        square root of this sum, or the square root of this sum divided by the
        number of summands if normalize == True.

        Parameters
        ----------
        otherGateSet : GateSet
            the other gate set to difference against.

        transformMx : numpy array, optional
            if not None, transform this gateset by  
            G => inv(transformMx) * G * transformMx, for each gate matrix G 
            (and similar for rho and E vectors) before taking the difference.
            This transformation is applied only for the difference and does
            not alter the values stored in this gateset.

        gateWeight : float, optional
           weighting factor for differences between gate elements.

        spamWeight : float, optional
           weighting factor for differences between elements of spam vectors.

        normalize : bool, optional
           if True (the default), the frobenius difference is defined by the
           sum of weighted squared-differences divided by the number of 
           differences.  If False, this final division is not performed.

        Returns
        -------
        float
        """
        d = 0; T = transformMx
        nSummands = 0.0
        if T is not None:
            Ti = _nla.inv(T)
            for gateLabel in self:
                d += gateWeight * _gt.frobeniusdist2( _np.dot(Ti,_np.dot(self[gateLabel],T)),otherGateSet[gateLabel] )
                nSummands += gateWeight * _np.size(self[gateLabel])

            for (i,rhoV) in enumerate(self.rhoVecs): 
                d += spamWeight * _gt.frobeniusdist2(_np.dot(Ti,rhoV),otherGateSet.rhoVecs[i])
                nSummands += spamWeight * _np.size(rhoV)

            for (i,Evec) in enumerate(self.EVecs):
                d += spamWeight * _gt.frobeniusdist2(_np.dot(_np.transpose(T),Evec),otherGateSet.EVecs[i])
                nSummands += spamWeight * _np.size(Evec)

            if self.identityVec is not None:
                d += spamWeight * _gt.frobeniusdist2(_np.dot(_np.transpose(T),self.identityVec),otherGateSet.identityVec)
                nSummands += spamWeight * _np.size(self.identityVec)


            #for (spamLabel,spamGate) in self.SPAMs.iteritems():
            #    if spamGate is not None:
            #        d += _bt.frobeniusdist( _np.dot(Ti,_np.dot(spamGate,T)),otherGateSet.SPAMs[spamLabel] )
        else:
            for gateLabel in self:
                d += gateWeight * _gt.frobeniusdist2(self[gateLabel],otherGateSet[gateLabel])
                nSummands += gateWeight * _np.size(self[gateLabel])

            for (i,rhoV) in enumerate(self.rhoVecs): 
                d += spamWeight * _gt.frobeniusdist2(rhoV,otherGateSet.rhoVecs[i])
                nSummands += spamWeight *  _np.size(rhoV)

            for (i,Evec) in enumerate(self.EVecs):
                d += spamWeight * _gt.frobeniusdist2(Evec,otherGateSet.EVecs[i])
                nSummands += spamWeight * _np.size(Evec)

            if self.identityVec is not None and otherGateSet.identityVec is not None:
                d += spamWeight * _gt.frobeniusdist2(self.identityVec,otherGateSet.identityVec)
                nSummands += spamWeight * _np.size(self.identityVec)

            #for (spamLabel,spamGate) in self.SPAMs.iteritems():
            #    if spamGate is not None:
            #        d += _bt.frobeniusdist(spamGate,otherGateSet.SPAMs[spamLabel] )

        if normalize and nSummands > 0: 
            return _np.sqrt( d / float(nSummands) )
        else:
            return _np.sqrt(d)


    def jtracedist(self, otherGateSet, transformMx=None):
        """
        Compute the Jamiolkowski trace distance between this
        gateset and otherGateSet, defined as the maximum
        of the trace distances between each corresponding gate,
        including spam gates.

        Parameters
        ----------
        otherGateSet : GateSet
            the other gate set to difference against.

        transformMx : numpy array, optional
            if not None, transform this gateset by  
            G => inv(transformMx) * G * transformMx, for each gate matrix G 
            (and similar for rho and E vectors) before taking the difference.
            This transformation is applied only for the difference and does
            not alter the values stored in this gateset.

        Returns
        -------
        float
        """
        T = transformMx
        if T is not None:
            Ti = _nla.inv(T)
            dists = [ _gt.jtracedist( _np.dot(Ti,_np.dot(self[gateLabel],T)), otherGateSet[gateLabel] ) for gateLabel in self ]
            for (spamLabel,spamGate) in self.SPAMs.iteritems():
                if spamGate is not None:
                    dists.append( _gt.jtracedist( _np.dot(Ti,_np.dot(spamGate,T)), otherGateSet.SPAMs[spamLabel] ) )
        else:
            dists = [ _gt.jtracedist(self[gateLabel], otherGateSet[gateLabel]) for gateLabel in self ]
            for (spamLabel,spamGate) in self.SPAMs.iteritems():
                if spamGate is not None:
                    dists.append( _gt.jtracedist(spamGate, otherGateSet.SPAMs[spamLabel] ) )
        return max(dists)


    def diamonddist(self, otherGateSet, transformMx=None):
        """
        Compute the diamond-norm distance between this
        gateset and otherGateSet, defined as the maximum
        of the diamond-norm distances between each
        corresponding gate, including spam gates.

        Parameters
        ----------
        otherGateSet : GateSet
            the other gate set to difference against.

        transformMx : numpy array, optional
            if not None, transform this gateset by  
            G => inv(transformMx) * G * transformMx, for each gate matrix G 
            (and similar for rho and E vectors) before taking the difference.
            This transformation is applied only for the difference and does
            not alter the values stored in this gateset.

        Returns
        -------
        float
        """
        T = transformMx
        if T is not None:
            Ti = _nla.inv(T)
            dists = [ _gt.diamonddist( _np.dot(Ti,_np.dot(self[gateLabel],T)), otherGateSet[gateLabel] ) for gateLabel in self ]
            for (spamLabel,spamGate) in self.SPAMs.iteritems():
                if spamGate is not None:
                    dists.append( _gt.diamonddist( _np.dot(Ti,_np.dot(spamGate,T)), otherGateSet.SPAMs[spamLabel] ) )
        else:
            dists = [ _gt.diamonddist(self[gateLabel], otherGateSet[gateLabel]) for gateLabel in self ]
            for (spamLabel,spamGate) in self.SPAMs.iteritems():
                if spamGate is not None:
                    dists.append( _gt.diamonddist(spamGate, otherGateSet.SPAMs[spamLabel] ) )
        return max(dists)



    def copy(self):
        """ 
        Copy this gateset

        Returns
        -------
        GateSet
            a (deep) copy of this gateset.
        """
        newGateset = GateSet()
        newGateset.rhoVecs = [ rhoVec.copy() for rhoVec in self.rhoVecs ]
        newGateset.EVecs = [ EVec.copy() for EVec in self.EVecs ]
        newGateset.SPAM_labels = self.SPAM_labels.copy()
        newGateset.history = self.history[:]
        newGateset.assumeSumToOne = self.assumeSumToOne
        newGateset.identityVec = self.identityVec

        for (l,spam) in self.SPAMs.iteritems():
            newGateset.SPAMs[l] = spam.copy() if spam is not None else None

        for (l,g) in self.gates.iteritems():
            newGateset.set_gate(l, g.copy())

        return newGateset

    def __str__(self):
        s = ""
        for (i,rhoVec) in enumerate(self.rhoVecs):
            s += "rhoVec[%d] = " % i + _mt.mx_to_string(_np.transpose(rhoVec)) + "\n"
        s += "\n"
        for (i,EVec) in enumerate(self.EVecs):
            s += "EVec[%d] = " % i + _mt.mx_to_string(_np.transpose(EVec)) + "\n"
        s += "\n"
        for (l,gatemx) in self.iteritems():
            s += l + " = \n" + _mt.mx_to_string(gatemx) + "\n\n"
        return s

        
    def iterall(self):  
        """
        Returns
        -------
        iterator
            an iterator over all gate matrices, including the "SPAM gates"
        """
        for (label,gatemx) in self.iteritems():
            yield (label, gatemx)
        for (label,spamgatemx) in self.SPAMs.iteritems(): 
            yield (label, spamgatemx)

    #def _toStdDict(self):
    #    """ 
    #    Deprecated!! 
    #    Return this GateSet's data as a normal dictionary for easy pickling
    #    """
    #    ret = { }
    #    for (i,rhoVec) in enumerate(self.rhoVecs):
    #        ret["rhoVec%d" % i] = rhoVec
    #    for (i,EVec) in enumerate(self.EVecs):
    #        ret["EVec%d" % i] = EVec
    #    for (label,gate) in self.gates.iteritems():
    #        ret[label] = gate
    #    return ret

    def get_nongauge_projector(self, gates=True,G0=True,SPAM=True,SP0=True):
        # We want to divide the GateSet-space H (a Hilbert space, 56-dimensional in the 1Q, 3-gate, 2-vec case)
        # into the direct sum of gauge and non-gauge spaces, and find projectors onto each
        # sub-space (the non-gauge space in particular).  
        #
        # Within the GateSet H-space lies a gauge-manifold of maximum chi2 (16-dimensional in 1Q case),
        #  where the gauge-invariant chi2 function is constant.  At a single point (GateSet) P on this manifold,
        #  chosen by (somewhat arbitrarily) fixing the gauge, we can talk about the tangent space
        #  at that point.  This tangent space is spanned by some basis (~16 elements for the 1Q case),
        #  which associate with the infinitesimal gauge transform ?generators? on the full space.
        #  The subspace of H spanned by the derivatives w.r.t gauge transformations at P (a GateSet) spans
        #  the gauge space, and the complement of this (in H) is the non-gauge space.
        #
        #  An element of the gauge group can be written gg = exp(-K), where K is a n x n matrix.  If K is
        #   hermitian then gg is unitary, but this need not be the case.  A gauge transform acts on a 
        #   gatset via Gateset => gg^-1 G gg, i.e. G => exp(K) G exp(-K).  We care about the action of
        #   infinitesimal gauge tranformations (b/c the *derivative* vectors span the tangent space), 
        #   which act as:
        #    G => (I+K) G (I-K) = G + [K,G] + ignored(K^2), where [K,G] := KG-GK
        #
        # To produce a projector onto the gauge-space, we compute the *column* vectors
        #  dG_ij = [K_ij,G], where K_ij is the i,j-th matrix unit (56x1 in the 1Q case, 16 such column vectors)
        #  and then form a projector in the standard way.
        #  (to project onto |v1>, |v2>, etc., form P = sum_i |v_i><v_i|)
        #
        #Typically nGateParams < len(dG_ij) and linear system is overconstrained
        #   and no solution is expected.  If no solution exists, simply ignore
        #
        # So we form P = sum_ij dG_ij * transpose(dG_ij) (56x56 in 1Q case)
        #              = dG * transpose(dG)              where dG_ij form the *columns* of dG (56x16 in 1Q case) 
        # But dG_ij are not orthonormal, so really we need a slight modification,
        #  otherwise P'P' != P' as must be the case for a projector:
        # 
        # P' = dG * (transpose(dG) * dG)^-1 * transpose(dG) (see wikipedia on projectors)
        #
        #    or equivalently (I think)
        #
        # P' = pseudo-inv(P)*P  
        #
        #  since the pseudo-inv is defined by P*pseudo-inv(P) = I and so P'P' == P'
        #  and P' is our gauge-projector!

        # Note: In the general case of parameterized gates (i.e. non-fully parameterized gates), there are fewer
        #   gate parameters than the size of dG_ij ( < 56 in 1Q case).  In this general case, we want to know
        #   what (if any) change in gate parameters produces the change dG_ij of the gate matrix elements.
        #   That is, we solve dG_ij = derivWRTParams * dParams_ij for dParams_ij, where derivWRTParams is 
        #   the derivative of the gate matrix elements with respect to the gate parameters (derivWRTParams 
        #   is 56x(nGateParams) and dParams_ij is (nGateParams)x1 in the 1Q case) -- then substitute dG_ij
        #   with dParams_ij above to arrive at a (nGateParams)x(nGateParams) projector (compatible with
        #   hessian computed by gateset).
        #
        #   Actually, we want to know if what changes gate parameters
        #   produce changes in the span of all the dG_ij, so really we want the intersection of the space
        #   defined by the columns of derivWRTParams (the "gate-parameter range" space) and the dG_ij vectors.
        #
        #   This intersection is determined by nullspace( derivWRTParams | dG ), where the pipe denotes
        #     concatenating the matrices together.  Then, if x is a basis vector of the nullspace
        #     x[0:nGateParams] is the basis vector for the intersection space within the gate parameter space,
        #     that is, the analogue of dParams_ij in the single-dG_ij introduction above.
        # 
        #   Still, we just substitue these dParams_ij vectors (as many as the nullspace dimension) for dG_ij
        #   above to get the general case projector.


        #Use a gateset object to hold & then vectorize the derivatives wrt each gauge transform basis element (each ij)
        #  **If G0 == False, then don't include gauge basis elements corresponding to the first row (since we assume
        #     gauge transforms will be TP constrained in this case)
        dim = self.gate_dim
        nParams = self.num_params(gates,G0,SPAM,SP0)

        #Note: gateset object (gsDeriv) must have all elements of gate mxs and spam vectors as parameters in order
        #  to match deriv_wrt_params calls, which give derivs wrt 
        gsDeriv = self.copy() 
        for gateLabel in self:
            gsDeriv.set_gate(gateLabel, _gate.FullyParameterizedGate(_np.zeros((dim,dim),'d')))
        for k,rhoVec in enumerate(gsDeriv.rhoVecs): gsDeriv.set_rhovec(_np.zeros((dim,1),'d'), k)
        for k,EVec in enumerate(gsDeriv.EVecs):     gsDeriv.set_evec(_np.zeros((dim,1),'d'), k)

        nElements = gsDeriv.num_elements()

        if gates == True: gatesToInclude = self.keys() #all gates
        elif gates == False: gatesToInclude = [] #no gates

        zMx = _np.zeros( (dim,dim), 'd')

        dG = _np.empty( (nElements, dim**2), 'd' )
        for i in xrange(dim): #Note: always range over all rows: this is *generator* mx, not gauge mx itself
            for j in xrange(dim):
                unitMx = _bt._mut(i,j,dim)
                #DEBUG: gsDeriv = self.copy() -- should delete this after debugging is done since doesn't work for parameterized gates
                if SPAM:
                    for k,rhoVec in enumerate(self.rhoVecs):
                        gsDeriv.set_rhovec( _np.dot(unitMx, rhoVec), k)
                    for k,EVec in enumerate(self.EVecs):
                        gsDeriv.set_evec( -_np.dot(EVec.T, unitMx).T, k)
                #else don't consider spam space as part of gateset-space => leave spam derivs zero
                    
                for gateLabel,gateMx in self.iteritems():
                    if gateLabel in gatesToInclude:
                        gsDeriv.get_gate(gateLabel).set_value( _np.dot(unitMx,gateMx)-_np.dot(gateMx,unitMx) )
                    #else leave gate as zeros

                #Note: vectorize *everything* in this gateset of FullyParameterizedGate
                #      objects to match the number of gateset *elements*.  (so all True's to to_vector)
                dG[:,i*dim+j] = gsDeriv.to_vector(True,True,True,True) 


        dP = self.deriv_wrt_params(gates,G0,SPAM,SP0)  #TODO maybe make this a hidden method if it's only used here...
        M = _np.concatenate( (dP,dG), axis=1 )
        
        def nullspace(m, tol=1e-7): #get the nullspace of a matrix
            u,s,vh = _np.linalg.svd(m)
            rank = (s > tol).sum()
            return vh[rank:].T.copy()

        nullsp = nullspace(M) #columns are nullspace basis vectors
        gen_dG = nullsp[0:nParams,:] #take upper (gate-param-segment) of vectors for basis
                                     # of subspace intersection in gate-parameter space
        #Note: gen_dG == "generalized dG", and is (nParams)x(nullSpaceDim==gaugeSpaceDim), so P 
        #  (below) is (nParams)x(nParams) as desired.

        #DEBUG
        #for iRow in range(nElements):
        #    pNorm = _np.linalg.norm(dP[iRow])
        #    if pNorm < 1e-6: 
        #        print "Row %d of dP is zero" % iRow
        #
        #print "------------------------------"
        #print "nParams = ",nParams
        #print "nElements = ",nElements
        #print " shape(M) = ",M.shape
        #print " shape(dG) = ",dG.shape
        #print " shape(dP) = ",dP.shape
        #print " shape(gen_dG) = ",gen_dG.shape
        #print " shape(nullsp) = ",nullsp.shape
        #print " rank(dG) = ",_np.linalg.matrix_rank(dG)
        #print " rank(dP) = ",_np.linalg.matrix_rank(dP)
        #print "------------------------------"
        #assert(_np.linalg.norm(_np.imag(gen_dG)) < 1e-9) #gen_dG is real


        # ORIG WAY: use psuedo-inverse to normalize projector.  Ran into problems where
        #  default rcond == 1e-15 didn't work for 2-qubit case, but still more stable than inv method below
        P = _np.dot(gen_dG, _np.transpose(gen_dG)) # almost a projector, but cols of dG are not orthonormal
        Pp = _np.dot( _np.linalg.pinv(P, rcond=1e-7), P ) # make P into a true projector (onto gauge space)

        # ALT WAY: use inverse of dG^T*dG to normalize projector (see wikipedia on projectors, dG => A)
        #  This *should* give the same thing as above, but numerical differences indicate the pinv method
        #  is prefereable (so long as rcond=1e-7 is ok in general...)
        #  Check: P'*P' = (dG (dGT dG)^1 dGT)(dG (dGT dG)^-1 dGT) = (dG (dGT dG)^1 dGT) = P'
        #invGG = _np.linalg.inv(_np.dot(_np.transpose(gen_dG), gen_dG))
        #Pp_alt = _np.dot(gen_dG, _np.dot(invGG, _np.transpose(gen_dG))) # a true projector (onto gauge space)
        #print "Pp - Pp_alt norm diff = ", _np.linalg.norm(Pp_alt - Pp)

        ret = _np.identity(nParams,'d') - Pp # projector onto the non-gauge space

        # Check ranks to make sure everything is consistent.  If either of these assertions fail,
        #  then pinv's rcond or some other numerical tolerances probably need adjustment.
        #print "Rank P = ",_np.linalg.matrix_rank(P)
        #print "Rank Pp = ",_np.linalg.matrix_rank(Pp, P_RANK_TOL)
        #print "Rank 1-Pp = ",_np.linalg.matrix_rank(_np.identity(nParams,'d') - Pp, P_RANK_TOL)
          #print " Evals(1-Pp) = \n","\n".join([ "%d: %g" % (i,ev) \
          #    for i,ev in enumerate(_np.sort(_np.linalg.eigvals(_np.identity(nParams,'d') - Pp))) ])

        rank_P = _np.linalg.matrix_rank(P, P_RANK_TOL) # original un-normalized projector onto gauge space
          # Note: use P_RANK_TOL here even though projector is *un-normalized* since sometimes P will
          #  have eigenvalues 1e-17 and one or two 1e-11 that should still be "zero" but aren't when
          #  no tolerance is given.  Perhaps a more custom tolerance based on the singular values of P
          #  but different from numpy's default tolerance would be appropriate here.

        assert( rank_P == _np.linalg.matrix_rank(Pp, P_RANK_TOL)) #rank shouldn't change with normalization
        assert( (nParams - rank_P) == _np.linalg.matrix_rank(ret, P_RANK_TOL) ) # dimension of orthogonal space
        return ret





    def get_nongauge_projector_ex(self, nonGaugeMixMx, gates=True,G0=True,SPAM=True,SP0=True):
        dim = self.gate_dim
        nParams = self.num_params(gates,G0,SPAM,SP0)

        #Note: gateset object (gsDeriv) must have all elements of gate mxs and spam vectors as parameters in order
        #  to match deriv_wrt_params calls, which give derivs wrt 
        gsDeriv = self.copy() 
        for gateLabel in self:
            gsDeriv.set_gate(gateLabel, _gate.FullyParameterizedGate(_np.zeros((dim,dim),'d')))
        for k,rhoVec in enumerate(gsDeriv.rhoVecs): gsDeriv.set_rhovec(_np.zeros((dim,1),'d'), k)
        for k,EVec in enumerate(gsDeriv.EVecs):     gsDeriv.set_evec(_np.zeros((dim,1),'d'), k)

        nElements = gsDeriv.num_elements()

        if gates == True: gatesToInclude = self.keys() #all gates
        elif gates == False: gatesToInclude = [] #no gates

        zMx = _np.zeros( (dim,dim), 'd')

        dG = _np.empty( (nElements, dim**2), 'd' )
        for i in xrange(dim): #Note: always range over all rows: this is *generator* mx, not gauge mx itself
            for j in xrange(dim):
                unitMx = _bt._mut(i,j,dim)
                gsDeriv = self.copy()
                if SPAM:
                    for k,rhoVec in enumerate(self.rhoVecs):
                        gsDeriv.set_rhovec( _np.dot(unitMx, rhoVec), k)
                    for k,EVec in enumerate(self.EVecs):
                        gsDeriv.set_evec( -_np.dot(EVec.T, unitMx).T, k)
                #else don't consider spam space as part of gateset-space => leave spam derivs zero
                    
                for gateLabel,gateMx in self.iteritems():
                    if gateLabel in gatesToInclude:
                        gsDeriv.get_gate(gateLabel).set_value( _np.dot(unitMx,gateMx)-_np.dot(gateMx,unitMx) )
                    #else leave gate as zeros

                #Note: vectorize *everything* in this gateset of FullyParameterizedGate
                #      objects to match the number of gateset *elements*.  (so all True's to to_vector)
                dG[:,i*dim+j] = gsDeriv.to_vector(True,True,True,True) 


        dP = self.deriv_wrt_params(gates,G0,SPAM,SP0)  #TODO maybe make this a hidden method if it's only used here...
        M = _np.concatenate( (dP,dG), axis=1 )
        
        def nullspace(m, tol=1e-7): #get the nullspace of a matrix
            u,s,vh = _np.linalg.svd(m)
            rank = (s > tol).sum()
            return vh[rank:].T.copy()

        nullsp = nullspace(M) #columns are nullspace basis vectors
        gen_dG = nullsp[0:nParams,:] #take upper (gate-param-segment) of vectors for basis
                                     # of subspace intersection in gate-parameter space
        #Note: gen_dG == "generalized dG", and is (nParams)x(nullSpaceDim==gaugeSpaceDim), so P 
        #  (below) is (nParams)x(nParams) as desired.


        #EXTRA ----------------------------------------

        # nullspace of gen_dG^T (mx with gauge direction vecs as rows) gives non-gauge directions
        nonGaugeDirections = nullspace(gen_dG.T) #columns are non-gauge directions

        #for each column of gen_dG, which is a gauge direction in gateset parameter space,
        # we add some amount of non-gauge direction, given by coefficients of the 
        # numNonGaugeParams non-gauge directions.
        gen_dG = gen_dG + _np.dot( nonGaugeDirections, nonGaugeMixMx) #add non-gauge direction components in
         # dims: (nParams,nGaugeParams) + (nParams,nNonGaugeParams) * (nNonGaugeParams,nGaugeParams)
         # nonGaugeMixMx is a (nNonGaugeParams,nGaugeParams) matrix whose i-th column specifies the 
         #  coefficents to multipy each of the non-gauge directions by before adding them to the i-th 
         #  direction to project out (i.e. what were the pure gauge directions).

        #DEBUG
        #print "gen_dG shape = ",gen_dG.shape
        #print "NGD shape = ",nonGaugeDirections.shape
        #print "NGD rank = ",_np.linalg.matrix_rank(nonGaugeDirections, P_RANK_TOL)


        #END EXTRA ----------------------------------------

        # ORIG WAY: use psuedo-inverse to normalize projector.  Ran into problems where
        #  default rcond == 1e-15 didn't work for 2-qubit case, but still more stable than inv method below
        P = _np.dot(gen_dG, _np.transpose(gen_dG)) # almost a projector, but cols of dG are not orthonormal
        Pp = _np.dot( _np.linalg.pinv(P, rcond=1e-7), P ) # make P into a true projector (onto gauge space)

        # ALT WAY: use inverse of dG^T*dG to normalize projector (see wikipedia on projectors, dG => A)
        #  This *should* give the same thing as above, but numerical differences indicate the pinv method
        #  is prefereable (so long as rcond=1e-7 is ok in general...)
        #  Check: P'*P' = (dG (dGT dG)^1 dGT)(dG (dGT dG)^-1 dGT) = (dG (dGT dG)^1 dGT) = P'
        #invGG = _np.linalg.inv(_np.dot(_np.transpose(gen_dG), gen_dG))
        #Pp_alt = _np.dot(gen_dG, _np.dot(invGG, _np.transpose(gen_dG))) # a true projector (onto gauge space)
        #print "Pp - Pp_alt norm diff = ", _np.linalg.norm(Pp_alt - Pp)

        ret = _np.identity(nParams,'d') - Pp # projector onto the non-gauge space

        # Check ranks to make sure everything is consistent.  If either of these assertions fail,
        #  then pinv's rcond or some other numerical tolerances probably need adjustment.
        #print "Rank P = ",_np.linalg.matrix_rank(P)
        #print "Rank Pp = ",_np.linalg.matrix_rank(Pp, P_RANK_TOL)
        #print "Rank 1-Pp = ",_np.linalg.matrix_rank(_np.identity(nParams,'d') - Pp, P_RANK_TOL)
          #print " Evals(1-Pp) = \n","\n".join([ "%d: %g" % (i,ev) \
          #    for i,ev in enumerate(_np.sort(_np.linalg.eigvals(_np.identity(nParams,'d') - Pp))) ])

        rank_P = _np.linalg.matrix_rank(P, P_RANK_TOL) # original un-normalized projector onto gauge space
          # Note: use P_RANK_TOL here even though projector is *un-normalized* since sometimes P will
          #  have eigenvalues 1e-17 and one or two 1e-11 that should still be "zero" but aren't when
          #  no tolerance is given.  Perhaps a more custom tolerance based on the singular values of P
          #  but different from numpy's default tolerance would be appropriate here.

        assert( rank_P == _np.linalg.matrix_rank(Pp, P_RANK_TOL)) #rank shouldn't change with normalization
        assert( (nParams - rank_P) == _np.linalg.matrix_rank(ret, P_RANK_TOL) ) # dimension of orthogonal space
        return ret

        
    def depolarize(self, gate_noise=None, spam_noise=None, max_gate_noise=None,
                   max_spam_noise=None, seed=None):
        """
        Apply depolarization uniformly or randomly to this gateset's gate
        and/or SPAM elements, and return the result, without modifying the
        original (this) gateset.  You must specify either gate_noise or 
        max_gate_noise (for the amount of gate depolarization), and  either
        spam_noise or max_spam_noise (for spam depolarization).
        
        Parameters
        ----------
        gate_noise : float, optional
          apply depolarizing noise of strength 1-noise to all
          gates in the gateset. (Multiplies each assumed-Pauli-basis gate 
          matrix by the diagonal matrix with (1.0-noise) along all
          the diagonal elements except the first (the identity).
    
        spam_noise : float, optional
          apply depolarizing noise of strength 1-noise to all
          SPAM vectors in the gateset. (Multiplies the non-identity
          part of each assumed-Pauli-basis state preparation vector
          and measurement vector by (1.0-noise).

        max_gate_noise : float, optional
          specified instead of 'gate_noise'; apply a random depolarization
          with maximum strength 1-max_noise to each gate in the gateset.
    
        max_spam_noise : float, optional
          specified instead of 'spam_noise'; apply a random depolarization
          with maximum strength 1-max_noise to SPAM vector in the gateset.
    
        seed : int, optional
          if not None, seed numpy's random number generator with this value
          before generating random depolarizations.
        
        Returns
        -------
        GateSet
            the depolarized GateSet
        """
        newGateset = self.copy() # start by just copying the current gateset
        gateDim = self.get_dimension()
        rndm = _np.random.RandomState(seed)

        if max_gate_noise is not None:
            if gate_noise is not None: 
                raise ValueError("Must specify at most one of 'gate_noise' and 'max_gate_noise' NOT both")
    
            #Apply random depolarization to each gate
            r = max_gate_noise * rndm.random_sample( len(self) )
            for (i,label) in enumerate(self):
                D = _np.diag( [1]+[1-r[i]]*(gateDim-1) )
                newGateset.set_gate(label, _gate.FullyParameterizedGate( _np.dot(D,self[label]) ))
                
        elif gate_noise is not None:
            #Apply the same depolarization to each gate
            D = _np.diag( [1]+[1-gate_noise]*(gateDim-1) )
            for (i,label) in enumerate(self):
                newGateset.set_gate(label, _gate.FullyParameterizedGate( _np.dot(D,self[label]) ))
    
        if max_spam_noise is not None:
            if spam_noise is not None: 
                raise ValueError("Must specify at most  one of 'noise' and 'max_noise' NOT both")
    
            #Apply random depolarization to each rho and E vector
            r = max_spam_noise * rndm.random_sample( len(self.rhoVecs) )
            for (i,rhoVec) in enumerate(self.rhoVecs):
                D = _np.diag( [1]+[1-r[i]]*(gateDim-1) )
                newGateset.set_rhovec( _np.dot(D,rhoVec), i)
    
            r = max_spam_noise * rndm.random_sample( len(self.EVecs) )
            for (i,EVec) in enumerate(self.EVecs):
                D = _np.diag( [1]+[1-r[i]]*(gateDim-1) )
                newGateset.set_evec( _np.dot(D,EVec), i)    
                
        elif spam_noise is not None:
            #Apply the same depolarization to each gate
            D = _np.diag( [1]+[1-spam_noise]*(gateDim-1) )
            for (i,rhoVec) in enumerate(self.rhoVecs):
                newGateset.set_rhovec( _np.dot(D,rhoVec), i)
            for (i,EVec) in enumerate(self.EVecs):
                newGateset.set_evec( _np.dot(D,EVec), i)

        return newGateset
    
    
    def rotate(self, rotate=None, max_rotate=None, seed=None):
        """
        Apply rotation uniformly or randomly to this gateset,
        and return the result, without modifying the original
        (this) gateset.  You must specify either 'rotate' or
        'max_rotate'. This method currently only works on
        single-qubit gatesets.
    
        Parameters
        ----------
        rotate : float or tuple of floats, optional
          if a single float, apply rotation of rotate radians along each of
          the pauli-product axes (X,Y,Z for 1-qubit) of all gates in the gateset.
          For a 1-qubit gateset, a 3-tuple of floats can be specifed to apply
          separate rotations along the X, Y, and Z axes.  For a 2-qubit gateset,
          a 15-tuple of floats can be specified to apply separate rotations along
          the IX, IY, IZ, XI, XX, XY, XZ, YI, YX, YY, YZ, ZI, ZX, ZY, ZZ axes.
    
        max_rotate : float, optional
          specified instead of 'rotate'; apply a random rotation with
          maximum max_rotate radians along each of the relevant axes
          of each each gate in the gateset.  That is, rotations of a 
          particular gate around different axes are different random amounts.
    
        seed : int, optional
          if  not None, seed numpy's random number generator with this value
          before generating random depolarizations.
        
        Returns
        -------
        GateSet
            the rotated GateSet
        """
        newGateset = self.copy() # start by just copying gateset
        dim = self.get_dimension()
    
        for (i,rhoVec) in enumerate(self.rhoVecs):
            newGateset.set_rhovec( rhoVec, i )  
        for (i,EVec) in enumerate(self.EVecs):
            newGateset.set_evec( EVec, i )
    
        if dim not in (4, 16):
            raise ValueError("Gateset dimension must be either 4 (single-qubit) or 16 (two-qubit)")

        rndm = _np.random.RandomState(seed)
    
        if max_rotate is not None:
            if rotate is not None: 
                raise ValueError("Must specify exactly one of 'rotate' and 'max_rotate' NOT both")
    
            #Apply random rotation to each gate
            if dim == 4:
                r = max_rotate * rndm.random_sample( len(self) * 3 )
                for (i,label) in enumerate(self):
                    rot = r[3*i:3*(i+1)]
                    newGateset.set_gate(label, _gate.FullyParameterizedGate( _np.dot( 
                                _bt.single_qubit_gate(rot[0]/2.0,rot[1]/2.0,rot[2]/2.0), self[label]) ))
            elif dim == 16:
                r = max_rotate * rndm.random_sample( len(self) * 15 )
                for (i,label) in enumerate(self):
                    rot = r[15*i:15*(i+1)]
                    newGateset.set_gate(label, _gate.FullyParameterizedGate( _np.dot( 
                                _bt.two_qubit_gate(rot[0]/2.0,rot[1]/2.0,rot[2]/2.0,
                                                   rot[3]/2.0,rot[4]/2.0,rot[5]/2.0,
                                                   rot[6]/2.0,rot[7]/2.0,rot[8]/2.0,
                                                   rot[9]/2.0,rot[10]/2.0,rot[11]/2.0,
                                                   rot[12]/2.0,rot[13]/2.0,rot[14]/2.0,
                                                   ), self[label]) ))
            #else: raise ValueError("Invalid gateset dimension") # checked above
                
        elif rotate is not None:

            if dim == 4:
                #Specify rotation by a single value (to mean this rotation along each axis) or a 3-tuple
                if type(rotate) in (float,int): rx,ry,rz = rotate,rotate,rotate
                elif type(rotate) in (tuple,list):
                    if len(rotate) != 3:
                        raise ValueError("Rotation, when specified as a tuple must be of length 3, not: %s" % str(rotate))
                    (rx,ry,rz) = rotate
                else: raise ValueError("Rotation must be specifed as a single number or as a lenght-3 list, not: %s" % str(rotate))
                    
                for (i,label) in enumerate(self):
                    newGateset.set_gate(label, _gate.FullyParameterizedGate( _np.dot( 
                                _bt.single_qubit_gate(rx/2.0,ry/2.0,rz/2.0), self[label]) ))

            elif dim == 16:
                #Specify rotation by a single value (to mean this rotation along each axis) or a 15-tuple
                if type(rotate) in (float,int): 
                    rix,riy,riz = rotate,rotate,rotate
                    rxi,rxx,rxy,rxz = rotate,rotate,rotate,rotate
                    ryi,ryx,ryy,ryz = rotate,rotate,rotate,rotate
                    rzi,rzx,rzy,rzz = rotate,rotate,rotate,rotate
                elif type(rotate) in (tuple,list):
                    if len(rotate) != 15:
                        raise ValueError("Rotation, when specified as a tuple must be of length 15, not: %s" % str(rotate))
                    (rix,riy,riz,rxi,rxx,rxy,rxz,ryi,ryx,ryy,ryz,rzi,rzx,rzy,rzz) = rotate
                else: raise ValueError("Rotation must be specifed as a single number or as a lenght-15 list, not: %s" % str(rotate))
                    
                for (i,label) in enumerate(self):
                    newGateset.set_gate(label, _gate.FullyParameterizedGate( _np.dot( 
                                _bt.two_qubit_gate(rix/2.0,riy/2.0,riz/2.0,
                                                 rxi/2.0,rxx/2.0,rxy/2.0,rxz/2.0,
                                                 ryi/2.0,ryx/2.0,ryy/2.0,ryz/2.0,
                                                 rzi/2.0,rzx/2.0,rzy/2.0,rzz/2.0,), self[label]) ))
            #else: raise ValueError("Invalid gateset dimension") # checked above
    
        else: raise ValueError("Must specify either 'rotate' or 'max_rotate' -- neither was non-None")
        return newGateset
    
    
    def randomize_with_unitary(self,scale,seed=None):
        """
        Apply a random unitary to each element of a gateset, and return the
        result, without modifying the original (this) gateset. This method
        currently only works on single- and two-qubit gatesets, and *assumes*
        that the gate matrices of this gateset are being interpreted in the
        Pauli-product basis.
    
        Parameters
        ----------
        scale : float
          maximum element magnitude in the generator of each random unitary transform.
    
        seed : int, optional
          if not None, seed numpy's random number generator with this value
          before generating random depolarizations.
        
        Returns
        -------
        GateSet
            the randomized GateSet
        """
        gs_pauli = self.copy() # assumes self is in the pauli-product basis!!    
        rndm = _np.random.RandomState(seed)
    
        gate_dim = gs_pauli.get_dimension()
        if gate_dim == 4: unitary_dim = 2
        elif gate_dim == 16: unitary_dim = 4
        else: raise ValueError("Gateset dimension must be either 4 (single-qubit) or 16 (two-qubit)")
    
        for gateLabel in gs_pauli.keys():
            randMat = scale * (rndm.randn(unitary_dim,unitary_dim) + 1j * rndm.randn(unitary_dim,unitary_dim))
            randMat = _np.dot(_np.transpose(_np.conjugate(randMat)),randMat) # make randMat Hermetian: (A_dag*A)^dag = (A_dag*A)
            randU   = _scipy.linalg.expm(-1j*randMat)
    
            if unitary_dim == 2:
                randUPP = _bt.unitary_to_pauligate_1q(randU)
            elif unitary_dim == 4:
                randUPP = _bt.unitary_to_pauligate_2q(randU)
            else: raise ValueError("Gateset dimension must be either 4 (single-qubit) or 16 (two-qubit)")
    
            gs_pauli.set_gate(gateLabel, _gate.FullyParameterizedGate(_np.dot(randUPP,gs_pauli[gateLabel])))
    
        return gs_pauli
    
    
    def increase_dimension(self, newDimension):
        """
        Enlarge the spam vectors and gate matrices of gateset to a specified
        dimension, and return the resulting inflated gateset.  Spam vectors 
        are zero-padded and gate matrices are padded with 1's on the diagonal
        and zeros on the off-diagonal (effectively padded by identity operation).
    
        Parameters
        ----------
        newDimension : int
          the dimension of the returned gateset.  That is,
          the returned gateset will have rho and E vectors that
          have shape (newDimension,1) and gate matrices with shape
          (newDimension,newDimension)
        
        Returns
        -------
        GateSet
            the increased-dimension GateSet
        """
    
        curDim = self.get_dimension()
        assert(newDimension > curDim)
        
        new_gateset = self.copy()
        new_gateset.gate_dim = newDimension;
    
        addedDim = newDimension-curDim
        vec_zeroPad = _np.zeros( (addedDim,1), 'd')
    
        #Increase dimension of rhoVecs and EVecs by zero-padding
        for i,rhoVec in enumerate(self.rhoVecs):
            assert( len(self.rhoVecs[i]) == curDim )
            new_gateset.rhoVecs[i] = _np.concatenate( (self.rhoVecs[i], vec_zeroPad) )
    
        for i,EVec in enumerate(self.EVecs):
            assert( len(self.EVecs[i]) == curDim )
            new_gateset.EVecs[i] = _np.concatenate( (self.EVecs[i], vec_zeroPad) )
    
        #Increase dimension of identityVec by zero-padding
        if self.identityVec is not None:
            new_gateset.identityVec = _np.concatenate( (self.identityVec, vec_zeroPad) )
    
        #Increase dimension of gates by assuming they act as identity on additional (unknown) space
        for gateLabel,gate in self.iteritems():
            assert( gate.shape == (curDim,curDim) )
            newGate = _np.zeros( (newDimension,newDimension) )
            newGate[ 0:curDim, 0:curDim ] = gate[:,:]
            for i in xrange(curDim,newDimension): newGate[i,i] = 1.0
            new_gateset.set_gate(gateLabel, _gate.FullyParameterizedGate(newGate))
    
        new_gateset.make_spams()
        return new_gateset
    
    
    def decrease_dimension(self, newDimension):
        """
        Shrink the spam vectors and gate matrices of gateset to a specified
        dimension, and return the resulting gate set.
    
        Parameters
        ----------
        newDimension : int
          the dimension of the returned gateset.  That is,
          the returned gateset will have rho and E vectors that
          have shape (newDimension,1) and gate matrices with shape
          (newDimension,newDimension)
        
        Returns
        -------
        GateSet
            the decreased-dimension GateSet
        """
    
        curDim = self.get_dimension()
        assert(newDimension < curDim)
        
        new_gateset = self.copy()
        new_gateset.gate_dim = newDimension
    
        #Decrease dimension of rhoVecs and EVecs by truncation
        for i,rhoVec in enumerate(self.rhoVecs):
            assert( len(self.rhoVecs[i]) == curDim )
            new_gateset.rhoVecs[i] = self.rhoVecs[i][0:newDimension,:]
    
        for i,EVec in enumerate(self.EVecs):
            assert( len(self.EVecs[i]) == curDim )
            new_gateset.EVecs[i] = self.EVecs[i][0:newDimension,:]
    
        #Decrease dimension of identityVec by trunction
        if self.identityVec is not None:
            new_gateset.identityVec = self.identityVec[0:newDimension,:]
    
        #Decrease dimension of gates by truncation
        for gateLabel,gate in self.iteritems():
            assert( gate.shape == (curDim,curDim) )
            newGate = _np.zeros( (newDimension,newDimension) )
            newGate[ :, : ] = gate[0:newDimension,0:newDimension]
            new_gateset.set_gate(gateLabel, _gate.FullyParameterizedGate(newGate))
    
        new_gateset.make_spams()
        return new_gateset
    
    def kick(self, absmag=1.0, bias=0, seed=None):
        """
        Kick gateset by adding to each gate a random matrix with values
        uniformly distributed in the interval [bias-absmag,bias+absmag],
        and return the resulting "kicked" gate set.
    
        Parameters
        ----------
        absmag : float, optional
            The maximum magnitude of the entries in the "kick" matrix
            relative to bias.
            
        bias : float, optional
            The bias of the entries in the "kick" matrix.

        seed : int, optional
          if not None, seed numpy's random number generator with this value
          before generating random depolarizations.
    
        Returns
        -------
        GateSet
            the kicked gate set.
        """
        kicked_gs = self.copy()
        rndm = _np.random.RandomState(seed)
        for gateLabel,gate in self.iteritems():
            delta = absmag * 2.0*(rndm.random_sample(gate.shape)-0.5) + bias
            kicked_gs.set_gate(gateLabel, _gate.FullyParameterizedGate(kicked_gs[gateLabel] + delta))
        #kicked_gs.make_spams() #if we modify rhoVecs or EVecs
        return kicked_gs
    
    
    def print_info(self):
        """
        Print to stdout relevant information about this gateset,
          including the Choi matrices and their eigenvalues.
        
        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        print self
        print "\n"
        print "Choi Matrices:"
        for (label,gate) in self.iteritems():
            print "Choi(%s) in pauli basis = \n" % label, 
            _mt.mx_to_string_complex(_jt.jamiolkowski_iso(gate))
            print "  --eigenvals = ", sorted( 
                [ev.real for ev in _np.linalg.eigvals(
                        _jt.jamiolkowski_iso(gate))] ),"\n"
        print "Sum of negative Choi eigenvalues = ", _jt.sum_of_negative_choi_evals(self)
    
        rhovec_penalty = sum( [ _lf.rhovec_penalty(rhoVec) for rhoVec in self.rhoVecs ] )
        evec_penalty   = sum( [ _lf.evec_penalty(EVec)     for EVec   in self.EVecs ] )
        print "rhoVec Penalty (>0 if invalid rhoVecs) = ", rhovec_penalty
        print "EVec Penalty (>0 if invalid EVecs) = ", evec_penalty



