"""Defines the Instrument class"""
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
import collections as _collections
import numpy as _np
import warnings as _warnings

from ..tools import matrixtools as _mt

#from . import labeldicts as _ld
from . import gatesetmember as _gm
from . import gate as _gate


def convert(instrument, typ, basis):
    if typ == "TP":
        if isinstance(instrument, TPInstrument):
            return instrument
        else:
            return TPInstrument(list(instrument.items()))
    elif typ in ("full","static"):
        gate_list = [(k,_gate.convert(g,typ,basis)) for k,g in instrument.items()]
        return Instrument(gate_list)
    else:
        raise ValueError("Cannot convert an instrument to type %s" % typ)


class Instrument(_gm.GateSetMember, _collections.OrderedDict):
    """ 
    Meant to correspond to a quantum instrument in theory, this class
    generalizes that notion slightly to include a collection of gates that may
    or may not have all of the properties associated by a mathematical quantum
    instrument.
    """
    def __init__(self, gate_matrices, items=[]):
        """
        Creates a new Instrument object.

        Parameters
        ----------
        gates : dict of Gate objects
            A dict (or list of key,value pairs) of the gates.
        """
        self._readonly = False #until init is done
        if len(items)>0:
            assert(gate_matrices is None), "`items` was given when gate_matrices != None"

        dim = None
        
        if gate_matrices is not None:
            if isinstance(gate_matrices,dict):
                matrix_list = [(k,v) for k,v in gate_matrices.items()] #gives definite ordering
            elif isinstance(gate_matrices,list):
                matrix_list = gate_matrices # assume it's is already an ordered (key,value) list
            else:
                raise ValueError("Invalid `gate_matrices` arg of type %s" % type(gate_matrices))

            items = []
            for k,v in matrix_list:
                gate = v if isinstance(v, _gate.Gate) else \
                       _gate.FullyParameterizedGate(v)
                if dim is None: dim = gate.dim
                assert(dim == gate.dim),"All gates must have the same dimension!"
                items.append( (k,gate) )

        _collections.OrderedDict.__init__(self, items)
        _gm.GateSetMember.__init__(self, dim)
        self._paramvec = self._build_paramvec()
        self._readonly = True


    #No good way to update Instrument on the fly yet...
    #def _update_paramvec(self, modified_obj=None):
    #    """Updates self._paramvec after a member of this GateSet is modified"""
    #    for obj in self.values():
    #        assert(obj.gpindices is self), "Cannot add/adjust parameter vector!"
    #
    #    #update parameters changed by modified_obj
    #    self._paramvec[modified_obj.gpindices] = modified_obj.to_vector()
    #
    #    #re-initialze any members that also depend on the updated parameters
    #    modified_indices = set(modified_obj.gpindices_as_array())
    #    for obj in self.values()
    #        if obj is modified_obj: continue
    #        if modified_indices.intersection(obj.gpindices_as_array()):
    #            obj.from_vector(self._paramvec[obj.gpindices])

        
    def _build_paramvec(self):
        """ Resizes self._paramvec and updates gpindices & parent members as needed,
            and will initialize new elements of _paramvec, but does NOT change
            existing elements of _paramvec (use _update_paramvec for this)"""
        v = _np.empty(0,'d'); off = 0

        # Step 2: add parameters that don't exist yet
        for obj in self.values():
            if obj.gpindices is None or obj.parent is not self:
                #Assume all parameters of obj are new independent parameters
                v = _np.insert(v, off, obj.to_vector())
                num_new_params = obj.allocate_gpindices( off, self )
                off += num_new_params
            else:
                inds = obj.gpindices_as_array()
                M = max(inds) if len(inds)>0 else -1; L = len(v)
                if M >= L:
                    #Some indices specified by obj are absent, and must be created.
                    w = obj.to_vector()
                    v = _np.concatenate((v, _np.empty(M+1-L,'d')),axis=0) # [v.resize(M+1) doesn't work]
                    for ii,i in enumerate(inds):
                        if i >= L: v[i] = w[ii]
                off = M+1
        return v

    def __setitem__(self, key, value):
        if self._readonly: raise ValueError("Cannot alter Instrument elements")
        else: return _collections.OrderedDict.__setitem__(self, key, value)
        
    def __reduce__(self):
        """ Needed for OrderedDict-derived classes (to set dict items) """
        #need to *not* pickle parent, as __reduce__ bypasses GateSetMember.__getstate__
        dict_to_pickle = self.__dict__.copy()
        dict_to_pickle['_parent'] = None

        #Python 2.7: remove elements of __dict__ that get initialized by OrderedDict impl
        if '_OrderedDict__root' in dict_to_pickle: del dict_to_pickle['_OrderedDict__root']
        if '_OrderedDict__map' in dict_to_pickle: del dict_to_pickle['_OrderedDict__map']
        
        return (Instrument, (None, list(self.items())), dict_to_pickle)

    def __pygsti_reduce__(self):
        return self.__reduce__()

        
    def compile_gates(self,prefix=""):
        """
        Returns a dictionary of gates that belong to the Instrument's parent
        `GateSet` - that is, whose `gpindices` are set to all or a subset of
        this instruments's gpindices.  These are used internally within
        computations involving the parent `GateSet`.

        Parameters
        ----------
        prefix : str
            A string, usually identitying this instrument, which may be used
            to prefix the compiled gate keys.

        Returns
        -------
        OrderedDict of Gates
        """
        #Create a "compiled" (GateSet-referencing) set of element gates
        if prefix: prefix += "_"
        compiled = _collections.OrderedDict()
        for k,g in self.items():
            comp = g.copy()
            comp.set_gpindices( _gm._compose_gpindices(self.gpindices,
                                                       g.gpindices), self.parent)
            compiled[prefix + k] = comp
        return compiled


    def num_elements(self):
        """
        Return the number of total gate elements in this instrument.
        This is in general different from the number of *parameters*,
        which are the number of free variables used to generate all of
        the matrix *elements*.

        Returns
        -------
        int
        """
        return sum([ g.size for g in self.values() ])


    def num_params(self):
        """
        Get the number of independent parameters which specify this Instrument.

        Returns
        -------
        int
           the number of independent parameters.
        """
        return len(self._paramvec)


    def to_vector(self):
        """
        Extract a vector of the underlying gate parameters from this Instrument.

        Returns
        -------
        numpy array
            a 1D numpy array with length == num_params().
        """
        return self._paramvec


    def from_vector(self, v):
        """
        Initialize the Instrument using a vector of its parameters.

        Parameters
        ----------
        v : numpy array
            The 1D vector of gate parameters.  Length
            must == num_params().

        Returns
        -------
        None
        """
        assert(len(v) == self.num_params())
        for gate in self.values():
            gate.from_vector( v[gate.gpindices] )
        self._paramvec = v

        
    def transform(self, S):
        """
        Update Instrument element matrix G with inv(S) * G * S.

        Parameters
        ----------
        S : GaugeGroupElement
            A gauge group element which specifies the "S" matrix 
            (and it's inverse) used in the above similarity transform.            
        """
        #Note: since each Mi is a linear function of MT and the Di, we can just
        # transform the MT and Di (self.param_gates) and re-init the elements.
        for gate in self.values():
            gate.transform(S)
            self._paramvec[gate.gpindices] = gate.to_vector()
        self.dirty = True


    def depolarize(self, amount):
        """
        Depolarize this Instrument by the given `amount`.

        Parameters
        ----------
        amount : float or tuple
            The amount to depolarize by.  If a tuple, it must have length
            equal to one less than the dimension of the gate. All but the
            first element of each spam vector (often corresponding to the 
            identity element) are multiplied by `amount` (if a float) or
            the corresponding `amount[i]` (if a tuple).

        Returns
        -------
        None
        """
        #Note: since each Mi is a linear function of MT and the Di, we can just
        # depolarize the MT and Di (self.param_gates) and re-init the elements.
        for gate in self.values():
            gate.depolarize(amount)
            self._paramvec[gate.gpindices] = gate.to_vector()
        self.dirty = True

        
    def rotate(self, amount, mxBasis='gm'):
        """
        Rotate this instrument by the given `amount`.

        Parameters
        ----------
        amount : tuple of floats, optional
            Specifies the rotation "coefficients" along each of the non-identity
            Pauli-product axes.  The gate's matrix `G` is composed with a
            rotation operation `R`  (so `G` -> `dot(R, G)` ) where `R` is the
            unitary superoperator corresponding to the unitary operator 
            `U = exp( sum_k( i * rotate[k] / 2.0 * Pauli_k ) )`.  Here `Pauli_k`
            ranges over all of the non-identity un-normalized Pauli operators.

        mxBasis : {'std', 'gm', 'pp', 'qt'} or Basis object
            The source and destination basis, respectively.  Allowed
            values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
            and Qutrit (qt) (or a custom basis object).

        Returns
        -------
        None
        """
        #Note: since each Mi is a linear function of MT and the Di, we can just
        # rotate the MT and Di (self.param_gates) and re-init the elements.
        for gate in self.values():
            gate.rotate(amount,mxBasis)
            self._paramvec[gate.gpindices] = gate.to_vector()
        self.dirty = True

        
    def copy(self, parent=None):
        """
        Copy this Instrument.

        Returns
        -------
        Instrument
            A copy of this Instrument
        """
        copied_items = [ (k,v.copy()) for k,v in self.items() ]
        return self._copy_gpindices( Instrument(copied_items), parent)

    def __str__(self):
        s = "Instrument with elements:\n"
        for lbl,element in self.items():
            s += "%s:\n%s\n" % (lbl, _mt.mx_to_string(element.base, width=4, prec=2))
        return s




class TPInstrument(_gm.GateSetMember, _collections.OrderedDict):
    """ 
    A trace-preservng quantum instrument which is a collection of gates whose
    sum is a trace-preserving map.  The instrument's elements may or may not
    have all of the properties associated by a mathematical quantum instrument.

    If M1,M2,...Mn are the elements of the instrument, then we parameterize
    1. MT = (M1+M2+...Mn) as a TPParmeterizedGate
    2. Di = Mi - MT for i = 1..(n-1) as FullyParameterizedGates

    So to recover M1...Mn we compute:
    Mi = Di + MT for i = 1...(n-1)
       = -(n-2)*MT-sum(Di) = -(n-2)*MT-[(MT-Mi)-n*MT] for i == (n-1)
    """
    #Scratch:
    #    Scratch
    # M1+M2+M3+M4  MT
    #   -M2-M3-M4  M1-MT
    #-M1   -M3-M4  M2-MT
    #-M1-M2   -M4  M3-MT
    #
    #(M1-MT) + (M2-MT) + (M3-MT) = (MT-M4) - 3*MT = -2*MT-M4
    # M4 = -(sum(Di)+(4-2=2)*MT) = -(sum(all)+(4-3=1)*MT)
    #n=2 case: (M1-MT) = (MT-M2)-MT = -M2, so M2 = -sum(Di)
    
    def __init__(self, gate_matrices, items=[]):
        """
        Creates a new Instrument object.

        Parameters
        ----------
        gates : dict of numpy arrays
            A dict (or list of key,value pairs) of the gate matrices whose sum
            must be a trace-preserving (TP) map.
        """
        self._readonly = False #until init is done
        if len(items)>0:
            assert(gate_matrices is None), "`items` was given when gate_matrices != None"

        dim = None
        self.param_gates = [] #first element is TP sum (MT), following
           #elements are fully-param'd (Mi-Mt) for i=0...n-2

        #Note: when un-pickling using items arg, these members will
        # remain the above values, but *will* be set when state dict is copied
        # in (so unpickling works as desired)
        
        if gate_matrices is not None:
            if isinstance(gate_matrices,dict):
                matrix_list = [(k,v) for k,v in gate_matrices.items()] #gives definite ordering
            elif isinstance(gate_matrices,list):
                matrix_list = gate_matrices # assume it's is already an ordered (key,value) list
            else:
                raise ValueError("Invalid `gate_matrices` arg of type %s" % type(gate_matrices))

            # Create gate objects that are used to parameterize this instrument
            MT = _gate.TPParameterizedGate( sum([v for k,v in matrix_list]) )
            MT.set_gpindices( slice(0, MT.num_params()), self)
            self.param_gates.append( MT )

            dim = MT.dim; off = MT.num_params()
            for k,v in matrix_list[:-1]:
                Di = _gate.FullyParameterizedGate(v-MT)
                Di.set_gpindices( slice(off, off+Di.num_params()), self )
                assert(Di.dim == dim)
                self.param_gates.append( Di ); off += Di.num_params()
            
            #Create a TPInstrumentGate for each gate matrix
            # Note: TPInstrumentGate sets it's own parent and gpindices
            items = [ (k, _gate.TPInstrumentGate(self.param_gates,i)) 
                      for i,(k,v) in enumerate(matrix_list) ]

            #DEBUG
            #print("POST INIT PARAM GATES:")
            #for i,v in enumerate(self.param_gates):
            #    print(i,":\n",v)
            #
            #print("POST INIT ITEMS:")
            #for k,v in items:
            #    print(k,":\n",v)


        _collections.OrderedDict.__init__(self,items)
        _gm.GateSetMember.__init__(self,dim)
        self._readonly = True

    def __setitem__(self, key, value):
        if self._readonly: raise ValueError("Cannot alter POVM elements")
        else: return _collections.OrderedDict.__setitem__(self, key, value)
        
        
    def __reduce__(self):
        """ Needed for OrderedDict-derived classes (to set dict items) """
        #Don't pickle TPInstrumentGates b/c they'll each pickle the same
        # param_gates and I don't this will unpickle correctly.  Instead, just
        # strip the numpy array from each element and call __init__ again when
        # unpickling:
        gate_matrices = [ (lbl,_np.asarray(val)) for lbl,val in self.items()]
        return (TPInstrument, (gate_matrices,[]), {'_gpindices': self._gpindices})

    def __pygsti_reduce__(self):
        return self.__reduce__()


    def compile_gates(self, prefix=""):
        """
        Returns a dictionary of gates that belong to the Instrument's parent
        `GateSet` - that is, whose `gpindices` are set to all or a subset of
        this instruments's gpindices.  These are used internally within
        computations involving the parent `GateSet`.

        Parameters
        ----------
        prefix : str
            A string, usually identitying this instrument, which may be used
            to prefix the compiled gate keys.

        Returns
        -------
        OrderedDict of Gates
        """
        #Create a "compiled" (GateSet-referencing) set of param gates
        param_compiled = []
        for g in self.param_gates:
            comp = g.copy()
            comp.set_gpindices( _gm._compose_gpindices(self.gpindices,
                                                       g.gpindices), self.parent)
            param_compiled.append(comp)

        # Create "compiled" elements, which infer their parent and
        # gpindices from the set of "param-gates" they're constructed with.
        if prefix: prefix += "_"
        compiled = _collections.OrderedDict(
            [ (prefix + k, _gate.TPInstrumentGate(param_compiled,i)) 
              for i,k in enumerate(self.keys()) ] )
        return compiled

    def num_elements(self):
        """
        Return the number of total gate elements in this instrument.
        This is in general different from the number of *parameters*,
        which are the number of free variables used to generate all of
        the matrix *elements*.

        Returns
        -------
        int
        """
        return sum([ g.size for g in self.values() ])


    def num_params(self):
        """
        Get the number of independent parameters which specify this Instrument.

        Returns
        -------
        int
           the number of independent parameters.
        """
        return sum([g.num_params() for g in self.param_gates])


    def to_vector(self):
        """
        Extract a vector of the underlying gate parameters from this Instrument.

        Returns
        -------
        numpy array
            a 1D numpy array with length == num_params().
        """
        v = _np.empty(self.num_params(),'d')
        for gate in self.param_gates:
            v[gate.gpindices] = gate.to_vector()
        return v


    def from_vector(self, v):
        """
        Initialize the Instrument using a vector of its parameters.

        Parameters
        ----------
        v : numpy array
            The 1D vector of gate parameters.  Length
            must == num_params().

        Returns
        -------
        None
        """
        for gate in self.param_gates:
            gate.from_vector( v[gate.gpindices] )
        for instGate in self.values():
            instGate._construct_matrix()
            

    def transform(self, S):
        """
        Update Instrument element matrix G with inv(S) * G * S.

        Parameters
        ----------
        S : GaugeGroupElement
            A gauge group element which specifies the "S" matrix 
            (and it's inverse) used in the above similarity transform.            
        """
        #Note: since each Mi is a linear function of MT and the Di, we can just
        # transform the MT and Di (self.param_gates) and re-init the elements.
        for gate in self.param_gates:
            gate.transform(S)

        for element in self.values():
            element._construct_matrix() # construct from param gates
        self.dirty = True


    def depolarize(self, amount):
        """
        Depolarize this Instrument by the given `amount`.

        Parameters
        ----------
        amount : float or tuple
            The amount to depolarize by.  If a tuple, it must have length
            equal to one less than the dimension of the gate. All but the
            first element of each spam vector (often corresponding to the 
            identity element) are multiplied by `amount` (if a float) or
            the corresponding `amount[i]` (if a tuple).

        Returns
        -------
        None
        """
        #Note: since each Mi is a linear function of MT and the Di, we can just
        # depolarize the MT and Di (self.param_gates) and re-init the elements.
        for gate in self.param_gates:
            gate.depolarize(amount)

        for element in self.values():
            element._construct_matrix() # construct from param gates
        self.dirty = True

        
    def rotate(self, amount, mxBasis='gm'):
        """
        Rotate this instrument by the given `amount`.

        Parameters
        ----------
        amount : tuple of floats, optional
            Specifies the rotation "coefficients" along each of the non-identity
            Pauli-product axes.  The gate's matrix `G` is composed with a
            rotation operation `R`  (so `G` -> `dot(R, G)` ) where `R` is the
            unitary superoperator corresponding to the unitary operator 
            `U = exp( sum_k( i * rotate[k] / 2.0 * Pauli_k ) )`.  Here `Pauli_k`
            ranges over all of the non-identity un-normalized Pauli operators.

        mxBasis : {'std', 'gm', 'pp', 'qt'} or Basis object
            The source and destination basis, respectively.  Allowed
            values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
            and Qutrit (qt) (or a custom basis object).

        Returns
        -------
        None
        """
        #Note: since each Mi is a linear function of MT and the Di, we can just
        # rotate the MT and Di (self.param_gates) and re-init the elements.
        for gate in self.param_gates:
            gate.rotate(amount,mxBasis)

        for element in self.values():
            element._construct_matrix() # construct from param gates
        self.dirty = True

        
    def copy(self, parent=None):
        """
        Copy this Instrument.

        Returns
        -------
        Instrument
            A copy of this Instrument
        """
        #Note: items will get copied in constructor, so we don't need to here.
        return self._copy_gpindices( TPInstrument( list(self.items()) ), parent)

    def __str__(self):
        s = "TPInstrument with elements:\n"
        for lbl,element in self.items():
            s += "%s:\n%s\n" % (lbl, _mt.mx_to_string(element.base, width=4, prec=2))
        return s
