"""
Defines the TPInstrument class
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************
import collections as _collections
import numpy as _np

from pygsti.modelmembers.instruments.tpinstrumentop import TPInstrumentOp as _TPInstrumentOp
from pygsti.modelmembers import modelmember as _mm
from pygsti.modelmembers import operations as _op

from pygsti.evotypes import Evotype as _Evotype
from pygsti.baseobjs import statespace as _statespace
from pygsti.tools import matrixtools as _mt
from pygsti.tools import slicetools as _slct
from pygsti.baseobjs.label import Label as _Label
from pygsti.baseobjs.statespace import StateSpace as _StateSpace


class TPInstrument(_mm.ModelMember, _collections.OrderedDict):
    """
    A trace-preservng quantum instrument.

    This is essentially a collection of operations whose sum is a
    trace-preserving map.  The instrument's elements may or may not have all of
    the properties associated by a mathematical quantum instrument.

    If `M1,M2,...Mn` are the elements of the instrument, then we parameterize
    
    1. MT = (M1+M2+...Mn) as a TPParmeterizedGate
    2. Di = Mi - MT for `i = 1..(n-1)` as FullyParameterizedGates

    So to recover `M1...Mn` we compute:
    Mi = Di + MT for i = `1...(n-1)`
    = -(n-2)*MT-sum(Di) = -(n-2)*MT-[(MT-Mi)-n*MT] for i == (n-1)

    Parameters
    ----------
    op_matrices : dict of numpy arrays
        A dict (or list of key,value pairs) of the operation matrices whose sum
        must be a trace-preserving (TP) map.

    evotype : Evotype or str, optional
        The evolution type.  The special value `"default"` is equivalent
        to specifying the value of `pygsti.evotypes.Evotype.default_evotype`.

    state_space : StateSpace, optional
        The state space for this Instrument.  If `None`, the space is inferred
        from the first effect vector.  If `len(effects) == 0` in this case,
        an error is raised.

    items : list or dict, optional
        Initial values.  This should only be used internally in de-serialization.
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

    def __init__(self, op_matrices, evotype="default", state_space=None, called_from_reduce=False, items=[]):

        self._readonly = False  # until init is done
        if len(items) > 0:
            assert(op_matrices is None), "`items` was given when op_matrices != None"

        evotype = _Evotype.cast(evotype)
        self.param_ops = []  # first element is TP sum (MT), following
        #elements are fully-param'd (Mi-Mt) for i=0...n-2

        #Note: when un-pickling using items arg, these members will
        # remain the above values, but *will* be set when state dict is copied
        # in (so unpickling works as desired)

        if op_matrices is not None:
            if isinstance(op_matrices, dict):
                matrix_list = [(k, v) for k, v in op_matrices.items()]  # gives definite ordering
            elif isinstance(op_matrices, list):
                matrix_list = op_matrices  # assume it's is already an ordered (key,value) list
            else:
                raise ValueError("Invalid `op_matrices` arg of type %s" % type(op_matrices))

            assert(len(matrix_list) > 0 or state_space is not None), \
                "Must specify `state_space` when there are no instrument members!"
            state_space = _statespace.default_space_for_dim(matrix_list[0][1].shape[0]) if (state_space is None) \
                else _statespace.StateSpace.cast(state_space)

            # Create gate objects that are used to parameterize this instrument
            MT_mx = sum([v for k, v in matrix_list])  # sum-of-instrument-members matrix
            MT = _op.FullTPOp(MT_mx, None, evotype, state_space)
            self.param_ops.append(MT)

            dim = MT.dim
            for k, v in matrix_list[:-1]:
                Di = _op.FullArbitraryOp(v - MT_mx, None, evotype, state_space)
                assert(Di.dim == dim)
                self.param_ops.append(Di)

            #Create a TPInstrumentOp for each operation matrix
            # Note: TPInstrumentOp sets it's own parent and gpindices
            items = [(k, _TPInstrumentOp(self.param_ops, i, None))
                     for i, (k, v) in enumerate(matrix_list)]

            #DEBUG
            #print("POST INIT PARAM GATES:")
            #for i,v in enumerate(self.param_ops):
            #    print(i,":\n",v)
            #
            #print("POST nINIT ITEMS:")
            #for k,v in items:
            #    print(k,":\n",v)
        else:
            assert(state_space is not None), "`state_space` cannot be `None` when there are no members!"

        _collections.OrderedDict.__init__(self, items)
        _mm.ModelMember.__init__(self, state_space, evotype)
        if not called_from_reduce:  # if called from reduce, gpindices are already initialized
            self.init_gpindices()  # initialize our gpindices based on sub-members
        self._readonly = True

    def submembers(self):
        """
        Get the ModelMember-derived objects contained in this one.

        Returns
        -------
        list
        """
        return list(self.values())  # TPInstrumentOp objects that have self.param_ops as submembers

    def to_memoized_dict(self, mmg_memo):
        """Create a serializable dict with references to other objects in the memo.

        Parameters
        ----------
        mmg_memo: dict
            Memo dict from a ModelMemberGraph, i.e. keys are object ids and values
            are ModelMemberGraphNodes (which contain the serialize_id). This is NOT
            the same as other memos in ModelMember (e.g. copy, allocate_gpindices, etc.).

        Returns
        -------
        mm_dict: dict
            A dict representation of this ModelMember ready for serialization
            This must have at least the following fields:
            module, class, submembers, params, state_space, evotype
            Additional fields may be added by derived classes.
        """
        mm_dict = super().to_memoized_dict(mmg_memo)

        mm_dict['member_labels'] = list(self.keys())  # labels of the submember effects

        return mm_dict

    @classmethod
    def _from_memoized_dict(cls, mm_dict, serial_memo):
        state_space = _StateSpace.from_nice_serialization(mm_dict['state_space'])

        # Alternate __init__ because we already have members built:
        lbl_member_pairs = [(lbl, serial_memo[subm_serial_id])
                            for lbl, subm_serial_id in zip(mm_dict['member_labels'], mm_dict['submembers'])]
        MT_member = next(filter(lambda pair: pair[1].index == len(lbl_member_pairs) - 1, lbl_member_pairs))
        if type(MT_member) is tuple:
            param_ops = []
            for lbl in lbl_member_pairs:
                param_ops += [lbl[1]]
        else: 
            param_ops = MT_member.submembers()  # the final (TP) member has all the param_ops as its submembers
         

        ret = TPInstrument.__new__(TPInstrument)
        ret.param_ops = param_ops
        ret._readonly = False
        _collections.OrderedDict.__init__(ret, lbl_member_pairs)
        _mm.ModelMember.__init__(ret, state_space, mm_dict['evotype'])
        ret._readonly = True
        return ret

    def _is_similar(self, other, rtol, atol):
        """ Returns True if `other` model member (which it guaranteed to be the same type as self) has
            the same local structure, i.e., not considering parameter values or submembers """
        # Note: same as for Instrument (in instrument.py)
        return list(self.keys()) == list(other.keys())

    def __setitem__(self, key, value):
        if self._readonly: raise ValueError("Cannot alter Instrument elements")
        else: return _collections.OrderedDict.__setitem__(self, key, value)

    def __reduce__(self):
        """ Needed for OrderedDict-derived classes (to set dict items) """
        #Don't pickle TPInstrumentGates b/c they'll each pickle the same
        # param_ops and I don't this will unpickle correctly.  Instead, just
        # strip the numpy array from each element and call __init__ again when
        # unpickling:
        op_matrices = [(lbl, _np.asarray(val)) for lbl, val in self.items()]
        return (TPInstrument, (op_matrices, self.evotype, self.state_space, []),
                {'init_gpindices': self._gpindices, '_submember_rpindices': self._submember_rpindices})

    def __setstate__(self, state):
        # Re-initialize sub-members (which aren't pickled by re-created via __init__) using the saved gpindices
        # Note: the parent Model will later re-link to itself, so ok that parent is None here
        self._submember_rpindices = state['_submember_rpindices']
        self.set_gpindices(state['init_gpindices'], parent=None)

    def __pygsti_reduce__(self):
        return self.__reduce__()

    def simplify_operations(self, prefix=""):
        """
        Creates a dictionary of simplified instrument operations.

        Returns a dictionary of operations that belong to the Instrument's parent
        `Model` - that is, whose `gpindices` are set to all or a subset of
        this instruments's gpindices.  These are used internally within
        computations involving the parent `Model`.

        Parameters
        ----------
        prefix : str
            A string, usually identitying this instrument, which may be used
            to prefix the simplified gate keys.

        Returns
        -------
        OrderedDict of Gates
        """
        # Create "simplified" elements, which infer their parent and
        # gpindices from the set of "param-gates" they're constructed with.
        if isinstance(prefix, _Label):  # Deal with case when prefix isn't just a string
            simplified = _collections.OrderedDict(
                [(_Label(prefix.name + "_" + k, prefix.sslbls), v)
                 for k, v in self.items()])
        else:
            if prefix: prefix += "_"
            simplified = _collections.OrderedDict(
                [(prefix + k, v)
                 for k, v in self.items()])
        return simplified

    @property
    def parameter_labels(self):  # same as in Instrument CONSOLIDATE?
        """
        An array of labels (usually strings) describing this model member's parameters.
        """
        plabels_per_local_index = _collections.defaultdict(list)
        for operation, factorgate_local_inds in zip(self.submembers(), self._submember_rpindices):
            for i, plbl in zip(_slct.to_array(factorgate_local_inds), operation.parameter_labels):
                plabels_per_local_index[i].append(plbl)

        vl = _np.empty(self.num_params, dtype=object)
        for i in range(self.num_params):
            vl[i] = ', '.join(plabels_per_local_index[i])
        return vl

    @property
    def num_elements(self):  # same as in Instrument CONSOLIDATE?
        """
        Return the number of total gate elements in this instrument.

        This is in general different from the number of *parameters*,
        which are the number of free variables used to generate all of
        the matrix *elements*.

        Returns
        -------
        int
        """
        return sum([g.size for g in self.values()])

    @property
    def num_params(self):  # same as in Instrument CONSOLIDATE?
        """
        Get the number of independent parameters which specify this Instrument.

        Returns
        -------
        int
            the number of independent parameters.
        """
        return len(self.gpindices_as_array())

    def to_vector(self):  # same as in Instrument but w/.submembers() CONSOLIDATE?
        """
        Extract a vector of the underlying gate parameters from this Instrument.

        Returns
        -------
        numpy array
            a 1D numpy array with length == num_params().
        """
        assert(self.gpindices is not None), "Must set an Instrument's .gpindices before calling to_vector"
        v = _np.empty(self.num_params, 'd')
        for operation, factor_local_inds in zip(self.submembers(), self._submember_rpindices):
            v[factor_local_inds] = operation.to_vector()
        return v

    def from_vector(self, v, close=False, dirty_value=True):
        """
        Initialize the Instrument using a vector of its parameters.

        Parameters
        ----------
        v : numpy array
            The 1D vector of gate parameters.  Length
            must == num_params().

        close : bool, optional
            Whether `v` is close to this Instrument's current
            set of parameters.  Under some circumstances, when this
            is true this call can be completed more quickly.

        dirty_value : bool, optional
            The value to set this object's "dirty flag" to before exiting this
            call.  This is passed as an argument so it can be updated *recursively*.
            Leave this set to `True` unless you know what you're doing.

        Returns
        -------
        None
        """
        assert(self.gpindices is not None), "Must set an Instrument's .gpindices before calling from_vector"
        for operation, factor_local_inds in zip(self.submembers(), self._submember_rpindices):
            operation.from_vector(v[factor_local_inds], close, dirty_value)
        for instGate in self.values():
            instGate._construct_matrix()
        self.dirty = dirty_value

    def transform_inplace(self, s):
        """
        Update each Instrument element matrix `O` with `inv(s) * O * s`.

        Parameters
        ----------
        s : GaugeGroupElement
            A gauge group element which specifies the "s" matrix
            (and it's inverse) used in the above similarity transform.

        Returns
        -------
        None
        """
        #Note: since each Mi is a linear function of MT and the Di, we can just
        # transform the MT and Di (self.param_ops) and re-init the elements.
        for gate in self.param_ops:
            gate.transform_inplace(s)

        for element in self.values():
            element._construct_matrix()  # construct from param gates
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
        # depolarize the MT and Di (self.param_ops) and re-init the elements.
        for gate in self.param_ops:
            gate.depolarize(amount)

        for element in self.values():
            element._construct_matrix()  # construct from param gates
        self.dirty = True

    def rotate(self, amount, mx_basis='gm'):
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

        mx_basis : {'std', 'gm', 'pp', 'qt'} or Basis object
            The source and destination basis, respectively.  Allowed
            values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
            and Qutrit (qt) (or a custom basis object).

        Returns
        -------
        None
        """
        #Note: since each Mi is a linear function of MT and the Di, we can just
        # rotate the MT and Di (self.param_ops) and re-init the elements.
        for gate in self.param_ops:
            gate.rotate(amount, mx_basis)

        for element in self.values():
            element._construct_matrix()  # construct from param gates
        self.dirty = True

    def __str__(self):
        s = "TPInstrument with elements:\n"
        for lbl, element in self.items():
            s += "%s:\n%s\n" % (lbl, _mt.mx_to_string(element.to_dense(), width=4, prec=2))
        return s
