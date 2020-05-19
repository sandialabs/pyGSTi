"""
Defines the POVM class
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
import itertools as _itertools
import numpy as _np
import warnings as _warnings
import functools as _functools

#from . import labeldicts as _ld
from . import modelmember as _gm
from . import spamvec as _sv
from . import operation as _op
from . import labeldicts as _ld
from .label import Label as _Label
from ..tools import matrixtools as _mt
from ..tools import basistools as _bt
from ..tools import optools as _gt


#Thoughts:
# what are POVM objs needed for?
# - construction of Effect vectors: allocating a pool of
#    shared parameters that multiple SPAMVecs use
#    - how should Model add items?
#      "default allocator" inserts new params into _paramvec when gpindices is None
#       (or is made None b/c parent is different) and sets gpindices accordingly
#      Could an alternate allocator allocate a POVM, which asks for/presensts a
#      block of indices, and after receiving this block adds effect vec to Model
#      which use the indices in this block? - maybe when Model inserts a POVM
#      it rebuilds paramvec as usual but doesn't insert it's effects into Model
#      (maybe not really inserting but "allocating/integrating" it - meaning it's
#       gpindices is set) until after the POVM's block of indices is allocated?
#    - maybe concept of "allocation" is a good one - meaning when an objects
#       gpindices and parent are set, and there's room in the Model's _paramvec
#       for the parameters.
#    - currently, a gates are "allocated" by _rebuild_paramvec when their
#       gpindices is None (if gpindices is not None, the indices can get
#       "shifted" but not "allocated" (check this!)
#    - maybe good to alert an object when it has be "allocated" to a Model;
#       a LinearOperator may do nothing, but a POVM might then allocate its member effects.
#       E.G:  POVM created = creates objects all with None gpindices
#             POVM assigned to a Model => Model allocates POVM & calls POVM.allocated_callback()
#             POVM.allocated_callback() allocates (on behalf of Model b/c POVM owns those indices?) its member effects -
#               maybe needs to add them to Model.effects so they're accounted for later & calls
#               SPAMVec.allocated_callback()
#             SPAMVec.allocated_callback() does nothing.
#    - it seems good for Model to keep track directly of allocated preps, gates, & effects OR else it will need to alert
#      objects when they're allocated indices shift so they can shift their member's
#      indices... (POVM.shifted_callback())
#    - at this point, could just add set_gpindices and shift_gpindices members to ModelMember, though not all indices
#      necessarily shift by same amt...
# - grouping a set of effect vectors together for iterating
#    over (just holding the names seems sufficient)

# Conclusions/philosphy: 12/8/2017
# - povms and instruments will hold their members, but member SPAMVec or LinearOperator objects
#   will have the Model as their parent, and have gpindices which reference the Model.
# - it is the parent object's (e.g. a Model, POVM, or Instrument) which is responsible
#   for setting the gpindices of its members.  The gpindices is set via a property or method
#   call, and parent objects will thereby set the gpindices of their contained elements.

#

def convert(povm, to_type, basis, extra=None):
    """
    Convert a POVM to a new type of parameterization.

    This potentially creates a new object.  Raises ValueError for invalid conversions.

    Parameters
    ----------
    povm : POVM
        POVM to convert

    to_type : {"full","TP","static","static unitary","H+S terms",
        "H+S clifford terms","clifford"}
        The type of parameterizaton to convert to.  See
        :method:`Model.set_all_parameterizations` for more details.
        TODO docstring: update the options here.

    basis : {'std', 'gm', 'pp', 'qt'} or Basis object
        The basis for `povm`.  Allowed values are Matrix-unit (std),
        Gell-Mann (gm), Pauli-product (pp), and Qutrit (qt)
        (or a custom basis object).

    extra : object, optional
        Unused.

    Returns
    -------
    POVM
        The converted POVM vector, usually a distinct
        object from the object passed as input.
    """
    if to_type in ("full", "static", "static unitary"):
        converted_effects = [(lbl, _sv.convert(vec, to_type, basis))
                             for lbl, vec in povm.items()]
        return UnconstrainedPOVM(converted_effects)

    elif to_type == "TP":
        if isinstance(povm, TPPOVM):
            return povm  # no conversion necessary
        else:
            converted_effects = [(lbl, _sv.convert(vec, "full", basis))
                                 for lbl, vec in povm.items()]
            return TPPOVM(converted_effects)

    elif _gt.is_valid_lindblad_paramtype(to_type):

        # A LindbladPOVM needs a *static* base/reference POVM
        #  with the appropriate evotype.  If we can convert `povm` to such a
        #  thing we win.  (the error generator is initialized as just the identity below)

        nQubits = int(round(_np.log2(povm.dim) / 2.0))  # Linblad ops always work on density-matrices, never states
        bQubits = bool(_np.isclose(nQubits, _np.log2(povm.dim) / 2.0))  # integer # of qubits?
        proj_basis = "pp" if (basis == "pp" or bQubits) else basis

        _, evotype = _gt.split_lindblad_paramtype(to_type)

        if isinstance(povm, ComputationalBasisPOVM):  # special easy case
            assert(povm.nqubits == nQubits)
            base_povm = ComputationalBasisPOVM(nQubits, evotype)
        else:
            base_items = [(lbl, _sv._convert_to_lindblad_base(Evec, "effect", evotype, basis))
                          for lbl, Evec in povm.items()]
            base_povm = UnconstrainedPOVM(base_items)

        # purevecs = extra if (extra is not None) else None # UNUSED
        cls = _op.LindbladDenseOp if (povm.dim <= 64 and evotype == "densitymx") \
            else _op.LindbladOp
        povmNoiseMap = cls.from_operation_obj(_np.identity(povm.dim, 'd'), to_type,
                                              None, proj_basis, basis, truncate=True)
        return LindbladPOVM(povmNoiseMap, base_povm, basis)

    elif to_type == "clifford":
        if isinstance(povm, ComputationalBasisPOVM) and povm._evotype == "stabilizer":
            return povm

        #OLD
        ##Try to figure out whether this POVM acts on states or density matrices
        #if any([ (isinstance(Evec,DenseSPAMVec) and _np.iscomplexobj(Evec.base)) # PURE STATE?
        #         for Evec in povm.values()]):
        #    nqubits = int(round(_np.log2(povm.dim)))
        #else:
        #    nqubits = int(round(_np.log2(povm.dim))) // 2

        #Assume `povm` already represents state-vec ops, since otherwise we'd
        # need to change dimension
        nqubits = int(round(_np.log2(povm.dim)))

        #Check if `povm` happens to be a Z-basis POVM on `nqubits`
        v = (_np.array([1, 0], 'd'), _np.array([0, 1], 'd'))  # (v0,v1) - eigenstates of sigma_z
        for zvals, lbl in zip(_itertools.product(*([(0, 1)] * nqubits)), povm.keys()):
            testvec = _functools.reduce(_np.kron, [v[i] for i in zvals])
            if not _np.allclose(testvec, povm[lbl].todense()):
                raise ValueError("Cannot convert POVM into a Z-basis stabilizer state POVM")

        #If no errors, then return a stabilizer POVM
        return ComputationalBasisPOVM(nqubits, 'stabilizer')

    else:
        raise ValueError("Invalid to_type argument: %s" % to_type)


class POVM(_gm.ModelMember, _collections.OrderedDict):
    """
    A generalized positive operator-valued measure (POVM).

    Meant to correspond to a  positive operator-valued measure,
    in theory, this class generalizes that notion slightly to
    include a collection of effect vectors that may or may not
    have all of the properties associated by a mathematical POVM.

    Parameters
    ----------
    dim : int
        The dimension of the Hilbert-Schmidt space of the effect vectors.

    evotype : str
        The evolution type.

    items : list or dict, optional
        Initial values.  This should only be used internally in de-serialization.
    """

    def __init__(self, dim, evotype, items=[]):
        self._readonly = False  # until init is done
        _collections.OrderedDict.__init__(self, items)
        _gm.ModelMember.__init__(self, dim, evotype)
        self._readonly = True
        assert(self.dim == dim)

    def __setitem__(self, key, value):
        if self._readonly: raise ValueError("Cannot alter POVM elements")
        else: return _collections.OrderedDict.__setitem__(self, key, value)

    def __pygsti_reduce__(self):
        return self.__reduce__()

    def num_params(self):
        """
        Get the number of independent parameters which specify this POVM.

        Returns
        -------
        int
            the number of independent parameters.
        """
        return 0  # default == no parameters

    def to_vector(self):
        """
        Extract a vector of the underlying gate parameters from this POVM.

        Returns
        -------
        numpy array
            a 1D numpy array with length == num_params().
        """
        return _np.array([], 'd')  # no parameters

    def from_vector(self, v, close=False, nodirty=False):
        """
        Initialize this POVM using a vector of its parameters.

        Parameters
        ----------
        v : numpy array
            The 1D vector of POVM parameters.  Length
            must == num_params().

        close : bool, optional
            Whether `v` is close to this POVM's current
            set of parameters.  Under some circumstances, when this
            is true this call can be completed more quickly.

        nodirty : bool, optional
            Whether this POVM should refrain from setting it's dirty
            flag as a result of this call.  `False` is the safe option, as
            this call potentially changes this POVM's parameters.

        Returns
        -------
        None
        """
        assert(len(v) == 0)  # should be no parameters

    def transform(self, s):  #INPLACE
        """
        Update each POVM effect E as s^T * E.

        Note that this is equivalent to the *transpose* of the effect vectors
        being mapped as `E^T -> E^T * s`.

        Parameters
        ----------
        s : GaugeGroupElement
            A gauge group element which specifies the "s" matrix
            (and it's inverse) used in the above similarity transform.

        Returns
        -------
        None
        """
        raise ValueError("Cannot transform a %s object" % self.__class__.__name__)
        #self.dirty = True

    def depolarize(self, amount):
        """
        Depolarize this POVM by the given `amount`.

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
        raise ValueError("Cannot depolarize a %s object" % self.__class__.__name__)
        #self.dirty = True

    def num_elements(self):
        """
        Return the number of total spam vector elements in this povm.

        This is in general different from the number of *parameters*,
        which are the number of free variables used to generate all of
        the vector *elements*.

        Returns
        -------
        int
        """
        return sum([E.size for E in self.values()])

    def acton(self, state):
        """
        Compute the outcome probabilities the result from acting on `state` with this POVM.

        Parameters
        ----------
        state : SPAMVec
            The state to act on

        Returns
        -------
        OrderedDict
            A dictionary whose keys are the outcome labels (strings)
            and whose values are the probabilities of seeing each outcome.
        """
        assert(self._evotype in ('densitymx', 'statevec', 'stabilizer')), \
            "probabilities(...) cannot be used with the %s evolution type!" % self._evotype
        assert(state._evotype == self._evotype), "Evolution type mismatch: %s != %s" % (self._evotype, state._evotype)

        staterep = state._rep
        outcome_probs = _collections.OrderedDict()
        for lbl, E in self.items():
            outcome_probs[lbl] = E._rep.probability(staterep)
        return outcome_probs

    def __str__(self):
        s = "%s with effect vectors:\n" % self.__class__.__name__
        for lbl, effect in self.items():
            s += "%s: %s\n" % (lbl, str(effect))
        return s


class _BasePOVM(POVM):
    """ The base behavior for both UnconstrainedPOVM and TPPOVM """

    def __init__(self, effects, preserve_sum=False):
        """
        Creates a new BasePOVM object.

        Parameters
        ----------
        effects : dict of SPAMVecs or array-like
            A dict (or list of key,value pairs) of the effect vectors.

        preserve_sum : bool, optional
            If true, the sum of `effects` is taken to be a constraint
            and so the final effect vector is made into a
            :class:`ComplementSPAMVec`.
        """
        dim = None
        self.Np = 0

        if isinstance(effects, dict):
            items = [(k, v) for k, v in effects.items()]  # gives definite ordering of effects
        elif isinstance(effects, list):
            items = effects  # assume effects is already an ordered (key,value) list
        else:
            raise ValueError("Invalid `effects` arg of type %s" % type(effects))

        if preserve_sum:
            assert(len(items) > 1), "Cannot create a TP-POVM with < 2 effects!"
            self.complement_label = items[-1][0]
            comp_val = _np.array(items[-1][1])  # current value of complement vec
        else:
            self.complement_label = None

        #Copy each effect vector and set it's parent and gpindices.
        # Assume each given effect vector's parameters are independent.
        copied_items = []
        evotype = None
        for k, v in items:
            if k == self.complement_label: continue
            effect = v if isinstance(v, _sv.SPAMVec) else \
                _sv.FullSPAMVec(v, typ="effect")
            if effect._prep_or_effect == "unknown": effect._prep_or_effect = "effect"  # backward compatibility
            assert(effect._prep_or_effect == "effect"), "Elements of POVMs must be *effect* SPAM vecs!"

            if evotype is None: evotype = effect._evotype
            else: assert(evotype == effect._evotype), \
                "All effect vectors must have the same evolution type"

            if dim is None: dim = effect.dim
            assert(dim == effect.dim), "All effect vectors must have the same dimension"

            N = effect.num_params()
            effect.set_gpindices(slice(self.Np, self.Np + N), self); self.Np += N
            copied_items.append((k, effect))
        items = copied_items

        if evotype is None:
            evotype = "densitymx"  # default (if no effects)

        #Add a complement effect if desired
        if self.complement_label is not None:  # len(items) > 0 by assert
            non_comp_effects = [v for k, v in items]
            identity_for_complement = _np.array(sum([v.reshape(comp_val.shape) for v in non_comp_effects])
                                                + comp_val, 'd')  # ensure shapes match before summing
            complement_effect = _sv.ComplementSPAMVec(
                identity_for_complement, non_comp_effects)
            complement_effect.set_gpindices(slice(0, self.Np), self)  # all parameters
            items.append((self.complement_label, complement_effect))

        super(_BasePOVM, self).__init__(dim, evotype, items)

    def _reset_member_gpindices(self):
        """
        Sets gpindices for all non-complement items.  Assumes all non-complement
        vectors have *independent* parameters (for now).
        """
        Np = 0
        for k, effect in self.items():
            if k == self.complement_label: continue
            N = effect.num_params()
            pslc = slice(Np, Np + N)
            if effect.gpindices != pslc:
                effect.set_gpindices(pslc, self)
            Np += N
        self.Np = Np

    def _rebuild_complement(self, identity_for_complement=None):
        """ Rebuild complement vector (in case other vectors have changed) """

        if self.complement_label is not None and self.complement_label in self:
            non_comp_effects = [v for k, v in self.items()
                                if k != self.complement_label]

            if identity_for_complement is None:
                identity_for_complement = self[self.complement_label].identity

            complement_effect = _sv.ComplementSPAMVec(
                identity_for_complement, non_comp_effects)
            complement_effect.set_gpindices(slice(0, self.Np), self)  # all parameters

            #Assign new complement effect without calling our __setitem__
            old_ro = self._readonly; self._readonly = False
            POVM.__setitem__(self, self.complement_label, complement_effect)
            self._readonly = old_ro

    def __setitem__(self, key, value):
        if not self._readonly:  # when readonly == False, we're initializing
            return super(_BasePOVM, self).__setitem__(key, value)

        if key == self.complement_label:
            raise KeyError("Cannot directly assign the complement effect vector!")
        value = value.copy() if isinstance(value, _sv.SPAMVec) else \
            _sv.FullSPAMVec(value, typ='effect')
        _collections.OrderedDict.__setitem__(self, key, value)
        self._reset_member_gpindices()
        self._rebuild_complement()

    def simplify_effects(self, prefix=""):
        """
        Creates a dictionary of simplified effect vectors.

        Returns a dictionary of effect SPAMVecs that belong to the POVM's parent
        `Model` - that is, whose `gpindices` are set to all or a subset of
        this POVM's gpindices.  Such effect vectors are used internally within
        computations involving the parent `Model`.

        Parameters
        ----------
        prefix : str
            A string, usually identitying this POVM, which may be used
            to prefix the simplified gate keys.

        Returns
        -------
        OrderedDict of SPAMVecs
        """
        if prefix: prefix = prefix + "_"
        simplified = _collections.OrderedDict()
        for lbl, effect in self.items():
            if lbl == self.complement_label: continue
            simplified[prefix + lbl] = effect.copy()
            simplified[prefix + lbl].set_gpindices(
                _gm._compose_gpindices(self.gpindices, effect.gpindices),
                self.parent)

        if self.complement_label:
            lbl = self.complement_label
            simplified[prefix + lbl] = _sv.ComplementSPAMVec(
                self[lbl].identity, [v for k, v in simplified.items()])
            self._copy_gpindices(simplified[prefix + lbl], self.parent)  # set gpindices
            # of complement vector to the same as POVM (it uses *all* params)
        return simplified

    def num_params(self):
        """
        Get the number of independent parameters which specify this POVM.

        Returns
        -------
        int
           the number of independent parameters.
        """
        return self.Np

    def to_vector(self):
        """
        Extract a vector of the underlying gate parameters from this POVM.

        Returns
        -------
        numpy array
            a 1D numpy array with length == num_params().
        """
        v = _np.empty(self.num_params(), 'd')
        for lbl, effect in self.items():
            if lbl == self.complement_label: continue
            v[effect.gpindices] = effect.to_vector()
        return v

    def from_vector(self, v, close=False, nodirty=False):
        """
        Initialize this POVM using a vector of its parameters.

        Parameters
        ----------
        v : numpy array
            The 1D vector of POVM parameters.  Length
            must == num_params().

        close : bool, optional
            Whether `v` is close to this POVM's current
            set of parameters.  Under some circumstances, when this
            is true this call can be completed more quickly.

        nodirty : bool, optional
            Whether this POVM should refrain from setting it's dirty
            flag as a result of this call.  `False` is the safe option, as
            this call potentially changes this POVM's parameters.

        Returns
        -------
        None
        """
        for lbl, effect in self.items():
            if lbl == self.complement_label: continue
            effect.from_vector(v[effect.gpindices], close, nodirty)
        if self.complement_label:  # re-init Ec
            self[self.complement_label]._construct_vector()

    def transform(self, s):
        """
        Update each POVM effect E as s^T * E.

        Note that this is equivalent to the *transpose* of the effect vectors
        being mapped as `E^T -> E^T * s`.

        Parameters
        ----------
        s : GaugeGroupElement
            A gauge group element which specifies the "s" matrix
            (and it's inverse) used in the above similarity transform.
        """
        for lbl, effect in self.items():
            if lbl == self.complement_label: continue
            effect.transform(s, 'effect')

        if self.complement_label:
            #Other effects being transformed transforms the complement,
            # so just check that the transform preserves the identity.
            TOL = 1e-6
            identityVec = self[self.complement_label].identity.todense().reshape((-1, 1))
            SmxT = _np.transpose(s.get_transform_matrix())
            assert(_np.linalg.norm(identityVec - _np.dot(SmxT, identityVec)) < TOL),\
                ("Cannot transform complement effect in a way that doesn't"
                 " preserve the identity!")
            self[self.complement_label]._construct_vector()

        self.dirty = True

    def depolarize(self, amount):
        """
        Depolarize this POVM by the given `amount`.

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
        for lbl, effect in self.items():
            if lbl == self.complement_label:
                #Don't depolarize complements since this will depol the
                # other effects via their shared params - cleanup will update
                # any complement vectors
                continue
            effect.depolarize(amount)

        if self.complement_label:
            # depolarization of other effects "depolarizes" the complement
            self[self.complement_label]._construct_vector()
        self.dirty = True


class UnconstrainedPOVM(_BasePOVM):
    """
    A POVM that just holds a set of effect vectors, parameterized individually however you want.

    Parameters
    ----------
    effects : dict of SPAMVecs or array-like
        A dict (or list of key,value pairs) of the effect vectors.
    """

    def __init__(self, effects):
        """
        Creates a new POVM object.

        Parameters
        ----------
        effects : dict of SPAMVecs or array-like
            A dict (or list of key,value pairs) of the effect vectors.
        """
        super(UnconstrainedPOVM, self).__init__(effects, preserve_sum=False)

    def __reduce__(self):
        """ Needed for OrderedDict-derived classes (to set dict items) """
        assert(self.complement_label is None)
        effects = [(lbl, effect.copy()) for lbl, effect in self.items()]
        return (UnconstrainedPOVM, (effects,), {'_gpindices': self._gpindices})


class TPPOVM(_BasePOVM):
    """
    A POVM whose sum-of-effects is constrained to what, by definition, we call the "identity".

    Parameters
    ----------
    effects : dict of SPAMVecs or array-like
        A dict (or list of key,value pairs) of the effect vectors.  The
        final effect vector will be stripped of any existing
        parameterization and turned into a ComplementSPAMVec which has
        no additional parameters and is always equal to
        `identity - sum(other_effects`, where `identity` is the sum of
        `effects` when this __init__ call is made.
    """

    def __init__(self, effects):
        """
        Creates a new POVM object.

        Parameters
        ----------
        effects : dict of SPAMVecs or array-like
            A dict (or list of key,value pairs) of the effect vectors.  The
            final effect vector will be stripped of any existing
            parameterization and turned into a ComplementSPAMVec which has
            no additional parameters and is always equal to
            `identity - sum(other_effects`, where `identity` is the sum of
            `effects` when this __init__ call is made.
        """
        super(TPPOVM, self).__init__(effects, preserve_sum=True)

    def __reduce__(self):
        """ Needed for OrderedDict-derived classes (to set dict items) """
        assert(self.complement_label is not None)
        effects = [(lbl, effect.copy()) for lbl, effect in self.items()
                   if lbl != self.complement_label]

        #add complement effect as a std numpy array - it will get
        # re-created correctly by __init__ w/preserve_sum == True
        effects.append((self.complement_label,
                        self[self.complement_label].todense().reshape((-1, 1))))

        return (TPPOVM, (effects,), {'_gpindices': self._gpindices})


class TensorProdPOVM(POVM):
    """
    A POVM that is effectively the tensor product of several other POVMs (which can be TP).

    Parameters
    ----------
    factor_povms : list of POVMs
        POVMs that will be tensor-producted together.
    """

    def __init__(self, factor_povms):
        """
        Creates a new TensorProdPOVM object.

        Parameters
        ----------
        factor_povms : list of POVMs
            POVMs that will be tensor-producted together.
        """
        dim = _np.product([povm.dim for povm in factor_povms])

        # self.factorPOVMs
        #  Copy each POVM and set it's parent and gpindices.
        #  Assume each one's parameters are independent.
        self.factorPOVMs = [povm.copy() for povm in factor_povms]

        off = 0; evotype = None
        for povm in self.factorPOVMs:
            N = povm.num_params()
            povm.set_gpindices(slice(off, off + N), self); off += N

            if evotype is None: evotype = povm._evotype
            else: assert(evotype == povm._evotype), \
                "All factor povms must have the same evolution type"

        if evotype is None:
            evotype = "densitymx"  # default (if there are no factors)

        items = []  # init as empty (lazy creation of members)
        self._factor_keys = tuple((list(povm.keys()) for povm in factor_povms))
        self._factor_lbllens = []
        for fkeys in self._factor_keys:
            assert(len(fkeys) > 0), "Each factor POVM must have at least one effect!"
            l = len(list(fkeys)[0])  # length of the first outcome label (a string)
            assert(all([len(elbl) == l for elbl in fkeys])), \
                "All the effect labels for a given factor POVM must be the *same* length!"
            self._factor_lbllens.append(l)

        super(TensorProdPOVM, self).__init__(dim, evotype, items)

    def __contains__(self, key):
        """ For lazy creation of effect vectors """
        i = 0
        for fkeys, lbllen in zip(self._factor_keys, self._factor_lbllens):
            if key[i:i + lbllen] not in fkeys: return False
            i += lbllen
        return True

    def __iter__(self):
        return self.keys()

    def __len__(self):
        return _np.product([len(fk) for fk in self._factor_keys])

    def keys(self):
        """
        An iterator over the effect (outcome) labels of this POVM.
        """
        for k in _itertools.product(*self._factor_keys):
            yield "".join(k)

    def values(self):
        """
        An iterator over the effect SPAM vectors of this POVM.
        """
        for k in self.keys():
            yield self[k]

    def items(self):
        """
        An iterator over the (effect_label, effect_vector) items in this POVM.
        """
        for k in self.keys():
            yield k, self[k]

    def __getitem__(self, key):
        """ For lazy creation of effect vectors """
        if _collections.OrderedDict.__contains__(self, key):
            return _collections.OrderedDict.__getitem__(self, key)
        elif key in self:  # calls __contains__ to efficiently check for membership
            #create effect vector now that it's been requested (lazy creation)
            elbls = []; i = 0  # decompose key into separate factor-effect labels
            for fkeys, lbllen in zip(self._factor_keys, self._factor_lbllens):
                elbls.append(key[i:i + lbllen]); i += lbllen
            # infers parent & gpindices from factor_povms
            effect = _sv.TensorProdSPAMVec('effect', self.factorPOVMs, elbls)
            _collections.OrderedDict.__setitem__(self, key, effect)
            return effect
        else: raise KeyError("%s is not an outcome label of this TensorProdPOVM" % key)

    def __reduce__(self):
        """ Needed for OrderedDict-derived classes (to set dict items) """
        return (TensorProdPOVM, ([povm.copy() for povm in self.factorPOVMs],),
                {'_gpindices': self._gpindices})  # preserve gpindices (but not parent)

    def simplify_effects(self, prefix=""):
        """
        Creates a dictionary of simplified effect vectors.

        Returns a dictionary of effect SPAMVecs that belong to the POVM's parent
        `Model` - that is, whose `gpindices` are set to all or a subset of
        this POVM's gpindices.  Such effect vectors are used internally within
        computations involving the parent `Model`.

        Parameters
        ----------
        prefix : str
            A string, usually identitying this POVM, which may be used
            to prefix the simplified gate keys.

        Returns
        -------
        OrderedDict of SPAMVecs
        """
        #Note: calling from_vector(...) on the simplified effect vectors (in
        # order) - e.g. within the finite differencing in MapForwardSimulator -  must
        # be able to properly initialize them, so need to set gpindices
        # appropriately.

        #Create a "simplified" (Model-referencing) set of factor POVMs
        factorPOVMs_simplified = []
        for p in self.factorPOVMs:
            povm = p.copy()
            povm.set_gpindices(_gm._compose_gpindices(self.gpindices,
                                                      p.gpindices), self.parent)
            factorPOVMs_simplified.append(povm)

        # Create "simplified" effect vectors, which infer their parent and
        # gpindices from the set of "factor-POVMs" they're constructed with.
        # Currently simplify *all* the effects, creating those that haven't been yet (lazy creation)
        if prefix: prefix += "_"
        simplified = _collections.OrderedDict(
            [(prefix + k, _sv.TensorProdSPAMVec('effect', factorPOVMs_simplified, self[k].effectLbls))
             for k in self.keys()])
        return simplified

    def num_params(self):
        """
        Get the number of independent parameters which specify this POVM.

        Returns
        -------
        int
            the number of independent parameters.
        """
        return sum([povm.num_params() for povm in self.factorPOVMs])

    def to_vector(self):
        """
        Extract a vector of the underlying gate parameters from this POVM.

        Returns
        -------
        numpy array
            a 1D numpy array with length == num_params().
        """
        v = _np.empty(self.num_params(), 'd')
        for povm in self.factorPOVMs:
            v[povm.gpindices] = povm.to_vector()
        return v

    def from_vector(self, v, close=False, nodirty=False):
        """
        Initialize this POVM using a vector of its parameters.

        Parameters
        ----------
        v : numpy array
            The 1D vector of POVM parameters.  Length
            must == num_params().

        close : bool, optional
            Whether `v` is close to this POVM's current
            set of parameters.  Under some circumstances, when this
            is true this call can be completed more quickly.

        nodirty : bool, optional
            Whether this POVM should refrain from setting it's dirty
            flag as a result of this call.  `False` is the safe option, as
            this call potentially changes this POVM's parameters.

        Returns
        -------
        None
        """
        for povm in self.factorPOVMs:
            povm.from_vector(v[povm.gpindices], close, nodirty)

    def depolarize(self, amount):
        """
        Depolarize this POVM by the given `amount`.

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
        for povm in self.factorPOVMs:
            povm.depolarize(amount)

        #No need to re-init effect vectors since they don't store a (dense)
        # version of their vector - they just create it from factor_povms on demand
        self.dirty = True

    def __str__(self):
        s = "Tensor-product POVM with %d factor POVMs\n" % len(self.factorPOVMs)
        #s += " and final effect labels " + ", ".join(self.keys()) + "\n"
        for i, povm in enumerate(self.factorPOVMs):
            s += "Factor %d: " % i
            s += str(povm)

        #s = "Tensor-product POVM with effect labels:\n"
        #s += ", ".join(self.keys()) + "\n"
        #s += " Effects (one per column):\n"
        #s += _mt.mx_to_string( _np.concatenate( [effect.todense() for effect in self.values()],
        #                                   axis=1), width=6, prec=2)
        return s


class ComputationalBasisPOVM(POVM):
    """
    A POVM that "measures" states in the computational "Z" basis.

    Parameters
    ----------
    nqubits : int
        The number of qubits

    evotype : {"densitymx", "statevec", "stabilizer", "svterm", "cterm"}
        The type of evolution being performed.

    qubit_filter : list, optional
        An optional list of integers specifying a subset
        of the qubits to be measured.
    """

    def __init__(self, nqubits, evotype, qubit_filter=None):
        """
        Creates a new StabilizerZPOVM object.

        Parameters
        ----------
        nqubits : int
            The number of qubits

        evotype : {"densitymx", "statevec", "stabilizer", "svterm", "cterm"}
            The type of evolution being performed.

        qubit_filter : list, optional
            An optional list of integers specifying a subset
            of the qubits to be measured.
        """
        if qubit_filter is not None:
            raise NotImplementedError("Still need to implement qubit_filter functionality")

        self.nqubits = nqubits
        self.qubit_filter = qubit_filter

        #LATER - do something with qubit_filter here
        # qubits = self.qubit_filter if (self.qubit_filter is not None) else list(range(self.nqubits))

        items = []  # init as empty (lazy creation of members)

        assert(evotype in ("statevec", "densitymx", "stabilizer", "svterm", "cterm"))
        dim = 4**nqubits if (evotype in ("densitymx", "svterm", "cterm")) else 2**nqubits
        super(ComputationalBasisPOVM, self).__init__(dim, evotype, items)

    def __contains__(self, key):
        """ For lazy creation of effect vectors """
        fkeys = ('0', '1')
        return bool(len(key) == self.nqubits
                    and all([(letter in fkeys) for letter in key]))

    def __iter__(self):
        return self.keys()

    def __len__(self):
        return 2**self.nqubits

    def keys(self):
        """
        An iterator over the effect (outcome) labels of this POVM.
        """
        iterover = [('0', '1')] * self.nqubits
        for k in _itertools.product(*iterover):
            yield "".join(k)

    def values(self):
        """
        An iterator over the effect SPAM vectors of this POVM.
        """
        for k in self.keys():
            yield self[k]

    def items(self):
        """
        An iterator over the (effect_label, effect_vector) items in this POVM.
        """
        for k in self.keys():
            yield k, self[k]

    def __getitem__(self, key):
        """ For lazy creation of effect vectors """
        if _collections.OrderedDict.__contains__(self, key):
            return _collections.OrderedDict.__getitem__(self, key)
        elif key in self:  # calls __contains__ to efficiently check for membership
            #create effect vector now that it's been requested (lazy creation)
            # decompose key into separate factor-effect labels
            outcomes = [(0 if letter == '0' else 1) for letter in key]
            effect = _sv.ComputationalSPAMVec(outcomes, self._evotype, "effect")  # "statevec" or "densitymx"
            effect.set_gpindices(slice(0, 0, None), self.parent)  # computational vecs have no params
            _collections.OrderedDict.__setitem__(self, key, effect)
            return effect
        else: raise KeyError("%s is not an outcome label of this StabilizerZPOVM" % key)

    def __reduce__(self):
        """ Needed for OrderedDict-derived classes (to set dict items) """
        return (ComputationalBasisPOVM, (self.nqubits, self._evotype, self.qubit_filter),
                {'_gpindices': self._gpindices})  # preserve gpindices (but not parent)

    def simplify_effects(self, prefix=""):
        """
        Creates a dictionary of simplified effect vectors.

        Returns a dictionary of effect SPAMVecs that belong to the POVM's parent
        `Model` - that is, whose `gpindices` are set to all or a subset of
        this POVM's gpindices.  Such effect vectors are used internally within
        computations involving the parent `Model`.

        Parameters
        ----------
        prefix : str
            A string, usually identitying this POVM, which may be used
            to prefix the simplified gate keys.

        Returns
        -------
        OrderedDict of SPAMVecs
        """
        # Create "simplified" effect vectors, which infer their parent and
        # gpindices from the set of "factor-POVMs" they're constructed with.
        if prefix: prefix += "_"
        simplified = _collections.OrderedDict(
            [(prefix + k, self[k]) for k in self.keys()])
        return simplified

    def __str__(self):
        s = "Computational(Z)-basis POVM on %d qubits and filter %s\n" \
            % (self.nqubits, str(self.qubit_filter))
        return s


class LindbladPOVM(POVM):
    """
    A POVM that is effectively a *single* Lindblad-parameterized gate followed by a computational-basis POVM.

    Parameters
    ----------
    errormap : MapOperator
        The error generator action and parameterization, encapsulated in
        a gate object.  Usually a :class:`LindbladOp`
        or :class:`ComposedOp` object.  (This argument is *not* copied,
        to allow LindbladSPAMVecs to share error generator
        parameters with other gates and spam vectors.)

    povm : POVM, optional
        A sub-POVM which supplies the set of "reference" effect vectors
        that `errormap` acts on to produce the final effect vectors of
        this LindbladPOVM.  This POVM must be *static*
        (have zero parameters) and its evolution type must match that of
        `errormap`.  If None, then a :class:`ComputationalBasisPOVM` is
        used on the number of qubits appropriate to `errormap`'s dimension.

    mx_basis : {'std', 'gm', 'pp', 'qt'} or Basis object
        The basis for this spam vector. Allowed values are Matrix-unit (std),
        Gell-Mann (gm), Pauli-product (pp), and Qutrit (qt) (or a custom
        basis object).  If None, then this is extracted (if possible) from
        `errormap`.
    """

    def __init__(self, errormap, povm=None, mx_basis=None):
        """
        Creates a new LindbladPOVM object.

        Parameters
        ----------
        errormap : MapOperator
            The error generator action and parameterization, encapsulated in
            a gate object.  Usually a :class:`LindbladOp`
            or :class:`ComposedOp` object.  (This argument is *not* copied,
            to allow LindbladSPAMVecs to share error generator
            parameters with other gates and spam vectors.)

        povm : POVM, optional
            A sub-POVM which supplies the set of "reference" effect vectors
            that `errormap` acts on to produce the final effect vectors of
            this LindbladPOVM.  This POVM must be *static*
            (have zero parameters) and its evolution type must match that of
            `errormap`.  If None, then a :class:`ComputationalBasisPOVM` is
            used on the number of qubits appropriate to `errormap`'s dimension.

        mx_basis : {'std', 'gm', 'pp', 'qt'} or Basis object
            The basis for this spam vector. Allowed values are Matrix-unit (std),
            Gell-Mann (gm), Pauli-product (pp), and Qutrit (qt) (or a custom
            basis object).  If None, then this is extracted (if possible) from
            `errormap`.
        """
        self.error_map = errormap
        dim = self.error_map.dim

        if mx_basis is None:
            if isinstance(errormap, _op.LindbladOp):
                mx_basis = errormap.errorgen.matrix_basis
            else:
                raise ValueError("Cannot extract a matrix-basis from `errormap` (type %s)"
                                 % str(type(errormap)))

        self.matrix_basis = mx_basis
        evotype = self.error_map._evotype

        if povm is None:
            nqubits = int(round(_np.log2(dim) / 2))
            assert(_np.isclose(nqubits, _np.log2(dim) / 2)), \
                ("A default computational-basis POVM can only be used with an"
                 " integral number of qubits!")
            povm = ComputationalBasisPOVM(nqubits, evotype)
        else:
            assert(povm._evotype == evotype), \
                ("Evolution type of `povm` (%s) must match that of "
                 "`errormap` (%s)!") % (povm._evotype, evotype)
            assert(povm.num_params() == 0), \
                "Given `povm` must be static (have 0 parameters)!"
        self.base_povm = povm

        items = []  # init as empty (lazy creation of members)
        super(LindbladPOVM, self).__init__(dim, evotype, items)

    def __contains__(self, key):
        """ For lazy creation of effect vectors """
        return bool(key in self.base_povm)

    def __iter__(self):
        return self.keys()

    def __len__(self):
        return len(self.base_povm)

    def keys(self):
        """
        An iterator over the effect (outcome) labels of this POVM.
        """
        for k in self.base_povm.keys():
            yield k

    def values(self):
        """
        An iterator over the effect SPAM vectors of this POVM.
        """
        for k in self.keys():
            yield self[k]

    def items(self):
        """
        An iterator over the (effect_label, effect_vector) items in this POVM.
        """
        for k in self.keys():
            yield k, self[k]

    def __getitem__(self, key):
        """ For lazy creation of effect vectors """
        if _collections.OrderedDict.__contains__(self, key):
            return _collections.OrderedDict.__getitem__(self, key)
        elif key in self:  # calls __contains__ to efficiently check for membership
            #create effect vector now that it's been requested (lazy creation)
            pureVec = self.base_povm[key]
            effect = _sv.LindbladSPAMVec(pureVec, self.error_map, "effect")
            effect.set_gpindices(self.error_map.gpindices, self.parent)
            # initialize gpindices of "child" effect (should be in simplify_effects?)
            _collections.OrderedDict.__setitem__(self, key, effect)
            return effect
        else: raise KeyError("%s is not an outcome label of this LindbladPOVM" % key)

    def __reduce__(self):
        """ Needed for OrderedDict-derived classes (to set dict items) """
        return (LindbladPOVM, (self.error_map.copy(), self.base_povm.copy(), self.matrix_basis),
                {'_gpindices': self._gpindices})  # preserve gpindices (but not parent)

    def allocate_gpindices(self, starting_index, parent, memo=None):
        """
        Sets gpindices array for this object or any objects it contains (i.e. depends upon).

        Indices may be obtained from contained objects which have already been
        initialized (e.g. if a contained object is shared with other top-level
        objects), or given new indices starting with `starting_index`.

        Parameters
        ----------
        starting_index : int
            The starting index for un-allocated parameters.

        parent : Model or ModelMember
            The parent whose parameter array gpindices references.

        memo : set, optional
            Used to prevent duplicate calls and self-referencing loops.  If
            `memo` contains an object's id (`id(self)`) then this routine
            will exit immediately.

        Returns
        -------
        num_new : int
            The number of *new* allocated parameters (so
            the parent should mark as allocated parameter
            indices `starting_index` to `starting_index + new_new`).
        """
        if memo is None: memo = set()
        if id(self) in memo: return 0
        memo.add(id(self))

        assert(self.base_povm.num_params() == 0)  # so no need to do anything w/base_povm
        num_new_params = self.error_map.allocate_gpindices(starting_index, parent, memo)  # *same* parent as self
        _gm.ModelMember.set_gpindices(
            self, self.error_map.gpindices, parent)
        return num_new_params

    def submembers(self):
        """
        Get the ModelMember-derived objects contained in this one.

        Returns
        -------
        list
        """
        return [self.error_map]

    def relink_parent(self, parent):  # Unnecessary?
        """
        Sets the parent of this object *without* altering its gpindices.

        In addition to setting the parent of this object, this method
        sets the parent of any objects this object contains (i.e.
        depends upon) - much like allocate_gpindices.  To ensure a valid
        parent is not overwritten, the existing parent *must be None*
        prior to this call.

        Parameters
        ----------
        parent : Model or ModelMember
            The parent of this POVM.

        Returns
        -------
        None
        """
        self.error_map.relink_parent(parent)
        _gm.ModelMember.relink_parent(self, parent)

    def set_gpindices(self, gpindices, parent, memo=None):
        """
        Set the parent and indices into the parent's parameter vector that are used by this ModelMember object.

        Parameters
        ----------
        gpindices : slice or integer ndarray
            The indices of this objects parameters in its parent's array.

        parent : Model or ModelMember
            The parent whose parameter array gpindices references.

        memo : dict, optional
            A memo dict used to avoid circular references.

        Returns
        -------
        None
        """
        if memo is None: memo = set()
        elif id(self) in memo: return
        memo.add(id(self))

        assert(self.base_povm.num_params() == 0)  # so no need to do anything w/base_povm
        self.error_map.set_gpindices(gpindices, parent, memo)
        self.terms = {}  # clear terms cache since param indices have changed now
        _gm.ModelMember._set_only_my_gpindices(self, gpindices, parent)

    def simplify_effects(self, prefix=""):
        """
        Creates a dictionary of simplified effect vectors.

        Returns a dictionary of effect SPAMVecs that belong to the POVM's parent
        `Model` - that is, whose `gpindices` are set to all or a subset of
        this POVM's gpindices.  Such effect vectors are used internally within
        computations involving the parent `Model`.

        Parameters
        ----------
        prefix : str
            A string, usually identitying this POVM, which may be used
            to prefix the simplified gate keys.

        Returns
        -------
        OrderedDict of SPAMVecs
        """
        # Create "simplified" effect vectors, which infer their parent and
        # gpindices from the set of "factor-POVMs" they're constructed with.
        if prefix: prefix += "_"
        simplified = _collections.OrderedDict(
            [(prefix + k, self[k]) for k in self.keys()])
        return simplified

    def num_params(self):
        """
        Get the number of independent parameters which specify this POVM.

        Returns
        -------
        int
            the number of independent parameters.
        """
        # Recall self.base_povm.num_params() == 0
        return self.error_map.num_params()

    def to_vector(self):
        """
        Extract a vector of the underlying gate parameters from this POVM.

        Returns
        -------
        numpy array
            a 1D numpy array with length == num_params().
        """
        # Recall self.base_povm.num_params() == 0
        return self.error_map.to_vector()

    def from_vector(self, v, close=False, nodirty=False):
        """
        Initialize this POVM using a vector of its parameters.

        Parameters
        ----------
        v : numpy array
            The 1D vector of POVM parameters.  Length
            must == num_params().

        close : bool, optional
            Whether `v` is close to this POVM's current
            set of parameters.  Under some circumstances, when this
            is true this call can be completed more quickly.

        nodirty : bool, optional
            Whether this POVM should refrain from setting it's dirty
            flag as a result of this call.  `False` is the safe option, as
            this call potentially changes this POVM's parameters.

        Returns
        -------
        None
        """
        # Recall self.base_povm.num_params() == 0
        self.error_map.from_vector(v, close, nodirty)

    def transform(self, s):  #INPLACE
        """
        Update each POVM effect E as s^T * E.

        Note that this is equivalent to the *transpose* of the effect vectors
        being mapped as `E^T -> E^T * s`.

        Parameters
        ----------
        s : GaugeGroupElement
            A gauge group element which specifies the "s" matrix
            (and it's inverse) used in the above similarity transform.

        Returns
        -------
        None
        """
        self.error_map.spam_transform(s, 'effect')  # only do this *once*
        for lbl, effect in self.items():
            effect._update_rep()  # these two lines mimic the bookeepging in
            effect.dirty = True   # a "effect.transform(s, 'effect')" call.
        self.dirty = True

    def __str__(self):
        s = "Lindblad-parameterized POVM of length %d\n" \
            % (len(self))
        return s


class MarginalizedPOVM(POVM):
    """
    A POVM whose effects are the sums of sets of effect vectors in a parent POVM.

    Namely the effects of the parent POVN whose labels have the same *character*
    at certain "marginalized" indices are summed together.

    Parameters
    ----------
    povm_to_marginalize : POVM
        The POVM to marginalize (the "parent" POVM).

    all_sslbls : StateSpaceLabels or tuple
        The state space labels of the parent POVM, which should have as many
        labels (factors) as the parent POVM's outcome/effect labels have characters.

    sslbls_after_marginalizing : tuple
        The subset of `all_sslbls` that should be *kept* after marginalizing.
    """

    def __init__(self, povm_to_marginalize, all_sslbls, sslbls_after_marginalizing):
        """
        Create a MarginalizedPOVM.

        Create a marginalized POVM by adding together sets of effect vectors whose labels
        have the same *character* at marginalized indices.  This assumes that the POVM
        being marginalized has a particular (though common) effect-label structure whereby
        each state-space sector corresponds to a single character, e.g. "0010" for a 4-qubt POVM.

        Parameters
        ----------
        povm_to_marginalize : POVM
            The POVM to marginalize (the "parent" POVM).

        all_sslbls : StateSpaceLabels or tuple
            The state space labels of the parent POVM, which should have as many
            labels (factors) as the parent POVM's outcome/effect labels have characters.

        sslbls_after_marginalizing : tuple
            The subset of `all_sslbls` that should be *kept* after marginalizing.
        """
        self.povm_to_marginalize = povm_to_marginalize

        if isinstance(all_sslbls, _ld.StateSpaceLabels):
            assert(len(all_sslbls.labels) == 1), "all_sslbls should only have a single tensor product block!"
            all_sslbls = all_sslbls.labels[0]

        #now all_sslbls is a tuple of labels, like sslbls_after_marginalizing
        self.sslbls_to_marginalize = all_sslbls
        self.sslbls_after_marginalizing = sslbls_after_marginalizing
        indices_to_keep = set([list(all_sslbls).index(l) for l in sslbls_after_marginalizing])
        indices_to_remove = set(range(len(all_sslbls))) - indices_to_keep
        self.indices_to_marginalize = sorted(indices_to_remove, reverse=True)

        elements_to_sum = {}
        for k in self.povm_to_marginalize.keys():
            mk = self.marginalize_effect_label(k)
            if mk in elements_to_sum:
                elements_to_sum[mk].append(k)
            else:
                elements_to_sum[mk] = [k]
        self._elements_to_sum = {k: tuple(v) for k, v in elements_to_sum.items()}  # convert to tuples
        super(MarginalizedPOVM, self).__init__(self.povm_to_marginalize.dim, self.povm_to_marginalize._evotype)

    def marginalize_effect_label(self, elbl):
        """
        Removes the "marginalized" characters from `elbl`, resulting in a marginalized POVM effect label.

        Parameters
        ----------
        elbl : str
            Effect label (typically of the parent POVM) to marginalize.
        """
        assert(len(elbl) == len(self.sslbls_to_marginalize))
        for i in self.indices_to_marginalize:
            elbl = elbl[:i] + elbl[i + 1:]  # remove i-th character
        return elbl

    def __contains__(self, key):
        """ For lazy creation of effect vectors """
        return bool(key in self._elements_to_sum)

    def __iter__(self):
        return self.keys()

    def __len__(self):
        return len(self._elements_to_sum)

    def keys(self):
        """
        An iterator over the effect (outcome) labels of this POVM.
        """
        for k in self._elements_to_sum.keys():
            yield k

    def values(self):
        """
        An iterator over the effect SPAM vectors of this POVM.
        """
        for k in self.keys():
            yield self[k]

    def items(self):
        """
        An iterator over the (effect_label, effect_vector) items in this POVM.
        """
        for k in self.keys():
            yield k, self[k]

    def __getitem__(self, key):
        """ For lazy creation of effect vectors """
        if _collections.OrderedDict.__contains__(self, key):
            return _collections.OrderedDict.__getitem__(self, key)
        elif key in self:  # calls __contains__ to efficiently check for membership
            #create effect vector now that it's been requested (lazy creation)
            #FUTURE: maybe have a "SumSPAMVec" that can add spamvecs to preserve paramterization and avoid dense reps
            effect_vec = None  # Note: currently all marginalized POVMs are *static*, since
            # we don't have a good general way to add parameterized effect vectors.

            for k in self._elements_to_sum[key]:
                e = self.povm_to_marginalize[k]
                if effect_vec is None:
                    effect_vec = e.todense()
                else:
                    effect_vec += e.todense()
            effect = _sv.StaticSPAMVec(effect_vec, self._evotype, 'effect')
            effect.set_gpindices(slice(0, 0), self.parent)
            _collections.OrderedDict.__setitem__(self, key, effect)
            return effect
        else: raise KeyError("%s is not an outcome label of this MarginalizedPOVM" % key)

    def __reduce__(self):
        """ Needed for OrderedDict-derived classes (to set dict items) """
        return (MarginalizedPOVM, (self.povm_to_marginalize, self.sslbls_to_marginalize,
                                   self.sslbls_after_marginalizing),
                {'_gpindices': self._gpindices})  # preserve gpindices (but not parent)

    #May need to implement this in future if we allow non-static MarginalizedPOVMs
    #def allocate_gpindices(self, starting_index, parent, memo=None):
    #    """
    #    Sets gpindices array for this object or any objects it
    #    contains (i.e. depends upon).  Indices may be obtained
    #    from contained objects which have already been initialized
    #    (e.g. if a contained object is shared with other
    #     top-level objects), or given new indices starting with
    #    `starting_index`.
    #
    #    Parameters
    #    ----------
    #    starting_index : int
    #        The starting index for un-allocated parameters.
    #
    #    parent : Model or ModelMember
    #        The parent whose parameter array gpindices references.
    #
    #    memo : set, optional
    #        Used to prevent duplicate calls and self-referencing loops.  If
    #        `memo` contains an object's id (`id(self)`) then this routine
    #        will exit immediately.
    #
    #    Returns
    #    -------
    #    num_new: int
    #        The number of *new* allocated parameters (so
    #        the parent should mark as allocated parameter
    #        indices `starting_index` to `starting_index + new_new`).
    #    """
    #    if memo is None: memo = set()
    #    if id(self) in memo: return 0
    #    memo.add(id(self))
    #
    #    assert(self.base_povm.num_params() == 0)  # so no need to do anything w/base_povm
    #    num_new_params = self.error_map.allocate_gpindices(starting_index, parent, memo)  # *same* parent as self
    #    _gm.ModelMember.set_gpindices(
    #        self, self.error_map.gpindices, parent)
    #    return num_new_params

    #def submembers(self):
    #    """
    #    Get the ModelMember-derived objects contained in this one.
    #
    #    Returns
    #    -------
    #    list
    #    """
    #    return [self.povm_to_marginalize]
    #
    #def relink_parent(self, parent):  # Unnecessary?
    #    """
    #    Sets the parent of this object *without* altering its gpindices.
    #
    #    In addition to setting the parent of this object, this method
    #    sets the parent of any objects this object contains (i.e.
    #    depends upon) - much like allocate_gpindices.  To ensure a valid
    #    parent is not overwritten, the existing parent *must be None*
    #    prior to this call.
    #    """
    #    self.povm_to_marginalize.relink_parent(parent)
    #    _gm.ModelMember.relink_parent(self, parent)

    #def set_gpindices(self, gpindices, parent, memo=None):
    #    """
    #    Set the parent and indices into the parent's parameter vector that
    #    are used by this ModelMember object.
    #
    #    Parameters
    #    ----------
    #    gpindices : slice or integer ndarray
    #        The indices of this objects parameters in its parent's array.
    #
    #    parent : Model or ModelMember
    #        The parent whose parameter array gpindices references.
    #
    #    Returns
    #    -------
    #    None
    #    """
    #    if memo is None: memo = set()
    #    elif id(self) in memo: return
    #    memo.add(id(self))
    #
    #    assert(self.base_povm.num_params() == 0)  # so no need to do anything w/base_povm
    #    self.error_map.set_gpindices(gpindices, parent, memo)
    #    self.terms = {}  # clear terms cache since param indices have changed now
    #    _gm.ModelMember._set_only_my_gpindices(self, gpindices, parent)

    def simplify_effects(self, prefix=""):
        """
        Creates a dictionary of simplified effect vectors.

        Returns a dictionary of effect SPAMVecs that belong to the POVM's parent
        `Model` - that is, whose `gpindices` are set to all or a subset of
        this POVM's gpindices.  Such effect vectors are used internally within
        computations involving the parent `Model`.

        Parameters
        ----------
        prefix : str
            A string, usually identitying this POVM, which may be used
            to prefix the simplified gate keys.

        Returns
        -------
        OrderedDict of SPAMVecs
        """
        # Create "simplified" effect vectors, which infer their parent and
        # gpindices from the set of "factor-POVMs" they're constructed with.
        if isinstance(prefix, _Label):  # Deal with case when prefix isn't just a string
            simplified = _collections.OrderedDict(
                [(_Label(prefix.name + '_' + k, prefix.sslbls), self[k]) for k in self.keys()])
        else:
            if prefix: prefix += "_"
            simplified = _collections.OrderedDict(
                [(prefix + k, self[k]) for k in self.keys()])
        return simplified

    def __str__(self):
        s = "Marginalized POVM of length %d\n" \
            % (len(self))
        return s
