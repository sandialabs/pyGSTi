""" Defines the GateSet class and supporting functionality."""
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

import numpy as _np
import scipy as _scipy
import itertools as _itertools
import collections as _collections
import warnings as _warnings
import time as _time
import bisect as _bisect

from ..tools import matrixtools as _mt
from ..tools import gatetools as _gt
from ..tools import slicetools as _slct
from ..tools import likelihoodfns as _lf
from ..tools import jamiolkowski as _jt
from ..tools import compattools as _compat
from ..tools import basistools as _bt
from ..tools import listtools as _lt

from . import gatesetmember as _gm
from . import gatestring as _gs
from . import gate as _gate
from . import spamvec as _sv
from . import povm as _povm
from . import instrument as _instrument
from . import labeldicts as _ld
from . import gaugegroup as _gg
from . import gatematrixcalc as _gatematrixcalc
from . import gatemapcalc as _gatemapcalc
from . import gatetermcalc as _gatetermcalc
from . import autogator as _autogator

from ..baseobjs import VerbosityPrinter as _VerbosityPrinter
from ..baseobjs import Basis as _Basis
from ..baseobjs import Label as _Label


class GateSet(object):
    """
    Encapsulates a set of gate, state preparation, and POVM effect operations.

    A GateSet stores a set of labeled Gate objects and provides dictionary-like
    access to their matrices.  State preparation and POVM effect operations are
    represented as column vectors.
    """

    #Whether access to gates & spam vecs via GateSet indexing is allowed
    _strict = False

    #Whether to perform extra parameter-vector integrity checks
    _pcheck = False

    def __init__(self, default_param="full",
                 prep_prefix="rho", effect_prefix="E", gate_prefix="G",
                 povm_prefix="M", instrument_prefix="I", sim_type="auto"):
        """
        Initialize a gate set.

        Parameters
        ----------
        default_param : {"full", "TP", "CPTP", etc.}, optional
            Specifies the default gate and SPAM vector parameterization type.
            Can be any value allowed by :method:`set_all_parameterizations`,
            which also gives a description of each parameterization type.

        prep_prefix, effect_prefix, gate_prefix,
        povm_prefix, instrument_prefix : string, optional
            Key prefixes designating state preparations, POVM effects,
            gates, POVM, and instruments respectively.  These prefixes allow
            the GateSet to determine what type of object a key corresponds to.

        sim_type : {"auto", "matrix", "map", "termorder:<X>"}
            The type of gate sequence / circuit simulation used to compute any
            requested probabilities, e.g. from :method:`probs` or
            :method:`bulk_probs`.  The default value of `"auto"` automatically
            selects the simulation type, and is usually what you want. Allowed
            values are:

            - "matrix" : gate_matrix-gate_matrix products are computed and
              cached to get composite gates which can then quickly simulate
              a circuit for any preparation and outcome.  High memory demand;
              best for a small number of (1 or 2) qubits.
            - "map" : gate_matrix-state_vector products are repeatedly computed
              to simulate circuits.  Slower for a small number of qubits, but
              faster and more memory efficient for higher numbers of qubits (3+).
            - "termorder:<X>" : Use Taylor expansions of gates in error rates
              to compute probabilities out to some maximum order <X> (an
              integer) in these rates.
        """
        assert(default_param in ('full','TP','CPTP','H+S','S','static',
                                         'H+S terms','clifford','H+S clifford terms'))

        #default_e_param = "full" if default_param == "TP" else default_param

        #Gate dimension of this GateSet (None => unset, to be determined)
        self._dim = None
        self._evotype = None

        #Name and dimension (or list of dims) of the *basis*
        # that the gates and SPAM vectors are expressed in.  This
        # is for interpretational purposes only, and is reset often
        # (for instance, when reading GateSet params from a vector)
        self.reset_basis()

        #SPAM vectors
        self.preps = _ld.OrderedMemberDict(self, default_param, prep_prefix, "spamvec")
        self.povms = _ld.OrderedMemberDict(self, default_param, povm_prefix, "povm")
        self.effects_prefix = effect_prefix
        #self.effects = _ld.OrderedMemberDict(self, default_e_param, effect_prefix, "spamvec")

        #SPAM labels: key = label, value = (prepLabel, effectLabel)
        #self.spamdefs = _ld.OrderedSPAMLabelDict('remainder')

        #Gates
        self.gates = _ld.OrderedMemberDict(self, default_param, gate_prefix, "gate")
        self.instruments = _ld.OrderedMemberDict(self, default_param, instrument_prefix, "instrument")

        if sim_type != "auto":
            self.set_simtype(sim_type)
        else:
            #defer setting sim_type until _dim is set (via _check_dim in labeldicts.py)
            self._calcClass = None
            self._sim_type = sim_type
            self._sim_args = []

        self._default_gauge_group = None
        self._paramvec = _np.zeros(0, 'd')
        self._rebuild_paramvec()
        self._autogator = _autogator.SimpleCompositionAutoGator(self) # the default

        super(GateSet, self).__init__()

    def set_simtype(self, sim_type, calc_cache=None):
        #Calculator selection based on simulation type

        if sim_type == "auto":
            default_param = self.gates.default_param # assume the same for other dicts
            if default_param in ("H+S terms","H+S clifford terms"):
                sim_type = "termorder:1"
            else:
                d = self._dim if (self._dim is not None) else 0
                sim_type = "matrix" if d <= 16 else "map"

        simtype_and_args = sim_type.split(":")
        sim_type = simtype_and_args[0]
        if sim_type == "matrix":      c = _gatematrixcalc.GateMatrixCalc
        elif sim_type == "map":       c = _gatemapcalc.GateMapCalc
        elif sim_type == "termorder": c = _gatetermcalc.GateTermCalc
        else: raise ValueError("Invalid `sim_type` (%s)" % sim_type)

        self._calcClass = c
        self._sim_type = sim_type
        self._sim_args = list(simtype_and_args[1:])

        if sim_type == "termorder":
            cache = calc_cache if (calc_cache is not None) else {} # make a temp cache if none is given
            self._sim_args.append(cache) # add calculation cache as another argument
            



    def _embedGate(self, gateTargetLabels, gateVal):
        """
        Called by OrderedMemberDict._auto_embed to create an embedded-gate
        object that embeds `gateVal` into the sub-space of
        `self.stateSpaceLabels` given by `gateTargetLabels`.

        Parameters
        ----------
        gateTargetLabels : list
            A list of `gateVal`'s target state space labels.

        gateVal : Gate
            The gate object to embed.  Note this should be a legitimate
            Gate-derived object and not just a numpy array.

        Returns
        -------
        Gate
            A gate of the full gate set dimension.
        """
        if self.dim is None:
            raise ValueError("Must set gateset dimension before adding auto-embedded gates.")
        if self.stateSpaceLabels is None:
            raise ValueError("Must set gateset.stateSpaceLabels before adding auto-embedded gates.")

        if gateVal.dim == self.dim:
            return gateVal # if gate operates on full dimension, no need to embed

        # TODO: REMOVE - I think this is redundant w/case above and less general
        #if len(self.stateSpaceLabels.labels) == 1 and \
        #   set(self.stateSpaceLabels.labels[0]) == set(gateTargetLabels):
        #    return gateVal # if gate operates on *all* the labels, no need to embed

        if self._sim_type == "matrix":
            return _gate.EmbeddedGate(self.stateSpaceLabels, gateTargetLabels, gateVal)
        elif self._sim_type in ("map","termorder"):
            return _gate.EmbeddedGateMap(self.stateSpaceLabels, gateTargetLabels, gateVal)
        else:
            assert(False), "Invalid GateSet sim type == %s" % str(self._sim_type)


    @property
    def default_gauge_group(self):
        """
        Gets the default gauge group for performing gauge
        transformations on this GateSet.
        """
        return self._default_gauge_group

    @default_gauge_group.setter
    def default_gauge_group(self, value):
        self._default_gauge_group = value


    @property
    def dim(self):
        """
        The dimension of the gateset, which equals d when the gate
        matrices have shape d x d and spam vectors have shape d x 1.

        Returns
        -------
        int
            gateset dimension
        """
        return self._dim


    def get_dimension(self):
        """
        Get the dimension of the gateset, which equals d when the gate
        matrices have shape d x d and spam vectors have shape d x 1.
        Equivalent to gateset.dim.

        Returns
        -------
        int
            gateset dimension
        """
        return self._dim

    @property
    def prep(self):
        """
        The unique state preparation in this gateset, if one exists.  If not,
        a ValueError is raised.
        """
        if len(self.preps) != 1:
            raise ValueError("'.prep' can only be used on gate sets" +
                             " with a *single* state prep.  This GateSet has" +
                             " %d state preps!" % len(self.preps))
        return list(self.preps.values())[0]


    @property
    def effects(self):
        """
        The unique POVM in this gateset, if one exists.  If not,
        a ValueError is raised.
        """
        if len(self.povms) != 1:
            raise ValueError("'.effects' can only be used on gate sets" +
                             " with a *single* POVM.  This GateSet has" +
                             " %d POVMS!" % len(self.povms))
        return list(self.povms.values())[0]


    def get_basis_name(self):
        """ DEPRECATED: use `<this object>.basis.name` instead. """
        _warnings.warn('gs.get_basis_name() is deprecated. ' + \
                'Use gs.basis.name instead.')
        return self.basis.name

    def get_basis_dimension(self):
        """ DEPRECATED: use `<this object>.basis.dim.dmDim` instead. """
        _warnings.warn('gs.get_basis_dimension() is deprecated. ' + \
                'Use gs.basis.dim.dmDim (same functionality) or gs.basis.dim.blockDims (full blockDims) instead')
        return self.basis.dim.dmDim

    def set_basis(self, name, dimension):
        """ DEPRECATED: use `<this object>.basis = Basis(...) instead. """
        _warnings.warn('gs.set_basis() is deprecated. ' + \
                'Use gs.basis = Basis({}, {}) ' + \
                '(or another method of basis construction, ' + \
                'like gs.basis = Basis([(\'std\', 2), (\'gm\', 2)])) ' + \
                'instead.'.format(name, dimension))
        self.basis = _Basis(name, dimension)

    def reset_basis(self):
        """
        "Forgets" the basis name and dimension by setting
        these quantities to "unkown" and None, respectively.
        """
        self.basis = _Basis('unknown', None)
        self.stateSpaceLabels = None

    def get_prep_labels(self):
        """
        DEPRECATED. Get the labels of state preparation vectors.

        Returns
        -------
        list of strings
        """
        assert(False),"Deprecated!"
        #return list(self.preps.keys())


    def get_effect_labels(self):
        """
        DEPRECATED. Get all the effect vector labels.

        Returns
        -------
        list of strings
        """
        assert(False),"Deprecated!"
        #return list(self.effects.keys())


    def get_preps(self):
        """
        DEPRECATED.
        Get an list of all the state prepartion vectors.  These
        vectors are copies of internally stored data and thus
        can be modified without altering the gateset.

        Returns
        -------
        list of numpy arrays
            list of state preparation vectors of shape (dim, 1).
        """
        assert(False),"Deprecated!"
        #return [ self.preps[l].copy() for l in self.get_prep_labels() ]

    def get_effects(self):
        """
        DEPRECATED.
        Get an list of all the POVM effect vectors.  This list will include
        the "compliment" effect vector at the end of the list if one has been
        specified.  Also, the returned vectors are copies of internally stored
        data and thus can be modified without altering the gateset.

        Returns
        -------
        list of numpy arrays
            list of POVM effect vectors of shape (dim, 1).
        """
        assert(False),"Deprecated!"
        #return [ self.effects[l].copy() for l in self.get_effect_labels() ]


    def num_preps(self):
        """
        DEPRECATED. Get the number of state preparation vectors

        Returns
        -------
        int
        """
        assert(False),"Deprecated!"
        #return len(self.preps)


    def num_effects(self):
        """
        DEPRECATED. Get the number of effect vectors.

        Returns
        -------
        int
        """
        assert(False),"Deprecated!"
        #return len(self.effects)


    def get_reverse_spam_defs(self):
        """
        Get a reverse-lookup dictionary for spam labels.

        Returns
        -------
        OrderedDict
            a dictionary with keys == (prepLabel,effectLabel) tuples and
            values == SPAM labels.
        """
        assert(False),"Deprecated!"
        #d = _collections.OrderedDict()
        #for label in self.spamdefs:
        #    d[  self.spamdefs[label] ] = label
        #return d

    def get_spam_labels(self):
        """
        Get a list of all the spam labels.

        Returns
        -------
        list of strings
        """
        assert(False),"Deprecated!"
        #return list(self.spamdefs.keys())


    def get_spamgate(self, spamLabel):
        """
        Construct the SPAM gate associated with
        a given spam label.

        Parameters
        ----------
        spamLabel : str
           the spam label to construct a "spam gate" for.

        Returns
        -------
        numpy array
        """
        assert(False),"Deprecated!"
        #return self._calc()._make_spamgate(spamLabel)



    def __setitem__(self, label, value):
        """
        Set a Gate or SPAM vector associated with a given label.

        Parameters
        ----------
        label : string
            the gate or SPAM vector label.

        value : numpy array or Gate or SPAMVec
            a gate matrix, SPAM vector, or object, which must have the
            appropriate dimension for the GateSet and appropriate type
            given the prefix of the label.
        """
        if GateSet._strict:
            raise KeyError("Strict-mode: invalid key %s" % repr(label))

        if not isinstance(label, _Label): label = _Label(label)

        if label.has_prefix(self.preps._prefix):
            self.preps[label] = value
        elif label.has_prefix(self.povms._prefix):
            self.povms[label] = value
        #elif label.has_prefix(self.effects._prefix):
        #    self.effects[label] = value
        elif label.has_prefix(self.gates._prefix):
            self.gates[label] = value
        elif label.has_prefix(self.instruments._prefix, typ="any"):
            self.instruments[label] = value
        else:
            raise KeyError("Key %s has an invalid prefix" % label)

    def __getitem__(self, label):
        """
        Get a Gate or SPAM vector associated with a given label.

        Parameters
        ----------
        label : string
            the gate or SPAM vector label.
        """
        if GateSet._strict:
            raise KeyError("Strict-mode: invalid key %s" % label)

        if not isinstance(label, _Label): label = _Label(label)

        if label.has_prefix(self.preps._prefix):
            return self.preps[label]
        elif label.has_prefix(self.povms._prefix):
            return self.povms[label]
        #elif label.name.startswith(self.effects._prefix):
        #    return self.effects[label]
        elif label.has_prefix(self.gates._prefix):
            return self.gates[label]
        elif label.has_prefix(self.instruments._prefix, typ="any"):
            return self.instruments[label]
        else:
            raise KeyError("Key %s has an invalid prefix" % label)


    def set_all_parameterizations(self, parameterization_type, extra=None):
        """
        Convert all gates and SPAM vectors to a specific parameterization
        type.

        Parameters
        ----------
        parameterization_type : string
            The gate and SPAM vector parameterization type.  Allowed
            values are:

            - "full" : each gate / SPAM element is an independent parameter
            - "TP" : Trace-Preserving gates and state preps
            - "CPTP" : Completely-Positive-Trace-Preserving gates
            - "H+S" : Hamiltonian and Pauli-Stochastic errors only
            - "S" : Pauli-Stochastic errors only
            - "static" : no parameters
            - "static unitary" : no parameters; convert superops to unitaries
            - "H+S terms" : like H+S, but support "svterm" evolution type
            - "H+S clifford terms" : like H+S, but support "cterm" evo. type
            - "clifford" : no parameters; convert unitaries to Clifford symplecitics.

        extra : dict, optional
            For `"H+S terms"` type, this may specify a dictionary
            of unitary gates and pure state vectors to be used
            as the *ideal* operation of each gate/SPAM vector.
        """
        typ = parameterization_type
        assert(parameterization_type in ('full','TP','CPTP','H+S','S','static',
                                         'H+S terms','clifford','H+S clifford terms',
                                         'static unitary'))

        povmtyp = rtyp = ityp = "TP" if typ in ("CPTP","H+S","S") else typ
        #rtyp = "CPTP" if typ in ("H+S","S") else typ #TESTING, but CPTP spamvec still unreliable

        #Update dim and evolution type so that setting converted elements works correctly
        orig_dim = self.dim
        orig_evotype = self._evotype
        if typ in ('full','TP','CPTP','H+S','S','static'):
            self._evotype = "densitymx"
            if self._sim_type not in ("matrix","map"):
                self.set_simtype("matrix" if self.dim <= 16 else "map")

        elif typ == 'clifford':
            self._evotype = "stabilizer"
            self.set_simtype("map")

        elif typ == 'H+S terms':
            self._evotype = "svterm"
            if self._sim_type != "termorder":
                self.set_simtype("termorder:1")

        elif typ == 'H+S clifford terms':
            self._evotype = "cterm"
            if self._sim_type != "termorder":
                self.set_simtype("termorder:1")

        elif typ == 'static unitary':
            assert(self._evotype == "densitymx"), \
                "Can only convert to 'static unitary' from a density-matrix evolution type."
            self._evotype = "statevec"
            self._dim = int(round(_np.sqrt(self.dim))) # reduce dimension d -> sqrt(d)
            if self._sim_type not in ("matrix","map"):
                self.set_simtype("matrix" if self.dim <= 4 else "map")

        basis = self.basis
        if extra is None: extra = {}

        for lbl,gate in self.gates.items():
            self.gates[lbl] = _gate.convert(gate, typ, basis,
                                            extra.get(lbl,None))

        for lbl,inst in self.instruments.items():
            self.instruments[lbl] = _instrument.convert(inst, ityp, basis,
                                                        extra.get(lbl,None))

        for lbl,vec in self.preps.items():
            self.preps[lbl] = _sv.convert(vec, rtyp, basis,
                                          extra.get(lbl,None))

        for lbl,povm in self.povms.items():
            self.povms[lbl] = _povm.convert(povm, povmtyp, basis,
                                            extra.get(lbl,None))

        if typ == 'full':
            self.default_gauge_group = _gg.FullGaugeGroup(self.dim)
        elif typ == 'TP':
            self.default_gauge_group = _gg.TPGaugeGroup(self.dim)
        elif typ == 'CPTP':
            self.default_gauge_group = _gg.UnitaryGaugeGroup(self.dim, basis)
        else: # typ in ('static','H+S','S', 'H+S terms')
            self.default_gauge_group = _gg.TrivialGaugeGroup(self.dim)


    #def __getstate__(self):
    #    #Returns self.__dict__ by default, which is fine

    def __setstate__(self, stateDict):
        if "effects" in stateDict:
            #unpickling an OLD-version GateSet - like a re-__init__
            #print("DB: UNPICKLING AN OLD GATESET"); print("Keys = ",stateDict.keys())
            default_param = "full"
            self.preps = _ld.OrderedMemberDict(self, default_param, "rho", "spamvec")
            self.povms = _ld.OrderedMemberDict(self, default_param, "M", "povm")
            self.effects_prefix = 'E'
            self.gates = _ld.OrderedMemberDict(self, default_param, "G", "gate")
            self.instruments = _ld.OrderedMemberDict(self, default_param, "I", "instrument")
            self._paramvec = _np.zeros(0, 'd')
            self._rebuild_paramvec()

            self._dim = stateDict['_dim']
            self._calcClass = stateDict.get('_calcClass',_gatematrixcalc.GateMatrixCalc)
            self._evotype = "densitymx"
            self._autogator = None # the default for *old* gatesets
            self._default_gauge_group = stateDict['_default_gauge_group']
            self.basis = stateDict.get('basis', _Basis('unknown', None))
            if self.basis.name == "unknown" and '_basisNameAndDim' in stateDict:
                self.basis = _Basis(stateDict['_basisNameAndDim'][0],
                                    stateDict['_basisNameAndDim'][1])

            assert(len(stateDict['preps']) <= 1), "Cannot convert GateSets with multiple preps!"
            for lbl,gate in stateDict['gates'].items(): self.gates[lbl] = gate
            for lbl,vec in stateDict['preps'].items(): self.preps[lbl] = vec

            effect_vecs = []; remL = stateDict['_remainderlabel']
            comp_lbl = None
            for sl,(prepLbl,ELbl) in stateDict['spamdefs'].items():
                assert((prepLbl,ELbl) != (remL,remL)), "Cannot convert sum-to-one spamlabel!"
                if ELbl == remL:  comp_lbl = str(sl)
                else: effect_vecs.append( (str(sl), stateDict['effects'][ELbl]) )
            if comp_lbl is not None:
                comp_vec = stateDict['_povm_identity'] - sum([v for sl,v in effect_vecs])
                effect_vecs.append( (comp_lbl, comp_vec) )
                self.povms['Mdefault'] = _povm.TPPOVM(effect_vecs)
            else:
                self.povms['Mdefault'] = _povm.UnconstrainedPOVM(effect_vecs)


        else:
            self.__dict__.update(stateDict)

        #Additionally, must re-connect this gateset as the parent
        # of relevant OrderedDict-derived classes, which *don't*
        # preserve this information upon pickling so as to avoid
        # circular pickling...
        self.preps.parent = self
        self.povms.parent = self
        #self.effects.parent = self
        self.gates.parent = self
        self.instruments.parent = self
        for o in self.preps.values(): o._parent = self
        for o in self.povms.values(): o._parent = self
        #for o in self.effects.values(): o._parent = self
        for o in self.gates.values(): o._parent = self
        for o in self.instruments.values(): o._parent = self



    def num_params(self):
        """
        Return the number of free parameters when vectorizing
        this gateset.

        Returns
        -------
        int
            the number of gateset parameters.
        """
        return len(self._paramvec)


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
        rhoSize = [ rho.size for rho in self.preps.values() ]
        povmSize = [ povm.num_elements() for povm in self.povms.values() ]
        gateSize = [ gate.size for gate in self.gates.values() ]
        instSize = [ i.num_elements() for i in self.instruments.values() ]
        return sum(rhoSize) + sum(povmSize) + sum(gateSize) + sum(instSize)


    def num_nongauge_params(self):
        """
        Return the number of non-gauge parameters when vectorizing
        this gateset according to the optional parameters.

        Returns
        -------
        int
            the number of non-gauge gateset parameters.
        """
        return self.num_params() - self.num_gauge_params()


    def num_gauge_params(self):
        """
        Return the number of gauge parameters when vectorizing
        this gateset according to the optional parameters.

        Returns
        -------
        int
            the number of gauge gateset parameters.
        """
        dPG = self._calc()._buildup_dPG()
        gaugeDirs = _mt.nullspace_qr(dPG) #cols are gauge directions
        return _np.linalg.matrix_rank(gaugeDirs[0:self.num_params(),:])


    def _check_paramvec(self, debug=False):
        if debug: print("---- GateSet._check_paramvec ----")
        TOL=1e-8
        for lbl,obj in self.iter_objs():
            if debug: print(lbl,":",obj.num_params(),obj.gpindices)
            w = obj.to_vector()
            msg = "None" if (obj.parent is None) else id(obj.parent)
            assert(obj.parent is self), "%s's parent is not set correctly (%s)!" % (lbl,msg)
            if obj.gpindices is not None and len(w) > 0:
                if _np.linalg.norm(self._paramvec[obj.gpindices]-w) > TOL:
                    raise ValueError("%s is out of sync with paramvec!!!" % lbl)


    def _clean_paramvec(self):
        """ Updates _paramvec corresponding to any "dirty" elements, which may
            have been modified without out knowing, leaving _paramvec out of
            sync with the element's internal data.  It *may* be necessary
            to resolve conflicts where multiple dirty elements want different
            values for a single parameter.  This method is used as a safety net
            that tries to insure _paramvec & GateSet elements are consistent
            before their use."""
        dirty = False; TOL=1e-8
        for _,obj in self.iter_objs():
            if obj.dirty:
                w = obj.to_vector()
                if _np.linalg.norm(self._paramvec[obj.gpindices]-w) > TOL:
                    self._paramvec[obj.gpindices] = w; dirty = True

        if dirty:
            #re-update everything to ensure consistency
            #print("DEBUG: non-trivailly CLEANED paramvec due to dirty elements")
            self.from_vector(self._paramvec,False)

        if GateSet._pcheck: self._check_paramvec()


    def _update_paramvec(self, modified_obj=None):
        """Updates self._paramvec after a member of this GateSet is modified"""
        self._rebuild_paramvec() # prepares _paramvec & gpindices

        #update parameters changed by modified_obj
        self._paramvec[modified_obj.gpindices] = modified_obj.to_vector()

        #re-initialze any members that also depend on the updated parameters
        modified_indices = set(modified_obj.gpindices_as_array())
        for _,obj in self.iter_objs():
            if obj is modified_obj: continue
            if modified_indices.intersection(obj.gpindices_as_array()):
                obj.from_vector(self._paramvec[obj.gpindices])


    def _rebuild_paramvec(self):
        """ Resizes self._paramvec and updates gpindices & parent members as needed,
            and will initialize new elements of _paramvec, but does NOT change
            existing elements of _paramvec (use _update_paramvec for this)"""
        v = self._paramvec; Np = self.num_params()
        off = 0; shift = 0

        #ellist = ", ".join(list(self.preps.keys()) +list(self.povms.keys()) +list(self.gates.keys()))
        #print("DEBUG: rebuilding... %s" % ellist)

        #Step 1: remove any unused indices from paramvec and shift accordingly
        used_gpindices = set()
        for _,obj in self.iter_objs():
            if obj.gpindices is not None:
                assert(obj.parent is self), "Member's parent is not set correctly!"
                used_gpindices.update( obj.gpindices_as_array() )
            else:
                assert(obj.parent is self or obj.parent is None)
                #Note: ok for objects to have parent == None if their gpindices is also None

        indices_to_remove = sorted(set(range(Np)) - used_gpindices)

        if len(indices_to_remove) > 0:
            #print("DEBUG: Removing %d params:"  % len(indices_to_remove), indices_to_remove)
            v = _np.delete(v, indices_to_remove)
            get_shift = lambda j: _bisect.bisect_left(indices_to_remove, j)
            memo = set() #keep track of which object's gpindices have been set
            for lbl,obj in self.iter_objs():
                if obj.gpindices is not None:
                    if id(obj) in memo: continue #already processed
                    if isinstance(obj.gpindices, slice):
                        new_inds = _slct.shift(obj.gpindices,
                                               -get_shift(obj.gpindices.start))
                    else:
                        new_inds = []
                        for i in obj.gpindices:
                            new_inds.append(i - get_shift(i))
                        new_inds = _np.array(new_inds,_np.int64)
                    obj.set_gpindices( new_inds, self, memo)


        # Step 2: add parameters that don't exist yet
        memo = set() #keep track of which object's gpindices have been set
        for _,obj in self.iter_objs():

            if shift > 0 and obj.gpindices is not None:
                if isinstance(obj.gpindices, slice):
                    obj.set_gpindices(_slct.shift(obj.gpindices, shift), self, memo)
                else:
                    obj.set_gpindices(obj.gpindices+shift, self, memo)  #works for integer arrays

            if obj.gpindices is None or obj.parent is not self:
                #Assume all parameters of obj are new independent parameters
                num_new_params = obj.allocate_gpindices( off, self )
                objvec = obj.to_vector() #may include more than "new" indices
                if num_new_params > 0:
                    new_local_inds = _gm._decompose_gpindices(obj.gpindices, slice(off,off+num_new_params))
                    assert(len(objvec[new_local_inds]) == num_new_params)
                    v = _np.insert(v, off, objvec[new_local_inds])
                #print("objvec len = ",len(objvec), "num_new_params=",num_new_params," gpinds=",obj.gpindices) #," loc=",new_local_inds)

                #obj.set_gpindices( slice(off, off+obj.num_params()), self )
                #shift += obj.num_params()
                #off += obj.num_params()

                shift += num_new_params
                off += num_new_params
                #print("DEBUG: %s: alloc'd & inserted %d new params.  indices = " % (lbl,obj.num_params()), obj.gpindices, " off=",off)
            else:
                inds = obj.gpindices_as_array()
                M = max(inds) if len(inds)>0 else -1; L = len(v)
                #print("DEBUG: %s: existing indices = " % (lbl), obj.gpindices, " M=",M," L=",L)
                if M >= L:
                    #Some indices specified by obj are absent, and must be created.
                    w = obj.to_vector()
                    v = _np.concatenate((v, _np.empty(M+1-L,'d')),axis=0) # [v.resize(M+1) doesn't work]
                    shift += M+1-L
                    for ii,i in enumerate(inds):
                        if i >= L: v[i] = w[ii]
                    #print("DEBUG:    --> added %d new params" % (M+1-L))
                off = M+1

        self._paramvec = v
        #print("DEBUG: Done rebuild: %d params" % len(v))

    def _init_virtual_obj(self, obj):
        """ 
        Initializes a "virtual object" - an object (e.g. Gate) that *could* be a
        member of the GateSet but won't be, as it's just built for temporary
        use (e.g. the parallel action of several "base" gates).  As such
        we need to fully initialize its parent and gpindices members so it 
        knows it belongs to this GateSet BUT it's not allowed to add any new
        parameters (they'd just be temporary).  It's also assumed that virtual
        objects don't need to be to/from-vectored as there are already enough
        real (non-virtual) gates/spamvecs/etc. to accomplish this.
        """
        if obj.gpindices is not None:
            assert(obj.parent is self), "Virtual obj has incorrect parent already set!"
            return # if parent is already set we assume obj has already been init

        #Assume all parameters of obj are new independent parameters
        num_new_params = obj.allocate_gpindices(self.num_params(), self)
        assert(num_new_params == 0),"Virtual object is requesting %d new params!" % num_new_params


    def to_vector(self):
        """
        Returns the gateset vectorized according to the optional parameters.

        Returns
        -------
        numpy array
            The vectorized gateset parameters.
        """
        self._clean_paramvec()
        return self._paramvec


    def from_vector(self, v, reset_basis=True):
        """
        The inverse of to_vector.  Loads values of gates and rho and E vecs from
        from the vector `v`.  Note that `v` does not specify the number of
        gates, etc., and their labels: this information must be contained in
        this `GateSet` prior to calling `from_vector`.  In practice, this just
        means you should call the `from_vector` method using the same `GateSet`
        that was used to generate the vector `v` in the first place.
        """
        assert( len(v) == self.num_params() )

        self._paramvec = v.copy()
        for _,obj in self.iter_objs():
            obj.from_vector( v[obj.gpindices] )
            obj.dirty = False #object is known to be consistent with _paramvec

        if reset_basis:
            self.reset_basis()
            # assume the vector we're loading isn't producing gates & vectors in
            # a known basis.
        if GateSet._pcheck: self._check_paramvec()


    def deriv_wrt_params(self):
        """
        Construct a matrix whose columns are the vectorized derivatives of all
        the gateset's raw matrix and vector *elements* (placed in a vector)
        with respect to each single gateset parameter.

        Thus, each column has length equal to the number of elements in the
        gateset, and there are num_params() columns.  In the case of a "fully
        parameterized gateset" (i.e. all gate matrices and SPAM vectors are
        fully parameterized) then the resulting matrix will be the (square)
        identity matrix.

        Returns
        -------
        numpy array
            2D array of derivatives.
        """
        return self._calc().deriv_wrt_params()


    def get_nongauge_projector(self, itemWeights=None, nonGaugeMixMx=None):
        """
        Construct a projector onto the non-gauge parameter space, useful for
        isolating the gauge degrees of freedom from the non-gauge degrees of
        freedom.

        Parameters
        ----------
        itemWeights : dict, optional
            Dictionary of weighting factors for individual gates and spam operators.
            Keys can be gate, state preparation, POVM effect, spam labels, or the
            special strings "gates" or "spam" whic represent the entire set of gate
            or SPAM operators, respectively.  Values are floating point numbers.
            These weights define the metric used to compute the non-gauge space,
            *orthogonal* the gauge space, that is projected onto.

        nonGaugeMixMx : numpy array, optional
            An array of shape (nNonGaugeParams,nGaugeParams) specifying how to
            mix the non-gauge degrees of freedom into the gauge degrees of
            freedom that are projected out by the returned object.  This argument
            essentially sets the off-diagonal block of the metric used for
            orthogonality in the "gauge + non-gauge" space.  It is for advanced
            usage and typically left as None (the default).
.

        Returns
        -------
        numpy array
           The projection operator as a N x N matrix, where N is the number
           of parameters (obtained via num_params()).  This projector acts on
           parameter-space, and has rank equal to the number of non-gauge
           degrees of freedom.
        """
        return self._calc().get_nongauge_projector(itemWeights, nonGaugeMixMx)


    def transform(self, S):
        """
        Update each of the gate matrices G in this gateset with inv(S) * G * S,
        each rhoVec with inv(S) * rhoVec, and each EVec with EVec * S

        Parameters
        ----------
        S : GaugeGroupElement
            A gauge group element which specifies the "S" matrix
            (and it's inverse) used in the above similarity transform.
        """
        for rhoVec in self.preps.values():
            rhoVec.transform(S,'prep')

        for povm in self.povms.values():
            povm.transform(S)

        for gateObj in self.gates.values():
            gateObj.transform(S)

        for instrument in self.instruments.values():
            instrument.transform(S)

        self._clean_paramvec() #transform may leave dirty members


    def _calc(self):
        if not hasattr(self,"_calcClass"): #for backward compatibility
            self._calcClass = _gatematrixcalc.GateMatrixCalc

        compiled_effects = _collections.OrderedDict()
        for povm_lbl,povm in self.povms.items():
            for k,e in povm.compile_effects(povm_lbl).items():
                compiled_effects[k] = e

        compiled_gates = _collections.OrderedDict()
        for k,g in self.gates.items(): compiled_gates[k] = g
        for inst_lbl,inst in self.instruments.items():
            for k,g in inst.compile_gates(inst_lbl).items():
                compiled_gates[k] = g

        kwargs = {}
        if self._sim_type == "termorder":
            kwargs['max_order'] = int(self._sim_args[0])
            kwargs['cache'] = self._sim_args[-1] # always the list argument

        assert(self._calcClass is not None), "Gateset does not have a calculator setup yet!"
        return self._calcClass(self._dim, compiled_gates, self.preps,
                               compiled_effects, self._paramvec, self._autogator, **kwargs)


    def split_gatestring(self, gatestring, erroron=('prep','povm')):
        """
        Splits a gate string into prepLabel + gatesOnlyString + povmLabel
        components.  If `gatestring` does not contain a prep label or a
        povm label a default label is returned if one exists.

        Parameters
        ----------
        gatestring : GateString
            A gate string, possibly beginning with a state preparation
            label and ending with a povm label.

        erroron : tuple of {'prep','povm'}
            A ValueError is raised if a preparation or povm label cannot be
            resolved when 'prep' or 'povm' is included in 'erroron'.  Otherwise
            `None` is returned in place of unresolvable labels.  An exception
            is when this gateset has no preps or povms, in which case `None`
            is always returned and errors are never raised, since in this
            case one usually doesn't expect to use the GateSet to compute
            probabilities (e.g. in germ selection).

        Returns
        -------
        prepLabel : str or None
        gatesOnlyString : GateString
        povmLabel : str or None
        """
        if len(gatestring) > 0 and gatestring[0] in self.preps:
            prep_lbl = gatestring[0]
            gatestring = gatestring[1:]
        elif len(self.preps) == 1:
            prep_lbl = list(self.preps.keys())[0]
        else:
            if 'prep' in erroron and len(self.preps) > 0:
                raise ValueError("Cannot resolve state prep in %s" % gatestring)
            else: prep_lbl = None

        if len(gatestring) > 0 and gatestring[-1] in self.povms:
            povm_lbl = gatestring[-1]
            gatestring = gatestring[:-1]
        elif len(self.povms) == 1:
            povm_lbl = list(self.povms.keys())[0]
        else:
            if 'povm' in erroron and len(self.povms) > 0:
                raise ValueError("Cannot resolve POVM in %s" % gatestring)
            else: povm_lbl = None

        return prep_lbl, gatestring, povm_lbl


    def compile_gatestrings(self, gatestrings, dataset=None):
        """
        Compiles a list of :class:`GateString`s.

        Gate strings must be "compiled" before probabilities can be computed for
        them. Each string corresponds to some number of "outcomes", indexed by an
        "outcome label" that is a tuple of POVM-effect or instrument-element
        labels like "0".  Compiling creates maps between gate strings and their
        outcomes and the structures used in probability computation (see return
        values below).

        Parameters
        ----------
        gatestrings : list of GateStrings
            The list to compile.

        dataset : DataSet, optional
            If not None, restrict what is compiled to only those
            probabilities corresponding to non-zero counts (observed
            outcomes) in this data set.

        Returns
        -------
        raw_spamTuples_dict : collections.OrderedDict
            A dictionary whose keys are raw gate sequences (containing just
            "compiled" gates, i.e. not instruments), and whose values are
            lists of (preplbl, effectlbl) tuples.  The effectlbl names a
            "compiled" effect vector; preplbl is just a prep label. Each tuple
            corresponds to a single "final element" of the computation, e.g. a
            probability.  The ordering is important - and is why this needs to be
            an ordered dictionary - when the lists of tuples are concatenated (by
            key) the resulting tuple orderings corresponds to the final-element
            axis of an output array that is being filled (computed).

        elIndices : collections.OrderedDict
            A dictionary whose keys are integer indices into `gatestrings` and
            whose values are slices and/or integer-arrays into the space/axis of
            final elements.  Thus, to get the final elements corresponding to
            `gatestrings[i]`, use `filledArray[ elIndices[i] ]`.

        outcomes : collections.OrderedDict
            A dictionary whose keys are integer indices into `gatestrings` and
            whose values are lists of outcome labels (an outcome label is a tuple
            of POVM-effect and/or instrument-element labels).  Thus, to obtain
            what outcomes the i-th gate strings's final elements
            (`filledArray[ elIndices[i] ]`)  correspond to, use `outcomes[i]`.

        nTotElements : int
            The total number of "final elements" - this is how big of an array
            is need to hold all of the probabilities `gatestrings` generates.
        """
        # gateset.compile -> odict[raw_gstr] = spamTuples, elementIndices, nElements
        # dataset.compile -> outcomeLabels[i] = list_of_ds_outcomes, elementIndices, nElements
        # compile all gsplaq strs -> elementIndices[(i,j)],

        gatestrings = [ _gs.GateString(gs) for gs in gatestrings ] # cast to GateStrings

        #Indexed by raw gate string
        raw_spamTuples_dict = _collections.OrderedDict()  # final
        raw_gateOutcomes_dict = _collections.OrderedDict()
        raw_offsets = _collections.OrderedDict()

        #Indexed by parent index (an integer)
        elIndicesByParent = _collections.OrderedDict() # final
        outcomesByParent = _collections.OrderedDict()  # final
        elIndsToOutcomesByParent = _collections.OrderedDict()

        # Helper dict: (rhoLbl,POVM_ELbl) -> (Elbl,) mapping
        spamTupleToOutcome = { None : ("NONE",) } #Dummy label for placeholding (see resolveSPAM below)
        for prep_lbl in self.preps:
            for povm_lbl in self.povms:
                for elbl in self.povms[povm_lbl]:
                    spamTupleToOutcome[ (prep_lbl, povm_lbl + "_" + elbl) ] = (elbl,)


        def resolveSPAM(gatestring):
            """ Determines spam tuples that correspond to gatestring
                and strips any spam-related pieces off """
            prep_lbl, gatestring, povm_lbl = \
                self.split_gatestring(gatestring)
            if prep_lbl is None or povm_lbl is None:
                spamtups = [ None ] #put a single "dummy" spam-tuple placeholder
                  # so that there's a single "element" for each compiled string,
                  # which means that the usual "lookup" or "elIndices" will map
                  # original gatestring-list indices to compiled-string, i.e.,
                  # evalTree index, which is useful when computing products
                  # (often the case when a GateSet has no preps or povms,
                  #  e.g. in germ selection)
            else:
                spamtups = [ (prep_lbl, povm_lbl + "_" + elbl)
                             for elbl in self.povms[povm_lbl]]
            return gatestring, spamtups

        def process(s, spamtuples, observed_outcomes, elIndsToOutcomes,
                    gate_outcomes=(), start=0):
            """
            Implements recursive processing of a string. Separately
            implements two different behaviors:
              "add" : add entries to raw_spamTuples_dict and raw_gateOutcomes_dict
              "index" : adds entries to elIndicesByParent and outcomesByParent
                        assuming that raw_spamTuples_dict and raw_gateOutcomes_dict
                        are already build (and won't be modified anymore).
            """
            for i,gate_label in enumerate(s[start:],start=start):

                # OLD: now allow "gate-level" labels which can contain
                # multiple (parallel) instrument labels
                #if gate_label in self.instruments: 
                #    #we've found an instrument - recurse!
                #    for inst_el_lbl in self.instruments[gate_label]:
                #        compiled_el_lbl = gate_label + "_" + inst_el_lbl
                #        process(s[0:i] + _gs.GateString((compiled_el_lbl,)) + s[i+1:],
                #                spamtuples, elIndsToOutcomes, gate_outcomes + (inst_el_lbl,), i+1)
                #    break

                if any([ sub_gl in self.instruments for sub_gl in gate_label.components]):
                    # we've found an instrument - recurse!
                    sublabel_tups_to_iter = [] # one per label component (may be only 1)
                    for sub_gl in gate_label.components:
                        if sub_gl in self.instruments:
                            sublabel_tups_to_iter.append( [ (sub_gl,inst_el_lbl)
                                                            for inst_el_lbl in self.instruments[sub_gl].keys() ])
                        else:
                            sublabel_tups_to_iter.append( [(sub_gl,None)] ) # just a single element
                            
                    for sublabel_tups in _itertools.product(*sublabel_tups_to_iter):
                        sublabels = [] # the sub-labels of the overall gate label to add
                        outcomes = [] # the outcome tuple associated with this overall label
                        for sub_gl,inst_el_lbl in sublabel_tups:
                            if inst_el_lbl is not None:
                                sublabels.append(sub_gl + "_" + inst_el_lbl)
                                outcomes.append(inst_el_lbl)
                            else:
                                sublabels.append(sub_gl)
                                
                        compiled_el_lbl = _Label(sublabels)
                        compiled_el_outcomes = tuple(outcomes)
                        process(s[0:i] + _gs.GateString((compiled_el_lbl,)) + s[i+1:],
                                spamtuples, observed_outcomes, elIndsToOutcomes,
                                gate_outcomes + compiled_el_outcomes, i+1)
                    break
                    
            else: #no instruments -- add "raw" gate string s
                if s in raw_spamTuples_dict:
                    assert(gate_outcomes == raw_gateOutcomes_dict[s]) #DEBUG
                    #if action == "add":
                    od = raw_spamTuples_dict[s] # ordered dict
                    for spamtup in spamtuples:
                        outcome_tup = gate_outcomes + spamTupleToOutcome[spamtup]
                        if (observed_outcomes is not None) and \
                           (outcome_tup not in observed_outcomes): continue
                           # don't add spamtuples we don't observe
                        
                        spamtup_indx = od.get(spamtup,None)
                        if spamtup is None:
                            # although we've seen this raw string, we haven't
                            # seen spamtup yet - add it at end
                            spamtup_indx = len(od)
                            od[spamtup] = spamtup_indx

                        #Link the current iParent to this index (even if it was already going to be computed)
                        elIndsToOutcomes[(s,spamtup_indx)] = outcome_tup

                    #OLD: raw_spamTuples_dict[s] = _lt.remove_duplicates(raw_spamTuples_dict[s] + spamtuples)
                    # Note: there should only be duplicates if there are duplicates in
                    # original `gatestring_list` - check this?

                    #OLD - "index" case
                    #elif action == "index":  # fill *ByParent dicts
                    #    assert(iParent is not None)
                    #    offset = raw_offsets[s]
                    #    all_spamtuples = raw_spamTuples_dict[s] # an OrderedDict
                    #    final_outcomes = [ spamTupleToOutcome[x] for x in spamtuples ]
                    #    my_spamTuple_indices = [ offset+all_spamtuples[x] for x in spamtuples ]
                    #    my_outcome_tuples =  [ gate_outcomes + x for x in final_outcomes ]
                    #    for i,tup in zip(my_spamTuple_indices,my_outcome_tuples):
                    #        if i not in elIndicesByParent[iParent]: #don't duplicate existing indices
                    #            elIndicesByParent[iParent][i] = True #just a placeholder; just care about (ordered) keys
                    #            outcomesByParent[iParent].append(tup)
                    #        else: assert(tup in outcomesByParent) # pragma: no check
                    #                # double-check - could REMOVE for speed in future

                else:
                    # Note: store elements of raw_spamTuples_dict as dicts for
                    # now, for faster lookup during "index" mode
                    outcome_tuples =  [ gate_outcomes + spamTupleToOutcome[x] for x in spamtuples ]
                    
                    if observed_outcomes is not None:
                        # only add els of `spamtuples` corresponding to observed data (w/indexes starting at 0)
                        spamtup_dict = _collections.OrderedDict(); ist = 0
                        for spamtup,outcome_tup in zip(spamtuples, outcome_tuples):
                            if outcome_tup in observed_outcomes:
                                spamtup_dict[spamtup] = ist
                                elIndsToOutcomes[(s,ist)] = outcome_tup
                                ist += 1
                    else:
                        # add all els of `spamtuples` (w/indexes starting at 0)
                        spamtup_dict = _collections.OrderedDict( [
                            (spamtup,i) for i,spamtup in enumerate(spamtuples) ] )

                        # OLD: spamTuple_indices = range(len(spamtuples))
                        # OLD: for ist,out_tup in zip(spamTuple_indices,outcome_tuples):
                        for ist,out_tup in enumerate(outcome_tuples): # ist = spamtuple index
                            elIndsToOutcomes[(s,ist)] = out_tup # element index is given by (parent_gatestring, spamtuple_index) tuple
                              # Note: works even if `i` already exists - doesn't reorder keys then

                    raw_spamTuples_dict[s] = spamtup_dict
                    raw_gateOutcomes_dict[s] = gate_outcomes #DEBUG

        #Begin actual processing work:

        # Step1: recursively populate raw_spamTuples_dict,
        #        raw_gateOutcomes_dict, and elIndsToOutcomesByParent
        resolved_gatestrings = list(map(resolveSPAM, gatestrings))
        for iParent,(gstr,spamtuples) in enumerate(resolved_gatestrings):
            elIndsToOutcomesByParent[iParent] = _collections.OrderedDict()
            oouts = None if (dataset is None) else set(dataset[gstr].outcomes)
            process(gstr,spamtuples, oouts, elIndsToOutcomesByParent[iParent])

        # Step2: fill raw_offsets dictionary
        off = 0
        for raw_str, spamtuples in raw_spamTuples_dict.items():
            raw_offsets[raw_str] = off; off += len(spamtuples)
        nTotElements = off

        # Step3: split elIndsToOutcomesByParent into
        #        elIndicesByParent and outcomesByParent
        for iParent,elIndsToOutcomes in elIndsToOutcomesByParent.items():
            elIndicesByParent[iParent] = []
            outcomesByParent[iParent] = []
            for (raw_str,rel_spamtup_indx),outcomes in elIndsToOutcomes.items():
                elIndicesByParent[iParent].append( raw_offsets[raw_str]+rel_spamtup_indx )
                outcomesByParent[iParent].append( outcomes )
            elIndicesByParent[iParent] = _slct.list_to_slice(elIndicesByParent[iParent], array_ok=True)

        #Step3b: convert elements of raw_spamTuples_dict from OrderedDicts
        # to lists not that we don't need to use them for lookups anymore.
        for s in list(raw_spamTuples_dict.keys()):
            raw_spamTuples_dict[s] = list(raw_spamTuples_dict[s].keys())


        #Step4: change lists/slices -> index arrays for user convenience
        elIndicesByParent = _collections.OrderedDict(
            [ (k, (v if isinstance(v,slice) else _np.array(v,_np.int64)) )
              for k,v in elIndicesByParent.items()] )


        ##DEBUG: SANITY CHECK
        #if len(gatestrings) > 1:
        #    for k,gstr in enumerate(gatestrings):
        #        _,outcomes_k = self.compile_gatestring(gstr)
        #        nIndices = _slct.length(elIndicesByParent[k]) if isinstance(elIndicesByParent[k], slice) \
        #                      else len(elIndicesByParent[k])
        #        assert(len(outcomes_k) == nIndices)
        #        assert(outcomes_k == outcomesByParent[k])

        #print("GateSet.compile debug:")
        #print("input = ",gatestrings)
        #print("raw_dict = ", raw_spamTuples_dict)
        #print("elIndices = ", elIndicesByParent)
        #print("outcomes = ", outcomesByParent)
        #print("total els = ",nTotElements)

        return (raw_spamTuples_dict, elIndicesByParent,
                outcomesByParent, nTotElements)


    def compile_gatestring(self, gatestring):
        """
        Compiles a single :class:`GateString`.

        Parameters
        ----------
        gatestring : GateString
            The gate string to compile

        Returns
        -------
        raw_spamTuples_dict : collections.OrderedDict
            A dictionary whose keys are raw gate sequences (containing just
            "compiled" gates, i.e. not instruments), and whose values are
            lists of (preplbl, effectlbl) tuples.  The effectlbl names a
            "compiled" effect vector; preplbl is just a prep label. Each tuple
            corresponds to a single "final element" of the computation for this
            gate string.  The ordering is important - and is why this needs to be
            an ordered dictionary - when the lists of tuples are concatenated (by
            key) the resulting tuple orderings corresponds to the final-element
            axis of an output array that is being filled (computed).

        outcomes : list
            A list of outcome labels (an outcome label is a tuple
            of POVM-effect and/or instrument-element labels), corresponding to
            the final elements.
        """
        raw_dict,_,outcomes,nEls = self.compile_gatestrings([gatestring])
        assert(len(outcomes[0]) == nEls)
        return raw_dict,outcomes[0]


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
        gatestring = _gs.GateString(gatestring) # cast to GateString
        return self._calc().product(gatestring, bScale)


    def dproduct(self, gatestring, flat=False):
        """
        Compute the derivative of a specified sequence of gate labels.

        Parameters
        ----------
        gatestring : GateString or tuple of gate labels
          The sequence of gate labels.

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
        gatestring = _gs.GateString(gatestring) # cast to GateString
        return self._calc().dproduct(gatestring, flat)


    def hproduct(self, gatestring, flat=False):
        """
        Compute the hessian of a specified sequence of gate labels.

        Parameters
        ----------
        gatestring : GateString or tuple of gate labels
          The sequence of gate labels.

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
        gatestring = _gs.GateString(gatestring) # cast to GateString
        return self._calc().hproduct(gatestring, flat)


#    def pr(self, spamLabel, gatestring, clipTo=None, bUseScaling=True):
#        """
#        Compute the probability of the given gate sequence, where initialization
#        & measurement operations are together specified by spamLabel.
#
#        Parameters
#        ----------
#        spamLabel : string
#           the label specifying the state prep and measure operations
#
#        gatestring : GateString or tuple of gate labels
#          The sequence of gate labels specifying the gate string.
#
#        clipTo : 2-tuple, optional
#          (min,max) to clip return value if not None.
#
#        bUseScaling : bool, optional
#          Whether to use a post-scaled product internally.  If False, this
#          routine will run slightly faster, but with a chance that the
#          product will overflow and the subsequent trace operation will
#          yield nan as the returned probability.
#
#        Returns
#        -------
#        float
#        """
#        return self._calc().pr(spamLabel, gatestring, clipTo, bUseScaling)
#
#
#    def dpr(self, spamLabel, gatestring,
#            returnPr=False,clipTo=None):
#        """
#        Compute the derivative of a probability generated by a gate string and
#        spam label as a 1 x M numpy array, where M is the number of gateset
#        parameters.
#
#        Parameters
#        ----------
#        spamLabel : string
#           the label specifying the state prep and measure operations
#
#        gatestring : GateString or tuple of gate labels
#          The sequence of gate labels specifying the gate string.
#
#        returnPr : bool, optional
#          when set to True, additionally return the probability itself.
#
#        clipTo : 2-tuple, optional
#           (min,max) to clip returned probability to if not None.
#           Only relevant when returnPr == True.
#
#        Returns
#        -------
#        derivative : numpy array
#            a 1 x M numpy array of derivatives of the probability w.r.t.
#            each gateset parameter (M is the length of the vectorized gateset).
#
#        probability : float
#            only returned if returnPr == True.
#        """
#        return self._calc().dpr(spamLabel, gatestring,returnPr,clipTo)
#
#
#    def hpr(self, spamLabel, gatestring,
#            returnPr=False,returnDeriv=False,clipTo=None):
#        """
#        Compute the Hessian of a probability generated by a gate string and
#        spam label as a 1 x M x M array, where M is the number of gateset
#        parameters.
#
#        Parameters
#        ----------
#        spamLabel : string
#           the label specifying the state prep and measure operations
#
#        gatestring : GateString or tuple of gate labels
#          The sequence of gate labels specifying the gate string.
#
#        returnPr : bool, optional
#          when set to True, additionally return the probability itself.
#
#        returnDeriv : bool, optional
#          when set to True, additionally return the derivative of the
#          probability.
#
#        clipTo : 2-tuple, optional
#           (min,max) to clip returned probability to if not None.
#           Only relevant when returnPr == True.
#
#        Returns
#        -------
#        hessian : numpy array
#            a 1 x M x M array, where M is the number of gateset parameters.
#            hessian[0,j,k] is the derivative of the probability w.r.t. the
#            k-th then the j-th gateset parameter.
#
#        derivative : numpy array
#            only returned if returnDeriv == True. A 1 x M numpy array of
#            derivatives of the probability w.r.t. each gateset parameter.
#
#        probability : float
#            only returned if returnPr == True.
#        """
#        return self._calc().hpr(spamLabel, gatestring,
#                                returnPr,returnDeriv,clipTo)


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
        return self._calc().probs(self.compile_gatestring(gatestring), clipTo)


    def dprobs(self, gatestring, returnPr=False,clipTo=None):
        """
        Construct a dictionary containing the probability derivatives of every
        spam label for a given gate string.

        Parameters
        ----------
        gatestring : GateString or tuple of gate labels
          The sequence of gate labels specifying the gate string.

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
        return self._calc().dprobs(self.compile_gatestring(gatestring),
                                   returnPr,clipTo)


    def hprobs(self, gatestring, returnPr=False,returnDeriv=False,clipTo=None):
        """
        Construct a dictionary containing the probability derivatives of every
        spam label for a given gate string.

        Parameters
        ----------
        gatestring : GateString or tuple of gate labels
          The sequence of gate labels specifying the gate string.

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
        return self._calc().hprobs(self.compile_gatestring(gatestring),
                                   returnPr, returnDeriv, clipTo)


    def bulk_evaltree_from_resources(self, gatestring_list, comm=None, memLimit=None,
                                     distributeMethod="default", subcalls=[],
                                     dataset=None, verbosity=0):
        """
        Create an evaluation tree based on available memory and CPUs.

        This tree can be used by other Bulk_* functions, and is it's own
        function so that for many calls to Bulk_* made with the same
        gatestring_list, only a single call to bulk_evaltree is needed.

        Parameters
        ----------
        gatestring_list : list of (tuples or GateStrings)
            Each element specifies a gate string to include in the evaluation tree.

        comm : mpi4py.MPI.Comm
            When not None, an MPI communicator for distributing computations
            across multiple processors.

        memLimit : int, optional
            A rough memory limit in bytes which is used to determine subtree
            number and size.

        distributeMethod : {"gatestrings", "deriv"}
            How to distribute calculation amongst processors (only has effect
            when comm is not None).  "gatestrings" will divide the list of
            gatestrings and thereby result in more subtrees; "deriv" will divide
            the columns of any jacobian matrices, thereby resulting in fewer
            (larger) subtrees.

        subcalls : list, optional
            A list of the names of the GateSet functions that will be called
            using the returned evaluation tree, which are necessary for
            estimating memory usage (for comparison to memLimit).  If
            memLimit is None, then there's no need to specify `subcalls`.

        dataset : DataSet, optional
            If not None, restrict what is computed to only those
            probabilities corresponding to non-zero counts (observed
            outcomes) in this data set.

        verbosity : int, optional
            How much detail to send to stdout.

        Returns
        -------
        evt : EvalTree
            The evaluation tree object, split as necesary.
        paramBlockSize1 : int or None
            The maximum size of 1st-deriv-dimension parameter blocks
            (i.e. the maximum number of parameters to compute at once
             in calls to dprobs, etc., usually specified as wrtBlockSize
             or wrtBlockSize1).
        paramBlockSize2 : int or None
            The maximum size of 2nd-deriv-dimension parameter blocks
            (i.e. the maximum number of parameters to compute at once
             in calls to hprobs, etc., usually specified as wrtBlockSize2).
        """

        # Let np = # param groups, so 1 <= np <= num_params, size of each param group = num_params/np
        # Let ng = # gate string groups == # subtrees, so 1 <= ng <= max_split_num; size of each group = size of corresponding subtree
        # With nprocs processors, split into Ng comms of ~nprocs/Ng procs each.  These comms are each assigned
        #  some number of gate string groups, where their ~nprocs/Ng processors are used to partition the np param
        #  groups. Note that 1 <= Ng <= min(ng,nprocs).
        # Notes:
        #  - making np or ng > nprocs can be useful for saving memory.  Raising np saves *Jacobian* and *Hessian*
        #     function memory without evaltree overhead, and I think will typically be preferred over raising
        #     ng which will also save Product function memory but will incur evaltree overhead.
        #  - any given CPU will be running a *single* (ng-index,np-index) pair at any given time, and so many
        #     memory estimates only depend on ng and np, not on Ng.  (The exception is when a routine *gathers*
        #     the end results from a divided computation.)
        #  - "gatestrings" distributeMethod: never distribute num_params (np == 1, Ng == nprocs always).
        #     Choose ng such that ng >= nprocs, memEstimate(ng,np=1) < memLimit, and ng % nprocs == 0 (ng % Ng == 0).
        #  - "deriv" distributeMethod: if possible, set ng=1, nprocs <= np <= num_params, Ng = 1 (np % nprocs == 0?)
        #     If memory constraints don't allow this, set np = num_params, Ng ~= nprocs/num_params (but Ng >= 1),
        #     and ng set by memEstimate and ng % Ng == 0 (so comms are kept busy)
        #
        # find ng, np, Ng such that:
        # - memEstimate(ng,np,Ng) < memLimit
        # - full cpu usage:
        #       - np*ng >= nprocs (all procs used)
        #       - ng % Ng == 0 (all subtree comms kept busy)
        #     -nice, but not essential:
        #       - num_params % np == 0 (each param group has same size)
        #       - np % (nprocs/Ng) == 0 would be nice (all procs have same num of param groups to process)

        printer = _VerbosityPrinter.build_printer(verbosity, comm)

        nprocs = 1 if comm is None else comm.Get_size()
        num_params = self.num_params()
        evt_cache = {} # cache of eval trees based on # min subtrees, to avoid re-computation
        C = 1.0/(1024.0**3)
        calc = self._calc()

        bNp2Matters = ("bulk_fill_hprobs" in subcalls) or ("bulk_hprobs_by_block" in subcalls)

        if memLimit is not None:
            if memLimit <= 0:
                raise MemoryError("Attempted evaltree generation " +
                                  "w/memlimit = %g <= 0!" % memLimit)
            printer.log("Evaltree generation (%s) w/mem limit = %.2fGB"
                        % (distributeMethod, memLimit*C))

        def memEstimate(ng,np1,np2,Ng,fastCacheSz=False,verb=0,cacheSize=None):
            """ Returns a memory estimate based on arguments """
            tm = _time.time()

            nFinalStrs = int(round(len(gatestring_list) / ng)) #may not need to be an int...

            if cacheSize is None:
                #Get cache size
                if not fastCacheSz:
                    #Slower (but more accurate way)
                    if ng not in evt_cache:
                        evt_cache[ng] = self.bulk_evaltree(
                            gatestring_list, minSubtrees=ng, numSubtreeComms=Ng,
                            dataset=dataset)
                        # FUTURE: make a _bulk_evaltree_precompiled version that takes compiled
                        # gate strings as input so don't have to recompile every time we hit this line.
                    cacheSize = max([s.cache_size() for s in evt_cache[ng][0].get_sub_trees()])
                    nFinalStrs = max([s.num_final_strings() for s in evt_cache[ng][0].get_sub_trees()])
                else:
                    #heuristic (but fast)
                    cacheSize = calc.estimate_cache_size(nFinalStrs)


            mem = calc.estimate_mem_usage(subcalls,cacheSize,ng,Ng,np1,np2,nFinalStrs)

            if verb == 1:
                if (not fastCacheSz):
                    fast_estimate = calc.estimate_mem_usage(
                        subcalls, cacheSize, ng, Ng, np1, np2, nFinalStrs)
                    fc_est_str = " (%.2fGB fc)" % (fast_estimate*C)
                else: fc_est_str = ""

                printer.log(" mem(%d subtrees, %d,%d param-grps, %d proc-grps)"
                            % (ng, np1, np2, Ng) + " in %.0fs = %.2fGB%s"
                            % (_time.time()-tm, mem*C, fc_est_str))
            elif verb == 2:
                wrtLen1 = (num_params+np1-1) // np1 # ceiling(num_params / np1)
                wrtLen2 = (num_params+np2-1) // np2 # ceiling(num_params / np2)
                nSubtreesPerProc = (ng+Ng-1) // Ng # ceiling(ng / Ng)
                printer.log(" Memory estimate = %.2fGB" % (mem*C) +
                     " (cache=%d, wrtLen1=%d, wrtLen2=%d, subsPerProc=%d)." %
                            (cacheSize, wrtLen1, wrtLen2, nSubtreesPerProc))
                #printer.log("  subcalls = %s" % str(subcalls))
                #printer.log("  cacheSize = %d" % cacheSize)
                #printer.log("  wrtLen = %d" % wrtLen)
                #printer.log("  nSubtreesPerProc = %d" % nSubtreesPerProc)

            return mem

        if distributeMethod == "default":
            distributeMethod = calc.default_distribute_method()

        if distributeMethod == "gatestrings":
            Nstrs = len(gatestring_list)
            np1 = 1; np2 = 1; Ng = min(nprocs,Nstrs)
            ng = Ng
            if memLimit is not None:
                #Increase ng in amounts of Ng (so ng % Ng == 0).  Start
                # with fast cacheSize computation then switch to slow
                while memEstimate(ng,np1,np2,Ng,False) > memLimit:
                    ng += Ng
                    if ng >= Nstrs:
                        # even "maximal" splitting (num trees == num strings)
                        # won't help - see if we can squeeze the this maximally-split tree
                        # to have zero cachesize
                        if Nstrs not in evt_cache:
                            memEstimate(Nstrs,np1,np2,Ng,verb=1)
                        if hasattr(evt_cache[Nstrs],"squeeze") and \
                           memEstimate(Nstrs,np1,np2,Ng,cacheSize=0) <= memLimit:
                            evt_cache[Nstrs].squeeze(0) #To get here, need to use higher-dim gatesets
                        else:
                            raise MemoryError("Cannot split or squeeze tree to achieve memory limit")

                mem_estimate = memEstimate(ng,np1,np2,Ng,verb=1)
                while mem_estimate > memLimit:
                    ng += Ng; next = memEstimate(ng,np1,np2,Ng,verb=1)
                    if(next >= mem_estimate): raise MemoryError("Not enough memory: splitting unproductive")
                    mem_estimate = next

                   #Note: could do these while loops smarter, e.g. binary search-like?
                   #  or assume memEstimate scales linearly in ng? E.g:
                   #     if memLimit < memEstimate:
                   #         reductionFactor = float(memEstimate) / float(memLimit)
                   #         maxTreeSize = int(nstrs / reductionFactor)
            else:
                memEstimate(ng,np1,np2,Ng) # to compute & cache final EvalTree


        elif distributeMethod == "deriv":

            def set_Ng(desired_Ng):
                """ Set Ng, the number of subTree processor groups, such
                    that Ng divides nprocs evenly or vice versa. """
                if desired_Ng >= nprocs:
                    return nprocs * int(_np.ceil(1.*desired_Ng/nprocs))
                else:
                    fctrs = sorted(_mt.prime_factors(nprocs)); i=1
                    if int(_np.ceil(desired_Ng)) in fctrs:
                        return int(_np.ceil(desired_Ng)) #we got lucky
                    while _np.product(fctrs[0:i]) < desired_Ng: i+=1
                    return _np.product(fctrs[0:i])

            ng = Ng = 1
            if bNp2Matters:
                if nprocs > num_params**2:
                    np1 = np2 = num_params
                    ng = Ng = set_Ng(nprocs / num_params**2) #Note __future__ division
                elif nprocs > num_params:
                    np1 = num_params
                    np2 = int(_np.ceil(nprocs / num_params))
                else:
                    np1 = nprocs; np2 = 1
            else:
                np2 = 1
                if nprocs > num_params:
                    np1 = num_params
                    ng = Ng = set_Ng(nprocs / num_params)
                else:
                    np1 = nprocs

            if memLimit is not None:

                ok = False
                if (not ok) and np1 < num_params:
                    #First try to decrease mem consumption by increasing np1
                    memEstimate(ng,np1,np2,Ng,verb=1) #initial estimate (to screen)
                    for n in range(np1, num_params+1, nprocs):
                        if memEstimate(ng,n,np2,Ng) < memLimit:
                            np1 = n; ok=True; break
                    else: np1 = num_params

                if (not ok) and bNp2Matters and np2 < num_params:
                    #Next try to decrease mem consumption by increasing np2
                    for n in range(np2, num_params+1):
                        if memEstimate(ng,np1,n,Ng) < memLimit:
                            np2 = n; ok=True; break
                    else: np2 = num_params

                if not ok:
                    #Finally, increase ng in amounts of Ng (so ng % Ng == 0).  Start
                    # with fast cacheSize computation then switch to slow
                    while memEstimate(ng,np1,np2,Ng,True) > memLimit: ng += Ng
                    mem_estimate = memEstimate(ng,np1,np2,Ng,verb=1)
                    while mem_estimate > memLimit:
                        ng += Ng; next = memEstimate(ng,np1,np2,Ng,verb=1)
                        if next >= mem_estimate:
                            raise MemoryError("Not enough memory: splitting unproductive")
                        mem_estimate = next
            else:
                memEstimate(ng,np1,np2,Ng) # to compute & cache final EvalTree

        elif distributeMethod == "balanced":
            # try to minimize "unbalanced" procs
            #np = gcf(num_params, nprocs)
            #ng = Ng = max(nprocs / np, 1)
            #if memLimit is not None:
            #    while memEstimate(ng,np1,np2,Ng) > memLimit: ng += Ng #so ng % Ng == 0
            raise NotImplementedError("balanced distribution still todo")

        # Retrieve final EvalTree (already computed from estimates above)
        assert (ng in evt_cache), "Tree Caching Error"
        evt,lookup,outcome_lookup = evt_cache[ng]
        evt.distribution['numSubtreeComms'] = Ng

        paramBlkSize1 = num_params / np1
        paramBlkSize2 = num_params / np2   #the *average* param block size
          # (in general *not* an integer), which ensures that the intended # of
          # param blocks is communicatd to gsCalc.py routines (taking ceiling or
          # floor can lead to inefficient MPI distribution)

        printer.log("Created evaluation tree with %d subtrees.  " % ng
                    + "Will divide %d procs into %d (subtree-processing)" % (nprocs,Ng))
        if bNp2Matters:
            printer.log(" groups of ~%d procs each, to distribute over " % (nprocs/Ng)
                        + "(%d,%d) params (taken as %d,%d param groups of ~%d,%d params)."
                        % (num_params,num_params, np1,np2, paramBlkSize1,paramBlkSize2))
        else:
            printer.log(" groups of ~%d procs each, to distribute over " % (nprocs/Ng)
                        + "%d params (taken as %d param groups of ~%d params)."
                        % (num_params, np1, paramBlkSize1))

        if memLimit is not None:
            memEstimate(ng,np1,np2,Ng,False,verb=2) #print mem estimate details

        if (comm is None or comm.Get_rank() == 0) and evt.is_split():
            if printer.verbosity >= 2: evt.print_analysis()

        if np1 == 1: # (paramBlkSize == num_params)
            paramBlkSize1 = None # == all parameters, and may speed logic in dprobs, etc.
        else:
            if comm is not None:
                blkSizeTest = comm.bcast(paramBlkSize1,root=0)
                assert(abs(blkSizeTest-paramBlkSize1) < 1e-3)
                  #all procs should have *same* paramBlkSize1

        if np2 == 1: # (paramBlkSize == num_params)
            paramBlkSize2 = None # == all parameters, and may speed logic in hprobs, etc.
        else:
            if comm is not None:
                blkSizeTest = comm.bcast(paramBlkSize2,root=0)
                assert(abs(blkSizeTest-paramBlkSize2) < 1e-3)
                  #all procs should have *same* paramBlkSize2

        return evt, paramBlkSize1, paramBlkSize2, lookup, outcome_lookup



    def bulk_evaltree(self, gatestring_list, minSubtrees=None, maxTreeSize=None,
                      numSubtreeComms=1, dataset=None, verbosity=0):
        """
        Create an evaluation tree for all the gate strings in gatestring_list.

        This tree can be used by other Bulk_* functions, and is it's own
        function so that for many calls to Bulk_* made with the same
        gatestring_list, only a single call to bulk_evaltree is needed.

        Parameters
        ----------
        gatestring_list : list of (tuples or GateStrings)
            Each element specifies a gate string to include in the evaluation tree.

        minSubtrees : int (optional)
            The minimum number of subtrees the resulting EvalTree must have.

        maxTreeSize : int (optional)
            The maximum size allowed for the single un-split tree or any of
            its subtrees.

        numSubtreeComms : int, optional
            The number of processor groups (communicators)
            to divide the subtrees of the EvalTree among
            when calling its `distribute` method.

        dataset : DataSet, optional
            If not None, restrict what is computed to only those
            probabilities corresponding to non-zero counts (observed
            outcomes) in this data set.

        verbosity : int, optional
            How much detail to send to stdout.

        Returns
        -------
        evt : EvalTree
            An evaluation tree object.

        elIndices : collections.OrderedDict
            A dictionary whose keys are integer indices into `gatestring_list` and
            whose values are slices and/or integer-arrays into the space/axis of
            final elements returned by the 'bulk fill' routines.  Thus, to get the
            final elements corresponding to `gatestrings[i]`, use
            `filledArray[ elIndices[i] ]`.

        outcomes : collections.OrderedDict
            A dictionary whose keys are integer indices into `gatestring_list` and
            whose values are lists of outcome labels (an outcome label is a tuple
            of POVM-effect and/or instrument-element labels).  Thus, to obtain
            what outcomes the i-th gate strings's final elements
            (`filledArray[ elIndices[i] ]`)  correspond to, use `outcomes[i]`.
        """
        tm = _time.time()
        printer = _VerbosityPrinter.build_printer(verbosity)

        compiled_gatestrings, elIndices, outcomes, nEls = \
                            self.compile_gatestrings(gatestring_list, dataset)

        evalTree = self._calc().construct_evaltree()
        evalTree.initialize(compiled_gatestrings, numSubtreeComms)

        printer.log("bulk_evaltree: created initial tree (%d strs) in %.0fs" %
                    (len(gatestring_list),_time.time()-tm)); tm = _time.time()

        if maxTreeSize is not None:
            elIndices = evalTree.split(elIndices, maxTreeSize, None, printer) # won't split if unnecessary

        if minSubtrees is not None:
            if not evalTree.is_split() or len(evalTree.get_sub_trees()) < minSubtrees:
                evalTree.original_index_lookup = None # reset this so we can re-split TODO: cleaner
                elIndices = evalTree.split(elIndices, None, minSubtrees, printer)
                if maxTreeSize is not None and \
                        any([ len(sub)>maxTreeSize for sub in evalTree.get_sub_trees()]):
                    _warnings.warn("Could not create a tree with minSubtrees=%d" % minSubtrees
                                   + " and maxTreeSize=%d" % maxTreeSize)
                    evalTree.original_index_lookup = None # reset this so we can re-split TODO: cleaner
                    elIndices = evalTree.split(elIndices, maxTreeSize, None) # fall back to split for max size

        if maxTreeSize is not None or minSubtrees is not None:
            printer.log("bulk_evaltree: split tree (%d subtrees) in %.0fs"
                        % (len(evalTree.get_sub_trees()),_time.time()-tm))

        assert(evalTree.num_final_elements() == nEls)
        return evalTree, elIndices, outcomes


    def bulk_product(self, evalTree, bScale=False, comm=None):
        """
        Compute the products of many gate strings at once.

        Parameters
        ----------
        evalTree : EvalTree
           given by a prior call to bulk_evaltree.  Specifies the gate strings
           to compute the bulk operation on.

        bScale : bool, optional
           When True, return a scaling factor (see below).

        comm : mpi4py.MPI.Comm, optional
           When not None, an MPI communicator for distributing the computation
           across multiple processors.  This is done over gate strings when a
           *split* evalTree is given, otherwise no parallelization is performed.


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
        return self._calc().bulk_product(evalTree, bScale, comm)


    def bulk_dproduct(self, evalTree, flat=False, bReturnProds=False,
                      bScale=False, comm=None):
        """
        Compute the derivative of many gate strings at once.

        Parameters
        ----------
        evalTree : EvalTree
           given by a prior call to bulk_evaltree.  Specifies the gate strings
           to compute the bulk operation on.

        flat : bool, optional
          Affects the shape of the returned derivative array (see below).

        bReturnProds : bool, optional
          when set to True, additionally return the products.

        bScale : bool, optional
          When True, return a scaling factor (see below).

        comm : mpi4py.MPI.Comm, optional
           When not None, an MPI communicator for distributing the computation
           across multiple processors.  Distribution is first done over the set
           of parameters being differentiated with respect to.  If there are
           more processors than gateset parameters, distribution over a split
           evalTree (if given) is possible.


        Returns
        -------
        derivs : numpy array

          * if `flat` is ``False``, an array of shape S x M x G x G, where:

            - S = len(gatestring_list)
            - M = the length of the vectorized gateset
            - G = the linear dimension of a gate matrix (G x G gate matrices)

            and ``derivs[i,j,k,l]`` holds the derivative of the (k,l)-th entry
            of the i-th gate string product with respect to the j-th gateset
            parameter.

          * if `flat` is ``True``, an array of shape S*N x M where:

            - N = the number of entries in a single flattened gate (ordering
              same as numpy.flatten),
            - S,M = as above,

            and ``deriv[i,j]`` holds the derivative of the ``(i % G^2)``-th
            entry of the ``(i / G^2)``-th flattened gate string product  with
            respect to the j-th gateset parameter.

        products : numpy array
          Only returned when `bReturnProds` is ``True``.  An array of shape
          S x G x G; ``products[i]`` is the i-th gate string product.

        scaleVals : numpy array
          Only returned when `bScale` is ``True``.  An array of shape S such
          that ``scaleVals[i]`` contains the multiplicative scaling needed for
          the derivatives and/or products for the i-th gate string.
        """
        return self._calc().bulk_dproduct(evalTree, flat, bReturnProds,
                                          bScale, comm)


    def bulk_hproduct(self, evalTree, flat=False, bReturnDProdsAndProds=False,
                      bScale=False, comm=None):
        """
        Return the Hessian of many gate string products at once.

        Parameters
        ----------
        evalTree : EvalTree
           given by a prior call to bulk_evaltree.  Specifies the gate strings
           to compute the bulk operation on.

        flat : bool, optional
          Affects the shape of the returned derivative array (see below).

        bReturnDProdsAndProds : bool, optional
          when set to True, additionally return the probabilities and
          their derivatives.

        bScale : bool, optional
          When True, return a scaling factor (see below).

        comm : mpi4py.MPI.Comm, optional
           When not None, an MPI communicator for distributing the computation
           across multiple processors.  Distribution is first done over the
           set of parameters being differentiated with respect to when the
           *second* derivative is taken.  If there are more processors than
           gateset parameters, distribution over a split evalTree (if given)
           is possible.


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
        ret = self._calc().bulk_hproduct(
            evalTree, flat, bReturnDProdsAndProds, bScale, comm)
        if bReturnDProdsAndProds:
            return ret[0:2] + ret[3:] #remove ret[2] == deriv wrt filter2,
                         # which isn't an input param for GateSet version
        else: return ret


#Deprecated: use bulk_probs, bulk_dprobs, and bulk_hprobs
#    def bulk_pr(self, spamLabel, evalTree, clipTo=None, check=False, comm=None):
#        """
#        Compute the probabilities of the gate sequences given by evalTree,
#        where initialization & measurement operations are always the same
#        and are together specified by spamLabel.
#
#        Parameters
#        ----------
#        spamLabel : string
#           the label specifying the state prep and measure operations
#
#        evalTree : EvalTree
#           given by a prior call to bulk_evaltree.  Specifies the gate strings
#           to compute the bulk operation on.
#
#        clipTo : 2-tuple, optional
#           (min,max) to clip return value if not None.
#
#        check : boolean, optional
#          If True, perform extra checks within code to verify correctness,
#          generating warnings when checks fail.  Used for testing, and runs
#          much slower when True.
#
#        comm : mpi4py.MPI.Comm, optional
#           When not None, an MPI communicator for distributing the computation
#           across multiple processors.  Distribution is performed over
#           subtrees of evalTree (if it is split).
#
#
#        Returns
#        -------
#        numpy array
#          An array of length equal to the number of gate strings containing
#          the (float) probabilities.
#        """
#        return self._calc().bulk_pr(spamLabel, evalTree, clipTo, check, comm)
#
#
#    def bulk_dpr(self, spamLabel, evalTree,
#                 returnPr=False,clipTo=None,check=False,
#                 comm=None,wrtBlockSize=None):
#
#        """
#        Compute the derivatives of the probabilities generated by a each gate
#        sequence given by evalTree, where initialization
#        & measurement operations are always the same and are
#        together specified by spamLabel.
#
#        Parameters
#        ----------
#        spamLabel : string
#           the label specifying the state prep and measure operations
#
#        evalTree : EvalTree
#           given by a prior call to bulk_evaltree.  Specifies the gate strings
#           to compute the bulk operation on.
#
#        returnPr : bool, optional
#          when set to True, additionally return the probabilities.
#
#        clipTo : 2-tuple, optional
#           (min,max) to clip returned probability to if not None.
#           Only relevant when returnPr == True.
#
#        check : boolean, optional
#          If True, perform extra checks within code to verify correctness,
#          generating warnings when checks fail.  Used for testing, and runs
#          much slower when True.
#
#        comm : mpi4py.MPI.Comm, optional
#           When not None, an MPI communicator for distributing the computation
#           across multiple processors.  Distribution is first performed over
#           subtrees of evalTree (if it is split), and then over blocks (subsets)
#           of the parameters being differentiated with respect to (see
#           wrtBlockSize).
#
#        wrtBlockSize : int or float, optional
#          The maximum average number of derivative columns to compute *products*
#          for simultaneously.  None means compute all columns at once.
#          The minimum of wrtBlockSize and the size that makes maximal
#          use of available processors is used as the final block size. Use
#          this argument to reduce amount of intermediate memory required.
#
#
#        Returns
#        -------
#        dprobs : numpy array
#            An array of shape S x M, where
#
#            - S == the number of gate strings
#            - M == the length of the vectorized gateset
#
#            and dprobs[i,j] holds the derivative of the i-th probability w.r.t.
#            the j-th gateset parameter.
#
#        probs : numpy array
#            Only returned when returnPr == True. An array of shape S containing
#            the probabilities of each gate string.
#        """
#        return self._calc().bulk_dpr(spamLabel, evalTree, returnPr,clipTo,
#                                     check, comm, None, wrtBlockSize)
#
#
#    def bulk_hpr(self, spamLabel, evalTree,
#                 returnPr=False,returnDeriv=False,
#                 clipTo=None,check=False,comm=None,
#                 wrtBlockSize1=None, wrtBlockSize2=None):
#
#        """
#        Compute the 2nd derivatives of the probabilities generated by a each gate
#        sequence given by evalTree, where initialization & measurement
#        operations are always the same and are together specified by spamLabel.
#
#        Parameters
#        ----------
#        spamLabel : string
#          the label specifying the state prep and measure operations
#
#        evalTree : EvalTree
#          given by a prior call to bulk_evaltree.  Specifies the gate strings
#          to compute the bulk operation on.
#
#        returnPr : bool, optional
#          when set to True, additionally return the probabilities.
#
#        returnDeriv : bool, optional
#          when set to True, additionally return the probability derivatives.
#
#        clipTo : 2-tuple, optional
#          (min,max) to clip returned probability to if not None.
#          Only relevant when returnPr == True.
#
#        check : boolean, optional
#          If True, perform extra checks within code to verify correctness,
#          generating warnings when checks fail.  Used for testing, and runs
#          much slower when True.
#
#        comm : mpi4py.MPI.Comm, optional
#           When not None, an MPI communicator for distributing the computation
#           across multiple processors.
#
#        wrtBlockSize2, wrtBlockSize2 : int or float, optional
#          The maximum number of 1st (row) and 2nd (col) derivatives to compute
#          *products* for simultaneously.  None means compute all requested
#          rows or columns at once.  The  minimum of wrtBlockSize and the size
#          that makes maximal use of available processors is used as the final
#          block size.  These arguments must be None if the corresponding
#          wrtFilter is not None.  Set this to non-None to reduce amount of
#          intermediate memory required.
#
#
#        Returns
#        -------
#        hessians : numpy array
#            a S x M x M array, where
#
#            - S == the number of gate strings
#            - M == the length of the vectorized gateset
#
#            and hessians[i,j,k] is the derivative of the i-th probability
#            w.r.t. the k-th then the j-th gateset parameter.
#
#        derivs : numpy array
#            only returned if returnDeriv == True. A S x M array where
#            derivs[i,j] holds the derivative of the i-th probability
#            w.r.t. the j-th gateset parameter.
#
#        probabilities : numpy array
#            only returned if returnPr == True.  A length-S array
#            containing the probabilities for each gate string.
#        """
#        return self._calc().bulk_hpr(spamLabel, evalTree, returnPr,returnDeriv,
#                                    clipTo, check, comm, None, None,
#                                     wrtBlockSize1, wrtBlockSize2)


    def bulk_probs(self, gatestring_list, clipTo=None, check=False,
                   comm=None, memLimit=None, dataset=None):
        """
        Construct a dictionary containing the probabilities
        for an entire list of gate sequences.

        Parameters
        ----------
        gatestring_list : list of (tuples or GateStrings)
          Each element specifies a gate string to compute quantities for.

        clipTo : 2-tuple, optional
           (min,max) to clip return value if not None.

        check : boolean, optional
          If True, perform extra checks within code to verify correctness,
          generating warnings when checks fail.  Used for testing, and runs
          much slower when True.

        comm : mpi4py.MPI.Comm, optional
           When not None, an MPI communicator for distributing the computation
           across multiple processors.  Distribution is performed over
           subtrees of evalTree (if it is split).

        memLimit : int, optional
            A rough memory limit in bytes which is used to determine processor
            allocation.

        dataset : DataSet, optional
            If not None, restrict what is computed to only those
            probabilities corresponding to non-zero counts (observed
            outcomes) in this data set.


        Returns
        -------
        probs : dictionary
            A dictionary such that `probs[gstr]` is an ordered dictionary of
            `(outcome, p)` tuples, where `outcome` is a tuple of labels
            and `p` is the corresponding probability.
        """
        gatestring_list = [ _gs.GateString(gs) for gs in gatestring_list]  # cast to GateStrings
        #OLD: evalTree, elIndices, outcomes = self.bulk_evaltree(gatestring_list, dataset=dataset)
        evalTree, _, _, elIndices, outcomes = self.bulk_evaltree_from_resources(
            gatestring_list, comm, memLimit, subcalls=['bulk_fill_probs'],
            dataset=dataset, verbosity=0) # FUTURE (maybe make verbosity into an arg?)
        return self._calc().bulk_probs(gatestring_list, evalTree, elIndices,
                                       outcomes, clipTo, check, comm)


    def bulk_dprobs(self, gatestring_list, returnPr=False,clipTo=None,
                    check=False,comm=None,wrtBlockSize=None,dataset=None):

        """
        Construct a dictionary containing the probability-derivatives
        for an entire list of gate sequences.

        Parameters
        ----------
        gatestring_list : list of (tuples or GateStrings)
          Each element specifies a gate string to compute quantities for.

        returnPr : bool, optional
          when set to True, additionally return the probabilities.

        clipTo : 2-tuple, optional
           (min,max) to clip returned probability to if not None.
           Only relevant when returnPr == True.

        check : boolean, optional
          If True, perform extra checks within code to verify correctness,
          generating warnings when checks fail.  Used for testing, and runs
          much slower when True.

        comm : mpi4py.MPI.Comm, optional
           When not None, an MPI communicator for distributing the computation
           across multiple processors.  Distribution is first performed over
           subtrees of evalTree (if it is split), and then over blocks (subsets)
           of the parameters being differentiated with respect to (see
           wrtBlockSize).

        wrtBlockSize : int or float, optional
          The maximum average number of derivative columns to compute *products*
          for simultaneously.  None means compute all columns at once.
          The minimum of wrtBlockSize and the size that makes maximal
          use of available processors is used as the final block size. Use
          this argument to reduce amount of intermediate memory required.

        dataset : DataSet, optional
            If not None, restrict what is computed to only those
            probabilities corresponding to non-zero counts (observed
            outcomes) in this data set.


        Returns
        -------
        dprobs : dictionary
            A dictionary such that `probs[gstr]` is an ordered dictionary of
            `(outcome, dp, p)` tuples, where `outcome` is a tuple of labels,
            `p` is the corresponding probability, and `dp` is an array containing
            the derivative of `p` with respect to each parameter.  If `returnPr`
            if False, then `p` is not included in the tuples (so they're just
            `(outcome, dp)`).
        """
        gatestring_list = [ _gs.GateString(gs) for gs in gatestring_list]  # cast to GateStrings
        evalTree, elIndices, outcomes = self.bulk_evaltree(gatestring_list, dataset=dataset)
        return self._calc().bulk_dprobs(gatestring_list, evalTree, elIndices,
                                        outcomes, returnPr,clipTo,
                                        check, comm, None, wrtBlockSize)


    def bulk_hprobs(self, gatestring_list, returnPr=False,returnDeriv=False,
                    clipTo=None, check=False, comm=None,
                    wrtBlockSize1=None, wrtBlockSize2=None, dataset=None):

        """
        Construct a dictionary containing the probability-Hessians
        for an entire list of gate sequences.

        Parameters
        ----------
        gatestring_list : list of (tuples or GateStrings)
          Each element specifies a gate string to compute quantities for.

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

        comm : mpi4py.MPI.Comm, optional
           When not None, an MPI communicator for distributing the computation
           across multiple processors.

        wrtBlockSize2, wrtBlockSize2 : int or float, optional
          The maximum number of 1st (row) and 2nd (col) derivatives to compute
          *products* for simultaneously.  None means compute all requested
          rows or columns at once.  The  minimum of wrtBlockSize and the size
          that makes maximal use of available processors is used as the final
          block size.  These arguments must be None if the corresponding
          wrtFilter is not None.  Set this to non-None to reduce amount of
          intermediate memory required.

        dataset : DataSet, optional
            If not None, restrict what is computed to only those
            probabilities corresponding to non-zero counts (observed
            outcomes) in this data set.


        Returns
        -------
        hprobs : dictionary
            A dictionary such that `probs[gstr]` is an ordered dictionary of
            `(outcome, hp, dp, p)` tuples, where `outcome` is a tuple of labels,
            `p` is the corresponding probability, `dp` is a 1D array containing
            the derivative of `p` with respect to each parameter, and `hp` is a
            2D array containing the Hessian of `p` with respect to each parameter.
            If `returnPr` if False, then `p` is not included in the tuples.
            If `returnDeriv` if False, then `dp` is not included in the tuples.
        """
        gatestring_list = [ _gs.GateString(gs) for gs in gatestring_list]  # cast to GateStrings
        evalTree, elIndices, outcomes = self.bulk_evaltree(gatestring_list, dataset=dataset)
        return self._calc().bulk_hprobs(gatestring_list, evalTree, elIndices,
                                        outcomes, returnPr, returnDeriv,
                                        clipTo, check, comm, None, None,
                                        wrtBlockSize1, wrtBlockSize2)


    def bulk_fill_probs(self, mxToFill, evalTree, clipTo=None, check=False, comm=None):
        """
        Compute the outcome probabilities for an entire tree of gate strings.

        This routine fills a 1D array, `mxToFill` with the probabilities
        corresponding to the *compiled* gate strings found in an evaluation
        tree, `evalTree`.  An initial list of (general) :class:`GateString`
        objects is *compiled* into a lists of gate-only sequences along with
        a mapping of final elements (i.e. probabilities) to gate-only sequence
        and prep/effect pairs.  The evaluation tree organizes how to efficiently
        compute the gate-only sequences.  This routine fills in `mxToFill`, which
        must have length equal to the number of final elements (this can be
        obtained by `evalTree.num_final_elements()`.  To interpret which elements
        correspond to which strings and outcomes, you'll need the mappings
        generated when the original list of `GateStrings` was compiled.

        Parameters
        ----------
        mxToFill : numpy ndarray
          an already-allocated 1D numpy array of length equal to the
          total number of computed elements (i.e. evalTree.num_final_elements())

        evalTree : EvalTree
           given by a prior call to bulk_evaltree.  Specifies the *compiled* gate
           strings to compute the bulk operation on.

        clipTo : 2-tuple, optional
           (min,max) to clip return value if not None.

        check : boolean, optional
          If True, perform extra checks within code to verify correctness,
          generating warnings when checks fail.  Used for testing, and runs
          much slower when True.

        comm : mpi4py.MPI.Comm, optional
           When not None, an MPI communicator for distributing the computation
           across multiple processors.  Distribution is performed over
           subtrees of evalTree (if it is split).


        Returns
        -------
        None
        """
        return self._calc().bulk_fill_probs(mxToFill,
                                            evalTree, clipTo, check, comm)


    def bulk_fill_dprobs(self, mxToFill, evalTree, prMxToFill=None,clipTo=None,
                         check=False,comm=None, wrtBlockSize=None,
                         profiler=None, gatherMemLimit=None):
        """
        Compute the outcome probability-derivatives for an entire tree of gate
        strings.

        Similar to `bulk_fill_probs(...)`, but fills a 2D array with
        probability-derivatives for each "final element" of `evalTree`.

        Parameters
        ----------
        mxToFill : numpy ndarray
          an already-allocated ExM numpy array where E is the total number of
          computed elements (i.e. evalTree.num_final_elements()) and M is the
          number of gate set parameters.

        evalTree : EvalTree
           given by a prior call to bulk_evaltree.  Specifies the *compiled* gate
           strings to compute the bulk operation on.

        prMxToFill : numpy array, optional
          when not None, an already-allocated length-E numpy array that is filled
          with probabilities, just like in bulk_fill_probs(...).

        clipTo : 2-tuple, optional
           (min,max) to clip return value if not None.

        check : boolean, optional
          If True, perform extra checks within code to verify correctness,
          generating warnings when checks fail.  Used for testing, and runs
          much slower when True.

        comm : mpi4py.MPI.Comm, optional
           When not None, an MPI communicator for distributing the computation
           across multiple processors.  Distribution is first performed over
           subtrees of evalTree (if it is split), and then over blocks (subsets)
           of the parameters being differentiated with respect to (see
           wrtBlockSize).

        wrtBlockSize : int or float, optional
          The maximum average number of derivative columns to compute *products*
          for simultaneously.  None means compute all columns at once.
          The minimum of wrtBlockSize and the size that makes maximal
          use of available processors is used as the final block size. Use
          this argument to reduce amount of intermediate memory required.

        profiler : Profiler, optional
          A profiler object used for to track timing and memory usage.

        gatherMemLimit : int, optional
          A memory limit in bytes to impose upon the "gather" operations
          performed as a part of MPI processor syncronization.

        Returns
        -------
        None
        """
        return self._calc().bulk_fill_dprobs(mxToFill,
                                             evalTree, prMxToFill, clipTo,
                                             check, comm, None, wrtBlockSize,
                                             profiler, gatherMemLimit)


    def bulk_fill_hprobs(self, mxToFill, evalTree=None,
                         prMxToFill=None, derivMxToFill=None,
                         clipTo=None, check=False, comm=None,
                         wrtBlockSize1=None, wrtBlockSize2=None,
                         gatherMemLimit=None):

        """
        Compute the outcome probability-Hessians for an entire tree of gate
        strings.

        Similar to `bulk_fill_probs(...)`, but fills a 3D array with
        probability-Hessians for each "final element" of `evalTree`.

        Parameters
        ----------
        mxToFill : numpy ndarray
          an already-allocated ExMxM numpy array where E is the total number of
          computed elements (i.e. evalTree.num_final_elements()) and M1 & M2 are
          the number of selected gate-set parameters (by wrtFilter1 and wrtFilter2).

        evalTree : EvalTree
           given by a prior call to bulk_evaltree.  Specifies the *compiled* gate
           strings to compute the bulk operation on.

        prMxToFill : numpy array, optional
          when not None, an already-allocated length-E numpy array that is filled
          with probabilities, just like in bulk_fill_probs(...).

        derivMxToFill1, derivMxToFill2 : numpy array, optional
          when not None, an already-allocated ExM numpy array that is filled
          with probability derivatives, similar to bulk_fill_dprobs(...), but
          where M is the number of gateset parameters selected for the 1st and 2nd
          differentiation, respectively (i.e. by wrtFilter1 and wrtFilter2).

        clipTo : 2-tuple, optional
           (min,max) to clip return value if not None.

        check : boolean, optional
          If True, perform extra checks within code to verify correctness,
          generating warnings when checks fail.  Used for testing, and runs
          much slower when True.

        comm : mpi4py.MPI.Comm, optional
           When not None, an MPI communicator for distributing the computation
           across multiple processors.  Distribution is first performed over
           subtrees of evalTree (if it is split), and then over blocks (subsets)
           of the parameters being differentiated with respect to (see
           wrtBlockSize).

        wrtBlockSize2, wrtBlockSize2 : int or float, optional
          The maximum number of 1st (row) and 2nd (col) derivatives to compute
          *products* for simultaneously.  None means compute all requested
          rows or columns at once.  The  minimum of wrtBlockSize and the size
          that makes maximal use of available processors is used as the final
          block size.  These arguments must be None if the corresponding
          wrtFilter is not None.  Set this to non-None to reduce amount of
          intermediate memory required.

        profiler : Profiler, optional
          A profiler object used for to track timing and memory usage.

        gatherMemLimit : int, optional
          A memory limit in bytes to impose upon the "gather" operations
          performed as a part of MPI processor syncronization.

        Returns
        -------
        None
        """
        return self._calc().bulk_fill_hprobs(mxToFill,
                                     evalTree, prMxToFill, derivMxToFill, None,
                                     clipTo, check, comm, None, None,
                                     wrtBlockSize1,wrtBlockSize2,gatherMemLimit)


    def bulk_hprobs_by_block(self, evalTree, wrtSlicesList,
                              bReturnDProbs12=False, comm=None):
        """
        Constructs a generator that computes the 2nd derivatives of the
        probabilities generated by a each gate sequence given by evalTree
        column-by-column.

        This routine can be useful when memory constraints make constructing
        the entire Hessian at once impractical, and one is able to compute
        reduce results from a single column of the Hessian at a time.  For
        example, the Hessian of a function of many gate sequence probabilities
        can often be computed column-by-column from the using the columns of
        the gate sequences.


        Parameters
        ----------
        spam_label_rows : dictionary
          a dictionary with keys == spam labels and values which
          are integer row indices into mxToFill, specifying the
          correspondence between rows of mxToFill and spam labels.

        evalTree : EvalTree
           given by a prior call to bulk_evaltree.  Specifies the gate strings
           to compute the bulk operation on.  This tree *cannot* be split.

        wrtSlicesList : list
            A list of `(rowSlice,colSlice)` 2-tuples, each of which specify
            a "block" of the Hessian to compute.  Iterating over the output
            of this function iterates over these computed blocks, in the order
            given by `wrtSlicesList`.  `rowSlice` and `colSlice` must by Python
            `slice` objects.

        bReturnDProbs12 : boolean, optional
           If true, the generator computes a 2-tuple: (hessian_col, d12_col),
           where d12_col is a column of the matrix d12 defined by:
           d12[iSpamLabel,iGateStr,p1,p2] = dP/d(p1)*dP/d(p2) where P is is
           the probability generated by the sequence and spam label indexed
           by iGateStr and iSpamLabel.  d12 has the same dimensions as the
           Hessian, and turns out to be useful when computing the Hessian
           of functions of the probabilities.

        comm : mpi4py.MPI.Comm, optional
           When not None, an MPI communicator for distributing the computation
           across multiple processors.  Distribution is performed as in
           bulk_product, bulk_dproduct, and bulk_hproduct.


        Returns
        -------
        block_generator
          A generator which, when iterated, yields the 3-tuple
          `(rowSlice, colSlice, hprobs)` or `(rowSlice, colSlice, dprobs12)`
          (the latter if `bReturnDProbs12 == True`).  `rowSlice` and `colSlice`
          are slices directly from `wrtSlicesList`. `hprobs` and `dprobs12` are
          arrays of shape K x S x B x B', where:

          - K is the length of spam_label_rows,
          - S is the number of gate strings (i.e. evalTree.num_final_strings()),
          - B is the number of parameter rows (the length of rowSlice)
          - B' is the number of parameter columns (the length of colSlice)

          If `mx` and `dp` the outputs of :func:`bulk_fill_hprobs`
          (i.e. args `mxToFill` and `derivMxToFill`), then:

          - `hprobs == mx[:,:,rowSlice,colSlice]`
          - `dprobs12 == dp[:,:,rowSlice,None] * dp[:,:,None,colSlice]`
        """
        return self._calc().bulk_hprobs_by_block(
             evalTree, wrtSlicesList,
            bReturnDProbs12, comm)


    def frobeniusdist(self, otherGateSet, transformMx=None,
                      itemWeights=None, normalize=True):
        """
        Compute the weighted frobenius norm of the difference between this
        gateset and otherGateSet.  Differences in each corresponding gate
        matrix and spam vector element are squared, weighted (using
        `itemWeights` as applicable), then summed.  The value returned is the
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

        itemWeights : dict, optional
           Dictionary of weighting factors for individual gates and spam
           operators. Weights are applied multiplicatively to the squared
           differences, i.e., (*before* the final square root is taken).  Keys
           can be gate, state preparation, POVM effect, or spam labels, as well
           as the two special labels `"gates"` and `"spam"` which apply to all
           of the gate or SPAM elements, respectively (but are overridden by
           specific element values).  Values are floating point numbers.
           By default, all weights are 1.0.

        normalize : bool, optional
           if True (the default), the sum of weighted squared-differences
           is divided by the weighted number of differences before the
           final square root is taken.  If False, the division is not performed.

        Returns
        -------
        float
        """
        return self._calc().frobeniusdist(otherGateSet._calc(), transformMx,
                                          itemWeights, normalize)

    def residuals(self, otherGateSet, transformMx=None, itemWeights=None):
        """
        Compute the weighted residuals between two gate sets (the differences
        in corresponding gate matrix and spam vector elements).

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

        itemWeights : dict, optional
           Dictionary of weighting factors for individual gates and spam
           operators. Weights applied such that they act multiplicatively on
           the *squared* differences, so that the residuals themselves are
           scaled by the square roots of these weights.  Keys can be gate, state
           preparation, POVM effect, or spam labels, as well as the two special
           labels `"gates"` and `"spam"` which apply to all of the gate or SPAM
           elements, respectively (but are overridden by specific element
           values).  Values are floating point numbers.  By default, all weights
           are 1.0.

        Returns
        -------
        residuals : numpy.ndarray
            A 1D array of residuals (differences w.r.t. other)
        nSummands : int
            The (weighted) number of elements accounted for by the residuals.
        """
        return self._calc().residuals(otherGateSet._calc(), transformMx, itemWeights)


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
        return self._calc().jtracedist(otherGateSet._calc(), transformMx)


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
        return self._calc().diamonddist(otherGateSet._calc(), transformMx)


    def tpdist(self):
        """
        Compute the "distance" between this gateset and the space of
        trace-preserving (TP) maps, defined as the sqrt of the sum-of-squared
        deviations among the first row of all gate matrices and the
        first element of all state preparations.
        """
        penalty = 0.0
        for gateMx in list(self.gates.values()):
            penalty += abs(gateMx[0,0] - 1.0)**2
            for k in range(1,gateMx.shape[1]):
                penalty += abs(gateMx[0,k])**2

        gate_dim = self.get_dimension()
        firstEl = 1.0 / gate_dim**0.25
        for rhoVec in list(self.preps.values()):
            penalty += abs(rhoVec[0,0] - firstEl)**2

        return _np.sqrt(penalty)


    def strdiff(self, otherGateSet):
        """
        Return a string describing
        the frobenius distances between
        each corresponding gate, state prep,
        and POVM effect.

        Parameters
        ----------
        otherGateSet : GateSet
            the other gate set to difference against.

        Returns
        -------
        str
        """
        s =  "Gateset Difference:\n"
        s += " Preps:\n"
        for lbl in self.preps:
            s += "  %s = %g\n" % \
                (lbl, _np.linalg.norm(self.preps[lbl]-otherGateSet.preps[lbl]))

        s += " POVMs:\n"
        for povm_lbl,povm in self.povms.items():
            s += "  %s: " % povm_lbl
            for lbl in povm:
                s += "    %s = %g\n" % \
                     (lbl, _np.linalg.norm(povm[lbl]-otherGateSet.povms[povm_lbl][lbl]))

        s += " Gates:\n"
        for lbl in self.gates:
            s += "  %s = %g\n" % \
                (lbl, _np.linalg.norm(self.gates[lbl]-otherGateSet.gates[lbl]))

        if len(self.instruments) > 0:
            s += " Instruments:\n"
            for inst_lbl,inst in self.instruments.items():
                s += "  %s: " % inst_lbl
                for lbl in inst:
                    s += "    %s = %g\n" % \
                         (lbl, _np.linalg.norm(inst[lbl]-otherGateSet.instruments[inst_lbl][lbl]))

        return s


    def copy(self):
        """
        Copy this gateset

        Returns
        -------
        GateSet
            a (deep) copy of this gateset.
        """
        if GateSet._pcheck: self._check_paramvec()

        newGateset = GateSet()
        newGateset._evotype = self._evotype
        newGateset.preps = self.preps.copy(newGateset)
        newGateset.povms = self.povms.copy(newGateset)
        #newGateset.effects = self.effects.copy(newGateset)
        newGateset.gates = self.gates.copy(newGateset)
        newGateset.instruments = self.instruments.copy(newGateset)
        newGateset._paramvec = self._paramvec.copy()
        #newGateset._rebuild_paramvec() # unnecessary, as copy will change parent and copy gpindices

        if GateSet._pcheck: newGateset._check_paramvec()

        #newGateset.spamdefs = self.spamdefs.copy()
        newGateset._dim = self._dim
        newGateset._default_gauge_group = self._default_gauge_group #Note: SHALLOW copy

        if not hasattr(self,"_calcClass"): #for backward compatibility
            self._calcClass = _gatematrixcalc.GateMatrixCalc
        newGateset._calcClass = self._calcClass

        if not hasattr(self,"_sim_type"): #for backward compatibility
            self._sim_type = "dmmatrix"
            self._sim_args = []
        newGateset._sim_type = self._sim_type
        newGateset._sim_args = self._sim_args

        if not hasattr(self,"basis") and hasattr(self,'_basisNameAndDim'): #for backward compatibility
            self.basis = _Basis(self._basisNameAndDim[0],self._basisNameAndDim[1])
        newGateset.basis = self.basis.copy()
        newGateset.stateSpaceLabels = self.stateSpaceLabels # TODO: copy()?
        if GateSet._pcheck: self._check_paramvec()

        return newGateset

    def __str__(self):
        s = ""
        for lbl,vec in self.preps.items():
            s += "%s = " % lbl + str(vec) + "\n"
        s += "\n"
        for lbl,povm in self.povms.items():
            s += "%s = " % lbl + str(povm) + "\n"
        s += "\n"
        for lbl,gate in self.gates.items():
            s += "%s = \n" % lbl + str(gate) + "\n\n"
        for lbl,inst in self.instruments.items():
            s += "%s = " % lbl + str(inst) + "\n"
        s += "\n"

        return s


    def iter_objs(self):
        for lbl,obj in _itertools.chain(self.preps.items(),
                                        self.povms.items(),
                                        self.gates.items(),
                                        self.instruments.items()):
            yield (lbl,obj)

    def iter_gates(self):
        """
        Returns
        -------
        iterator
            an iterator over all (gateLabel,gate) pairs
        """
        assert(False),"Deprecated!"
        #for (label,gate) in self.gates.items():
        #    yield (label, gate)

    def iter_preps(self):
        """
        Returns
        -------
        iterator
            an iterator over all (prepLabel,vector) pairs
        """
        assert(False),"Deprecated!"
        #for (label,vec) in self.preps.items():
        #    yield (label, vec)

    def iter_effects(self):
        """
        Returns
        -------
        iterator
            an iterator over all (effectLabel,vector) pairs
        """
        assert(False),"Deprecated!"
        #for (label,vec) in self.effects.items():
        #    yield (label, vec)



#TODO: how to handle these given possibility of different parameterizations...
#  -- maybe only allow these methods to be called when using a "full" parameterization?
#  -- or perhaps better to *move* them to the parameterization class
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
         apply depolarizing noise of strength ``1-gate_noise`` to all gates in
          the gateset. (Multiplies each assumed-Pauli-basis gate matrix by the
          diagonal matrix with ``(1.0-gate_noise)`` along all the diagonal
          elements except the first (the identity).

        spam_noise : float, optional
          apply depolarizing noise of strength ``1-spam_noise`` to all SPAM
          vectors in the gateset. (Multiplies the non-identity part of each
          assumed-Pauli-basis state preparation vector and measurement vector
          by ``(1.0-spam_noise)``).

        max_gate_noise : float, optional

          specified instead of `gate_noise`; apply a random depolarization
          with maximum strength ``1-max_gate_noise`` to each gate in the
          gateset.

        max_spam_noise : float, optional
          specified instead of `spam_noise`; apply a random depolarization
          with maximum strength ``1-max_spam_noise`` to SPAM vector in the
          gateset.

        seed : int, optional
          if not ``None``, seed numpy's random number generator with this value
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
            r = max_gate_noise * rndm.random_sample(len(self.gates))
            for i,label in enumerate(self.gates):
                newGateset.gates[label].depolarize(r[i])
            r = max_gate_noise * rndm.random_sample(len(self.instruments))
            for i,label in enumerate(self.instruments):
                newGateset.instruments[label].depolarize(r[i])

        elif gate_noise is not None:
            #Apply the same depolarization to each gate
            for label in self.gates:
                newGateset.gates[label].depolarize(gate_noise)
            for label in self.instruments:
                newGateset.instruments[label].depolarize(gate_noise)

        if max_spam_noise is not None:
            if spam_noise is not None:
                raise ValueError("Must specify at most  one of 'noise' and 'max_noise' NOT both")

            #Apply random depolarization to each rho and E vector
            r = max_spam_noise * rndm.random_sample( len(self.preps) )
            for (i,lbl) in enumerate(self.preps):
                newGateset.preps[lbl].depolarize(r[i])
            r = max_spam_noise * rndm.random_sample( len(self.povms) )
            for label in self.povms:
                newGateset.povms[label].depolarize(r[i])

        elif spam_noise is not None:
            #Apply the same depolarization to each gate
            D = _np.diag( [1]+[1-spam_noise]*(gateDim-1) )
            for lbl in self.preps:
                newGateset.preps[lbl].depolarize(spam_noise)

            # Just depolarize the preps - leave POVMs alone
            #for label in self.povms:
            #    newGateset.povms[label].depolarize(spam_noise)

        newGateset._clean_paramvec() #depolarize may leave dirty members
        return newGateset


    def rotate(self, rotate=None, max_rotate=None, seed=None):
        """
        Apply a rotation uniformly (the same rotation applied to each gate)
        or randomly (different random rotations to each gate) to this gateset,
        and return the result, without modifying the original (this) gateset.

        You must specify either 'rotate' or 'max_rotate'. This method currently
        only works on n-qubit gatesets.

        Parameters
        ----------
        rotate : tuple of floats, optional
            If you specify the `rotate` argument, then the same rotation
            operation is applied to each gate.  That is, each gate's matrix `G`
            is composed with a rotation operation `R`  (so `G` -> `dot(R, G)` )
            where `R` is the unitary superoperator corresponding to the unitary
            operator `U = exp( sum_k( i * rotate[k] / 2.0 * Pauli_k ) )`.  Here
            `Pauli_k` ranges over all of the non-identity un-normalized Pauli
            operators (e.g. {X,Y,Z} for 1 qubit, {IX, IY, IZ, XI, XX, XY, XZ,
            YI, YX, YY, YZ, ZI, ZX, ZY, ZZ} for 2 qubits).

        max_rotate : float, optional
            If `max_rotate` is specified (*instead* of `rotate`), then pyGSTi
            randomly generates a different `rotate` tuple, and applies the
            corresponding rotation, to each gate in this `GateSet`.  Each
            component of each tuple is drawn uniformly from [0, `max_rotate`).

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
        myBasis = self.basis

        if max_rotate is not None:
            if rotate is not None:
                raise ValueError("Must specify exactly one of 'rotate' and 'max_rotate' NOT both")

            #Apply random rotation to each gate
            rndm = _np.random.RandomState(seed)
            r = max_rotate * rndm.random_sample( len(self.gates) * (dim-1) )
            for i,label in enumerate(self.gates):
                rot = _np.array(r[(dim-1)*i:(dim-1)*(i+1)])
                newGateset.gates[label].rotate(rot, myBasis)
            r = max_rotate * rndm.random_sample( len(self.instruments) * (dim-1) )
            for i,label in enumerate(self.instruments):
                rot = _np.array(r[(dim-1)*i:(dim-1)*(i+1)])
                newGateset.instruments[label].rotate(rot, myBasis)


        elif rotate is not None:
            assert(len(rotate) == dim-1), \
                "Invalid 'rotate' argument. You must supply a tuple of length %d" % (dim-1)
            for label in self.gates:
                newGateset.gates[label].rotate(rotate, myBasis)
            for label in self.instruments:
                newGateset.instruments[label].rotate(rotate, myBasis)

        else: raise ValueError("Must specify either 'rotate' or 'max_rotate' "
                               + "-- neither was non-None")

        newGateset._clean_paramvec() #rotate may leave dirty members
        return newGateset


    def randomize_with_unitary(self, scale, seed=None, randState=None):
        """Create a new gateset with random unitary perturbations.

        Apply a random unitary to each element of a gateset, and return the
        result, without modifying the original (this) gateset. This method
        works on GateSet as long as the dimension is a perfect square.

        Parameters
        ----------
        scale : float
          maximum element magnitude in the generator of each random unitary
          transform.

        seed : int, optional
          if not None, seed numpy's random number generator with this value
          before generating random depolarizations.

        randState : numpy.random.RandomState
            A RandomState object to generate samples from. Can be useful to set
            instead of `seed` if you want reproducible distribution samples
            across multiple random function calls but you don't want to bother
            with manually incrementing seeds between those calls.

        Returns
        -------
        GateSet
            the randomized GateSet
        """
        if randState is None:
            rndm = _np.random.RandomState(seed)
        else:
            rndm = randState

        gate_dim = self.get_dimension()
        unitary_dim = int(round(_np.sqrt(gate_dim)))
        assert( unitary_dim**2 == gate_dim ), \
            "GateSet dimension must be a perfect square, %d is not" % gate_dim

        gs_randomized = self.copy()

        for gateLabel,gate in self.gates.items():
            randMat = scale * (rndm.randn(unitary_dim,unitary_dim) \
                                   + 1j * rndm.randn(unitary_dim,unitary_dim))
            randMat = _np.transpose(_np.conjugate(randMat)) + randMat
                  # make randMat Hermetian: (A_dag + A)^dag = (A_dag + A)
            randUnitary   = _scipy.linalg.expm(-1j*randMat)

            randGate = _gt.unitary_to_process_mx(randUnitary) #in std basis
            randGate = _bt.change_basis(randGate, "std", self.basis)

            gs_randomized.gates[gateLabel] = _gate.FullyParameterizedGate(
                            _np.dot(randGate,gate))

        #Note: this function does NOT randomize instruments

        return gs_randomized


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

        new_gateset = GateSet("full", self.preps._prefix, self.effects_prefix,
                              self.gates._prefix, self.povms._prefix,
                              self.instruments._prefix, self._sim_type)
        new_gateset._dim = newDimension
        new_gateset.reset_basis() #FUTURE: maybe user can specify how increase is being done?

        addedDim = newDimension-curDim
        vec_zeroPad = _np.zeros( (addedDim,1), 'd')

        #Increase dimension of rhoVecs and EVecs by zero-padding
        for lbl,rhoVec in self.preps.items():
            assert( len(rhoVec) == curDim )
            new_gateset.preps[lbl] = \
                _sv.FullyParameterizedSPAMVec(_np.concatenate( (rhoVec, vec_zeroPad) ))

        for lbl,povm in self.povms.items():
            assert( povm.dim == curDim )
            effects = [ (elbl,_np.concatenate( (EVec, vec_zeroPad) ))
                        for elbl,EVec in povm.items() ]

            if isinstance(povm, _povm.TPPOVM):
                new_gateset.povms[lbl] = _povm.TPPOVM(effects)
            else:
                new_gateset.povms[lbl] = _povm.UnconstrainedPOVM(effects) #everything else

        #Increase dimension of gates by assuming they act as identity on additional (unknown) space
        for gateLabel,gate in self.gates.items():
            assert( gate.shape == (curDim,curDim) )
            newGate = _np.zeros( (newDimension,newDimension) )
            newGate[ 0:curDim, 0:curDim ] = gate[:,:]
            for i in range(curDim,newDimension): newGate[i,i] = 1.0
            new_gateset.gates[gateLabel] = _gate.FullyParameterizedGate(newGate)

        for instLabel,inst in self.instruments.items():
            inst_gates = []
            for outcomeLbl,gate in inst.items():
                newGate = _np.zeros( (newDimension,newDimension) )
                newGate[ 0:curDim, 0:curDim ] = gate[:,:]
                for i in range(curDim,newDimension): newGate[i,i] = 1.0
                inst_gates.append( (outcomeLbl,_gate.FullyParameterizedGate(newGate)) )
            new_gateset.instruments[instLabel] = _instrument.Instrument(inst_gates)

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

        new_gateset = GateSet("full", self.preps._prefix, self.effects_prefix,
                              self.gates._prefix, self.povms._prefix,
                              self.instruments._prefix, self._sim_type)
        new_gateset._dim = newDimension
        new_gateset.reset_basis() #FUTURE: maybe user can specify how decrease is being done?

        #Decrease dimension of rhoVecs and EVecs by truncation
        for lbl,rhoVec in self.preps.items():
            assert( len(rhoVec) == curDim )
            new_gateset.preps[lbl] = \
                _sv.FullyParameterizedSPAMVec(rhoVec[0:newDimension,:])

        for lbl,povm in self.povms.items():
            assert( povm.dim == curDim )
            effects = [ (elbl,EVec[0:newDimension,:]) for elbl,EVec in povm.items()]

            if isinstance(povm, _povm.TPPOVM):
                new_gateset.povms[lbl] = _povm.TPPOVM(effects)
            else:
                new_gateset.povms[lbl] = _povm.UnconstrainedPOVM(effects) #everything else


        #Decrease dimension of gates by truncation
        for gateLabel,gate in self.gates.items():
            assert( gate.shape == (curDim,curDim) )
            newGate = _np.zeros( (newDimension,newDimension) )
            newGate[ :, : ] = gate[0:newDimension,0:newDimension]
            new_gateset.gates[gateLabel] = _gate.FullyParameterizedGate(newGate)

        for instLabel,inst in self.instruments.items():
            inst_gates = []
            for outcomeLbl,gate in inst.items():
                newGate = _np.zeros( (newDimension,newDimension) )
                newGate[ :, : ] = gate[0:newDimension,0:newDimension]
                inst_gates.append( (outcomeLbl,_gate.FullyParameterizedGate(newGate)) )
            new_gateset.instruments[instLabel] = _instrument.Instrument(inst_gates)

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
        for gateLabel,gate in self.gates.items():
            delta = absmag * 2.0*(rndm.random_sample(gate.shape)-0.5) + bias
            kicked_gs.gates[gateLabel] = _gate.FullyParameterizedGate(
                                            kicked_gs.gates[gateLabel] + delta )

        #Note: does not alter intruments!
        return kicked_gs


    def get_clifford_symplectic_reps(self, gatelabel_filter=None):
        """
        Constructs a dictionary of the symplectic representations for all
        the Clifford gates in this gateset.  Non-:class:`CliffordGate` gates
        will be ignored and their entries omitted from the returned dictionary.

        Parameters
        ----------
        gatelabel_filter : iterable, optional
            A list, tuple, or set of gate labels whose symplectic
            representations should be returned (if they exist).

        Returns
        -------
        dict
            keys are gate labels and/or just the root names of gates
            (without any state space indices/labels).  Values are
            `(symplectic_matrix, phase_vector)` tuples.
        """
        gfilter = set(gatelabel_filter) if gatelabel_filter is not None \
                  else None

        srep_dict = {}
        for gl,gate in self.gates.items():
            if (gfilter is not None) and (gl not in gfilter): continue

            if isinstance(gate, _gate.EmbeddedGateMap):
                assert(isinstance(gate.embedded_gate, _gate.CliffordGate)), \
                    "EmbeddedClifforGate contains a non-CliffordGate!"
                lbl = gl.name # strip state space labels off since this is a
                              # symplectic rep for the *embedded* gate
                srep = (gate.embedded_gate.smatrix,gate.embedded_gate.svector)
            elif isinstance(gate, _gate.CliffordGate):
                lbl = gl.name 
                srep = (gate.smatrix,gate.svector)
            else:
                lbl = srep = None

            if srep:
                if lbl in srep_dict:
                    assert(srep == srep_dict[lbl]), \
                        "Inconsistent symplectic reps for %s label!" % lbl
                else:
                    srep_dict[lbl] = srep

        return srep_dict



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
        print(self)
        print("\n")
        print("Basis = ",self.basis.name)
        print("Choi Matrices:")
        for (label,gate) in self.gates.items():
            print(("Choi(%s) in pauli basis = \n" % label,
            _mt.mx_to_string_complex(_jt.jamiolkowski_iso(gate))))
            print(("  --eigenvals = ", sorted(
                [ev.real for ev in _np.linalg.eigvals(
                        _jt.jamiolkowski_iso(gate))] ),"\n"))
        print(("Sum of negative Choi eigenvalues = ", _jt.sum_of_negative_choi_evals(self)))
