"""
Defines the ExplicitOpModel class and supporting functionality.
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import numpy as _np
import scipy as _scipy
import itertools as _itertools
import collections as _collections
import warnings as _warnings
import time as _time
import uuid as _uuid
import bisect as _bisect
import copy as _copy

from ..tools import matrixtools as _mt
from ..tools import optools as _gt
from ..tools import slicetools as _slct
from ..tools import likelihoodfns as _lf
from ..tools import jamiolkowski as _jt
from ..tools import basistools as _bt
from ..tools import listtools as _lt
from ..tools import symplectic as _symp

from . import model as _mdl
from . import modelmember as _gm
from . import circuit as _cir
from . import operation as _op
from . import spamvec as _sv
from . import povm as _povm
from . import instrument as _instrument
from . import labeldicts as _ld
from . import gaugegroup as _gg
from . import matrixforwardsim as _matrixfwdsim
from . import mapforwardsim as _mapfwdsim
from . import termforwardsim as _termfwdsim
from . import explicitcalc as _explicitcalc
from . import opfactory as _opfactory

from .verbosityprinter import VerbosityPrinter as _VerbosityPrinter
from .basis import BuiltinBasis as _BuiltinBasis, DirectSumBasis as _DirectSumBasis
from .label import Label as _Label, CircuitLabel as _CircuitLabel
from .layerrules import LayerRules as _LayerRules
from .modelparaminterposer import LinearInterposer as _LinearInterposer


class ExplicitOpModel(_mdl.OpModel):
    """
    Encapsulates a set of gate, state preparation, and POVM effect operations.

    An ExplictOpModel stores a set of labeled LinearOperator objects and
    provides dictionary-like access to their matrices.  State preparation
    and POVM effect operations are represented as column vectors.

    Parameters
    ----------
    state_space_labels : StateSpaceLabels or list or tuple
        The decomposition (with labels) of (pure) state-space this model
        acts upon.  Regardless of whether the model contains operators or
        superoperators, this argument describes the Hilbert space dimension
        and imposed structure.  If a list or tuple is given, it must be
        of a from that can be passed to `StateSpaceLabels.__init__`.

    basis : {"auto","pp","gm","qt","std","sv"} or Basis
        The basis used for the state space by dense operator representations.

    default_param : {"full", "TP", "CPTP", etc.}, optional
        Specifies the default gate and SPAM vector parameterization type.
        Can be any value allowed by :method:`set_all_parameterizations`,
        which also gives a description of each parameterization type.

    prep_prefix: string, optional
        Key prefixe for state preparations, allowing the model to determing what
        type of object a key corresponds to.

    effect_prefix : string, optional
        Key prefix for POVM effects, allowing the model to determing what
        type of object a key corresponds to.

    gate_prefix : string, optional
        Key prefix for gates, allowing the model to determing what
        type of object a key corresponds to.

    povm_prefix : string, optional
        Key prefix for POVMs, allowing the model to determing what
        type of object a key corresponds to.

    instrument_prefix : string, optional
        Key prefix for instruments, allowing the model to determing what
        type of object a key corresponds to.

    simulator : ForwardSimulator or {"auto", "matrix", "map"}
        The circuit simulator used to compute any
        requested probabilities, e.g. from :method:`probs` or
        :method:`bulk_probs`.  The default value of `"auto"` automatically
        selects the simulation type, and is usually what you want. Other
        special allowed values are:

        - "matrix" : op_matrix-op_matrix products are computed and
          cached to get composite gates which can then quickly simulate
          a circuit for any preparation and outcome.  High memory demand;
          best for a small number of (1 or 2) qubits.
        - "map" : op_matrix-state_vector products are repeatedly computed
          to simulate circuits.  Slower for a small number of qubits, but
          faster and more memory efficient for higher numbers of qubits (3+).

    evotype : {"densitymx", "statevec", "stabilizer", "svterm", "cterm"}
        The evolution type of this model, describing how states are
        represented, allowing compatibility checks with (super)operator
        objects.
    """

    #Whether access to gates & spam vecs via Model indexing is allowed
    _strict = False

    def __init__(self, state_space_labels, basis="auto", default_param="full",
                 prep_prefix="rho", effect_prefix="E", gate_prefix="G",
                 povm_prefix="M", instrument_prefix="I", simulator="auto",
                 evotype="densitymx"):
        """
        Initialize an ExplictOpModel.

        Parameters
        ----------
        state_space_labels : StateSpaceLabels or list or tuple
            The decomposition (with labels) of (pure) state-space this model
            acts upon.  Regardless of whether the model contains operators or
            superoperators, this argument describes the Hilbert space dimension
            and imposed structure.  If a list or tuple is given, it must be
            of a from that can be passed to `StateSpaceLabels.__init__`.

        basis : {"auto","pp","gm","qt","std","sv"} or Basis
            The basis used for the state space by dense operator representations.

        default_param : {"full", "TP", "CPTP", etc.}, optional
            Specifies the default gate and SPAM vector parameterization type.
            Can be any value allowed by :method:`set_all_parameterizations`,
            which also gives a description of each parameterization type.

        prep_prefix, effect_prefix, gate_prefix,
        povm_prefix, instrument_prefix : string, optional
            Key prefixes designating state preparations, POVM effects,
            gates, POVM, and instruments respectively.  These prefixes allow
            the Model to determine what type of object a key corresponds to.

        simulator : ForwardSimulator or {"auto", "matrix", "map"}
            The circuit simulator used to compute any
            requested probabilities, e.g. from :method:`probs` or
            :method:`bulk_probs`.

        evotype : {"densitymx", "statevec", "stabilizer", "svterm", "cterm"}
            The evolution type of this model, describing how states are
            represented, allowing compatibility checks with (super)operator
            objects.
        """
        #More options now (TODO enumerate?)
        #assert(default_param in ('full','TP','CPTP','H+S','S','static',
        #                         'H+S terms','clifford','H+S clifford terms'))
        def flagfn(typ): return {'auto_embed': True, 'match_parent_dim': True,
                                 'match_parent_evotype': True, 'cast_to_type': typ}

        self.preps = _ld.OrderedMemberDict(self, default_param, prep_prefix, flagfn("spamvec"))
        self.povms = _ld.OrderedMemberDict(self, default_param, povm_prefix, flagfn("povm"))
        self.operations = _ld.OrderedMemberDict(self, default_param, gate_prefix, flagfn("operation"))
        self.instruments = _ld.OrderedMemberDict(self, default_param, instrument_prefix, flagfn("instrument"))
        self.factories = _ld.OrderedMemberDict(self, default_param, gate_prefix, flagfn("factory"))
        self.effects_prefix = effect_prefix
        self._default_gauge_group = None
        self.fogi_info = None

        if basis == "auto":
            basis = "pp" if evotype in ("densitymx", "svterm", "cterm") \
                else "sv"  # ( if evotype in ("statevec","stabilizer") )

        super(ExplicitOpModel, self).__init__(state_space_labels, basis, evotype, ExplicitLayerRules(), simulator)

    @property
    def _primitive_prep_label_dict(self):
        return self.preps

    @property
    def _primitive_povm_label_dict(self):
        return self.povms

    @property
    def _primitive_op_label_dict(self):
        return self.operations

    @property
    def _primitive_instrument_label_dict(self):
        return self.instruments

    #Functions required for base class functionality

    def _iter_parameterized_objs(self):
        for lbl, obj in _itertools.chain(self.preps.items(),
                                         self.povms.items(),
                                         self.operations.items(),
                                         self.instruments.items(),
                                         self.factories.items()):
            yield (lbl, obj)

    def _excalc(self):
        """ Create & return a special explicit-model calculator for this model """

        self._clean_paramvec()  # ensures paramvec is rebuild if needed
        simplified_effects = _collections.OrderedDict()
        for povm_lbl, povm in self.povms.items():
            for k, e in povm.simplify_effects(povm_lbl).items():
                simplified_effects[k] = e

        simplified_ops = _collections.OrderedDict()
        for k, g in self.operations.items(): simplified_ops[k] = g
        for inst_lbl, inst in self.instruments.items():
            for k, g in inst.simplify_operations(inst_lbl).items():
                simplified_ops[k] = g
        simplified_preps = self.preps

        return _explicitcalc.ExplicitOpModelCalc(self.dim, simplified_preps, simplified_ops,
                                                 simplified_effects, self.num_params)

    #Unneeded - just use string processing & rely on effect labels *not* having underscores in them
    #def simplify_spamtuple_to_outcome_label(self, simplified_spamTuple):
    #    #TODO: make this more efficient (prep lbl isn't even used!)
    #    for prep_lbl in self.preps:
    #        for povm_lbl in self.povms:
    #            for elbl in self.povms[povm_lbl]:
    #                if simplified_spamTuple == (prep_lbl, povm_lbl + "_" + elbl):
    #                    return (elbl,) # outcome "label" (a tuple)
    #    raise ValueError("No outcome label found for simplified spam_tuple: ", simplified_spamTuple)

    def _embed_operation(self, op_target_labels, op_val, force=False):
        """
        Called by OrderedMemberDict._auto_embed to create an embedded-gate
        object that embeds `op_val` into the sub-space of
        `self.state_space_labels` given by `op_target_labels`.

        Parameters
        ----------
        op_target_labels : list
            A list of `op_val`'s target state space labels.

        op_val : LinearOperator
            The gate object to embed.  Note this should be a legitimate
            LinearOperator-derived object and not just a numpy array.

        force : bool, optional
            Always wrap with an embedded LinearOperator, even if the
            dimension of `op_val` is the full model dimension.

        Returns
        -------
        LinearOperator
            A gate of the full model dimension.
        """
        if self.dim is None:
            raise ValueError("Must set model dimension before adding auto-embedded gates.")
        if self.state_space_labels is None:
            raise ValueError("Must set model.state_space_labels before adding auto-embedded gates.")

        if op_val.dim == self.dim and not force:
            return op_val  # if gate operates on full dimension, no need to embed.

        if isinstance(self._sim, _matrixfwdsim.MatrixForwardSimulator):
            return _op.EmbeddedDenseOp(self.state_space_labels, op_target_labels, op_val)
        else:  # all other types, e.g. "map" and "termorder"
            return _op.EmbeddedOp(self.state_space_labels, op_target_labels, op_val)

    @property
    def default_gauge_group(self):
        """
        Gets the default gauge group for performing gauge transformations on this Model.

        Returns
        -------
        GaugeGroup
        """
        return self._default_gauge_group

    @default_gauge_group.setter
    def default_gauge_group(self, value):
        """
        The default gauge group.
        """
        self._default_gauge_group = value

    @property
    def prep(self):
        """
        The unique state preparation in this model, if one exists.

        If not, a ValueError is raised.

        Returns
        -------
        SPAMVec
        """
        if len(self.preps) != 1:
            raise ValueError("'.prep' can only be used on models"
                             " with a *single* state prep.  This Model has"
                             " %d state preps!" % len(self.preps))
        return list(self.preps.values())[0]

    @property
    def effects(self):
        """
        The unique POVM in this model, if one exists.

        If not, a ValueError is raised.

        Returns
        -------
        POVM
        """
        if len(self.povms) != 1:
            raise ValueError("'.effects' can only be used on models"
                             " with a *single* POVM.  This Model has"
                             " %d POVMS!" % len(self.povms))
        return list(self.povms.values())[0]

    def __setitem__(self, label, value):
        """
        Set an operator or SPAM vector associated with a given label.

        Parameters
        ----------
        label : string
            the gate or SPAM vector label.

        value : numpy array or LinearOperator or SPAMVec
            a operation matrix, SPAM vector, or object, which must have the
            appropriate dimension for the Model and appropriate type
            given the prefix of the label.
        """
        if ExplicitOpModel._strict:
            raise KeyError("Strict-mode: invalid key %s" % repr(label))

        if not isinstance(label, _Label): label = _Label(label)

        if label == _Label(()):  # special case
            self.operations[label] = value
        elif label.has_prefix(self.preps._prefix):
            self.preps[label] = value
        elif label.has_prefix(self.povms._prefix):
            self.povms[label] = value
        elif label.has_prefix(self.operations._prefix):
            self.operations[label] = value
        elif label.has_prefix(self.instruments._prefix, typ="any"):
            self.instruments[label] = value
        else:
            raise KeyError("Key %s has an invalid prefix" % label)

    def __getitem__(self, label):
        """
        Get an operation or SPAM vector associated with a given label.

        Parameters
        ----------
        label : string
            the gate or SPAM vector label.
        """
        if ExplicitOpModel._strict:
            raise KeyError("Strict-mode: invalid key %s" % label)

        if not isinstance(label, _Label): label = _Label(label)

        if label == _Label(()):  # special case
            return self.operations[label]
        elif label.has_prefix(self.preps._prefix):
            return self.preps[label]
        elif label.has_prefix(self.povms._prefix):
            return self.povms[label]
        elif label.has_prefix(self.operations._prefix):
            return self.operations[label]
        elif label.has_prefix(self.instruments._prefix, typ="any"):
            return self.instruments[label]
        else:
            raise KeyError("Key %s has an invalid prefix" % label)

    def set_all_parameterizations(self, parameterization_type, extra=None):
        """
        Convert all gates and SPAM vectors to a specific parameterization type.

        Parameters
        ----------
        parameterization_type : string
            The gate and SPAM vector parameterization type.  Allowed
            values are (where '*' means " terms" and " clifford terms"
            evolution-type suffixes are allowed):

            - "full" : each gate / SPAM element is an independent parameter
            - "TP" : Trace-Preserving gates and state preps
            - "static" : no parameters
            - "static unitary" : no parameters; convert superops to unitaries
            - "clifford" : no parameters; convert unitaries to Clifford symplecitics.
            - "GLND*" : General unconstrained Lindbladian
            - "CPTP*" : Completely-Positive-Trace-Preserving
            - "H+S+A*" : Hamiltoian, Pauli-Stochastic, and Affine errors
            - "H+S*" : Hamiltonian and Pauli-Stochastic errors
            - "S+A*" : Pauli-Stochastic and Affine errors
            - "S*" : Pauli-Stochastic errors
            - "H+D+A*" : Hamiltoian, Depolarization, and Affine errors
            - "H+D*" : Hamiltonian and Depolarization errors
            - "D+A*" : Depolarization and Affine errors
            - "D*" : Depolarization errors
            - Any of the above with "S" replaced with "s" or "D" replaced with
              "d". This removes the CPTP constraint on the Gates and SPAM (and
              as such is seldom used).

        extra : dict, optional
            For `"H+S terms"` type, this may specify a dictionary
            of unitary gates and pure state vectors to be used
            as the *ideal* operation of each gate/SPAM vector.

        Returns
        -------
        None
        """
        typ = parameterization_type

        #More options now (TODO enumerate?)
        #assert(parameterization_type in ('full','TP','CPTP','H+S','S','static',
        #                                 'H+S terms','clifford','H+S clifford terms',
        #                                 'static unitary'))

        #Update dim and evolution type so that setting converted elements works correctly
        baseType = typ  # the default - only updated if a lindblad param type

        if typ == 'static unitary':
            assert(self._evotype == "densitymx"), \
                "Can only convert to 'static unitary' from a density-matrix evolution type."
            self._evotype = "statevec"
            self._dim = int(round(_np.sqrt(self.dim)))  # reduce dimension d -> sqrt(d)
            if not isinstance(self._sim, (_matrixfwdsim.MatrixForwardSimulator, _mapfwdsim.MapForwardSimulator)):
                self._sim = _matrixfwdsim.MatrixForwardSimulator(self) if self.dim <= 4 else \
                    _mapfwdsim.MapForwardSimulator(self, max_cache_size=0)

        elif typ == 'clifford':
            self._evotype = "stabilizer"
            self._sim = _mapfwdsim.SimpleMapForwardSimulator(self)
            #self._sim = _mapfwdsim.MapForwardSimulator(self, max_cache_size=0)

        elif _gt.is_valid_lindblad_paramtype(typ):
            baseType, evotype = _gt.split_lindblad_paramtype(typ)
            self._evotype = evotype
            if evotype == "densitymx":
                if not isinstance(self._sim, (_matrixfwdsim.MatrixForwardSimulator, _mapfwdsim.MapForwardSimulator)):
                    self._sim = _matrixfwdsim.MatrixForwardSimulator(self) if self.dim <= 16 else \
                        _mapfwdsim.MapForwardSimulator(self, max_cache_size=0)
            elif evotype in ("svterm", "cterm"):
                if not isinstance(self._sim, _termfwdsim.TermForwardSimulator):
                    self._sim = _termfwdsim.TermForwardSimulator(self)

        elif typ in ('static', 'full', 'TP', 'CPTP', 'linear'):  # assume all other parameterizations are densitymx type
            self._evotype = "densitymx"
            if not isinstance(self._sim, (_matrixfwdsim.MatrixForwardSimulator, _mapfwdsim.MapForwardSimulator)):
                self._sim = _matrixfwdsim.MatrixForwardSimulator(self) if self.dim <= 16 else \
                    _mapfwdsim.MapForwardSimulator(self, max_cache_size=0)
        else:
            raise ValueError("Invalid parameterization type: %s" % str(typ))

        basis = self.basis
        if extra is None: extra = {}

        povmtyp = rtyp = typ  # assume spam types are available to all objects
        ityp = "TP" if _gt.is_valid_lindblad_paramtype(typ) else typ

        for lbl, gate in self.operations.items():
            self.operations[lbl] = _op.convert(gate, typ, basis,
                                               extra.get(lbl, None))

        for lbl, inst in self.instruments.items():
            self.instruments[lbl] = _instrument.convert(inst, ityp, basis,
                                                        extra.get(lbl, None))

        for lbl, vec in self.preps.items():
            self.preps[lbl] = _sv.convert(vec, rtyp, basis,
                                          extra.get(lbl, None))

        for lbl, povm in self.povms.items():
            self.povms[lbl] = _povm.convert(povm, povmtyp, basis,
                                            extra.get(lbl, None))

        if typ == 'full':
            self.default_gauge_group = _gg.FullGaugeGroup(self.dim)
        elif typ == 'TP':
            self.default_gauge_group = _gg.TPGaugeGroup(self.dim)
        elif typ == 'CPTP':
            self.default_gauge_group = _gg.UnitaryGaugeGroup(self.dim, basis)
        else:  # typ in ('static','H+S','S', 'H+S terms', ...)
            self.default_gauge_group = _gg.TrivialGaugeGroup(self.dim)

    def __setstate__(self, state_dict):

        if "gates" in state_dict:
            #Unpickling an OLD-version Model (or GateSet)
            _warnings.warn("Unpickling deprecated-format ExplicitOpModel (GateSet).  Please re-save/pickle asap.")
            self.operations = state_dict['gates']
            self._state_space_labels = state_dict['stateSpaceLabels']
            self._paramlbls = None
            del state_dict['gates']
            del state_dict['_autogator']
            del state_dict['auto_idle_gatename']
            del state_dict['stateSpaceLabels']

        if "effects" in state_dict:
            raise ValueError(("This model (GateSet) object is too old to unpickle - "
                              "try using pyGSTi v0.9.6 to upgrade it to a version "
                              "that this version can upgrade to the current version."))

        #Backward compatibility:
        if 'basis' in state_dict:
            state_dict['_basis'] = state_dict['basis']; del state_dict['basis']
        if 'state_space_labels' in state_dict:
            state_dict['_state_space_labels'] = state_dict['state_space_labels']; del state_dict['_state_space_labels']
        if 'factories' not in state_dict:
            ops = state_dict['operations']
            state_dict['factories'] = _ld.OrderedMemberDict(self, ops.default_param, ops._prefix, ops.flags)

        super().__setstate__(state_dict)  # ~ self.__dict__.update(state_dict)

        if 'uuid' not in state_dict:
            self.uuid = _uuid.uuid4()  # create a new uuid

        #Additionally, must re-connect this model as the parent
        # of relevant OrderedDict-derived classes, which *don't*
        # preserve this information upon pickling so as to avoid
        # circular pickling...
        self.preps.parent = self
        self.povms.parent = self
        #self.effects.parent = self
        self.operations.parent = self
        self.instruments.parent = self
        self.factories.parent = self
        for o in self.preps.values(): o.relink_parent(self)
        for o in self.povms.values(): o.relink_parent(self)
        #for o in self.effects.values(): o.relink_parent(self)
        for o in self.operations.values(): o.relink_parent(self)
        for o in self.instruments.values(): o.relink_parent(self)
        for o in self.factories.values(): o.relink_parent(self)

    @property
    def num_elements(self):
        """
        Return the number of total operation matrix and spam vector elements in this model.

        This is in general different from the number of *parameters* in the
        model, which are the number of free variables used to generate all of
        the matrix and vector *elements*.

        Returns
        -------
        int
            the number of model elements.
        """
        rhoSize = [rho.size for rho in self.preps.values()]
        povmSize = [povm.num_elements for povm in self.povms.values()]
        opSize = [gate.size for gate in self.operations.values()]
        instSize = [i.num_elements for i in self.instruments.values()]
        #Don't count self.factories?
        return sum(rhoSize) + sum(povmSize) + sum(opSize) + sum(instSize)

    @property
    def num_nongauge_params(self):
        """
        Return the number of non-gauge parameters in this model.

        Returns
        -------
        int
            the number of non-gauge model parameters.
        """
        return self.num_params - self.num_gauge_params

    @property
    def num_gauge_params(self):
        """
        Return the number of gauge parameters in this model.

        Returns
        -------
        int
            the number of gauge model parameters.
        """
        if self._evotype not in ("densitymx", "statevec"):
            return 0  # punt on computing number of gauge parameters for other evotypes
        if self.num_params == 0:
            return 0  # save the trouble of getting gauge params when there are no params to begin with
        dPG = self._excalc()._buildup_dpg()
        gaugeDirs = _mt.nullspace_qr(dPG)  # cols are gauge directions
        if gaugeDirs.size == 0:  # if there are *no* gauge directions
            return 0  # calling matrix_rank on a length-0 array => error
        return _np.linalg.matrix_rank(gaugeDirs[0:self.num_params, :])

    def deriv_wrt_params(self):
        """
        The element-wise derivative of all this models' operations.

        Constructs a matrix whose columns are the vectorized derivatives of all
        the model's raw matrix and vector *elements* (placed in a vector)
        with respect to each single model parameter.

        Thus, each column has length equal to the number of elements in the
        model, and there are num_params() columns.  In the case of a "fully
        parameterized model" (i.e. all operation matrices and SPAM vectors are
        fully parameterized) then the resulting matrix will be the (square)
        identity matrix.

        Returns
        -------
        numpy array
            2D array of derivatives.
        """
        return self._excalc().deriv_wrt_params()

    def compute_nongauge_projector(self, item_weights=None, non_gauge_mix_mx=None):
        """
        Construct a projector onto the non-gauge parameter space.

        Useful for isolating the gauge degrees of freedom from the non-gauge
        degrees of freedom.

        Parameters
        ----------
        item_weights : dict, optional
            Dictionary of weighting factors for individual gates and spam operators.
            Keys can be gate, state preparation, POVM effect, spam labels, or the
            special strings "gates" or "spam" whic represent the entire set of gate
            or SPAM operators, respectively.  Values are floating point numbers.
            These weights define the metric used to compute the non-gauge space,
            *orthogonal* the gauge space, that is projected onto.

        non_gauge_mix_mx : numpy array, optional
            An array of shape (n_non_gauge_params,n_gauge_params) specifying how to
            mix the non-gauge degrees of freedom into the gauge degrees of
            freedom that are projected out by the returned object.  This argument
            essentially sets the off-diagonal block of the metric used for
            orthogonality in the "gauge + non-gauge" space.  It is for advanced
            usage and typically left as None (the default).

        Returns
        -------
        numpy array
           The projection operator as a N x N matrix, where N is the number
           of parameters (obtained via num_params()).  This projector acts on
           parameter-space, and has rank equal to the number of non-gauge
           degrees of freedom.
        """
        return self._excalc().nongauge_projector(item_weights, non_gauge_mix_mx)

    def transform_inplace(self, s):
        """
        Gauge transform this model.

        Update each of the operation matrices G in this model with inv(s) * G * s,
        each rhoVec with inv(s) * rhoVec, and each EVec with EVec * s

        Parameters
        ----------
        s : GaugeGroupElement
            A gauge group element which specifies the "s" matrix
            (and it's inverse) used in the above similarity transform.

        Returns
        -------
        None
        """
        for rhoVec in self.preps.values():
            rhoVec.transform_inplace(s, 'prep')

        for povm in self.povms.values():
            povm.transform_inplace(s)

        for opObj in self.operations.values():
            opObj.transform_inplace(s)

        for instrument in self.instruments.values():
            instrument.transform_inplace(s)

        for factory in self.factories.values():
            factory.transform_inplace(s)

        self._clean_paramvec()  # transform may leave dirty members

    def frobeniusdist(self, other_model, transform_mx=None,
                      item_weights=None, normalize=True):
        """
        Compute the weighted frobenius norm of the difference between this model and other_model.

        Differences in each corresponding gate matrix and spam vector element
        are squared, weighted (using `item_weights` as applicable), then summed.
        The value returned is the square root of this sum, or the square root of
        this sum divided by the number of summands if normalize == True.

        Parameters
        ----------
        other_model : Model
            the other model to difference against.

        transform_mx : numpy array, optional
            if not None, transform this model by
            G => inv(transform_mx) * G * transform_mx, for each operation matrix G
            (and similar for rho and E vectors) before taking the difference.
            This transformation is applied only for the difference and does
            not alter the values stored in this model.

        item_weights : dict, optional
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
        return self._excalc().frobeniusdist(other_model._excalc(), transform_mx,
                                            item_weights, normalize)

    def residuals(self, other_model, transform_mx=None, item_weights=None):
        """
        Compute the weighted residuals between two models.

        Residuals are the differences in corresponding operation matrix and spam
        vector elements.

        Parameters
        ----------
        other_model : Model
            the other model to difference against.

        transform_mx : numpy array, optional
            if not None, transform this model by
            G => inv(transform_mx) * G * transform_mx, for each operation matrix G
            (and similar for rho and E vectors) before taking the difference.
            This transformation is applied only for the difference and does
            not alter the values stored in this model.

        item_weights : dict, optional
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
        return self._excalc().residuals(other_model._excalc(), transform_mx, item_weights)

    def jtracedist(self, other_model, transform_mx=None, include_spam=True):
        """
        Compute the Jamiolkowski trace distance between this model and `other_model`.

        This is defined as the maximum of the trace distances between each
        corresponding gate, including spam gates.

        Parameters
        ----------
        other_model : Model
            the other model to difference against.

        transform_mx : numpy array, optional
            if not None, transform this model by
            G => inv(transform_mx) * G * transform_mx, for each operation matrix G
            (and similar for rho and E vectors) before taking the difference.
            This transformation is applied only for the difference and does
            not alter the values stored in this model.

        include_spam : bool, optional
            Whether to add to the max-trace-distance the frobenius distances
            between corresponding SPAM vectors.

        Returns
        -------
        float
        """
        return self._excalc().jtracedist(other_model._excalc(), transform_mx, include_spam)

    def diamonddist(self, other_model, transform_mx=None, include_spam=True):
        """
        Compute the diamond-norm distance between this model and `other_model`.

        This is defined as the maximum of the diamond-norm distances between each
        corresponding gate, including spam gates.

        Parameters
        ----------
        other_model : Model
            the other model to difference against.

        transform_mx : numpy array, optional
            if not None, transform this model by
            G => inv(transform_mx) * G * transform_mx, for each operation matrix G
            (and similar for rho and E vectors) before taking the difference.
            This transformation is applied only for the difference and does
            not alter the values stored in this model.

        include_spam : bool, optional
            Whether to add to the max-diamond-distance the frobenius distances
            between corresponding SPAM vectors.

        Returns
        -------
        float
        """
        return self._excalc().diamonddist(other_model._excalc(), transform_mx, include_spam)

    def _tpdist(self):
        """
        Compute the "distance" between this model and the space of trace-preserving (TP) maps.

        This is defined as the square root of the sum-of-squared deviations
        among the first row of all operation matrices and the first element of
        all state preparations.

        Returns
        -------
        float
        """
        penalty = 0.0
        for operationMx in list(self.operations.values()):
            penalty += abs(operationMx[0, 0] - 1.0)**2
            for k in range(1, operationMx.shape[1]):
                penalty += abs(operationMx[0, k])**2

        op_dim = self.dim
        firstEl = 1.0 / op_dim**0.25
        for rhoVec in list(self.preps.values()):
            penalty += abs(rhoVec[0, 0] - firstEl)**2

        return _np.sqrt(penalty)

    def strdiff(self, other_model, metric='frobenius'):
        """
        Return a string describing the distances between this model and `other_model`.

        The returned string displays differences between each corresponding gate,
        state prep, and POVM effect.

        Parameters
        ----------
        other_model : Model
            the other model to difference against.

        metric : {'frobenius', 'infidelity', 'diamond'}
            Which distance metric to use.

        Returns
        -------
        str
        """

        if metric == 'frobenius':
            def dist(a, b): return _np.linalg.norm(a - b)
            def vecdist(a, b): return _np.linalg.norm(a - b)
        elif metric == 'infidelity':
            def dist(a, b): return _gt.entanglement_infidelity(a, b, self.basis)
            def vecdist(a, b): return _np.linalg.norm(a - b)
        elif metric == 'diamond':
            def dist(a, b): return 0.5 * _gt.diamondist(a, b, self.basis)
            def vecdist(a, b): return _np.linalg.norm(a - b)
        else:
            raise ValueError("Invalid `metric` argument: %s" % metric)

        s = "Model Difference:\n"
        s += " Preps:\n"
        for lbl in self.preps:
            s += "  %s = %g\n" % \
                (str(lbl), vecdist(self.preps[lbl].to_dense(), other_model.preps[lbl].to_dense()))

        s += " POVMs:\n"
        for povm_lbl, povm in self.povms.items():
            s += "  %s: " % str(povm_lbl)
            for lbl in povm:
                s += "    %s = %g\n" % \
                     (lbl, vecdist(povm[lbl].to_dense(), other_model.povms[povm_lbl][lbl].to_dense()))

        s += " Gates:\n"
        for lbl in self.operations:
            s += "  %s = %g\n" % \
                (str(lbl), dist(self.operations[lbl].to_dense(), other_model.operations[lbl].to_dense()))

        if len(self.instruments) > 0:
            s += " Instruments:\n"
            for inst_lbl, inst in self.instruments.items():
                s += "  %s: " % str(inst_lbl)
                for lbl in inst:
                    s += "    %s = %g\n" % (str(lbl), dist(
                        inst[lbl].to_dense(), other_model.instruments[inst_lbl][lbl].to_dense()))

        #Note: no way to different factories easily

        return s

    def _init_copy(self, copy_into, memo):
        """
        Copies any "tricky" member of this model into `copy_into`, before
        deep copying everything else within a .copy() operation.
        """

        # Copy special base class members first
        super(ExplicitOpModel, self)._init_copy(copy_into, memo)

        # Copy our "tricky" members
        copy_into.preps = self.preps.copy(copy_into, memo)
        copy_into.povms = self.povms.copy(copy_into, memo)
        copy_into.operations = self.operations.copy(copy_into, memo)
        copy_into.instruments = self.instruments.copy(copy_into, memo)
        copy_into.factories = self.factories.copy(copy_into, memo)
        copy_into._default_gauge_group = self._default_gauge_group  # Note: SHALLOW copy

    def __str__(self):
        s = ""
        for lbl, vec in self.preps.items():
            s += "%s = " % str(lbl) + str(vec) + "\n"
        s += "\n"
        for lbl, povm in self.povms.items():
            s += "%s = " % str(lbl) + str(povm) + "\n"
        s += "\n"
        for lbl, gate in self.operations.items():
            s += "%s = \n" % str(lbl) + str(gate) + "\n\n"
        for lbl, inst in self.instruments.items():
            s += "%s = " % str(lbl) + str(inst) + "\n"
        for lbl, factory in self.factories.items():
            s += "%s = (factory)" % lbl + '\n'
        s += "\n"

        return s

    def all_objects(self):
        """
        Iterate over all of the (label, operator object) entities in this model.

        This iterator runs over all state preparations, POVMS, operations,
        and instruments.
        """
        for lbl, obj in _itertools.chain(self.preps.items(),
                                         self.povms.items(),
                                         self.operations.items(),
                                         self.instruments.items(),
                                         self.factories.items()):
            yield (lbl, obj)

#TODO: how to handle these given possibility of different parameterizations...
#  -- maybe only allow these methods to be called when using a "full" parameterization?
#  -- or perhaps better to *move* them to the parameterization class
    def depolarize(self, op_noise=None, spam_noise=None, max_op_noise=None,
                   max_spam_noise=None, seed=None):
        """
        Apply depolarization uniformly or randomly to this model's gate and/or SPAM elements.

        The result is returned without modifying the original (this) model.  You
        must specify either `op_noise` or `max_op_noise` (for the amount of gate
        depolarization), and either `spam_noise` or `max_spam_noise` (for spam
        depolarization).

        Parameters
        ----------
        op_noise : float, optional
            apply depolarizing noise of strength ``1-op_noise`` to all gates in
             the model. (Multiplies each assumed-Pauli-basis operation matrix by the
             diagonal matrix with ``(1.0-op_noise)`` along all the diagonal
             elements except the first (the identity).

        spam_noise : float, optional
            apply depolarizing noise of strength ``1-spam_noise`` to all SPAM
            vectors in the model. (Multiplies the non-identity part of each
            assumed-Pauli-basis state preparation vector and measurement vector
            by ``(1.0-spam_noise)``).

        max_op_noise : float, optional
            specified instead of `op_noise`; apply a random depolarization
            with maximum strength ``1-max_op_noise`` to each gate in the
            model.

        max_spam_noise : float, optional
            specified instead of `spam_noise`; apply a random depolarization
            with maximum strength ``1-max_spam_noise`` to SPAM vector in the
            model.

        seed : int, optional
            if not ``None``, seed numpy's random number generator with this value
            before generating random depolarizations.

        Returns
        -------
        Model
            the depolarized Model
        """
        newModel = self.copy()  # start by just copying the current model
        rndm = _np.random.RandomState(seed)

        if max_op_noise is not None:
            if op_noise is not None:
                raise ValueError("Must specify at most one of 'op_noise' and 'max_op_noise' NOT both")

            #Apply random depolarization to each gate
            r = max_op_noise * rndm.random_sample(len(self.operations))
            for i, label in enumerate(self.operations):
                newModel.operations[label].depolarize(r[i])
            r = max_op_noise * rndm.random_sample(len(self.instruments))
            for i, label in enumerate(self.instruments):
                newModel.instruments[label].depolarize(r[i])
            r = max_op_noise * rndm.random_sample(len(self.factories))
            for i, label in enumerate(self.factories):
                newModel.factories[label].depolarize(r[i])

        elif op_noise is not None:
            #Apply the same depolarization to each gate
            for label in self.operations:
                newModel.operations[label].depolarize(op_noise)
            for label in self.instruments:
                newModel.instruments[label].depolarize(op_noise)
            for label in self.factories:
                newModel.factories[label].depolarize(op_noise)

        if max_spam_noise is not None:
            if spam_noise is not None:
                raise ValueError("Must specify at most  one of 'noise' and 'max_noise' NOT both")

            #Apply random depolarization to each rho and E vector
            r = max_spam_noise * rndm.random_sample(len(self.preps))
            for (i, lbl) in enumerate(self.preps):
                newModel.preps[lbl].depolarize(r[i])
            r = max_spam_noise * rndm.random_sample(len(self.povms))
            for label in self.povms:
                newModel.povms[label].depolarize(r[i])

        elif spam_noise is not None:
            #Apply the same depolarization to each gate
            for lbl in self.preps:
                newModel.preps[lbl].depolarize(spam_noise)

            # Just depolarize the preps - leave POVMs alone
            #for label in self.povms:
            #    newModel.povms[label].depolarize(spam_noise)

        newModel._clean_paramvec()  # depolarize may leave dirty members
        return newModel

    def rotate(self, rotate=None, max_rotate=None, seed=None):
        """
        Apply a rotation uniformly or randomly to this model.

        Uniformly means the same rotation applied to each gate and
        randomly means different random rotations are applied to each gate of
        this model.  The result is returned without modifying the original (this) model.

        You must specify either `rotate` or `max_rotate`. This method currently
        only works on n-qubit models.

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
            corresponding rotation, to each gate in this `Model`.  Each
            component of each tuple is drawn uniformly from [0, `max_rotate`).

        seed : int, optional
            if  not None, seed numpy's random number generator with this value
            before generating random depolarizations.

        Returns
        -------
        Model
            the rotated Model
        """
        newModel = self.copy()  # start by just copying model
        dim = self.dim
        myBasis = self.basis

        if max_rotate is not None:
            if rotate is not None:
                raise ValueError("Must specify exactly one of 'rotate' and 'max_rotate' NOT both")

            #Apply random rotation to each gate
            rndm = _np.random.RandomState(seed)
            r = max_rotate * rndm.random_sample(len(self.operations) * (dim - 1))
            for i, label in enumerate(self.operations):
                rot = _np.array(r[(dim - 1) * i:(dim - 1) * (i + 1)])
                newModel.operations[label].rotate(rot, myBasis)
            r = max_rotate * rndm.random_sample(len(self.instruments) * (dim - 1))
            for i, label in enumerate(self.instruments):
                rot = _np.array(r[(dim - 1) * i:(dim - 1) * (i + 1)])
                newModel.instruments[label].rotate(rot, myBasis)
            r = max_rotate * rndm.random_sample(len(self.factories) * (dim - 1))
            for i, label in enumerate(self.factories):
                rot = _np.array(r[(dim - 1) * i:(dim - 1) * (i + 1)])
                newModel.factories[label].rotate(rot, myBasis)

        elif rotate is not None:
            assert(len(rotate) == dim - 1), \
                "Invalid 'rotate' argument. You must supply a tuple of length %d" % (dim - 1)
            for label in self.operations:
                newModel.operations[label].rotate(rotate, myBasis)
            for label in self.instruments:
                newModel.instruments[label].rotate(rotate, myBasis)
            for label in self.factories:
                newModel.factories[label].rotate(rotate, myBasis)

        else: raise ValueError("Must specify either 'rotate' or 'max_rotate' "
                               + "-- neither was non-None")

        newModel._clean_paramvec()  # rotate may leave dirty members
        return newModel

    def randomize_with_unitary(self, scale, seed=None, rand_state=None):
        """
        Create a new model with random unitary perturbations.

        Apply a random unitary to each element of a model, and return the
        result, without modifying the original (this) model. This method
        works on Model as long as the dimension is a perfect square.

        Parameters
        ----------
        scale : float
            maximum element magnitude in the generator of each random unitary
            transform.

        seed : int, optional
            if not None, seed numpy's random number generator with this value
            before generating random depolarizations.

        rand_state : numpy.random.RandomState
            A RandomState object to generate samples from. Can be useful to set
            instead of `seed` if you want reproducible distribution samples
            across multiple random function calls but you don't want to bother
            with manually incrementing seeds between those calls.

        Returns
        -------
        Model
            the randomized Model
        """
        if rand_state is None:
            rndm = _np.random.RandomState(seed)
        else:
            rndm = rand_state

        op_dim = self.dim
        unitary_dim = int(round(_np.sqrt(op_dim)))
        assert(unitary_dim**2 == op_dim), \
            "Model dimension must be a perfect square, %d is not" % op_dim

        mdl_randomized = self.copy()

        for opLabel, gate in self.operations.items():
            randMat = scale * (rndm.randn(unitary_dim, unitary_dim)
                               + 1j * rndm.randn(unitary_dim, unitary_dim))
            randMat = _np.transpose(_np.conjugate(randMat)) + randMat
            # make randMat Hermetian: (A_dag + A)^dag = (A_dag + A)
            randUnitary = _scipy.linalg.expm(-1j * randMat)

            randOp = _gt.unitary_to_process_mx(randUnitary)  # in std basis
            randOp = _bt.change_basis(randOp, "std", self.basis)

            mdl_randomized.operations[opLabel] = _op.FullDenseOp(
                _np.dot(randOp, gate))

        #Note: this function does NOT randomize instruments

        return mdl_randomized

    def increase_dimension(self, new_dimension):
        """
        Enlarge the dimension of this model.

        Enlarge the spam vectors and operation matrices of model to a specified
        dimension, and return the resulting inflated model.  Spam vectors
        are zero-padded and operation matrices are padded with 1's on the diagonal
        and zeros on the off-diagonal (effectively padded by identity operation).

        Parameters
        ----------
        new_dimension : int
            the dimension of the returned model.  That is,
            the returned model will have rho and E vectors that
            have shape (new_dimension,1) and operation matrices with shape
            (new_dimension,new_dimension)

        Returns
        -------
        Model
            the increased-dimension Model
        """

        curDim = self.dim
        assert(new_dimension > curDim)

        #For now, just create a dumb default state space labels and basis for the new model:
        sslbls = [('L%d' % i,) for i in range(new_dimension)]  # interpret as independent classical levels
        dumb_basis = _DirectSumBasis([_BuiltinBasis('gm', 1)] * new_dimension,
                                     name="Unknown")  # - just act on diagonal density mx
        new_model = ExplicitOpModel(sslbls, dumb_basis, "full", self.preps._prefix, self.effects_prefix,
                                    self.operations._prefix, self.povms._prefix,
                                    self.instruments._prefix, self._sim.copy())
        #new_model._dim = new_dimension # dim will be set when elements are added
        #new_model.reset_basis() #FUTURE: maybe user can specify how increase is being done?

        addedDim = new_dimension - curDim
        vec_zeroPad = _np.zeros((addedDim, 1), 'd')

        #Increase dimension of rhoVecs and EVecs by zero-padding
        for lbl, rhoVec in self.preps.items():
            assert(len(rhoVec) == curDim)
            new_model.preps[lbl] = \
                _sv.FullSPAMVec(_np.concatenate((rhoVec, vec_zeroPad)))

        for lbl, povm in self.povms.items():
            assert(povm.dim == curDim)
            effects = [(elbl, _np.concatenate((EVec, vec_zeroPad)))
                       for elbl, EVec in povm.items()]

            if isinstance(povm, _povm.TPPOVM):
                new_model.povms[lbl] = _povm.TPPOVM(effects)
            else:
                new_model.povms[lbl] = _povm.UnconstrainedPOVM(effects)  # everything else

        #Increase dimension of gates by assuming they act as identity on additional (unknown) space
        for opLabel, gate in self.operations.items():
            assert(gate.shape == (curDim, curDim))
            newOp = _np.zeros((new_dimension, new_dimension))
            newOp[0:curDim, 0:curDim] = gate[:, :]
            for i in range(curDim, new_dimension): newOp[i, i] = 1.0
            new_model.operations[opLabel] = _op.FullDenseOp(newOp)

        for instLabel, inst in self.instruments.items():
            inst_ops = []
            for outcomeLbl, gate in inst.items():
                newOp = _np.zeros((new_dimension, new_dimension))
                newOp[0:curDim, 0:curDim] = gate[:, :]
                for i in range(curDim, new_dimension): newOp[i, i] = 1.0
                inst_ops.append((outcomeLbl, _op.FullDenseOp(newOp)))
            new_model.instruments[instLabel] = _instrument.Instrument(inst_ops)

        if len(self.factories) > 0:
            raise NotImplementedError("Changing dimension of models with factories is not supported yet!")

        return new_model

    def _decrease_dimension(self, new_dimension):
        """
        Decrease the dimension of this model.

        Shrink the spam vectors and operation matrices of model to a specified
        dimension, and return the resulting model.

        Parameters
        ----------
        new_dimension : int
            the dimension of the returned model.  That is,
            the returned model will have rho and E vectors that
            have shape (new_dimension,1) and operation matrices with shape
            (new_dimension,new_dimension)

        Returns
        -------
        Model
            the decreased-dimension Model
        """
        curDim = self.dim
        assert(new_dimension < curDim)

        #For now, just create a dumb default state space labels and basis for the new model:
        sslbls = [('L%d' % i,) for i in range(new_dimension)]  # interpret as independent classical levels
        dumb_basis = _DirectSumBasis([_BuiltinBasis('gm', 1)] * new_dimension,
                                     name="Unknown")  # - just act on diagonal density mx
        new_model = ExplicitOpModel(sslbls, dumb_basis, "full", self.preps._prefix, self.effects_prefix,
                                    self.operations._prefix, self.povms._prefix,
                                    self.instruments._prefix, self._sim.copy())
        #new_model._dim = new_dimension # dim will be set when elements are added
        #new_model.reset_basis() #FUTURE: maybe user can specify how decrease is being done?

        #Decrease dimension of rhoVecs and EVecs by truncation
        for lbl, rhoVec in self.preps.items():
            assert(len(rhoVec) == curDim)
            new_model.preps[lbl] = \
                _sv.FullSPAMVec(rhoVec[0:new_dimension, :])

        for lbl, povm in self.povms.items():
            assert(povm.dim == curDim)
            effects = [(elbl, EVec[0:new_dimension, :]) for elbl, EVec in povm.items()]

            if isinstance(povm, _povm.TPPOVM):
                new_model.povms[lbl] = _povm.TPPOVM(effects)
            else:
                new_model.povms[lbl] = _povm.UnconstrainedPOVM(effects)  # everything else

        #Decrease dimension of gates by truncation
        for opLabel, gate in self.operations.items():
            assert(gate.shape == (curDim, curDim))
            newOp = _np.zeros((new_dimension, new_dimension))
            newOp[:, :] = gate[0:new_dimension, 0:new_dimension]
            new_model.operations[opLabel] = _op.FullDenseOp(newOp)

        for instLabel, inst in self.instruments.items():
            inst_ops = []
            for outcomeLbl, gate in inst.items():
                newOp = _np.zeros((new_dimension, new_dimension))
                newOp[:, :] = gate[0:new_dimension, 0:new_dimension]
                inst_ops.append((outcomeLbl, _op.FullDenseOp(newOp)))
            new_model.instruments[instLabel] = _instrument.Instrument(inst_ops)

        if len(self.factories) > 0:
            raise NotImplementedError("Changing dimension of models with factories is not supported yet!")

        return new_model

    def kick(self, absmag=1.0, bias=0, seed=None):
        """
        "Kick" this model by adding to each gate a random matrix.

        The random matrices have values uniformly distributed in the interval
        [bias-absmag,bias+absmag].

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
        Model
            the kicked model.
        """
        kicked_gs = self.copy()
        rndm = _np.random.RandomState(seed)
        for opLabel, gate in self.operations.items():
            delta = absmag * 2.0 * (rndm.random_sample(gate.shape) - 0.5) + bias
            kicked_gs.operations[opLabel] = _op.FullDenseOp(
                kicked_gs.operations[opLabel] + delta)

        #Note: does not alter intruments!
        return kicked_gs

    def compute_clifford_symplectic_reps(self, oplabel_filter=None):
        """
        Constructs a dictionary of the symplectic representations for all the Clifford gates in this model.

        Non-:class:`CliffordOp` gates will be ignored and their entries omitted
        from the returned dictionary.

        Parameters
        ----------
        oplabel_filter : iterable, optional
            A list, tuple, or set of operation labels whose symplectic
            representations should be returned (if they exist).

        Returns
        -------
        dict
            keys are operation labels and/or just the root names of gates
            (without any state space indices/labels).  Values are
            `(symplectic_matrix, phase_vector)` tuples.
        """
        gfilter = set(oplabel_filter) if oplabel_filter is not None \
            else None

        srep_dict = {}

        for gl, gate in self.operations.items():
            if (gfilter is not None) and (gl not in gfilter): continue

            if isinstance(gate, _op.EmbeddedOp):
                assert(isinstance(gate.embedded_op, _op.CliffordOp)), \
                    "EmbeddedClifforGate contains a non-CliffordOp!"
                lbl = gl.name  # strip state space labels off since this is a
                # symplectic rep for the *embedded* gate
                srep = (gate.embedded_op.smatrix, gate.embedded_op.svector)
            elif isinstance(gate, _op.CliffordOp):
                lbl = gl.name
                srep = (gate.smatrix, gate.svector)
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
        Print to stdout relevant information about this model.

        This information includes the Choi matrices and their eigenvalues.

        Returns
        -------
        None
        """
        print(self)
        print("\n")
        print("Basis = ", self.basis.name)
        print("Choi Matrices:")
        for (label, gate) in self.operations.items():
            print(("Choi(%s) in pauli basis = \n" % label,
                   _mt.mx_to_string_complex(_jt.jamiolkowski_iso(gate))))
            print(("  --eigenvals = ", sorted(
                [ev.real for ev in _np.linalg.eigvals(
                    _jt.jamiolkowski_iso(gate))]), "\n"))
        print(("Sum of negative Choi eigenvalues = ", _jt.sum_of_negative_choi_eigenvalues(self)))

    def _effect_labels_for_povm(self, povm_lbl):
        """
        Gets the effect labels corresponding to the possible outcomes of POVM label `povm_lbl`.

        Parameters
        ----------
        povm_lbl : Label
            POVM label.

        Returns
        -------
        list
            A list of strings which label the POVM outcomes.
        """
        return tuple(self.povms[povm_lbl].keys())

    def _member_labels_for_instrument(self, inst_lbl):
        """
        Gets the member labels corresponding to the possible outcomes of the instrument labeled by `inst_lbl`.

        Parameters
        ----------
        inst_lbl : Label
            Instrument label.

        Returns
        -------
        list
            A list of strings which label the instrument members.
        """
        return tuple(self.instruments[inst_lbl].keys())

    def _reinit_opcaches(self):
        self._opcaches.clear()

        # Add expanded instrument and POVM operations to cache so these are accessible to circuit calcs
        simplified_effects = _collections.OrderedDict()
        for povm_lbl, povm in self.povms.items():
            for k, e in povm.simplify_effects(povm_lbl).items():
                simplified_effects[k] = e

        simplified_ops = _collections.OrderedDict()
        for inst_lbl, inst in self.instruments.items():
            for k, g in inst.simplify_operations(inst_lbl).items():
                simplified_ops[k] = g

        self._opcaches['povm-layers'] = simplified_effects
        self._opcaches['op-layers'] = simplified_ops

    # Gauge invariant errorgens and/or parameters support (initial & experimental)
    def _construct_gauge_space_for_model(self, primitive_op_labels, ham_basis, other_basis, other_mode="all",
                                         reduce_to_model_space=True):
        ham_elem_labels = [('H', bel) for bel in ham_basis.labels[1:]]
        other_elem_labels = [('S', bel) for bel in other_basis.labels[1:]] if other_mode != "all" else \
            [('S', bel1, bel2) for bel1 in other_basis.labels[1:] for bel2 in other_basis.labels[1:]]

        ham_gauge_action_mxs = []
        other_gauge_action_mxs = []

        for op_label in primitive_op_labels:  # Note: "ga" stands for "gauge action" in variable names below
            op = self.operations[op_label]
            op_mx = op.unitary_postfactor.to_dense()
            U = _bt.change_basis(op_mx, self.basis, 'std')
            ham_ga = _gt.first_order_ham_gauge_action_matrix(U, ham_basis)
            other_ga = _gt.first_order_other_gauge_action_matrix(U, other_basis, other_mode)
            ham_gauge_action_mxs.append(ham_ga)
            other_gauge_action_mxs.append(other_ga)

        ham_gauge_action_dict = {lbl: amx for lbl, amx in zip(primitive_op_labels, ham_gauge_action_mxs)}
        other_gauge_action_dict = {lbl: amx for lbl, amx in zip(primitive_op_labels, other_gauge_action_mxs)}

        #Stack matrices to form "base" gauge action matrix for op_set
        ham_ga_mx = _np.concatenate(ham_gauge_action_mxs, axis=0)
        other_ga_mx = _np.concatenate(other_gauge_action_mxs, axis=0)

        # Intersect gauge action with the space of elementary errorgens present in the model.
        # We may need to eliminate some rows of X_ga matrices, and (only) keep linear combos
        # of the columns that are zero on these rows.
        present_ham_elem_lbls = set()
        present_other_elem_lbls = set()
        for op_label in primitive_op_labels:
            op = self.operations[op_label]
            lbls = op.errorgen_coefficient_labels()  # length num_coeffs
            present_ham_elem_lbls.update([(op_label, lbl) for lbl in lbls if lbl[0] == 'H'])
            present_other_elem_lbls.update([(op_label, lbl) for lbl in lbls if lbl[0] == 'S'])

        full_ham_elem_labels = [(op_label, elem_lbl) for op_label in primitive_op_labels
                                for elem_lbl in ham_elem_labels]
        assert(present_ham_elem_lbls.issubset(full_ham_elem_labels)), \
            "The given space of hamiltonian elementary gauge-gens must encompass all those in model ops!"
        disallowed_ham_space_labels = set(full_ham_elem_labels) - present_ham_elem_lbls
        disallowed_row_indices = [full_ham_elem_labels.index(disallowed_lbl)
                                  for disallowed_lbl in disallowed_ham_space_labels]

        if reduce_to_model_space and len(disallowed_row_indices) > 0:
            disallowed_rows = _np.take(ham_ga_mx, disallowed_row_indices, axis=0)
            allowed_ham_linear_combos = _mt.nice_nullspace(disallowed_rows, tol=1e-4)
            ham_ga_mx = _np.delete(ham_ga_mx, disallowed_row_indices, axis=0)
        else:
            allowed_ham_linear_combos = None

        full_other_elem_labels = [(op_label, elem_lbl) for op_label in primitive_op_labels
                                  for elem_lbl in other_elem_labels]
        assert(present_other_elem_lbls.issubset(full_other_elem_labels)), \
            "The given space of 'other' elementary gauge-gens must encompass all those in model ops!"
        disallowed_other_space_labels = set(full_other_elem_labels) - present_other_elem_lbls
        disallowed_row_indices = [full_other_elem_labels.index(disallowed_lbl)
                                  for disallowed_lbl in disallowed_other_space_labels]

        if reduce_to_model_space and len(disallowed_row_indices) > 0:
            disallowed_rows = _np.take(other_ga_mx, disallowed_row_indices, axis=0)
            allowed_other_linear_combos = _mt.nice_nullspace(disallowed_rows, tol=1e-4)
            other_ga_mx = _np.delete(other_ga_mx, disallowed_row_indices, axis=0)
        else:
            allowed_other_linear_combos = None

        #HACK: remove spam "other" gauge direction for now
        #if allowed_other_linear_combos is not None:
        #    spamGauge = _np.ones((allowed_other_linear_combos.shape[0], 1), 'd')
        #    test = _np.concatenate((allowed_other_linear_combos, spamGauge), axis=1)
        #    if _np.linalg.matrix_rank(test) < test.shape[1]:
        #        overlaps = _np.abs(_np.dot(spamGauge.T, allowed_other_linear_combos))
        #        allowed_other_linear_combos = _np.delete(allowed_other_linear_combos, [_np.argmax(overlaps)], axis=1)
        #        print("Removed SPAM gauge direction")

        return ham_ga_mx, other_ga_mx, allowed_ham_linear_combos, allowed_other_linear_combos, \
            ham_gauge_action_dict, other_gauge_action_dict

    def _compute_fogi_constructively(self, primitive_op_labels, basis, mode=None,
                                     gauge_linear_combos=None, op_label_abbrevs=None, reduce_to_model_space=True,
                                     dependent_fogi_action='drop'):

        assert(dependent_fogi_action in ('drop', 'mark'))
        typ = 'H' if (mode is None) else 'S'
        if typ == 'H':
            elem_labels = [('H', bel) for bel in basis.labels[1:]]
        else:
            elem_labels = [('S', bel) for bel in basis.labels[1:]] if mode != "all" else \
                [('S', bel1, bel2) for bel1 in basis.labels[1:] for bel2 in basis.labels[1:]]

        #Get lists of the present (existing within the model) labels for each operation
        labels_for_op = {op_label: elem_labels[:] for op_label in primitive_op_labels}  # COPY lists!
        disallowed_row_indices = {}
        if op_label_abbrevs is None: op_label_abbrevs = {}

        if reduce_to_model_space:
            for op_label in primitive_op_labels:
                op = self.operations[op_label]
                lbls = op.errorgen_coefficient_labels()
                present_elem_lbls = set(filter(lambda lbl: lbl[0] == typ, lbls))
                disallowed_labels = set(elem_labels) - present_elem_lbls
                disallowed_row_indices[op_label] = [elem_labels.index(disallowed_lbl)
                                                    for disallowed_lbl in disallowed_labels]
                for i in sorted(disallowed_row_indices[op_label], reverse=True):
                    del labels_for_op[op_label][i]

        gauge_elemgen_labels = elem_labels

        op_errgen_indices = {}; off = 0
        for op_label in primitive_op_labels:
            op_errgen_indices[op_label] = slice(off, off + len(labels_for_op[op_label]))
            off += len(labels_for_op[op_label])
        num_elem_errgens = off

        #Step 1: construct FOGI quantities and reference frame for each op
        ccomms = {}
        gauge_action_for_op = {}
        fogi_vecs = _np.zeros((num_elem_errgens, 0), 'd')
        fogi_names = []
        fogi_abbrev_names = []
        fogi_gauge_dirs = []
        dependent_vec_indices = []

        for op_label in primitive_op_labels:
            #print("##", op_label)
            op = self.operations[op_label]
            op_mx = op.unitary_postfactor.to_dense()
            U = _bt.change_basis(op_mx, self.basis, 'std')
            ga = _gt.first_order_ham_gauge_action_matrix(U, basis) if typ == 'H' else \
                _gt.first_order_other_gauge_action_matrix(U, basis, other_mode=mode)

            if gauge_linear_combos is not None:
                ga = _np.dot(ga, gauge_linear_combos)

            if disallowed_row_indices.get(op_label, None) is not None:
                assert(_np.linalg.norm(_np.take(ga, disallowed_row_indices[op_label], axis=0)) < 1e-6)
                ga = _np.delete(ga, disallowed_row_indices[op_label], axis=0)

            #Get commutant and communtant-complement spaces
            commutant = _mt.nice_nullspace(ga)  # columns = *gauge* elem gen directions
            op_elemgen_labels = labels_for_op[op_label]

            # Note: local/new_fogi_vecs are orthogonal but not necessarily normalized (so need to
            #  normalize to get inverse, but *don't* need pseudo-inverse).
            local_fogi_vecs = _mt.nice_nullspace(ga.T)  # "conjugate space" to gauge action
            new_fogi_vecs = _np.zeros((fogi_vecs.shape[0], local_fogi_vecs.shape[1]), 'd')
            new_fogi_vecs[op_errgen_indices[op_label], :] = local_fogi_vecs  # "juice" this op
            fogi_vecs = _np.concatenate((fogi_vecs, new_fogi_vecs), axis=1)
            fogi_gauge_dirs.extend([None] * new_fogi_vecs.shape[1])  # local qtys don't have corresp. gauge directions
            errgen_names = _gauge_names(local_fogi_vecs, op_elemgen_labels)
            errgen_names_abbrev = _gauge_names(local_fogi_vecs, op_elemgen_labels, abbrev=True)
            fogi_names.extend(["%s_%s" % ((("(%s)" % egname) if (' ' in egname) else egname),
                                          op_label_abbrevs.get(op_label, str(op_label)))
                               for egname in errgen_names])
            fogi_abbrev_names.extend(errgen_names_abbrev)

            complement = _mt.nice_nullspace(commutant.T)  # complement of commutant - where op if faithful rep
            ccomms[(op_label,)] = complement
            gauge_action_for_op[op_label] = ga

            #print("Commutant:"); _mt.print_mx(commutant)
            #print("Names: ", errgen_names)
            #print("Complement:"); _mt.print_mx(complement)

        smaller_sets = [(op_label,) for op_label in primitive_op_labels]
        max_size = len(primitive_op_labels)
        num_indep_fogi_vecs = fogi_vecs.shape[1]
        for set_size in range(1, max_size):
            larger_sets = []
            num_indep_vecs_from_smaller_sets = num_indep_fogi_vecs
            num_vecs_from_smaller_sets = fogi_vecs.shape[1]
            for op_label in primitive_op_labels:
                for existing_set in smaller_sets:
                    if op_label in existing_set: continue
                    new_set = tuple(sorted(existing_set + (op_label,)))
                    if new_set in larger_sets: continue

                    # print("\n##", existing_set, "+", op_label)

                    # Merge existing set + op_label => new set of larger size
                    ccommA = ccomms.get(existing_set, None)  # Note: commutant-complements are in *gauge* space,
                    ccommB = ccomms[(op_label,)]  # so they're all the same dimension.

                    if ccommA is not None and ccommA.shape[1] > 0 and ccommB.shape[1] > 0:
                        # merging with an empty complement does nothing (no intersection, same ccomm)
                        intersection_space = _mt.intersection_space(ccommA, ccommB)
                        union_space = _mt.union_space(ccommA, ccommB)

                        if intersection_space.shape[1] > 0:
                            # Then each basis vector of the intersection space defines a gauge-invariant ("fogi")
                            # direction via the difference between that gauge direction's action on A and B:
                            gauge_action = _np.concatenate([gauge_action_for_op[ol] for ol in existing_set]
                                                           + [gauge_action_for_op[op_label]], axis=0)
                            n = sum([gauge_action_for_op[ol].shape[0] for ol in existing_set])  # boundary btwn A & B
                            inv_diff_gauge_action = _np.concatenate((_np.linalg.pinv(gauge_action[0:n, :], rcond=1e-7),
                                                                     -_np.linalg.pinv(gauge_action[n:, :], rcond=1e-7)),
                                                                    axis=1).T
                            local_fogi_vecs = _np.dot(inv_diff_gauge_action, intersection_space)
                            assert(_np.linalg.norm(_np.dot(gauge_action.T, local_fogi_vecs)) < 1e-8)

                            new_fogi_vecs = _np.zeros((fogi_vecs.shape[0], local_fogi_vecs.shape[1]), 'd'); off = 0
                            for ol in existing_set + (op_label,):  # NOT new_set here b/c concat order below
                                n = len(labels_for_op[ol])
                                new_fogi_vecs[op_errgen_indices[ol], :] = local_fogi_vecs[off:off + n, :]; off += n

                            indep_cols = []  # debug = []
                            if dependent_fogi_action == "drop":
                                for j in range(new_fogi_vecs.shape[1]):
                                    test = _np.concatenate((fogi_vecs, new_fogi_vecs[:, j:j + 1]), axis=1)
                                    if _np.linalg.matrix_rank(test, tol=1e-7) == num_indep_fogi_vecs + 1:
                                        indep_cols.append(j)
                                        fogi_vecs = test
                                        num_indep_fogi_vecs += 1
                                        #debug.append("IND")
                                    #else:
                                    #    debug.append('-')
                            elif dependent_fogi_action == "mark":
                                for j in range(new_fogi_vecs.shape[1]):
                                    test = _np.concatenate((fogi_vecs[:, 0:num_vecs_from_smaller_sets],
                                                            new_fogi_vecs[:, j:j + 1]), axis=1)
                                    test2 = _np.concatenate((fogi_vecs, new_fogi_vecs[:, j:j + 1]), axis=1)
                                    U, s, Vh = _np.linalg.svd(test2)
                                    if _np.linalg.matrix_rank(test2, tol=1e-7) == num_indep_fogi_vecs + 1:
                                        # new vec is indep w/everything
                                        indep_cols.append(j)
                                        fogi_vecs = test2
                                        num_indep_fogi_vecs += 1
                                        #debug.append("IND")
                                    elif _np.linalg.matrix_rank(test, tol=1e-7) == num_indep_vecs_from_smaller_sets + 1:
                                        # new vec is indep w/fogi vecs from smaller sets, so dependency must just be
                                        # among other vecs for this same size.  Which vecs we keep is arbitrary here,
                                        # so keep this vec and "mark" it as a linearly dependent vec.
                                        indep_cols.append(j)  # keep this vec
                                        fogi_vecs = test2
                                        dependent_vec_indices.append(fogi_vecs.shape[1] - 1)  # but mark it
                                        #debug.append("DEP")
                                    #else:
                                    #    debug.append('-')
                                    # else new vec is dependent on *smaller* size fogi vecs - omit it then
                            #print("DEBUG: ",debug)  # TODO: REMOVE this and 'debug' uses above

                            indep_intersection_space = _np.take(intersection_space, indep_cols, axis=1)
                            indep_intersection_space = _np.dot(gauge_linear_combos, indep_intersection_space) \
                                if (gauge_linear_combos is not None) else indep_intersection_space
                            intersection_names = _gauge_names(indep_intersection_space, gauge_elemgen_labels)
                            intersection_names_abbrev = _gauge_names(indep_intersection_space, gauge_elemgen_labels,
                                                                     abbrev=True)
                            fogi_names.extend(["ga(%s)_%s - ga(%s)_%s" % (
                                iname, "|".join([op_label_abbrevs.get(l, str(l)) for l in existing_set]),
                                iname, op_label_abbrevs.get(op_label, str(op_label))) for iname in intersection_names])
                            fogi_abbrev_names.extend(["ga(%s)" % iname for iname in intersection_names_abbrev])
                            fogi_gauge_dirs.extend([intersection_space[:, j] for j in indep_cols])  # gauge directions

                            #print("Fogi vecs:\n"); _mt.print_mx(local_fogi_vecs)
                            #print("Ham Intersection names: ", intersection_names)

                        ccomms[new_set] = union_space
                        #print("Complement:\n"); _mt.print_mx(union_space)

                    larger_sets.append(new_set)

            smaller_sets = larger_sets

        #big_gauge_action = _np.concatenate([other_gauge_action[ol] for ol in primitive_op_labels], axis=0)  # DEBUG

        #print("Ham Fogi vecs:\n"); _mt.print_mx(ham_fogi_vecs, width=5, prec=1)
        #print("Ham Names = \n", '\n'.join(["%d: %s" % (i, v) for i, v in enumerate(ham_fogi_names)]))
        #print("Rank = ", _np.linalg.matrix_rank(ham_fogi_vecs))

        #print("Other Fogi vecs:\n"); _mt.print_mx(other_fogi_vecs, width=5, prec=1)
        #print("Other Names = \n", '\n'.join(other_fogi_names))
        #print("Rank = ", _np.linalg.matrix_rank(other_fogi_vecs))
        #import bpdb; bpdb.set_trace()  # DEBUG

        return (fogi_vecs, fogi_names, fogi_abbrev_names, labels_for_op, dependent_vec_indices,
                fogi_gauge_dirs, op_errgen_indices)

    def _compute_fogi_via_nullspaces(self, primitive_op_labels, ham_basis, other_basis, other_mode="all",
                                     ham_gauge_linear_combos=None, other_gauge_linear_combos=None,
                                     op_label_abbrevs=None, reduce_to_model_space=True):
        num_ham_elem_errgens = (len(ham_basis) - 1)
        num_other_elem_errgens = (len(other_basis) - 1)**2 if other_mode == "all" else (len(other_basis) - 1)
        ham_elem_labels = [('H', bel) for bel in ham_basis.labels[1:]]
        other_elem_labels = [('S', bel) for bel in other_basis.labels[1:]] if other_mode != "all" else \
            [('S', bel1, bel2) for bel1 in other_basis.labels[1:] for bel2 in other_basis.labels[1:]]
        assert(len(ham_elem_labels) == num_ham_elem_errgens)
        assert(len(other_elem_labels) == num_other_elem_errgens)

        #Get lists of the present (existing within the model) labels for each operation
        ham_labels_for_op = {op_label: ham_elem_labels[:] for op_label in primitive_op_labels}  # COPY lists!
        other_labels_for_op = {op_label: other_elem_labels[:] for op_label in primitive_op_labels}  # ditto
        if reduce_to_model_space:
            for op_label in primitive_op_labels:
                op = self.operations[op_label]
                lbls = op.errorgen_coefficient_labels()
                present_ham_elem_lbls = set(filter(lambda lbl: lbl[0] == 'H', lbls))
                present_other_elem_lbls = set(filter(lambda lbl: lbl[0] == 'S', lbls))

                disallowed_ham_space_labels = set(ham_elem_labels) - present_ham_elem_lbls
                disallowed_row_indices = [ham_elem_labels.index(disallowed_lbl)
                                          for disallowed_lbl in disallowed_ham_space_labels]
                for i in sorted(disallowed_row_indices, reverse=True):
                    del ham_labels_for_op[op_label][i]

                disallowed_other_space_labels = set(other_elem_labels) - present_other_elem_lbls
                disallowed_row_indices = [other_elem_labels.index(disallowed_lbl)
                                          for disallowed_lbl in disallowed_other_space_labels]
                for i in sorted(disallowed_row_indices, reverse=True):
                    del other_labels_for_op[op_label][i]

        #Step 1: construct nullspaces associated with sets of operations
        ham_nullspaces = {}
        other_nullspaces = {}
        max_size = len(primitive_op_labels)
        for set_size in range(1, max_size + 1):
            ham_nullspaces[set_size] = {}  # dict mapping operation-sets of `set_size` to nullspaces
            other_nullspaces[set_size] = {}

            for op_set in _itertools.combinations(primitive_op_labels, set_size):
                #print(op_set)
                ham_gauge_action_mxs = []
                other_gauge_action_mxs = []
                ham_rows_by_op = {}; h_off = 0
                other_rows_by_op = {}; o_off = 0
                for op_label in op_set:  # Note: "ga" stands for "gauge action" in variable names below
                    op = self.operations[op_label]
                    if isinstance(op, _op.LindbladOp):
                        op_mx = op.unitary_postfactor.to_dense()
                    else:
                        assert(False), "STOP - you probably don't want to do this!"
                        op_mx = op.to_dense()
                    U = _bt.change_basis(op_mx, self.basis, 'std')
                    ham_ga = _gt.first_order_ham_gauge_action_matrix(U, ham_basis)
                    other_ga = _gt.first_order_other_gauge_action_matrix(U, other_basis, other_mode)

                    if ham_gauge_linear_combos is not None:
                        ham_ga = _np.dot(ham_ga, ham_gauge_linear_combos)
                    if other_gauge_linear_combos is not None:
                        other_ga = _np.dot(other_ga, other_gauge_linear_combos)

                    ham_gauge_action_mxs.append(ham_ga)
                    other_gauge_action_mxs.append(other_ga)
                    reduced_ham_nrows = len(ham_labels_for_op[op_label])  # ham_ga.shape[0] when unrestricted
                    reduced_other_nrows = len(other_labels_for_op[op_label])  # other_ga.shape[0] when unrestricted
                    ham_rows_by_op[op_label] = slice(h_off, h_off + reduced_ham_nrows); h_off += reduced_ham_nrows
                    other_rows_by_op[op_label] = slice(o_off, o_off + reduced_other_nrows); o_off += reduced_other_nrows
                    assert(ham_ga.shape[0] == num_ham_elem_errgens)
                    assert(other_ga.shape[0] == num_other_elem_errgens)

                #Stack matrices to form "base" gauge action matrix for op_set
                ham_ga_mx = _np.concatenate(ham_gauge_action_mxs, axis=0)
                other_ga_mx = _np.concatenate(other_gauge_action_mxs, axis=0)

                # Intersect gauge action with the space of elementary errorgens present in the model.
                # We may need to eliminate some rows of X_ga matrices, and (only) keep linear combos
                # of the columns that are zero on these rows.
                present_ham_elem_lbls = set()
                present_other_elem_lbls = set()
                for op_label in op_set:
                    op = self.operations[op_label]
                    lbls = op.errorgen_coefficient_labels()  # length num_coeffs
                    present_ham_elem_lbls.update([(op_label, lbl) for lbl in lbls if lbl[0] == 'H'])
                    present_other_elem_lbls.update([(op_label, lbl) for lbl in lbls if lbl[0] == 'S'])

                full_ham_elem_labels = [(op_label, elem_lbl) for op_label in op_set
                                        for elem_lbl in ham_elem_labels]
                assert(present_ham_elem_lbls.issubset(full_ham_elem_labels)), \
                    "The given space of hamiltonian elementary gauge-gens must encompass all those in model ops!"
                disallowed_ham_space_labels = set(full_ham_elem_labels) - present_ham_elem_lbls
                disallowed_row_indices = [full_ham_elem_labels.index(disallowed_lbl)
                                          for disallowed_lbl in disallowed_ham_space_labels]

                if reduce_to_model_space and len(disallowed_row_indices) > 0:
                    #disallowed_rows = _np.take(ham_ga_mx, disallowed_row_indices, axis=0)
                    #allowed_linear_combos = _mt.nice_nullspace(disallowed_rows, tol=1e-4)
                    #ham_ga_mx = _np.dot(ham_ga_mx, allowed_linear_combos)
                    ham_ga_mx = _np.delete(ham_ga_mx, disallowed_row_indices, axis=0)

                full_other_elem_labels = [(op_label, elem_lbl) for op_label in op_set
                                          for elem_lbl in other_elem_labels]
                assert(present_other_elem_lbls.issubset(full_other_elem_labels)), \
                    "The given space of 'other' elementary gauge-gens must encompass all those in model ops!"
                disallowed_other_space_labels = set(full_other_elem_labels) - present_other_elem_lbls
                disallowed_row_indices = [full_other_elem_labels.index(disallowed_lbl)
                                          for disallowed_lbl in disallowed_other_space_labels]

                if reduce_to_model_space and len(disallowed_row_indices) > 0:
                    #disallowed_rows = _np.take(other_ga_mx, disallowed_row_indices, axis=0)
                    #allowed_linear_combos = _mt.nice_nullspace(disallowed_rows, tol=1e-4)
                    #other_ga_mx = _np.dot(other_ga_mx, allowed_linear_combos)
                    other_ga_mx = _np.delete(other_ga_mx, disallowed_row_indices, axis=0)

                #Add all known (already tabulated) nullspace directions so that we avoid getting them again
                # when we compute the nullspace of the gauge action matrix below.
                for previous_size in range(1, set_size + 1):  # include current size!
                    for previous_op_set, (nullsp, previous_rows) in ham_nullspaces[previous_size].items():
                        padded_nullsp = _np.zeros((ham_ga_mx.shape[0], nullsp.shape[1]), 'd')
                        for op in previous_op_set:
                            if op not in ham_rows_by_op: continue
                            padded_nullsp[ham_rows_by_op[op], :] = nullsp[previous_rows[op], :]
                        ham_ga_mx = _np.concatenate((ham_ga_mx, padded_nullsp), axis=1)

                    for previous_op_set, (nullsp, previous_rows) in other_nullspaces[previous_size].items():
                        padded_nullsp = _np.zeros((other_ga_mx.shape[0], nullsp.shape[1]), other_ga_mx.dtype)
                        for op in previous_op_set:
                            if op not in other_rows_by_op: continue
                            padded_nullsp[other_rows_by_op[op], :] = nullsp[previous_rows[op], :]
                        other_ga_mx = _np.concatenate((other_ga_mx, padded_nullsp), axis=1)

                #Finally, compute the nullspace of the resulting gauge-action + already-tallied matrix:
                nullspace = _mt.nice_nullspace(ham_ga_mx.T)
                ham_nullspaces[set_size][op_set] = (nullspace, ham_rows_by_op)
                #DEBUG: print("  NULLSP DIM = ",nullspace.shape[1])
                #DEBUG: labels = [(op_label, elem_lbl) for op_label in op_set
                #DEBUG:           for elem_lbl in ham_labels_for_op[op_label]]
                #DEBUG: print("\n".join(fogi_names(nullspace, labels, op_label_abbrevs)))

                nullspace = _mt.nice_nullspace(other_ga_mx.T)
                other_nullspaces[set_size][op_set] = (nullspace, other_rows_by_op)

        # Step 2: convert these per-operation-set nullspaces into vectors over a single "full"
        #  space of all the elementary error generators (as given by ham_basis, other_basis, & other_mode)

        # Note: "full" designation is for space of all elementary error generators as given by their
        #  supplied ham_basis, other_basis, and other_mode.

        # Construct full-space vectors for each nullspace vector found by crawling through
        #  the X_nullspaces dictionary and embedding values as needed.
        ham_rows_by_op = {}; off = 0
        for op_label in primitive_op_labels:
            ham_rows_by_op[op_label] = slice(off, off + len(ham_labels_for_op[op_label]))
            off += len(ham_labels_for_op[op_label])
        full_ham_fogi_vecs = _np.empty((off, 0), 'd')
        for size in range(1, max_size + 1):
            for op_set, (nullsp, op_set_rows) in ham_nullspaces[size].items():
                padded_nullsp = _np.zeros((full_ham_fogi_vecs.shape[0], nullsp.shape[1]), 'd')
                for op in op_set:
                    padded_nullsp[ham_rows_by_op[op], :] = nullsp[op_set_rows[op], :]
                full_ham_fogi_vecs = _np.concatenate((full_ham_fogi_vecs, padded_nullsp), axis=1)

        other_rows_by_op = {}; off = 0
        for op_label in primitive_op_labels:
            other_rows_by_op[op_label] = slice(off, off + len(other_labels_for_op[op_label]))
            off += len(other_labels_for_op[op_label])
        full_other_fogi_vecs = _np.empty((off, 0), complex)
        for size in range(1, max_size + 1):
            for op_set, (nullsp, op_set_rows) in other_nullspaces[size].items():
                padded_nullsp = _np.zeros((full_other_fogi_vecs.shape[0], nullsp.shape[1]), complex)
                for op in op_set:
                    padded_nullsp[other_rows_by_op[op], :] = nullsp[op_set_rows[op], :]
                full_other_fogi_vecs = _np.concatenate((full_other_fogi_vecs, padded_nullsp), axis=1)

        assert(_np.linalg.matrix_rank(full_ham_fogi_vecs) == full_ham_fogi_vecs.shape[1])
        assert(_np.linalg.matrix_rank(full_other_fogi_vecs) == full_other_fogi_vecs.shape[1])

        # Returns the vectors of FOGI (first order gauge invariant) linear combos as well
        # as lists of labels for the columns & rows, respectively.
        return (full_ham_fogi_vecs, ham_labels_for_op), (full_other_fogi_vecs, other_labels_for_op)

    def _recompute_fogi_names(self, op_label_abbrevs=None):
        # Step 4: compute names (by a heuristic) for each FOGI vec
        full_ham_fogi_vecs = self.fogi_info['ham_vecs']
        full_ham_space_labels = self.fogi_info['ham_fullspace_labels']
        full_other_fogi_vecs = self.fogi_info['other_vecs']
        full_other_space_labels = self.fogi_info['other_fullspace_labels']
        self.fogi_info['ham_fogi_labels'] = _fogi_names(full_ham_fogi_vecs, full_ham_space_labels, op_label_abbrevs)
        self.fogi_info['other_fogi_labels'] = _fogi_names(full_other_fogi_vecs, full_other_space_labels,
                                                          op_label_abbrevs)

    def _add_reparameterization(self, primitive_op_labels,
                                full_ham_fogi_vecs, full_ham_space_labels,
                                full_other_fogi_vecs, full_other_space_labels):
        # Create re-parameterization map from "fogi" parameters to old/existing model parameters
        #  MX(fogi_coeffs -> op_coeffs)  e.g. full_ham_fogi_vecs
        #  Deriv(op_params -> op_coeffs)
        #  fogi_deriv(fogi_params -> fogi_coeffs)  - near I (these are nearly identical apart from some squaring?)
        #  so:    d(op_params) = inv(deriv) * MX * fogi_deriv * d(fogi_params)
        #         d(op_params)/d(fogi_params) = inv(Deriv) * MX * fogi_deriv
        #  To first order: op_params = (inv(Deriv) * MX * fogi_deriv) * fogi_params := F * fogi_params
        # (fogi_params == "model params")

        # To compute F,
        # -let fogi_deriv == I   (shape nFogi,nFogi)
        # -MX is shape (nFullSpace, nFogi)
        # -deriv is shape (nOpCoeffs, nOpParams), inv(deriv) = (nOpParams, nOpCoeffs)
        #    - need Deriv of shape (nOpParams, nFullSpace) - build by placing deriv mxs in gpindices rows and
        #      correct cols).  We'll require that deriv be square (op has same #params as coeffs) and is *invertible*
        #      (raise error otherwise).  Then we can construct inv(Deriv) by placing inv(deriv) into inv(Deriv) by
        #      rows->gpindices and cols->elem_label-match.

        nOpParams = self.num_params  # the number of parameters *before* any reparameterization.  TODO: better way?
        full_ham_space_labels_indx = _collections.OrderedDict(
            [(lbl, i) for i, lbl in enumerate(full_ham_space_labels)])
        full_other_space_labels_indx = _collections.OrderedDict(
            [(lbl, i) for i, lbl in enumerate(full_other_space_labels)])

        invDeriv_ham = _np.zeros((nOpParams, full_ham_fogi_vecs.shape[0]), 'd')
        invDeriv_other = _np.zeros((nOpParams, full_other_fogi_vecs.shape[0]), 'd')

        used_param_indices = set()
        for op_label in primitive_op_labels:
            op = self.operations[op_label]
            lbls = op.errorgen_coefficient_labels()  # length num_coeffs
            param_indices = op.gpindices_as_array()  # length num_params
            deriv = op.errorgen_coefficients_array_deriv_wrt_params()  # shape == (num_coeffs, num_params)
            inv_deriv = _np.linalg.inv(deriv)
            used_param_indices.update(param_indices)

            for i, lbl in enumerate(lbls):
                if lbl[0] == 'H':
                    invDeriv_ham[param_indices, full_ham_space_labels_indx[(op_label, lbl)]] = inv_deriv[:, i]
                else:  # lbl[0] == 'S':
                    invDeriv_other[param_indices, full_other_space_labels_indx[(op_label, lbl)]] = inv_deriv[:, i]

        unused_param_indices = sorted(list(set(range(nOpParams)) - used_param_indices))
        prefix_mx = _np.zeros((nOpParams, len(unused_param_indices)), 'd')
        for j, indx in enumerate(unused_param_indices):
            prefix_mx[indx, j] = 1.0

        F_ham = _np.dot(invDeriv_ham, full_ham_fogi_vecs)
        F_other = _np.dot(invDeriv_other, full_other_fogi_vecs)
        F = _np.concatenate((prefix_mx, F_ham, F_other), axis=1)

        #Not sure if these are needed: "coefficients" have names, but maybe "parameters" shoudn't?
        #fogi_param_names = ["P%d" % i for i in range(len(unused_param_indices))] \
        #    + ham_fogi_vec_names + other_fogi_vec_names

        return _LinearInterposer(F)

    def setup_fogi(self, ham_basis, other_basis, other_mode="all",
                   op_label_abbrevs=None, reparameterize=False, reduce_to_model_space=True, constructive=False,
                   dependent_fogi_action='drop'):
        """
        TODO: docstring
        """
        primitive_op_labels = list(self.operations.keys())

        (ham_gauge_action, other_gauge_action, ham_gauge_linear_combos, other_gauge_linear_combos,
         ham_ga_dict, other_ga_dict) = \
            self._construct_gauge_space_for_model(primitive_op_labels, ham_basis, other_basis, other_mode,
                                                  reduce_to_model_space)

        if constructive:
            (ham_fogi_vecs, ham_fogi_labels, abbrev_ham_fogi_labels, ham_elem_errgen_labels_for_op,
             ham_dependent_vec_indices, ham_fogi_gauge_dirs, ham_op_errgen_indices) = \
                self._compute_fogi_constructively(primitive_op_labels, ham_basis, None,
                                                  ham_gauge_linear_combos, op_label_abbrevs, reduce_to_model_space,
                                                  dependent_fogi_action)
            (other_fogi_vecs, other_fogi_labels, abbrev_other_fogi_labels, other_elem_errgen_labels_for_op,
             other_dependent_vec_indices, other_fogi_gauge_dirs, other_op_errgen_indices) = \
                self._compute_fogi_constructively(primitive_op_labels, other_basis, other_mode,
                                                  other_gauge_linear_combos, op_label_abbrevs, reduce_to_model_space,
                                                  dependent_fogi_action)
        else:
            assert(dependent_fogi_action == 'drop')
            (ham_fogi_vecs, ham_elem_errgen_labels_for_op), (other_fogi_vecs, other_elem_errgen_labels_for_op) = \
                self._compute_fogi_via_nullspaces(primitive_op_labels, ham_basis, other_basis, other_mode,
                                                  ham_gauge_linear_combos, other_gauge_linear_combos, op_label_abbrevs,
                                                  reduce_to_model_space)
            ham_fogi_labels = other_fogi_labels = None  # nullspace method does not construct meaningful labels.
            abbrev_ham_fogi_labels = abbrev_other_fogi_labels = None
            ham_fogi_gauge_dirs = other_fogi_gauge_dirs = None  # nullspace method doesn't compute these
            ham_op_errgen_indices = other_op_errgen_indices = None  # these ARE computed - TODO: return it

        full_ham_space_labels = [(op_label, elem_lbl) for op_label in primitive_op_labels
                                 for elem_lbl in ham_elem_errgen_labels_for_op[op_label]]
        full_other_space_labels = [(op_label, elem_lbl) for op_label in primitive_op_labels
                                   for elem_lbl in other_elem_errgen_labels_for_op[op_label]]
        assert(len(full_ham_space_labels) == ham_fogi_vecs.shape[0])
        assert(len(full_other_space_labels) == other_fogi_vecs.shape[0])

        if reparameterize:
            self.param_interposer = self._add_reparameterization(primitive_op_labels,
                                                                 ham_fogi_vecs, full_ham_space_labels,
                                                                 other_fogi_vecs, full_other_space_labels)

        ham_gauge_vecs = _mt.nice_nullspace(ham_fogi_vecs.T)
        other_gauge_vecs = _mt.nice_nullspace(other_fogi_vecs.T)

        #TESTING (REMOVE)
        #if ham_gauge_linear_combos is not None:
        #    ham_gauge_action = _np.dot(ham_gauge_action, ham_gauge_linear_combos)
        #if other_gauge_linear_combos is not None:
        #    other_gauge_action = _np.dot(other_gauge_action, other_gauge_linear_combos)

        pinv_ham_gauge_action = _np.linalg.pinv(ham_gauge_action, rcond=1e-7)
        pinv_other_gauge_action = _np.linalg.pinv(other_gauge_action, rcond=1e-7)
        ham_gauge_directions = _np.dot(pinv_ham_gauge_action, ham_gauge_vecs)
        other_gauge_directions = _np.dot(pinv_other_gauge_action, other_gauge_vecs)

        raw_ham_fogi_labels = _fogi_names(ham_fogi_vecs, full_ham_space_labels, op_label_abbrevs)
        raw_other_fogi_labels = _fogi_names(other_fogi_vecs, full_other_space_labels, op_label_abbrevs)

        self.fogi_info = {'primitive_op_labels': primitive_op_labels,
                          'ham_op_errgen_indices': ham_op_errgen_indices,
                          'ham_vecs': ham_fogi_vecs,
                          'ham_elgen_labels': full_ham_space_labels,
                          'ham_elgen_labels_by_op': ham_elem_errgen_labels_for_op,
                          'ham_fogi_labels': ham_fogi_labels,
                          'ham_fogi_labels_raw': raw_ham_fogi_labels,
                          'ham_fogi_labels_abbrev': abbrev_ham_fogi_labels,
                          'ham_gauge_vecs': ham_gauge_vecs,
                          'ham_gauge_directions': ham_gauge_directions,
                          'ham_dependent_vec_indices': ham_dependent_vec_indices,
                          'ham_gauge_action_mxs': ham_ga_dict,
                          'ham_fogi_gauge_directions': ham_fogi_gauge_dirs,
                          'other_op_errgen_indices': other_op_errgen_indices,
                          'other_vecs': other_fogi_vecs,
                          'other_elgen_labels': full_other_space_labels,
                          'other_elgen_labels_by_op': other_elem_errgen_labels_for_op,
                          'other_fogi_labels': other_fogi_labels,
                          'other_fogi_labels_raw': raw_other_fogi_labels,
                          'other_fogi_labels_abbrev': abbrev_other_fogi_labels,
                          'other_gauge_vecs': other_gauge_vecs,
                          'other_gauge_directions': other_gauge_directions,
                          'other_dependent_vec_indices': other_dependent_vec_indices,
                          'other_gauge_action_mxs': other_ga_dict,
                          'other_fogi_gauge_directions': other_fogi_gauge_dirs,
                          'dependent_vec_indices': (ham_dependent_vec_indices + [i + ham_fogi_vecs.shape[1]
                                                                                 for i in other_dependent_vec_indices]),
                          }

        #if not constructive:
        #    self._recompute_fogi_names(op_label_abbrevs)

        assert(_np.linalg.norm(_np.dot(ham_gauge_action.T, ham_fogi_vecs)) < 1e-8)
        assert(_np.linalg.norm(_np.dot(other_gauge_action.T, other_fogi_vecs)) < 1e-8)

        #Check that pseudo-inverse was computed correctly (~ matrices are full rank)
        # full_other_vec = other_vecs * fogi
        # fogi = pinv_other_vecs * full_other_vec = (pinv_other_vecs * other_vecs) * fogi
        # so need (pinv_other_vecs * other_vecs) == identity
        if dependent_fogi_action == 'drop':
            assert(_np.linalg.norm(_np.dot(self.fogi_info['ham_vecs'].T, _np.linalg.pinv(self.fogi_info['ham_vecs'].T))
                                   - _np.identity(self.fogi_info['ham_vecs'].shape[1], 'd')) < 1e-6)
            assert(_np.linalg.norm(_np.dot(self.fogi_info['other_vecs'].T,
                                           _np.linalg.pinv(self.fogi_info['other_vecs'].T))
                                   - _np.identity(self.fogi_info['other_vecs'].shape[1], 'd')) < 1e-6)

        assert(_np.linalg.norm(_np.dot(self.fogi_info['ham_gauge_vecs'].T,
                                       _np.linalg.pinv(self.fogi_info['ham_gauge_vecs'].T))
                               - _np.identity(self.fogi_info['ham_gauge_vecs'].shape[1], 'd')) < 1e-6)
        assert(_np.linalg.norm(_np.dot(self.fogi_info['other_gauge_vecs'].T,
                                       _np.linalg.pinv(self.fogi_info['other_gauge_vecs'].T))
                               - _np.identity(self.fogi_info['other_gauge_vecs'].shape[1], 'd')) < 1e-6)

    #def errorgen_coefficients_array(self):
    #    pass

    def fogi_errorgen_coefficient_labels(self, complete=False, raw=False):
        assert(self.fogi_info is not None)
        ham_labels = self.fogi_info['ham_fogi_labels_raw'] if (self.fogi_info['ham_fogi_labels'] is None or raw) \
            else self.fogi_info['ham_fogi_labels']
        other_labels = self.fogi_info['other_fogi_labels_raw'] if (self.fogi_info['other_fogi_labels'] is None or raw) \
            else self.fogi_info['other_fogi_labels']
        fogi_labels = ham_labels + other_labels
        if not complete:
            return fogi_labels
        else:
            return fogi_labels + ["Gauge%d" % i for i in range(self.fogi_info['ham_gauge_vecs'].shape[1]
                                                               + self.fogi_info['other_gauge_vecs'].shape[1])]

    def fogi_errorgen_coefficients_array(self, complete=False, normalized_elem_gens=True):
        #  op_coeffs = dot(ham_vec, fogi_coeffs)
        # to get model's (Ham) linear combinations do:
        assert(self.fogi_info is not None)

        if not normalized_elem_gens:
            def rescale(coeffs):
                """ HACK: rescales errorgen coefficients for normalized-Pauli-basis elementary error gens
                         to be coefficients for the usual un-normalied-Pauli-basis elementary gens.  This
                         is only needed in the Hamiltonian case, as the non-ham "elementary" gen has a
                         factor of d2 baked into it. """
                d2 = _np.sqrt(self.dim); d = _np.sqrt(d2)
                return {lbl: (val / d if lbl[0] == 'H' else val) for lbl, val in coeffs.items()}

            op_coeffs = {op_label: rescale(self.operations[op_label].errorgen_coefficients())
                         for op_label in self.fogi_info['primitive_op_labels']}
        else:
            op_coeffs = {op_label: self.operations[op_label].errorgen_coefficients()
                         for op_label in self.fogi_info['primitive_op_labels']}

        ham_vecs = self.fogi_info['ham_vecs']
        other_vecs = self.fogi_info['other_vecs']
        if complete:
            ham_vecs = _np.concatenate((self.fogi_info['ham_vecs'],
                                        self.fogi_info['ham_gauge_vecs']), axis=1)
            other_vecs = _np.concatenate((self.fogi_info['other_vecs'],
                                          self.fogi_info['other_gauge_vecs']), axis=1)

        full_ham_vec = _np.zeros(self.fogi_info['ham_vecs'].shape[0], 'd')
        for i, (op_label, elem_lbl) in enumerate(self.fogi_info['ham_elgen_labels']):
            full_ham_vec[i] += op_coeffs[op_label].get(elem_lbl, 0.0)
        fogi_ham_coeffs = _np.dot(ham_vecs.T, full_ham_vec)

        full_other_vec = _np.zeros(self.fogi_info['other_vecs'].shape[0], complex)
        for i, (op_label, elem_lbl) in enumerate(self.fogi_info['other_elgen_labels']):
            full_other_vec[i] += op_coeffs[op_label].get(elem_lbl, 0.0)
        fogi_other_coeffs = _np.dot(other_vecs.T, full_other_vec)

        if complete:  # need to weave together results to put gauge at bottom.
            nHamFOGI = self.fogi_info['ham_vecs'].shape[1]
            nOtherFOGI = self.fogi_info['other_vecs'].shape[1]
            return _np.concatenate((fogi_ham_coeffs[0:nHamFOGI], fogi_other_coeffs[0:nOtherFOGI],
                                    fogi_ham_coeffs[nHamFOGI:], fogi_other_coeffs[nOtherFOGI:]))
        else:  # just concat the two sets of FIGO coeffs.
            return _np.concatenate((fogi_ham_coeffs, fogi_other_coeffs))

    def set_fogi_errorgen_coefficients_array(self, fogi_coefficients, complete=False, normalized_elem_gens=True):

        #TODO REMOVE
        #ham_vecs_pinv = self.fogi_info['ham_vecs_pinv']
        #other_vecs_pinv = self.fogi_info['other_vecs_pinv']
        #if complete:
        #    # Do pinv's properly here -- I think this is the same as using the commented code below, but
        #    # haven't checked yet.
        #    ham_vecs_pinv = _np.linalg.pinv(_np.concatenate((self.fogi_info['ham_vecs'],
        #                                                     self.fogi_info['ham_gauge_vecs']), axis=1), rcond=1e-7)
        #    other_vecs_pinv = _np.linalg.pinv(_np.concatenate((self.fogi_info['other_vecs'],
        #                                                       self.fogi_info['other_gauge_vecs']), axis=1),rcond=1e-7)
        #    # _np.concatenate((ham_vecs_pinv, self.fogi_info['ham_gauge_vecs_pinv']), axis=0)
        #    # _np.concatenate((other_vecs_pinv, self.fogi_info['other_gauge_vecs_pinv']), axis=0)
        #    assert(_np.linalg.matrix_rank(ham_vecs_pinv) == max(ham_vecs_pinv.shape))  # should be square & full rank!
        #    assert(_np.linalg.matrix_rank(other_vecs_pinv) == max(other_vecs_pinv.shape))  # ditto

        # to set model's (Ham) linear combinations do:
        assert(self.fogi_info is not None)
        nHamVecs = self.fogi_info['ham_vecs'].shape[1]
        ham_vecs = self.fogi_info['ham_vecs']
        other_vecs = self.fogi_info['other_vecs']
        if complete:
            #rearrange FOGI coefficients so H and S parts are separated
            nHamGauge = self.fogi_info['ham_gauge_vecs'].shape[1]
            nFOGI = nHamVecs + self.fogi_info['other_vecs'].shape[1]
            fc = fogi_coefficients  # shorthand
            fogi_coefficients = _np.concatenate((fc[0:nHamVecs], fc[nFOGI:nFOGI + nHamGauge],
                                                 fc[nHamVecs:nFOGI], fc[nFOGI + nHamGauge:]))

            nHamVecs += nHamGauge
            ham_vecs = _np.concatenate((ham_vecs, self.fogi_info['ham_gauge_vecs']), axis=1)
            other_vecs = _np.concatenate((other_vecs, self.fogi_info['other_gauge_vecs']), axis=1)
            assert(_np.linalg.matrix_rank(ham_vecs) == max(ham_vecs.shape))  # should be square & full rank!
            assert(_np.linalg.matrix_rank(other_vecs) == max(other_vecs.shape))  # ditto

        full_ham_vec = _np.dot(_np.linalg.pinv(ham_vecs.T, rcond=1e-7), fogi_coefficients[0:nHamVecs])
        full_other_vec = _np.dot(_np.linalg.pinv(other_vecs.T, rcond=1e-7), fogi_coefficients[nHamVecs:])
        op_coeffs = {op_label: {}
                     for op_label in self.fogi_info['primitive_op_labels']}
        for (op_label, elem_lbl), coeff_value in zip(self.fogi_info['ham_elgen_labels'], full_ham_vec):
            op_coeffs[op_label][elem_lbl] = coeff_value
        for (op_label, elem_lbl), coeff_value in zip(self.fogi_info['other_elgen_labels'], full_other_vec):
            op_coeffs[op_label][elem_lbl] = coeff_value

        if not normalized_elem_gens:
            def inv_rescale(coeffs):  # the inverse of the rescaling applied in fogi_errorgen_coefficients_array
                d2 = _np.sqrt(self.dim); d = _np.sqrt(d2)
                return {lbl: (val * d if lbl[0] == 'H' else val) for lbl, val in coeffs.items()}
        else:
            def inv_rescale(coeffs): return coeffs

        for op_label, coeff_dict in op_coeffs.items():
            self.operations[op_label].set_errorgen_coefficients(inv_rescale(coeff_dict))


class ExplicitLayerRules(_LayerRules):
    """ Rule: layer must appear explicitly as a "primitive op" """
    def prep_layer_operator(self, model, layerlbl, caches):
        """
        Create the operator corresponding to `layerlbl`.

        Parameters
        ----------
        layerlbl : Label
            A circuit layer label.

        Returns
        -------
        POVM or SPAMVec
        """
        # No need for caching preps
        return model.preps[layerlbl]  # don't cache this - it's not a new operator

    def povm_layer_operator(self, model, layerlbl, caches):
        """
        Create the operator corresponding to `layerlbl`.

        Parameters
        ----------
        layerlbl : Label
            A circuit layer label.

        Returns
        -------
        POVM or SPAMVec
        """
        if layerlbl in caches['povm-layers']: return caches['povm-layers'][layerlbl]
        return model.povms[layerlbl]  # don't cache this - it's not a new operator

    def operation_layer_operator(self, model, layerlbl, caches):
        """
        Create the operator corresponding to `layerlbl`.

        Parameters
        ----------
        layerlbl : Label
            A circuit layer label.

        Returns
        -------
        LinearOperator
        """
        if layerlbl in caches['op-layers']: return caches['op-layers'][layerlbl]
        if isinstance(layerlbl, _CircuitLabel):
            dense = isinstance(model._sim, _matrixfwdsim.MatrixForwardSimulator)  # True => create dense-matrix gates
            op = self._create_op_for_circuitlabel(model, layerlbl, dense)
            caches['op-layers'][layerlbl] = op
            return op
        elif layerlbl in model.operations:
            return model.operations[layerlbl]
        else:
            return _opfactory.op_from_factories(model.factories, layerlbl)


def _fogi_names(fogi_vecs, full_space_labels, op_label_abbrevs):
    if op_label_abbrevs is None: op_label_abbrevs = {}
    fogi_vec_names = []
    for j in range(fogi_vecs.shape[1]):
        name = ""
        for i, (op_lbl, elem_lbl) in enumerate(full_space_labels):
            val = fogi_vecs[i, j]
            if abs(val) < 1e-6: continue
            sign = ' + ' if val > 0 else ' - '
            abs_val_str = '' if _np.isclose(abs(val), 1.0) else ("%g " % abs(val))  # was %.1g
            name += sign + abs_val_str + "%s(%s)_%s" % (elem_lbl[0], ','.join(elem_lbl[1:]),
                                                        op_label_abbrevs.get(op_lbl, str(op_lbl)))
        if name.startswith(' + '): name = name[3:]  # strip leading +
        if name.startswith(' - '): name = '-' + name[3:]  # strip leading spaces
        fogi_vec_names.append(name)
    return fogi_vec_names


def _gauge_names(gauge_vecs, gauge_space_labels, abbrev=False):
    gauge_vec_names = []
    for j in range(gauge_vecs.shape[1]):
        name = ""
        for i, elem_lbl in enumerate(gauge_space_labels):
            val = gauge_vecs[i, j]
            if abs(val) < 1e-6: continue
            sign = ' + ' if val > 0 else ' - '
            abs_val_str = '' if _np.isclose(abs(val), 1.0) else ("%g " % abs(val))  # was %.1g
            if abbrev:
                name += sign + abs_val_str + "%s" % (','.join(elem_lbl[1:]))  # 'H' or 'S'
            else:
                name += sign + abs_val_str + "%s(%s)" % (elem_lbl[0], ','.join(elem_lbl[1:]))
        if name.startswith(' + '): name = name[3:]  # strip leading +
        if name.startswith(' - '): name = '-' + name[3:]  # strip leading spaces
        gauge_vec_names.append(name)
    return gauge_vec_names
