"""
Defines the ExplicitOpModel class and supporting functionality.
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import collections as _collections
import itertools as _itertools
import uuid as _uuid
import warnings as _warnings

import numpy as _np
import scipy as _scipy

from pygsti.models import explicitcalc as _explicitcalc
from pygsti.models import model as _mdl, gaugegroup as _gg
from pygsti.models.memberdict import OrderedMemberDict as _OrderedMemberDict
from pygsti.models.layerrules import LayerRules as _LayerRules
from pygsti.models.modelparaminterposer import ModelParamsInterposer as _ModelParamsInterposer
from pygsti.models.fogistore import FirstOrderGaugeInvariantStore as _FirstOrderGaugeInvariantStore
from pygsti.models.gaugegroup import (
    GaugeGroup as _GaugeGroup,
    GaugeGroupElement as _GaugeGroupElement
)
from pygsti.forwardsims.forwardsim import ForwardSimulator as _FSim
from pygsti.forwardsims import matrixforwardsim as _matrixfwdsim
from pygsti.modelmembers import instruments as _instrument
from pygsti.modelmembers import operations as _op
from pygsti.modelmembers import povms as _povm
from pygsti.modelmembers import states as _state
from pygsti.modelmembers.modelmembergraph import ModelMemberGraph as _MMGraph
from pygsti.modelmembers.operations import opfactory as _opfactory
from pygsti.baseobjs.basis import Basis as _Basis
from pygsti.baseobjs.basis import BuiltinBasis as _BuiltinBasis, DirectSumBasis as _DirectSumBasis
from pygsti.baseobjs.label import Label as _Label, CircuitLabel as _CircuitLabel
from pygsti.baseobjs import statespace as _statespace
from pygsti.tools import basistools as _bt
from pygsti.tools import jamiolkowski as _jt
from pygsti.tools import matrixtools as _mt
from pygsti.tools import optools as _ot
from pygsti.tools import fogitools as _fogit
from pygsti.tools import slicetools as _slct
from pygsti.tools import listtools as _lt
from pygsti import SpaceT
from pygsti.tools.legacytools import deprecate as _deprecated_fn


class ExplicitOpModel(_mdl.OpModel):
    """
    Encapsulates a set of gate, state preparation, and POVM effect operations.

    An ExplictOpModel stores a set of labeled LinearOperator objects and
    provides dictionary-like access to their matrices.  State preparation
    and POVM effect operations are represented as column vectors.

    Parameters
    ----------
    state_space : StateSpace
        The state space for this model.

    basis : {"pp","gm","qt","std","sv"} or Basis, optional
        The basis used for the state space by dense superoperator representations.

    default_param : {"full", "TP", "CPTP", etc.}, optional
        Specifies the default gate and SPAM vector parameterization type.
        Can be any value allowed by :meth:`set_all_parameterizations`,
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
        requested probabilities, e.g. from :meth:`probs` or
        :meth:`bulk_probs`.  The default value of `"auto"` automatically
        selects the simulation type, and is usually what you want. Other
        special allowed values are:

        - "matrix" : op_matrix-op_matrix products are computed and
          cached to get composite gates which can then quickly simulate
          a circuit for any preparation and outcome.  High memory demand;
          best for a small number of (1 or 2) qubits.
        - "map" : op_matrix-state_vector products are repeatedly computed
          to simulate circuits.  Slower for a small number of qubits, but
          faster and more memory efficient for higher numbers of qubits (3+).

    evotype : Evotype or str, optional
        The evolution type of this model, describing how states are
        represented.  The special value `"default"` is equivalent
        to specifying the value of `pygsti.evotypes.Evotype.default_evotype`.
    """

    #Whether access to gates & spam vecs via Model indexing is allowed
    _strict = False

    def __init__(self, state_space, basis="pp", default_gate_type="full",
                 default_prep_type="auto", default_povm_type="auto",
                 default_instrument_type="auto", prep_prefix="rho", effect_prefix="E",
                 gate_prefix="G", povm_prefix="M", instrument_prefix="I",
                 simulator="auto", evotype="default"):
        #More options now (TODO enumerate?)
        #assert(default_param in ('full','TP','CPTP','H+S','S','static',
        #                         'H+S terms','clifford','H+S clifford terms'))
        def flagfn(typ): return {'auto_embed': True, 'match_parent_statespace': True,
                                 'match_parent_evotype': True, 'cast_to_type': typ}

        if default_prep_type == "auto":
            default_prep_type = _state.state_type_from_op_type(default_gate_type)
        if default_povm_type == "auto":
            default_povm_type = _povm.povm_type_from_op_type(default_gate_type)
        if default_instrument_type == "auto":
            default_instrument_type = _instrument.instrument_type_from_op_type(default_gate_type)

        self.preps = _OrderedMemberDict(self, default_prep_type, prep_prefix, flagfn("state"))
        self.povms = _OrderedMemberDict(self, default_povm_type, povm_prefix, flagfn("povm"))
        self.operations = _OrderedMemberDict(self, default_gate_type, gate_prefix, flagfn("operation"))
        self.instruments = _OrderedMemberDict(self, default_instrument_type, instrument_prefix, flagfn("instrument"))
        self.factories = _OrderedMemberDict(self, default_gate_type, gate_prefix, flagfn("factory"))
        self.covariance_function = None
        self.effects_prefix = effect_prefix
        self._default_gauge_group = None
        
        super(ExplicitOpModel, self).__init__(state_space, basis, evotype, ExplicitLayerRules(), simulator)

    @property
    def _primitive_prep_label_dict(self):
        return self.preps

    @property
    def _primitive_povm_label_dict(self):
        return self.povms

    @property
    def _primitive_op_label_dict(self):
        # don't include 'implied' ops as primitive ops -- FUTURE - maybe should include empty layer ([])?
        return _collections.OrderedDict([(k, None) for k in self.operations
                                         if not (k.name.startswith('{') and k.name.endswith('}'))])

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
        if self.covariance_function is not None:
            yield ('Covariance Function', self.covariance_function)

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

        return _explicitcalc.ExplicitOpModelCalc(self.state_space.dim, simplified_preps, simplified_ops,
                                                 simplified_effects, self.num_params, self._param_interposer)

    def _embed_operation(self, op_target_labels, op_val, force=False):
        """
        Called by OrderedMemberDict._auto_embed to create an embedded-gate
        object that embeds `op_val` into the sub-space of
        `self.state_space` given by `op_target_labels`.

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
        if self.state_space is None:
            raise ValueError("Must set model state space before adding auto-embedded gates.")

        if op_val.state_space == self.state_space and not force:
            return op_val  # if gate operates on full dimension, no need to embed.

        return _op.EmbeddedOp(self.state_space, op_target_labels, op_val)

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
        State
        """
        if len(self.preps) != 1:
            raise ValueError("'.prep' can only be used on models"
                             " with a *single* state prep.  This Model has"
                             " %d state preps!" % len(self.preps))
        return list(self.preps.values())[0]

    @property
    def effects(self):
        """
        The effect vectors from the unique POVM in this model, if one exists.

        If not, a ValueError is raised.

        Returns
        -------
        list of POVMEffects
        """
        if len(self.povms) != 1:
            raise ValueError("'.effects' can only be used on models"
                             " with a *single* POVM.  This Model has"
                             " %d POVMS!" % len(self.povms))
        return list(self.povms.values())[0]

    def __setitem__(self, label, value):
        """
        Set an operator, state, or POVM associated with a given label.

        Parameters
        ----------
        label : string
            the gate or SPAM vector label.

        value : numpy array or LinearOperator or State or POVM
            a operation matrix, state vector, or POVM, which must have the
            appropriate state space for the Model and appropriate type
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
        Get an operation, state, or POVM associated with a given label.

        Parameters
        ----------
        label : string
            the gate, state vector, or POVM label.
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

    def convert_members_inplace(self, to_type, categories_to_convert='all', labels_to_convert='all',
                                ideal_model=None, flatten_structure=False, set_default_gauge_group=False, 
                                cptp_truncation_tol= 1e-7, spam_cp_penalty = 1e-7):
        """
        Method for converting the parameterizations of modelmembers within this model to new ones in-place.

        Parameters
        ----------
        to_type :  str
            String specifier for the parameterization type to convert to.
        
        categories_to_convert : str or list of str, optional (default 'all')
            Categories of modelmembers to perform conversion on. Allowed options are:
            'all', 'ops' or 'operations' (these two are aliases for the same option),
            'instruments', 'povms' or 'preps'.
        
        labels_to_convert : str or list of `Label`, optional (default 'all')
            A string specifier, or list of `Label` objects, specifying the set
            of objects (state preparations, operations, instruments, etc.) within the model 
            to apply the conversion to.

         ideal_model : `Model`, optional (default None)
            A model containing modelmembers instantiated such that they all correspond to the ideal
            actions of the given gate set elements. It is recommended that this be specified when
            converting to an error-generator-based parameterization.
        
        flatten_structure : bool, optional (default False)
            When `False`, the sub-members of composed and embedded operations
            are separately converted, leaving the original modelmember structure
            unchanged.  When `True`, composed and embedded operations are "flattened"
            into a single modelmember parameterized according to the requested `to_type`.

        set_default_gauge_group : bool, optional (default False)
            A flag specifying whether the default gauge group for the model should be updated
            to the default value associated with the specified value of `to_type`.
            See `set_default_gauge_group_for_member_type` for more on these default gauge groups.

        cptp_truncation_tol : float, optional (default 1e-7)
            Tolerance term used to enforce the CPTP constraint on gates when moving between different
            parameterizations.

        spam_cp_penalty : float, optional (default 1e-7)
            Penalty term used to enforce the CP constraint on SPAM when moving between different
            parameterizations.        
        """
        if isinstance(categories_to_convert, str): categories_to_convert = (categories_to_convert,)
        if any([c in categories_to_convert for c in ('all', 'ops', 'operations')]):
            for lbl, gate in self.operations.items():
                if labels_to_convert == 'all' or lbl in labels_to_convert:
                    ideal = ideal_model.operations.get(lbl, None) if (ideal_model is not None) else None
                    self.operations[lbl] = _op.convert(gate, to_type, self.basis, ideal, flatten_structure, cptp_truncation_tol)
        if any([c in categories_to_convert for c in ('all', 'instruments')]):
            for lbl, inst in self.instruments.items():
                if labels_to_convert == 'all' or lbl in labels_to_convert:
                    ideal = ideal_model.instruments.get(lbl, None) if (ideal_model is not None) else None
                    self.instruments[lbl] = _instrument.convert(inst, to_type, self.basis, ideal, flatten_structure)
        if any([c in categories_to_convert for c in ('all', 'preps')]):
            for lbl, prep in self.preps.items():
                if labels_to_convert == 'all' or lbl in labels_to_convert:
                    ideal = ideal_model.preps.get(lbl, None) if (ideal_model is not None) else None
                    self.preps[lbl] = _state.convert(prep, to_type, self.basis, ideal, flatten_structure, cp_penalty=spam_cp_penalty)
        if any([c in categories_to_convert for c in ('all', 'povms')]):
            for lbl, povm in self.povms.items():
                if labels_to_convert == 'all' or lbl in labels_to_convert:
                    ideal = ideal_model.povms.get(lbl, None) if (ideal_model is not None) else None
                    self.povms[lbl] = _povm.convert(povm, to_type, self.basis, ideal, flatten_structure, cp_penalty=spam_cp_penalty)

        self._clean_paramvec()  # param indices were probabaly updated
        if set_default_gauge_group:
            self.set_default_gauge_group_for_member_type(to_type)

    def set_default_gauge_group_for_member_type(self, member_type):
        """ 
        Updates the default gauge group to the default value for the specified modelmember type.

        Parameters
        ----------
        member_type : str
            A string specifier for the modelmember type used to select the gauge group type.
            Mapping is the following:

            - 'full' -> `FullGaugeGroup`
            - 'full TP', 'TP', `TPGaugeGroup`
            - 'CPTP' or Anything that is a valid lindblad type -> `UnitaryGaugeGroup`
            - Otherwise -> `TrivialGaugeGroup`
        """
        if member_type == 'full':
            self.default_gauge_group = _gg.FullGaugeGroup(self.state_space, self.basis, self.evotype)
        elif member_type in ('full TP', 'TP'):  # TODO: get from verbose_conversion dictionary of modelmembers?
            self.default_gauge_group = _gg.TPGaugeGroup(self.state_space, self.basis, self.evotype)
        elif _ot.is_valid_lindblad_paramtype(member_type) or member_type == 'CPTP':
            self.default_gauge_group = _gg.UnitaryGaugeGroup(self.state_space, self.basis, self.evotype)
        else:  
            self.default_gauge_group = _gg.TrivialGaugeGroup(self.state_space)

    def set_all_parameterizations(self, gate_type, prep_type="auto", povm_type="auto",
                                  instrument_type="auto", ideal_model=None, cptp_truncation_tol = 1e-6, spam_cp_penalty = 1e-7):
        """
        Convert all gates, states, and POVMs to a specific parameterization type.

        Parameters
        ----------
        parameterization_type : string
            The gate, state, and POVM parameterization type.  Allowed
            values are (where '*' means " terms" and " clifford terms"
            evolution-type suffixes are allowed):

            - "full" : each gate / state / POVM effect element is an independent parameter
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
              "d". This removes the CPTP constraint on the gates and SPAM
              operations (and as such is seldom used).

        ideal_model : Model, optional
            This may specify an ideal model of unitary gates and pure state vectors
            to be used as the *ideal* operation of each gate/SPAM operation, which
            is particularly useful as target for CPTP-based conversions.
            
        cptp_truncation_tol : float, optional (default 1e-6)
            Tolerance used for conversion to CPTP parameterizations. When converting to
            CPTP models negative eigenvalues of the choi matrix representation of a superoperator
            are truncated, which can result in a change in the PTM for that operator. This tolerance
            indicates the maximum amount of truncation induced deviation from the original operations
            (measured by frobenius distance) we're willing to accept without marking the conversion
            as failed.
        spam_cp_penalty : float, optional (default 0.5)
            Converting SPAM operations to an error generator representation may 
            introduce trivial gauge degrees of freedom. These gauge degrees of freedom 
            are called trivial because they quite literally do not change the dense representation 
            (i.e. Hilbert-Schmidt vectors) at all. Despite being trivial, error generators along 
            this trivial gauge orbit may be non-CP, so this cptp penalty is used to favor channels 
            within this gauge orbit which are CPTP.

        Returns
        -------
        None
        """
        typ = gate_type

        # Set ideal model to static when used as targets (specifically needed for CPTP prep/povms)
        static_model = None
        if ideal_model is not None:
            static_model = ideal_model.copy()
            static_model.set_all_parameterizations('static')

        rtyp = _state.state_type_from_op_type(gate_type) if prep_type == "auto" else prep_type
        povmtyp = _povm.povm_type_from_op_type(gate_type) if povm_type == "auto" else povm_type
        ityp = _instrument.instrument_type_from_op_type(gate_type) if instrument_type == "auto" else instrument_type

        try:
            self.convert_members_inplace(typ, 'operations', 'all', flatten_structure=True, ideal_model=static_model, cptp_truncation_tol = cptp_truncation_tol)
            self.convert_members_inplace(ityp, 'instruments', 'all', flatten_structure=True, ideal_model=static_model, cptp_truncation_tol = cptp_truncation_tol)
            self.convert_members_inplace(rtyp, 'preps', 'all', flatten_structure=True, ideal_model=static_model, cptp_truncation_tol = cptp_truncation_tol, spam_cp_penalty = spam_cp_penalty)
            self.convert_members_inplace(povmtyp, 'povms', 'all', flatten_structure=True, ideal_model=static_model, cptp_truncation_tol = cptp_truncation_tol, spam_cp_penalty = spam_cp_penalty)
        except ValueError as e:
            raise ValueError("Failed to convert members. If converting to CPTP-based models, " +
                "try providing an ideal_model to avoid possible branch cuts.") from e
        
        self.set_default_gauge_group_for_member_type(typ)

    def __setstate__(self, state_dict):

        if "gates" in state_dict:
            #Unpickling an OLD-version Model (or GateSet)
            _warnings.warn("Unpickling deprecated-format ExplicitOpModel (GateSet).  Please re-save/pickle asap.")
            self.operations = state_dict['gates']
            self.state_space = state_dict['stateSpaceLabels']
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
            state_dict['state_space'] = state_dict['state_space_labels']; del state_dict['state_space_labels']
        if 'factories' not in state_dict:
            ops = state_dict['operations']
            state_dict['factories'] = _OrderedMemberDict(self, ops.default_param, ops._prefix, ops.flags)

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
        if self.covariance_function is not None:
            self.covariance_function.relink_parent(self)

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
        rhoSize = [rho.hilbert_schmidt_size for rho in self.preps.values()]
        povmSize = [povm.num_elements for povm in self.povms.values()]
        opSize = [gate.hilbert_schmidt_size for gate in self.operations.values()]
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
        #Note maybe we need some way for some evotypes to punt here? (and just return 0?)
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

    def compute_nongauge_and_gauge_spaces(self, item_weights=None, non_gauge_mix_mx=None):
        """
        TODO: docstring
        """
        return self._excalc().nongauge_and_gauge_spaces(item_weights, non_gauge_mix_mx)

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
            special strings "gates" or "spam" which represent the entire set of gate
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
            rhoVec.transform_inplace(s)

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
            between corresponding SPAM operations.

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
            between corresponding SPAM operations.

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

        op_dim = self.state_space.dim
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
            def dist(a, b): return _ot.entanglement_infidelity(a, b, self.basis)
            def vecdist(a, b): return _np.linalg.norm(a - b)
        elif metric == 'diamond':
            def dist(a, b): return 0.5 * _ot.diamondist(a, b, self.basis)
            def vecdist(a, b): return _np.linalg.norm(a - b)
        else:
            raise ValueError("Invalid `metric` argument: %s" % metric)

        on_space = 'HilbertSchmidt'
        s = "Model Difference:\n"
        s += " Preps:\n"
        for lbl in self.preps:
            s += "  %s = %g\n" % \
                (str(lbl), vecdist(self.preps[lbl].to_dense(on_space), other_model.preps[lbl].to_dense(on_space)))

        s += " POVMs:\n"
        for povm_lbl, povm in self.povms.items():
            s += "  %s: " % str(povm_lbl)
            for lbl in povm:
                s += "    %s = %g\n" % \
                     (lbl, vecdist(povm[lbl].to_dense(on_space), other_model.povms[povm_lbl][lbl].to_dense(on_space)))

        s += " Gates:\n"
        for lbl in self.operations:
            s += "  %s = %g\n" % \
                (str(lbl), dist(self.operations[lbl].to_dense(on_space),
                                other_model.operations[lbl].to_dense(on_space)))

        if len(self.instruments) > 0:
            s += " Instruments:\n"
            for inst_lbl, inst in self.instruments.items():
                s += "  %s: " % str(inst_lbl)
                for lbl in inst:
                    s += "    %s = %g\n" % (str(lbl), dist(
                        inst[lbl].to_dense(on_space), other_model.instruments[inst_lbl][lbl].to_dense(on_space)))

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
            s += "%s = (factory)" % str(lbl) + '\n'
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
            opeations (state and POVM effects) in the model. (Multiplies the
            non-identity part of each assumed-Pauli-basis state preparation
            vector and measurement vector by ``(1.0-spam_noise)``).

        max_op_noise : float, optional
            specified instead of `op_noise`; apply a random depolarization
            with maximum strength ``1-max_op_noise`` to each gate in the
            model.

        max_spam_noise : float, optional
            specified instead of `spam_noise`; apply a random depolarization
            with maximum strength ``1-max_spam_noise`` to each state preparation
            and POVM in the model.

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
        dim = self.state_space.dim
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

        op_dim = self.state_space.dim
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

            randOp = _ot.unitary_to_superop(randUnitary, self.basis)

            mdl_randomized.operations[opLabel] = _op.FullArbitraryOp(_np.dot(randOp, gate.to_dense("HilbertSchmidt")))

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

        if isinstance(new_dimension, _statespace.StateSpace):
            state_space = new_dimension
            new_dimension = state_space.dim
        else:
            state_space = _statespace.default_space_for_dim(new_dimension)

        curDim = self.state_space.dim
        assert(new_dimension > curDim)

        #For now, just create a dumb default state space labels and basis for the new model:
        sslbls = [('L%d' % i,) for i in range(new_dimension)]  # interpret as independent classical levels
        dumb_basis = _DirectSumBasis([_BuiltinBasis('gm', 1)] * new_dimension,
                                     name="Unknown")  # - just act on diagonal density mx
        new_model = ExplicitOpModel(sslbls, dumb_basis, "full", "auto", "auto", "auto",
                                    self.preps._prefix, self.effects_prefix,
                                    self.operations._prefix, self.povms._prefix,
                                    self.instruments._prefix, self._sim.copy())
        #new_model._dim = new_dimension # dim will be set when elements are added
        #new_model.reset_basis() #FUTURE: maybe user can specify how increase is being done?

        addedDim = new_dimension - curDim
        vec_zeroPad = _np.zeros((addedDim, 1), 'd')
        evotype = self.evotype

        #Increase dimension of rhoVecs and EVecs by zero-padding
        for lbl, rhoVec in self.preps.items():
            assert(len(rhoVec) == curDim)
            new_model.preps[lbl] = \
                _state.FullState(_np.concatenate((rhoVec, vec_zeroPad)), dumb_basis, evotype, state_space)

        for lbl, povm in self.povms.items():
            assert(povm.state_space.dim == curDim)
            effects = [(elbl, _np.concatenate((EVec, vec_zeroPad)))
                       for elbl, EVec in povm.items()]

            if isinstance(povm, _povm.TPPOVM):
                new_model.povms[lbl] = _povm.TPPOVM(effects, evotype, state_space)
            else:
                new_model.povms[lbl] = _povm.UnconstrainedPOVM(effects, evotype, state_space)  # everything else

        #Increase dimension of gates by assuming they act as identity on additional (unknown) space
        for opLabel, gate in self.operations.items():
            assert(gate.shape == (curDim, curDim))
            newOp = _np.zeros((new_dimension, new_dimension))
            newOp[0:curDim, 0:curDim] = gate[:, :]
            for i in range(curDim, new_dimension): newOp[i, i] = 1.0
            new_model.operations[opLabel] = _op.FullArbitraryOp(newOp, dumb_basis, evotype, state_space)

        for instLabel, inst in self.instruments.items():
            inst_ops = []
            for outcomeLbl, gate in inst.items():
                newOp = _np.zeros((new_dimension, new_dimension))
                newOp[0:curDim, 0:curDim] = gate[:, :]
                for i in range(curDim, new_dimension): newOp[i, i] = 1.0
                inst_ops.append((outcomeLbl, _op.FullArbitraryOp(newOp, dumb_basis, evotype, state_space)))
            new_model.instruments[instLabel] = _instrument.Instrument(inst_ops, evotype, state_space)

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
        if isinstance(new_dimension, _statespace.StateSpace):
            state_space = new_dimension
            new_dimension = state_space.dim
        else:
            state_space = _statespace.default_space_for_dim(new_dimension)

        curDim = self.state_space.dim
        assert(new_dimension < curDim)

        #For now, just create a dumb default state space labels and basis for the new model:
        sslbls = [('L%d' % i,) for i in range(new_dimension)]  # interpret as independent classical levels
        dumb_basis = _DirectSumBasis([_BuiltinBasis('gm', 1)] * new_dimension,
                                     name="Unknown")  # - just act on diagonal density mx
        new_model = ExplicitOpModel(sslbls, dumb_basis, "full", "auto", "auto", "auto",
                                    self.preps._prefix, self.effects_prefix,
                                    self.operations._prefix, self.povms._prefix,
                                    self.instruments._prefix, self._sim.copy())
        #new_model._dim = new_dimension # dim will be set when elements are added
        #new_model.reset_basis() #FUTURE: maybe user can specify how decrease is being done?

        #Decrease dimension of rhoVecs and EVecs by truncation
        for lbl, rhoVec in self.preps.items():
            assert(len(rhoVec) == curDim)
            new_model.preps[lbl] = \
                _state.FullState(rhoVec[0:new_dimension, :], dumb_basis, self.evotype, state_space)

        for lbl, povm in self.povms.items():
            assert(povm.state_space.dim == curDim)
            effects = [(elbl, EVec[0:new_dimension, :]) for elbl, EVec in povm.items()]

            if isinstance(povm, _povm.TPPOVM):
                new_model.povms[lbl] = _povm.TPPOVM(effects, self.evotype, state_space)
            else:
                new_model.povms[lbl] = _povm.UnconstrainedPOVM(effects, self.evotype, state_space)  # everything else

        #Decrease dimension of gates by truncation
        for opLabel, gate in self.operations.items():
            assert(gate.shape == (curDim, curDim))
            newOp = _np.zeros((new_dimension, new_dimension))
            newOp[:, :] = gate[0:new_dimension, 0:new_dimension]
            new_model.operations[opLabel] = _op.FullArbitraryOp(newOp, evotype=self.evotype, state_space=state_space)

        for instLabel, inst in self.instruments.items():
            inst_ops = []
            for outcomeLbl, gate in inst.items():
                newOp = _np.zeros((new_dimension, new_dimension))
                newOp[:, :] = gate[0:new_dimension, 0:new_dimension]
                inst_ops.append((outcomeLbl, _op.FullArbitraryOp(newOp, evotype=self.evotype, state_space=state_space)))
            new_model.instruments[instLabel] = _instrument.Instrument(inst_ops, self.evotype, state_space)

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
            kicked_gs.operations[opLabel] = _op.FullArbitraryOp(kicked_gs.operations[opLabel] + delta)

        #Note: does not alter intruments!
        return kicked_gs

    def compute_clifford_symplectic_reps(self, oplabel_filter=None):
        """
        Constructs a dictionary of the symplectic representations for all the Clifford gates in this model.

        Non-:class:`StaticCliffordOp` gates will be ignored and their entries omitted
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
                assert(isinstance(gate.embedded_op, _op.StaticCliffordOp)), \
                    "EmbeddedClifforGate contains a non-StaticCliffordOp!"
                lbl = gl.name  # strip state space labels off since this is a
                # symplectic rep for the *embedded* gate
                srep = (gate.embedded_op.smatrix, gate.embedded_op.svector)
            elif isinstance(gate, _op.StaticCliffordOp):
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

    @_deprecated_fn
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

    def create_processor_spec(self, qudit_labels='auto'):
        """
        Create a processor specification from this model with the given qudit labels.

        Currently this only works for models on qudits.

        Parameters
        ----------
        qudit_labels : tuple or `"auto"`, optional
            A tuple of qudit labels, e.g. ('Q0', 'Q1') or (0, 1).  `"auto"`
            uses the labels in this model's state space labels.

        Returns
        -------
        QuditProcessorSpec or QubitProcessorSpec
        """
        from pygsti.processors import QubitProcessorSpec as _QubitProcessorSpec
        from pygsti.processors import QuditProcessorSpec as _QuditProcessorSpec
        #go through ops, building up availability and unitaries, then create procesor spec...

        nqudits = self.state_space.num_qudits
        gate_unitaries = _collections.OrderedDict()
        all_sslbls = self.state_space.sole_tensor_product_block_labels
        all_udims = [self.state_space.label_udimension(lbl) for lbl in all_sslbls]
        availability = {}

        def extract_unitary(Umx, U_sslbls, extracted_sslbls):
            if extracted_sslbls is None: return Umx  # no extraction to be done
            extracted_sslbls = list(extracted_sslbls)
            extracted_indices = [U_sslbls.index(lbl) for lbl in extracted_sslbls]
            extracted_udims = [self.state_space.label_udimension(lbl) for lbl in extracted_sslbls]

            # can assume all lbls are qudits, so increment associated with qudit k is (2^(N-1-k) for qubits):
            all_inc = _np.flip(_np.cumprod(list(reversed(all_udims[1:] + [1]))))
            extracted_inc = all_inc[extracted_indices]

            # assume this is a kronecker product (check this in FUTURE?), so just fill extracted
            # unitary by fixing all non-extracted qudits (assumed identity-action on these) to 0
            # and looping over extracted ones:
            U_extracted = _np.zeros((_np.prod(extracted_udims), _np.prod(extracted_udims)), complex)
            for ii, itup in enumerate(_itertools.product(*[range(ud) for ud in extracted_udims])):
                i = _np.dot(extracted_inc, itup)
                for jj, jtup in enumerate(_itertools.product(*[range(ud) for ud in extracted_udims])):
                    j = _np.dot(extracted_inc, jtup)
                    U_extracted[ii, jj] = Umx[i, j]
            return U_extracted

        def add_availability(opkey, op):
            if opkey == _Label(()) or opkey.is_simple:
                if opkey == _Label(()):  # special case: turn empty tuple labels into "{idle}" gate in processor spec
                    gn = "{idle}"
                    sslbls = None
                elif opkey.is_simple:
                    gn = opkey.name
                    sslbls = opkey.sslbls
                    #if sslbls is not None:
                    #    observed_sslbls.update(sslbls)

                if gn not in gate_unitaries or gate_unitaries[gn] is None:
                    U = _ot.superop_to_unitary(op.to_dense("HilbertSchmidt"), self.basis) \
                        if (op is not None) else None  # U == None indicates "unknown, up until this point"

                    Ulocal = extract_unitary(U, all_sslbls, sslbls)
                    gate_unitaries[gn] = Ulocal

                    if gn in availability:
                        if sslbls not in availability[gn]:
                            availability[gn].append(sslbls)
                    else:
                        availability[gn] = [sslbls]
                elif sslbls not in availability[gn]:
                    availability[gn].append(sslbls)

            else:  # a COMPOUND label with components => process each component separately
                for component in opkey.components:
                    add_availability(component, None)  # recursive call - the reason we need this to be a function!

        #observed_sslbls = set()
        for opkey, op in self.operations.items():  # TODO: need to deal with special () idle label
            add_availability(opkey, op)

        #Check that there aren't any undetermined unitaries
        unknown_unitaries = [k for k, v in gate_unitaries.items() if v is None]
        if len(unknown_unitaries) > 0:
            raise ValueError("Unitary not specfied for %s gate(s)!" % str(unknown_unitaries))

        if qudit_labels == 'auto':
            qudit_labels = self.state_space.sole_tensor_product_block_labels
            #OR: qudit_labels = self.state_space.qudit_labels  # only works for a QuditSpace
            #OR: qudit_labels = self.state_space.qubit_labels  # only works for a QubitSpace
            #OR: qubit_labels = tuple(sorted(observed_sslbls))

        if qudit_labels is None:  # special case of legacy explicit models where all gates have availability [None]
            qudit_labels = tuple(range(nqudits))

        assert(len(qudit_labels) == nqudits), \
            "Length of `qudit_labels` must equal %d (not %d)!" % (nqudits, len(qudit_labels))

        if all([udim == 2 for udim in all_udims]):
            return _QubitProcessorSpec(nqudits, list(gate_unitaries.keys()), gate_unitaries, availability,
                                       qubit_labels=qudit_labels,
                                       instrument_names=list(self.instruments.keys()), nonstd_instruments=self.instruments)
        else:
            return _QuditProcessorSpec(qudit_labels, all_udims, list(gate_unitaries.keys()), gate_unitaries,
                                       availability,
                                       instrument_names=list(self.instruments.keys()), nonstd_instruments=self.instruments)

    def create_modelmember_graph(self):
        return _MMGraph({
            'preps': self.preps,
            'povms': self.povms,
            'operations': self.operations,
            'instruments': self.instruments,
            'factories': self.factories,
        })

    def _to_nice_serialization(self):
        state = super()._to_nice_serialization()
        state.update({'basis': self.basis.to_nice_serialization(),
                      'default_gate_type': self.operations.default_param,
                      'default_prep_type': self.preps.default_param,
                      'default_povm_type': self.povms.default_param,
                      'default_instrument_type': self.instruments.default_param,
                      'prep_prefix': self.preps._prefix,
                      'effect_prefix': self.effects_prefix,
                      'gate_prefix': self.operations._prefix,
                      'povm_prefix': self.povms._prefix,
                      'instrument_prefix': self.instruments._prefix,
                      'evotype': str(self.evotype),  # TODO or serialize?
                      'simulator': self.sim.to_nice_serialization(),
                      'default_gauge_group': (self.default_gauge_group.to_nice_serialization()
                                              if (self.default_gauge_group is not None) else None),
                      'parameter_interposer': (self._param_interposer.to_nice_serialization()
                                               if (self._param_interposer is not None) else None),
                      'fogi_store': (self.fogi_store.to_nice_serialization()
                                               if (self.fogi_store is not None) else None)
                      })

        mmgraph = self.create_modelmember_graph()
        state['modelmembers'] = mmgraph.create_serialization_dict()
        return state

    @classmethod
    def _from_nice_serialization(cls, state):
        state_space = _statespace.StateSpace.from_nice_serialization(state['state_space'])
        basis = _Basis.from_nice_serialization(state['basis'])
        simulator = _FSim.from_nice_serialization(state['simulator'])
        default_gauge_group = _GaugeGroup.from_nice_serialization(state['default_gauge_group']) \
            if (state['default_gauge_group'] is not None) else None
        param_interposer = _ModelParamsInterposer.from_nice_serialization(state['parameter_interposer']) \
            if (state['parameter_interposer'] is not None) else None
        
        fogi_store = _FirstOrderGaugeInvariantStore.from_nice_serialization(state['fogi_store']) \
            if (state.get('fogi_store', None) is not None) else None
        param_labels = state.get('parameter_labels', None)
        param_bounds = state.get('parameter_bounds', None)

        mdl = cls(state_space, basis, state['default_gate_type'],
                  state['default_prep_type'], state['default_povm_type'],
                  state['default_instrument_type'], state['prep_prefix'], state['effect_prefix'],
                  state['gate_prefix'], state['povm_prefix'], state['instrument_prefix'],
                  simulator, state['evotype'])

        modelmembers = _MMGraph.load_modelmembers_from_serialization_dict(state['modelmembers'], mdl)
        mdl.preps.update(modelmembers.get('preps', {}))
        mdl.povms.update(modelmembers.get('povms', {}))
        mdl.operations.update(modelmembers.get('operations', {}))
        mdl.instruments.update(modelmembers.get('instruments', {}))
        mdl.factories.update(modelmembers.get('factories', {}))
        mdl._clean_paramvec()
        mdl.default_gauge_group = default_gauge_group
        mdl.param_interposer = param_interposer
        mdl.fogi_store = fogi_store

        Np = len(mdl._paramlbls)  # _clean_paramvec sets up ._paramlbls so its length == # of params
        if param_labels and len(param_labels) == Np:
            mdl._paramlbls[:] = [_lt.lists_to_tuples(lbl) for lbl in param_labels]
        if param_bounds is not None:
            param_bounds = cls._decodemx(param_bounds)
            if param_bounds.shape == (Np, 2):
                mdl._param_bounds

        return mdl

    def errorgen_coefficients(self, normalized_elem_gens=True):
        """TODO: docstring - returns a nested dict containing all the error generator coefficients for all
           the operations in this model. """
        if not normalized_elem_gens:
            def rescale(coeffs):
                """ HACK: rescales errorgen coefficients for normalized-Pauli-basis elementary error gens
                         to be coefficients for the usual un-normalied-Pauli-basis elementary gens.  This
                         is only needed in the Hamiltonian case, as the non-ham "elementary" gen has a
                         factor of d2 baked into it.
                """
                d2 = _np.sqrt(self.dim); d = _np.sqrt(d2)
                return {lbl: (val / d if lbl.errorgen_type == 'H' else val) for lbl, val in coeffs.items()}

            op_coeffs = {op_label: rescale(self.operations[op_label].errorgen_coefficients())
                         for op_label in self.operations}
            op_coeffs.update({prep_label: rescale(self.preps[prep_label].errorgen_coefficients())
                              for prep_label in self.preps})
            op_coeffs.update({povm_label: rescale(self.povms[povm_label].errorgen_coefficients())
                              for povm_label in self.povms})

        else:
            op_coeffs = {op_label: self.operations[op_label].errorgen_coefficients()
                         for op_label in self.operations}
            op_coeffs.update({prep_label: self.preps[prep_label].errorgen_coefficients()
                              for prep_label in self.preps})
            op_coeffs.update({povm_label: self.povms[povm_label].errorgen_coefficients()
                              for povm_label in self.povms})

        return op_coeffs

    def _op_decomposition(self, op_label):
        """Returns the target and error-generator-containing error map parts of the operation for `op_label` """
        return self.operations[op_label], self.operations[op_label]


def transform_composed_model(mdl: ExplicitOpModel, s : _GaugeGroupElement) -> ExplicitOpModel:
    """
    Return a copy of `mdl` whose members have been gauge-transformed by `s`,
    while retaining the parameterization of `mdl`.
    
    This function's implementation requires that `mdl` use ComposedState for
    stateprep and ComposedPOVM for measurements. It does NOT require that
    operations be represented with ComposedOp. It ignores any factories that
    might be present in mdl.
    """
    from pygsti.models import ExplicitOpModel
    assert isinstance(mdl, ExplicitOpModel)
    if len(mdl.factories) > 0:
        _warnings.warn('The returned model will not retain the factories in mdl.')
    if len(mdl.instruments) > 0:
        raise NotImplementedError('Models with instruments are not supported.')

    oldmdl = mdl

    def mycopy(_m):
        # This function is a hack. It makes us robust to errors
        # arising from copy.deepcopy in (re)linking model members
        # to parent model objects.
        s = _m.to_nice_serialization()
        t = ExplicitOpModel.from_nice_serialization(s)
        return t
    
    mdl = mycopy(oldmdl)

    from pygsti.modelmembers.operations import ComposedOp, StaticArbitraryOp
    from pygsti.modelmembers.povms import ComposedPOVM
    from pygsti.modelmembers.states import ComposedState

    U    = StaticArbitraryOp(s.transform_matrix,         basis=oldmdl.basis)
    invU = StaticArbitraryOp(s.transform_matrix_inverse, basis=oldmdl.basis) 

    # NOTE: the operations passed to ComposedOp are interpreted in
    # reverse order. For example, ComposedOp([X,Y,Z]) is applied to
    # a vector v as Z @ Y @ X @ v.

    for key, rho in oldmdl.preps.items():
        # replace each ComposedState superket `rhoVec` with `invU @ rhoVec`;
        # do this by packing invU into a new ComposedState's error map.
        assert isinstance(rho, ComposedState)
        static_rho = rho.state_vec
        errmap  = ComposedOp([rho.error_map, invU])
        mdl.preps[key] = ComposedState(static_rho, errmap)

    for key, povm in oldmdl.povms.items():
        # replace each ComposedPOVM `p` with another ComposedPOVM `q`, where
        # effects `Evec` belonging to `p` are mapped to effects `EVec @ U` 
        # belonging to `q`. Do this by packing U into the error map of `q`.
        assert isinstance(povm, ComposedPOVM)
        static_povm = povm.base_povm
        errmap = ComposedOp([U, povm.error_map])
        mdl.povms[key] = ComposedPOVM(errmap, static_povm, mx_basis=oldmdl.basis)

    for key, op in oldmdl.operations.items():
        # replace each operation `G` with `invU @ G @ U`.
        op_s = ComposedOp([U, op, invU])
        mdl.operations[key] = op_s

    mdl._clean_paramvec()  # transform may leave dirty members
    return mdl


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
        State
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
        POVM or POVMEffect
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
            op = self._create_op_for_circuitlabel(model, layerlbl)
            caches['op-layers'][layerlbl] = op
            return op
        elif layerlbl in model.operations:
            return model.operations[layerlbl]
        else:
            return _opfactory.op_from_factories(model.factories, layerlbl)
