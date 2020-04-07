""" Defines the ImplicitOpModel class and supporting functionality."""
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
from . import simplifierhelper as _sh
from . import layerlizard as _ll

from .verbosityprinter import VerbosityPrinter as _VerbosityPrinter
from .basis import Basis as _Basis
from .label import Label as _Label


class ImplicitOpModel(_mdl.OpModel):
    """
    An ImplicitOpModel represents a flexible QIP model whereby only the
    building blocks for layer operations are stored, and custom layer-lizard
    logic is used to construct layer operations from these blocks on an
    on-demand basis.
    """

    def __init__(self,
                 state_space_labels,
                 basis="pp",
                 primitive_labels=None,
                 layer_lizard_class=_ll.ImplicitLayerLizard,
                 layer_lizard_args=(),
                 simplifier_helper_class=None,
                 sim_type="auto",
                 evotype="densitymx"):
        """
        Creates a new ImplicitOpModel.  Usually only called from derived
        classes `__init__` functions.

        Parameters
        ----------
        state_space_labels : StateSpaceLabels or list or tuple
            The decomposition (with labels) of (pure) state-space this model
            acts upon.  Regardless of whether the model contains operators or
            superoperators, this argument describes the Hilbert space dimension
            and imposed structure.  If a list or tuple is given, it must be
            of a from that can be passed to `StateSpaceLabels.__init__`.

        basis : Basis
            The basis used for the state space by dense operator representations.

        primitive_labels : dict, optional
            A dictionary of lists with keys `"preps"`, `"povms"`, `"ops"` and
            `"instruments`" giving the primitive-layer labels for each member
            type.  This information is needed for interfacing with the LGST
            algorithm and for circuit compiling.

        layer_lizard_class : class, optional
            The class of the layer lizard to use, which should usually be derived
            from :class:`ImplicitLayerLizard` and will be created using:
            `layer_lizard_class(simplified_prep_blks, simplified_op_blks, simplified_effect_blks, self)`

        layer_lizard_args : tuple, optional
            Additional arguments reserved for the custom layer lizard class.
            These arguments are not passed to the `layer_lizard_class`'s
            constructor, but are stored in the model's `._lizardArgs` member and
            may be accessed from within the layer lizard object (which gets a
            reference to the model upon initialization).

        simplifier_helper_class : class, optional
            The :class:`SimplifierHelper`-derived type used to provide the
            mimial interface needed for circuit compiling.  Initalized
            using `simplifier_helper_class(self)`.

        sim_type : {"auto", "matrix", "map", "termorder:X"}
            The type of forward simulator this model should use.  `"auto"`
            tries to determine the best type automatically.

        evotype : {"densitymx", "statevec", "stabilizer", "svterm", "cterm"}
            The evolution type of this model, describing how states are
            represented, allowing compatibility checks with (super)operator
            objects.
        """

        self.prep_blks = _collections.OrderedDict()
        self.povm_blks = _collections.OrderedDict()
        self.operation_blks = _collections.OrderedDict()
        self.instrument_blks = _collections.OrderedDict()
        self.factories = _collections.OrderedDict()

        if primitive_labels is None: primitive_labels = {}
        self._primitive_prep_labels = primitive_labels.get('preps', ())
        self._primitive_povm_labels = primitive_labels.get('povms', ())
        self._primitive_op_labels = primitive_labels.get('ops', ())
        self._primitive_instrument_labels = primitive_labels.get('instruments', ())

        self._lizardClass = layer_lizard_class
        self._lizardArgs = layer_lizard_args

        if simplifier_helper_class is None:
            simplifier_helper_class = _sh.ImplicitModelSimplifierHelper
            # by default, assume *_blk members have keys which match the simple
            # labels found in the circuits this model can simulate.
        self.simplifier_helper_class = simplifier_helper_class
        super(ImplicitOpModel, self).__init__(state_space_labels, basis, evotype,
                                              None, sim_type)
        self._shlp = simplifier_helper_class(self)

    def get_primitive_prep_labels(self):
        """ Return the primitive state preparation labels of this model"""
        return self._primitive_prep_labels

    def set_primitive_prep_labels(self, lbls):
        """ Set the primitive state preparation labels of this model"""
        self._primitive_prep_labels = tuple(lbls)

    def get_primitive_povm_labels(self):
        """ Return the primitive POVM labels of this model"""
        return self._primitive_povm_labels

    def set_primitive_povm_labels(self, lbls):
        """ Set the primitive POVM labels of this model"""
        self._primitive_povm_labels = tuple(lbls)

    def get_primitive_op_labels(self):
        """ Return the primitive operation labels of this model"""
        return self._primitive_op_labels

    def set_primitive_op_labels(self, lbls):
        """ Set the primitive operation labels of this model"""
        self._primitive_op_labels = tuple(lbls)

    def get_primitive_instrument_labels(self):
        """ Return the primitive instrument labels of this model"""
        return self._primitive_instrument_labels

    def set_primitive_instrument_labels(self, lbls):
        """ Set the primitive instrument labels of this model"""
        self._primitive_instrument_labels = tuple(lbls)

    #Functions required for base class functionality

    def _iter_parameterized_objs(self):
        for dictlbl, objdict in _itertools.chain(self.prep_blks.items(),
                                                 self.povm_blks.items(),
                                                 self.operation_blks.items(),
                                                 self.instrument_blks.items(),
                                                 self.factories.items()):
            for lbl, obj in objdict.items():
                yield (_Label(dictlbl + ":" + lbl.name, lbl.sslbls), obj)

    def _layer_lizard(self):
        """ (simplified op server) """
        self._clean_paramvec()  # just to be safe
        return self._lizardClass(self.prep_blks, self.operation_blks, self.povm_blks, self.instrument_blks, self)
        # maybe add a self.factories arg? (but factories aren't really "simplified"...
        # use self._lizardArgs internally?

    def _init_copy(self, copyInto):
        """
        Copies any "tricky" member of this model into `copyInto`, before
        deep copying everything else within a .copy() operation.
        """
        # Copy special base class members first
        super(ImplicitOpModel, self)._init_copy(copyInto)

        # Copy our "tricky" members
        copyInto.prep_blks = _collections.OrderedDict([(lbl, prepdict.copy(copyInto))
                                                       for lbl, prepdict in self.prep_blks.items()])
        copyInto.povm_blks = _collections.OrderedDict([(lbl, povmdict.copy(copyInto))
                                                       for lbl, povmdict in self.povm_blks.items()])
        copyInto.operation_blks = _collections.OrderedDict([(lbl, opdict.copy(copyInto))
                                                            for lbl, opdict in self.operation_blks.items()])
        copyInto.instrument_blks = _collections.OrderedDict([(lbl, idict.copy(copyInto))
                                                             for lbl, idict in self.instrument_blks.items()])
        copyInto.factories = _collections.OrderedDict([(lbl, fdict.copy(copyInto))
                                                       for lbl, fdict in self.factories.items()])

        copyInto._state_space_labels = self._state_space_labels.copy()  # needed by simplifier helper
        copyInto._shlp = self.simplifier_helper_class(copyInto)

    def __setstate__(self, stateDict):
        self.__dict__.update(stateDict)
        if 'uuid' not in stateDict:
            self.uuid = _uuid.uuid4()  # create a new uuid

        if 'factories' not in stateDict:
            self.factories = _collections.OrderedDict()  # backward compatibility (temporary)

        #Additionally, must re-connect this model as the parent
        # of relevant OrderedDict-derived classes, which *don't*
        # preserve this information upon pickling so as to avoid
        # circular pickling...
        for prepdict in self.prep_blks.values():
            prepdict.parent = self
            for o in prepdict.values(): o.relink_parent(self)
        for povmdict in self.povm_blks.values():
            povmdict.parent = self
            for o in povmdict.values(): o.relink_parent(self)
        for opdict in self.operation_blks.values():
            opdict.parent = self
            for o in opdict.values(): o.relink_parent(self)
        for idict in self.instrument_blks.values():
            idict.parent = self
            for o in idict.values(): o.relink_parent(self)
        for fdict in self.factories.values():
            fdict.parent = self
            for o in fdict.values(): o.relink_parent(self)

    def get_clifford_symplectic_reps(self, oplabel_filter=None):
        """
        Constructs a dictionary of the symplectic representations for all
        the Clifford gates in this model.  Non-:class:`CliffordOp` gates
        will be ignored and their entries omitted from the returned dictionary.

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

        for gl in self.get_primitive_op_labels():
            gate = self.operation_blks['layers'][gl]
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

    def __str__(self):
        s = ""
        for dictlbl, d in self.prep_blks.items():
            for lbl, vec in d.items():
                s += "%s:%s = " % (str(dictlbl), str(lbl)) + str(vec) + "\n"
        s += "\n"
        for dictlbl, d in self.povm_blks.items():
            for lbl, povm in d.items():
                s += "%s:%s = " % (str(dictlbl), str(lbl)) + str(povm) + "\n"
        s += "\n"
        for dictlbl, d in self.operation_blks.items():
            for lbl, gate in d.items():
                s += "%s:%s = \n" % (str(dictlbl), str(lbl)) + str(gate) + "\n\n"
        for dictlbl, d in self.instrument_blks.items():
            for lbl, inst in d.items():
                s += "%s:%s = " % (str(dictlbl), str(lbl)) + str(inst) + "\n"
        s += "\n"
        for dictlbl, d in self.factories.items():
            for lbl, factory in d.items():
                s += "%s:%s = " % (str(dictlbl), str(lbl)) + str(factory) + "\n"
        s += "\n"

        return s
