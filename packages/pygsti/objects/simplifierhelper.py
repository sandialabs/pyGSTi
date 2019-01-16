""" Defines the SimplifierHelper class and supporting functionality."""
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
import uuid as _uuid
import bisect as _bisect
import copy as _copy

from ..tools import matrixtools as _mt
from ..tools import optools as _gt
from ..tools import slicetools as _slct
from ..tools import likelihoodfns as _lf
from ..tools import jamiolkowski as _jt
from ..tools import compattools as _compat
from ..tools import basistools as _bt
from ..tools import listtools as _lt
from ..tools import symplectic as _symp

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

from ..baseobjs import VerbosityPrinter as _VerbosityPrinter
from ..baseobjs import Basis as _Basis
from ..baseobjs import Label as _Label


class SimplifierHelper(object):
    """
    Defines the minimal interface for performing :class:`Circuit` "compiling"
    (pre-processing for forward simulators, which only deal with preps, ops, 
    and effects) needed by :class:`Model`.

    To simplify a circuit a `Model` doesn't, for instance, need to know *all*
    possible state preparation labels, as a dict of preparation operations
    would provide - it only needs a function to check if a given value is a
    viable state-preparation label.
    """
    pass #TODO docstring - FILL IN functions & docstrings
        
class BasicSimplifierHelper(SimplifierHelper):
    """
    Performs the work of a :class:`SimplifierHelper` using user-supplied lists
    """
    def __init__(self, preplbls, povmlbls, instrumentlbls,
                 povm_effect_lbls, instrument_member_lbls):
        """
        Create a new BasicSimplifierHelper.

        preplbls, povmlbls, instrumentlbls, povm_effect_lbls,
        instrument_member_lbls : list
            Lists of all the state-preparation, POVM, instrument,
            POVM-effect, and instrument-member labels of a model.
        """
        self.preplbls = preplbls
        self.povmlbls = povmlbls
        self.instrumentlbls = instrumentlbls
        self.povm_effect_lbls = povm_effect_lbls
        self.instrument_member_lbls = instrument_member_lbls
    
    def is_prep_lbl(self, lbl):
        return lbl in self.preplbls
    
    def is_povm_lbl(self, lbl):
        return lbl in self.povmlbls
    
    def is_instrument_lbl(self, lbl):
        return lbl in self.instrumentlbls
    
    def get_default_prep_lbl(self):
        return self.preplbls[0] \
            if len(self.preplbls) == 1 else None
    
    def get_default_povm_lbl(self):
        return self.povmlbls[0] \
            if len(self.povmlbls) == 1 else None

    def has_preps(self):
        return len(self.preplbls) > 0

    def has_povms(self):
        return len(self.povmlbls) > 0

    def get_effect_labels_for_povm(self, povm_lbl):
        return self.povm_effect_lbls[povm_lbl]

    def get_member_labels_for_instrument(self, inst_lbl):
        return self.instrument_member_lbls[inst_lbl]

class MemberDictSimplifierHelper(SimplifierHelper):
    """
    Performs the work of a :class:`SimplifierHelper` using a set of
    `OrderedMemberDict` objects, such as those contained in an
    :class:`ExplicitOpModel`.
    """
    def __init__(self, preps, povms, instruments):
        """
        Create a new MemberDictSimplifierHelper.

        Parameters
        ----------
        preps, povms, instruments : OrderedMemberDict
        """
        self.preps = preps        
        self.povms = povms
        self.instruments = instruments
    
    def is_prep_lbl(self, lbl):
        return lbl in self.preps
    
    def is_povm_lbl(self, lbl):
        return lbl in self.povms
    
    def is_instrument_lbl(self, lbl):
        return lbl in self.instruments
    
    def get_default_prep_lbl(self):
        return tuple(self.preps.keys())[0] \
            if len(self.preps) == 1 else None
    
    def get_default_povm_lbl(self):
        return tuple(self.povms.keys())[0] \
            if len(self.povms) == 1 else None

    def has_preps(self):
        return len(self.preps) > 0

    def has_povms(self):
        return len(self.povms) > 0

    def get_effect_labels_for_povm(self, povm_lbl):
        return tuple(self.povms[povm_lbl].keys())

    def get_member_labels_for_instrument(self, inst_lbl):
        return tuple(self.instruments[inst_lbl].keys())


class MemberDictDictSimplifierHelper(SimplifierHelper):
    """
    Performs the work of a :class:`SimplifierHelper` using a set of
    dictionaries of `OrderedMemberDict` objects, such as those
    contained in an :class:`ImplicitOpModel`.
    """
    def __init__(self, prep_blks, povm_blks, instrument_blks):
        """
        Create a new MemberDictDictSimplifierHelper.

        Parameters
        ----------
        prep_blks, povm_blks, instrument_blks : dict of OrderedMemberDict
        """
        self.prep_blks = prep_blks
        self.povm_blks = povm_blks
        self.instrument_blks = instrument_blks
    
    def is_prep_lbl(self, lbl):
        return any([(lbl in prepdict) for prepdict in self.prep_blks.values()])
    
    def is_povm_lbl(self, lbl):
        return any([(lbl in povmdict) for povmdict in self.povm_blks.values()])
    
    def is_instrument_lbl(self, lbl):
        return any([(lbl in idict) for idict in self.instrument_blks.values()])
    
    def get_default_prep_lbl(self):
        npreps = sum([ len(prepdict) for prepdict in self.prep_blks.values()])
        if npreps == 1:
            for prepdict in self.prep_blks.values():
                if len(prepdict) > 0:
                    return tuple(prepdict.keys())[0]
            assert(False), "Logic error: one prepdict should have had lenght > 0!"
        else:
            return None
    
    def get_default_povm_lbl(self):
        npovms = sum([ len(povmdict) for povmdict in self.povm_blks.values()])
        if npovms == 1:
            for povmdict in self.povm_blks.values():
                if len(povmdict) > 0:
                    return tuple(povmdict.keys())[0]
            assert(False), "Logic error: one povmdict should have had lenght > 0!"
        else:
            return None

    def has_preps(self):
        return any([ (len(prepdict) > 0) for prepdict in self.prep_blks.values()])

    def has_povms(self):
        return any([ (len(povmdict) > 0) for povmdict in self.povm_blks.values()])

    def get_effect_labels_for_povm(self, povm_lbl):
        for povmdict in self.povm_blks.values():
            if povm_lbl in povmdict:
                return tuple(povmdict[povm_lbl].keys())
        raise KeyError("No POVM labeled %s!" % povm_lbl)

    def get_member_labels_for_instrument(self, inst_lbl):
        for idict in self.instrument_blks.values():
            if inst_lbl in idict:
                return tuple(idict[inst_lbl].keys())
        raise KeyError("No instrument labeled %s!" % inst_lbl)


class ImplicitModelSimplifierHelper(MemberDictDictSimplifierHelper):
    """ Performs the work of a "Simplifier Helper" using user-supplied dicts """
    def __init__(self, implicitModel):
        """ Create a new ImplicitModelSimplifierHelper. """
        super(ImplicitModelSimplifierHelper,self).__init__(
            implicitModel.prep_blks, implicitModel.povm_blks, implicitModel.instrument_blks)
