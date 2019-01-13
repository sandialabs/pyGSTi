""" Defines the Model class and supporting functionality."""
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

#create model 

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

class Model(object):
    """
    A predictive model for a Quantum Information Processor (QIP).

    The main function of a `Model` object is to compute the outcome
    probabilities of :class:`Circuit` objects based on the action of the
    model's ideal operations plus (potentially) noise which makes the
    outcome probabilities deviate from the perfect ones.
    """

    #Whether to perform extra parameter-vector integrity checks
    _pcheck = False

    def __init__(self, state_space_labels, basis, evotype, simplifier_helper, sim_type="auto"):
        """
        Creates a new Model.  Rarely used except from derived classes
        `__init__` functions.

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

        evotype : {"densitymx", "statevec", "stabilizer", "svterm", "cterm"}
            The evolution type of this model, describing how states are
            represented, allowing compatibility checks with (super)operator 
            objects.

        simplifier_helper : SimplifierHelper
            Provides a minimal interface for compiling circuits for forward
            simulation.

        sim_type : {"auto", "matrix", "map", "termorder:X"}
            The type of forward simulator this model should use.  `"auto"`
            tries to determine the best type automatically.
        """
        self._evotype = evotype
        self.set_state_space(state_space_labels, basis)
          #sets self._state_space_labels, self._basis, self._dim

        self.set_simtype(sim_type)
          #sets self._calcClass, self._sim_type, self._sim_args

        self._paramvec = _np.zeros(0, 'd')
        self._shlp = simplifier_helper
        self._paramlbls = None # a placeholder for FUTURE functionality
        self._need_to_rebuild = True #whether we call _rebuild_paramvec() in to_vector() or num_params()
        self.dirty = False #indicates when objects and _paramvec may be out of sync
        #OLD TODO REMOVE self._rebuild_paramvec()
        
        self.uuid = _uuid.uuid4() # a Model's uuid is like a persistent id(), useful for hashing
        super(Model, self).__init__()

    ##########################################
    ## Get/Set methods
    ##########################################

    @property
    def simtype(self):
        """ Forward simulation type """
        return self._sim_type

    @property
    def evotype(self):
        """ Evolution type """
        return self._evotype

    @property
    def state_space_labels(self):
        """ State space labels """
        return self._state_space_labels

    @property
    def basis(self):
        """ The basis used to represent dense (super)operators of this model """
        return self._basis

    @basis.setter
    def basis(self, basis):
        if isinstance(basis, _Basis):
            assert(basis.dim == self.state_space_labels.dim)
            self._basis = basis
        else: #create a basis with the proper dimension
            self._basis = _Basis(basis, self.state_space_labels.dim)
    
    def set_simtype(self, sim_type, calc_cache=None):
        """
        Reset the forward simulation type of this model.

        Parameters
        ----------
        sim_type : {"auto", "matrix", "map", "termorder:X"}
            The type of forward simulator this model should use.  `"auto"`
            tries to determine the best type automatically.

        calc_cache : dict or None
            A cache of pre-computed values used in Taylor-term-based forward
            simulation.

        Returns
        -------
        None
        """
        #Calculator selection based on simulation type

        if sim_type == "auto":
            default_param = self.operations.default_param # assume the same for other dicts
            if _gt.is_valid_lindblad_paramtype(default_param) and \
               _gt.split_lindblad_paramtype(default_param)[1] in ("svterm","cterm"):
                sim_type = "termorder:1"
            else:
                d = self._dim if (self._dim is not None) else 0
                sim_type = "matrix" if d <= 16 else "map"

        simtype_and_args = sim_type.split(":")
        sim_type = simtype_and_args[0]
        if sim_type == "matrix":      c = _matrixfwdsim.MatrixForwardSimulator
        elif sim_type == "map":       c = _mapfwdsim.MapForwardSimulator
        elif sim_type == "termorder": c = _termfwdsim.TermForwardSimulator
        else: raise ValueError("Invalid `sim_type` (%s)" % sim_type)

        self._calcClass = c
        self._sim_type = sim_type
        self._sim_args = list(simtype_and_args[1:])

        if sim_type == "termorder":
            cache = calc_cache if (calc_cache is not None) else {} # make a temp cache if none is given
            self._sim_args.append(cache) # add calculation cache as another argument

    def reset_basis(self):
        """
        "Forgets" the current basis, so that
        self.basis becomes a dummy Basis w/name "unknown".
        """
        self._basis = _Basis('unknown', None)

    def set_state_space(self, lbls, basis="pp"):
        """
        Sets labels for the components of the Hilbert space upon which 
        the gates of this Model act.

        Parameters
        ----------
        lbls : list or tuple or StateSpaceLabels object
            A list of state-space labels (can be strings or integers), e.g.
            `['Q0','Q1']` or a :class:`StateSpaceLabels` object.

        basis : Basis or str
            A :class:`Basis` object or a basis name (like `"pp"`), specifying
            the basis used to interpret the operators in this Model.  If a
            `Basis` object, then its dimensions must match those of `lbls`.
        
        Returns
        -------
        None
        """
        if isinstance(lbls, _ld.StateSpaceLabels):
            self._state_space_labels = lbls
        else:
            self._state_space_labels = _ld.StateSpaceLabels(lbls)
        self.basis = basis # invokes basis setter to set self._basis

        #Operator dimension of this Model
        if self._evotype in ("densitymx","svterm","cterm"):
            self._dim = self.state_space_labels.dim.opDim
        else:
            self._dim = self.state_space_labels.dim.dmDim #operator dim for *state* vectors
            # FUTURE: have a Basis for state *vectors*?

    @property
    def dim(self):
        """
        The dimension of the model, which equals d when the gate
        matrices have shape d x d and spam vectors have shape d x 1.

        Returns
        -------
        int
            model dimension
        """
        return self._dim


    def get_dimension(self):
        """
        Get the dimension of the model, which equals d when the gate
        matrices have shape d x d and spam vectors have shape d x 1.
        Equivalent to model.dim.

        Returns
        -------
        int
            model dimension
        """
        return self._dim



            
    ####################################################
    ## Parameter vector maintenance
    ####################################################

    def num_params(self):
        """
        Return the number of free parameters when vectorizing
        this model.

        Returns
        -------
        int
            the number of model parameters.
        """
        self._clean_paramvec()
        return len(self._paramvec)
    
    def _iter_parameterized_objs(self):
        raise NotImplementedError("Derived Model classes should implement _iter_parameterized_objs")
        #return # default is to have no parameterized objects

    def _check_paramvec(self, debug=False):
        if debug: print("---- Model._check_paramvec ----")

        TOL=1e-8
        for lbl,obj in self._iter_parameterized_objs():
            if debug: print(lbl,":",obj.num_params(),obj.gpindices)
            w = obj.to_vector()
            msg = "None" if (obj.parent is None) else id(obj.parent)
            assert(obj.parent is self), "%s's parent is not set correctly (%s)!" % (lbl,msg)
            if obj.gpindices is not None and len(w) > 0:
                if _np.linalg.norm(self._paramvec[obj.gpindices]-w) > TOL:
                    if debug: print(lbl,".to_vector() = ",w," but Model's paramvec = ",self._paramvec[obj.gpindices])
                    raise ValueError("%s is out of sync with paramvec!!!" % lbl)
            if self.dirty==False and obj.dirty:
                raise ValueError("%s is dirty but Model.dirty=False!!" % lbl)


    def _clean_paramvec(self):
        """ Updates _paramvec corresponding to any "dirty" elements, which may
            have been modified without out knowing, leaving _paramvec out of
            sync with the element's internal data.  It *may* be necessary
            to resolve conflicts where multiple dirty elements want different
            values for a single parameter.  This method is used as a safety net
            that tries to insure _paramvec & Model elements are consistent
            before their use."""

        #print("Cleaning Paramvec (dirty=%s, rebuild=%s)" % (self.dirty, self._need_to_rebuild))
        if self._need_to_rebuild:
            self._rebuild_paramvec()
            self._need_to_rebuild = False

        if self.dirty: # if any member object is dirty (ModelMember.dirty setter should set this value)
            TOL=1e-8

            #Note: lbl args used *just* for potential debugging - could strip out once
            # we're confident this code always works.
            def clean_single_obj(obj,lbl): # sync an object's to_vector result w/_paramvec
                if obj.dirty:
                    w = obj.to_vector()
                    chk_norm = _np.linalg.norm(self._paramvec[obj.gpindices]-w)
                    #print(lbl, " is dirty! vec = ", w, "  chk_norm = ",chk_norm)
                    if (not _np.isfinite(chk_norm)) or chk_norm > TOL:
                        self._paramvec[obj.gpindices] = w
                    obj.dirty = False
                        
            def clean_obj(obj,lbl): # recursive so works with objects that have sub-members
                for i,subm in enumerate(obj.submembers()):
                    clean_obj(subm, _Label(lbl.name+":%d"%i,lbl.sslbls))
                clean_single_obj(obj,lbl)

            def reset_dirty(obj): # recursive so works with objects that have sub-members
                for i,subm in enumerate(obj.submembers()): reset_dirty(subm)
                obj.dirty = False
            
            for lbl,obj in self._iter_parameterized_objs():
                clean_obj(obj,lbl)

            #re-update everything to ensure consistency ~ self.from_vector(self._paramvec)
            #print("DEBUG: non-trivially CLEANED paramvec due to dirty elements")
            for _,obj in self._iter_parameterized_objs():
                obj.from_vector( self._paramvec[obj.gpindices] )
                reset_dirty(obj) # like "obj.dirty = False" but recursive
                  #object is known to be consistent with _paramvec
            
        if Model._pcheck: self._check_paramvec()


    def _mark_for_rebuild(self, modified_obj=None):
        #re-initialze any members that also depend on the updated parameters
        self._need_to_rebuild = True
        for _,o in self._iter_parameterized_objs():
            if o._obj_refcount(modified_obj) > 0:
                o.clear_gpindices() # ~ o.gpindices = None but works w/submembers
                                    # (so params for this obj will be rebuilt)
        self.dirty = True
          #since it's likely we'll set at least one of our object's .dirty flags
          # to True (and said object may have parent=None and so won't
          # auto-propagate up to set this model's dirty flag (self.dirty)

        
    #TODO REMOVE: unneeded now that we do *lazy* rebuilding of paramvec (now set self.need_to_rebuild=True)
    #def _update_paramvec(self, modified_obj=None):
    #    """Updates self._paramvec after a member of this Model is modified"""
    #    self._rebuild_paramvec() # prepares _paramvec & gpindices
    #
    #    #update parameters changed by modified_obj
    #    self._paramvec[modified_obj.gpindices] = modified_obj.to_vector()
    #
    #    #re-initialze any members that also depend on the updated parameters
    #    modified_indices = set(modified_obj.gpindices_as_array())
    #    for _,obj in self._iter_parameterized_objs():
    #        if obj is modified_obj: continue
    #        if modified_indices.intersection(obj.gpindices_as_array()):
    #            obj.from_vector(self._paramvec[obj.gpindices])

    def _print_gpindices(self):
        print("PRINTING MODEL GPINDICES!!!")
        for lbl,obj in self._iter_parameterized_objs():
            print("LABEL ",lbl)
            obj._print_gpindices()

    def _rebuild_paramvec(self):
        """ Resizes self._paramvec and updates gpindices & parent members as needed,
            and will initialize new elements of _paramvec, but does NOT change
            existing elements of _paramvec (use _update_paramvec for this)"""
        v = self._paramvec; Np = len(self._paramvec) #NOT self.num_params() since the latter calls us!
        off = 0; shift = 0

        #ellist = ", ".join(map(str,list(self.preps.keys()) +list(self.povms.keys()) +list(self.operations.keys())))
        #print("DEBUG: rebuilding... %s" % ellist)

        #Step 1: remove any unused indices from paramvec and shift accordingly
        used_gpindices = set()
        for _,obj in self._iter_parameterized_objs():
            if obj.gpindices is not None:
                assert(obj.parent is self), "Member's parent is not set correctly (%s)!" % str(obj.parent)
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
            for _,obj in self._iter_parameterized_objs():
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
        for lbl,obj in self._iter_parameterized_objs():

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
                #print("DEBUG: %s: alloc'd & inserted %d new params.  indices = " % (str(lbl),obj.num_params()), obj.gpindices, " off=",off)
            else:
                inds = obj.gpindices_as_array()
                M = max(inds) if len(inds)>0 else -1; L = len(v)
                #print("DEBUG: %s: existing indices = " % (str(lbl)), obj.gpindices, " M=",M," L=",L)
                if M >= L:
                    #Some indices specified by obj are absent, and must be created.
                    w = obj.to_vector()
                    v = _np.concatenate((v, _np.empty(M+1-L,'d')),axis=0) # [v.resize(M+1) doesn't work]
                    shift += M+1-L
                    for ii,i in enumerate(inds):
                        if i >= L: v[i] = w[ii]
                    #print("DEBUG:    --> added %d new params" % (M+1-L))
                if M >= 0: # M == -1 signifies this object has no parameters, so we'll just leave `off` alone
                    off = M+1

        self._paramvec = v
        #print("DEBUG: Done rebuild: %d params" % len(v))

    def _init_virtual_obj(self, obj):
        """ 
        Initializes a "virtual object" - an object (e.g. LinearOperator) that *could* be a
        member of the Model but won't be, as it's just built for temporary
        use (e.g. the parallel action of several "base" gates).  As such
        we need to fully initialize its parent and gpindices members so it 
        knows it belongs to this Model BUT it's not allowed to add any new
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

    def _obj_refcount(self, obj):
        """ Number of references to `obj` contained within this Model """
        cnt = 0
        for _,o in self._iter_parameterized_objs():
            cnt += o._obj_refcount(obj)
        return cnt

    def to_vector(self):
        """
        Returns the model vectorized according to the optional parameters.

        Returns
        -------
        numpy array
            The vectorized model parameters.
        """
        self._clean_paramvec() # will rebuild if needed
        return self._paramvec


    def from_vector(self, v, reset_basis=False):
        """
        The inverse of to_vector.  Loads values of gates and rho and E vecs from
        from the vector `v`.  Note that `v` does not specify the number of
        gates, etc., and their labels: this information must be contained in
        this `Model` prior to calling `from_vector`.  In practice, this just
        means you should call the `from_vector` method using the same `Model`
        that was used to generate the vector `v` in the first place.
        """
        assert( len(v) == self.num_params() )
        
        self._paramvec = v.copy()
        for _,obj in self._iter_parameterized_objs():
            obj.from_vector( v[obj.gpindices] )
            obj.dirty = False #object is known to be consistent with _paramvec

        if reset_basis:
            self.reset_basis() 
            # assume the vector we're loading isn't producing gates & vectors in
            # a known basis.
        if Model._pcheck: self._check_paramvec()


    ######################################
    ## Compilation
    ######################################
        
    def _layer_lizard(self):
        """ Return a layer lizard for this model """
        raise NotImplementedError("Derived Model classes should implement this!")
    
    def _calc(self):
        """ Create & return a forward-simulator ("calculator") for this model """
        self._clean_paramvec()
        layer_lizard = self._layer_lizard() 
        
        kwargs = {}
        if self._sim_type == "termorder":
            kwargs['max_order'] = int(self._sim_args[0])
            kwargs['cache'] = self._sim_args[-1] # always the list argument

        assert(self._calcClass is not None), "Model does not have a calculator setup yet!"
        return self._calcClass(self._dim, layer_lizard, self._paramvec, **kwargs) #fwdsim class


    def split_circuit(self, circuit, erroron=('prep','povm')):
        """
        Splits a operation sequence into prepLabel + opsOnlyString + povmLabel
        components.  If `circuit` does not contain a prep label or a
        povm label a default label is returned if one exists.

        Parameters
        ----------
        circuit : Circuit
            A operation sequence, possibly beginning with a state preparation
            label and ending with a povm label.

        erroron : tuple of {'prep','povm'}
            A ValueError is raised if a preparation or povm label cannot be
            resolved when 'prep' or 'povm' is included in 'erroron'.  Otherwise
            `None` is returned in place of unresolvable labels.  An exception
            is when this model has no preps or povms, in which case `None`
            is always returned and errors are never raised, since in this
            case one usually doesn't expect to use the Model to compute
            probabilities (e.g. in germ selection).

        Returns
        -------
        prepLabel : str or None
        opsOnlyString : Circuit
        povmLabel : str or None
        """
        if len(circuit) > 0 and self._shlp.is_prep_lbl(circuit[0]):
            prep_lbl = circuit[0]
            circuit = circuit[1:]
        elif self._shlp.get_default_prep_lbl() is not None:
            prep_lbl = self._shlp.get_default_prep_lbl()
        else:
            if 'prep' in erroron and self._shlp.has_preps():
                raise ValueError("Cannot resolve state prep in %s" % circuit)
            else: prep_lbl = None

        if len(circuit) > 0 and self._shlp.is_povm_lbl(circuit[-1]):
            povm_lbl = circuit[-1]
            circuit = circuit[:-1]
        elif self._shlp.get_default_povm_lbl() is not None:
            povm_lbl = self._shlp.get_default_povm_lbl()
        else:
            if 'povm' in erroron and self._shlp.has_povms():
                raise ValueError("Cannot resolve POVM in %s" % circuit)
            else: povm_lbl = None

        return prep_lbl, circuit, povm_lbl


    def simplify_circuits(self, circuits, dataset=None):
        """
        Simplifies a list of :class:`Circuit`s.

        Circuits must be "simplified" before probabilities can be computed for
        them. Each string corresponds to some number of "outcomes", indexed by an
        "outcome label" that is a tuple of POVM-effect or instrument-element
        labels like "0".  Compiling creates maps between operation sequences and their
        outcomes and the structures used in probability computation (see return
        values below).

        Parameters
        ----------
        circuits : list of Circuits
            The list to simplify.

        dataset : DataSet, optional
            If not None, restrict what is simplified to only those
            probabilities corresponding to non-zero counts (observed
            outcomes) in this data set.

        Returns
        -------
        raw_spamTuples_dict : collections.OrderedDict
            A dictionary whose keys are raw operation sequences (containing just
            "simplified" gates, i.e. not instruments), and whose values are
            lists of (preplbl, effectlbl) tuples.  The effectlbl names a
            "simplified" effect vector; preplbl is just a prep label. Each tuple
            corresponds to a single "final element" of the computation, e.g. a
            probability.  The ordering is important - and is why this needs to be
            an ordered dictionary - when the lists of tuples are concatenated (by
            key) the resulting tuple orderings corresponds to the final-element
            axis of an output array that is being filled (computed).

        elIndices : collections.OrderedDict
            A dictionary whose keys are integer indices into `circuits` and
            whose values are slices and/or integer-arrays into the space/axis of
            final elements.  Thus, to get the final elements corresponding to
            `circuits[i]`, use `filledArray[ elIndices[i] ]`.

        outcomes : collections.OrderedDict
            A dictionary whose keys are integer indices into `circuits` and
            whose values are lists of outcome labels (an outcome label is a tuple
            of POVM-effect and/or instrument-element labels).  Thus, to obtain
            what outcomes the i-th operation sequences's final elements
            (`filledArray[ elIndices[i] ]`)  correspond to, use `outcomes[i]`.

        nTotElements : int
            The total number of "final elements" - this is how big of an array
            is need to hold all of the probabilities `circuits` generates.
        """
        # model.simplify -> odict[raw_gstr] = spamTuples, elementIndices, nElements
        # dataset.simplify -> outcomeLabels[i] = list_of_ds_outcomes, elementIndices, nElements
        # simplify all gsplaq strs -> elementIndices[(i,j)],

        circuits = [ _cir.Circuit(opstr) for opstr in circuits ] # cast to Circuits

        #Indexed by raw operation sequence
        raw_spamTuples_dict = _collections.OrderedDict()  # final
        raw_opOutcomes_dict = _collections.OrderedDict()
        raw_offsets = _collections.OrderedDict()

        #Indexed by parent index (an integer)
        elIndicesByParent = _collections.OrderedDict() # final
        outcomesByParent = _collections.OrderedDict()  # final
        elIndsToOutcomesByParent = _collections.OrderedDict()

        # Helper dict: (rhoLbl,POVM_ELbl) -> (Elbl,) mapping
        def spamTupleToOutcome(spamTuple):
            if spamTuple is None:
                return ("NONE",) #Dummy label for placeholding (see resolveSPAM below)
            else:
                prep_lbl, povm_and_effect_lbl = spamTuple
                last_underscore = povm_and_effect_lbl.rindex('_')
                effect_lbl = povm_and_effect_lbl[last_underscore+1:]
                return (effect_lbl,) # effect label *is* the outcome

        def resolveSPAM(circuit):
            """ Determines spam tuples that correspond to circuit
                and strips any spam-related pieces off """
            prep_lbl, circuit, povm_lbl = \
                self.split_circuit(circuit)
            if prep_lbl is None or povm_lbl is None:
                spamtups = [ None ] #put a single "dummy" spam-tuple placeholder
                  # so that there's a single "element" for each simplified string,
                  # which means that the usual "lookup" or "elIndices" will map
                  # original circuit-list indices to simplified-string, i.e.,
                  # evalTree index, which is useful when computing products
                  # (often the case when a Model has no preps or povms,
                  #  e.g. in germ selection)
            else:
                if dataset is not None:
                    #Then we don't need to consider *all* possible spam tuples -
                    # just the ones that are observed, i.e. that correspond to
                    # a final element in the "full" (tuple) outcome labels that
                    # were observed.
                    observed_povm_outcomes = sorted(set(
                        [full_out_tup[-1] for full_out_tup in dataset[circuit].outcomes] ))
                    spamtups = [ (prep_lbl, povm_lbl + "_" + oout)
                                 for oout in observed_povm_outcomes ]
                      # elbl = oout[-1] -- the last element corresponds
                      # to the POVM (earlier ones = instruments)
                else:
                    spamtups = [ (prep_lbl, povm_lbl + "_" + elbl)
                                 for elbl in self._shlp.get_effect_labels_for_povm(povm_lbl) ]
            return circuit, spamtups

        def process(s, spamtuples, observed_outcomes, elIndsToOutcomes,
                    op_outcomes=(), start=0):
            """
            Implements recursive processing of a string. Separately
            implements two different behaviors:
              "add" : add entries to raw_spamTuples_dict and raw_opOutcomes_dict
              "index" : adds entries to elIndicesByParent and outcomesByParent
                        assuming that raw_spamTuples_dict and raw_opOutcomes_dict
                        are already build (and won't be modified anymore).
            """
            for i,op_label in enumerate(s[start:],start=start):

                # OLD: now allow "gate-level" labels which can contain
                # multiple (parallel) instrument labels
                #if op_label in self.instruments: 
                #    #we've found an instrument - recurse!
                #    for inst_el_lbl in self.instruments[op_label]:
                #        simplified_el_lbl = op_label + "_" + inst_el_lbl
                #        process(s[0:i] + _cir.Circuit((simplified_el_lbl,)) + s[i+1:],
                #                spamtuples, elIndsToOutcomes, op_outcomes + (inst_el_lbl,), i+1)
                #    break

                if any([ self._shlp.is_instrument_lbl(sub_gl) for sub_gl in op_label.components]):
                    # we've found an instrument - recurse!
                    sublabel_tups_to_iter = [] # one per label component (may be only 1)
                    for sub_gl in op_label.components:
                        if self._shlp.is_instrument_lbl(sub_gl):
                            sublabel_tups_to_iter.append( [ (sub_gl,inst_el_lbl)
                                                            for inst_el_lbl in self._shlp.get_member_labels_for_instrument(sub_gl) ])
                        else:
                            sublabel_tups_to_iter.append( [(sub_gl,None)] ) # just a single element
                            
                    for sublabel_tups in _itertools.product(*sublabel_tups_to_iter):
                        sublabels = [] # the sub-labels of the overall operation label to add
                        outcomes = [] # the outcome tuple associated with this overall label
                        for sub_gl,inst_el_lbl in sublabel_tups:
                            if inst_el_lbl is not None:
                                sublabels.append(sub_gl + "_" + inst_el_lbl)
                                outcomes.append(inst_el_lbl)
                            else:
                                sublabels.append(sub_gl)
                                
                        simplified_el_lbl = _Label(sublabels)
                        simplified_el_outcomes = tuple(outcomes)
                        process(s[0:i] + _cir.Circuit((simplified_el_lbl,)) + s[i+1:],
                                spamtuples, observed_outcomes, elIndsToOutcomes,
                                op_outcomes + simplified_el_outcomes, i+1)
                    break
                    
            else: #no instruments -- add "raw" operation sequence s
                if s in raw_spamTuples_dict:
                    assert(op_outcomes == raw_opOutcomes_dict[s]) #DEBUG
                    #if action == "add":
                    od = raw_spamTuples_dict[s] # ordered dict
                    for spamtup in spamtuples:
                        outcome_tup = op_outcomes + spamTupleToOutcome(spamtup)
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
                else:
                    # Note: store elements of raw_spamTuples_dict as dicts for
                    # now, for faster lookup during "index" mode
                    outcome_tuples =  [ op_outcomes + spamTupleToOutcome(x) for x in spamtuples ]
                    
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

                        for ist,out_tup in enumerate(outcome_tuples): # ist = spamtuple index
                            elIndsToOutcomes[(s,ist)] = out_tup # element index is given by (parent_circuit, spamtuple_index) tuple
                              # Note: works even if `i` already exists - doesn't reorder keys then

                    raw_spamTuples_dict[s] = spamtup_dict
                    raw_opOutcomes_dict[s] = op_outcomes #DEBUG

        #Begin actual processing work:

        # Step1: recursively populate raw_spamTuples_dict,
        #        raw_opOutcomes_dict, and elIndsToOutcomesByParent
        resolved_circuits = list(map(resolveSPAM, circuits))
        for iParent,(opstr,spamtuples) in enumerate(resolved_circuits):
            elIndsToOutcomesByParent[iParent] = _collections.OrderedDict()
            oouts = None if (dataset is None) else set(dataset[opstr].outcomes)
            process(opstr,spamtuples, oouts, elIndsToOutcomesByParent[iParent])

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
        #if len(circuits) > 1:
        #    for k,opstr in enumerate(circuits):
        #        _,outcomes_k = self.simplify_circuit(opstr)
        #        nIndices = _slct.length(elIndicesByParent[k]) if isinstance(elIndicesByParent[k], slice) \
        #                      else len(elIndicesByParent[k])
        #        assert(len(outcomes_k) == nIndices)
        #        assert(outcomes_k == outcomesByParent[k])

        #print("Model.simplify debug:")
        #print("input = ",'\n'.join(["%d: %s" % (i,repr(c)) for i,c in enumerate(circuits)]))
        #print("raw_dict = ", raw_spamTuples_dict)
        #print("elIndices = ", elIndicesByParent)
        #print("outcomes = ", outcomesByParent)
        #print("total els = ",nTotElements)

        return (raw_spamTuples_dict, elIndicesByParent,
                outcomesByParent, nTotElements)


    def simplify_circuit(self, circuit):
        """
        Simplifies a single :class:`Circuit`.

        Parameters
        ----------
        circuit : Circuit
            The operation sequence to simplify

        Returns
        -------
        raw_spamTuples_dict : collections.OrderedDict
            A dictionary whose keys are raw operation sequences (containing just
            "simplified" gates, i.e. not instruments), and whose values are
            lists of (preplbl, effectlbl) tuples.  The effectlbl names a
            "simplified" effect vector; preplbl is just a prep label. Each tuple
            corresponds to a single "final element" of the computation for this
            operation sequence.  The ordering is important - and is why this needs to be
            an ordered dictionary - when the lists of tuples are concatenated (by
            key) the resulting tuple orderings corresponds to the final-element
            axis of an output array that is being filled (computed).

        outcomes : list
            A list of outcome labels (an outcome label is a tuple
            of POVM-effect and/or instrument-element labels), corresponding to
            the final elements.
        """
        raw_dict,_,outcomes,nEls = self.simplify_circuits([circuit])
        assert(len(outcomes[0]) == nEls)
        return raw_dict,outcomes[0]

    def probs(self, circuit, clipTo=None):
        """
        Construct a dictionary containing the probabilities of every spam label
        given a operation sequence.

        Parameters
        ----------
        circuit : Circuit or tuple of operation labels
          The sequence of operation labels specifying the operation sequence.

        clipTo : 2-tuple, optional
           (min,max) to clip probabilities to if not None.

        Returns
        -------
        probs : dictionary
            A dictionary such that
            probs[SL] = pr(SL,circuit,clipTo)
            for each spam label (string) SL.
        """
        return self._calc().probs(self.simplify_circuit(circuit), clipTo)


    def dprobs(self, circuit, returnPr=False,clipTo=None):
        """
        Construct a dictionary containing the probability derivatives of every
        spam label for a given operation sequence.

        Parameters
        ----------
        circuit : Circuit or tuple of operation labels
          The sequence of operation labels specifying the operation sequence.

        returnPr : bool, optional
          when set to True, additionally return the probabilities.

        clipTo : 2-tuple, optional
           (min,max) to clip returned probability to if not None.
           Only relevant when returnPr == True.

        Returns
        -------
        dprobs : dictionary
            A dictionary such that
            dprobs[SL] = dpr(SL,circuit,gates,G0,SPAM,SP0,returnPr,clipTo)
            for each spam label (string) SL.
        """
        return self._calc().dprobs(self.simplify_circuit(circuit),
                                   returnPr,clipTo)


    def hprobs(self, circuit, returnPr=False,returnDeriv=False,clipTo=None):
        """
        Construct a dictionary containing the probability derivatives of every
        spam label for a given operation sequence.

        Parameters
        ----------
        circuit : Circuit or tuple of operation labels
          The sequence of operation labels specifying the operation sequence.

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
            hprobs[SL] = hpr(SL,circuit,gates,G0,SPAM,SP0,returnPr,returnDeriv,clipTo)
            for each spam label (string) SL.
        """
        return self._calc().hprobs(self.simplify_circuit(circuit),
                                   returnPr, returnDeriv, clipTo)


    def bulk_evaltree_from_resources(self, circuit_list, comm=None, memLimit=None,
                                     distributeMethod="default", subcalls=[],
                                     dataset=None, verbosity=0):
        """
        Create an evaluation tree based on available memory and CPUs.

        This tree can be used by other Bulk_* functions, and is it's own
        function so that for many calls to Bulk_* made with the same
        circuit_list, only a single call to bulk_evaltree is needed.

        Parameters
        ----------
        circuit_list : list of (tuples or Circuits)
            Each element specifies a operation sequence to include in the evaluation tree.

        comm : mpi4py.MPI.Comm
            When not None, an MPI communicator for distributing computations
            across multiple processors.

        memLimit : int, optional
            A rough memory limit in bytes which is used to determine subtree
            number and size.

        distributeMethod : {"circuits", "deriv"}
            How to distribute calculation amongst processors (only has effect
            when comm is not None).  "circuits" will divide the list of
            circuits and thereby result in more subtrees; "deriv" will divide
            the columns of any jacobian matrices, thereby resulting in fewer
            (larger) subtrees.

        subcalls : list, optional
            A list of the names of the Model functions that will be called
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
        # Let ng = # operation sequence groups == # subtrees, so 1 <= ng <= max_split_num; size of each group = size of corresponding subtree
        # With nprocs processors, split into Ng comms of ~nprocs/Ng procs each.  These comms are each assigned
        #  some number of operation sequence groups, where their ~nprocs/Ng processors are used to partition the np param
        #  groups. Note that 1 <= Ng <= min(ng,nprocs).
        # Notes:
        #  - making np or ng > nprocs can be useful for saving memory.  Raising np saves *Jacobian* and *Hessian*
        #     function memory without evaltree overhead, and I think will typically be preferred over raising
        #     ng which will also save Product function memory but will incur evaltree overhead.
        #  - any given CPU will be running a *single* (ng-index,np-index) pair at any given time, and so many
        #     memory estimates only depend on ng and np, not on Ng.  (The exception is when a routine *gathers*
        #     the end results from a divided computation.)
        #  - "circuits" distributeMethod: never distribute num_params (np == 1, Ng == nprocs always).
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

            nFinalStrs = int(round(len(circuit_list) / ng)) #may not need to be an int...

            if cacheSize is None:
                #Get cache size
                if not fastCacheSz:
                    #Slower (but more accurate way)
                    if ng not in evt_cache:
                        evt_cache[ng] = self.bulk_evaltree(
                            circuit_list, minSubtrees=ng, numSubtreeComms=Ng,
                            dataset=dataset)                        
                        # FUTURE: make a _bulk_evaltree_presimplified version that takes simplified
                        # operation sequences as input so don't have to re-simplify every time we hit this line.
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

        if distributeMethod == "circuits":
            Nstrs = len(circuit_list)
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
                            evt_cache[Nstrs].squeeze(0) #To get here, need to use higher-dim models
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
                    np1 = np2 = max(num_params,1)
                    ng = Ng = set_Ng(nprocs / max(num_params**2,1)) #Note __future__ division
                elif nprocs > num_params:
                    np1 = max(num_params,1)
                    np2 = int(_np.ceil(nprocs / max(num_params,1)))
                else:
                    np1 = nprocs; np2 = 1
            else:
                np2 = 1
                if nprocs > num_params:
                    np1 = max(num_params,1)
                    ng = Ng = set_Ng(nprocs / max(num_params,1))
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



    def bulk_evaltree(self, circuit_list, minSubtrees=None, maxTreeSize=None,
                      numSubtreeComms=1, dataset=None, verbosity=0):
        """
        Create an evaluation tree for all the operation sequences in circuit_list.

        This tree can be used by other Bulk_* functions, and is it's own
        function so that for many calls to Bulk_* made with the same
        circuit_list, only a single call to bulk_evaltree is needed.

        Parameters
        ----------
        circuit_list : list of (tuples or Circuits)
            Each element specifies a operation sequence to include in the evaluation tree.

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
            A dictionary whose keys are integer indices into `circuit_list` and
            whose values are slices and/or integer-arrays into the space/axis of
            final elements returned by the 'bulk fill' routines.  Thus, to get the
            final elements corresponding to `circuits[i]`, use
            `filledArray[ elIndices[i] ]`.

        outcomes : collections.OrderedDict
            A dictionary whose keys are integer indices into `circuit_list` and
            whose values are lists of outcome labels (an outcome label is a tuple
            of POVM-effect and/or instrument-element labels).  Thus, to obtain
            what outcomes the i-th operation sequences's final elements
            (`filledArray[ elIndices[i] ]`)  correspond to, use `outcomes[i]`.
        """
        tm = _time.time()
        printer = _VerbosityPrinter.build_printer(verbosity)

        toCircuit = lambda x : x if isinstance(x,_cir.Circuit) else _cir.Circuit(x)
        circuit_list = list(map(toCircuit,circuit_list)) # make sure simplify_circuits is given Circuits
        simplified_circuits, elIndices, outcomes, nEls = \
                            self.simplify_circuits(circuit_list, dataset)

        evalTree = self._calc().construct_evaltree()
        evalTree.initialize(simplified_circuits, numSubtreeComms)

        printer.log("bulk_evaltree: created initial tree (%d strs) in %.0fs" %
                    (len(circuit_list),_time.time()-tm)); tm = _time.time()

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


    def bulk_probs(self, circuit_list, clipTo=None, check=False,
                   comm=None, memLimit=None, dataset=None, smartc=None):
        """
        Construct a dictionary containing the probabilities
        for an entire list of operation sequences.

        Parameters
        ----------
        circuit_list : list of (tuples or Circuits)
          Each element specifies a operation sequence to compute quantities for.

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

        smartc : SmartCache, optional
            A cache object to cache & use previously cached values inside this
            function.


        Returns
        -------
        probs : dictionary
            A dictionary such that `probs[opstr]` is an ordered dictionary of
            `(outcome, p)` tuples, where `outcome` is a tuple of labels
            and `p` is the corresponding probability.
        """
        circuit_list = [ _cir.Circuit(opstr) for opstr in circuit_list]  # cast to Circuits
        evalTree, _, _, elIndices, outcomes = self.bulk_evaltree_from_resources(
            circuit_list, comm, memLimit, subcalls=['bulk_fill_probs'],
            dataset=dataset, verbosity=0) # FUTURE (maybe make verbosity into an arg?)

        return self._calc().bulk_probs(circuit_list, evalTree, elIndices,
                                       outcomes, clipTo, check, comm, smartc)


    def bulk_dprobs(self, circuit_list, returnPr=False,clipTo=None,
                    check=False,comm=None,wrtBlockSize=None,dataset=None):

        """
        Construct a dictionary containing the probability-derivatives
        for an entire list of operation sequences.

        Parameters
        ----------
        circuit_list : list of (tuples or Circuits)
          Each element specifies a operation sequence to compute quantities for.

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
            A dictionary such that `probs[opstr]` is an ordered dictionary of
            `(outcome, dp, p)` tuples, where `outcome` is a tuple of labels,
            `p` is the corresponding probability, and `dp` is an array containing
            the derivative of `p` with respect to each parameter.  If `returnPr`
            if False, then `p` is not included in the tuples (so they're just
            `(outcome, dp)`).
        """
        circuit_list = [ _cir.Circuit(opstr) for opstr in circuit_list]  # cast to Circuits
        evalTree, elIndices, outcomes = self.bulk_evaltree(circuit_list, dataset=dataset)
        return self._calc().bulk_dprobs(circuit_list, evalTree, elIndices,
                                        outcomes, returnPr,clipTo,
                                        check, comm, None, wrtBlockSize)


    def bulk_hprobs(self, circuit_list, returnPr=False,returnDeriv=False,
                    clipTo=None, check=False, comm=None,
                    wrtBlockSize1=None, wrtBlockSize2=None, dataset=None):

        """
        Construct a dictionary containing the probability-Hessians
        for an entire list of operation sequences.

        Parameters
        ----------
        circuit_list : list of (tuples or Circuits)
          Each element specifies a operation sequence to compute quantities for.

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
            A dictionary such that `probs[opstr]` is an ordered dictionary of
            `(outcome, hp, dp, p)` tuples, where `outcome` is a tuple of labels,
            `p` is the corresponding probability, `dp` is a 1D array containing
            the derivative of `p` with respect to each parameter, and `hp` is a
            2D array containing the Hessian of `p` with respect to each parameter.
            If `returnPr` if False, then `p` is not included in the tuples.
            If `returnDeriv` if False, then `dp` is not included in the tuples.
        """
        circuit_list = [ _cir.Circuit(opstr) for opstr in circuit_list]  # cast to Circuits
        evalTree, elIndices, outcomes = self.bulk_evaltree(circuit_list, dataset=dataset)
        return self._calc().bulk_hprobs(circuit_list, evalTree, elIndices,
                                        outcomes, returnPr, returnDeriv,
                                        clipTo, check, comm, None, None,
                                        wrtBlockSize1, wrtBlockSize2)


    def bulk_fill_probs(self, mxToFill, evalTree, clipTo=None, check=False, comm=None):
        """
        Compute the outcome probabilities for an entire tree of operation sequences.

        This routine fills a 1D array, `mxToFill` with the probabilities
        corresponding to the *simplified* operation sequences found in an evaluation
        tree, `evalTree`.  An initial list of (general) :class:`Circuit`
        objects is *simplified* into a lists of gate-only sequences along with
        a mapping of final elements (i.e. probabilities) to gate-only sequence
        and prep/effect pairs.  The evaluation tree organizes how to efficiently
        compute the gate-only sequences.  This routine fills in `mxToFill`, which
        must have length equal to the number of final elements (this can be
        obtained by `evalTree.num_final_elements()`.  To interpret which elements
        correspond to which strings and outcomes, you'll need the mappings
        generated when the original list of `Circuits` was simplified.

        Parameters
        ----------
        mxToFill : numpy ndarray
          an already-allocated 1D numpy array of length equal to the
          total number of computed elements (i.e. evalTree.num_final_elements())

        evalTree : EvalTree
           given by a prior call to bulk_evaltree.  Specifies the *simplified* gate
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
          number of model parameters.

        evalTree : EvalTree
           given by a prior call to bulk_evaltree.  Specifies the *simplified* gate
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
           given by a prior call to bulk_evaltree.  Specifies the *simplified* gate
           strings to compute the bulk operation on.

        prMxToFill : numpy array, optional
          when not None, an already-allocated length-E numpy array that is filled
          with probabilities, just like in bulk_fill_probs(...).

        derivMxToFill1, derivMxToFill2 : numpy array, optional
          when not None, an already-allocated ExM numpy array that is filled
          with probability derivatives, similar to bulk_fill_dprobs(...), but
          where M is the number of model parameters selected for the 1st and 2nd
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
        the operation sequences.


        Parameters
        ----------
        spam_label_rows : dictionary
          a dictionary with keys == spam labels and values which
          are integer row indices into mxToFill, specifying the
          correspondence between rows of mxToFill and spam labels.

        evalTree : EvalTree
           given by a prior call to bulk_evaltree.  Specifies the operation sequences
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
           d12[iSpamLabel,iOpStr,p1,p2] = dP/d(p1)*dP/d(p2) where P is is
           the probability generated by the sequence and spam label indexed
           by iOpStr and iSpamLabel.  d12 has the same dimensions as the
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
          - S is the number of operation sequences (i.e. evalTree.num_final_strings()),
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

    def _init_copy(self,copyInto):
        """
        Copies any "tricky" member of this model into `copyInto`, before
        deep copying everything else within a .copy() operation.
        """
        self._clean_paramvec() # make sure _paramvec is valid before copying (necessary?)
        copyInto.uuid = _uuid.uuid4() # new uuid for a copy (don't duplicate!)
        copyInto._shlp = None # must be set by a derived-class _init_copy() method
        copyInto._need_to_rebuild = True # copy will have all gpindices = None, etc.


    def copy(self):
        """
        Copy this model.

        Returns
        -------
        Model
            a (deep) copy of this model.
        """
        self._clean_paramvec() # ensure _paramvec is rebuilt if needed
        if Model._pcheck: self._check_paramvec()
        
        #Avoid having to reconstruct everything via __init__;
        # essentially deepcopy this object, but give the
        # class opportunity to initialize tricky members instead
        # of letting deepcopy do it.
        newModel = type(self).__new__(self.__class__) # empty object

        #first call _init_copy to initialize any tricky members
        # (like those that contain references to self or other members)
        self._init_copy(newModel)
        
        for attr,val in self.__dict__.items():
            if not hasattr(newModel,attr):
                setattr(newModel,attr,_copy.deepcopy(val))

        if Model._pcheck: newModel._check_paramvec()
        return newModel

    
    def __str__(self):
        pass

    def __hash__(self):
        if self.uuid is not None:
            return hash(self.uuid)
        else:
            raise TypeError('Use digest hash')






class ExplicitOpModel(Model):
    """
    Encapsulates a set of gate, state preparation, and POVM effect operations.

    An ExplictOpModel stores a set of labeled LinearOperator objects and
    provides dictionary-like access to their matrices.  State preparation
    and POVM effect operations are represented as column vectors.
    """

    #Whether access to gates & spam vecs via Model indexing is allowed
    _strict = False

    def __init__(self, state_space_labels, basis="pp", default_param="full",
                 prep_prefix="rho", effect_prefix="E", gate_prefix="G",
                 povm_prefix="M", instrument_prefix="I", sim_type="auto",
                 evotype="densitymx"):
                 #REMOVE auto_idle_name=None):
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

        basis : Basis
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

        sim_type : {"auto", "matrix", "map", "termorder:<X>"}
            The type of gate sequence / circuit simulation used to compute any
            requested probabilities, e.g. from :method:`probs` or
            :method:`bulk_probs`.  The default value of `"auto"` automatically
            selects the simulation type, and is usually what you want. Allowed
            values are:

            - "matrix" : op_matrix-op_matrix products are computed and
              cached to get composite gates which can then quickly simulate
              a circuit for any preparation and outcome.  High memory demand;
              best for a small number of (1 or 2) qubits.
            - "map" : op_matrix-state_vector products are repeatedly computed
              to simulate circuits.  Slower for a small number of qubits, but
              faster and more memory efficient for higher numbers of qubits (3+).
            - "termorder:<X>" : Use Taylor expansions of gates in error rates
              to compute probabilities out to some maximum order <X> (an
              integer) in these rates.

        evotype : {"densitymx", "statevec", "stabilizer", "svterm", "cterm"}
            The evolution type of this model, describing how states are
            represented, allowing compatibility checks with (super)operator 
            objects.
        """        
        #More options now (TODO enumerate?)
        #assert(default_param in ('full','TP','CPTP','H+S','S','static',
        #                         'H+S terms','clifford','H+S clifford terms'))
        flagfn = lambda typ : { 'auto_embed': True, 'match_parent_dim': True,
                                'match_parent_evotype': True, 'cast_to_type': typ }
        
        self.preps = _ld.OrderedMemberDict(self, default_param, prep_prefix, flagfn("spamvec"))
        self.povms = _ld.OrderedMemberDict(self, default_param, povm_prefix, flagfn("povm"))
        self.operations = _ld.OrderedMemberDict(self, default_param, gate_prefix, flagfn("operation"))
        self.instruments = _ld.OrderedMemberDict(self, default_param, instrument_prefix, flagfn("instrument"))
        self.effects_prefix = effect_prefix
        
        self._default_gauge_group = None

        chelper = MemberDictSimplifierHelper(self.preps, self.povms, self.instruments)
        super(ExplicitOpModel, self).__init__(state_space_labels, basis, evotype, chelper, sim_type)


    def get_primitive_prep_labels(self):
        """ Return the primitive state preparation labels of this model"""
        return tuple(self.preps.keys())

    def set_primitive_prep_labels(self, lbls):
        """ Set the primitive state preparation labels of this model"""
        raise ValueError(("Cannot set the primitive labels of an ExplicitOpModel "
                          "(they're determined by the keys of the model.operations dict)."))

    def get_primitive_povm_labels(self):
        """ Return the primitive POVM labels of this model"""
        return tuple(self.povms.keys())

    def set_primitive_povm_labels(self, lbls):
        """ Set the primitive POVM labels of this model"""
        raise ValueError(("Cannot set the primitive labels of an ExplicitOpModel "
                          "(they're determined by the keys of the model.povms dict)."))
    
    def get_primitive_op_labels(self):
        """ Return the primitive operation labels of this model"""
        return tuple(self.operations.keys())

    def set_primitive_op_labels(self, lbls):
        """ Set the primitive operation labels of this model"""
        raise ValueError(("Cannot set the primitive labels of an ExplicitOpModel "
                          "(they're determined by the keys of the model.operations dict)."))

    def get_primitive_instrument_labels(self):
        """ Return the primitive instrument labels of this model"""
        return tuple(self.instruments.keys())

    def set_primitive_instrument_labels(self):
        """ Set the primitive instrument labels of this model"""
        raise ValueError(("Cannot set the primitive labels of an ExplicitOpModel "
                          "(they're determined by the keys of the model.instrument dict)."))


    #Functions required for base class functionality
    
    def _iter_parameterized_objs(self):
        for lbl,obj in _itertools.chain(self.preps.items(),
                                        self.povms.items(),
                                        self.operations.items(),
                                        self.instruments.items()):
            yield (lbl,obj)
    
    def _layer_lizard(self):
        """ Return a layer lizard for this model """
        self._clean_paramvec() # just to be safe

        simplified_effects = _collections.OrderedDict()
        for povm_lbl,povm in self.povms.items():
            for k,e in povm.simplify_effects(povm_lbl).items():
                simplified_effects[k] = e
        
        simplified_ops = _collections.OrderedDict()
        for k,g in self.operations.items(): simplified_ops[k] = g
        for inst_lbl,inst in self.instruments.items():
            for k,g in inst.simplify_operations(inst_lbl).items():
                simplified_ops[k] = g
        simplified_preps = self.preps

        return ExplicitLayerLizard(simplified_preps, simplified_ops, simplified_effects, self)

    def _excalc(self):
        """ Create & return a special explicit-model calculator for this model """

        self._clean_paramvec() #ensures paramvec is rebuild if needed
        simplified_effects = _collections.OrderedDict()
        for povm_lbl,povm in self.povms.items():
            for k,e in povm.simplify_effects(povm_lbl).items():
                simplified_effects[k] = e
        
        simplified_ops = _collections.OrderedDict()
        for k,g in self.operations.items(): simplified_ops[k] = g
        for inst_lbl,inst in self.instruments.items():
            for k,g in inst.simplify_operations(inst_lbl).items():
                simplified_ops[k] = g
        simplified_preps = self.preps

        return _explicitcalc.ExplicitOpModel_Calc(self.dim, simplified_preps, simplified_ops,
                                                  simplified_effects, self.num_params())

    #Unneeded - just use string processing & rely on effect labels *not* having underscores in them
    #def simplify_spamtuple_to_outcome_label(self, simplified_spamTuple):
    #    #TODO: make this more efficient (prep lbl isn't even used!)
    #    for prep_lbl in self.preps:
    #        for povm_lbl in self.povms:
    #            for elbl in self.povms[povm_lbl]:
    #                if simplified_spamTuple == (prep_lbl, povm_lbl + "_" + elbl):
    #                    return (elbl,) # outcome "label" (a tuple)
    #    raise ValueError("No outcome label found for simplified spamTuple: ", simplified_spamTuple)


    def _embedOperation(self, opTargetLabels, opVal, force=False):
        """
        Called by OrderedMemberDict._auto_embed to create an embedded-gate
        object that embeds `opVal` into the sub-space of
        `self.state_space_labels` given by `opTargetLabels`.

        Parameters
        ----------
        opTargetLabels : list
            A list of `opVal`'s target state space labels.

        opVal : LinearOperator
            The gate object to embed.  Note this should be a legitimate
            LinearOperator-derived object and not just a numpy array.

        force : bool, optional
            Always wrap with an embedded LinearOperator, even if the
            dimension of `opVal` is the full model dimension.

        Returns
        -------
        LinearOperator
            A gate of the full model dimension.
        """
        if self.dim is None:
            raise ValueError("Must set model dimension before adding auto-embedded gates.")
        if self.state_space_labels is None:
            raise ValueError("Must set model.state_space_labels before adding auto-embedded gates.")

        if opVal.dim == self.dim and not force:
            return opVal # if gate operates on full dimension, no need to embed.

        if self._sim_type == "matrix":
            return _op.EmbeddedDenseOp(self.state_space_labels, opTargetLabels, opVal)
        elif self._sim_type in ("map","termorder"):
            return _op.EmbeddedOp(self.state_space_labels, opTargetLabels, opVal)
        else:
            assert(False), "Invalid Model sim type == %s" % str(self._sim_type)


    @property
    def default_gauge_group(self):
        """
        Gets the default gauge group for performing gauge
        transformations on this Model.
        """
        return self._default_gauge_group

    @default_gauge_group.setter
    def default_gauge_group(self, value):
        self._default_gauge_group = value


    @property
    def prep(self):
        """
        The unique state preparation in this model, if one exists.  If not,
        a ValueError is raised.
        """
        if len(self.preps) != 1:
            raise ValueError("'.prep' can only be used on models" +
                             " with a *single* state prep.  This Model has" +
                             " %d state preps!" % len(self.preps))
        return list(self.preps.values())[0]


    @property
    def effects(self):
        """
        The unique POVM in this model, if one exists.  If not,
        a ValueError is raised.
        """
        if len(self.povms) != 1:
            raise ValueError("'.effects' can only be used on models" +
                             " with a *single* POVM.  This Model has" +
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

        if label.has_prefix(self.preps._prefix):
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

        if label.has_prefix(self.preps._prefix):
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
        Convert all gates and SPAM vectors to a specific parameterization
        type.

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
        """
        typ = parameterization_type

        #More options now (TODO enumerate?)
        #assert(parameterization_type in ('full','TP','CPTP','H+S','S','static',
        #                                 'H+S terms','clifford','H+S clifford terms',
        #                                 'static unitary'))

        #Update dim and evolution type so that setting converted elements works correctly
        orig_dim = self.dim
        orig_evotype = self._evotype
        baseType = typ # the default - only updated if a lindblad param type
                
        if typ == 'static unitary':
            assert(self._evotype == "densitymx"), \
                "Can only convert to 'static unitary' from a density-matrix evolution type."
            self._evotype = "statevec"
            self._dim = int(round(_np.sqrt(self.dim))) # reduce dimension d -> sqrt(d)
            if self._sim_type not in ("matrix","map"):
                self.set_simtype("matrix" if self.dim <= 4 else "map")

        elif typ == 'clifford':
            self._evotype = "stabilizer"
            self.set_simtype("map")

        elif _gt.is_valid_lindblad_paramtype(typ):
            baseType,evotype = _gt.split_lindblad_paramtype(typ)
            self._evotype = evotype
            if evotype == "densitymx":
                if self._sim_type not in ("matrix","map"):
                    self.set_simtype("matrix" if self.dim <= 16 else "map")
            elif evotype in ("svterm","cterm"):
                if self._sim_type != "termorder":
                    self.set_simtype("termorder:1")

        else: # assume all other parameterizations are densitymx type
            self._evotype = "densitymx"
            if self._sim_type not in ("matrix","map"):
                self.set_simtype("matrix" if self.dim <= 16 else "map")

        basis = self.basis
        if extra is None: extra = {}

        povmtyp = rtyp = typ #assume spam types are available to all objects
        ityp = "TP" if _gt.is_valid_lindblad_paramtype(typ) else typ

        for lbl,gate in self.operations.items():
            self.operations[lbl] = _op.convert(gate, typ, basis,
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
        else: # typ in ('static','H+S','S', 'H+S terms', ...)
            self.default_gauge_group = _gg.TrivialGaugeGroup(self.dim)

            
    #def __getstate__(self):
    #    #Returns self.__dict__ by default, which is fine

    def __setstate__(self, stateDict):
        
        if "gates" in stateDict:
            #Unpickling an OLD-version Model (or GateSet)
            _warnings.warn("Unpickling deprecated-format ExplicitOpModel (GateSet).  Please re-save/pickle asap.")
            self.operations = stateDict['gates']
            self._state_space_labels = stateDict['stateSpaceLabels']
            self._paramlbls = None
            self._shlp = MemberDictSimplifierHelper(stateDict['preps'], stateDict['povms'], stateDict['instruments'])
            del stateDict['gates']
            del stateDict['_autogator']
            del stateDict['auto_idle_gatename']
            del stateDict['stateSpaceLabels']

        if "effects" in stateDict:
            raise ValueError(("This model (GateSet) object is too old to unpickle - "
                              "try using pyGSTi v0.9.6 to upgrade it to a version "
                              "that this version can upgrade to the current version."))

        #Backward compatibility:
        if 'basis' in stateDict:
            stateDict['_basis'] = stateDict['basis']; del stateDict['basis']
        if 'state_space_labels' in stateDict:
            stateDict['_state_space_labels'] = stateDict['state_space_labels']; del stateDict['_state_space_labels']

        #TODO REMOVE
        #if "effects" in stateDict: #
        #    #unpickling an OLD-version Model - like a re-__init__
        #    #print("DB: UNPICKLING AN OLD GATESET"); print("Keys = ",stateDict.keys())
        #    default_param = "full"
        #    self.preps = _ld.OrderedMemberDict(self, default_param, "rho", "spamvec")
        #    self.povms = _ld.OrderedMemberDict(self, default_param, "M", "povm")
        #    self.effects_prefix = 'E'
        #    self.operations = _ld.OrderedMemberDict(self, default_param, "G", "gate")
        #    self.instruments = _ld.OrderedMemberDict(self, default_param, "I", "instrument")
        #    self._paramvec = _np.zeros(0, 'd')
        #    self._rebuild_paramvec()
        #
        #    self._dim = stateDict['_dim']
        #    self._calcClass = stateDict.get('_calcClass',_matrixfwdsim.MatrixForwardSimulator)
        #    self._evotype = "densitymx"
        #    self.basis = stateDict.get('basis', _Basis('unknown', None))
        #    if self.basis.name == "unknown" and '_basisNameAndDim' in stateDict:
        #        self.basis = _Basis(stateDict['_basisNameAndDim'][0],
        #                            stateDict['_basisNameAndDim'][1])
        #
        #    self._default_gauge_group = stateDict['_default_gauge_group']
        #
        #    assert(len(stateDict['preps']) <= 1), "Cannot convert Models with multiple preps!"
        #    for lbl,gate in stateDict['gates'].items(): self.operations[lbl] = gate
        #    for lbl,vec in stateDict['preps'].items(): self.preps[lbl] = vec
        #
        #    effect_vecs = []; remL = stateDict['_remainderlabel']
        #    comp_lbl = None
        #    for sl,(prepLbl,ELbl) in stateDict['spamdefs'].items():
        #        assert((prepLbl,ELbl) != (remL,remL)), "Cannot convert sum-to-one spamlabel!"
        #        if ELbl == remL:  comp_lbl = str(sl)
        #        else: effect_vecs.append( (str(sl), stateDict['effects'][ELbl]) )
        #    if comp_lbl is not None:
        #        comp_vec = stateDict['_povm_identity'] - sum([v for sl,v in effect_vecs])
        #        effect_vecs.append( (comp_lbl, comp_vec) )
        #        self.povms['Mdefault'] = _povm.TPPOVM(effect_vecs)
        #    else:
        #        self.povms['Mdefault'] = _povm.UnconstrainedPOVM(effect_vecs)
        #
        #else:
        self.__dict__.update(stateDict)

        if 'uuid' not in stateDict:
            self.uuid = _uuid.uuid4() #create a new uuid

        #TODO REMOVE
        #if 'auto_idle_gatename' not in stateDict:
        #    self.auto_idle_gatename = None

        #Additionally, must re-connect this model as the parent
        # of relevant OrderedDict-derived classes, which *don't*
        # preserve this information upon pickling so as to avoid
        # circular pickling...
        self.preps.parent = self
        self.povms.parent = self
        #self.effects.parent = self
        self.operations.parent = self
        self.instruments.parent = self
        for o in self.preps.values(): o.relink_parent(self)
        for o in self.povms.values(): o.relink_parent( self)
        #for o in self.effects.values(): o.relink_parent(self)
        for o in self.operations.values(): o.relink_parent(self)
        for o in self.instruments.values(): o.relink_parent(self)


    def num_elements(self):
        """
        Return the number of total operation matrix and spam vector
        elements in this model.  This is in general different
        from the number of *parameters* in the model, which
        are the number of free variables used to generate all of
        the matrix and vector *elements*.

        Returns
        -------
        int
            the number of model elements.
        """
        rhoSize = [ rho.size for rho in self.preps.values() ]
        povmSize = [ povm.num_elements() for povm in self.povms.values() ]
        opSize = [ gate.size for gate in self.operations.values() ]
        instSize = [ i.num_elements() for i in self.instruments.values() ]
        return sum(rhoSize) + sum(povmSize) + sum(opSize) + sum(instSize)


    def num_nongauge_params(self):
        """
        Return the number of non-gauge parameters when vectorizing
        this model according to the optional parameters.

        Returns
        -------
        int
            the number of non-gauge model parameters.
        """
        return self.num_params() - self.num_gauge_params()


    def num_gauge_params(self):
        """
        Return the number of gauge parameters when vectorizing
        this model according to the optional parameters.

        Returns
        -------
        int
            the number of gauge model parameters.
        """
        dPG = self._excalc()._buildup_dPG()
        gaugeDirs = _mt.nullspace_qr(dPG) #cols are gauge directions
        return _np.linalg.matrix_rank(gaugeDirs[0:self.num_params(),:])


    def deriv_wrt_params(self):
        """
        Construct a matrix whose columns are the vectorized derivatives of all
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
        return self._excalc().get_nongauge_projector(itemWeights, nonGaugeMixMx)


    def transform(self, S):
        """
        Update each of the operation matrices G in this model with inv(S) * G * S,
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

        for opObj in self.operations.values():
            opObj.transform(S)

        for instrument in self.instruments.values():
            instrument.transform(S)

        self._clean_paramvec() #transform may leave dirty members




    def product(self, circuit, bScale=False):
        """
        Compute the product of a specified sequence of operation labels.

        Note: Operator matrices are multiplied in the reversed order of the tuple. That is,
        the first element of circuit can be thought of as the first gate operation
        performed, which is on the far right of the product of matrices.

        Parameters
        ----------
        circuit : Circuit or tuple of operation labels
            The sequence of operation labels.

        bScale : bool, optional
            When True, return a scaling factor (see below).

        Returns
        -------
        product : numpy array
            The product or scaled product of the operation matrices.

        scale : float
            Only returned when bScale == True, in which case the
            actual product == product * scale.  The purpose of this
            is to allow a trace or other linear operation to be done
            prior to the scaling.
        """
        circuit = _cir.Circuit(circuit) # cast to Circuit
        return self._calc().product(circuit, bScale)


    def dproduct(self, circuit, flat=False):
        """
        Compute the derivative of a specified sequence of operation labels.

        Parameters
        ----------
        circuit : Circuit or tuple of operation labels
          The sequence of operation labels.

        flat : bool, optional
          Affects the shape of the returned derivative array (see below).

        Returns
        -------
        deriv : numpy array
            * if flat == False, a M x G x G array, where:

              - M == length of the vectorized model (number of model parameters)
              - G == the linear dimension of a operation matrix (G x G operation matrices).

              and deriv[i,j,k] holds the derivative of the (j,k)-th entry of the product
              with respect to the i-th model parameter.

            * if flat == True, a N x M array, where:

              - N == the number of entries in a single flattened gate (ordering as numpy.flatten)
              - M == length of the vectorized model (number of model parameters)

              and deriv[i,j] holds the derivative of the i-th entry of the flattened
              product with respect to the j-th model parameter.
        """
        circuit = _cir.Circuit(circuit) # cast to Circuit
        return self._calc().dproduct(circuit, flat)


    def hproduct(self, circuit, flat=False):
        """
        Compute the hessian of a specified sequence of operation labels.

        Parameters
        ----------
        circuit : Circuit or tuple of operation labels
          The sequence of operation labels.

        flat : bool, optional
          Affects the shape of the returned derivative array (see below).

        Returns
        -------
        hessian : numpy array
            * if flat == False, a  M x M x G x G numpy array, where:

              - M == length of the vectorized model (number of model parameters)
              - G == the linear dimension of a operation matrix (G x G operation matrices).

              and hessian[i,j,k,l] holds the derivative of the (k,l)-th entry of the product
              with respect to the j-th then i-th model parameters.

            * if flat == True, a  N x M x M numpy array, where:

              - N == the number of entries in a single flattened gate (ordered as numpy.flatten)
              - M == length of the vectorized model (number of model parameters)

              and hessian[i,j,k] holds the derivative of the i-th entry of the flattened
              product with respect to the k-th then k-th model parameters.
        """
        circuit = _cir.Circuit(circuit) # cast to Circuit
        return self._calc().hproduct(circuit, flat)


    def bulk_product(self, evalTree, bScale=False, comm=None):
        """
        Compute the products of many operation sequences at once.

        Parameters
        ----------
        evalTree : EvalTree
           given by a prior call to bulk_evaltree.  Specifies the operation sequences
           to compute the bulk operation on.

        bScale : bool, optional
           When True, return a scaling factor (see below).

        comm : mpi4py.MPI.Comm, optional
           When not None, an MPI communicator for distributing the computation
           across multiple processors.  This is done over operation sequences when a
           *split* evalTree is given, otherwise no parallelization is performed.


        Returns
        -------
        prods : numpy array
            Array of shape S x G x G, where:

            - S == the number of operation sequences
            - G == the linear dimension of a operation matrix (G x G operation matrices).

        scaleValues : numpy array
            Only returned when bScale == True. A length-S array specifying
            the scaling that needs to be applied to the resulting products
            (final_product[i] = scaleValues[i] * prods[i]).
        """
        return self._calc().bulk_product(evalTree, bScale, comm)


    def bulk_dproduct(self, evalTree, flat=False, bReturnProds=False,
                      bScale=False, comm=None):
        """
        Compute the derivative of many operation sequences at once.

        Parameters
        ----------
        evalTree : EvalTree
           given by a prior call to bulk_evaltree.  Specifies the operation sequences
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
           more processors than model parameters, distribution over a split
           evalTree (if given) is possible.


        Returns
        -------
        derivs : numpy array

          * if `flat` is ``False``, an array of shape S x M x G x G, where:

            - S = len(circuit_list)
            - M = the length of the vectorized model
            - G = the linear dimension of a operation matrix (G x G operation matrices)

            and ``derivs[i,j,k,l]`` holds the derivative of the (k,l)-th entry
            of the i-th operation sequence product with respect to the j-th model
            parameter.

          * if `flat` is ``True``, an array of shape S*N x M where:

            - N = the number of entries in a single flattened gate (ordering
              same as numpy.flatten),
            - S,M = as above,

            and ``deriv[i,j]`` holds the derivative of the ``(i % G^2)``-th
            entry of the ``(i / G^2)``-th flattened operation sequence product  with
            respect to the j-th model parameter.

        products : numpy array
          Only returned when `bReturnProds` is ``True``.  An array of shape
          S x G x G; ``products[i]`` is the i-th operation sequence product.

        scaleVals : numpy array
          Only returned when `bScale` is ``True``.  An array of shape S such
          that ``scaleVals[i]`` contains the multiplicative scaling needed for
          the derivatives and/or products for the i-th operation sequence.
        """
        return self._calc().bulk_dproduct(evalTree, flat, bReturnProds,
                                          bScale, comm)


    def bulk_hproduct(self, evalTree, flat=False, bReturnDProdsAndProds=False,
                      bScale=False, comm=None):
        """
        Return the Hessian of many operation sequence products at once.

        Parameters
        ----------
        evalTree : EvalTree
           given by a prior call to bulk_evaltree.  Specifies the operation sequences
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
           model parameters, distribution over a split evalTree (if given)
           is possible.


        Returns
        -------
        hessians : numpy array
            * if flat == False, an  array of shape S x M x M x G x G, where

              - S == len(circuit_list)
              - M == the length of the vectorized model
              - G == the linear dimension of a operation matrix (G x G operation matrices)

              and hessians[i,j,k,l,m] holds the derivative of the (l,m)-th entry
              of the i-th operation sequence product with respect to the k-th then j-th
              model parameters.

            * if flat == True, an array of shape S*N x M x M where

              - N == the number of entries in a single flattened gate (ordering as numpy.flatten),
              - S,M == as above,

              and hessians[i,j,k] holds the derivative of the (i % G^2)-th entry
              of the (i / G^2)-th flattened operation sequence product with respect to
              the k-th then j-th model parameters.

        derivs : numpy array
          Only returned if bReturnDProdsAndProds == True.

          * if flat == False, an array of shape S x M x G x G, where

            - S == len(circuit_list)
            - M == the length of the vectorized model
            - G == the linear dimension of a operation matrix (G x G operation matrices)

            and derivs[i,j,k,l] holds the derivative of the (k,l)-th entry
            of the i-th operation sequence product with respect to the j-th model
            parameter.

          * if flat == True, an array of shape S*N x M where

            - N == the number of entries in a single flattened gate (ordering is
                   the same as that used by numpy.flatten),
            - S,M == as above,

            and deriv[i,j] holds the derivative of the (i % G^2)-th entry of
            the (i / G^2)-th flattened operation sequence product  with respect to
            the j-th model parameter.

        products : numpy array
          Only returned when bReturnDProdsAndProds == True.  An array of shape
          S x G x G; products[i] is the i-th operation sequence product.

        scaleVals : numpy array
          Only returned when bScale == True.  An array of shape S such that
          scaleVals[i] contains the multiplicative scaling needed for
          the hessians, derivatives, and/or products for the i-th operation sequence.
        """
        ret = self._calc().bulk_hproduct(
            evalTree, flat, bReturnDProdsAndProds, bScale, comm)
        if bReturnDProdsAndProds:
            return ret[0:2] + ret[3:] #remove ret[2] == deriv wrt filter2,
                         # which isn't an input param for Model version
        else: return ret


    def frobeniusdist(self, otherModel, transformMx=None,
                      itemWeights=None, normalize=True):
        """
        Compute the weighted frobenius norm of the difference between this
        model and otherModel.  Differences in each corresponding gate
        matrix and spam vector element are squared, weighted (using
        `itemWeights` as applicable), then summed.  The value returned is the
        square root of this sum, or the square root of this sum divided by the
        number of summands if normalize == True.

        Parameters
        ----------
        otherModel : Model
            the other model to difference against.

        transformMx : numpy array, optional
            if not None, transform this model by
            G => inv(transformMx) * G * transformMx, for each operation matrix G
            (and similar for rho and E vectors) before taking the difference.
            This transformation is applied only for the difference and does
            not alter the values stored in this model.

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
        return self._excalc().frobeniusdist(otherModel._excalc(), transformMx,
                                            itemWeights, normalize)

    def residuals(self, otherModel, transformMx=None, itemWeights=None):
        """
        Compute the weighted residuals between two models (the differences
        in corresponding operation matrix and spam vector elements).

        Parameters
        ----------
        otherModel : Model
            the other model to difference against.

        transformMx : numpy array, optional
            if not None, transform this model by
            G => inv(transformMx) * G * transformMx, for each operation matrix G
            (and similar for rho and E vectors) before taking the difference.
            This transformation is applied only for the difference and does
            not alter the values stored in this model.

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
        return self._excalc().residuals(otherModel._excalc(), transformMx, itemWeights)


    def jtracedist(self, otherModel, transformMx=None):
        """
        Compute the Jamiolkowski trace distance between this
        model and otherModel, defined as the maximum
        of the trace distances between each corresponding gate,
        including spam gates.

        Parameters
        ----------
        otherModel : Model
            the other model to difference against.

        transformMx : numpy array, optional
            if not None, transform this model by
            G => inv(transformMx) * G * transformMx, for each operation matrix G
            (and similar for rho and E vectors) before taking the difference.
            This transformation is applied only for the difference and does
            not alter the values stored in this model.

        Returns
        -------
        float
        """
        return self._excalc().jtracedist(otherModel._excalc(), transformMx)


    def diamonddist(self, otherModel, transformMx=None):
        """
        Compute the diamond-norm distance between this
        model and otherModel, defined as the maximum
        of the diamond-norm distances between each
        corresponding gate, including spam gates.

        Parameters
        ----------
        otherModel : Model
            the other model to difference against.

        transformMx : numpy array, optional
            if not None, transform this model by
            G => inv(transformMx) * G * transformMx, for each operation matrix G
            (and similar for rho and E vectors) before taking the difference.
            This transformation is applied only for the difference and does
            not alter the values stored in this model.

        Returns
        -------
        float
        """
        return self._excalc().diamonddist(otherModel._excalc(), transformMx)


    def tpdist(self):
        """
        Compute the "distance" between this model and the space of
        trace-preserving (TP) maps, defined as the sqrt of the sum-of-squared
        deviations among the first row of all operation matrices and the
        first element of all state preparations.
        """
        penalty = 0.0
        for operationMx in list(self.operations.values()):
            penalty += abs(operationMx[0,0] - 1.0)**2
            for k in range(1,operationMx.shape[1]):
                penalty += abs(operationMx[0,k])**2

        op_dim = self.get_dimension()
        firstEl = 1.0 / op_dim**0.25
        for rhoVec in list(self.preps.values()):
            penalty += abs(rhoVec[0,0] - firstEl)**2

        return _np.sqrt(penalty)


    def strdiff(self, otherModel):
        """
        Return a string describing
        the frobenius distances between
        each corresponding gate, state prep,
        and POVM effect.

        Parameters
        ----------
        otherModel : Model
            the other model to difference against.

        Returns
        -------
        str
        """
        s =  "Model Difference:\n"
        s += " Preps:\n"
        for lbl in self.preps:
            s += "  %s = %g\n" % \
                (lbl, _np.linalg.norm(self.preps[lbl].todense()-otherModel.preps[lbl].todense()))

        s += " POVMs:\n"
        for povm_lbl,povm in self.povms.items():
            s += "  %s: " % povm_lbl
            for lbl in povm:
                s += "    %s = %g\n" % \
                     (lbl, _np.linalg.norm(povm[lbl].todense()-otherModel.povms[povm_lbl][lbl].todense()))

        s += " Gates:\n"
        for lbl in self.operations:
            s += "  %s = %g\n" % \
                (lbl, _np.linalg.norm(self.operations[lbl].todense()-otherModel.operations[lbl].todense()))

        if len(self.instruments) > 0:
            s += " Instruments:\n"
            for inst_lbl,inst in self.instruments.items():
                s += "  %s: " % inst_lbl
                for lbl in inst:
                    s += "    %s = %g\n" % \
                         (lbl, _np.linalg.norm(inst[lbl].todense()-otherModel.instruments[inst_lbl][lbl].todense()))

        return s

    def _init_copy(self,copyInto):
        """
        Copies any "tricky" member of this model into `copyInto`, before
        deep copying everything else within a .copy() operation.
        """
        
        # Copy special base class members first
        super(ExplicitOpModel, self)._init_copy(copyInto)
        
        # Copy our "tricky" members
        copyInto.preps = self.preps.copy(copyInto)
        copyInto.povms = self.povms.copy(copyInto)
        copyInto.operations = self.operations.copy(copyInto)
        copyInto.instruments = self.instruments.copy(copyInto)
        copyInto._shlp = MemberDictSimplifierHelper(copyInto.preps, copyInto.povms, copyInto.instruments)

        copyInto._default_gauge_group = self._default_gauge_group #Note: SHALLOW copy


    def __str__(self):
        s = ""
        for lbl,vec in self.preps.items():
            s += "%s = " % str(lbl) + str(vec) + "\n"
        s += "\n"
        for lbl,povm in self.povms.items():
            s += "%s = " % str(lbl) + str(povm) + "\n"
        s += "\n"
        for lbl,gate in self.operations.items():
            s += "%s = \n" % str(lbl) + str(gate) + "\n\n"
        for lbl,inst in self.instruments.items():
            s += "%s = " % str(lbl) + str(inst) + "\n"
        s += "\n"

        return s


    def iter_objs(self):
        for lbl,obj in _itertools.chain(self.preps.items(),
                                        self.povms.items(),
                                        self.operations.items(),
                                        self.instruments.items()):
            yield (lbl,obj)


#TODO: how to handle these given possibility of different parameterizations...
#  -- maybe only allow these methods to be called when using a "full" parameterization?
#  -- or perhaps better to *move* them to the parameterization class
    def depolarize(self, op_noise=None, spam_noise=None, max_op_noise=None,
                   max_spam_noise=None, seed=None):
        """
        Apply depolarization uniformly or randomly to this model's gate
        and/or SPAM elements, and return the result, without modifying the
        original (this) model.  You must specify either op_noise or
        max_op_noise (for the amount of gate depolarization), and  either
        spam_noise or max_spam_noise (for spam depolarization).

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
        newModel = self.copy() # start by just copying the current model
        opDim = self.get_dimension()
        rndm = _np.random.RandomState(seed)

        if max_op_noise is not None:
            if op_noise is not None:
                raise ValueError("Must specify at most one of 'op_noise' and 'max_op_noise' NOT both")

            #Apply random depolarization to each gate
            r = max_op_noise * rndm.random_sample(len(self.operations))
            for i,label in enumerate(self.operations):
                newModel.operations[label].depolarize(r[i])
            r = max_op_noise * rndm.random_sample(len(self.instruments))
            for i,label in enumerate(self.instruments):
                newModel.instruments[label].depolarize(r[i])

        elif op_noise is not None:
            #Apply the same depolarization to each gate
            for label in self.operations:
                newModel.operations[label].depolarize(op_noise)
            for label in self.instruments:
                newModel.instruments[label].depolarize(op_noise)

        if max_spam_noise is not None:
            if spam_noise is not None:
                raise ValueError("Must specify at most  one of 'noise' and 'max_noise' NOT both")

            #Apply random depolarization to each rho and E vector
            r = max_spam_noise * rndm.random_sample( len(self.preps) )
            for (i,lbl) in enumerate(self.preps):
                newModel.preps[lbl].depolarize(r[i])
            r = max_spam_noise * rndm.random_sample( len(self.povms) )
            for label in self.povms:
                newModel.povms[label].depolarize(r[i])

        elif spam_noise is not None:
            #Apply the same depolarization to each gate
            D = _np.diag( [1]+[1-spam_noise]*(opDim-1) )
            for lbl in self.preps:
                newModel.preps[lbl].depolarize(spam_noise)

            # Just depolarize the preps - leave POVMs alone
            #for label in self.povms:
            #    newModel.povms[label].depolarize(spam_noise)

        newModel._clean_paramvec() #depolarize may leave dirty members
        return newModel


    def rotate(self, rotate=None, max_rotate=None, seed=None):
        """
        Apply a rotation uniformly (the same rotation applied to each gate)
        or randomly (different random rotations to each gate) to this model,
        and return the result, without modifying the original (this) model.

        You must specify either 'rotate' or 'max_rotate'. This method currently
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
        newModel = self.copy() # start by just copying model
        dim = self.get_dimension()
        myBasis = self.basis

        if max_rotate is not None:
            if rotate is not None:
                raise ValueError("Must specify exactly one of 'rotate' and 'max_rotate' NOT both")

            #Apply random rotation to each gate
            rndm = _np.random.RandomState(seed)
            r = max_rotate * rndm.random_sample( len(self.operations) * (dim-1) )
            for i,label in enumerate(self.operations):
                rot = _np.array(r[(dim-1)*i:(dim-1)*(i+1)])
                newModel.operations[label].rotate(rot, myBasis)
            r = max_rotate * rndm.random_sample( len(self.instruments) * (dim-1) )
            for i,label in enumerate(self.instruments):
                rot = _np.array(r[(dim-1)*i:(dim-1)*(i+1)])
                newModel.instruments[label].rotate(rot, myBasis)


        elif rotate is not None:
            assert(len(rotate) == dim-1), \
                "Invalid 'rotate' argument. You must supply a tuple of length %d" % (dim-1)
            for label in self.operations:
                newModel.operations[label].rotate(rotate, myBasis)
            for label in self.instruments:
                newModel.instruments[label].rotate(rotate, myBasis)

        else: raise ValueError("Must specify either 'rotate' or 'max_rotate' "
                               + "-- neither was non-None")

        newModel._clean_paramvec() #rotate may leave dirty members
        return newModel


    def randomize_with_unitary(self, scale, seed=None, randState=None):
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

        randState : numpy.random.RandomState
            A RandomState object to generate samples from. Can be useful to set
            instead of `seed` if you want reproducible distribution samples
            across multiple random function calls but you don't want to bother
            with manually incrementing seeds between those calls.

        Returns
        -------
        Model
            the randomized Model
        """
        if randState is None:
            rndm = _np.random.RandomState(seed)
        else:
            rndm = randState

        op_dim = self.get_dimension()
        unitary_dim = int(round(_np.sqrt(op_dim)))
        assert( unitary_dim**2 == op_dim ), \
            "Model dimension must be a perfect square, %d is not" % op_dim

        mdl_randomized = self.copy()

        for opLabel,gate in self.operations.items():
            randMat = scale * (rndm.randn(unitary_dim,unitary_dim) \
                                   + 1j * rndm.randn(unitary_dim,unitary_dim))
            randMat = _np.transpose(_np.conjugate(randMat)) + randMat
                  # make randMat Hermetian: (A_dag + A)^dag = (A_dag + A)
            randUnitary   = _scipy.linalg.expm(-1j*randMat)

            randOp = _gt.unitary_to_process_mx(randUnitary) #in std basis
            randOp = _bt.change_basis(randOp, "std", self.basis)

            mdl_randomized.operations[opLabel] = _op.FullDenseOp(
                            _np.dot(randOp,gate))

        #Note: this function does NOT randomize instruments

        return mdl_randomized


    def increase_dimension(self, newDimension):
        """
        Enlarge the spam vectors and operation matrices of model to a specified
        dimension, and return the resulting inflated model.  Spam vectors
        are zero-padded and operation matrices are padded with 1's on the diagonal
        and zeros on the off-diagonal (effectively padded by identity operation).

        Parameters
        ----------
        newDimension : int
          the dimension of the returned model.  That is,
          the returned model will have rho and E vectors that
          have shape (newDimension,1) and operation matrices with shape
          (newDimension,newDimension)

        Returns
        -------
        Model
            the increased-dimension Model
        """

        curDim = self.get_dimension()
        assert(newDimension > curDim)

        #For now, just create a dumb default state space labels and basis for the new model:
        sslbls = [('L%d'%i,) for i in range(newDimension)] # interpret as independent classical levels
        dumb_basis = _Basis('gm',[1]*newDimension) # act on diagonal density mx to get appropriate
        new_model = ExplicitOpModel(sslbls, dumb_basis, "full", self.preps._prefix, self.effects_prefix,
                              self.operations._prefix, self.povms._prefix,
                              self.instruments._prefix, self._sim_type)
        #new_model._dim = newDimension # dim will be set when elements are added
        new_model.reset_basis() #FUTURE: maybe user can specify how increase is being done?

        addedDim = newDimension-curDim
        vec_zeroPad = _np.zeros( (addedDim,1), 'd')

        #Increase dimension of rhoVecs and EVecs by zero-padding
        for lbl,rhoVec in self.preps.items():
            assert( len(rhoVec) == curDim )
            new_model.preps[lbl] = \
                _sv.FullSPAMVec(_np.concatenate( (rhoVec, vec_zeroPad) ))

        for lbl,povm in self.povms.items():
            assert( povm.dim == curDim )
            effects = [ (elbl,_np.concatenate( (EVec, vec_zeroPad) ))
                        for elbl,EVec in povm.items() ]

            if isinstance(povm, _povm.TPPOVM):
                new_model.povms[lbl] = _povm.TPPOVM(effects)
            else:
                new_model.povms[lbl] = _povm.UnconstrainedPOVM(effects) #everything else

        #Increase dimension of gates by assuming they act as identity on additional (unknown) space
        for opLabel,gate in self.operations.items():
            assert( gate.shape == (curDim,curDim) )
            newOp = _np.zeros( (newDimension,newDimension) )
            newOp[ 0:curDim, 0:curDim ] = gate[:,:]
            for i in range(curDim,newDimension): newOp[i,i] = 1.0
            new_model.operations[opLabel] = _op.FullDenseOp(newOp)

        for instLabel,inst in self.instruments.items():
            inst_ops = []
            for outcomeLbl,gate in inst.items():
                newOp = _np.zeros( (newDimension,newDimension) )
                newOp[ 0:curDim, 0:curDim ] = gate[:,:]
                for i in range(curDim,newDimension): newOp[i,i] = 1.0
                inst_ops.append( (outcomeLbl,_op.FullDenseOp(newOp)) )
            new_model.instruments[instLabel] = _instrument.Instrument(inst_ops)

        return new_model


    def decrease_dimension(self, newDimension):
        """
        Shrink the spam vectors and operation matrices of model to a specified
        dimension, and return the resulting model.

        Parameters
        ----------
        newDimension : int
          the dimension of the returned model.  That is,
          the returned model will have rho and E vectors that
          have shape (newDimension,1) and operation matrices with shape
          (newDimension,newDimension)

        Returns
        -------
        Model
            the decreased-dimension Model
        """
        curDim = self.get_dimension()
        assert(newDimension < curDim)

        #For now, just create a dumb default state space labels and basis for the new model:
        sslbls = [('L%d'%i,) for i in range(newDimension)] # interpret as independent classical levels
        dumb_basis = _Basis('gm',[1]*newDimension) # act on diagonal density mx to get appropriate
        new_model = ExplicitOpModel(sslbls, dumb_basis, "full", self.preps._prefix, self.effects_prefix,
                              self.operations._prefix, self.povms._prefix,
                              self.instruments._prefix, self._sim_type)
        #new_model._dim = newDimension # dim will be set when elements are added
        new_model.reset_basis() #FUTURE: maybe user can specify how decrease is being done?

        #Decrease dimension of rhoVecs and EVecs by truncation
        for lbl,rhoVec in self.preps.items():
            assert( len(rhoVec) == curDim )
            new_model.preps[lbl] = \
                _sv.FullSPAMVec(rhoVec[0:newDimension,:])

        for lbl,povm in self.povms.items():
            assert( povm.dim == curDim )
            effects = [ (elbl,EVec[0:newDimension,:]) for elbl,EVec in povm.items()]

            if isinstance(povm, _povm.TPPOVM):
                new_model.povms[lbl] = _povm.TPPOVM(effects)
            else:
                new_model.povms[lbl] = _povm.UnconstrainedPOVM(effects) #everything else


        #Decrease dimension of gates by truncation
        for opLabel,gate in self.operations.items():
            assert( gate.shape == (curDim,curDim) )
            newOp = _np.zeros( (newDimension,newDimension) )
            newOp[ :, : ] = gate[0:newDimension,0:newDimension]
            new_model.operations[opLabel] = _op.FullDenseOp(newOp)

        for instLabel,inst in self.instruments.items():
            inst_ops = []
            for outcomeLbl,gate in inst.items():
                newOp = _np.zeros( (newDimension,newDimension) )
                newOp[ :, : ] = gate[0:newDimension,0:newDimension]
                inst_ops.append( (outcomeLbl,_op.FullDenseOp(newOp)) )
            new_model.instruments[instLabel] = _instrument.Instrument(inst_ops)

        return new_model

    def kick(self, absmag=1.0, bias=0, seed=None):
        """
        Kick model by adding to each gate a random matrix with values
        uniformly distributed in the interval [bias-absmag,bias+absmag],
        and return the resulting "kicked" model.

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
        for opLabel,gate in self.operations.items():
            delta = absmag * 2.0*(rndm.random_sample(gate.shape)-0.5) + bias
            kicked_gs.operations[opLabel] = _op.FullDenseOp(
                                            kicked_gs.operations[opLabel] + delta )

        #Note: does not alter intruments!
        return kicked_gs


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

        #TODO REMOVE
        #if self.auto_idle_gatename is not None:
        #    # Special case: gatename for a 1-qubit perfect idle (not actually stored as gates)
        #    srep_dict[self.auto_idle_gatename] = _symp.unitary_to_symplectic(_np.identity(2,'d'))

        for gl,gate in self.operations.items():
            if (gfilter is not None) and (gl not in gfilter): continue

            if isinstance(gate, _op.EmbeddedOp):
                assert(isinstance(gate.embedded_op, _op.CliffordOp)), \
                    "EmbeddedClifforGate contains a non-CliffordOp!"
                lbl = gl.name # strip state space labels off since this is a
                              # symplectic rep for the *embedded* gate
                srep = (gate.embedded_op.smatrix,gate.embedded_op.svector)
            elif isinstance(gate, _op.CliffordOp):
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
        Print to stdout relevant information about this model,
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
        for (label,gate) in self.operations.items():
            print(("Choi(%s) in pauli basis = \n" % label,
            _mt.mx_to_string_complex(_jt.jamiolkowski_iso(gate))))
            print(("  --eigenvals = ", sorted(
                [ev.real for ev in _np.linalg.eigvals(
                        _jt.jamiolkowski_iso(gate))] ),"\n"))
        print(("Sum of negative Choi eigenvalues = ", _jt.sum_of_negative_choi_evals(self)))

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
        

class LayerLizard(object):
    """ 
    Helper class for interfacing a Model and a forward simulator
    (which just deals with *simplified* operations).  Can be thought
    of as a "server" of simplified operations for a forward simulator
    which pieces together layer operations from components.
    """
    pass # TODO docstring - add not-implemented members & docstrings?

    
class ExplicitLayerLizard(LayerLizard):
    """
    This layer lizard (see :class:`LayerLizard`) only serves up layer 
    operations it have been explicitly provided upon initialization.
    """
    def __init__(self,preps,ops,effects,model):
        """
        Creates a new ExplicitLayerLizard.

        Parameters
        ----------
        preps, ops, effects : OrderedMemberDict
            Dictionaries of simplified layer operations available for 
            serving to a forwared simulator.

        model : Model
            The model associated with the simplified operations.
        """
        self.preps, self.ops, self.effects = preps,ops,effects
        self.model = model
        
    def get_evotype(self):
        """ 
        Return the evolution type of the operations being served.

        Returns
        -------
        str
        """
        return self.model._evotype

    def get_prep(self,layerlbl):
        """
        Return the (simplified) preparation layer operator given by `layerlbl`.

        Returns
        -------
        LinearOperator
        """
        return self.preps[layerlbl]
    
    def get_effect(self,layerlbl):
        """
        Return the (simplified) POVM effect layer operator given by `layerlbl`.

        Returns
        -------
        LinearOperator
        """
        return self.effects[layerlbl]
    
    def get_operation(self,layerlbl):
        """
        Return the (simplified) layer operation given by `layerlbl`.

        Returns
        -------
        LinearOperator
        """
        return self.ops[layerlbl]

    def from_vector(self, v):
        """
        Re-initialize the simplified operators from model-parameter-vector `v`.

        Parameters
        ----------
        v : numpy.ndarray
            A vector of parameters for `Model` associated with this layer lizard.
        """
        for _,obj in _itertools.chain(self.preps.items(),
                                      self.effects.items(),
                                      self.ops.items()):
            obj.from_vector( v[obj.gpindices] )


class ImplicitLayerLizard(LayerLizard):
    """ 
    This layer lizard (see :class:`LayerLizard`) is used as a base class for
    objects which serve up layer operations for implicit models (and so provide
    logic for how to construct layer operations from model components).
    """
    def __init__(self,preps,ops,effects,model):
        """
        Creates a new ExplicitLayerLizard.

        Parameters
        ----------
        preps, ops, effects : dict
            Dictionaries of :class:`OrderedMemberDict` objects, one per
            "category" of simplified operators.  These are stored and used
            to build layer operations for serving to a forwared simulator.

        model : Model
            The model associated with the simplified operations.
        """
        self.prep_blks, self.op_blks, self.effect_blks = preps,ops,effects
        self.model = model
        
    def get_prep(self,layerlbl):
        """
        Return the (simplified) preparation layer operator given by `layerlbl`.

        Returns
        -------
        LinearOperator
        """
        raise NotImplementedError("ImplicitLayerLizard-derived classes must implement `get_preps`")
    
    def get_effect(self,layerlbl):
        """
        Return the (simplified) POVM effect layer operator given by `layerlbl`.

        Returns
        -------
        LinearOperator
        """
        raise NotImplementedError("ImplicitLayerLizard-derived classes must implement `get_effect`")
    
    def get_operation(self,layerlbl):
        """
        Return the (simplified) layer operation given by `layerlbl`.

        Returns
        -------
        LinearOperator
        """
        raise NotImplementedError("ImplicitLayerLizard-derived classes must implement `get_operation`")

    def get_evotype(self):
        """ 
        Return the evolution type of the operations being served.

        Returns
        -------
        str
        """
        return self.model._evotype

    def from_vector(self, v):
        """
        Re-initialize the simplified operators from model-parameter-vector `v`.

        Parameters
        ----------
        v : numpy.ndarray
            A vector of parameters for `Model` associated with this layer lizard.
        """
        for _,objdict in _itertools.chain(self.prep_blks.items(),
                                          self.effect_blks.items(),
                                          self.op_blks.items()):
            for _,obj in objdict.items():
                obj.from_vector( v[obj.gpindices] )
        
    
class ImplicitOpModel(Model):
    """
    An ImplicitOpModel represents a flexible QIP model whereby only the
    building blocks for layer operations are stored, and custom layer-lizard
    logic is used to construct layer operations from these blocks on an
    on-demand basis.
    """

    def __init__(self, state_space_labels, basis="pp", primitive_labels=None, layer_lizard_class=ImplicitLayerLizard,
                 layer_lizard_args=(), simplifier_helper_class=None,
                 sim_type="auto", evotype="densitymx"):
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

        if primitive_labels is None: primitive_labels = {}
        self._primitive_prep_labels = primitive_labels.get('preps',())
        self._primitive_povm_labels = primitive_labels.get('povms',())
        self._primitive_op_labels = primitive_labels.get('ops',())
        self._primitive_instrument_labels = primitive_labels.get('instruments',())
        
        self._lizardClass = layer_lizard_class
        self._lizardArgs = layer_lizard_args
        
        if simplifier_helper_class is None:
            simplifier_helper_class = ImplicitModelSimplifierHelper
            # by default, assume *_blk members have keys which match the simple
            # labels found in the circuits this model can simulate.
        self.simplifier_helper_class = simplifier_helper_class
        simplifier_helper = simplifier_helper_class(self)
        super(ImplicitOpModel, self).__init__(state_space_labels, basis, evotype,
                                              simplifier_helper, sim_type)


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

    def set_primitive_instrument_labels(self):
        """ Set the primitive instrument labels of this model"""
        self._primitive_instrument_labels = tuple(lbls)

        
    #Functions required for base class functionality

    def _iter_parameterized_objs(self):
        for dictlbl,objdict in _itertools.chain(self.prep_blks.items(),
                                                self.povm_blks.items(),
                                                self.operation_blks.items(),
                                                self.instrument_blks.items()):
            for lbl,obj in objdict.items():
                yield (_Label(dictlbl+":"+lbl.name,lbl.sslbls),obj)
    
    def _layer_lizard(self):
        """ (simplified op server) """
        self._clean_paramvec() # just to be safe
        
        simplified_effect_blks = _collections.OrderedDict()
        for povm_dict_lbl,povmdict in self.povm_blks.items():
            simplified_effect_blks[povm_dict_lbl] = _collections.OrderedDict(
                [(k,e) for povm_lbl,povm in povmdict.items()
                 for k,e in povm.simplify_effects(povm_lbl).items() ])
        
        simplified_op_blks = self.operation_blks.copy() #no compilation needed
        for inst_dict_lbl,instdict in self.instrument_blks.items():
            if inst_dict_lbl not in simplified_op_blks: #only create when needed
                simplified_op_blks[inst_dict_lbl] = _collections.OrderedDict()
            for inst_lbl,inst in instdict.items():
                for k,g in inst.simplify_operations(inst_lbl).items():
                    simplified_op_blks[inst_dict_lbl][k] = g            

        simplified_prep_blks = self.prep_blks.copy() #no compilation needed

        return self._lizardClass(simplified_prep_blks, simplified_op_blks, simplified_effect_blks, self)
          # use self._lizardArgs internally?

    def _init_copy(self,copyInto):
        """
        Copies any "tricky" member of this model into `copyInto`, before
        deep copying everything else within a .copy() operation.
        """
        # Copy special base class members first
        super(ImplicitOpModel, self)._init_copy(copyInto)

        # Copy our "tricky" members
        copyInto.prep_blks = _collections.OrderedDict([ (lbl,prepdict.copy(copyInto))
                                                        for lbl,prepdict in self.prep_blks.items()])
        copyInto.povm_blks = _collections.OrderedDict([ (lbl,povmdict.copy(copyInto))
                                                        for lbl,povmdict in self.povm_blks.items()])
        copyInto.operation_blks = _collections.OrderedDict([ (lbl,opdict.copy(copyInto))
                                                        for lbl,opdict in self.operation_blks.items()])
        copyInto.instrument_blks = _collections.OrderedDict([ (lbl,idict.copy(copyInto))
                                                        for lbl,idict in self.instrument_blks.items()])
        copyInto._shlp = self.simplifier_helper_class(copyInto)


    def __setstate__(self, stateDict):
        self.__dict__.update(stateDict)
        if 'uuid' not in stateDict:
            self.uuid = _uuid.uuid4() #create a new uuid

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
                lbl = gl.name # strip state space labels off since this is a
                              # symplectic rep for the *embedded* gate
                srep = (gate.embedded_op.smatrix,gate.embedded_op.svector)
            elif isinstance(gate, _op.CliffordOp):
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



    def __str__(self):
        s = ""
        for dictlbl,d in self.prep_blks.items():
            for lbl,vec in d.items():
                s += "%s:%s = " % (str(dictlbl),str(lbl)) + str(vec) + "\n"
        s += "\n"
        for dictlbl,d in self.povm_blks.items():
            for lbl,povm in d.items():
                s += "%s:%s = " % (str(dictlbl),str(lbl)) + str(povm) + "\n"
        s += "\n"
        for dictlbl,d in self.operation_blks.items():
            for lbl,gate in d.items():
                s += "%s:%s = \n" % (str(dictlbl),str(lbl)) + str(gate) + "\n\n"
        for dictlbl,d in self.instrument_blks.items():
            for lbl,inst in d.items():
                s += "%s:%s = " % (str(dictlbl),str(lbl)) + str(inst) + "\n"
        s += "\n"

        return s

