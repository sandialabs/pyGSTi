"""
Defines the FirstOrderGaugeInvariantStore class and supporting functionality.
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import numpy as _np
import scipy.sparse as _sps
import copy as _copy
import warnings as _warnings
import itertools as _itertools
import collections as _collections
from pygsti.baseobjs import Basis as _Basis
from pygsti.tools import matrixtools as _mt
from pygsti.tools import optools as _ot
from pygsti.tools import slicetools as _slct
from pygsti.tools import fogitools as _fogit
from pygsti.baseobjs.nicelyserializable import NicelySerializable as _NicelySerializable
from pygsti.baseobjs.label import Label as _Label
from pygsti.baseobjs.errorgenlabel import GlobalElementaryErrorgenLabel as _GlobalElementaryErrorgenLabel
from pygsti.tools.slicetools import slice_hash as _slice_hash
from pygsti.baseobjs.errorgenspace import ErrorgenSpace as _ErrorgenSpace

class FirstOrderGaugeInvariantStore(_NicelySerializable):
    """
    An object that computes and stores the first-order-gauge-invariant quantities of a model.

    Currently, it is only compatible with :class:`ExplicitOpModel` objects.
    """
    def __init__(self, primitive_op_labels, gauge_space, elem_errorgen_labels_by_op, op_errorgen_indices, fogi_directions, 
                 fogi_metadata, dependent_dir_indices, fogv_directions, allop_gauge_action, gauge_space_directions, 
                 norm_order='auto', dependent_fogi_action='drop'):
        """

        Parameters
        ----------
        primitive_op_labels : tuple of Label
            Labels describing the gate set operations

        gauge_space : ErrorgenSpace
            Special way of intersecting the gauge spaces of all ops, see the beginning of from_gauge_action_matrices()
            for a more detailed description.

        elem_errorgen_labels_by_op : dict of Label keys with GlobalElementaryErrorgenLabel or GlobalElementaryErrorgenLabel values
            Elementary error generator labels for every operation in the gate set

        op_errorgen_indices : dict of Label keys with Slices values
            Specifies which indices from the error generator space labels correspond to which operation. For example,
            these allows us to index accordingly into the rows of allop_gauge_action which contain all error generators
            from every operation stacked on top of each other.

        fogi_directions : numpy array
            Array where each column is a FOGI direction, the rows represent the modelmember parameters from which they were
            derived (typically elementary error generators)

        fogi_metadata : list of dictionaries 
            Entry number k of this list contains metadata information about fogi quantity k (column k of fogi_directions). 
            It is organized as a dictionary with the following entries:

                - 'name' (str) : Label describing the FOGI quantity

                - 'abbrev' (str) : An abbreviated version of the label above TODO: more details 

                - 'r' (float) : constant that allows us to convert between the gauge-space-normalized
                       FOGI direction to the errgen-space-normalized version of it.
                       fogi_dir = r * dot(M, epsilon) = r * dot(inv_diff_gauge_action, int_vec)

                - 'gaugespace_dir' (numpy array) : A corresponding gauge direction TODO: more details on their relationship
                                    to FOGI quantities

                - 'opset' (list of Label) : The set of model member operations that are affected by the corresponding FOGI
                           quantity

                - 'raw' (str) : Another version of 'name' without abbreviations

        dependent_dir_indices : list of int
            Specifies the indices of a set of FOGI quantities which, if removed, the remaining FOGI quantities are linearly independent.

        fogv_directions : numpy array
            A list of gauge directions that span the gauge space of the target gate set

        allop_gauge_action : lil_matrix
            Stands for "All operations gauge action". It is a restricted version of each operation's gauge_action stacked together. Ignores elemgens that are not included in the 
            common gauge space. The common gauge space is the intersection of all the individual gauge spaces. TODO: I believe 
            this is done to remove the "trivial"/"meta" gauge from SPAM operations. We should verify this and if so, include
            it in this docstring

        gauge_space_directions : numpy array 
            Contains, when applicable, a gauge-space direction that
            correspondings to the FOGI quantity in fogi_directions.  Such a gauge-space direction
            exists for relational FOGI quantities, where the FOGI quantity is constructed by taking the
            *difference* of a gauge-space action (given by the gauge-space direction) on two operations
            (or sets of operations). TODO: How does one know the corresponding fogi direction? each
            one of these is computed from fogv_directions.

        norm_order : str or int

            Defines the order of the norm to normalize FOGI directions. It should be 1 for normalizing 'S' quantities and 2 for 'H',
            so 'auto' utilizes intelligence.

        dependent_fogi_action : 'drop' or 'mark'

            If 'drop', all linearly dependent FOGI directions are not returned, resulting in a linearly independent set of quantities. 
            If 'mark' linearly dependent FOGI directions are kept and marked by dependent_dir_indices.
        """
        super().__init__()
        self.primitive_op_labels = primitive_op_labels
        self.gauge_space = gauge_space
        self.elem_errorgen_labels_by_op = elem_errorgen_labels_by_op
        self.op_errorgen_indices = op_errorgen_indices
        self.fogi_directions = fogi_directions
        self.fogi_metadata = fogi_metadata
        self.dependent_dir_indices = dependent_dir_indices
        self.fogv_directions = fogv_directions
        self.allop_gauge_action = allop_gauge_action
        self.gauge_space_directions = gauge_space_directions
        self.norm_order = norm_order
        self._dependent_fogi_action = dependent_fogi_action

        self.errorgen_space_op_elem_labels = tuple([(op_label, elem_lbl) for op_label in self.primitive_op_labels
                                                    for elem_lbl in self.elem_errorgen_labels_by_op[op_label]])
        self.fogv_labels = ["%d gauge action" % i for i in range(self.fogv_directions.shape[1])]
    
    @classmethod
    def from_gauge_action_matrices(cls, gauge_action_matrices_by_op, gauge_action_gauge_spaces_by_op, errorgen_coefficient_labels_by_op,
                 op_label_abbrevs=None, dependent_fogi_action='drop', norm_order='auto'):
        """
        Receives the matrices describing the individual gauge action of each operation within the gate set, and massages
        them into appropriate from to be fed into construct_fogi_quantities() which finds FOGI quantities for the corresponding
        gate set. Aditionally, gauge space vectors are computed, and gauge directions that "correspond" to FOGI quantities are 
        identified. With all of this, a FOGIStore object is created and returned.

        Parameters
        ----------
        gauge_action_matrices_by_op : dict of Label keys and numpy array values
            Gauge action of the primitive operator described by Label. For SPAM, 
            "gauge action" matrix is just the identity on the *relevant* space of gauge transformations.
            For gates, it is K - UKU^dag where K is a gauge generator, and U is the ideal gate.

        gauge_action_gauge_spaces_by_op : dict of Label keys and numpy array values
            "Formatted" gauge spaces corresponding to each gauge action from gauge_action_matrices_by_op. 
            This formatting occurs in model.py:_format_gauge_action_matrix() TODO: More details

        errorgen_coefficient_labels_by_op : dict of Label keys and list of Label values
            If indexed by an operation Label, this variable returns a list of error generator labels
            that identify every row of their corresponding gauge action matrix in gauge_action_matrices_by_op

        op_label_abbrevs : dict of Label keys with str values
            The user is able to fill the values of this dictionary with any strings to represent
            the operations in the gate set. typically used to choose shorter names to avoid
            FOGI names that are too long/complicated to read. 


        norm_order : int or 'auto' (Defaults to 'auto')

            Defines the order of the norm to normalize FOGI directions. It should be 1 for 
            normalizing 'S' quantities and 2 for 'H', so 'auto' utilizes intelligence.

        dependent_fogi_action : 'drop' or 'mark' (Defaults to 'drop')

            If 'drop', all linearly dependent FOGI directions are not returned, resulting in a linearly 
            independent set of quantities. If 'mark', linearly dependent FOGI directions are kept and
            marked by dependent_dir_indices.
        """

        primitive_op_labels = tuple(gauge_action_matrices_by_op.keys())

        
        # Construct common gauge space by special way of intersecting the gauge spaces for all the ops
        # Note: the gauge_space of each op is constructed (see `setup_fogi`) so that the gauge action is
        #  zero on any elementary error generator not in the elementary-errorgen basis associated with
        #  gauge_space (gauge_space is the span of linear combos of a elemgen basis that was chosen to
        #  include all possible non-trivial (non-zero) gauge actions by the operator (gauge action is the
        #  *difference* K - UKU^dag in the Lindblad mapping under gauge transform exp(K),  L -> L + K - UKU^dag)
        common_gauge_space = None
        for op_label, gauge_space in gauge_action_gauge_spaces_by_op.items():
            
            if common_gauge_space is None:
                common_gauge_space = gauge_space
            else:
                common_gauge_space = common_gauge_space.intersection(gauge_space,
                                                                     free_on_unspecified_space=True,
                                                                     use_nice_nullspace=True)

        # column space of self.fogi_directions
        #FOGI DEBUG print("DEBUG common gauge space of has dim", common_gauge_space.vectors.shape[1])
        common_gauge_space.normalize()
        gauge_space = common_gauge_space

        # row space of self.fogi_directions - "errgen-set space" lookups
        # -- maybe make this into an "ErrgenSetSpace" object in FUTURE?
        elem_errorgen_labels_by_op = errorgen_coefficient_labels_by_op

        op_errorgen_indices = _fogit._create_op_errgen_indices_dict(primitive_op_labels,
                                                                         elem_errorgen_labels_by_op)
        errorgen_space_op_elem_labels = tuple([(op_label, elem_lbl) for op_label in primitive_op_labels
                                                    for elem_lbl in elem_errorgen_labels_by_op[op_label]])
        # above is same as flattened self.elem_errorgen_labels_by_op - labels final "row basis" of fogi dirs

        num_elem_errgens = sum([len(labels) for labels in elem_errorgen_labels_by_op.values()])
        allop_gauge_action = _sps.lil_matrix((num_elem_errgens, gauge_space.vectors.shape[1]), dtype=complex)

        # Now update (restrict) each op's gauge_action to use common gauge space
        # - need to write the vectors of the common (final) gauge space, w_i, as linear combos of
        #   the op's original gauge space vectors, v_i.
        # - ignore elemgens that are not in the op's orig_gauge_space's elemgen basis
        # W = V * alpha, and we want to find alpha.  W and V are the vectors in the op's elemgen basis
        #  (these could be seen as staring in the union of the common gauge's and op's elemgen
        #   bases - which would just be the common gauge's elemgen basis since it's strictly larger -
        #   restricted to the op's elemben basis)
        
        for op_label, orig_gauge_space in gauge_action_gauge_spaces_by_op.items():
            #FOGI DEBUG print("DEBUG: ", op_label, orig_gauge_space.vectors.shape, len(orig_gauge_space.elemgen_basis))
            gauge_action = gauge_action_matrices_by_op[op_label]  # a sparse matrix

            op_elemgen_lbls = orig_gauge_space.elemgen_basis.labels
            W = common_gauge_space.vectors[common_gauge_space.elemgen_basis.label_indices(op_elemgen_lbls), :]
            V = orig_gauge_space.vectors
            alpha = _np.dot(_np.linalg.pinv(V), W)  # make SPARSE compatible in future if space vectors are sparse
            alpha = _sps.csr_matrix(alpha)  # convert to dense -> CSR for now, as if we did sparse math above

            # update gauge action to use common gauge space
            sparse_gauge_action = gauge_action.dot(alpha)
            allop_gauge_action[op_errorgen_indices[op_label], :] = sparse_gauge_action[:, :]
            gauge_action_matrices_by_op[op_label] = sparse_gauge_action.toarray()  # make **DENSE** here
            # Hopefully matrices aren't too large after this reduction and dense matrices are ok,
            # otherwise we need to change downstream nullspace and pseduoinverse operations to be sparse compatible.

            #FUTURE: if update above creates zero-rows in gauge action matrix, maybe remove
            # these rows from the row basis, i.e. self.elem_errorgen_labels_by_op[op_label]


        gauge_action_for_op = gauge_action_matrices_by_op


        (indep_fogi_directions, indep_fogi_metadata, dep_fogi_directions, dep_fogi_metadata) = \
            _fogit.construct_fogi_quantities(primitive_op_labels, gauge_action_for_op,
                                             elem_errorgen_labels_by_op, op_errorgen_indices,
                                             gauge_space, op_label_abbrevs, dependent_fogi_action, norm_order)
        fogi_directions = _sps.hstack((indep_fogi_directions, dep_fogi_directions))
        fogi_metadata = indep_fogi_metadata + dep_fogi_metadata  # list concatenation
        dependent_dir_indices = _np.arange(len(indep_fogi_metadata), len(fogi_metadata))
        for j, meta in enumerate(fogi_metadata):
            meta['raw'] = _fogit.op_elem_vec_name(fogi_directions[:, j], errorgen_space_op_elem_labels,
                                                  op_label_abbrevs if (op_label_abbrevs is not None) else {})

        assert(len(errorgen_space_op_elem_labels) == fogi_directions.shape[0])

        # Note: currently PUNT on doing below with sparse math, as a sparse nullspace routine is unavailable

        #First order gauge *variant* directions (the complement of FOGI directions in errgen set space)
        # (directions in errorgen space that correspond to gauge transformations -- to first order)
        #fogv_directions = _mt.nice_nullspace(self.fogi_directions.T)  # can be dependent!
        fogv_directions = _mt.nullspace(fogi_directions.toarray().T)  # complement of fogi directions
        fogv_directions = _sps.csc_matrix(fogv_directions)  # as though we used sparse math above
        #self.fogv_labels = ["%s gauge action" % nm
        #                    for nm in _fogit.elem_vec_names(gauge_space_directions, gauge_elemgen_labels)]

        #Get gauge-space directions corresponding to the fogv directions
        # (pinv_allop_gauge_action takes errorgen-set -> gauge-gen space)
        pinv_allop_gauge_action = _np.linalg.pinv(allop_gauge_action.toarray(), rcond=1e-7)
        gauge_space_directions = pinv_allop_gauge_action @ fogv_directions.toarray()  # in gauge-generator space

        #Notes on error-gen vs gauge-gen space:
        # self.fogi_directions and self.fogv_directions are dual vectors in error-generator space,
        # i.e. with elements corresponding to the elementary error generators given by
        # self.errorgen_space_op_elem_labels.
        # self.fogi_gauge_space_directions contains, when applicable, a gauge-space direction that
        # correspondings to the FOGI quantity in self.fogi_directions.  Such a gauge-space direction
        # exists for relational FOGI quantities, where the FOGI quantity is constructed by taking the
        # *difference* of a gauge-space action (given by the gauge-space direction) on two operations
        # (or sets of operations).

        
        fogi_store = cls(primitive_op_labels, gauge_space, elem_errorgen_labels_by_op, op_errorgen_indices, fogi_directions, fogi_metadata, dependent_dir_indices, fogv_directions, allop_gauge_action, gauge_space_directions, norm_order, dependent_fogi_action)
        fogi_store._check_fogi_store()
        return fogi_store
        #Assertions to check that everything looks good
    def _check_fogi_store(self):
        fogi_dirs = self.fogi_directions.toarray()  # don't bother with sparse math yet
        fogv_dirs = self.fogv_directions.toarray()

        # We must reduce X_gauge_action to the "in-model gauge space" before testing if the computed vecs are FOGI:
        assert(_np.linalg.norm(_np.dot(self.allop_gauge_action.toarray().T, fogi_dirs)) < 1e-8)

        #Check that pseudo-inverse was computed correctly (~ matrices are full rank)
        # fogi_coeffs = dot(fogi_directions.T, elem_errorgen_vec), where elem_errorgen_vec is filled from model
        #                 params, since fogi_directions columns are *dual* vectors in error-gen space.  Thus,
        #                 to go in reverse:
        # elem_errogen_vec = dot(pinv_fogi_dirs_T, fogi_coeffs), where dot(fogi_directions.T, pinv_fogi_dirs_T) == I
        # (This will only be the case when fogi_vecs are linearly independent, so when dependent_indices == 'drop')

        if self._dependent_fogi_action == 'drop':
            #assert(_mt.columns_are_orthogonal(self.fogi_directions))  # not true unless we construct them so...
            assert(_np.linalg.norm(_np.dot(fogi_dirs.T, _np.linalg.pinv(fogi_dirs.T))
                                    - _np.identity(fogi_dirs.shape[1], 'd')) < 1e-6)

        # A similar relationship should always hold for the gauge directions, except for these we never
        #  keep linear dependencies
        assert(_mt.columns_are_orthogonal(fogv_dirs))
        assert(_np.linalg.norm(_np.dot(fogv_dirs.T, _np.linalg.pinv(fogv_dirs.T))
                                - _np.identity(fogv_dirs.shape[1], 'd')) < 1e-6)
    def _to_nice_serialization(self):
        state = super()._to_nice_serialization()
        ops_labels = self.primitive_op_labels
        encoded_metadata = []
        for object in self.fogi_metadata:
            encoded_gauge_space = self._encodemx(object['gaugespace_dir'])
            native_opset_labels = [label.to_native() for label in object['opset']]
            object_copy = object.copy()
            object_copy['gaugespace_dir'] = encoded_gauge_space
            object_copy['opset'] = native_opset_labels
            encoded_metadata.append(object_copy)
        
        state.update({'ops' : [op.to_native() for op in ops_labels],
                      'elem_errorgen_labels_by_op_keys' : [op.to_native() for op in self.elem_errorgen_labels_by_op.keys()],
                      'elem_errorgen_labels_by_op_values' : [([label.__str__() for label in errgen_labels]) for errgen_labels in self.elem_errorgen_labels_by_op.values()],
                      'op_errorgen_indices_keys' : [op.to_native() for op in self.op_errorgen_indices.keys()],
                      'op_errorgen_indices_values' : [_slice_hash(slice) for slice in self.op_errorgen_indices.values()],
                      'fogv_directions' : self._encodemx(self.fogv_directions),
                      'fogi_directions' : self._encodemx(self.fogi_directions),
                      'fogi_metadata' : encoded_metadata,
                      'dependent_dir_indices' : self._encodemx(self.dependent_dir_indices),
                      'allop_gauge_action' : self._encodemx(self.allop_gauge_action),
                      'gauge_space_directions' : self._encodemx(self.gauge_space_directions),
                      'norm_order' : self.norm_order,
                      'dependent_fogi_action' : self._dependent_fogi_action,
                      'gauge_space' : self.gauge_space._to_nice_serialization()
                      })
        return state
    @classmethod
    def _from_nice_serialization(cls, state):  
        primitive_op_labels = [_Label(op) for op in state['ops']]
        elem_errorgen_labels_by_op_keys = [_Label(op) for op in state['elem_errorgen_labels_by_op_keys']]
        elem_errorgen_labels_by_op_values = tuple(tuple(_GlobalElementaryErrorgenLabel.cast(label) for label in errgen_labels) for errgen_labels in state['elem_errorgen_labels_by_op_values'])
        elem_errorgen_labels_by_op = {op_label : errgen_labels for op_label, errgen_labels in zip(elem_errorgen_labels_by_op_keys, elem_errorgen_labels_by_op_values)}
        op_errorgen_indices_keys = [_Label(op) for op in state['op_errorgen_indices_keys']]
        op_errorgen_indices =  {op_label :slice(*slice_args) for op_label, slice_args in zip(op_errorgen_indices_keys, state['op_errorgen_indices_values']) }
        fogv_directions = cls._decodemx(state['fogv_directions'])
        fogi_directions = cls._decodemx(state['fogi_directions'])
        
        for object in state['fogi_metadata']:
            object['gaugespace_dir'] = cls._decodemx(object['gaugespace_dir'])
            object['opset'] = tuple(_Label(label) for label in object['opset'])
        fogi_metadata = state['fogi_metadata']
        dependent_dir_indices = cls._decodemx(state['dependent_dir_indices'])
        allop_gauge_action = cls._decodemx(state['allop_gauge_action'])
        gauge_space_directions = cls._decodemx(state['gauge_space_directions'])
        dependent_fogi_action = state['dependent_fogi_action']
        gauge_space = _ErrorgenSpace.from_nice_serialization(state['gauge_space'])
        norm_order = state['norm_order']

        return cls(primitive_op_labels, gauge_space, elem_errorgen_labels_by_op, op_errorgen_indices, fogi_directions, fogi_metadata, dependent_dir_indices, fogv_directions, allop_gauge_action, gauge_space_directions, norm_order, dependent_fogi_action)
    
    def find_nice_fogiv_directions(self):
        # BELOW: an attempt to find nice FOGV directions - but we'd like all the vecs to be
        #  orthogonal and this seems to interfere with that, so we'll just leave the fogv dirs messy for now.
        #
        # # like to find LCs mix s.t.  dot(gauge_space_directions, mix) ~= identity, so use pinv
        # # then propagate this mixing to fogv_directions = dot(allop_gauge_action, mixed_gauge_space_directions)
        # mix = _np.linalg.pinv(gauge_space_directions)[:, 0:fogv_directions.shape[1]]  # use "full-rank" part of pinv
        # mixed_gauge_space_dirs = _np.dot(gauge_space_directions, mix)
        #
        # #TODO - better mix matrix?
        # #print("gauge_space_directions shape = ",gauge_space_directions.shape)
        # #print("mixed_gauge_space_dirs = ");  _mt.print_mx(gauge_space_directions, width=6, prec=2)
        # #U, s, Vh = _np.linalg.svd(gauge_space_directions, full_matrices=True)
        # #inv_s = _np.array([1/x if abs(x) > 1e-4 else 0 for x in s])
        # #print("shapes = ",U.shape, s.shape, Vh.shape)
        # #print("s = ",s)
        # #print(_np.linalg.norm(_np.dot(U,_np.conjugate(U.T)) - _np.identity(U.shape[0])))
        # #_mt.print_mx(U, width=6, prec=2)
        # #print("U * Udag = ")
        # #_mt.print_mx(_np.dot(U,_np.conjugate(U.T)), width=6, prec=2)
        # #print(_np.linalg.norm(_np.dot(Vh,Vh.T) - _np.identity(Vh.shape[0])))
        # #full_mix = _np.dot(Vh.T, _np.dot(_np.diag(inv_s), U.T))  # _np.linalg.pinv(gauge_space_directions)
        # #full_mixed_gauge_space_dirs = _np.dot(gauge_space_directions, full_mix)
        # #print("full_mixed_gauge_space_dirs = ");  _mt.print_mx(full_mixed_gauge_space_dirs, width=6, prec=2)
        #
        # self.fogv_labels = ["%s gauge action" % nm
        #                     for nm in _fogit.elem_vec_names(mixed_gauge_space_dirs, gauge_elemgen_labels)]
        # self.fogv_directions = _np.dot(self.allop_gauge_action, mixed_gauge_space_dirs)
        # # - directions in errorgen space that correspond to gauge transformations (to first order)

        #UNUSED - maybe useful just as a check? otherwise REMOVE
        #pinv_allop_gauge_action = _np.linalg.pinv(self.allop_gauge_action, rcond=1e-7)  # maps error -> gauge-gen space
        #gauge_space_directions = _np.dot(pinv_allop_gauge_action, self.fogv_directions)  # in gauge-generator space
        #assert(_np.linalg.matrix_rank(gauge_space_directions) <= self._gauge_space_dim)  # should be nearly full rank
        pass
    
    def __eq__(self, other):

        if other is None:
            return False
        assert isinstance(other, FirstOrderGaugeInvariantStore), 'Object provided is not of type FirstOrderGaugeInvariantStore'

        if [label.__str__() for label in self.primitive_op_labels] != [label.__str__() for label in other.primitive_op_labels]:
            return False
        
        if not self.gauge_space.__eq__(other.gauge_space):
            return False
        
        if [op.__str__() for op in self.elem_errorgen_labels_by_op.keys()] != [op.__str__() for op in other.elem_errorgen_labels_by_op.keys()]:
            return False
        
        for errgens_self, errgens_other in zip(list(self.elem_errorgen_labels_by_op.values()), list(other.elem_errorgen_labels_by_op.values())):
            for errgen_self, errgen_other in zip(errgens_self, errgens_other):
                if errgen_self.__str__() != errgen_other.__str__():
                    return False
        
        if not (self.fogi_directions != other.fogi_directions).nnz == 0:
            return False
        
        for object_self, object_other in zip(self.fogi_metadata, other.fogi_metadata):

            if (object_self['name'] != object_other['name']) or (object_self['abbrev'] != object_other['abbrev']) or (object_self['r'] != object_other['r']) \
                or (object_self['raw'] != object_other['raw']):
                return False
            #Case for when one of the spaces is None but the other isn't
            if (object_self['gaugespace_dir'] is not None and object_other['gaugespace_dir'] is None) or \
            (object_self['gaugespace_dir'] is None and object_other['gaugespace_dir'] is not None):
                return False
            #If they are both not none, check that their matrices are equal
            if object_self['gaugespace_dir'] is not None and object_other['gaugespace_dir'] is not None:
                if not _np.allclose(object_self['gaugespace_dir'], object_other['gaugespace_dir']):
                    return False
            if [label.__str__() for label in object_self['opset']] != [label.__str__() for label in object_other['opset']]:
                return False
        
        if not _np.allclose(self.dependent_dir_indices, other.dependent_dir_indices):
            return False
        
        if not (self.fogv_directions != other.fogv_directions).nnz == 0:
            return False 
        
        if not (self.allop_gauge_action != other.allop_gauge_action).nnz == 0:
            return False
        #Case for when one of the spaces is none but the other isn't
        if (self.gauge_space_directions is not None and other.gauge_space_directions is None) or \
            (self.gauge_space_directions is None and other.gauge_space_directions is not None):
            return False
        #If they are both not none, check that their matrices are equal
        if self.gauge_space_directions is not None and other.gauge_space_directions is not None:
            if not _np.allclose(self.gauge_space_directions, other.gauge_space_directions):
                return False
        
        if self.norm_order != other.norm_order or self._dependent_fogi_action != other._dependent_fogi_action:
            return False
        return True
        
        '''
        self.primitive_op_labels = primitive_op_labels
        self.gauge_space = gauge_space
        self.elem_errorgen_labels_by_op = elem_errorgen_labels_by_op
        self.op_errorgen_indices = op_errorgen_indices
        self.fogi_directions = fogi_directions
        self.fogi_metadata = fogi_metadata
        self.dependent_dir_indices = dependent_dir_indices
        self.fogv_directions = fogv_directions
        self.allop_gauge_action = allop_gauge_action
        self.gauge_space_directions = gauge_space_directions
        self.norm_order = norm_order
        self._dependent_fogi_action = dependent_fogi_action

        self.errorgen_space_op_elem_labels = tuple([(op_label, elem_lbl) for op_label in self.primitive_op_labels
                                                    for elem_lbl in self.elem_errorgen_labels_by_op[op_label]])
        self.fogv_labels = ["%d gauge action" % i for i in range(self.fogv_directions.shape[1])]'''
    @property
    def errorgen_space_dim(self):
        return self.fogi_directions.shape[0]

    @property
    def gauge_space_dim(self):
        return self.gauge_space.vectors.shape[1]

    @property
    def num_fogi_directions(self):
        return self.fogi_directions.shape[1]

    @property
    def num_fogv_directions(self):
        return self.fogv_directions.shape[1]

    def fogi_errorgen_direction_labels(self, typ='normal'):
        """ typ can be 'raw' or 'abbrev' too """
        if typ == 'normal': return tuple([meta['name'] for meta in self.fogi_metadata])
        elif typ == 'raw': return tuple([meta['raw'] for meta in self.fogi_metadata])
        elif typ == 'abrev': return tuple([meta['abbrev'] for meta in self.fogi_metadata])
        else: raise ValueError("Invalid `typ` argument: %s" % str(typ))

    def fogv_errorgen_direction_labels(self, typ='normal'):
        if typ == 'normal': labels = self.fogv_labels
        else: labels = [''] * len(self.fogv_labels)
        return tuple(labels)

    def errorgen_vec_to_fogi_components_array(self, errorgen_vec):
        fogi_coeffs = self.fogi_directions.transpose().dot(errorgen_vec)
        assert(_np.linalg.norm(fogi_coeffs.imag) < 1e-8)
        return fogi_coeffs.real

    def errorgen_vec_to_fogv_components_array(self, errorgen_vec):
        fogv_coeffs = self.fogv_directions.transpose().dot(errorgen_vec)
        assert(_np.linalg.norm(fogv_coeffs.imag) < 1e-8)
        return fogv_coeffs.real

    def opcoeffs_to_fogi_components_array(self, op_coeffs):
        errorgen_vec = _np.zeros(self.errorgen_space_dim, 'd')
        for i, (op_label, elem_lbl) in enumerate(self.errorgen_space_op_elem_labels):
            errorgen_vec[i] += op_coeffs[op_label].get(elem_lbl, 0.0)
        return self.errorgen_vec_to_fogi_components_array(errorgen_vec)

    def opcoeffs_to_fogv_components_array(self, op_coeffs):
        errorgen_vec = _np.zeros(self.errorgen_space_dim, 'd')
        for i, (op_label, elem_lbl) in enumerate(self.errorgen_space_op_elem_labels):
            errorgen_vec[i] += op_coeffs[op_label].get(elem_lbl, 0.0)
        return self.errorgen_vec_to_fogv_components_array(errorgen_vec)

    def opcoeffs_to_fogiv_components_array(self, op_coeffs):
        errorgen_vec = _np.zeros(self.errorgen_space_dim, 'd')
        for i, (op_label, elem_lbl) in enumerate(self.errorgen_space_op_elem_labels):
            errorgen_vec[i] += op_coeffs[op_label].get(elem_lbl, 0.0)
        out1 = self.errorgen_vec_to_fogi_components_array(errorgen_vec)
        out2 = self.errorgen_vec_to_fogv_components_array(errorgen_vec)
        return out1, out2

    def fogi_components_array_to_errorgen_vec(self, fogi_components):
        assert(self._dependent_fogi_action == 'drop'), \
            ("Cannot convert *from* fogi components to an errorgen-set vec when fogi directions are linearly-dependent!"
             "  (Set `dependent_fogi_action='drop'` to ensure directions are independent.)")
        # DENSE - need to use sparse solve to enact sparse pinv on vector TODO
        return _np.dot(_np.linalg.pinv(self.fogi_directions.toarray().T, rcond=1e-7), fogi_components)

    def fogv_components_array_to_errorgen_vec(self, fogv_components):
        assert(self._dependent_fogi_action == 'drop'), \
            ("Cannot convert *from* fogv components to an errorgen-set vec when fogi directions are linearly-dependent!"
             "  (Set `dependent_fogi_action='drop'` to ensure directions are independent.)")
        # DENSE - need to use sparse solve to enact sparse pinv on vector TODO
        return _np.dot(_np.linalg.pinv(self.fogv_directions.toarray().T, rcond=1e-7), fogv_components)

    def fogiv_components_array_to_errorgen_vec(self, fogi_components, fogv_components):
        assert(self._dependent_fogi_action == 'drop'), \
            ("Cannot convert *from* fogiv components to an errorgen-set vec when fogi directions are "
             "linearly-dependent!  (Set `dependent_fogi_action='drop'` to ensure directions are independent.)")
        # DENSE - need to use sparse solve to enact sparse pinv on vector TODO
        return _np.dot(_np.linalg.pinv(
            _np.concatenate((self.fogi_directions.toarray(), self.fogv_directions.toarray()), axis=1).T,
            rcond=1e-7), _np.concatenate((fogi_components, fogv_components)))

    def errorgen_vec_to_opcoeffs(self, errorgen_vec):
        op_coeffs = {op_label: {} for op_label in self.primitive_op_labels}
        for (op_label, elem_lbl), coeff_value in zip(self.errorgen_space_op_elem_labels, errorgen_vec):
            op_coeffs[op_label][elem_lbl] = coeff_value
        return op_coeffs

    def fogi_components_array_to_opcoeffs(self, fogi_components):
        return self.errorgen_vec_to_opcoeffs(self.fogi_components_array_to_errorgen_vec(fogi_components))

    def fogv_components_array_to_opcoeffs(self, fogv_components):
        return self.errorgen_vec_to_opcoeffs(self.fogv_components_array_to_errorgen_vec(fogv_components))

    def fogiv_components_array_to_opcoeffs(self, fogi_components, fogv_components):
        return self.errorgen_vec_to_opcoeffs(self.fogiv_components_array_to_errorgen_vec(
            fogi_components, fogv_components))

    def create_binned_fogi_infos(self, tol=1e-5):
        """
        Creates an 'info' dictionary for each FOGI quantity and places it within a
        nested dictionary structure by the operators involved, the types of error generators,
        and the qubits acted upon (a.k.a. the "target" qubits).
        TODO: docstring

        Returns
        -------
        dict
        """

        # Construct a dict of information for each elementary error-gen basis element (the basis for error-gen space)
        elemgen_info = {}
        for k, (op_label, eglabel) in enumerate(self.errorgen_space_op_elem_labels):
            elemgen_info[k] = {
                'type': eglabel.errorgen_type,
                'qubits': eglabel.sslbls,
                'op_label': op_label,
                'elemgen_label': eglabel,
            }

        bins = {}
        dependent_indices = set(self.dependent_dir_indices)  # indices of one set of linearly dep. fogi dirs
        for i, meta in enumerate(self.fogi_metadata):
            fogi_dir = self.fogi_directions[:, i].toarray().ravel()

            label = meta['name']
            label_raw = meta['raw']
            label_abbrev = meta['abbrev']
            gauge_dir = meta['gaugespace_dir']
            r_factor = meta['r']

            present_elgen_indices = _np.where(_np.abs(fogi_dir) > tol)[0]

            #Aggregate elemgen_info data for all elemgens that contribute to this FOGI qty (as determined by `tol`)
            ops_involved = set(); qubits_acted_upon = set(); types = set()  # basismx = None
            for k in present_elgen_indices:
                k_info = elemgen_info[k]
                ops_involved.add(k_info['op_label'])
                qubits_acted_upon.update(k_info['qubits'])
                types.add(k_info['type'])

            #Create the "info" dictionary for this FOGI quantity
            info = {'op_set': ops_involved,
                    'types': types,
                    'qubits': qubits_acted_upon,
                    'fogi_index': i,
                    'label': label,
                    'label_raw': label_raw,
                    'label_abbrev': label_abbrev,
                    'dependent': bool(i in dependent_indices),
                    'gauge_dir': gauge_dir,
                    'fogi_dir': fogi_dir,
                    'r_factor': r_factor
                    }
            ops_involved = tuple(sorted(ops_involved))
            types = tuple(sorted(types))
            qubits_acted_upon = tuple(sorted(qubits_acted_upon))
            if ops_involved not in bins: bins[ops_involved] = {}
            if types not in bins[ops_involved]: bins[ops_involved][types] = {}
            if qubits_acted_upon not in bins[ops_involved][types]: bins[ops_involved][types][qubits_acted_upon] = []
            bins[ops_involved][types][qubits_acted_upon].append(info)

        return bins

    def create_elementary_errorgen_space(self, op_elem_errgen_labels):
        """
        Construct a matrix whose column space spans the given list of elementary error generators.

        Parameters
        ----------
        op_elem_errgen_labels : iterable
            A list of `(operation_label, elementary_error_generator_label)` tuples, where
            `operation_label` is one of the primitive operation labels in `self.primitive_op_labels`
            and `elementary_error_generator_label` is a :class:`GlobalElementaryErrorgenLabel`
            object.

        Returns
        -------
        numpy.ndarray
            A two-dimensional array of shape `(self.errorgen_space_dim, len(op_elem_errgen_labels))`.
            Columns correspond to elements of `op_elem_errgen_labels` and the rowspace is the
            full elementary error generator space of this FOGI analysis.
        """
        lbl_to_index = {}
        for op_label in self.primitive_op_labels:
            elem_errgen_lbls = self.elem_errorgen_labels_by_op[op_label]
            elem_errgen_indices = _slct.indices(self.op_errorgen_indices[op_label])
            assert(len(elem_errgen_indices) == len(elem_errgen_lbls))
            lbl_to_index.update({(op_label, lbl): index for lbl, index in zip(elem_errgen_lbls, elem_errgen_indices)})

        ret = _np.zeros((self.fogi_directions.shape[0], len(op_elem_errgen_labels)))
        for i, lbl in enumerate(op_elem_errgen_labels):
            ret[lbl_to_index[lbl], i] = 1.0
        return ret

    #UNUSED in our final analyses so far - deprecate in favor of create_fogi_aggregate_single_op_space?
    #TODO: maybe can combine with function below, expanding it to take an op_set?
    def create_fogi_aggregate_space(self, op_set='all', errorgen_types='all', target='all'):
        """
        Construct a matrix with columns equal to the FOGI directions within the specified categories.

        Projecting a model's error-generator vector onto such a space can be used
        to obtain the contribution of a desired subset of the model's errors.

        Parameters
        ----------
        op_set : tuple or "all"
            Restrict to (intrinsic) FOGI quantities of a single operation, given as a 1-tuple,
            e.g. `(('Gxpi2',0),)` or relational error between multiple operations, e.g.,
            `(('Gxpi2',0), ('Gypi2',0))`.  The special `"all"` value includes quantities on
            all operations (no restriction).

        errorgen_types : tuple of {"H", "S"} or "all"
            Restrict to FOGI quantities containing elementary error generators of the given
            type(s).  Note that a tuple with multiple types only selects FOGI quantities
            containing *all* the give types (e.g., `('H','S')` will not match quantities
            composed of solely Hamiltonian-type generators).  The special `"all"` value includes
            quantities of all types (no restriction).

        target : tuple or "all"
            A tuple of state space (qubit) labels to restrict to, e.g., `('Q0','Q1')`.
            Note that includeing multiple labels selects only those quantities that
            target *all* the labels. The special `"all"` value includes quantities
            on all targets (no restriction).

        Returns
        -------
        numpy.ndarray
            A two-dimensional array with `self.errorgen_space_dim` rows and a number of
            columns dependent on the number of FOGI quantities matching the argument
            criteria.
        """
        binned_infos = self.create_binned_fogi_infos()

        selected_infos = []
        for ops, infos_by_type in binned_infos.items():
            if op_set == 'all' or ops == op_set:
                for type_tup, infos_by_target in infos_by_type.items():
                    if errorgen_types == 'all' or type_tup == errorgen_types:
                        for tgt, info_lst in infos_by_target.items():
                            if target == 'all' or tgt == target:
                                selected_infos.extend(info_lst)

        fogi_indices = [info['fogi_index'] for info in selected_infos]
        space = _np.take(self.fogi_directions, fogi_indices, axis=1)
        return space

    def create_fogi_aggregate_single_op_space(self, op_label, errorgen_type='H',
                                              intrinsic_or_relational='intrinsic', target='all'):
        """
        Construct a matrix with columns spanning a particular FOGI subspace for a single operation.

        This is a subspace of the full error-generator space of this FOGI analysis,
        and projecting a model's error-generator vector onto this space can be used
        to obtain the contribution of a desired subset of the `op_label`'s errors.

        Parameters
        ----------
        op_label : Label or str
            The operation to construct a subspace for.  This should be an element of
            `self.primitive_op_labels`.

        errorgen_type : {"H", "S", "all"}
            Potentially restrict to the subspace containing just Hamiltonian (H) or Pauli
            stochastic (S) errors.  `"all"` imposes no restriction.

        intrinsic_or_relational : {"intrinsic", "relational", "all"}
            Restrict to intrinsic or relational errors (or not, using `"all"`).

        target : tuple or "all"
            A tuple of state space (qubit) labels to restrict to, e.g., `('Q0','Q1')`.
            Note that including multiple labels selects only those quantities that
            target *all* the labels. The special `"all"` value includes quantities
            on all targets (no restriction).

        Returns
        -------
        numpy.ndarray
            A two-dimensional array with `self.errorgen_space_dim` rows and a number of
            columns dependent on the dimension of the selected subspace.
        """
        binned_infos = self.create_binned_fogi_infos()

        elem_errgen_lbls = self.elem_errorgen_labels_by_op[op_label]
        elem_errgen_indices = _slct.indices(self.op_errorgen_indices[op_label])
        assert(len(elem_errgen_indices) == len(elem_errgen_lbls))

        op_elem_space = _np.zeros((self.fogi_directions.shape[0], len(elem_errgen_indices)))
        for i, index in enumerate(elem_errgen_indices):
            op_elem_space[index, i] = 1.0

        if target == 'all' and errorgen_type == 'all':
            on_target_elem_errgen_indices = elem_errgen_indices
        else:
            on_target_elem_errgen_indices = []
            for index, lbl in zip(elem_errgen_indices, elem_errgen_lbls):
                if errorgen_type == 'all' or errorgen_type == lbl.errorgen_type:
                    support = lbl.support
                    if (target == 'all') or (target == support):
                        on_target_elem_errgen_indices.append(index)

        support_elem_space = _np.zeros((self.fogi_directions.shape[0], len(on_target_elem_errgen_indices)))
        for i, index in enumerate(on_target_elem_errgen_indices):
            support_elem_space[index, i] = 1.0
        #P_support_elem_space = support_elem_space @ np.linalg.pinv(support_elem_space)

        if intrinsic_or_relational in ('intrinsic', 'relational'):
            # easy case - can just use FOGIs to identify intrinsic errors
            selected_infos = []
            for ops, infos_by_type in binned_infos.items():
                if ops == (op_label,):
                    for types, infos_by_target in infos_by_type.items():  # use all types
                        #if types == (egtype,):
                        for _, info_lst in infos_by_target.items():  # use all targets here
                            selected_infos.extend(info_lst)
            fogi_indices = [info['fogi_index'] for info in selected_infos]
            full_int_space = _np.take(self.fogi_directions.toarray(), fogi_indices, axis=1)

            #space = P_op_elem_space @ full_int_space

            if intrinsic_or_relational == 'intrinsic':
                # full intrinsic space is a subspace of op_elem_space but perhaps not of support_elem_space
                #target_int_space = P_op_elem_space @ full_int_space
                support_int_space = _mt.intersection_space(support_elem_space, full_int_space, use_nice_nullspace=True)
                space = support_int_space
            elif intrinsic_or_relational == 'relational':
                local_support_space = op_elem_space.T @ support_elem_space
                local_int_space = op_elem_space.T @ full_int_space
                local_rel_space = _mt.nice_nullspace(local_int_space.T)
                support_rel_space = _mt.intersection_space(local_support_space, local_rel_space,
                                                           use_nice_nullspace=True)
                space = op_elem_space @ support_rel_space

        elif intrinsic_or_relational == 'all':
            space = support_elem_space
        else:
            raise ValueError("Invalid intrinsic_or_relational value: `%s`" % str(intrinsic_or_relational))

        col_indices = _mt.independent_columns(space)
        space = space[:, col_indices]
        return space

    @classmethod
    def merge_binned_fogi_infos(cls, binned_fogi_infos, index_offsets):
        """
        Merge together multiple FOGI-info dictionaries created by :meth:`create_binned_fogi_infos`.

        Parameters
        ----------
        binned_fogi_infos : list
            A list of FOGI-info dictionaries.

        index_offsets : list
            A list of length `len(binned_fogi_infos)` that gives the offset
            into an assumed-to-exist corresponding vector of components for
            all the FOGI infos.

        Returns
        -------
        dict
            The merged dictionary
        """
        def _merge_into(dest, src, offset, nlevels_to_merge, store_index):
            if nlevels_to_merge == 0:  # special last-level case where src and dest are *lists*
                for info in src:
                    new_info = _copy.deepcopy(info)
                    new_info['fogi_index'] += offset
                    new_info['store_index'] = store_index
                    dest.append(new_info)
            else:
                for k, d in src.items():
                    if k not in dest: dest[k] = {} if (nlevels_to_merge > 1) else []  # last level = list
                    _merge_into(dest[k], d, offset, nlevels_to_merge - 1, store_index)

        bins = {}
        nLevels = 3  # ops_involved, types, qubits_acted_upon
        for i, (sub_bins, offset) in enumerate(zip(binned_fogi_infos, index_offsets)):
            _merge_into(bins, sub_bins, offset, nLevels, i)
        return bins
