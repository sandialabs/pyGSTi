from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
""" Defines the GateSet class and supporting functionality."""

import numpy as _np
import numpy.linalg as _nla
import scipy as _scipy
import itertools as _itertools
import collections as _collections
import warnings as _warnings
import time as _time

from ..tools import matrixtools as _mt
from ..tools import basistools as _bt
from ..tools import likelihoodfns as _lf
from ..tools import jamiolkowski as _jt

from . import evaltree as _evaltree
from . import gate as _gate
from . import spamvec as _sv
from . import labeldicts as _ld
from . import gscalc as _gscalc
from . import gaugegroup as _gg

from .verbosityprinter import VerbosityPrinter

# Tolerace for matrix_rank when finding rank of a *normalized* projection
# matrix.  This is a legitimate tolerace since a normalized projection matrix
# should have just 0 and 1 eigenvalues, and thus a tolerace << 1.0 will work
# well.
P_RANK_TOL = 1e-7


class GateSet(object):
    """
    Encapsulates a set of gate, state preparation, and POVM effect operations.

    A GateSet stores a set of labeled Gate objects and provides dictionary-like
    access to their matrices.  State preparation and POVM effect operations are
    represented as column vectors.
    """

    #Whether access to gates & spam vecs via GateSet indexing is allowed
    _strict = False

    def __init__(self, default_param="full",
                 prep_prefix="rho", effect_prefix="E", gate_prefix="G",
                 remainder_label="remainder", identity_label="identity"):
        """
        Initialize a gate set.

        Parameters
        ----------
        default_param : {"full", "TP", "static"}, optional
            Specifies the default gate and SPAM vector parameterization type.
            "full" : by default gates and vectors are fully parameterized.
            "TP" : by default the first row of gates and the first element of
            vectors is not parameterized and fixed so gate set is trace-
            preserving.
            "static" : by default gates and vectors are not parameterized.

        prep_prefix, effect_prefix, gate_prefix : string, optional
            Key prefixes designating state preparations, POVM effects,
            and gates, respectively.  These prefixes allow the GateSet to
            determine what type of object a each key corresponds to.

        remainder_label : string, optional
            The special string used within SPAM label values to indicate
            special behavior, either the use of a "complement" POVM vector or
            a SPAM label which generates probabilities such that the sum of all
            SPAM label probabilities equals one.

        identity_label : string, optional
            The special string which acts as the key to get and set the
            identity vector.

        """

        assert(default_param in ('full','TP','static'))
        default_e_param = "full" if default_param == "TP" else default_param

        #Gate dimension of this GateSet (None => unset, to be determined)
        self._dim = None

        #Name and dimension (or list of dims) of the *basis*
        # that the gates and SPAM vectors are expressed in.  This
        # is for interpretational purposes only, and is reset often
        # (for instance, when reading GateSet params from a vector)
        self._basisNameAndDim = ("unknown",None)

        #SPAM vectors
        self.preps = _ld.OrderedSPAMVecDict(self,default_param,
                                              None, prep_prefix)
        self.effects = _ld.OrderedSPAMVecDict(self, default_e_param,
                                            remainder_label, effect_prefix)
        self._povm_identity = None #the identity vector in whatever basis is
                                #being used (needed only if "-1" EVec is used)

        #SPAM labels: key = label, value = (prepLabel, effectLabel)
        self.spamdefs = _ld.OrderedSPAMLabelDict(remainder_label)

        #Gates
        self.gates = _ld.OrderedGateDict(self, default_param, gate_prefix)

        self._remainderlabel = remainder_label
        self._identitylabel = identity_label
        self._default_gauge_group = None

        super(GateSet, self).__init__()

    @property
    def povm_identity(self):
        return self._povm_identity

    @povm_identity.setter
    def povm_identity(self, value):
        if value is None:
            self._povm_identity = None
            return

        if self._dim is None:     self._dim = len(value)
        if self._dim != len(value):
            raise ValueError("Cannot add vector with dimension" +
                             "%d to gateset of dimension %d"
                             % (len(value),self._dim))
        if self.povm_identity is not None:
            self._povm_identity.set_vector(value)
        else:
            self._povm_identity = _sv.FullyParameterizedSPAMVec(value)
              # fully parameterized, even though not vectorized (so
              # can gauge transform it)

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


    def get_basis_name(self):
        """
        Returns the name abbreviation of the basis, essentially identifying
        its type.  The gate matrices and SPAM vectors within the GateSet
        are to be interpreted within this basis.  Note that the dimension of
        the (matrix) elements of the basis can be obtained by
        get_basis_dimension(...).  The basis abbreviations use by pyGSTi are
        "gm" (a Gell-Mann basis), "pp" (a Pauli-product basis), and "std"
        (a matrix unit, or "standard" basis).  If the basis is unknown, the
        string "unknown" is returned.

        Returns
        -------
        str
            basis abbreviation (or "unknown")
        """
        return self._basisNameAndDim[0]


    def get_basis_dimension(self):
        """
        Get the dimension of the basis matrices, or more generally,
        the structure of the density matrix space as a list of integer
        dimensions.  In the latter case, the dimension of each basis
        element (a matrix) is d x d, where d equals the sum of the
        returned list of integers. (In the former case, d equals the
        single returned integers.)  This density-matrix-space
        structure can be used, along with the basis name (cf.
        get_basis_name), to construct that basis of matrices used
        to express the gate matrices and SPAM vectors of this
        GateSet.

        Returns
        -------
        int or list
            density-matrix dimension or a list of integers
            specifying the dimension of each term in a
            direct sum decomposition of the density matrix
            space.
        """
        return self._basisNameAndDim[1]


    def set_basis(self, basisName, basisDimension):
        """
        Sets the basis name and dimension.  See
        get_basis_name and gate_basis_dimension for
        details on these quantities.

        Parameters
        ----------
        basisName : str
           The name abbreviation for the basis. Typically in {"pp","gm","std"}

        basisDimension : int or list
           The dimension of the density matrix space this basis spans, or a
           list specifying the dimensions of terms in a direct-sum
           decomposition of the density matrix space.
        """
        self._basisNameAndDim = (basisName, basisDimension)


    def reset_basis(self):
        """
        "Forgets" the basis name and dimension by setting
        these quantities to "unkown" and None, respectively.
        """
        self._basisNameAndDim = ("unknown", None)


    def get_prep_labels(self):
        """
        Get the labels of state preparation vectors.

        Returns
        -------
        list of strings
        """
        return list(self.preps.keys())


    def get_effect_labels(self):
        """
        Get all the effect vector labels present in a SPAM label.  This
        may include the special "remainder" label signifying the "complement"
        effect vector, equal to Identity - sum(other effect vectors).

        Returns
        -------
        list of strings
        """
        labels = list(self.effects.keys())
        if any( [effectLabel == self._remainderlabel and
                 prepLabel != self._remainderlabel
                 for prepLabel,effectLabel in list(self.spamdefs.values())] ):
            labels.append( self._remainderlabel )
        return labels


    def get_preps(self):
        """
        Get an list of all the state prepartion vectors.  These
        vectors are copies of internally stored data and thus
        can be modified without altering the gateset.

        Returns
        -------
        list of numpy arrays
            list of state preparation vectors of shape (dim, 1).
        """
        return [ self.preps[l].copy() for l in self.get_prep_labels() ]

    def get_effects(self):
        """
        Get an list of all the POVM effect vectors.  This list will include
        the "compliment" effect vector at the end of the list if one has been
        specified.  Also, the returned vectors are copies of internally stored
        data and thus can be modified without altering the gateset.

        Returns
        -------
        list of numpy arrays
            list of POVM effect vectors of shape (dim, 1).
        """
        return [ self.effects[l].copy() for l in self.get_effect_labels() ]


    def num_preps(self):
        """
        Get the number of state preparation vectors

        Returns
        -------
        int
        """
        return len(self.preps)

    def num_effects(self):
        """
        Get the number of effect vectors, including a "complement" effect
        vector, equal to Identity - sum(other effect vectors)

        Returns
        -------
        int
        """
        bHaveComplementEvec = \
            any( [effectLabel == self._remainderlabel and
                  prepLabel != self._remainderlabel
                  for prepLabel,effectLabel in list(self.spamdefs.values())] )
        return len(self.effects) + ( 1 if bHaveComplementEvec else 0 )


    #def add_spam_definition(self, prepLabel, effectLabel, spamLabel):
    #    """
    #    Adds a new spam label.  That is, associates the SPAM
    #      pair (prepLabel,effectLabel) with the given spamLabel.  Same
    #      as:  gateset.spamdefs[spamLabel] = (prepLabel, effectLabel)
    #
    #    Parameters
    #    ----------
    #    prepLabel : string
    #        state preparation label.
    #
    #    effectLabel : string
    #        POVM effect label.
    #
    #    spamLabel : string
    #        the "spam label" to associate with (prepLabel,effectLabel).
    #    """
    #    self.spamdefs[spamLabel] = (prepLabel,effectLabel)


    def get_reverse_spam_defs(self):
        """
        Get a reverse-lookup dictionary for spam labels.

        Returns
        -------
        OrderedDict
            a dictionary with keys == (prepLabel,effectLabel) tuples and
            values == SPAM labels.
        """
        d = _collections.OrderedDict()
        for label in self.spamdefs:
            d[  self.spamdefs[label] ] = label
        return d

    def get_spam_labels(self):
        """
        Get a list of all the spam labels.

        Returns
        -------
        list of strings
        """
        return list(self.spamdefs.keys())


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
        return self._calc()._make_spamgate(spamLabel)



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
            raise KeyError("Strict-mode: invalid key %s" % label)

        if label.startswith(self.preps._prefix):
            self.preps[label] = value
        elif label.startswith(self.effects._prefix) \
                or label == self._remainderlabel:
            self.effects[label] = value
        elif label.startswith(self.gates._prefix):
            self.gates[label] = value
        elif label == self._identitylabel:
            self.povm_identity = value
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

        if label.startswith(self.preps._prefix):
            return self.preps[label]
        elif label.startswith(self.effects._prefix) \
                or label == self._remainderlabel:
            return self.effects[label]
        elif label.startswith(self.gates._prefix):
            return self.gates[label]
        elif label == self._identitylabel:
            return self.povm_identity
        else:
            raise KeyError("Key %s has an invalid prefix" % label)

    def set_all_parameterizations(self, parameterization_type):
        """
        Convert all gates and SPAM vectors to a specific parameterization
        type.

        Parameters
        ----------
        parameterization_type : {"full", "TP", "static"}
            The gate and SPAM vector parameterization type:

        """
        typ = parameterization_type
        assert(parameterization_type in ('full','TP','static'))
        etyp = "full" if typ == "TP" else typ #EVecs never "TP"

        for lbl,gate in self.gates.items():
            self.gates[lbl] = _gate.convert(gate, typ)

        for lbl,vec in self.preps.items():
            self.preps[lbl] = _sv.convert(vec, typ)

        for lbl,vec in self.effects.items():
            self.effects[lbl] = _sv.convert(vec, etyp)

        if typ == 'full': 
            self.default_gauge_group = _gg.FullGaugeGroup(self.dim)
        elif typ == 'TP': 
            self.default_gauge_group = _gg.TPGaugeGroup(self.dim)
        elif typ == 'static': 
            self.default_gauge_group = None
        
        #Note: self.povm_identity should *always* be fully
        # paramterized, and is not changed by this method.



    #def __getstate__(self):
    #    #Returns self.__dict__ by default, which is fine

    def __setstate__(self, stateDict):
        self.__dict__.update(stateDict)
        #Additionally, must re-connect this gateset as the parent
        # of relevant OrderedDict-derived classes, which *don't*
        # preserve this information upon pickling so as to avoid
        # circular pickling...
        self.preps.parent = self
        self.effects.parent = self
        self.gates.parent = self


    def num_params(self):
        """
        Return the number of free parameters when vectorizing
        this gateset.

        Returns
        -------
        int
            the number of gateset parameters.
        """
        L = sum([ rhoVec.num_params() for rhoVec in list(self.preps.values()) ])
        L += sum([ EVec.num_params() for EVec in list(self.effects.values()) ])
        L += sum([ gate.num_params() for gate in list(self.gates.values())])
        return L


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
        rhoSize = [ rho.size for rho in list(self.preps.values()) ]
        eSize   = [ E.size for E in list(self.effects.values()) ]
        gateSize = [ gate.size for gate in list(self.gates.values()) ]
        return sum(rhoSize) + sum(eSize) + sum(gateSize)


    def num_nongauge_params(self):
        """
        Return the number of non-gauge parameters when vectorizing
        this gateset according to the optional parameters.

        Returns
        -------
        int
            the number of non-gauge gateset parameters.
        """
        P = self.get_nongauge_projector()
        return _np.linalg.matrix_rank(P, P_RANK_TOL)


    def num_gauge_params(self):
        """
        Return the number of gauge parameters when vectorizing
        this gateset according to the optional parameters.

        Returns
        -------
        int
            the number of gauge gateset parameters.
        """
        return self.num_params() - self.num_nongauge_params()


    def to_vector(self):
        """
        Returns the gateset vectorized according to the optional parameters.

        Returns
        -------
        numpy array
            The vectorized gateset parameters.
        """
        v = _np.empty( self.num_params() )

        objs_to_vectorize = \
            list(self.preps.values()) + list(self.effects.values()) + list(self.gates.values())

        off = 0
        for obj in objs_to_vectorize:
            np = obj.num_params()
            v[off:off+np] = obj.to_vector(); off += np

        return v


    def from_vector(self, v):
        """
        The inverse of to_vector.  Loads values of gates and/or rho and E vecs from
        from a vector v according to the optional parameters. Note that neither v
        nor the optional parameters specify what number of gates and their labels,
        and that this information must be contained in the gateset prior to calling
        from_vector.  In practice, this just means you should call the from_vector method
        of the gateset that was used to generate the vector v in the first place.
        """
        assert( len(v) == self.num_params() )

        objs_to_vectorize = \
            list(self.preps.values()) + list(self.effects.values()) + list(self.gates.values())

        off = 0
        for obj in objs_to_vectorize:
            np = obj.num_params()
            obj.from_vector( v[off:off+np] ); off += np

        self.reset_basis()
          # assume the vector we're loading isn't producing gates & vectors in
          # a known basis.


    def get_vector_offsets(self):
        """
        Returns the offsets of individual components in the vectorized
        gateset according to the optional parameters.

        Returns
        -------
        dict
            A dictionary whose keys are either SPAM vector or a gate label
            and whose values are (start,next_start) tuples of integers
            indicating the start and end+1 indices of the component.
        """
        off = 0
        offsets = {}
        for label,obj in _itertools.chain(iter(self.preps.items()),
                                          iter(self.effects.items()),
                                          iter(self.gates.items()) ):
            np = obj.num_params()
            offsets[label] = (off,off+np); off += np

        return offsets


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
        deriv = _np.zeros( (self.num_elements(), self.num_params()), 'd' )

        objs_to_vectorize = \
            list(self.preps.values()) + list(self.effects.values()) + list(self.gates.values())

        eo = po = 0 # element & parameter offsets
        for obj in objs_to_vectorize:
            ne, np = obj.size, obj.num_params() #number of els & params
            #print "DB: setting [%d:%d, %d:%d] = \n%s" % (eo,eo+ne,po,po+np,obj.deriv_wrt_params())
            deriv[eo:eo+ne,po:po+np] = obj.deriv_wrt_params()
            eo += ne; po += np

        return deriv


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

        # We want to divide the GateSet-space H (a Hilbert space, 56-dimensional in the 1Q, 3-gate, 2-vec case)
        # into the direct sum of gauge and non-gauge spaces, and find projectors onto each
        # sub-space (the non-gauge space in particular).
        #
        # Within the GateSet H-space lies a gauge-manifold of maximum chi2 (16-dimensional in 1Q case),
        #  where the gauge-invariant chi2 function is constant.  At a single point (GateSet) P on this manifold,
        #  chosen by (somewhat arbitrarily) fixing the gauge, we can talk about the tangent space
        #  at that point.  This tangent space is spanned by some basis (~16 elements for the 1Q case),
        #  which associate with the infinitesimal gauge transform ?generators? on the full space.
        #  The subspace of H spanned by the derivatives w.r.t gauge transformations at P (a GateSet) spans
        #  the gauge space, and the complement of this (in H) is the non-gauge space.
        #
        #  An element of the gauge group can be written gg = exp(-K), where K is a n x n matrix.  If K is
        #   hermitian then gg is unitary, but this need not be the case.  A gauge transform acts on a
        #   gatset via Gateset => gg^-1 G gg, i.e. G => exp(K) G exp(-K).  We care about the action of
        #   infinitesimal gauge tranformations (b/c the *derivative* vectors span the tangent space),
        #   which act as:
        #    G => (I+K) G (I-K) = G + [K,G] + ignored(K^2), where [K,G] := KG-GK
        #
        # To produce a projector onto the gauge-space, we compute the *column* vectors
        #  dG_ij = [K_ij,G], where K_ij is the i,j-th matrix unit (56x1 in the 1Q case, 16 such column vectors)
        #  and then form a projector in the standard way.
        #  (to project onto |v1>, |v2>, etc., form P = sum_i |v_i><v_i|)
        #
        #Typically nGateParams < len(dG_ij) and linear system is overconstrained
        #   and no solution is expected.  If no solution exists, simply ignore
        #
        # So we form P = sum_ij dG_ij * transpose(dG_ij) (56x56 in 1Q case)
        #              = dG * transpose(dG)              where dG_ij form the *columns* of dG (56x16 in 1Q case)
        # But dG_ij are not orthonormal, so really we need a slight modification,
        #  otherwise P'P' != P' as must be the case for a projector:
        #
        # P' = dG * (transpose(dG) * dG)^-1 * transpose(dG) (see wikipedia on projectors)
        #
        #    or equivalently (I think)
        #
        # P' = pseudo-inv(P)*P
        #
        #  since the pseudo-inv is defined by P*pseudo-inv(P) = I and so P'P' == P'
        #  and P' is our gauge-projector!

        # Note: In the general case of parameterized gates (i.e. non-fully parameterized gates), there are fewer
        #   gate parameters than the size of dG_ij ( < 56 in 1Q case).  In this general case, we want to know
        #   what (if any) change in gate parameters produces the change dG_ij of the gate matrix elements.
        #   That is, we solve dG_ij = derivWRTParams * dParams_ij for dParams_ij, where derivWRTParams is
        #   the derivative of the gate matrix elements with respect to the gate parameters (derivWRTParams
        #   is 56x(nGateParams) and dParams_ij is (nGateParams)x1 in the 1Q case) -- then substitute dG_ij
        #   with dParams_ij above to arrive at a (nGateParams)x(nGateParams) projector (compatible with
        #   hessian computed by gateset).
        #
        #   Actually, we want to know if what changes gate parameters
        #   produce changes in the span of all the dG_ij, so really we want the intersection of the space
        #   defined by the columns of derivWRTParams (the "gate-parameter range" space) and the dG_ij vectors.
        #
        #   This intersection is determined by nullspace( derivWRTParams | dG ), where the pipe denotes
        #     concatenating the matrices together.  Then, if x is a basis vector of the nullspace
        #     x[0:nGateParams] is the basis vector for the intersection space within the gate parameter space,
        #     that is, the analogue of dParams_ij in the single-dG_ij introduction above.
        #
        #   Still, we just substitue these dParams_ij vectors (as many as the nullspace dimension) for dG_ij
        #   above to get the general case projector.


        #Use a GateSet object to hold & then vectorize the derivatives wrt each gauge transform basis element (each ij)
        dim = self._dim
        nParams = self.num_params()
        nElements = self.num_elements()

        #This was considered as optional behavior, but better to just delete qtys from GateSet
        ##whether elements of the raw gateset matrices/SPAM vectors that are not
        ## parameterized at all should be ignored.   By ignoring changes to such
        ## elements, they are treated as not being a part of the "gateset"
        #bIgnoreUnparameterizedEls = True

        #Note: gateset object (gsDeriv) must have all elements of gate
        # mxs and spam vectors as parameters (i.e. be "fully parameterized") in
        # order to match deriv_wrt_params call, which gives derivatives wrt
        # *all* elements of a gate set.
        gsDeriv = GateSet("full", self.preps._prefix, self.effects._prefix,
                          self.gates._prefix, self._remainderlabel,
                          self._identitylabel)
        for gateLabel in self.gates:
            gsDeriv.gates[gateLabel] = _np.zeros((dim,dim),'d')
        for prepLabel in self.preps:
            gsDeriv.preps[prepLabel] = _np.zeros((dim,1),'d')
        for effectLabel in self.effects:
            gsDeriv.effects[effectLabel] = _np.zeros((dim,1),'d')

        assert(gsDeriv.num_elements() == gsDeriv.num_params() == nElements)

        dG = _np.empty( (nElements, dim**2), 'd' )
        for i in range(dim):      # always range over all rows: this is the
            for j in range(dim):  # *generator* mx, not gauge mx itself
                unitMx = _bt._mut(i,j,dim)
                for lbl,rhoVec in self.preps.items():
                    gsDeriv.preps[lbl] = _np.dot(unitMx, rhoVec)
                for lbl,EVec in self.effects.items():
                    gsDeriv.effects[lbl] =  -_np.dot(EVec.T, unitMx).T
                for lbl,gate in self.gates.items():
                    gsDeriv.gates[lbl] = _np.dot(unitMx,gate) - \
                                         _np.dot(gate,unitMx)

                #Note: vectorize all the parameters in this full-
                # parameterization object, which gives a vector of length
                # equal to the number of gateset *elements*.
                dG[:,i*dim+j] = gsDeriv.to_vector()

        dP = self.deriv_wrt_params()

        #if bIgnoreUnparameterizedEls:
        #    for i in range(dP.shape[0]):
        #        if _np.isclose( _np.linalg.norm(dP[i,:]), 0):
        #            dG[i,:] = 0 #if i-th element not parameterized,
        #                        # clear dG row corresponding to it.

        M = _np.concatenate( (dP,dG), axis=1 )
        nullsp = _mt.nullspace(M) #columns are nullspace basis vectors
        gen_dG = nullsp[0:nParams,:] #take upper (gate-param-segment) of vectors for basis
                                     # of subspace intersection in gate-parameter space
        #Note: gen_dG == "generalized dG", and is (nParams)x(nullSpaceDim==gaugeSpaceDim), so P
        #  (below) is (nParams)x(nParams) as desired.

        #DEBUG
        #for iRow in range(nElements):
        #    pNorm = _np.linalg.norm(dP[iRow])
        #    if pNorm < 1e-6:
        #        print "Row %d of dP is zero" % iRow
        #
        #print "------------------------------"
        #print "nParams = ",nParams
        #print "nElements = ",nElements
        #print " shape(M) = ",M.shape
        #print " shape(dG) = ",dG.shape
        #print " shape(dP) = ",dP.shape
        #print " shape(gen_dG) = ",gen_dG.shape
        #print " shape(nullsp) = ",nullsp.shape
        #print " rank(dG) = ",_np.linalg.matrix_rank(dG)
        #print " rank(dP) = ",_np.linalg.matrix_rank(dP)
        #print "------------------------------"
        #assert(_np.linalg.norm(_np.imag(gen_dG)) < 1e-9) #gen_dG is real

        if nonGaugeMixMx is not None:
            msg = "You've set both nonGaugeMixMx and itemWeights, both of which"\
                + " set the gauge metric... You probably don't want to do this."
            assert(itemWeights is None), msg

            # BEGIN GAUGE MIX ----------------------------------------
            # nullspace of gen_dG^T (mx with gauge direction vecs as rows) gives non-gauge directions
            nonGaugeDirections = _mt.nullspace(gen_dG.T) #columns are non-gauge directions

            #for each column of gen_dG, which is a gauge direction in gateset parameter space,
            # we add some amount of non-gauge direction, given by coefficients of the
            # numNonGaugeParams non-gauge directions.
            gen_dG = gen_dG + _np.dot( nonGaugeDirections, nonGaugeMixMx) #add non-gauge direction components in
             # dims: (nParams,nGaugeParams) + (nParams,nNonGaugeParams) * (nNonGaugeParams,nGaugeParams)
             # nonGaugeMixMx is a (nNonGaugeParams,nGaugeParams) matrix whose i-th column specifies the
             #  coefficents to multipy each of the non-gauge directions by before adding them to the i-th
             #  direction to project out (i.e. what were the pure gauge directions).

            #DEBUG
            #print "gen_dG shape = ",gen_dG.shape
            #print "NGD shape = ",nonGaugeDirections.shape
            #print "NGD rank = ",_np.linalg.matrix_rank(nonGaugeDirections, P_RANK_TOL)
            #END GAUGE MIX ----------------------------------------

        # Build final non-gauge projector by getting a mx of column vectors 
        # orthogonal to the cols fo gen_dG:
        #     gen_dG^T * gen_ndG = 0 => nullspace(gen_dG^T)
        # or for a general metric W:
        #     gen_dG^T * W * gen_ndG = 0 => nullspace(gen_dG^T * W)
        # (This is instead of construction as I - gaugeSpaceProjector)

        if itemWeights is not None:
            metric_diag = _np.ones(self.num_params(), 'd')
            offsets = self.get_vector_offsets()
            gateWeight = itemWeights.get('gates', 1.0)
            spamWeight = itemWeights.get('spam', 1.0)
            for lbl in self.gates:
                i,j = offsets[lbl]
                metric_diag[i:j] = itemWeights.get(lbl, gateWeight)
            for lbl in _itertools.chain(iter(self.preps),iter(self.effects)):
                i,j = offsets[lbl]
                metric_diag[i:j] = itemWeights.get(lbl, spamWeight)
            metric = _np.diag(metric_diag)
            gen_ndG = _mt.nullspace(_np.dot(gen_dG.T,metric))
        else:
            gen_ndG = _mt.nullspace(gen_dG.T) #cols are non-gauge directions
                

        # ORIG WAY: use psuedo-inverse to normalize projector.  Ran into problems where
        #  default rcond == 1e-15 didn't work for 2-qubit case, but still more stable than inv method below
        P = _np.dot(gen_ndG, _np.transpose(gen_ndG)) # almost a projector, but cols of dG are not orthonormal
        Pp = _np.dot( _np.linalg.pinv(P, rcond=1e-7), P ) # make P into a true projector (onto gauge space)

        # ALT WAY: use inverse of dG^T*dG to normalize projector (see wikipedia on projectors, dG => A)
        #  This *should* give the same thing as above, but numerical differences indicate the pinv method
        #  is prefereable (so long as rcond=1e-7 is ok in general...)
        #  Check: P'*P' = (dG (dGT dG)^1 dGT)(dG (dGT dG)^-1 dGT) = (dG (dGT dG)^1 dGT) = P'
        #invGG = _np.linalg.inv(_np.dot(_np.transpose(gen_ndG), gen_ndG))
        #Pp_alt = _np.dot(gen_ndG, _np.dot(invGG, _np.transpose(gen_ndG))) # a true projector (onto gauge space)
        #print "Pp - Pp_alt norm diff = ", _np.linalg.norm(Pp_alt - Pp)

        #OLD: ret = _np.identity(nParams,'d') - Pp 
        # Check ranks to make sure everything is consistent.  If either of these assertions fail,
        #  then pinv's rcond or some other numerical tolerances probably need adjustment.
        #print "Rank P = ",_np.linalg.matrix_rank(P)
        #print "Rank Pp = ",_np.linalg.matrix_rank(Pp, P_RANK_TOL)
        #print "Rank 1-Pp = ",_np.linalg.matrix_rank(_np.identity(nParams,'d') - Pp, P_RANK_TOL)
        #print " Evals(1-Pp) = \n","\n".join([ "%d: %g" % (i,ev) \
        #       for i,ev in enumerate(_np.sort(_np.linalg.eigvals(_np.identity(nParams,'d') - Pp))) ])
        
        try:
            rank_P = _np.linalg.matrix_rank(P, P_RANK_TOL) # original un-normalized projector
            
            # Note: use P_RANK_TOL here even though projector is *un-normalized* since sometimes P will
            #  have eigenvalues 1e-17 and one or two 1e-11 that should still be "zero" but aren't when
            #  no tolerance is given.  Perhaps a more custom tolerance based on the singular values of P
            #  but different from numpy's default tolerance would be appropriate here.

            assert( rank_P == _np.linalg.matrix_rank(Pp, P_RANK_TOL)) #rank shouldn't change with normalization
            #assert( (nParams - rank_P) == _np.linalg.matrix_rank(ret, P_RANK_TOL) ) # dimension of orthogonal space
        except(_np.linalg.linalg.LinAlgError):
            _warnings.warn("Linear algebra error (probably a non-convergent" +
                           "SVD) ignored during matric rank checks in " +
                           "GateSet.get_nongauge_projector(...) ")
            
        return Pp 
        #OLD: return ret


    def transform(self, S):
        """
        Update each of the gate matrices G in this gateset with inv(S) * G * S,
        each rhoVec with inv(S) * rhoVec, and each EVec with EVec * S

        Parameters
        ----------
        S : GaugeGroup.element
            A gauge group element which specifies the "S" matrix 
            (and it's inverse) used in the above similarity transform.
        """
        for rhoVec in list(self.preps.values()):
            rhoVec.transform(S,'prep')

        for EVec in list(self.effects.values()):
            EVec.transform(S,'effect')

        if self.povm_identity is not None: # same as Es
            self.povm_identity.transform(S,'effect')

        for gateObj in list(self.gates.values()):
            gateObj.transform(S)


    def _calc(self):
        return _gscalc.GateSetCalculator(self._dim, self.gates, self.preps,
                                         self.effects, self.povm_identity,
                                         self.spamdefs, self._remainderlabel,
                                         self._identitylabel)

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
        return self._calc().hproduct(gatestring, flat)


    def pr(self, spamLabel, gatestring, clipTo=None, bUseScaling=True):
        """
        Compute the probability of the given gate sequence, where initialization
        & measurement operations are together specified by spamLabel.

        Parameters
        ----------
        spamLabel : string
           the label specifying the state prep and measure operations

        gatestring : GateString or tuple of gate labels
          The sequence of gate labels specifying the gate string.

        clipTo : 2-tuple, optional
          (min,max) to clip return value if not None.

        bUseScaling : bool, optional
          Whether to use a post-scaled product internally.  If False, this
          routine will run slightly faster, but with a chance that the
          product will overflow and the subsequent trace operation will
          yield nan as the returned probability.

        Returns
        -------
        float
        """
        return self._calc().pr(spamLabel, gatestring, clipTo, bUseScaling)


    def dpr(self, spamLabel, gatestring,
            returnPr=False,clipTo=None):
        """
        Compute the derivative of a probability generated by a gate string and
        spam label as a 1 x M numpy array, where M is the number of gateset
        parameters.

        Parameters
        ----------
        spamLabel : string
           the label specifying the state prep and measure operations

        gatestring : GateString or tuple of gate labels
          The sequence of gate labels specifying the gate string.

        returnPr : bool, optional
          when set to True, additionally return the probability itself.

        clipTo : 2-tuple, optional
           (min,max) to clip returned probability to if not None.
           Only relevant when returnPr == True.

        Returns
        -------
        derivative : numpy array
            a 1 x M numpy array of derivatives of the probability w.r.t.
            each gateset parameter (M is the length of the vectorized gateset).

        probability : float
            only returned if returnPr == True.
        """
        return self._calc().dpr(spamLabel, gatestring,returnPr,clipTo)


    def hpr(self, spamLabel, gatestring,
            returnPr=False,returnDeriv=False,clipTo=None):
        """
        Compute the Hessian of a probability generated by a gate string and
        spam label as a 1 x M x M array, where M is the number of gateset
        parameters.

        Parameters
        ----------
        spamLabel : string
           the label specifying the state prep and measure operations

        gatestring : GateString or tuple of gate labels
          The sequence of gate labels specifying the gate string.

        returnPr : bool, optional
          when set to True, additionally return the probability itself.

        returnDeriv : bool, optional
          when set to True, additionally return the derivative of the
          probability.

        clipTo : 2-tuple, optional
           (min,max) to clip returned probability to if not None.
           Only relevant when returnPr == True.

        Returns
        -------
        hessian : numpy array
            a 1 x M x M array, where M is the number of gateset parameters.
            hessian[0,j,k] is the derivative of the probability w.r.t. the
            k-th then the j-th gateset parameter.

        derivative : numpy array
            only returned if returnDeriv == True. A 1 x M numpy array of
            derivatives of the probability w.r.t. each gateset parameter.

        probability : float
            only returned if returnPr == True.
        """
        return self._calc().hpr(spamLabel, gatestring,
                                returnPr,returnDeriv,clipTo)


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
        return self._calc().probs(gatestring, clipTo)


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
        return self._calc().dprobs(gatestring,returnPr,clipTo)


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
        return self._calc().hprobs(gatestring, returnPr, returnDeriv, clipTo)


    def bulk_evaltree_from_resources(self, gatestring_list, comm=None, memLimit=None,
                                     distributeMethod="gatestrings", subcalls=[],
                                     verbosity=0):
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

        verbosity : int, optional
            How much detail to send to stdout.

        Returns
        -------
        evt : EvalTree
            The evaluation tree object, split as necesary.
        paramBlockSize : int or None
            The maximum size of parameter blocks (i.e. the maximum
            number of parameters to compute at once in calls to 
            dprobs, etc., usually specified as wrtBlockSize).
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

        printer = VerbosityPrinter.build_printer(verbosity, comm)

        nspam = len(self.get_spam_labels())
        nprocs = 1 if comm is None else comm.Get_size()
        num_params = self.num_params()
        dim = self._dim
        evt_cache = {} # cache of eval trees based on # min subtrees, to avoid re-computation
        floatSize = 8 # in bytes: TODO: a better way
        C = 1.0/(1024.0**3)

        bNp2Matters = ("bulk_fill_hprobs" in subcalls) or ("bulk_hprobs_by_block" in subcalls)

        if memLimit is not None:
            if memLimit <= 0:
                raise MemoryError("Attempted evaltree generation " +
                                  "w/memlimit = %g <= 0!" % memLimit)
            printer.log("Evaltree generation (%s) w/mem limit = %.2fGB"
                        % (distributeMethod, memLimit*C))

        def prime_factors(n):  #TODO: move this fn somewhere else
            i = 2; factors = []
            while i * i <= n:
                if n % i:
                    i += 1
                else:
                    n //= i
                    factors.append(i)
            if n > 1:
                factors.append(n)
            return factors


        def memEstimate(ng,np1,np2,Ng,fastCacheSz=False,verb=0):
            tm = _time.time()

            #Get cache size
            if not fastCacheSz:
                #Slower (but more accurate way)
                if ng not in evt_cache: evt_cache[ng] = self.bulk_evaltree(
                    gatestring_list,minSubtrees=ng,verbosity=printer-1)
                tstTree = evt_cache[ng]
                cacheSize = max([len(s) for s in tstTree.get_sub_trees()])
            else:
                #heuristic (but fast)
                cacheSize = int( 1.3 * len(gatestring_list) / ng )

            wrtLen1 = (num_params+np1-1) // np1 # ceiling(num_params / np1)
            wrtLen2 = (num_params+np2-1) // np2 # ceiling(num_params / np2)
            nSubtreesPerProc = (ng+Ng-1) // Ng # ceiling(ng / Ng)

            mem = 0
            for fnName in subcalls:
                if fnName == "bulk_fill_probs":
                    mem += cacheSize * dim * dim # product cache
                    mem += cacheSize # scale cache (exps)
                    mem += cacheSize # scale vals

                elif fnName == "bulk_fill_dprobs":
                    mem += cacheSize * wrtLen1 * dim * dim # dproduct cache
                    mem += cacheSize * dim * dim # product cache
                    mem += cacheSize # scale cache
                    mem += cacheSize # scale vals

                elif fnName == "bulk_fill_hprobs":
                    mem += cacheSize * wrtLen1 * wrtLen2 * dim * dim # hproduct cache
                    mem += cacheSize * (wrtLen1 + wrtLen2) * dim * dim # dproduct cache
                    mem += cacheSize * dim * dim # product cache
                    mem += cacheSize # scale cache
                    mem += cacheSize # scale vals

                elif fnName == "bulk_hprobs_by_block":
                    #Note: includes "results" memory since this is allocated within
                    # the generator and yielded, *not* allocated by the user.
                    mem += 2 * cacheSize * nspam * wrtLen1 * wrtLen2 # hprobs & dprobs12 results
                    mem += cacheSize * nspam * (wrtLen1 + wrtLen2) # dprobs1 & dprobs2
                    mem += cacheSize * wrtLen1 * wrtLen2 * dim * dim # hproduct cache
                    mem += cacheSize * (wrtLen1 + wrtLen2) * dim * dim # dproduct cache
                    mem += cacheSize * dim * dim # product cache
                    mem += cacheSize # scale cache
                    mem += cacheSize # scale vals

                else:
                    raise ValueError("Unknown subcall name: %s" % fnName)
            
            if verb == 1:
                fc_est_str = " (%.2fGB fc)" % (memEstimate(ng,np1,np2,Ng,True)*C)\
                    if (not fastCacheSz) else ""
                printer.log(" mem(%d subtrees, %d,%d param-grps, %d proc-grps)"
                            % (ng, np1, np2, Ng) + " in %.0fs = %.2fGB%s"
                            % (_time.time()-tm, mem*floatSize*C, fc_est_str))
            elif verb == 2:
                printer.log(" Memory estimate = %.2fGB" % (mem*floatSize*C) +
                     " (cache=%d, wrtLen1=%d, wrtLen2=%d, subsPerProc=%d)." %
                            (cacheSize, wrtLen1, wrtLen2, nSubtreesPerProc))
                #printer.log("  subcalls = %s" % str(subcalls))
                #printer.log("  cacheSize = %d" % cacheSize)
                #printer.log("  wrtLen = %d" % wrtLen)
                #printer.log("  nSubtreesPerProc = %d" % nSubtreesPerProc)
                #if "bulk_fill_dprobs" in subcalls:
                #    printer.log(" DB Detail: dprobs cache = %.2fGB" % 
                #                (8*cacheSize * wrtLen * dim * dim * C))
                #    printer.log(" DB Detail: probs cache = %.2fGB" % 
                #                (8*cacheSize * dim * dim * C))
            return mem * floatSize


        if distributeMethod == "gatestrings":
            np1 = 1; np2 = 1; Ng = nprocs
            ng = nprocs
            if memLimit is not None:
                #Increase ng in amounts of Ng (so ng % Ng == 0).  Start
                # with fast cacheSize computation then switch to slow
                while memEstimate(ng,np1,np2,Ng,True) > memLimit: ng += Ng
                mem_estimate = memEstimate(ng,np1,np2,Ng,verb=1)
                while mem_estimate > memLimit:
                    ng += Ng; next = memEstimate(ng,np1,np2,Ng,verb=1)
                    assert next < mem_estimate, \
                        "Not enough memory: splitting unproductive"
                    mem_estimate = next
                
                   #Note: could do these while loops smarter, e.g. binary search-like?
                   #  or assume memEstimate scales linearly in ng? E.g:
                   #     if memLimit < memEstimate:
                   #         reductionFactor = float(memEstimate) / float(memLimit)
                   #         maxTreeSize = int(nstrs / reductionFactor)
            else:
                memEstimate(ng,np1,np2,Ng) # to compute & cache final EvalTree

        elif distributeMethod == "deriv":

            #Set Ng, the number of subTree processor groups, such
            # that Ng divides nprocs evenly or vice versa
            def set_Ng(desired_Ng):
                if desired_Ng >= nprocs:
                    return nprocs * int(_np.ceil(1.*desired_Ng/nprocs))
                else:
                    fctrs = sorted(prime_factors(nprocs)); i=1
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

                    #OLD
                    #np1 = num_params
                    #np2 = num_params if bNp2Matters else 1
                    #ng = Ng = max(nprocs // (np1*np2), 1)
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
        evt = evt_cache[ng]; evt.distribution['numSubtreeComms'] = Ng

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

        return evt, paramBlkSize1, paramBlkSize2



    def bulk_evaltree(self, gatestring_list, minSubtrees=None, maxTreeSize=None,
                      numSubtreeComms=1, verbosity=0):
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

        verbosity : int, optional
            How much detail to send to stdout.

        Returns
        -------
        EvalTree
            An evaluation tree object.
        """
        tm = _time.time()
        printer = VerbosityPrinter.build_printer(verbosity)
        evalTree = _evaltree.EvalTree()
        evalTree.initialize([""] + list(self.gates.keys()), gatestring_list, numSubtreeComms)

        printer.log("bulk_evaltree: created initial tree (%d strs) in %.0fs" %
                    (len(gatestring_list),_time.time()-tm)); tm = _time.time()

        if maxTreeSize is not None:
            evalTree.split(maxTreeSize, None, printer) # won't split if unnecessary

        if minSubtrees is not None:
            if not evalTree.is_split() or len(evalTree.get_sub_trees()) < minSubtrees:
                evalTree.split(None, minSubtrees, printer)
                if maxTreeSize is not None and \
                        any([ len(sub)>maxTreeSize for sub in evalTree.get_sub_trees()]):
                    _warnings.warn("Could not create a tree with minSubtrees=%d" % minSubtrees
                                   + " and maxTreeSize=%d" % maxTreeSize)
                    evalTree.split(maxTreeSize, None) # fall back to split for max size
        
        if maxTreeSize is not None or minSubtrees is not None:
            printer.log("bulk_evaltree: split tree (%d subtrees) in %.0fs" 
                        % (len(evalTree.get_sub_trees()),_time.time()-tm))
        return evalTree


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


    def bulk_pr(self, spamLabel, evalTree, clipTo=None, check=False, comm=None):
        """
        Compute the probabilities of the gate sequences given by evalTree,
        where initialization & measurement operations are always the same
        and are together specified by spamLabel.

        Parameters
        ----------
        spamLabel : string
           the label specifying the state prep and measure operations

        evalTree : EvalTree
           given by a prior call to bulk_evaltree.  Specifies the gate strings
           to compute the bulk operation on.

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
        numpy array
          An array of length equal to the number of gate strings containing
          the (float) probabilities.
        """
        return self._calc().bulk_pr(spamLabel, evalTree, clipTo, check, comm)


    def bulk_dpr(self, spamLabel, evalTree,
                 returnPr=False,clipTo=None,check=False,
                 comm=None,wrtBlockSize=None):

        """
        Compute the derivatives of the probabilities generated by a each gate
        sequence given by evalTree, where initialization
        & measurement operations are always the same and are
        together specified by spamLabel.

        Parameters
        ----------
        spamLabel : string
           the label specifying the state prep and measure operations

        evalTree : EvalTree
           given by a prior call to bulk_evaltree.  Specifies the gate strings
           to compute the bulk operation on.

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


        Returns
        -------
        dprobs : numpy array
            An array of shape S x M, where

            - S == the number of gate strings
            - M == the length of the vectorized gateset

            and dprobs[i,j] holds the derivative of the i-th probability w.r.t.
            the j-th gateset parameter.

        probs : numpy array
            Only returned when returnPr == True. An array of shape S containing
            the probabilities of each gate string.
        """
        return self._calc().bulk_dpr(spamLabel, evalTree, returnPr,clipTo,
                                     check, comm, None, wrtBlockSize)


    def bulk_hpr(self, spamLabel, evalTree,
                 returnPr=False,returnDeriv=False,
                 clipTo=None,check=False,comm=None,
                 wrtBlockSize1=None, wrtBlockSize2=None):

        """
        Compute the 2nd derivatives of the probabilities generated by a each gate
        sequence given by evalTree, where initialization & measurement
        operations are always the same and are together specified by spamLabel.

        Parameters
        ----------
        spamLabel : string
          the label specifying the state prep and measure operations

        evalTree : EvalTree
          given by a prior call to bulk_evaltree.  Specifies the gate strings
          to compute the bulk operation on.

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


        Returns
        -------
        hessians : numpy array
            a S x M x M array, where

            - S == the number of gate strings
            - M == the length of the vectorized gateset

            and hessians[i,j,k] is the derivative of the i-th probability
            w.r.t. the k-th then the j-th gateset parameter.

        derivs : numpy array
            only returned if returnDeriv == True. A S x M array where
            derivs[i,j] holds the derivative of the i-th probability
            w.r.t. the j-th gateset parameter.

        probabilities : numpy array
            only returned if returnPr == True.  A length-S array
            containing the probabilities for each gate string.
        """
        return self._calc().bulk_hpr(spamLabel, evalTree, returnPr,returnDeriv,
                                    clipTo, check, comm, None, None,
                                     wrtBlockSize1, wrtBlockSize2)


    def bulk_probs(self, evalTree, clipTo=None, check=False, comm=None):
        """
        Construct a dictionary containing the bulk-probabilities
        for every spam label (each possible initialization &
        measurement pair) for each gate sequence given by
        evalTree.

        Parameters
        ----------
        evalTree : EvalTree
           given by a prior call to bulk_evaltree.  Specifies the gate strings
           to compute the bulk operation on.

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
        probs : dictionary
            A dictionary such that
            probs[SL] = bulk_pr(SL,evalTree,clipTo,check)
            for each spam label (string) SL.
        """
        return self._calc().bulk_probs(evalTree, clipTo, check, comm)


    def bulk_dprobs(self, evalTree, returnPr=False,clipTo=None,
                    check=False,comm=None,wrtBlockSize=None):

        """
        Construct a dictionary containing the bulk-probability-
        derivatives for every spam label (each possible
        initialization & measurement pair) for each gate
        sequence given by evalTree.

        Parameters
        ----------
        evalTree : EvalTree
           given by a prior call to bulk_evaltree.  Specifies the gate strings
           to compute the bulk operation on.

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


        Returns
        -------
        dprobs : dictionary
            A dictionary such that
            ``dprobs[SL] = bulk_dpr(SL,evalTree,gates,G0,SPAM,SP0,returnPr,clipTo,check)``
            for each spam label (string) SL.
        """
        return self._calc().bulk_dprobs(evalTree, returnPr,clipTo,
                                        check, comm, None, wrtBlockSize)


    def bulk_hprobs(self, evalTree, returnPr=False,returnDeriv=False,
                    clipTo=None, check=False, comm=None,
                    wrtBlockSize1=None, wrtBlockSize2=None):

        """
        Construct a dictionary containing the bulk-probability-
        Hessians for every spam label (each possible
        initialization & measurement pair) for each gate
        sequence given by evalTree.

        Parameters
        ----------
        evalTree : EvalTree
           given by a prior call to bulk_evaltree.  Specifies the gate strings
           to compute the bulk operation on.

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


        Returns
        -------
        hprobs : dictionary
            A dictionary such that
            ``hprobs[SL] = bulk_hpr(SL,evalTree,returnPr,returnDeriv,clipTo,check)``
            for each spam label (string) SL.
        """
        return self._calc().bulk_hprobs(evalTree, returnPr, returnDeriv,
                                        clipTo, check, comm, None, None,
                                        wrtBlockSize1, wrtBlockSize2)


    def bulk_fill_probs(self, mxToFill, spam_label_rows,
                       evalTree, clipTo=None, check=False, comm=None):
        """
        Identical to bulk_probs(...) except results are
        placed into rows of a pre-allocated array instead
        of being returned in a dictionary.

        Specifically, the probabilities for all gate strings
        and a given SPAM label are placed into the row of
        mxToFill specified by spam_label_rows[spamLabel].

        Parameters
        ----------
        mxToFill : numpy ndarray
          an already-allocated KxS numpy array, where K is larger
          than the maximum value in spam_label_rows and S is equal
          to the number of gate strings (i.e. evalTree.num_final_strings())

        spam_label_rows : dictionary
          a dictionary with keys == spam labels and values which
          are integer row indices into mxToFill, specifying the
          correspondence between rows of mxToFill and spam labels.

        evalTree : EvalTree
           given by a prior call to bulk_evaltree.  Specifies the gate strings
           to compute the bulk operation on.

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
        return self._calc().bulk_fill_probs(mxToFill, spam_label_rows,
                                            evalTree, clipTo, check, comm)


    def bulk_fill_dprobs(self, mxToFill, spam_label_rows,
                         evalTree, prMxToFill=None,clipTo=None,
                         check=False,comm=None, wrtBlockSize=None,
                         profiler=None, gatherMemLimit=None):

        """
        Identical to bulk_dprobs(...) except results are
        placed into rows of a pre-allocated array instead
        of being returned in a dictionary.

        Specifically, the probability derivatives for all gate
        strings and a given SPAM label are placed into
        mxToFill[ spam_label_rows[spamLabel] ].
        Optionally, probabilities can be placed into
        prMxToFill[ spam_label_rows[spamLabel] ]

        Parameters
        ----------
        mxToFill : numpy array
          an already-allocated KxSxM numpy array, where K is larger
          than the maximum value in spam_label_rows, S is equal
          to the number of gate strings (i.e. evalTree.num_final_strings()),
          and M is the length of the vectorized gateset.

        spam_label_rows : dictionary
          a dictionary with keys == spam labels and values which
          are integer row indices into mxToFill, specifying the
          correspondence between rows of mxToFill and spam labels.

        evalTree : EvalTree
           given by a prior call to bulk_evaltree.  Specifies the gate strings
           to compute the bulk operation on.

        prMxToFill : numpy array, optional
          when not None, an already-allocated KxS numpy array that is filled
          with the probabilities as per spam_label_rows, similar to
          bulk_fill_probs(...).

        clipTo : 2-tuple, optional
          (min,max) to clip returned probability to if not None.
          Only relevant when prMxToFill is not None.

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
        return self._calc().bulk_fill_dprobs(mxToFill, spam_label_rows,
                                             evalTree, prMxToFill, clipTo,
                                             check, comm, None, wrtBlockSize,
                                             profiler, gatherMemLimit)


    def bulk_fill_hprobs(self, mxToFill, spam_label_rows,
                         evalTree=None, prMxToFill=None, derivMxToFill=None,
                         clipTo=None, check=False, comm=None, 
                         wrtBlockSize1=None, wrtBlockSize2=None,
                         gatherMemLimit=None):

        """
        Identical to bulk_hprobs(...) except results are
        placed into rows of a pre-allocated array instead
        of being returned in a dictionary.

        Specifically, the probability hessians for all gate
        strings and a given SPAM label are placed into
        mxToFill[ spam_label_rows[spamLabel] ].
        Optionally, probabilities and/or derivatives can be placed into
        prMxToFill[ spam_label_rows[spamLabel] ] and
        derivMxToFill[ spam_label_rows[spamLabel] ] respectively.

        Parameters
        ----------
        mxToFill : numpy array
          an already-allocated KxSxMxM numpy array, where K is larger
          than the maximum value in spam_label_rows, S is equal
          to the number of gate strings (i.e. evalTree.num_final_strings()),
          and M is the length of the vectorized gateset.

        spam_label_rows : dictionary
          a dictionary with keys == spam labels and values which
          are integer row indices into mxToFill, specifying the
          correspondence between rows of mxToFill and spam labels.

        evalTree : EvalTree
           given by a prior call to bulk_evaltree.  Specifies the gate strings
           to compute the bulk operation on.

        prMxToFill : numpy array, optional
          when not None, an already-allocated KxS numpy array that is filled
          with the probabilities as per spam_label_rows, similar to
          bulk_fill_probs(...).

        derivMxToFill : numpy array, optional
          when not None, an already-allocated KxSxM numpy array that is filled
          with the probability derivatives as per spam_label_rows, similar to
          bulk_fill_dprobs(...).

        clipTo : 2-tuple
          (min,max) to clip returned probability to if not None.
          Only relevant when prMxToFill is not None.

        check : boolean, optional
          If True, perform extra checks within code to verify correctness,
          generating warnings when checks fail.  Used for testing, and runs
          much slower when True.

        comm : mpi4py.MPI.Comm, optional
           When not None, an MPI communicator for distributing the computation
           across multiple processors.  Distribution is first performed over
           subtrees of evalTree (if it is split), and then over blocks (subsets)
           of the parameters being second-differentiated with respect to (see
           wrtBlockSize).

        wrtBlockSize2, wrtBlockSize2 : int or float, optional
          The maximum number of 1st (row) and 2nd (col) derivatives to compute
          *products* for simultaneously.  None means compute all requested
          rows or columns at once.  The  minimum of wrtBlockSize and the size
          that makes maximal use of available processors is used as the final
          block size.  These arguments must be None if the corresponding
          wrtFilter is not None.  Set this to non-None to reduce amount of
          intermediate memory required.

        gatherMemLimit : int, optional
          A memory limit in bytes to impose upon the "gather" operations
          performed as a part of MPI processor syncronization.


        Returns
        -------
        None
        """
        return self._calc().bulk_fill_hprobs(mxToFill, spam_label_rows,
                                     evalTree, prMxToFill, derivMxToFill, None,
                                     clipTo, check, comm, None, None,
                                     wrtBlockSize1,wrtBlockSize2,gatherMemLimit)


    def bulk_hprobs_by_block(self, spam_label_rows, evalTree, wrtSlicesList,
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
            spam_label_rows, evalTree, wrtSlicesList,
            bReturnDProbs12, comm)
            

    def frobeniusdist(self, otherGateSet, transformMx=None,
                      gateWeight=1.0, spamWeight=1.0, itemWeights=None,
                      normalize=True):
        """
        Compute the weighted frobenius norm of the difference between this
        gateset and otherGateSet.  Differences in each corresponding gate
        matrix and spam vector element are squared, weighted (using gateWeight
        or spamWeight as applicable), then summed.  The value returned is the
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

        gateWeight : float, optional
           weighting factor for differences between gate elements.

        spamWeight : float, optional
           weighting factor for differences between elements of spam vectors.

        itemWeights : dict, optional
           Dictionary of weighting factors for individual gates and spam
           operators. Weights are applied multiplicatively to the squared
           differences, i.e., (*before* the final square root is taken).  Keys
           can be gate, state preparation, POVM effect, or spam labels.  Values
           are floating point numbers.  By default, weights are set by
           gateWeight and spamWeight.

        normalize : bool, optional
           if True (the default), the frobenius difference is defined by the
           sum of weighted squared-differences divided by the number of
           differences.  If False, this final division is not performed.

        Returns
        -------
        float
        """
        return self._calc().frobeniusdist(otherGateSet._calc(), transformMx,
                                          gateWeight, spamWeight, itemWeights,
                                          normalize)


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

        s += " Effects:\n"
        for lbl in self.effects:
            s += "  %s = %g\n" % \
                (lbl, _np.linalg.norm(self.effects[lbl]-otherGateSet.effects[lbl]))
        if self.povm_identity is not None:
            s += "  Identity = %g\n" % \
                _np.linalg.norm(self.povm_identity-otherGateSet.povm_identity)

        s += " Gates:\n"
        for lbl in self.gates:
            s += "  %s = %g\n" % \
                (lbl, _np.linalg.norm(self.gates[lbl]-otherGateSet.gates[lbl]))

        return s



    def copy(self):
        """
        Copy this gateset

        Returns
        -------
        GateSet
            a (deep) copy of this gateset.
        """
        newGateset = GateSet()
        newGateset.preps = self.preps.copy(newGateset)
        newGateset.effects = self.effects.copy(newGateset)
        newGateset.gates = self.gates.copy(newGateset)
        newGateset.spamdefs = self.spamdefs.copy()
        newGateset.povm_identity = self.povm_identity.copy()
        newGateset._dim = self._dim
        newGateset._basisNameAndDim = self._basisNameAndDim
        newGateset._remainderlabel = self._remainderlabel
        newGateset._identitylabel = self._identitylabel
        newGateset._default_gauge_group = self._default_gauge_group
        return newGateset

    def __str__(self):
        s = ""
        for (lbl,vec) in self.preps.items():
            s += "%s = " % lbl + _mt.mx_to_string(_np.transpose(vec)) + "\n"
        s += "\n"
        for (lbl,vec) in self.effects.items():
            s += "%s = " % lbl + _mt.mx_to_string(_np.transpose(vec)) + "\n"
        s += "\n"
        for (lbl,gate) in self.gates.items():
            s += "%s = \n" % lbl + _mt.mx_to_string(gate) + "\n\n"
        return s


    def iter_gates(self):
        """
        Returns
        -------
        iterator
            an iterator over all (gateLabel,gate) pairs
        """
        for (label,gate) in self.gates.items():
            yield (label, gate)

    def iter_preps(self):
        """
        Returns
        -------
        iterator
            an iterator over all (prepLabel,vector) pairs
        """
        for (label,vec) in self.preps.items():
            yield (label, vec)

    def iter_effects(self):
        """
        Returns
        -------
        iterator
            an iterator over all (effectLabel,vector) pairs
        """
        for (label,vec) in self.effects.items():
            yield (label, vec)



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
            r = max_gate_noise * rndm.random_sample( len(self.gates) )
            for (i,label) in enumerate(self.gates):
                D = _np.diag( [1]+[1-r[i]]*(gateDim-1) )
                newGateset.gates[label] = _gate.FullyParameterizedGate(
                                           _np.dot(D,self.gates[label]) )

        elif gate_noise is not None:
            #Apply the same depolarization to each gate
            D = _np.diag( [1]+[1-gate_noise]*(gateDim-1) )
            for (i,label) in enumerate(self.gates):
                newGateset.gates[label].depolarize(gate_noise)

        if max_spam_noise is not None:
            if spam_noise is not None:
                raise ValueError("Must specify at most  one of 'noise' and 'max_noise' NOT both")

            #Apply random depolarization to each rho and E vector
            r = max_spam_noise * rndm.random_sample( len(self.preps) )
            for (i,lbl) in enumerate(self.preps):
                D = _np.diag( [1]+[1-r[i]]*(gateDim-1) )
                newGateset.preps[lbl] = _sv.FullyParameterizedSPAMVec(
                                           _np.dot(D,self.preps[lbl]))

            r = max_spam_noise * rndm.random_sample( len(self.effects) )
            for (i,lbl) in enumerate(self.effects):
                D = _np.diag( [1]+[1-r[i]]*(gateDim-1) )
                newGateset.effects[lbl] = _sv.FullyParameterizedSPAMVec(
                                         _np.dot(D,self.effects[lbl]))

        elif spam_noise is not None:
            #Apply the same depolarization to each gate
            D = _np.diag( [1]+[1-spam_noise]*(gateDim-1) )
            for lbl,rhoVec in self.preps.items():
                newGateset.preps[lbl].depolarize(spam_noise)
            for lbl,EVec in self.effects.items():
                newGateset.effects[lbl].depolarize(spam_noise)

        return newGateset


    def rotate(self, rotate=None, max_rotate=None, seed=None):
        """
        Apply rotation uniformly or randomly to this gateset,
        and return the result, without modifying the original
        (this) gateset.  You must specify either 'rotate' or
        'max_rotate'. This method currently only works on
        single-qubit gatesets.

        Parameters
        ----------
        rotate : float or tuple of floats, optional
          if a single float, apply rotation of rotate radians along each of
          the pauli-product axes (X,Y,Z for 1-qubit) of all gates in the gateset.
          For a 1-qubit gateset, a 3-tuple of floats can be specifed to apply
          separate rotations along the X, Y, and Z axes.  For a 2-qubit gateset,
          a 15-tuple of floats can be specified to apply separate rotations along
          the IX, IY, IZ, XI, XX, XY, XZ, YI, YX, YY, YZ, ZI, ZX, ZY, ZZ axes.

        max_rotate : float, optional
          specified instead of 'rotate'; apply a random rotation with
          maximum max_rotate radians along each of the relevant axes
          of each each gate in the gateset.  That is, rotations of a
          particular gate around different axes are different random amounts.

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

        #NEEDED?
        #for (i,rhoVec) in enumerate(self.preps):
        #    newGateset.set_rhovec( rhoVec, i )
        #for (i,EVec) in enumerate(self.effects):
        #    newGateset.set_evec( EVec, i )

        if dim not in (4, 16):
            raise ValueError("Gateset dimension must be either 4 (single-qubit) or 16 (two-qubit)")

        rndm = _np.random.RandomState(seed)

        if max_rotate is not None:
            if rotate is not None:
                raise ValueError("Must specify exactly one of 'rotate' and 'max_rotate' NOT both")

            #Apply random rotation to each gate
            if dim == 4:
                r = max_rotate * rndm.random_sample( len(self.gates) * 3 )
                for (i,label) in enumerate(self.gates):
                    rot = r[3*i:3*(i+1)]
                    newGateset.gates[label] = _gate.FullyParameterizedGate(
                       _np.dot( _bt.single_qubit_gate(rot[0]/2.0,
                                rot[1]/2.0,rot[2]/2.0), self.gates[label]) )

            elif dim == 16:
                r = max_rotate * rndm.random_sample( len(self.gates) * 15 )
                for (i,label) in enumerate(self.gates):
                    rot = r[15*i:15*(i+1)]
                    newGateset.gates[label] = _gate.FullyParameterizedGate(
                     _np.dot( _bt.two_qubit_gate(rot[0]/2.0,rot[1]/2.0,rot[2]/2.0,
                                                 rot[3]/2.0,rot[4]/2.0,rot[5]/2.0,
                                                 rot[6]/2.0,rot[7]/2.0,rot[8]/2.0,
                                                 rot[9]/2.0,rot[10]/2.0,rot[11]/2.0,
                                                 rot[12]/2.0,rot[13]/2.0,rot[14]/2.0,
                                                 ), self.gates[label]) )
            #else: raise ValueError("Invalid gateset dimension") # checked above

        elif rotate is not None:

            if dim == 4:
                #Specify rotation by a single value (to mean this rotation along each axis) or a 3-tuple
                if type(rotate) in (float,int): rx,ry,rz = rotate,rotate,rotate
                elif type(rotate) in (tuple,list):
                    if len(rotate) != 3:
                        raise ValueError("Rotation, when specified as a tuple "
                                + "must be of length 3, not: %s" % str(rotate))
                    (rx,ry,rz) = rotate
                else: raise ValueError("Rotation must be specifed as a single "
                       + "number or as a lenght-3 list, not: %s" % str(rotate))

                for (i,label) in enumerate(self.gates):
                    newGateset.gates[label] = _gate.FullyParameterizedGate(
                        _np.dot(
                            _bt.single_qubit_gate(rx/2.0,ry/2.0,rz/2.0),
                            self.gates[label]) )

            elif dim == 16:
                #Specify rotation by a single value (to mean this rotation along each axis) or a 15-tuple
                if type(rotate) in (float,int):
                    rix,riy,riz = rotate,rotate,rotate
                    rxi,rxx,rxy,rxz = rotate,rotate,rotate,rotate
                    ryi,ryx,ryy,ryz = rotate,rotate,rotate,rotate
                    rzi,rzx,rzy,rzz = rotate,rotate,rotate,rotate
                elif type(rotate) in (tuple,list):
                    if len(rotate) != 15:
                        raise ValueError("Rotation, when specified as a tuple "
                             + "must be of length 15, not: %s" % str(rotate))
                    (rix,riy,riz,rxi,rxx,rxy,rxz,ryi,
                     ryx,ryy,ryz,rzi,rzx,rzy,rzz) = rotate
                else: raise ValueError("Rotation must be specifed as a single "
                      + "number or as a lenght-15 list, not: %s" % str(rotate))

                for (i,label) in enumerate(self.gates):
                    newGateset.gates[label] = _gate.FullyParameterizedGate(
                        _np.dot(
                            _bt.two_qubit_gate(rix/2.0,riy/2.0,riz/2.0,
                                               rxi/2.0,rxx/2.0,rxy/2.0,rxz/2.0,
                                               ryi/2.0,ryx/2.0,ryy/2.0,ryz/2.0,
                                               rzi/2.0,rzx/2.0,rzy/2.0,rzz/2.0,)
                            , self.gates[label]) )
            #else: raise ValueError("Invalid gateset dimension") # checked above

        else: raise ValueError("Must specify either 'rotate' or 'max_rotate' "
                               + "-- neither was non-None")
        return newGateset


    def randomize_with_unitary(self, scale, seed=None, randState=None):
        """Create a new gateset with random unitary perturbations.

        Apply a random unitary to each element of a gateset, and return the
        result, without modifying the original (this) gateset. This method
        currently only works on single- and two-qubit gatesets, and *assumes*
        that the gate matrices of this gateset are being interpreted in the
        Pauli-product basis.

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
        gs_pauli = self.copy() # assumes self is in the pauli-product basis!!
        if randState is None:
            rndm = _np.random.RandomState(seed)
        else:
            rndm = randState

        gate_dim = gs_pauli.get_dimension()
        if gate_dim == 4:
            unitary_dim = 2
        elif gate_dim == 16:
            unitary_dim = 4
        else: raise ValueError("Gateset dimension must be either 4"
                               " (single-qubit) or 16 (two-qubit)")

        for gateLabel in list(gs_pauli.gates.keys()):
            randMat = scale * (rndm.randn(unitary_dim,unitary_dim) \
                                   + 1j * rndm.randn(unitary_dim,unitary_dim))
#            randMat = _np.dot(_np.transpose(_np.conjugate(randMat)),randMat)
            randMat = _np.transpose(_np.conjugate(randMat)) + randMat
                        # make randMat Hermetian: (A_dag + A)^dag = (A_dag + A)
            randU   = _scipy.linalg.expm(-1j*randMat)

            if unitary_dim == 2:
                randUPP = _bt.unitary_to_pauligate_1q(randU)
            elif unitary_dim == 4:
                randUPP = _bt.unitary_to_pauligate_2q(randU)
            # No else case needed, unitary_dim MUST be at either 2 or 4 at this point
            #   (Redundant exception)

            gs_pauli.gates[gateLabel] = _gate.FullyParameterizedGate(
                            _np.dot(randUPP,gs_pauli.gates[gateLabel]))

        return gs_pauli


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

        new_gateset = GateSet("full", self.preps._prefix, self.effects._prefix,
                              self.gates._prefix, self._remainderlabel,
                              self._identitylabel)
        new_gateset._dim = newDimension
        new_gateset.reset_basis() #FUTURE: maybe user can specify how increase is being done?
        new_gateset.spamdefs.update( self.spamdefs )

        addedDim = newDimension-curDim
        vec_zeroPad = _np.zeros( (addedDim,1), 'd')

        #Increase dimension of rhoVecs and EVecs by zero-padding
        for lbl,rhoVec in self.preps.items():
            assert( len(rhoVec) == curDim )
            new_gateset.preps[lbl] = \
                _sv.FullyParameterizedSPAMVec(_np.concatenate( (rhoVec, vec_zeroPad) ))

        for lbl,EVec in self.effects.items():
            assert( len(EVec) == curDim )
            new_gateset.effects[lbl] = \
                _sv.FullyParameterizedSPAMVec(_np.concatenate( (EVec, vec_zeroPad) ))

        #Increase dimension of identityVec by zero-padding
        if self.povm_identity is not None:
            new_gateset.povm_identity = _sv.FullyParameterizedSPAMVec(
                            _np.concatenate( (self.povm_identity, vec_zeroPad) ))

        #Increase dimension of gates by assuming they act as identity on additional (unknown) space
        for gateLabel,gate in self.gates.items():
            assert( gate.shape == (curDim,curDim) )
            newGate = _np.zeros( (newDimension,newDimension) )
            newGate[ 0:curDim, 0:curDim ] = gate[:,:]
            for i in range(curDim,newDimension): newGate[i,i] = 1.0
            new_gateset.gates[gateLabel] = _gate.FullyParameterizedGate(newGate)

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

        new_gateset = GateSet("full", self.preps._prefix, self.effects._prefix,
                              self.gates._prefix, self._remainderlabel,
                              self._identitylabel)
        new_gateset._dim = newDimension
        new_gateset.reset_basis() #FUTURE: maybe user can specify how decrease is being done?
        new_gateset.spamdefs.update( self.spamdefs )

        #Decrease dimension of rhoVecs and EVecs by truncation
        for lbl,rhoVec in self.preps.items():
            assert( len(rhoVec) == curDim )
            new_gateset.preps[lbl] = \
                _sv.FullyParameterizedSPAMVec(rhoVec[0:newDimension,:])

        for lbl,EVec in self.effects.items():
            assert( len(EVec) == curDim )
            new_gateset.effects[lbl] = \
                _sv.FullyParameterizedSPAMVec(EVec[0:newDimension,:])

        #Decrease dimension of identityVec by trunction
        if self.povm_identity is not None:
            new_gateset.povm_identity = _sv.FullyParameterizedSPAMVec(
                                self.povm_identity[0:newDimension,:])

        #Decrease dimension of gates by truncation
        for gateLabel,gate in self.gates.items():
            assert( gate.shape == (curDim,curDim) )
            newGate = _np.zeros( (newDimension,newDimension) )
            newGate[ :, : ] = gate[0:newDimension,0:newDimension]
            new_gateset.gates[gateLabel] = _gate.FullyParameterizedGate(newGate)

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
        return kicked_gs


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
        print("Choi Matrices:")
        for (label,gate) in self.gates.items():
            print(("Choi(%s) in pauli basis = \n" % label,
            _mt.mx_to_string_complex(_jt.jamiolkowski_iso(gate))))
            print(("  --eigenvals = ", sorted(
                [ev.real for ev in _np.linalg.eigvals(
                        _jt.jamiolkowski_iso(gate))] ),"\n"))
        print(("Sum of negative Choi eigenvalues = ", _jt.sum_of_negative_choi_evals(self)))

        prep_penalty = sum( [ _lf.prep_penalty(rhoVec)
                                for rhoVec in list(self.preps.values()) ] )
        effect_penalty   = sum( [ _lf.effect_penalty(EVec)
                                for EVec in list(self.effects.values()) ] )
        print(("rhoVec Penalty (>0 if invalid rhoVecs) = ", prep_penalty))
        print(("EVec Penalty (>0 if invalid EVecs) = ", effect_penalty))



#########################################
## DEPRECATED
#########################################
#    def set_identity_vec(self, identityVec):
#        """
#        Set a the identity vector.  Calls
#        make_spams automatically.
#
#        Parameters
#        ----------
#        identityVec : numpy array
#            a column vector containing the identity vector.
#        """
#        self.parameterization.set_identityvec( identityVec )
#        self._rawupdate()
#
#    def get_identity_vec(self):
#        """
#        Get the identity vector in the basis being used by this gateset.
#
#        Returns
#        -------
#        numpy array
#            The identity column vector.  Note this is a reference to
#            the GateSet's internal object, so callers should copy the
#            vector before changing it.
#        """
#        return self.povm_identity
#
#    def set_rhovec(self, rhoVec, index=0):
#        """
#        Set a state prepartion vector by index.  Calls make_spams automatically.
#
#        Parameters
#        ----------
#        rhoVec : numpy array
#            a column vector containing the state preparation vector.
#
#        index : int, optional
#            the index of the state preparation vector to set.  Must
#            be <= num_preps(), where equality adds a new vector.
#        """
#        self.parameterization.set_rhovec(rhoVec, index)
#        self._rawupdate()
#
#
#    def get_rhovec(self, index=0):
#        """
#        Get a state prepartion vector by index.
#
#        Parameters
#        ----------
#        index : int, optional
#            the index of the vector to return.
#
#        Returns
#        -------
#        numpy array
#            a state preparation vector of shape (dim, 1).
#        """
#        return self.preps[index]
#
#
#
#
#    def set_evec(self, EVec, index=0):
#        """
#        Set a POVM effect vector by index.  Calls make_spams automatically.
#
#        Parameters
#        ----------
#        rhoVec : numpy array
#            a column vector containing the effect vector.
#
#        index : int, optional
#            the index of the effect vector to set.  Must
#            be <= num_effects(), where equality adds a new vector.
#        """
#        self.parameterization.set_evec(EVec, index)
#        self._rawupdate()
#
#
#    def get_evec(self, index=0):
#        """
#        Get a POVM effect vector by index.
#
#        Parameters
#        ----------
#        index : int, optional
#            the index of the vector to return.
#
#        Returns
#        -------
#        numpy array
#            an effect vector of shape (dim, 1).
#        """
#        if index == -1:
#            return self.povm_identity - sum(self.effects)
#        else:
#            return self.effects[index]
#
#
#    def get_rhovec_indices(self):
#        """
#        Get the indices of state preparation vectors.
#
#        Returns
#        -------
#        list of ints
#        """
#        return range(len(self.preps))
#
#    def get_evec_indices(self):
#        """
#        Get the indices of effect vectors, possibly including -1 as
#          the index of a "complement" effect vector,
#          equal to Identity - sum(other effect vectors)
#
#        Returns
#        -------
#        list of ints
#        """
#        inds = range(len(self.effects))
#        if any( [ (EIndx == -1 and rhoIndx != -1) for (rhoIndx,EIndx) in self.spamdefs.values() ]):
#            inds.append( -1 )
#        return inds
#
#    def add_spam_definition(self, rhoIndex, eIndex, label):
#        """
#        Adds a new spam label.  That is, associates the SPAM
#          pair (rhoIndex,eIndex) with the given label.  Calls
#          make_spams automatically.
#
#        Parameters
#        ----------
#        rhoIndex : int
#            state preparation vector index.
#
#        eIndex : int
#            POVM effect vector index.
#
#        label : string
#            the "spam label" to associate with (rhoIndex,eIndex).
#        """
#XXX
#        self.make_spams()
#
#    def get_reverse_spam_defs(self):
#        """
#        Get a reverse-lookup dictionary for spam labels.
#
#        Returns
#        -------
#        dict
#            a dictionary with keys == (rhoIndex,eIndex) tuples and
#            values == SPAM labels.
#        """
#        d = { }
#        for label in self.spamdefs:
#            d[  self.spamdefs[label] ] = label
#        return d
#
#    def get_spam_labels(self):
#        """
#        Get a list of all the spam labels.
#
#        Returns
#        -------
#        list of strings
#        """
#        return self.spamdefs.keys()
#
#
#    def __setitem__(self, label, gate):
#        """
#        Set the Gate matrix or object associated with a given label.
#
#        Parameters
#        ----------
#        label : string
#            the gate label.
#
#        gate : numpy array or Gate
#            a gate matrix or object, which must have the dimension
#            of the GateSet.
#        """
#        self.parameterization.set_gate(label,gate)
#        self._rawupdate(gateLabel=label)
#
#    def __reduce__(self):
#        #Used by pickle; needed because OrderedDict uses __reduce__ and we don't want
#        #  that one to be called, so we override...
#        return (GateSet, (), self.__getstate__())
#            #Construct a GateSet class, passing init no parameters, and state given by __getstate__
#
#    def __getstate__(self):
#        #Return the state (for pickling)
#        mystate = { 'parameterization': self.parameterization,
#                    'spamdefs': self.spamdefs,
#                    'SPAMs' : self.SPAMs,
#                    'history' : self.history,
#                    'assumeSumToOne' : self.assumeSumToOne
#                    }
#                    #'rhoVecs' : self.preps,
#                    #'EVecs': self.effects,
#                    #'identityVec': self.povm_identity,
#                    #'gate_dim': self.gate_dim,
#                    #'gates': self.gates,
#        return mystate
#
#    def __setstate__(self, stateDict):
#        #Initialize a GateSet from a state dictionary (for un-pickling)
#        #self.gate_dim = stateDict['gate_dim']
#        self.parameterization = stateDict['parameterization']
#        #self.preps = stateDict['rhoVecs']
#        #self.effects = stateDict['EVecs']
#        #self.gates = stateDict['gates']
#        #self.povm_identity = stateDict['identityVec']
#        self.spamdefs = stateDict['spamdefs']
#        self.SPAMs = stateDict['SPAMs']
#        self.history = stateDict['history']
#        self.assumeSumToOne = stateDict['assumeSumToOne']
#
#        #Don't serialize dictionary elements (gate matrices) of self since they
#        # are generated by the parameterization:
#        self._rawupdate() #sets gates, EVecs, rhoVecs, identityVec
#
#    def _rawupdate(self, gateLabel=None):
#        #TODO: filter using gateLabel?
#        for gateLabel, gateMatrix in self.parameterization.iter_gate_matrices():
#            super(GateSet, self).__setitem__(gateLabel, gateMatrix)
#
#        for index,rhoVec in self.parameterization.iter_rho_vectors():
#            if index < len(self.preps):
#                self.preps[index] = rhoVec
#            else:
#                self.preps.append(rhoVec)
#
#        for index,EVec in self.parameterization.iter_e_vectors():
#            if index < len(self.effects):
#                self.effects[index] = EVec
#            else:
#                self.effects.append(EVec)
#
#        self.povm_identity = self.parameterization.compute_identity_vector()
#        self.make_spams()
#
#    def get_gate(self,label):
#        """
#        Get the Gate matrix associated with a given label.
#
#        Parameters
#        ----------
#        label : string
#            the gate label.
#
#        Returns
#        -------
#        Gate
#        """
#        return self[label]
#
#    def set_gate(self,label,gate):
#        """
#        Set the Gate matrix or object associated with a given label.
#
#        Parameters
#        ----------
#        label : string
#            the gate label.
#
#        gate : numpy array or Gate
#            a gate matrix or object, which must have the dimension
#            of the GateSet.
#        """
#        self.parameterization.set_gate(label,gate)
#        self._rawupdate(gateLabel=label)
#
#
#    def update(self, *args, **kwargs): #So that our __setitem__ is always called
#        """ Updates the Gateset as a dictionary """
#        #raise ValueError("Update on gatesets is not implemented")
#        if args:
#            if len(args) > 1:
#                raise TypeError("update expected at most 1 arguments, got %d" % len(args))
#            other = dict(args[0])
#            for key in other:
#                self[key] = other[key]
#        for key in kwargs:
#            self[key] = kwargs[key]
#
#    def setdefault(self, key, value=None): #So that our __setitem__ is always called
#        raise ValueError("setdefault on gatesets is not implemented")
#
#
#    def log(self, strDescription, extra=None):
#        """
#        Append a message to the log of this gateset.
#
#        Parameters
#        ----------
#        strDescription : string
#            a string description
#
#        extra : anything, optional
#            any additional variable to log along with strDescription.
#        """
#        self.history.append( (strDescription,extra) )
