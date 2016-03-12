#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation              
#    This Software is released under the GPL license detailed    
#    in the file "license.txt" in the top-level pyGSTi directory 
#*****************************************************************
""" Defines the GateSetCalculator class"""

import warnings as _warnings
import numpy as _np
import numpy.linalg as _nla

from ..tools import gatetools as _gt

#import evaltree as _evaltree


# Smallness tolerances, used internally for conditional scaling required
# to control bulk products, their gradients, and their Hessians.
PSMALL = 1e-100
DSMALL = 1e-100
HSMALL = 1e-100

class GateSetCalculator(object):
    """
    Encapsulates a calulation tool used by gate set objects to perform product
    and derivatives-of-product calculations.

    This is contained in a class separate from GateSet to allow for additional
    gate set classes (e.g. ones which use entirely different -- non-gate-local
    -- parameterizations of gate matrices and SPAM vectors) access to these
    fundamental operations.
    """

    def __init__(self, dim, gates, rhovecs, evecs, identityvec, spamLabels,
                 remainderLabel):
        """
        Construct a new GateSetCalculator object.

        Parameters
        ----------
        dim : int
            The gate-dimension.  All gate matrices should be dim x dim, and all
            SPAM vectors should be dim x 1.

        gates, rhovecs, evecs : OrderedDict
            Ordered dictionaries of Gate, SPAMVec, and SPAMVec objects,
            respectively.  Must be *ordered* dictionaries to specify a
            well-defined column ordering when taking derivatives.

        identityvec : SPAMVec
            Identity vector (shape must be dim x 1) used when spamLabels
            contains the value (<rho_label>,"remainder"), which specifies
            a POVM effect that is the identity minus the sum of all the 
            effect vectors in evecs.

        spamLabels : OrderedDict
            A dictionary whose keys are the allowed SPAM labels, and whose
            values are 2-tuples comprised of a state preparation label 
            followed by a POVM effect label (both of which are strings,
            and keys of rhovecs and evecs, respectively, except for the
            special case when eith both or just the effect label is set
            to "remainder").            

        remainderLabel : string
            A string that may appear in the values of spamLabels to designate
            special behavior.
        """
        self._remainderLabel = remainderLabel
        self.dim = dim
        self.gates = gates
        self.rhovecs = rhovecs
        self.identityvec = identityvec
        self.spamLabels = spamLabels
        self.assumeSumToOne = bool( (self._remainderLabel,self._remainderLabel) in spamLabels)
          #Whether spamLabels contains the value ("remainder", "remainder"),
          #  which specifies a spam label that generates probabilities such that
          #  all SPAM label probabilities sum exactly to 1.0.



    def _is_remainder_spamlabel(label):
        """
        Returns whether or not the given SPAM label is the
        special "remainder" SPAM label which generates 
        probabilities such that all SPAM label probabilities
        sum exactly to 1.0.
        """
        return bool(self.spamLabels[label] == (self._remainderLabel, self._remainderLabel))

    def _get_evec(elabel):
        """
        Get a POVM effect vector by label.

        Parameters
        ----------
        elabel : string
            the label of the POVM effect vector to return.

        Returns
        -------
        numpy array
            an effect vector of shape (dim, 1).
        """
        if elabel == self._remainderLabel:
            return self.identityvec - sum(self.evecs)
        else:
            return self.evecs[elabel]

    def _make_spamgate(self, spamlabel):
        rhoLabel,eLabel = self.spamLabels[spamlabel]
        if rhoLabel == self._remainderLabel: 
            return None

        rho,E = self.rhovecs[rhoLabel], self._get_evec(eLabel)
        return _np.kron(rho, _np.conjugate(_np.transpose(E)))


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
        if bScale:
            scaledGatesAndExps = {}; 
            for (gateLabel,gatemx) in self.gates:
                ng = max(_nla.norm(gatemx),1.0)
                scaledGatesAndExps[gateLabel] = (gatemx / ng, _np.log(ng))

            scale_exp = 0
            G = _np.identity( self.dim )
            for lGate in gatestring:
                gate, ex = scaledGatesAndExps[lGate]
                H = _np.dot(gate,G)   # product of gates, starting with identity
                scale_exp += ex   # scale and keep track of exponent
                if H.max() < PSMALL and H.min() > -PSMALL:
                    nG = max(_nla.norm(G), _np.exp(-scale_exp))
                    G = _np.dot(gate,G/nG); scale_exp += _np.log(nG) 
                    #OLD: _np.dot(G/nG,gate); scale_exp += _np.log(nG) LEXICOGRAPHICAL VS MATRIX ORDER
                else: G = H

            old_err = _np.seterr(over='ignore')
            scale = _np.exp(scale_exp)
            _np.seterr(**old_err)

            return G, scale
        
        else:
            G = _np.identity( self.dim )
            for lGate in gatestring:
                G = _np.dot(self.gates[lGate],G)  # product of gates
                #OLD: G = _np.dot(G,self[lGate]) LEXICOGRAPHICAL VS MATRIX ORDER
            return G


    #Vectorizing Identities. (Vectorization)
    # Note when vectorizing op uses numpy.flatten rows are kept contiguous, so the first identity below is valid.
    # Below we use E(i,j) to denote the elementary matrix where all entries are zero except the (i,j) entry == 1
    
    # if vec(.) concatenates rows (which numpy.flatten does)
    # vec( A * E(0,1) * B ) = vec( mx w/ row_i = A[i,0] * B[row1] ) = A tensor B^T * vec( E(0,1) )
    # In general: vec( A * X * B ) = A tensor B^T * vec( X )
    
    # if vec(.) stacks columns
    # vec( A * E(0,1) * B ) = vec( mx w/ col_i = A[col0] * B[0,1] ) = B^T tensor A * vec( E(0,1) )
    # In general: vec( A * X * B ) = B^T tensor A * vec( X )

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

        # LEXICOGRAPHICAL VS MATRIX ORDER
        revGateLabelList = tuple(reversed(tuple(gatestring))) # we do matrix multiplication in this order (easier to think about)

        #  prod = G1 * G2 * .... * GN , a matrix
        #  dprod/d(gateLabel)_ij   = sum_{L s.t. G(L) == gatelabel} [ G1 ... G(L-1) dG(L)/dij G(L+1) ... GN ] , a matrix for each given (i,j)
        #  vec( dprod/d(gateLabel)_ij ) = sum_{L s.t. G(L) == gatelabel} [ (G1 ... G(L-1)) tensor (G(L+1) ... GN)^T vec( dG(L)/dij ) ]
        #                               = [ sum_{L s.t. G(L) == gatelabel} [ (G1 ... G(L-1)) tensor (G(L+1) ... GN)^T ]] * vec( dG(L)/dij) )
        #  if dG(L)/dij = E(i,j) 
        #                               = vec(i,j)-col of [ sum_{L s.t. G(L) == gatelabel} [ (G1 ... G(L-1)) tensor (G(L+1) ... GN)^T ]]
        # So for each gateLabel the matrix [ sum_{L s.t. GL == gatelabel} [ (G1 ... G(L-1)) tensor (G(L+1) ... GN)^T ]] has columns which 
        #  correspond to the vectorized derivatives of each of the product components (i.e. prod_kl) with respect to a given gateLabel_ij
        # This function returns a concatenated form of the above matrices, so that each column corresponds to a (gateLabel,i,j) tuple and
        #  each row corresponds to an element of the product (els of prod.flatten()).
        #
        # Note: if gate G(L) is just a matrix of parameters, then dG(L)/dij = E(i,j), an elementary matrix

        dim = self.dim

        #Cache partial products
        leftProds = [ ]
        G = _np.identity( dim ); leftProds.append(G)
        for gateLabel in revGateLabelList:
            G = _np.dot(G,self.gates[gateLabel]); leftProds.append(G)

        rightProdsT = [ ]
        G = _np.identity( dim ); rightProdsT.append( _np.transpose(G) )
        for gateLabel in reversed(revGateLabelList):
            G = _np.dot(self.gates[gateLabel],G); rightProdsT.append( _np.transpose(G) )

        # Initialize storage
        dprod_dgateLabel = { }; dgate_dgateLabel = {}
        for gateLabel,gate in self.gates.iteritems():
            dprod_dgateLabel[gateLabel] = _np.zeros( (dim**2, gate.num_params() ) )
            dgate_dgateLabel[gateLabel] = gate.deriv_wrt_params() # (dim**2, nParams[gateLabel])
            #Note: replace gate.num_params()  and .deriv_wrt_params() with something more
            # general like parameterizer.get_num_params(gateLabel), etc, to allow for non-gate-local
            # gatesets params? In this case, would expect output to be of shape (dim**2, nTotParams)
            # and would *add* these together instead of concatenating below.

        #Add contributions for each gate in list
        N = len(revGateLabelList)
        for (i,gateLabel) in enumerate(revGateLabelList):
                dprod_dgate = _np.kron( leftProds[i], rightProdsT[N-1-i] )  # (dim**2, dim**2)
                dprod_dgateLabel[gateLabel] += _np.dot( dprod_dgate, dgate_dgateLabel[gateLabel] ) # (dim**2, nParams[gateLabel])
            
        #Concatenate per-gateLabel results to get final result
        to_concat = [ dprod_dgateLabel[gateLabel] for gateLabel in self.gates ]
        flattened_dprod = _np.concatenate( to_concat, axis=1 ) # axes = (vectorized_gate_el_index,gateset_parameter)

        if flat:
            return flattened_dprod
        else:
            vec_gs_size = flattened_dprod.shape[1]
            return _np.swapaxes( flattened_dprod, 0, 1 ).reshape( (vec_gs_size, dim, dim) ) # axes = (gate_ij, prod_row, prod_col)

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

        gatesToVectorize1 = self.gates.keys() #which differentiation w.r.t. gates should be done
                                              # (which is all the differentiation done here)
        gatesToVectorize2 = self.gates.keys() # (possibility to later specify different sets of gates
                                              #to differentiate firstly and secondly with)

        # LEXICOGRAPHICAL VS MATRIX ORDER
        revGateLabelList = tuple(reversed(tuple(gatestring))) # we do matrix multiplication in this order (easier to think about)

        #  prod = G1 * G2 * .... * GN , a matrix
        #  dprod/d(gateLabel)_ij   = sum_{L s.t. GL == gatelabel} [ G1 ... G(L-1) dG(L)/dij G(L+1) ... GN ] , a matrix for each given (i,j)
        #  d2prod/d(gateLabel1)_kl*d(gateLabel2)_ij = sum_{M s.t. GM == gatelabel1} sum_{L s.t. GL == gatelabel2, M < L} 
        #                                                 [ G1 ... G(M-1) dG(M)/dkl G(M+1) ... G(L-1) dG(L)/dij G(L+1) ... GN ] + {similar with L < M} (if L == M ignore)
        #                                                 a matrix for each given (i,j,k,l)
        #  vec( d2prod/d(gateLabel1)_kl*d(gateLabel2)_ij ) = sum{...} [ G1 ...  G(M-1) dG(M)/dkl G(M+1) ... G(L-1) tensor (G(L+1) ... GN)^T vec( dG(L)/dij ) ]
        #                                                  = sum{...} [ unvec( G1 ...  G(M-1) tensor (G(M+1) ... G(L-1))^T vec( dG(M)/dkl ) )
        #                                                                tensor (G(L+1) ... GN)^T vec( dG(L)/dij ) ]
        #                                                  + sum{ L < M} [ G1 ...  G(L-1) tensor
        #                                                       ( unvec( G(L+1) ... G(M-1) tensor (G(M+1) ... GN)^T vec( dG(M)/dkl ) ) )^T vec( dG(L)/dij ) ]
        #
        #  Note: ignoring L == M terms assumes that d^2 G/(dij)^2 == 0, which is true IF each gate matrix element is at most 
        #        *linear* in each of the gate parameters.  If this is not the case, need Gate objects to have a 2nd-deriv method in addition of deriv_wrt_params
        #
        #  Note: unvec( X ) can be done efficiently by actually computing X^T ( note (A tensor B)^T = A^T tensor B^T ) and using numpy's reshape

        dim = self.dim

        #Cache partial products
        prods = {}
        ident = _np.identity( dim )
        for (i,gateLabel1) in enumerate(revGateLabelList): #loop over "starting" gate
            prods[ (i,i-1) ] = ident #product of no gates
            G = ident
            for (j,gateLabel2) in enumerate(revGateLabelList[i:],start=i): #loop over "ending" gate (>= starting gate)
                G = _np.dot(G,self.gates[gateLabel2])
                prods[ (i,j) ] = G
        prods[ (len(revGateLabelList),len(revGateLabelList)-1) ] = ident #product of no gates

        # Initialize storage
        dgate_dgateLabel = {}; nParams = {}
        for gateLabel in set(gatesToVectorize1).union(gatesToVectorize2):
            dgate_dgateLabel[gateLabel] = self.gates[gateLabel].deriv_wrt_params(bG0) # (dim**2, nParams[gateLabel])
            nParams[gateLabel] = dgate_dgateLabel[gateLabel].shape[1]

        d2prod_dgateLabels = { }; 
        for gateLabel1 in gatesToVectorize1:
            for gateLabel2 in gatesToVectorize2:
                d2prod_dgateLabels[(gateLabel1,gateLabel2)] = _np.zeros( (dim**2, nParams[gateLabel1], nParams[gateLabel2]), 'd')

        #Add contributions for each gate in list
        N = len(revGateLabelList)
        for m,gateLabel1 in enumerate(revGateLabelList):
            #OLD shortcut: if gateLabel1 in gatesToVectorize1: (and indent below)
            for l,gateLabel2 in enumerate(revGateLabelList):
                #OLD shortcut: if gateLabel2 in gatesToVectorize2: (and indent below)
                # FUTURE: we could add logic that accounts for the symmetry of the Hessian, so that 
                # if gl1 and gl2 are both in gatesToVectorize1 and gatesToVectorize2 we only compute d2(prod)/d(gl1)d(gl2)
                # and not d2(prod)/d(gl2)d(gl1) ...
                if m < l:
                    x0 = _np.kron(_np.transpose(prods[(0,m-1)]),prods[(m+1,l-1)])  # (dim**2, dim**2)
                    x  = _np.dot( _np.transpose(dgate_dgateLabel[gateLabel1]), x0); xv = x.view() # (nParams[gateLabel1],dim**2)
                    xv.shape = (nParams[gateLabel1], dim, dim) # (reshape without copying - throws error if copy is needed)
                    y = _np.dot( _np.kron(xv, _np.transpose(prods[(l+1,N-1)])), dgate_dgateLabel[gateLabel2] ) 
                    # above: (nParams1,dim**2,dim**2) * (dim**2,nParams[gateLabel2]) = (nParams1,dim**2,nParams2)
                    d2prod_dgateLabels[(gateLabel1,gateLabel2)] += _np.swapaxes(y,0,1)
                            # above: dim = (dim2, nParams1, nParams2); swapaxes takes (kl,vec_prod_indx,ij) => (vec_prod_indx,kl,ij)
                elif l < m:
                    x0 = _np.kron(_np.transpose(prods[(l+1,m-1)]),prods[(m+1,N-1)]) # (dim**2, dim**2)
                    x  = _np.dot( _np.transpose(dgate_dgateLabel[gateLabel1]), x0); xv = x.view() # (nParams[gateLabel1],dim**2)
                    xv.shape = (nParams[gateLabel1], dim, dim) # (reshape without copying - throws error if copy is needed)
                    xv = _np.swapaxes(xv,1,2) # transposes each of the now un-vectorized dim x dim mxs corresponding to a single kl
                    y = _np.dot( _np.kron(prods[(0,l-1)], xv), dgate_dgateLabel[gateLabel2] )
# above: (nParams1,dim**2,dim**2) * (dim**2,nParams[gateLabel2]) = (nParams1,dim**2,nParams2)
                    d2prod_dgateLabels[(gateLabel1,gateLabel2)] += _np.swapaxes(y,0,1)
                    # above: dim = (dim2, nParams1, nParams2); swapaxes takes (kl,vec_prod_indx,ij) => (vec_prod_indx,kl,ij)
               #else l==m, in which case there's no contribution since we assume all gate elements are at most linear in the parameters


        #Concatenate per-gateLabel results to get final result
        to_concat = []
        for gateLabel1 in gatesToVectorize1:
            to_concat.append( _np.concatenate( [ d2prod_dgateLabels[(gateLabel1,gateLabel2)] for gateLabel2 in gatesToVectorize2 ], axis=2 ) ) #concat along ij (nParams2)
        flattened_d2prod = _np.concatenate( to_concat, axis=1 ) # concat along kl (nParams1)

        if flat:
            return flattened_d2prod # axes = (vectorized_gate_el_index, gateset_parameter1, gateset_parameter2)
        else:
            vec_kl_size, vec_ij_size = flattened_d2prod.shape[1:3]
            return _np.rollaxis( flattened_d2prod, 0, 3 ).reshape( (vec_kl_size, vec_ij_size, dim, dim) )
            # axes = (gateset_parameter1, gateset_parameter2, gateset_element_row, gateset_element_col)


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

        if self._is_remainder_spamlabel(spamLabel):
            #then compute 1.0 - (all other spam label probabilities)
            otherSpamLabels = spamLabels.copy(); del otherSpamLabels[ otherSpamLabels.index(spamLabel) ]
            assert( not any([ self._is_remainder_spamlabel(sl) for sl in otherSpamLabels]) )
            return 1.0 - sum( [self.pr(gatestring, clipTo, bUseScaling) for sl in otherSpamLabels] )

        (rholabel,elabel) = self.spamLabels[spamLabel]
        rho = self.rhovecs[rholabel]
        E   = _np.conjugate(_np.transpose(self._get_evec(elabel)))
        
        if bUseScaling:
            old_err = _np.seterr(over='ignore')
            G,scale = self.product(gatestring, True)
            p = _np.dot(E, _np.dot(G, rho)) * scale # probability, with scaling applied (may generate overflow, but OK)

            #DEBUG: catch warnings to make sure correct (inf if value is large) evaluation occurs when there's a warning
            #bPrint = False
            #with _warnings.catch_warnings():
            #    _warnings.filterwarnings('error')
            #    try:
            #        test = _mt.trace( _np.dot(self.SPAMs[spamLabel],G) ) * scale
            #    except Warning: bPrint = True
            #if bPrint:  print 'Warning in Gateset.pr : scale=%g, trace=%g, p=%g' % (scale,_np.dot(self.SPAMs[spamLabel],G) ), p)
            _np.seterr(**old_err)

        else: #no scaling -- faster but susceptible to overflow
            G = self.product(gatestring, False)
            p = _np.dot(E, _np.dot(G, rho) )

        if _np.isnan(p): 
            if len(gatestring) < 10:
                strToPrint = str(gatestring)
            else: strToPrint = str(gatestring[0:10]) + " ... (len %d)" % len(gatestring)
            _warnings.warn("pr(%s) == nan" % strToPrint)
            #DEBUG: print "backtrace" of product leading up to nan

            #G = _np.identity( self.dim ); total_exp = 0.0
            #for i,lGate in enumerate(gateLabelList):
            #    G = _np.dot(G,self[lGate])  # product of gates, starting with G0
            #    nG = norm(G); G /= nG; total_exp += log(nG) # scale and keep track of exponent
            #
            #    p = _mt.trace( _np.dot(self.SPAMs[spamLabel],G) ) * exp(total_exp) # probability
            #    print "%d: p = %g, norm %g, exp %g\n%s" % (i,p,norm(G),total_exp,str(G))
            #    if _np.isnan(p): raise ValueError("STOP")

        if clipTo is not None: 
            return _np.clip(p,clipTo[0],clipTo[1])
        else: return p


    def dpr(self, spamLabel, gatestring, returnPr=False,clipTo=None):
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

        if self._is_remainder_spamlabel(spamLabel):
            #then compute Deriv[ 1.0 - (all other spam label probabilities) ]
            otherSpamLabels = self.get_spam_labels(); del otherSpamLabels[ otherSpamLabels.index(spamLabel) ]
            assert( not any([ self._is_remainder_spamlabel(sl) for sl in otherSpamLabels]) )
            otherResults = [self.dpr(sl, gatestring, returnPr, clipTo) for sl in otherSpamLabels]
            if returnPr: 
                return -1.0 * sum([dpr for dpr,p in otherResults]), 1.0 - sum([p for dpr,p in otherResults])
            else:
                return -1.0 * sum(otherResults)

        #  pr = Tr( |rho><E| * prod ) = sum E_k prod_kl rho_l
        #  dpr/d(gateLabel)_ij = sum E_k [dprod/d(gateLabel)_ij]_kl rho_l
        #  dpr/d(rho)_i = sum E_k prod_ki
        #  dpr/d(E)_i   = sum prod_il rho_l

        (rholabel,elabel) = self.spamLabels[spamLabel]
        rho = self.rhovecs[rholabel]
        E   = _np.conjugate(_np.transpose(self._get_evec(elabel)))

        #Derivs wrt Gates
        old_err = _np.seterr(over='ignore')
        prod,scale = self.product(gatestring, True)
        dprod_dGates = self.dproduct(gatestring); vec_gs_size = dprod_dGates.shape[0]
        dpr_dGates = _np.empty( (1, vec_gs_size) )
        for i in xrange(vec_gs_size):
            dpr_dGates[0,i] = float(_np.dot(E, _np.dot( dprod_dGates[i], rho)))
        
        if returnPr:
            p = _np.dot(E, _np.dot(prod, rho)) * scale  #may generate overflow, but OK
            if clipTo is not None:  p = _np.clip( p, clipTo[0], clipTo[1] )
            
        #Derivs wrt SPAM
        num_rho_params = [v.num_params() for v in self.rhovecs.values()]
        rho_offset = [ sum(num_rho_params[0:i]) for i in range(len(self.rhovecs)+1) ]
        rhoIndex = self.rhovecs.keys().index(rholabel)
        dpr_drhos = _np.zeros( (1, sum(num_rho_params)) )
        derivWrtAnyRhovec = scale * _np.dot(E,prod)
        dpr_drhos[0, rho_offset[rhoIndex]:rho_offset[rhoIndex+1]] = \
            _np.dot( derivWrtAnyRhovec, rho.deriv_wrt_params())  #may overflow, but OK

        num_e_params = [v.num_params() for v in self.evecs.values()]
        e_offset = [ sum(num_e_params[0:i]) for i in range(len(self.evecs)+1) ]
        dpr_dEs = _np.zeros( (1, sum(num_e_params)) ); 
        derivWrtAnyEvec = scale * _np.transpose(_np.dot(prod,rho)) # may overflow, but OK
           # (** doesn't depend on eIndex **) -- TODO: should also conjugate() here if complex?
        if elabel == self._remainderLabel:
            assert(self._remainderLabel not in self.evecs) # "remainder" should be a distint *special* label
            for ei,evec in enumerate(self.evecs.values()):  #compute Deriv w.r.t. [ 1 - sum_of_other_EVecs ]
                dpr_dEs[0, e_offset[ei]:e_offset[ei+1]] = \
                    -1.0 * _np.dot( derivWrtAnyEvec, evec.deriv_wrt_params() )
        else:
            eIndex = self.evecs.keys().index(elabel)
            dpr_dEs[0, e_offset[eIndex]:e_offset[eIndex+1]] = \
                _np.dot( derivWrtAnyEvec, self.evecs[elabel].deriv_wrt_params() )

        _np.seterr(**old_err)

        if returnPr:
              return _np.concatenate( (dpr_drhos,dpr_dEs,dpr_dGates), axis=1 ), p
        else: return _np.concatenate( (dpr_drhos,dpr_dEs,dpr_dGates), axis=1 )


    def hpr(self, spamLabel, gatestring, returnPr=False, returnDeriv=False,clipTo=None):
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

        if self._is_remainder_spamlabel(spamLabel):
            #then compute Hessian[ 1.0 - (all other spam label probabilities) ]
            otherSpamLabels = self.get_spam_labels(); del otherSpamLabels[ otherSpamLabels.index(spamLabel) ]
            assert( not any([ self._is_remainder_spamlabel(sl) for sl in otherSpamLabels]) )
            otherResults = [self.hpr(sl, gatestring, returnPr, returnDeriv, clipTo) for sl in otherSpamLabels]
            if returnDeriv: 
                if returnPr: return ( -1.0 * sum([hpr for hpr,dpr,p in otherResults]),
                                      -1.0 * sum([dpr for hpr,dpr,p in otherResults]), 
                                       1.0 - sum([p   for hpr,dpr,p in otherResults])   )
                else:        return ( -1.0 * sum([hpr for hpr,dpr in otherResults]),
                                      -1.0 * sum([dpr for hpr,dpr in otherResults])     )
            else:
                if returnPr: return ( -1.0 * sum([hpr for hpr,p in otherResults]),
                                       1.0 - sum([p   for hpr,p in otherResults])   )
                else:        return   -1.0 * sum(otherResults)


        #  pr = Tr( |rho><E| * prod ) = sum E_k prod_kl rho_l
        #  d2pr/d(gateLabel1)_mn d(gateLabel2)_ij = sum E_k [dprod/d(gateLabel1)_mn d(gateLabel2)_ij]_kl rho_l
        #  d2pr/d(rho)_i d(gateLabel)_mn = sum E_k [dprod/d(gateLabel)_mn]_ki     (and same for other diff order)
        #  d2pr/d(E)_i d(gateLabel)_mn   = sum [dprod/d(gateLabel)_mn]_il rho_l   (and same for other diff order)
        #  d2pr/d(E)_i d(rho)_j          = prod_ij                                (and same for other diff order)
        #  d2pr/d(E)_i d(E)_j            = 0
        #  d2pr/d(rho)_i d(rho)_j        = 0

        (rholabel,elabel) = self.spamLabels[spamLabel]
        rho = self.rhovecs[rholabel]
        E   = _np.conjugate(_np.transpose(self._get_evec(elabel)))

        d2prod_dGates = self.hproduct(gatestring)
        vec_gs_size = d2prod_dGates.shape[0]
        assert( d2prod_dGates.shape[0] == d2prod_dGates.shape[1] )

        d2pr_dGates2 = _np.empty( (1, vec_gs_size, vec_gs_size) )
        for i in xrange(vec_gs_size):
            for j in xrange(vec_gs_size):
                d2pr_dGates2[0,i,j] = float(_np.dot(E, _np.dot( d2prod_dGates[i,j], rho)))

        old_err = _np.seterr(over='ignore')

        prod,scale = self.product(gatestring, True)
        if returnPr:
            p = _np.dot(E, _np.dot(prod, rho)) * scale  #may generate overflow, but OK
            if clipTo is not None:  p = _np.clip( p, clipTo[0], clipTo[1] )

        dprod_dGates  = self.dproduct(gatestring)
        assert( dprod_dGates.shape[0] == vec_gs_size )
        if returnDeriv: # same as in dpr(...)
            dpr_dGates = _np.empty( (1, vec_gs_size) )
            for i in xrange(vec_gs_size):
                dpr_dGates[0,i] = float(_np.dot(E, _np.dot( dprod_dGates[i], rho)))


        #Derivs wrt SPAM
        num_rho_params = [v.num_params() for v in self.rhovecs.values()]
        num_e_params = [v.num_params() for v in self.evecs.values()]
        rho_offset = [ sum(num_rho_params[0:i]) for i in range(len(self.rhovecs)+1) ]
        e_offset = [ sum(num_e_params[0:i]) for i in range(len(self.evecs)+1) ]
        rhoIndex = self.rhovecs.keys().index(rholabel)

        if returnDeriv:  #same as in dpr(...)
            dpr_drhos = _np.zeros( (1, sum(num_rho_params)) )
            derivWrtAnyRhovec = scale * _np.dot(E,prod)
            dpr_drhos[0, rho_offset[rhoIndex]:rho_offset[rhoIndex+1]] = \
                _np.dot( derivWrtAnyRhovec, rho.deriv_wrt_params())  #may overflow, but OK

            dpr_dEs = _np.zeros( (1, sum(num_e_params)) ); 
            derivWrtAnyEvec = scale * _np.transpose(_np.dot(prod,rho)) # may overflow, but OK
            if elabel == self._remainderLabel:
                assert(self._remainderLabel not in self.evecs)
                for ei,evec in enumerate(self.evecs.values()):  #compute Deriv w.r.t. [ 1 - sum_of_other_EVecs ]
                    dpr_dEs[0, e_offset[ei]:e_offset[ei+1]] = \
                        -1.0 * _np.dot( derivWrtAnyEvec, evec.deriv_wrt_params() )
            else:
                eIndex = self.evecs.keys().index(elabel)
                dpr_dEs[0, e_offset[eIndex]:e_offset[eIndex+1]] = \
                    _np.dot( derivWrtAnyEvec, self.evecs[elabel].deriv_wrt_params() )
        
            dpr = _np.concatenate( (dpr_drhos,dpr_dEs,dpr_dGates), axis=1 )

        d2pr_drhos = _np.zeros( (1, vec_gs_size, sum(num_rho_params)) )
        d2pr_drhos[0, :, sum(num_rho_params[0:rhoIndex]):sum(num_rho_params[0:rhoIndex+1])] \
            = _np.dot( _np.dot(E,dprod_dGates), rho.deriv_wrt_params())[0] # (= [0,:,:])

        d2pr_dEs = _np.zeros( (1, vec_gs_size, sum(num_e_params)) )
        derivWrtAnyEvec = _np.squeeze(_np.dot(dprod_dGates,rho), axis=(2,))
        if elabel == self._remainderLabel:
            assert(self._remainderLabel not in self.evecs)
            for ei,evec in enumerate(self.evecs.values()): #similar to above, but now after a deriv w.r.t gates
                d2pr_dEs[0, :, e_offset[ei]:e_offset[ei+1]] = \
                    -1.0 * _np.dot( derivWrtAnyEvec, evec.deriv_wrt_params() )
        else:
            eIndex = self.evecs.keys().index(elabel)
            d2pr_dEs[0, :, e_offset[eIndex]:e_offset[eIndex+1]] = \
                _np.dot(derivWrtAnyEvec, self.evecs[elabel].deriv_wrt_params())
        
        d2pr_dErhos = _np.zeros( (1, sum(num_e_params), sum(num_rho_params)) )
        derivWrtAnyEvec = scale * _np.dot(prod, rho.deriv_wrt_params()) #may generate overflow, but OK

        if elabel == self._remainderLabel:
            for ei,evec in enumerate(self.evecs.values()): #similar to above, but now after also a deriv w.r.t rhos
                d2pr_dErhos[0, e_offset[ei]:e_offset[ei+1], rho_offset[rhoIndex]:rho_offset[rhoIndex+1]] = \
                    -1.0 * _np.dot( _np.transpose(evec.deriv_wrt_params()),derivWrtAnyEvec)
                # ET*P*rho -> drhoP -> ET*P*drho/drhoP = ((P*drho/drhoP)^T*E)^T -> dEp -> 
                # ((P*drho/drhoP)^T*dE/dEp)^T = dE/dEp^T*(P*drho/drhoP) = (d,eP)^T*(d,rhoP) = (eP,rhoP) OK!
        else:
            eIndex = self.evecs.keys().index(elabel)
            d2pr_dErhos[0, e_offset[eIndex]:e_offset[eIndex+1], 
                        rho_offset[rhoIndex]:rho_offset[rhoIndex+1]] = \
                        _np.dot( _np.transpose(self.evecs[elabel].deriv_wrt_params()),derivWrtAnyEvec)

        d2pr_d2rhos = _np.zeros( (1, sum(num_rho_params), sum(num_rho_params)) )
        d2pr_d2Es   = _np.zeros( (1, sum(num_e_params), sum(num_e_params)) )

        ret_row1 = _np.concatenate( ( d2pr_d2rhos, _np.transpose(d2pr_dErhos,(0,2,1)), _np.transpose(d2pr_drhos,(0,2,1)) ), axis=2) # wrt rho
        ret_row2 = _np.concatenate( ( d2pr_dErhos, d2pr_d2Es, _np.transpose(d2pr_dEs,(0,2,1)) ), axis=2 ) # wrt E
        ret_row3 = _np.concatenate( ( d2pr_drhos,d2pr_dEs,d2pr_dGates2), axis=2 ) #wrt gates
        ret = _np.concatenate( (ret_row1, ret_row2, ret_row3), axis=1 )

        _np.seterr(**old_err)

        if returnDeriv: 
            if returnPr: return ret, dpr, p
            else:        return ret, dpr
        else:
            if returnPr: return ret, p
            else:        return ret


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
        probs = { }
        if not self.assumeSumToOne:
            for spamLabel in self.spamLabels:
                probs[spamLabel] = self.pr(spamLabel, gatestring, clipTo)
        else:
            s = 0; lastLabel = None
            for spamLabel in self.spamLabels:
                if self._is_remainder_spamlabel(spamLabel):
                    assert(lastLabel is None) # ensure there is at most one "remainder" spam label
                    lastLabel = spamLabel; continue
                probs[spamLabel] = self.pr(spamLabel, gatestring, clipTo)
                s += probs[spamLabel]
            if lastLabel is not None: 
                probs[lastLabel] = 1.0 - s  #last spam label is computed so sum == 1
        return probs


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
        dprobs = { }
        if not self.assumeSumToOne:
            for spamLabel in self.spamLabels:
                dprobs[spamLabel] = self.dpr(spamLabel, gatestring, returnPr,clipTo)
        else:
            ds = None; s=0; lastLabel = None
            for spamLabel in self.spamLabels:
                if self._is_remainder_spamlabel(spamLabel):
                    assert(lastLabel is None) # ensure there is at most one dummy spam label
                    lastLabel = spamLabel; continue
                dprobs[spamLabel] = self.dpr(spamLabel, gatestring, returnPr,clipTo)
                if returnPr:
                    ds = dprobs[spamLabel][0] if ds is None else ds + dprobs[spamLabel][0]
                    s += dprobs[spamLabel][1]
                else:
                    ds = dprobs[spamLabel] if ds is None else ds + dprobs[spamLabel]
            if lastLabel is not None: 
                dprobs[lastLabel] = (-ds,1.0-s) if returnPr else -ds
        return dprobs



    def hprobs(self, gatestring, returnPr=False, returnDeriv=False, clipTo=None):
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
        hprobs = { }
        if not self.assumeSumToOne:
            for spamLabel in self.spamLabels:
                hprobs[spamLabel] = self.hpr(spamLabel, gatestring, returnPr,
                                             returnDeriv,clipTo)
        else:
            hs = None; ds=None; s=0; lastLabel = None
            for spamLabel in self.spamLabels:
                if self._is_remainder_spamlabel(spamLabel):
                    assert(lastLabel is None) # ensure there is at most one dummy spam label
                    lastLabel = spamLabel; continue
                hprobs[spamLabel] = self.hpr(spamLabel, gatestring, returnPr,
                                             returnDeriv,clipTo)
                if returnPr:
                    if returnDeriv:
                        hs = hprobs[spamLabel][0] if hs is None else hs + hprobs[spamLabel][0]
                        ds = hprobs[spamLabel][1] if ds is None else ds + hprobs[spamLabel][1]
                        s += hprobs[spamLabel][2]
                    else:
                        hs = hprobs[spamLabel][0] if hs is None else hs + hprobs[spamLabel][0]
                        s += hprobs[spamLabel][1]
                else:
                    if returnDeriv:
                        hs = hprobs[spamLabel][0] if hs is None else hs + hprobs[spamLabel][0]
                        ds = hprobs[spamLabel][1] if ds is None else ds + hprobs[spamLabel][1]
                    else:
                        hs = hprobs[spamLabel] if hs is None else hs + hprobs[spamLabel]

            if lastLabel is not None: 
                if returnPr:
                    hprobs[lastLabel] = (-hs,-ds,1.0-s) if returnDeriv else (-hs,1.0-s)
                else:
                    hprobs[lastLabel] = (-hs,-ds) if returnDeriv else -hs
                    
        return hprobs


#    def bulk_evaltree_beta(self, gatestring_list):
#        """
#          Returns an evaluation tree for all the gate 
#          strings in gatestring_list. Used by bulk_pr and
#          bulk_dpr, this is it's own function so that 
#          if many calls to bulk_pr and/or bulk_dpr are
#          made with the same gatestring_list, only a single
#          call to bulk_evaltree is needed.
#        """
#        evalTree = _evaltree.EvalTree()
#        evalTree.initialize_beta([""] + self.gates.keys(), gatestring_list)
#        return evalTree


#Keep this in GateSet for now since it's so simple
#    def bulk_evaltree(self, gatestring_list):
#        """
#        Create an evaluation tree for all the gate strings in gatestring_list.
#
#        This tree can be used by other Bulk_* functions, and is it's own 
#        function so that for many calls to Bulk_* made with the same 
#        gatestring_list, only a single call to bulk_evaltree is needed.
#        
#        Parameters
#        ----------
#        gatestring_list : list of (tuples or GateStrings)
#            Each element specifies a gate string to include in the evaluation tree.
#
#        Returns
#        -------
#        EvalTree
#            An evaluation tree object.
#        """
#        evalTree = _evaltree.EvalTree()
#        evalTree.initialize([""] + self.gates.keys(), gatestring_list)
#        return evalTree


    def bulk_product(self, evalTree, bScale=False):
        """
        Compute the products of many gate strings at once.

        Parameters
        ----------
        evalTree : EvalTree
           given by a prior call to bulk_evaltree.  Specifies the gate strings
           to compute the bulk operation on.

        bScale : bool, optional
           When True, return a scaling factor (see below).
              
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

        dim = self.dim
        assert(not evalTree.is_split()) #product functions can't use split trees (as there's really no point)

        cacheSize = len(evalTree)
        prodCache = _np.zeros( (cacheSize, dim, dim) )
        scaleCache = _np.zeros( cacheSize, 'd' )

        #First element of cache are given by evalTree's initial single- or zero-gate labels
        for i,gateLabel in enumerate(evalTree.get_init_labels()):
            if gateLabel == "": #special case of empty label == no gate
                prodCache[i] = _np.identity( dim )
            else:
                gate = self.gates[gateLabel].asarray()
                nG = max(_nla.norm(gate), 1.0)
                prodCache[i] = gate / nG
                scaleCache[i] = _np.log(nG)

        nZeroAndSingleStrs = len(evalTree.get_init_labels())

        #evaluate gate strings using tree (skip over the zero and single-gate-strings)
        #cnt = 0
        for (i,tup) in enumerate(evalTree[nZeroAndSingleStrs:],start=nZeroAndSingleStrs):

            # combine iLeft + iRight => i
            # LEXICOGRAPHICAL VS MATRIX ORDER Note: we reverse iLeft <=> iRight from evalTree because
            # (iRight,iLeft,iFinal) = tup implies gatestring[i] = gatestring[iLeft] + gatestring[iRight], but we want:
            (iRight,iLeft,iFinal) = tup   # since then matrixOf(gatestring[i]) = matrixOf(gatestring[iLeft]) * matrixOf(gatestring[iRight])
            L,R = prodCache[iLeft], prodCache[iRight]
            prodCache[i] = _np.dot(L,R)
            scaleCache[i] = scaleCache[iLeft] + scaleCache[iRight]

            if prodCache[i].max() < PSMALL and prodCache[i].min() > -PSMALL:
                nL,nR = max(_nla.norm(L), _np.exp(-scaleCache[iLeft]),1e-300), max(_nla.norm(R), _np.exp(-scaleCache[iRight]),1e-300)
                sL, sR = L/nL, R/nR
                prodCache[i] = _np.dot(sL,sR); scaleCache[i] += _np.log(nL) + _np.log(nR)
               
        #print "bulk_product DEBUG: %d rescalings out of %d products" % (cnt, len(evalTree)) 

        nanOrInfCacheIndices = (~_np.isfinite(prodCache)).nonzero()[0]  #may be duplicates (a list, not a set)
        assert( len(nanOrInfCacheIndices) == 0 ) # since all scaled gates start with norm <= 1, products should all have norm <= 1

        #use cached data to construct return values
        finalIndxList = evalTree.get_list_of_final_value_tree_indices()
        Gs = prodCache.take(  finalIndxList, axis=0 ) #shape == ( len(gatestring_list), dim, dim ), Gs[i] is product for i-th gate string
        scaleExps = scaleCache.take( finalIndxList )

        old_err = _np.seterr(over='ignore')
        scaleVals = _np.exp(scaleExps) #may overflow, but OK if infs occur here
        _np.seterr(**old_err)

        if bScale:
            return Gs, scaleVals
        else:
            old_err = _np.seterr(over='ignore')
            Gs = _np.swapaxes( _np.swapaxes(Gs,0,2) * scaleVals, 0,2)  #may overflow, but ok
            _np.seterr(**old_err)
            return Gs


    def bulk_dproduct(self, evalTree, flat=False, bReturnProds=False, bScale=False, memLimit=None):
        """
        Compute the derivative of a many gate strings at once.

        Parameters
        ----------
        evalTree : EvalTree
           given by a prior call to bulk_evaltree.  Specifies the gate strings
           to compute the bulk operation on.

        flat : bool, optional
          Affects the shape of the returned derivative array (see below).

        bReturnProds : bool, optional
          when set to True, additionally return the probabilities.

        bScale : bool, optional
          When True, return a scaling factor (see below).

        memLimit : int, optional
          A rough memory limit in bytes which restricts the amount of
          intermediate values that are computed and stored.
        
           
        Returns
        -------
        derivs : numpy array
          
          * if flat == False, an array of shape S x M x G x G, where:

            - S == len(gatestring_list)
            - M == the length of the vectorized gateset
            - G == the linear dimension of a gate matrix (G x G gate matrices)
            
            and derivs[i,j,k,l] holds the derivative of the (k,l)-th entry
            of the i-th gate string product with respect to the j-th gateset
            parameter.

          * if flat == True, an array of shape S*N x M where:

            - N == the number of entries in a single flattened gate (ordering same as numpy.flatten),
            - S,M == as above,
              
            and deriv[i,j] holds the derivative of the (i % G^2)-th entry of
            the (i / G^2)-th flattened gate string product  with respect to 
            the j-th gateset parameter.

        products : numpy array
          Only returned when bReturnProds == True.  An array of shape  
          S x G x G; products[i] is the i-th gate string product.

        scaleVals : numpy array
          Only returned when bScale == True.  An array of shape S such that
          scaleVals[i] contains the multiplicative scaling needed for
          the derivatives and/or products for the i-th gate string.
        """

        dim = self.dim
        assert(not evalTree.is_split()) #product functions can't use split trees (as there's really no point)

        nGateStrings = evalTree.num_final_strings() #len(gatestring_list)
        nGateDerivCols = sum([g.num_params() for g in self.gates.values()])
        deriv_shape = (nGateDerivCols, dim, dim)
        cacheSize = len(evalTree)

        ##DEBUG
        #nc = cacheSize; gd = dim; nd = nGateDerivCols; C = 8.0/1024.0**3
        #print "Memory estimate for bulk_dproduct: %d eval tree size, %d gate dim, %d gateset params" % (nc,gd,nd)
        #print "    ==> %g GB (p) + %g GB (dp) + %g GB (scale) = %g GB (total)" % \
        #    (nc*gd*gd*C, nc*nd*gd*gd*C,nc*C, (nc*gd*gd + nc*nd*gd*gd + nc)*C)

        memEstimate = 8*cacheSize*(1 + dim**2 * (1 + nGateDerivCols)) # in bytes (8* = 64-bit floats)
        if memLimit is not None and memEstimate > memLimit:
            C = 1.0/(1024.0**3) #conversion bytes => GB (memLimit assumed to be in bytes)
            raise MemoryError("Memory estimate of %dGB  exceeds limit of %dGB" % (memEstimate*C,memLimit*C))    

        prodCache = _np.zeros( (cacheSize, dim, dim) )
        dProdCache = _np.zeros( (cacheSize,) + deriv_shape )
        scaleCache = _np.zeros( cacheSize, 'd' )



        #print "DEBUG: cacheSize = ",cacheSize, " gate dim = ",dim, " deriv_shape = ",deriv_shape
        #print "  pc MEM estimate = ", cacheSize*dim*dim*8.0/(1024.0**2), "MB"
        #print "  dpc MEM estimate = ", cacheSize*_np.prod(deriv_shape)*8.0/(1024.0**2), "MB"
        #print "  sc MEM estimate = ", cacheSize*8.0/(1024.0**2), "MB"
        #import time
        #time.sleep(10)
        #print "Continuing..."        

        # This iteration **must** match that in bulk_evaltree
        #   in order to associate the right single-gate-strings w/indices
        for i,gateLabel in enumerate(evalTree.get_init_labels()):
            if gateLabel == "": #special case of empty label == no gate
                prodCache[i] = _np.identity( dim ); dProdCache[0] = _np.zeros( deriv_shape )
            else:
                dgate = self.dproduct( (gateLabel,) )
                nG = max(_nla.norm(self.gates[gateLabel]),1.0)
                prodCache[i]  = self.gates[gateLabel].asarray() / nG
                scaleCache[i] = _np.log(nG)
                dProdCache[i] = dgate / nG 
                
        nZeroAndSingleStrs = len(evalTree.get_init_labels())

        #evaluate gate strings using tree (skip over the zero and single-gate-strings)
        for (i,tup) in enumerate(evalTree[nZeroAndSingleStrs:],start=nZeroAndSingleStrs):

            # combine iLeft + iRight => i
            # LEXICOGRAPHICAL VS MATRIX ORDER Note: we reverse iLeft <=> iRight from evalTree because
            # (iRight,iLeft,iFinal) = tup implies gatestring[i] = gatestring[iLeft] + gatestring[iRight], but we want:
            (iRight,iLeft,iFinal) = tup   # since then matrixOf(gatestring[i]) = matrixOf(gatestring[iLeft]) * matrixOf(gatestring[iRight])
            L,R = prodCache[iLeft], prodCache[iRight]
            prodCache[i] = _np.dot(L,R)

            #if not prodCache[i].any(): #same as norm(prodCache[i]) == 0 but faster
            if prodCache[i].max() < PSMALL and prodCache[i].min() > -PSMALL:
                nL,nR = max(_nla.norm(L), _np.exp(-scaleCache[iLeft]),1e-300), max(_nla.norm(R), _np.exp(-scaleCache[iRight]),1e-300)
                sL, sR, sdL, sdR = L/nL, R/nR, dProdCache[iLeft]/nL, dProdCache[iRight]/nR
                prodCache[i] = _np.dot(sL,sR); dProdCache[i] = _np.dot(sdL, sR) + _np.swapaxes(_np.dot(sL, sdR),0,1)
                scaleCache[i] = scaleCache[iLeft] + scaleCache[iRight] + _np.log(nL) + _np.log(nR)
                if dProdCache[i].max() < DSMALL and dProdCache[i].min() > -DSMALL:
                    _warnings.warn("Scaled dProd small in order to keep prod managable.")
            else:
                dL,dR = dProdCache[iLeft], dProdCache[iRight]
                dProdCache[i] = _np.dot(dL, R) + _np.swapaxes(_np.dot(L, dR),0,1) #dot(dS, T) + dot(S, dT)
                scaleCache[i] = scaleCache[iLeft] + scaleCache[iRight]
                
                if dProdCache[i].max() < DSMALL and dProdCache[i].min() > -DSMALL:
                    nL,nR = max(_nla.norm(dL), _np.exp(-scaleCache[iLeft]),1e-300), max(_nla.norm(dR), _np.exp(-scaleCache[iRight]),1e-300)
                    sL, sR, sdL, sdR = L/nL, R/nR, dL/nL, dR/nR
                    prodCache[i] = _np.dot(sL,sR); dProdCache[i] = _np.dot(sdL, sR) + _np.swapaxes(_np.dot(sL, sdR),0,1)
                    scaleCache[i] = scaleCache[iLeft] + scaleCache[iRight] + _np.log(nL) + _np.log(nR)
                    if prodCache[i].max() < PSMALL and prodCache[i].min() > -PSMALL:
                        _warnings.warn("Scaled prod small in order to keep dProd managable.")
                
        nanOrInfCacheIndices = (~_np.isfinite(prodCache)).nonzero()[0] 
        assert( len(nanOrInfCacheIndices) == 0 ) # since all scaled gates start with norm <= 1, products should all have norm <= 1
        
#        #Possibly re-evaluate tree using slower method if there nan's or infs using the fast method
#        if len(nanOrInfCacheIndices) > 0:
#            iBeginScaled = min( evalTree[ min(nanOrInfCacheIndices) ][0:2] ) # first index in tree that *resulted* in a nan or inf
#            _warnings.warn("Nans and/or Infs triggered re-evaluation at indx %d of %d products" % (iBeginScaled,len(evalTree)))
#            for (i,tup) in enumerate(evalTree[iBeginScaled:],start=iBeginScaled):
#                (iLeft,iRight,iFinal) = tup
#                L,R = prodCache[iLeft], prodCache[iRight],
#                G = dot(L,R); nG = norm(G)
#                prodCache[i] = G / nG
#                dProdCache[i] = dot(dProdCache[iLeft], R) + swapaxes(dot(L, dProdCache[iRight]),0,1) / nG
#                scaleCache[i] = scaleCache[iLeft] + scaleCache[iRight] + log(nG)

        #use cached data to construct return values

        finalIndxList = evalTree.get_list_of_final_value_tree_indices()

        old_err = _np.seterr(over='ignore')
        scaleExps = scaleCache.take( finalIndxList )
        scaleVals = _np.exp(scaleExps) #may overflow, but OK if infs occur here
        _np.seterr(**old_err)

        if bReturnProds:
            Gs  = prodCache.take(  finalIndxList, axis=0 ) #shape == ( len(gatestring_list), dim, dim ), Gs[i] is product for i-th gate string
            dGs = dProdCache.take( finalIndxList, axis=0 ) #shape == ( len(gatestring_list), nGateDerivCols, dim, dim ), dGs[i] is dprod_dGates for ith string

            if not bScale:
                old_err = _np.seterr(over='ignore', invalid='ignore')
                Gs  = _np.swapaxes( _np.swapaxes(Gs,0,2) * scaleVals, 0,2)  #may overflow, but ok
                dGs = _np.swapaxes( _np.swapaxes(dGs,0,3) * scaleVals, 0,3) #may overflow or get nans (invalid), but ok
                dGs[_np.isnan(dGs)] = 0  #convert nans to zero, as these occur b/c an inf scaleVal is mult by a zero deriv value (see below)
                _np.seterr(**old_err)

            if flat: dGs =  _np.swapaxes( _np.swapaxes(dGs,0,1).reshape( (nGateDerivCols, nGateStrings*dim**2) ), 0,1 ) # cols = deriv cols, rows = flattened everything else
            return (dGs, Gs, scaleVals) if bScale else (dGs, Gs)

        else:
            dGs = dProdCache.take( finalIndxList, axis=0 ) #shape == ( len(gatestring_list), nGateDerivCols, dim, dim ), dGs[i] is dprod_dGates for ith string

            if not bScale:
                old_err = _np.seterr(over='ignore', invalid='ignore')
                dGs = _np.swapaxes( _np.swapaxes(dGs,0,3) * scaleVals, 0,3) #may overflow or get nans (invalid), but ok
                dGs[_np.isnan(dGs)] =  0 #convert nans to zero, as these occur b/c an inf scaleVal is mult by a zero deriv value, and we 
                                        # assume the zero deriv value trumps since we've renormed to keep all the products within decent bounds
                #assert( len( (_np.isnan(dGs)).nonzero()[0] ) == 0 ) 
                #assert( len( (_np.isinf(dGs)).nonzero()[0] ) == 0 ) 
                #dGs = clip(dGs,-1e300,1e300)
                _np.seterr(**old_err)

            if flat: dGs =  _np.swapaxes( _np.swapaxes(dGs,0,1).reshape( (nGateDerivCols, nGateStrings*dim**2) ), 0,1 ) # cols = deriv cols, rows = flattened everything else
            return (dGs, scaleVals) if bScale else dGs


    def bulk_hproduct(self, evalTree, flat=False, bReturnDProdsAndProds=False, bScale=False):
        
        """
        Return the Hessian of a many gate strings at once.

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

        dim = self.dim
        assert(not evalTree.is_split()) #product functions can't use split trees (as there's really no point)

        nGateStrings = evalTree.num_final_strings() #len(gatestring_list)
        nGateDerivCols = sum([g.num_params() for g in self.gates.values()])
        deriv_shape = (nGateDerivCols, dim, dim)
        hessn_shape = (nGateDerivCols, nGateDerivCols, dim, dim)

        cacheSize = len(evalTree)
        prodCache = _np.zeros( (cacheSize, dim, dim) )
        dProdCache = _np.zeros( (cacheSize,) + deriv_shape )
        hProdCache = _np.zeros( (cacheSize,) + hessn_shape )
        scaleCache = _np.zeros( cacheSize, 'd' )

        #print "DEBUG: cacheSize = ",cacheSize, " gate dim = ",dim, " deriv_shape = ",deriv_shape," hessn_shape = ",hessn_shape
        #print "  pc MEM estimate = ", cacheSize*dim*dim*8.0/(1024.0**2), "MB"
        #print "  dpc MEM estimate = ", cacheSize*_np.prod(deriv_shape)*8.0/(1024.0**2), "MB"
        #print "  hpc MEM estimate = ", cacheSize*_np.prod(hessn_shape)*8.0/(1024.0**2), "MB"
        #print "  sc MEM estimate = ", cacheSize*8.0/(1024.0**2), "MB"
        #import time
        #time.sleep(10)
        #print "Continuing..."        

        #First element of cache are given by evalTree's initial single- or zero-gate labels
        for i,gateLabel in enumerate(evalTree.get_init_labels()):
            if gateLabel == "": #special case of empty label == no gate
                prodCache[i]  = _np.identity( dim )
                dProdCache[i] = _np.zeros( deriv_shape )
                hProdCache[i] = _np.zeros( hessn_shape )
            else:
                hgate = self.hproduct( (gateLabel,) )
                dgate = self.dproduct( (gateLabel,) )
                nG = max(_nla.norm(self.gates[gateLabel]),1.0)
                prodCache[i]  = self.gates[gateLabel].asarray() / nG
                scaleCache[i] = _np.log(nG)
                dProdCache[i] = dgate / nG 
                hProdCache[i] = hgate / nG 
            
        nZeroAndSingleStrs = len(evalTree.get_init_labels())

        #evaluate gate strings using tree (skip over the zero and single-gate-strings)
        for (i,tup) in enumerate(evalTree[nZeroAndSingleStrs:],start=nZeroAndSingleStrs):

            # combine iLeft + iRight => i
            # LEXICOGRAPHICAL VS MATRIX ORDER Note: we reverse iLeft <=> iRight from evalTree because
            # (iRight,iLeft,iFinal) = tup implies gatestring[i] = gatestring[iLeft] + gatestring[iRight], but we want:
            (iRight,iLeft,iFinal) = tup   # since then matrixOf(gatestring[i]) = matrixOf(gatestring[iLeft]) * matrixOf(gatestring[iRight])
            L,R = prodCache[iLeft], prodCache[iRight]
            prodCache[i] = _np.dot(L,R)

            if prodCache[i].max() < PSMALL and prodCache[i].min() > -PSMALL:
                nL,nR = max(_nla.norm(L), _np.exp(-scaleCache[iLeft]),1e-300), max(_nla.norm(R), _np.exp(-scaleCache[iRight]),1e-300)
                sL, sR, sdL, sdR = L/nL, R/nR, dProdCache[iLeft]/nL, dProdCache[iRight]/nR
                shL, shR, sdLdR = hProdCache[iLeft]/nL, hProdCache[iRight]/nR, _np.swapaxes(_np.dot(sdL,sdR),1,2) #_np.einsum('ikm,jml->ijkl',sdL,sdR)
                prodCache[i] = _np.dot(sL,sR); dProdCache[i] = _np.dot(sdL, sR) + _np.swapaxes(_np.dot(sL, sdR),0,1)
                hProdCache[i] = _np.dot(shL, sR) + sdLdR + _np.swapaxes(sdLdR,0,1) + _np.swapaxes(_np.dot(sL,shR),0,2)
                scaleCache[i] = scaleCache[iLeft] + scaleCache[iRight] + _np.log(nL) + _np.log(nR)
                if dProdCache[i].max() < DSMALL and dProdCache[i].min() > -DSMALL:
                    _warnings.warn("Scaled dProd small in order to keep prod managable.")
                if hProdCache[i].max() < HSMALL and hProdCache[i].min() > -HSMALL:
                    _warnings.warn("Scaled hProd small in order to keep prod managable.")
            else:
                dL,dR = dProdCache[iLeft], dProdCache[iRight]
                dProdCache[i] = _np.dot(dL, R) + _np.swapaxes(_np.dot(L, dR),0,1) #dot(dS, T) + dot(S, dT)
                scaleCache[i] = scaleCache[iLeft] + scaleCache[iRight]

                hL,hR = hProdCache[iLeft], hProdCache[iRight]   
                dLdR = _np.swapaxes(_np.dot(dL,dR),1,2) #_np.einsum('ikm,jml->ijkl',dL,dR) # Note: L, R = GxG ; dL,dR = vgs x GxG ; hL,hR = vgs x vgs x GxG
                hProdCache[i] = _np.dot(hL, R) + dLdR + _np.swapaxes(dLdR,0,1) + _np.swapaxes(_np.dot(L,hR),0,2)

                if dProdCache[i].max() < DSMALL and dProdCache[i].min() > -DSMALL:
                    nL,nR = max(_nla.norm(dL), _np.exp(-scaleCache[iLeft]),1e-300), max(_nla.norm(dR), _np.exp(-scaleCache[iRight]),1e-300)
                    sL, sR, sdL, sdR = L/nL, R/nR, dL/nL, dR/nR
                    shL, shR, sdLdR = hL/nL, hR/nR, _np.swapaxes(_np.dot(sdL,sdR),1,2) #_np.einsum('ikm,jml->ijkl',sdL,sdR)
                    prodCache[i] = _np.dot(sL,sR); dProdCache[i] = _np.dot(sdL, sR) + _np.swapaxes(_np.dot(sL, sdR),0,1)
                    hProdCache[i] = _np.dot(shL, sR) + sdLdR + _np.swapaxes(sdLdR,0,1) + _np.swapaxes(_np.dot(sL,shR),0,2)
                    scaleCache[i] = scaleCache[iLeft] + scaleCache[iRight] + _np.log(nL) + _np.log(nR)
                    if prodCache[i].max() < PSMALL and prodCache[i].min() > -PSMALL:
                        _warnings.warn("Scaled prod small in order to keep dProd managable.")
                    if hProdCache[i].max() < HSMALL and hProdCache[i].min() > -HSMALL:
                        _warnings.warn("Scaled hProd small in order to keep dProd managable.")
                
        nanOrInfCacheIndices = (~_np.isfinite(prodCache)).nonzero()[0] 
        assert( len(nanOrInfCacheIndices) == 0 ) # since all scaled gates start with norm <= 1, products should all have norm <= 1
        


        #use cached data to construct return values
        finalIndxList = evalTree.get_list_of_final_value_tree_indices()
        old_err = _np.seterr(over='ignore')
        scaleExps = scaleCache.take( finalIndxList )
        scaleVals = _np.exp(scaleExps) #may overflow, but OK if infs occur here
        _np.seterr(**old_err)

        if bReturnDProdsAndProds:
            Gs  = prodCache.take(  finalIndxList, axis=0 ) #shape == ( len(gatestring_list), dim, dim ), Gs[i] is product for i-th gate string
            dGs = dProdCache.take( finalIndxList, axis=0 ) #shape == ( len(gatestring_list), nGateDerivCols, dim, dim ), dGs[i] is dprod_dGates for ith string
            hGs = hProdCache.take( finalIndxList, axis=0 ) #shape == ( len(gatestring_list), nGateDerivCols, nGateDerivCols, dim, dim ), hGs[i] 
                                                           # is hprod_dGates for ith string

            if not bScale:
                old_err = _np.seterr(over='ignore', invalid='ignore')
                Gs  = _np.swapaxes( _np.swapaxes(Gs,0,2) * scaleVals, 0,2)  #may overflow, but ok
                dGs = _np.swapaxes( _np.swapaxes(dGs,0,3) * scaleVals, 0,3) #may overflow or get nans (invalid), but ok
                hGs = _np.swapaxes( _np.swapaxes(hGs,0,4) * scaleVals, 0,4) #may overflow or get nans (invalid), but ok
                dGs[_np.isnan(dGs)] = 0  #convert nans to zero, as these occur b/c an inf scaleVal is mult by a zero deriv value (see below)
                hGs[_np.isnan(hGs)] = 0  #convert nans to zero, as these occur b/c an inf scaleVal is mult by a zero hessian value (see below)
                _np.seterr(**old_err)

            if flat: 
                dGs = _np.swapaxes( _np.swapaxes(dGs,0,1).reshape( (nGateDerivCols, nGateStrings*dim**2) ), 0,1 ) # cols = deriv cols, rows = flattened all else
                hGs = _np.rollaxis( _np.rollaxis(hGs,0,3).reshape( (nGateDerivCols, nGateDerivCols, nGateStrings*dim**2) ), 2) # cols = deriv cols, rows = all else
                
            return (hGs, dGs, Gs, scaleVals) if bScale else (hGs, dGs, Gs)

        else:
            hGs = hProdCache.take( finalIndxList, axis=0 ) #shape == ( len(gatestring_list), nGateDerivCols, nGateDerivCols, dim, dim )

            if not bScale:
                old_err = _np.seterr(over='ignore', invalid='ignore')
                hGs = _np.swapaxes( _np.swapaxes(hGs,0,4) * scaleVals, 0,4) #may overflow or get nans (invalid), but ok
                hGs[_np.isnan(hGs)] =  0 #convert nans to zero, as these occur b/c an inf scaleVal is mult by a zero hessian value, and we 
                                         # assume the zero hessian value trumps since we've renormed to keep all the products within decent bounds
                #assert( len( (_np.isnan(hGs)).nonzero()[0] ) == 0 ) 
                #assert( len( (_np.isinf(hGs)).nonzero()[0] ) == 0 ) 
                #hGs = clip(hGs,-1e300,1e300)
                _np.seterr(**old_err)

            if flat: hGs = _np.rollaxis( _np.rollaxis(hGs,0,3).reshape( (nGateDerivCols, nGateDerivCols, nGateStrings*dim**2) ), 2) # as above

            return (hGs, scaleVals) if bScale else hGs

    

    def bulk_pr(self, spamLabel, evalTree, clipTo=None, check=False):
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
           
        Returns
        -------
        numpy array
          An array of length equal to the number of gate strings containing
          the (float) probabilities.
        """

        nGateStrings = evalTree.num_final_strings() #len(gatestring_list)
        if evalTree.is_split():
            vp = _np.empty( nGateStrings, 'd' )

        (rholabel,elabel) = self.spamLabels[spamLabel]
        rho = self.rhovecs[rholabel]
        E   = _np.conjugate(_np.transpose(self._get_evec(elabel)))

        for evalSubTree in evalTree.get_sub_trees():
            Gs, scaleVals = self.bulk_product(evalSubTree, bScale=True)

            #Compute probability and save in return array
            # want vp[iFinal] = float(dot(E, dot(G, rho)))  ##OLD, slightly slower version: p = trace(dot(self.SPAMs[spamLabel], G))
            #  vp[i] = sum_k,l E[0,k] Gs[i,k,l] rho[l,0] * scaleVals[i]
            #  vp[i] = sum_k E[0,k] dot(Gs, rho)[i,k,0]  * scaleVals[i]
            #  vp[i] = dot( E, dot(Gs, rho))[0,i,0]      * scaleVals[i]
            #  vp    = squeeze( dot( E, dot(Gs, rho)), axis=(0,2) ) * scaleVals
            old_err = _np.seterr(over='ignore')
            sub_vp = _np.squeeze( _np.dot(E, _np.dot(Gs, rho)), axis=(0,2) ) * scaleVals  # shape == (len(gatestring_list),) ; may overflow but OK
            _np.seterr(**old_err)
        
            if evalTree.is_split():
                vp[ evalSubTree.myFinalToParentFinalMap ] = sub_vp
            else: vp = sub_vp

        #DEBUG: catch warnings to make sure correct (inf if value is large) evaluation occurs when there's a warning
        #bPrint = False
        #with _warnings.catch_warnings():
        #    _warnings.filterwarnings('error')
        #    try:
        #        vp = squeeze( dot(E, dot(Gs, rho)), axis=(0,2) ) * scaleVals
        #    except Warning: bPrint = True
        #if bPrint:  print 'Warning in Gateset.bulk_pr : scaleVals=',scaleVals,'\n vp=',vp
            
        if clipTo is not None:  
            vp = _np.clip( vp, clipTo[0], clipTo[1])
            #nClipped = len((_np.logical_or(vp < clipTo[0], vp > clipTo[1])).nonzero()[0])
            #if nClipped > 0: print "DEBUG: bulk_pr nClipped = ",nClipped

        if check: 
            # compare with older slower version that should do the same thing (for debugging)
            gatestring_list = evalTree.generate_gatestring_list()
            check_vp = _np.array( [ self.pr(spamLabel, gateString, clipTo) for gateString in gatestring_list ] )
            if _nla.norm(vp - check_vp) > 1e-6:
                _warnings.warn( "norm(vp-check_vp) = %g - %g = %g" % (_nla.norm(vp), _nla.norm(check_vp), _nla.norm(vp - check_vp)) )
                #for i,gs in enumerate(gatestring_list):
                #    if abs(vp[i] - check_vp[i]) > 1e-6: 
                #        check = self.pr(spamLabel, gs, clipTo, bDebug=True)
                #        print "Check = ",check
                #        print "Bulk scaled gates:"
                #        print " prodcache = \n",prodCache[i] 
                #        print " scaleCache = ",scaleCache[i]
                #        print " trace = ", squeeze( dot(E, dot(Gs, rho)), axis=(0,2) )[i]
                #        print " scaleVals = ",scaleVals
                #        #for k in range(1+len(self)):
                #        print "   %s => p=%g, check_p=%g, diff=%g" % (str(gs),vp[i],check_vp[i],abs(vp[i]-check_vp[i]))
                #        raise ValueError("STOP")

        return vp


    def bulk_dpr(self, spamLabel, evalTree, 
                 returnPr=False,clipTo=None,check=False,memLimit=None):

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

        if self._is_remainder_spamlabel(spamLabel):
            #then compute Deriv[ 1.0 - (all other spam label probabilities) ]
            otherSpamLabels = self.get_spam_labels(); del otherSpamLabels[ otherSpamLabels.index(spamLabel) ]
            assert( not any([ self._is_remainder_spamlabel(sl) for sl in otherSpamLabels]) )
            otherResults = [self.bulk_dpr(sl, evalTree, returnPr, clipTo) for sl in otherSpamLabels]
            if returnPr:
                return ( -1.0 * _np.sum([dpr for dpr,p in otherResults],axis=0),
                          1.0 - _np.sum([p   for dpr,p in otherResults],axis=0) )
            else:
                return -1.0 * _np.sum(otherResults, axis=0)

        (rholabel,elabel) = self.spamLabels[spamLabel]
        rho = self.rhovecs[rholabel]
        E   = _np.conjugate(_np.transpose(self._get_evec(elabel)))

        nGateStrings = evalTree.num_final_strings()
        num_rho_params = [v.num_params() for v in self.rhovecs.values()]
        num_e_params = [v.num_params() for v in self.evecs.values()]
        rho_offset = [ sum(num_rho_params[0:i]) for i in range(len(self.rhovecs)+1) ]
        e_offset = [ sum(num_e_params[0:i]) for i in range(len(self.evecs)+1) ]
        nDerivCols = sum(num_rho_params + num_e_params + [g.num_params() for g in self.gates.values()])

        if evalTree.is_split():
            vp = _np.empty( nGateStrings, 'd' )
            vdp = _np.empty( (nGateStrings, nDerivCols), 'd' )  

        for evalSubTree in evalTree.get_sub_trees():
            sub_nGateStrings = evalSubTree.num_final_strings()
            dGs, Gs, scaleVals = self.bulk_dproduct(evalSubTree, bScale=True,
                                                    bReturnProds=True, memLimit=memLimit)

            old_err = _np.seterr(over='ignore')
    
            #Compute probability and save in return array
            # want vp[iFinal] = float(dot(E, dot(G, rho)))  ##OLD, slightly slower version: p = trace(dot(self.SPAMs[spamLabel], G))
            #  vp[i] = sum_k,l E[0,k] Gs[i,k,l] rho[l,0]
            #  vp[i] = sum_k E[0,k] dot(Gs, rho)[i,k,0]
            #  vp[i] = dot( E, dot(Gs, rho))[0,i,0]
            #  vp    = squeeze( dot( E, dot(Gs, rho)), axis=(0,2) )
            if returnPr: 
                sub_vp = _np.squeeze( _np.dot(E, _np.dot(Gs, rho)), axis=(0,2) ) * scaleVals  # shape == (len(gatestring_list),) ; may overflow, but OK
    
            #Compute d(probability)/dGates and save in return list (now have G,dG => product, dprod_dGates)
            #  prod, dprod_dGates = G,dG
            # dp_dGates[i,j] = sum_k,l E[0,k] dGs[i,j,k,l] rho[l,0] 
            # dp_dGates[i,j] = sum_k E[0,k] dot( dGs, rho )[i,j,k,0]
            # dp_dGates[i,j] = dot( E, dot( dGs, rho ) )[0,i,j,0]
            # dp_dGates      = squeeze( dot( E, dot( dGs, rho ) ), axis=(0,3))
            old_err2 = _np.seterr(invalid='ignore', over='ignore')
            dp_dGates = _np.squeeze( _np.dot( E, _np.dot( dGs, rho ) ), axis=(0,3) ) * scaleVals[:,None] 
            _np.seterr(**old_err2)
               # may overflow, but OK ; shape == (len(gatestring_list), nGateDerivCols)
               # may also give invalid value due to scaleVals being inf and dot-prod being 0. In
               #  this case set to zero since we can't tell whether it's + or - inf anyway...
            dp_dGates[ _np.isnan(dp_dGates) ] = 0

            #DEBUG
            #assert( len( (_np.isnan(scaleVals)).nonzero()[0] ) == 0 ) 
            #xxx = _np.dot( E, _np.dot( dGs, rho ) )
            #assert( len( (_np.isnan(xxx)).nonzero()[0] ) == 0 )
            #if len( (_np.isnan(dp_dGates)).nonzero()[0] ) != 0:
            #    print "scaleVals = ",_np.min(scaleVals),", ",_np.max(scaleVals)
            #    print "xxx = ",_np.min(xxx),", ",_np.max(xxx)
            #    print len( (_np.isinf(xxx)).nonzero()[0] )
            #    print len( (_np.isinf(scaleVals)).nonzero()[0] )
            #    assert( len( (_np.isnan(dp_dGates)).nonzero()[0] ) == 0 )
    
            #SPAM -------------

            # Get: dp_drhos[i, rho_offset[rhoIndex]:rho_offset[rhoIndex+1]] = dot(E,Gs[i],drho/drhoP)
            # dp_drhos[i,J0+J] = sum_kl E[0,k] Gs[i,k,l] drhoP[l,J]
            # dp_drhos[i,J0+J] = dot(E, Gs, drhoP)[0,i,J]
            # dp_drhos[:,J0+J] = squeeze(dot(E, Gs, drhoP),axis=(0,))[:,J]
            rhoIndex = self.rhovecs.keys().index(rholabel)
            dp_drhos = _np.zeros( (sub_nGateStrings, sum(num_rho_params) ) )
            dp_drhos[: , rho_offset[rhoIndex]:rho_offset[rhoIndex+1] ] = \
                _np.squeeze(_np.dot(_np.dot(E, Gs), rho.deriv_wrt_params()),axis=(0,)) \
                * scaleVals[:,None] # may overflow, but OK
            
            # Get: dp_dEs[i, e_offset[eIndex]:e_offset[eIndex+1]] = dot(transpose(dE/dEP),Gs[i],rho))
            # dp_dEs[i,J0+J] = sum_lj dEPT[J,j] Gs[i,j,l] rho[l,0] 
            # dp_dEs[i,J0+J] = sum_j dEP[j,J] dot(Gs, rho)[i,j]
            # dp_dEs[i,J0+J] = sum_j dot(Gs, rho)[i,j,0] dEP[j,J]
            # dp_dEs[i,J0+J] = dot(squeeze(dot(Gs, rho),2), dEP)[i,J]
            # dp_dEs[:,J0+J] = dot(squeeze(dot(Gs, rho),axis=(2,)), dEP)[:,J]
            dp_dEs = _np.zeros( (sub_nGateStrings, sum(num_e_params)) )
            dp_dAnyE = _np.squeeze(_np.dot(Gs, rho),axis=(2,)) * scaleVals[:,None] #may overflow, but OK (deriv w.r.t any of self.evecs - independent of which)
            if elabel == self._remainderLabel:
                for ei,evec in enumerate(self.evecs.values()): #compute Deriv w.r.t. [ 1 - sum_of_other_EVecs ]
                    dp_dEs[:,e_offset[ei]:e_offset[ei+1]] = -1.0 * _np.dot(dp_dAnyE, evec.deriv_wrt_params())
            else:
                eIndex = self.evecs.keys().index(elabel)
                dp_dEs[:,e_offset[eIndex]:e_offset[eIndex+1]] = \
                    _np.dot(dp_dAnyE, self.evecs[elabel].deriv_wrt_params())
            sub_vdp = _np.concatenate( (dp_drhos,dp_dEs,dp_dGates), axis=1 )
    
            _np.seterr(**old_err)

            if evalTree.is_split():
                if returnPr: vp[ evalSubTree.myFinalToParentFinalMap ] = sub_vp
                vdp[ evalSubTree.myFinalToParentFinalMap, : ] = sub_vdp
            else: 
                if returnPr: vp = sub_vp
                vdp = sub_vdp

        if returnPr and clipTo is not None: #do this before check...
            vp = _np.clip( vp, clipTo[0], clipTo[1] )

        if check: 
            # compare with older slower version that should do the same thing (for debugging)
            gatestring_list = evalTree.generate_gatestring_list()
            check_vdp = _np.concatenate( [ self.dpr(spamLabel, gateString, False,clipTo) for gateString in gatestring_list ], axis=0 )
            check_vp = _np.array( [ self.pr(spamLabel, gateString, clipTo) for gateString in gatestring_list ] )

            if returnPr and _nla.norm(vp - check_vp) > 1e-6:
                _warnings.warn("norm(vp-check_vp) = %g - %g = %g" % (_nla.norm(vp), _nla.norm(check_vp), _nla.norm(vp - check_vp)))
                #for i,gs in enumerate(gatestring_list):
                #    if abs(vp[i] - check_vp[i]) > 1e-6: 
                #        print "   %s => p=%g, check_p=%g, diff=%g" % (str(gs),vp[i],check_vp[i],abs(vp[i]-check_vp[i]))
            if _nla.norm(vdp - check_vdp) > 1e-6:
                _warnings.warn("Norm(vdp-check_vdp) = %g - %g = %g" % (_nla.norm(vdp), _nla.norm(check_vdp), _nla.norm(vdp - check_vdp)))

        if returnPr: return vdp, vp
        else:        return vdp



    def bulk_hpr(self, spamLabel, evalTree, 
                 returnPr=False,returnDeriv=False,
                 clipTo=None,check=False):

        """
        Compute the derivatives of the probabilities generated by a each gate 
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

        if self._is_remainder_spamlabel(spamLabel):
            #then compute Hessian[ 1.0 - (all other spam label probabilities) ]
            otherSpamLabels = self.get_spam_labels(); del otherSpamLabels[ otherSpamLabels.index(spamLabel) ]
            assert( not any([ self._is_remainder_spamlabel(sl) for sl in otherSpamLabels]) )
            otherResults = [self.bulk_hpr(sl, evalTree, returnPr, returnDeriv, clipTo) for sl in otherSpamLabels]
            if returnDeriv: 
                if returnPr: return ( -1.0 * _np.sum([hpr for hpr,dpr,p in otherResults],axis=0),
                                      -1.0 * _np.sum([dpr for hpr,dpr,p in otherResults],axis=0), 
                                       1.0 - _np.sum([p   for hpr,dpr,p in otherResults],axis=0) )
                else:        return ( -1.0 * _np.sum([hpr for hpr,dpr in otherResults],axis=0),
                                      -1.0 * _np.sum([dpr for hpr,dpr in otherResults],axis=0)   )
            else:
                if returnPr: return ( -1.0 * _np.sum([hpr for hpr,p in otherResults],axis=0),
                                       1.0 - _np.sum([p   for hpr,p in otherResults],axis=0)  )
                else:        return   -1.0 * _np.sum(otherResults,axis=0)

        (rholabel,elabel) = self.spamLabels[spamLabel]
        rho = self.rhovecs[rholabel]
        E   = _np.conjugate(_np.transpose(self._get_evec(elabel)))

        nGateStrings = evalTree.num_final_strings()
        num_rho_params = [v.num_params() for v in self.rhovecs.values()]
        num_e_params = [v.num_params() for v in self.evecs.values()]
        rho_offset = [ sum(num_rho_params[0:i]) for i in range(len(self.rhovecs)+1) ]
        e_offset = [ sum(num_e_params[0:i]) for i in range(len(self.evecs)+1) ]
        nDerivCols = sum(num_rho_params + num_e_params + [g.num_params() for g in self.gates.values()])

        if evalTree.is_split():
            vp = _np.empty( nGateStrings, 'd' )
            vdp = _np.empty( (nGateStrings, nDerivCols), 'd' )  
            vhp = _np.empty( (nGateStrings, nDerivCols, nDerivCols), 'd' )

        for evalSubTree in evalTree.get_sub_trees():
            sub_nGateStrings = evalSubTree.num_final_strings()
            hGs, dGs, Gs, scaleVals = self.bulk_hproduct(evalSubTree, bScale=True, bReturnDProdsAndProds=True)
            old_err = _np.seterr(over='ignore')
    
            #Compute probability and save in return array
            # want vp[iFinal] = float(dot(E, dot(G, rho)))  ##OLD, slightly slower version: p = trace(dot(self.SPAMs[spamLabel], G))
            #  vp[i] = sum_k,l E[0,k] Gs[i,k,l] rho[l,0]
            #  vp[i] = sum_k E[0,k] dot(Gs, rho)[i,k,0]
            #  vp[i] = dot( E, dot(Gs, rho))[0,i,0]
            #  vp    = squeeze( dot( E, dot(Gs, rho)), axis=(0,2) )
            if returnPr: 
                sub_vp = _np.squeeze( _np.dot(E, _np.dot(Gs, rho)), axis=(0,2) ) * scaleVals  # shape == (len(gatestring_list),) ; may overflow, but OK
    
            #Compute d(probability)/dGates and save in return list (now have G,dG => product, dprod_dGates)
            #  prod, dprod_dGates = G,dG
            # dp_dGates[i,j] = sum_k,l E[0,k] dGs[i,j,k,l] rho[l,0] 
            # dp_dGates[i,j] = sum_k E[0,k] dot( dGs, rho )[i,j,k,0]
            # dp_dGates[i,j] = dot( E, dot( dGs, rho ) )[0,i,j,0]
            # dp_dGates      = squeeze( dot( E, dot( dGs, rho ) ), axis=(0,3))
            if returnDeriv:
                old_err2 = _np.seterr(invalid='ignore', over='ignore')
                dp_dGates = _np.squeeze( _np.dot( E, _np.dot( dGs, rho ) ), axis=(0,3) ) * scaleVals[:,None] 
                _np.seterr(**old_err2)
                # may overflow, but OK ; shape == (len(gatestring_list), nGateDerivCols)
                # may also give invalid value due to scaleVals being inf and dot-prod being 0. In
                #  this case set to zero since we can't tell whether it's + or - inf anyway...
                dp_dGates[ _np.isnan(dp_dGates) ] = 0
    
    
            #Compute d2(probability)/dGates2 and save in return list
            # d2pr_dGates2[i,j,k] = sum_l,m E[0,l] hGs[i,j,k,l,m] rho[m,0] 
            # d2pr_dGates2[i,j,k] = sum_l E[0,l] dot( dGs, rho )[i,j,k,l,0]
            # d2pr_dGates2[i,j,k] = dot( E, dot( dGs, rho ) )[0,i,j,k,0]
            # d2pr_dGates2        = squeeze( dot( E, dot( dGs, rho ) ), axis=(0,4))
            old_err2 = _np.seterr(invalid='ignore', over='ignore')
            d2pr_dGates2 = _np.squeeze( _np.dot( E, _np.dot( hGs, rho ) ), axis=(0,4) ) * scaleVals[:,None,None] 
            _np.seterr(**old_err2)
            # may overflow, but OK ; shape == (len(gatestring_list), nGateDerivCols, nGateDerivCols)
            # may also give invalid value due to scaleVals being inf and dot-prod being 0. In
            #  this case set to zero since we can't tell whether it's + or - inf anyway...
            d2pr_dGates2[ _np.isnan(d2pr_dGates2) ] = 0
    
    
            #SPAM ---------------------------------
            if returnDeriv: #same as in bulk_dpr - see comments there for details
                rhoIndex = self.rhovecs.keys().index(rholabel)
                dp_drhos = _np.zeros( (sub_nGateStrings, sum(num_rho_params) ) )
                dp_drhos[: , rho_offset[rhoIndex]:rho_offset[rhoIndex+1] ] = \
                    _np.squeeze(_np.dot(_np.dot(E, Gs), rho.deriv_wrt_params()),axis=(0,)) \
                    * scaleVals[:,None] # may overflow, but OK

                dp_dEs = _np.zeros( (sub_nGateStrings, sum(num_e_params)) )
                dp_dAnyE = _np.squeeze(_np.dot(Gs, rho),axis=(2,)) * scaleVals[:,None] #may overflow, but OK
                if elabel == self._remainderLabel:
                    for ei,evec in enumerate(self.evecs.values()): #compute Deriv w.r.t. [ 1 - sum_of_other_EVecs ]
                        dp_dEs[:,e_offset[ei]:e_offset[ei+1]] = -1.0 * _np.dot(dp_dAnyE, evec.deriv_wrt_params())
                else:
                    eIndex = self.evecs.keys().index(elabel)
                    dp_dEs[:,e_offset[eIndex]:e_offset[eIndex+1]] = \
                        _np.dot(dp_dAnyE, self.evecs[elabel].deriv_wrt_params())
                vdp = _np.concatenate( (dp_drhos,dp_dEs,dp_dGates), axis=1 )
                sub_vdp = vdp
    
            vec_gs_size = dGs.shape[1]
    
            # Get: d2pr_drhos[i, j, rho_offset[rhoIndex]:rho_offset[rhoIndex+1]] = dot(E,dGs[i,j],drho/drhoP))
            # d2pr_drhos[i,j,J0+J] = sum_kl E[0,k] dGs[i,j,k,l] drhoP[l,J]
            # d2pr_drhos[i,j,J0+J] = dot(E, dGs, drhoP)[0,i,j,J]
            # d2pr_drhos[:,:,J0+J] = squeeze(dot(E, dGs, drhoP),axis=(0,))[:,:,J]            
            d2pr_drhos = _np.zeros( (sub_nGateStrings, vec_gs_size, sum(num_rho_params)) )
            d2pr_drhos[:, :, rho_offset[rhoIndex]:rho_offset[rhoIndex+1]] = \
                _np.squeeze( _np.dot(_np.dot(E,dGs),rho.deriv_wrt_params()), axis=(0,)) \
                * scaleVals[:,None,None] #overflow OK
    
            # Get: d2pr_dEs[i, j, e_offset[eIndex]:e_offset[eIndex+1]] = dot(transpose(dE/dEP),dGs[i,j],rho)
            # d2pr_dEs[i,j,J0+J] = sum_kl dEPT[J,k] dGs[i,j,k,l] rho[l,0]
            # d2pr_dEs[i,j,J0+J] = sum_k dEP[k,J] dot(dGs, rho)[i,j,k,0]
            # d2pr_dEs[i,j,J0+J] = dot( squeeze(dot(dGs, rho),axis=(3,)), dEP)[i,j,J]
            # d2pr_dEs[:,:,J0+J] = dot( squeeze(dot(dGs, rho),axis=(3,)), dEP)[:,:,J]
            d2pr_dEs = _np.zeros( (sub_nGateStrings, vec_gs_size, sum(num_e_params)) )
            dp_dAnyE = _np.squeeze(_np.dot(dGs,rho), axis=(3,)) * scaleVals[:,None,None] #overflow OK
            if elabel == self._remainderLabel:
                for ei,evec in enumerate(self.evecs.values()):
                    d2pr_dEs[:, :, e_offset[ei]:e_offset[ei+1]] = -1.0 * _np.dot(dp_dAnyE, evec.deriv_wrt_params())
            else:
                eIndex = self.evecs.keys().index(elabel)
                d2pr_dEs[:, :, e_offset[eIndex]:e_offset[eIndex+1]] = \
                    _np.dot(dp_dAnyE, self.evecs[elabel].deriv_wrt_params())

    
            # Get: d2pr_dErhos[i, e_offset[eIndex]:e_offset[eIndex+1], e_offset[rhoIndex]:e_offset[rhoIndex+1]] =
            #    dEP^T * prod[i,:,:] * drhoP
            # d2pr_dErhos[i,J0+J,K0+K] = sum jk dEPT[J,j] prod[i,j,k] drhoP[k,K]
            # d2pr_dErhos[i,J0+J,K0+K] = sum j dEPT[J,j] dot(prod,drhoP)[i,j,K]
            # d2pr_dErhos[i,J0+J,K0+K] = dot(dEPT,prod,drhoP)[J,i,K]
            # d2pr_dErhos[i,J0+J,K0+K] = swapaxes(dot(dEPT,prod,drhoP),0,1)[i,J,K]
            # d2pr_dErhos[:,J0+J,K0+K] = swapaxes(dot(dEPT,prod,drhoP),0,1)[:,J,K]

#                    -1.0 * _np.dot( _np.transpose(evec.deriv_wrt_params()),derivWrtAnyEvec)
            d2pr_dErhos = _np.zeros( (sub_nGateStrings, sum(num_e_params), sum(num_rho_params)) )
            dp_dAnyE = _np.dot(Gs, rho.deriv_wrt_params()) * scaleVals[:,None,None] #overflow OK
            if elabel == self._remainderLabel:
                for ei,evec in enumerate(self.evecs.values()):
                    d2pr_dErhos[:, e_offset[ei]:e_offset[ei+1], rho_offset[rhoIndex]:rho_offset[rhoIndex+1]] = \
                        -1.0 * _np.dot(_np.transpose(evec.deriv_wrt_params()), dp_dAnyE )
            else:
                eIndex = self.evecs.keys().index(elabel)
                d2pr_dErhos[:, e_offset[eIndex]:e_offset[eIndex+1], rho_offset[rhoIndex]:rho_offset[rhoIndex+1]] = \
                    _np.dot(_np.transpose(self.evecs[elabel].deriv_wrt_params()), dp_dAnyE )
    
            d2pr_d2rhos = _np.zeros( (sub_nGateStrings, sum(num_rho_params), sum(num_rho_params)) )
            d2pr_d2Es   = _np.zeros( (sub_nGateStrings, sum(num_e_params), sum(num_e_params)) )
            #END SPAM -----------------------
    
            ret_row1 = _np.concatenate( ( d2pr_d2rhos, _np.transpose(d2pr_dErhos,(0,2,1)), _np.transpose(d2pr_drhos,(0,2,1)) ), axis=2) # wrt rho
            ret_row2 = _np.concatenate( ( d2pr_dErhos, d2pr_d2Es, _np.transpose(d2pr_dEs,(0,2,1)) ), axis=2 ) # wrt E
            ret_row3 = _np.concatenate( ( d2pr_drhos,d2pr_dEs,d2pr_dGates2), axis=2 ) #wrt gates
            sub_vhp = _np.concatenate( (ret_row1, ret_row2, ret_row3), axis=1 )

            _np.seterr(**old_err)

            if evalTree.is_split():
                if returnPr: vp[ evalSubTree.myFinalToParentFinalMap ] = sub_vp
                if returnDeriv: vdp[ evalSubTree.myFinalToParentFinalMap, : ] = sub_vdp
                vhp[ evalSubTree.myFinalToParentFinalMap, :, : ] = sub_vhp
            else: 
                if returnPr: vp = sub_vp
                if returnDeriv: vdp = sub_vdp
                vhp = sub_vhp
        

        if returnPr and clipTo is not None:  # do this before check...
            vp = _np.clip( vp, clipTo[0], clipTo[1] )

        if check: 
            # compare with older slower version that should do the same thing (for debugging)
            gatestring_list = evalTree.generate_gatestring_list()
            check_vhp = _np.concatenate( [ self.hpr(spamLabel, gateString, False,False,clipTo) for gateString in gatestring_list ], axis=0 )
            check_vdp = _np.concatenate( [ self.dpr(spamLabel, gateString, False,clipTo) for gateString in gatestring_list ], axis=0 )
            check_vp = _np.array( [ self.pr(spamLabel, gateString, clipTo) for gateString in gatestring_list ] )

            if returnPr and _nla.norm(vp - check_vp) > 1e-6:
                _warnings.warn("norm(vp-check_vp) = %g - %g = %g" % (_nla.norm(vp), _nla.norm(check_vp), _nla.norm(vp - check_vp)))
                #for i,gs in enumerate(gatestring_list):
                #    if abs(vp[i] - check_vp[i]) > 1e-6: 
                #        print "   %s => p=%g, check_p=%g, diff=%g" % (str(gs),vp[i],check_vp[i],abs(vp[i]-check_vp[i]))
            if returnDeriv and _nla.norm(vdp - check_vdp) > 1e-6:
                _warnings.warn("norm(vdp-check_vdp) = %g - %g = %g" % (_nla.norm(vdp), _nla.norm(check_vdp), _nla.norm(vdp - check_vdp)))
            if _nla.norm(vhp - check_vhp) > 1e-6:
                _warnings.warn("norm(vhp-check_vhp) = %g - %g = %g" % (_nla.norm(vhp), _nla.norm(check_vhp), _nla.norm(vhp - check_vhp)))

        if returnDeriv: 
            if returnPr: return vhp, vdp, vp
            else:        return vhp, vdp
        else:
            if returnPr: return vhp, vp
            else:        return vhp


    def bulk_probs(self, evalTree, clipTo=None, check=False):
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
           
        Returns
        -------
        probs : dictionary
            A dictionary such that 
            probs[SL] = bulk_pr(SL,evalTree,clipTo,check)
            for each spam label (string) SL.
        """
        probs = { }
        if not self.assumeSumToOne:
            for spamLabel in self.spamLabels:
                probs[spamLabel] = self.bulk_pr(spamLabel, evalTree, clipTo, check)
        else:
            s = _np.zeros( evalTree.num_final_strings(), 'd'); lastLabel = None
            for spamLabel in self.spamLabels:
                if self._is_remainder_spamlabel(spamLabel):
                    assert(lastLabel is None) # ensure there is at most one dummy spam label
                    lastLabel = spamLabel; continue
                probs[spamLabel] = self.bulk_pr(spamLabel, evalTree, clipTo, check)
                s += probs[spamLabel]
            if lastLabel is not None: probs[lastLabel] = 1.0 - s  #last spam label is computed so sum == 1
        return probs


    def bulk_dprobs(self, evalTree, 
                    returnPr=False,clipTo=None,
                    check=False,memLimit=None):

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

        memLimit : int, optional
          A rough memory limit in bytes which restricts the amount of
          intermediate values that are computed and stored.

        Returns
        -------
        dprobs : dictionary
            A dictionary such that 
            ``dprobs[SL] = bulk_dpr(SL,evalTree,gates,G0,SPAM,SP0,returnPr,clipTo,check,memLimit)``
            for each spam label (string) SL.
        """
        dprobs = { }
        if not self.assumeSumToOne:
            for spamLabel in self.spamLabels:
                dprobs[spamLabel] = self.bulk_dpr(spamLabel, evalTree,
                                                  returnPr,clipTo,check,memLimit)
        else:
            ds = None; lastLabel = None
            s = _np.zeros( evalTree.num_final_strings(), 'd')
            for spamLabel in self.spamLabels:
                if self._is_remainder_spamlabel(spamLabel):
                    assert(lastLabel is None) # ensure there is at most one dummy spam label
                    lastLabel = spamLabel; continue
                dprobs[spamLabel] = self.bulk_dpr(spamLabel, evalTree,
                                                  returnPr,clipTo,check,memLimit)
                if returnPr:
                    ds = dprobs[spamLabel][0] if ds is None else ds + dprobs[spamLabel][0]
                    s += dprobs[spamLabel][1]
                else:
                    ds = dprobs[spamLabel] if ds is None else ds + dprobs[spamLabel]                    
            if lastLabel is not None:
                dprobs[lastLabel] = (-ds,1.0-s) if returnPr else -ds
        return dprobs


    def bulk_hprobs(self, evalTree, 
                    returnPr=False,returnDeriv=False,clipTo=None,
                    check=False):

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

        Returns
        -------
        hprobs : dictionary
            A dictionary such that 
            ``hprobs[SL] = bulk_hpr(SL,evalTree,gates,G0,SPAM,SP0,returnPr,returnDeriv,clipTo,check)``
            for each spam label (string) SL.
        """
        hprobs = { }
        if not self.assumeSumToOne:
            for spamLabel in self.spamLabels:
                hprobs[spamLabel] = self.bulk_hpr(spamLabel, evalTree,
                                                  returnPr,returnDeriv,clipTo,check)
        else:
            hs = None; ds = None; lastLabel = None
            s = _np.zeros( evalTree.num_final_strings(), 'd')
            for spamLabel in self.spamLabels:
                if self._is_remainder_spamlabel(spamLabel):
                    assert(lastLabel is None) # ensure there is at most one dummy spam label
                    lastLabel = spamLabel; continue
                hprobs[spamLabel] = self.bulk_hpr(spamLabel, evalTree,
                                                  returnPr,returnDeriv,clipTo,check)

                if returnPr:
                    if returnDeriv:
                        hs = hprobs[spamLabel][0] if hs is None else hs + hprobs[spamLabel][0]
                        ds = hprobs[spamLabel][1] if ds is None else ds + hprobs[spamLabel][1]
                        s += hprobs[spamLabel][2]
                    else:
                        hs = hprobs[spamLabel][0] if hs is None else hs + hprobs[spamLabel][0]
                        s += hprobs[spamLabel][1]
                else:
                    if returnDeriv:
                        hs = hprobs[spamLabel][0] if hs is None else hs + hprobs[spamLabel][0]
                        ds = hprobs[spamLabel][1] if ds is None else ds + hprobs[spamLabel][1]
                    else:
                        hs = hprobs[spamLabel] if hs is None else hs + hprobs[spamLabel]

            if lastLabel is not None: 
                if returnPr:
                    hprobs[lastLabel] = (-hs,-ds,1.0-s) if returnDeriv else (-hs,1.0-s)
                else:
                    hprobs[lastLabel] = (-hs,-ds) if returnDeriv else -hs

        return hprobs



    def bulk_fill_probs(self, mxToFill, spam_label_rows, 
                       evalTree, clipTo=None, check=False):
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
           
        Returns
        -------
        None
        """
        if not self.assumeSumToOne:
            for spamLabel,rowIndex in spam_label_rows.iteritems():
                mxToFill[rowIndex] = self.bulk_pr(spamLabel, evalTree, clipTo, check)
        else:
            s = _np.zeros( evalTree.num_final_strings(), 'd'); lastLabel = None
            for spamLabel in self.spamLabels: #Note: must loop through all spam labels, even if not requested
                if self._is_remainder_spamlabel(spamLabel):
                    assert(lastLabel is None) # ensure there is at most one dummy spam label
                    lastLabel = spamLabel; continue
                probs = self.bulk_pr(spamLabel, evalTree, clipTo, check)
                s += probs

                if spam_label_rows.has_key(spamLabel):
                    mxToFill[ spam_label_rows[spamLabel] ] = probs

            if lastLabel is not None and spam_label_rows.has_key(lastLabel):
                mxToFill[ spam_label_rows[lastLabel] ] = 1.0 - s  #last spam label is computed so sum == 1


    def bulk_fill_dprobs(self, mxToFill, spam_label_rows, evalTree,
                         prMxToFill=None,clipTo=None,check=False,memLimit=None):

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

        Returns
        -------
        None
        """
        if not self.assumeSumToOne:
            if prMxToFill is not None:
                for spamLabel,rowIndex in spam_label_rows.iteritems():
                    mxToFill[rowIndex], prMxToFill[rowIndex] = \
                        self.bulk_dpr(spamLabel,evalTree,True,clipTo,check,memLimit)
            else:
                for spamLabel,rowIndex in spam_label_rows.iteritems():
                    mxToFill[rowIndex] = self.bulk_dpr(spamLabel,evalTree,False,clipTo,check,memLimit)

        else:
            ds = None; lastLabel = None
            s = _np.zeros( evalTree.num_final_strings(), 'd')

            if prMxToFill is not None: #then compute & fill probabilities too
                for spamLabel in self.spamLabels: #Note: must loop through all spam labels, even if not requested, in case prMxToFill is not None
                    if self._is_remainder_spamlabel(spamLabel):
                        assert(lastLabel is None) # ensure there is at most one dummy spam label
                        lastLabel = spamLabel; continue
                    dprobs, probs = self.bulk_dpr(spamLabel,evalTree,True,clipTo,check,memLimit)
                    ds = dprobs if ds is None else ds + dprobs
                    s += probs
                    if spam_label_rows.has_key(spamLabel):
                        mxToFill[ spam_label_rows[spamLabel] ] = dprobs
                        prMxToFill[ spam_label_rows[spamLabel] ] = probs

                if lastLabel is not None and spam_label_rows.has_key(lastLabel):
                    mxToFill[ spam_label_rows[lastLabel] ] = -ds
                    prMxToFill[ spam_label_rows[lastLabel] ] = 1.0-s

            else: #just compute derivatives of probabilities
                for spamLabel in self.spamLabels: #Note: must loop through all spam labels, even if not requested, in case prMxToFill is not None
                    if self._is_remainder_spamlabel(spamLabel):
                        assert(lastLabel is None) # ensure there is at most one dummy spam label
                        lastLabel = spamLabel; continue
                    dprobs = self.bulk_dpr(spamLabel,evalTree,False,clipTo,check,memLimit)
                    ds = dprobs if ds is None else ds + dprobs
                    if spam_label_rows.has_key(spamLabel):
                        mxToFill[ spam_label_rows[spamLabel] ] = dprobs

                if lastLabel is not None and spam_label_rows.has_key(lastLabel):
                    mxToFill[ spam_label_rows[lastLabel] ] = -ds


    def bulk_fill_hprobs(self, mxToFill, spam_label_rows, evalTree,
                         prMxToFill=None, derivMxToFill=None, clipTo=None,
                         check=False):

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

        Returns
        -------
        None
        """
        if not self.assumeSumToOne:
            if prMxToFill is not None:
                if derivMxToFill is not None:
                    for spamLabel,rowIndex in spam_label_rows.iteritems():
                        mxToFill[rowIndex], derivMxToFill[rowIndex], prMxToFill[rowIndex] = \
                            self.bulk_hpr(spamLabel,evalTree,True,True,clipTo,check)
                else:
                    for spamLabel,rowIndex in spam_label_rows.iteritems():
                        mxToFill[rowIndex], prMxToFill[rowIndex] = \
                            self.bulk_hpr(spamLabel,evalTree,True,False,clipTo,check)

            else:
                if derivMxToFill is not None:
                    for spamLabel,rowIndex in spam_label_rows.iteritems():
                        mxToFill[rowIndex], derivMxToFill[rowIndex] = \
                            self.bulk_hpr(spamLabel,evalTree,False,True,clipTo,check)
                else:
                    for spamLabel,rowIndex in spam_label_rows.iteritems():
                        mxToFill[rowIndex] = self.bulk_hpr(spamLabel, evalTree,
                                                           False,False,clipTo,check)

        else:  # assumeSumToOne == True

            hs = None; ds = None; lastLabel = None
            s = _np.zeros( evalTree.num_final_strings(), 'd')

            if prMxToFill is not None: #then compute & fill probabilities too
                if derivMxToFill is not None: #then compute & fill derivatives too
                    for spamLabel in self.spamLabels: #Note: must loop through all spam labels, even if not requested, in case prMxToFill is not None
                        if self._is_remainder_spamlabel(spamLabel):
                            assert(lastLabel is None) # ensure there is at most one dummy spam label
                            lastLabel = spamLabel; continue
                        hprobs, dprobs, probs = self.bulk_hpr(spamLabel, evalTree,
                                                              True,True,clipTo,check)
                        hs = hprobs if hs is None else hs + hprobs
                        ds = dprobs if ds is None else ds + dprobs
                        s += probs
                        if spam_label_rows.has_key(spamLabel):
                            mxToFill[ spam_label_rows[spamLabel] ] = hprobs
                            derivMxToFill[ spam_label_rows[spamLabel] ] = dprobs
                            prMxToFill[ spam_label_rows[spamLabel] ] = probs
    
                    if lastLabel is not None and spam_label_rows.has_key(lastLabel):
                        mxToFill[ spam_label_rows[lastLabel] ] = -hs
                        derivMxToFill[ spam_label_rows[lastLabel] ] = -ds
                        prMxToFill[ spam_label_rows[lastLabel] ] = 1.0-s

                else: #compute hessian & probs (no derivs)

                    for spamLabel in self.spamLabels: #Note: must loop through all spam labels, even if not requested, in case prMxToFill is not None
                        if self._is_remainder_spamlabel(spamLabel):
                            assert(lastLabel is None) # ensure there is at most one dummy spam label
                            lastLabel = spamLabel; continue
                        hprobs, probs = self.bulk_hpr(spamLabel, evalTree,
                                                      True,False,clipTo,check)
                        hs = hprobs if hs is None else hs + hprobs
                        s += probs
                        if spam_label_rows.has_key(spamLabel):
                            mxToFill[ spam_label_rows[spamLabel] ] = hprobs
                            prMxToFill[ spam_label_rows[spamLabel] ] = probs
    
                    if lastLabel is not None and spam_label_rows.has_key(lastLabel):
                        mxToFill[ spam_label_rows[lastLabel] ] = -hs
                        prMxToFill[ spam_label_rows[lastLabel] ] = 1.0-s


            else: 
                if derivMxToFill is not None: #compute hessians and derivatives (no probs)

                    for spamLabel in self.spamLabels: #Note: must loop through all spam labels, even if not requested, in case prMxToFill is not None
                        if self._is_remainder_spamlabel(spamLabel):
                            assert(lastLabel is None) # ensure there is at most one dummy spam label
                            lastLabel = spamLabel; continue
                        hprobs, dprobs = self.bulk_hpr(spamLabel, evalTree,
                                                       False,True,clipTo,check)
                        hs = hprobs if hs is None else hs + hprobs
                        ds = dprobs if ds is None else ds + dprobs
                        if spam_label_rows.has_key(spamLabel):
                            mxToFill[ spam_label_rows[spamLabel] ] = hprobs
                            derivMxToFill[ spam_label_rows[spamLabel] ] = dprobs
    
                    if lastLabel is not None and spam_label_rows.has_key(lastLabel):
                        mxToFill[ spam_label_rows[lastLabel] ] = -hs
                        derivMxToFill[ spam_label_rows[lastLabel] ] = -ds

                else: #just compute derivatives of probabilities

                    for spamLabel in self.spamLabels: #Note: must loop through all spam labels, even if not requested, in case prMxToFill is not None
                        if self._is_remainder_spamlabel(spamLabel):
                            assert(lastLabel is None) # ensure there is at most one dummy spam label
                            lastLabel = spamLabel; continue
                        hprobs = self.bulk_hpr(spamLabel, evalTree,
                                               False,False,clipTo,check)
                        hs = hprobs if hs is None else hs + hprobs
                        if spam_label_rows.has_key(spamLabel):
                            mxToFill[ spam_label_rows[spamLabel] ] = hprobs
    
                    if lastLabel is not None and spam_label_rows.has_key(lastLabel):
                        mxToFill[ spam_label_rows[lastLabel] ] = -hs



    def frobeniusdist(self, otherCalc, transformMx=None,
                      gateWeight=1.0, spamWeight=1.0, normalize=True):
        """
        Compute the weighted frobenius norm of the difference between two
        gatesets.  Differences in each corresponding gate matrix and spam
        vector element are squared, weighted (using gateWeight
        or spamWeight as applicable), then summed.  The value returned is the
        square root of this sum, or the square root of this sum divided by the
        number of summands if normalize == True.

        Parameters
        ----------
        otherCalc : GateSetCalculator
            the other gate set calculator to difference against.

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

        normalize : bool, optional
           if True (the default), the frobenius difference is defined by the
           sum of weighted squared-differences divided by the number of 
           differences.  If False, this final division is not performed.

        Returns
        -------
        float
        """
        d = 0; T = transformMx
        nSummands = 0.0
        if T is not None:
            Ti = _nla.inv(T)
            for gateLabel in self.gates:
                d += gateWeight * _gt.frobeniusdist2(_np.dot(
                    Ti,_np.dot(self.gates[gateLabel],T)),
                    otherCalc.gates[gateLabel] )
                nSummands += gateWeight * _np.size(self.gates[gateLabel])

            for (lbl,rhoV) in self.rhovecs.iteritems(): 
                d += spamWeight * _gt.frobeniusdist2(_np.dot(Ti,rhoV),
                                                     otherCalc.rhoVecs[lbl])
                nSummands += spamWeight * _np.size(rhoV)

            for (lbl,Evec) in self.evecs.iteritems():
                d += spamWeight * _gt.frobeniusdist2(_np.dot(
                    _np.transpose(T),Evec),otherCalc.evecs[lbl])
                nSummands += spamWeight * _np.size(Evec)

            if self.identityvec is not None:
                d += spamWeight * _gt.frobeniusdist2(_np.dot(
                    _np.transpose(T),self.identityvec),otherCalc.identityvec)
                nSummands += spamWeight * _np.size(self.identityvec)

        else:
            for gateLabel in self.gates:
                d += gateWeight * _gt.frobeniusdist2(self.gates[gateLabel],
                                                     otherCalc.gates[gateLabel])
                nSummands += gateWeight * _np.size(self[gateLabel])

            for (lbl,rhoV) in self.rhovecs.iteritems(): 
                d += spamWeight * _gt.frobeniusdist2(rhoV,
                                      otherCalc.rhovecs[lbl])
                nSummands += spamWeight *  _np.size(rhoV)

            for (lbl,Evec) in self.evecs.iteritems():
                d += spamWeight * _gt.frobeniusdist2(Evec,otherCalc.evecs[lbl])
                nSummands += spamWeight * _np.size(Evec)

            if self.identityvec is not None and \
               otherCalc.identityvec is not None:
                d += spamWeight * _gt.frobeniusdist2(self.identityvec,
                                                     otherCalc.identityvec)
                nSummands += spamWeight * _np.size(self.identityvec)

        if normalize and nSummands > 0: 
            return _np.sqrt( d / float(nSummands) )
        else:
            return _np.sqrt(d)


    def jtracedist(self, otherCalc, transformMx=None):
        """
        Compute the Jamiolkowski trace distance between two
        gatesets, defined as the maximum of the trace distances
        between each corresponding gate, including spam gates.

        Parameters
        ----------
        otherCalc : GateSetCalculator
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
        T = transformMx
        if T is not None:
            Ti = _nla.inv(T)
            dists = [ _gt.jtracedist( _np.dot(Ti,_np.dot(self.gates[gateLabel],
                                      T)), otherCalc.gates[gateLabel] ) 
                       for gateLabel in self.gates ]

            for spamLabel in self.spamLabels.values():
                spamGate = self._make_spamgate(spamLabel)
                spamGate2 = otherCalc._make_spamgate(spamLabel)
                if spamGate is not None and spamGate2 is not None:
                    dists.append( _gt.jtracedist( _np.dot(Ti,
                                  _np.dot(spamGate,T)), spamGate2 ) )
        else:
            dists = [ _gt.jtracedist(self[gateLabel], otherGateSet[gateLabel])
                      for gateLabel in self.gates ]

            for spamLabel in self.spamLabels.values():
                spamGate = self._make_spamgate(spamLabel)
                spamGate2 = otherCalc._make_spamgate(spamLabel)
                if spamGate is not None and spamGate2 is not None:
                    dists.append( _gt.jtracedist(spamGate, spamGate2 ) )

        return max(dists)


    def diamonddist(self, otherGateSet, transformMx=None):
        """
        Compute the diamond-norm distance between two
        gatesets, defined as the maximum
        of the diamond-norm distances between each
        corresponding gate, including spam gates.

        Parameters
        ----------
        otherCalc : GateSetCalculator
            the other gate set calculator to difference against.

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
        T = transformMx
        if T is not None:
            Ti = _nla.inv(T)
            dists = [ _gt.diamonddist( 
                    _np.dot(Ti,_np.dot(self.gates[gateLabel],T)),
                    otherGateSet.gates[gateLabel] )
                      for gateLabel in self.gates ]

            for spamLabel in self.spamLabels.values():
                spamGate = self._make_spamgate(spamLabel)
                spamGate2 = otherCalc._make_spamgate(spamLabel)
                if spamGate is not None and spamGate2 is not None:
                    dists.append( _gt.diamonddist( 
                            _np.dot(Ti,_np.dot(spamGate,T)),spamGate2 ) )
        else:
            dists = [ _gt.diamonddist(self.gates[gateLabel],
                                      otherGateSet[gateLabel])
                      for gateLabel in self ]

            for spamLabel in self.spamLabels.values():
                spamGate = self._make_spamgate(spamLabel)
                spamGate2 = otherCalc._make_spamgate(spamLabel)
                if spamGate is not None and spamGate2 is not None:
                    dists.append( _gt.diamonddist(spamGate, spamGate2) )

        return max(dists)
