#define NULL 0

#include <iostream>
#include <complex>
#include <assert.h>
#include <algorithm>    // std::find

#include "statecreps.h"
#include "opcreps.h"
//#include <pthread.h>

//using namespace std::complex_literals;

//#define DEBUG(x) x
#define DEBUG(x) 

namespace CReps_densitymx {

  /****************************************************************************\
  |* OpCRep                                                               *|
  \****************************************************************************/

  OpCRep::OpCRep(INT dim) {
    _dim = dim;
  }
  OpCRep::~OpCRep() { }


  /****************************************************************************\
  |* OpCRep_Dense                                                         *|
  \****************************************************************************/

  OpCRep_Dense::OpCRep_Dense(double* data, INT dim)
    :OpCRep(dim)
  {
    _dataptr = data;
  }
  OpCRep_Dense::~OpCRep_Dense() { }

  StateCRep* OpCRep_Dense::acton(StateCRep* state,
				       StateCRep* outstate) {
    DEBUG(std::cout << "Dense acton called!" << std::endl);
    DEBUG(state->print("INPUT"));
    INT k;
    for(INT i=0; i< _dim; i++) {
      outstate->_dataptr[i] = 0.0;
      k = i*_dim; // "row" offset into _dataptr, so dataptr[k+j] ~= dataptr[i,j]
      for(INT j=0; j< _dim; j++) {
	outstate->_dataptr[i] += _dataptr[k+j] * state->_dataptr[j];
      }
    }
    DEBUG(outstate->print("OUTPUT"));
    return outstate;
  }

  StateCRep* OpCRep_Dense::adjoint_acton(StateCRep* state,
					       StateCRep* outstate) {
    DEBUG(std::cout << "Dense adjoint_acton called!" << std::endl);
    DEBUG(state->print("INPUT"));
    for(INT i=0; i< _dim; i++) {
      outstate->_dataptr[i] = 0.0;
      for(INT j=0; j< _dim; j++) {
	outstate->_dataptr[i] += _dataptr[j*_dim+i] * state->_dataptr[j];
      }
    }
    DEBUG(outstate->print("OUTPUT"));
    return outstate;
  }


  /****************************************************************************\
  |* OpCRep_Embedded                                                      *|
  \****************************************************************************/

  OpCRep_Embedded::OpCRep_Embedded(OpCRep* embedded_gate_crep, INT* noop_incrementers,
				       INT* numBasisEls_noop_blankaction, INT* baseinds, INT* blocksizes,
				       INT embedded_dim, INT nComponentsInActiveBlock, INT iActiveBlock,
				       INT nBlocks, INT dim)
    :OpCRep(dim)
  {
    _embedded_gate_crep = embedded_gate_crep;
    _noop_incrementers = noop_incrementers;
    _numBasisEls_noop_blankaction = numBasisEls_noop_blankaction;
    _baseinds = baseinds;
    _blocksizes = blocksizes;
    _nComponents = nComponentsInActiveBlock;
    _embeddedDim = embedded_dim;
    _iActiveBlock = iActiveBlock;
    _nBlocks = nBlocks;
  }
  OpCRep_Embedded::~OpCRep_Embedded() { }
  
  StateCRep* OpCRep_Embedded::acton(StateCRep* state, StateCRep* out_state) {

    DEBUG(std::cout << "DB Embedded acton called!" << std::endl);
    DEBUG(state->print("INPUT"));
    //_fastcalc.embedded_fast_acton_sparse(self.embedded_op.acton,
    //                                         output_state, state,
    //                                         self.noop_incrementers,
    //                                         self.numBasisEls_noop_blankaction,
    //                                         self.baseinds)
    INT i, j, k, vec_index_noop = 0;
    INT nParts = _nComponents;
    INT nActionIndices = _embeddedDim;
    INT offset;

    double* state_data = state->_dataptr;
    double* outstate_data = out_state->_dataptr;

    //zero-out output state initially
    for(i=0; i<_dim; i++) outstate_data[i] = 0.0;

    INT b[100]; // could alloc dynamically (LATER?)
    assert(nParts <= 100); // need to increase size of static arrays above
    for(i=0; i<nParts; i++) b[i] = 0;

    // Temporary states for acting the embedded gate on a subset of the whole
    StateCRep subState1(nActionIndices);
    StateCRep subState2(nActionIndices);
    
    while(true) {
      // Act with embedded gate on appropriate sub-space of state
      // out_state[ inds ] += embedded_gate_acton( state[inds] ) (fancy index notn)
      // out_state[inds] += state[inds]
      for(k=0; k<nActionIndices; k++)
        subState1._dataptr[k] = state_data[ vec_index_noop+_baseinds[k] ];
      _embedded_gate_crep->acton(&subState1, &subState2);
      for(k=0; k<nActionIndices; k++)
        outstate_data[ vec_index_noop+_baseinds[k] ] += subState2._dataptr[k];
        
      // increment b ~ itertools.product & update vec_index_noop = _np.dot(self.multipliers, b)
      for(i=nParts-1; i >= 0; i--) {
	if(b[i]+1 < _numBasisEls_noop_blankaction[i]) {
	  b[i] += 1; vec_index_noop += _noop_incrementers[i];
	  break;
	}
	else {
	  b[i] = 0;
	}
      }
      if(i < 0) break;  // if didn't break out of loop above, then can't
    }                   // increment anything - break while(true) loop.

    //act on other blocks trivially:
    if(_nBlocks > 1) { // if there's more than one basis "block" (in direct sum)
      offset = 0;
      for(i=0; i<_nBlocks; i++) {
	if(i != _iActiveBlock) {
	  for(j=0; j<_blocksizes[i]; j++) // identity op on this block
	    outstate_data[offset+j] = state_data[offset+j];
	}
	offset += _blocksizes[i];
      }
    }

    DEBUG(out_state->print("OUTPUT"));
    return out_state;
  }


  StateCRep* OpCRep_Embedded::adjoint_acton(StateCRep* state, StateCRep* out_state) {

    //Note: exactly the same as acton(...) but calls embedded gate's adjoint_acton
    DEBUG(std::cout << " Embedded adjoint_acton called!" << std::endl);
    DEBUG(state->print("INPUT"));
    INT i, j, k, vec_index_noop = 0;
    INT nParts = _nComponents;
    INT nActionIndices = _embeddedDim;
    INT offset;

    double* state_data = state->_dataptr;
    double* outstate_data = out_state->_dataptr;

    //zero-out output state initially
    for(i=0; i<_dim; i++) outstate_data[i] = 0.0;

    INT b[100]; // could alloc dynamically (LATER?)
    assert(nParts <= 100); // need to increase size of static arrays above
    for(i=0; i<nParts; i++) b[i] = 0;

    // Temporary states for acting the embedded gate on a subset of the whole
    StateCRep subState1(nActionIndices);
    StateCRep subState2(nActionIndices);
    
    while(true) {
      // Act with embedded gate on appropriate sub-space of state
      // out_state[ inds ] += embedded_gate_acton( state[inds] ) (fancy index notn)
      // out_state[inds] += state[inds]
      for(k=0; k<nActionIndices; k++)
	subState1._dataptr[k] = state_data[ vec_index_noop+_baseinds[k] ];
      _embedded_gate_crep->adjoint_acton(&subState1, &subState2);
      for(k=0; k<nActionIndices; k++)
	outstate_data[ vec_index_noop+_baseinds[k] ] += subState2._dataptr[k];
        
      // increment b ~ itertools.product & update vec_index_noop = _np.dot(self.multipliers, b)
      for(i=nParts-1; i >= 0; i--) {
	if(b[i]+1 < _numBasisEls_noop_blankaction[i]) {
	  b[i] += 1; vec_index_noop += _noop_incrementers[i];
	  break;
	}
	else {
	  b[i] = 0;
	}
      }
      if(i < 0) break;  // if didn't break out of loop above, then can't
    }                   // increment anything - break while(true) loop.

    //act on other blocks trivially:
    if(_nBlocks > 1) { // if there's more than one basis "block" (in direct sum)
      offset = 0;
      for(i=0; i<_nBlocks; i++) {
	if(i != _iActiveBlock) {
	  for(j=0; j<_blocksizes[i]; j++) // identity op on this block
	    outstate_data[offset+j] = state_data[offset+j];
	  offset += _blocksizes[i];
	}
      }
    }

    DEBUG(out_state->print("OUTPUT"));
    return out_state;
  }



  /****************************************************************************\
  |* OpCRep_Composed                                                      *|
  \****************************************************************************/
  OpCRep_Composed::OpCRep_Composed(std::vector<OpCRep*> factor_gate_creps, INT dim)
    :OpCRep(dim),_factor_gate_creps(factor_gate_creps)
  {
  }
  OpCRep_Composed::~OpCRep_Composed() { }

  void OpCRep_Composed::reinit_factor_op_creps(std::vector<OpCRep*> new_factor_gate_creps) {
    _factor_gate_creps.clear(); //removes all elements
    _factor_gate_creps.insert(_factor_gate_creps.end(),
			      new_factor_gate_creps.begin(),
			      new_factor_gate_creps.end());  //inserts contents of new array
  }
  
  StateCRep* OpCRep_Composed::acton(StateCRep* state, StateCRep* out_state) {

    DEBUG(std::cout << "Composed acton called!" << std::endl);
    DEBUG(state->print("INPUT"));
    std::size_t nfactors = _factor_gate_creps.size();
    StateCRep *tmp2, *tmp1 = out_state; //tmp1 already alloc'd
    StateCRep* t; // for swapping

    //if length is 0 just copy state --> outstate
    if(nfactors == 0) {
      out_state->copy_from(state);
      return out_state;
    }

    //Act with first gate: output in tmp1
    _factor_gate_creps[0]->acton(state, tmp1);
    
    if(nfactors > 1) {
      StateCRep temp_state(_dim); tmp2 = &temp_state;

      //Act with additional gates: tmp1 -> tmp2 then swap, so output in tmp1
      for(std::size_t i=1; i < nfactors; i++) {
	_factor_gate_creps[i]->acton(tmp1,tmp2);
	t = tmp1; tmp1 = tmp2; tmp2 = t;
      }
      
      //tmp1 holds the output state now; if tmp1 == out_state
      // we're in luck, otherwise we need to copy it into out_state.
      if(tmp1 != out_state) {
	out_state->copy_from(tmp1);
      }
    }
    DEBUG(out_state->print("OUTPUT"));
    return out_state;
  }

  StateCRep* OpCRep_Composed::adjoint_acton(StateCRep* state, StateCRep* out_state) {

    DEBUG(std::cout << "Composed adjoint_acton called!" << std::endl);
    DEBUG(state->print("INPUT"));
    std::size_t nfactors = _factor_gate_creps.size();
    StateCRep *tmp2, *tmp1 = out_state; //tmp1 already alloc'd
    StateCRep* t; // for swapping

    //Note: same as acton(...) but reverse order of gates and perform adjoint_acton
    //Act with last gate: output in tmp1
    _factor_gate_creps[nfactors-1]->adjoint_acton(state, tmp1);
    
    if(nfactors > 1) {
      StateCRep temp_state(_dim); tmp2 = &temp_state;

      //Act with additional gates: tmp1 -> tmp2 then swap, so output in tmp1
      for(INT i=nfactors-2; i >= 0; i--) {
	_factor_gate_creps[i]->adjoint_acton(tmp1,tmp2);
	t = tmp1; tmp1 = tmp2; tmp2 = t;
      }
      
      //tmp1 holds the output state now; if tmp1 == out_state
      // we're in luck, otherwise we need to copy it into out_state.
      if(tmp1 != out_state) {
	out_state->copy_from(tmp1);
      }
    }
    DEBUG(out_state->print("OUTPUT"));
    return out_state;
  }

  /****************************************************************************\
  |* OpCRep_Sum                                                           *|
  \****************************************************************************/
  OpCRep_Sum::OpCRep_Sum(std::vector<OpCRep*> factor_creps, double* factor_coefficients, INT dim)
    :OpCRep(dim),_factor_creps(factor_creps),_factor_coeffs(factor_coefficients)
  {
  }
  OpCRep_Sum::~OpCRep_Sum() { }
  
  StateCRep* OpCRep_Sum::acton(StateCRep* state, StateCRep* out_state) {

    DEBUG(std::cout << "Sum acton called!" << std::endl);
    DEBUG(state->print("INPUT"));
    std::size_t nfactors = _factor_creps.size();
    StateCRep temp_state(_dim);

    //zero-out output state
    for(INT k=0; k<_dim; k++)
      out_state->_dataptr[k] = 0.0;

    //if length is 0 just return "0" state --> outstate
    if(nfactors == 0) return out_state;

    //Act with factors and accumulate into out_state
    if(_factor_coeffs == NULL) {
      for(std::size_t i=0; i < nfactors; i++) {
        _factor_creps[i]->acton(state,&temp_state);
        for(INT k=0; k<_dim; k++)
          out_state->_dataptr[k] += temp_state._dataptr[k];
      }
    } else {
      for(std::size_t i=0; i < nfactors; i++) {
        _factor_creps[i]->acton(state,&temp_state);
        for(INT k=0; k<_dim; k++)
          out_state->_dataptr[k] += _factor_coeffs[i] * temp_state._dataptr[k];
      }
    }
    DEBUG(out_state->print("OUTPUT"));
    return out_state;
  }

  StateCRep* OpCRep_Sum::adjoint_acton(StateCRep* state, StateCRep* out_state) {

    //Note: same as acton(...) but perform adjoint_acton
    DEBUG(std::cout << "Sum adjoint_acton called!" << std::endl);
    DEBUG(state->print("INPUT"));
    std::size_t nfactors = _factor_creps.size();
    StateCRep temp_state(_dim);

    //zero-out output state
    for(INT k=0; k<_dim; k++)
      out_state->_dataptr[k] = 0.0;

    //if length is 0 just return "0" state --> outstate
    if(nfactors == 0) return out_state;

    //Act with factors and accumulate into out_state
    if(_factor_coeffs == NULL) {
      for(std::size_t i=0; i < nfactors; i++) {
        _factor_creps[i]->adjoint_acton(state,&temp_state);
        for(INT k=0; k<_dim; k++)
          out_state->_dataptr[k] += temp_state._dataptr[k];
      }
    } else {
      for(std::size_t i=0; i < nfactors; i++) {
        _factor_creps[i]->adjoint_acton(state,&temp_state);
        for(INT k=0; k<_dim; k++)
          out_state->_dataptr[k] += _factor_coeffs[i] * temp_state._dataptr[k];
      }
    }
    DEBUG(out_state->print("OUTPUT"));
    return out_state;
  }

  /****************************************************************************\
  |* OpCRep_Repeated                                                   *|
  \****************************************************************************/

  OpCRep_Repeated::OpCRep_Repeated(OpCRep* repeated_crep, INT num_repetitions, INT dim)
    :OpCRep(dim)
  {
    _repeated_crep = repeated_crep;
    _num_repetitions = num_repetitions;
  }

  OpCRep_Repeated::~OpCRep_Repeated() { }

  StateCRep* OpCRep_Repeated::acton(StateCRep* state, StateCRep* out_state) {

    DEBUG(std::cout << "Repeated acton called!" << std::endl);
    DEBUG(state->print("INPUT"));
    StateCRep *tmp2, *tmp1 = out_state; //tmp1 already alloc'd
    StateCRep* t; // for swapping

    //if num_repetitions is 0 just copy state --> outstate
    if(_num_repetitions == 0) {
      out_state->copy_from(state);
      return out_state;
    }

    //Act with first gate: output in tmp1
    _repeated_crep->acton(state, tmp1);
    
    if(_num_repetitions > 1) {
      StateCRep temp_state(_dim); tmp2 = &temp_state;

      //Act with additional gates: tmp1 -> tmp2 then swap, so output in tmp1
      for(INT i=1; i < _num_repetitions; i++) {
	_repeated_crep->acton(tmp1,tmp2);
	t = tmp1; tmp1 = tmp2; tmp2 = t;
      }
      
      //tmp1 holds the output state now; if tmp1 == out_state
      // we're in luck, otherwise we need to copy it into out_state.
      if(tmp1 != out_state) {
	out_state->copy_from(tmp1);
      }
    }
    DEBUG(out_state->print("OUTPUT"));
    return out_state;
  }

  StateCRep* OpCRep_Repeated::adjoint_acton(StateCRep* state, StateCRep* out_state) {

    DEBUG(std::cout << "Repeated adjoint_acton called!" << std::endl);
    DEBUG(state->print("INPUT"));
    StateCRep *tmp2, *tmp1 = out_state; //tmp1 already alloc'd
    StateCRep* t; // for swapping

    //Note: same as acton(...) but perform adjoint_acton
    //if num_repetitions is 0 just copy state --> outstate
    if(_num_repetitions == 0) {
      out_state->copy_from(state);
      return out_state;
    }

    //Act with first gate: output in tmp1
    _repeated_crep->adjoint_acton(state, tmp1);
    
    if(_num_repetitions > 1) {
      StateCRep temp_state(_dim); tmp2 = &temp_state;

      //Act with additional gates: tmp1 -> tmp2 then swap, so output in tmp1
      for(INT i=1; i < _num_repetitions; i++) {
	_repeated_crep->adjoint_acton(tmp1,tmp2);
	t = tmp1; tmp1 = tmp2; tmp2 = t;
      }
      
      //tmp1 holds the output state now; if tmp1 == out_state
      // we're in luck, otherwise we need to copy it into out_state.
      if(tmp1 != out_state) {
	out_state->copy_from(tmp1);
      }
    }
    DEBUG(out_state->print("OUTPUT"));
    return out_state;
  }



  /****************************************************************************\
  |* OpCRep_ExpErrorgen                                                      *|
  \****************************************************************************/

  OpCRep_ExpErrorgen::OpCRep_ExpErrorgen(OpCRep* errgen_rep,
			double mu, double eta, INT m_star, INT s, INT dim)
    :OpCRep(dim)
  {
    _errgen_rep = errgen_rep;
    
    _mu = mu;
    _eta = eta;
    _m_star = m_star;
    _s = s;
  }
      
  OpCRep_ExpErrorgen::~OpCRep_ExpErrorgen() { }

  StateCRep* OpCRep_ExpErrorgen::acton(StateCRep* state, StateCRep* out_state)
  {
    //INT i, j;
    StateCRep* init_state = new StateCRep(_dim);
    DEBUG(std::cout << "Lindblad acton called!" << std::endl);
    DEBUG(state->print("INPUT"));

    init_state->copy_from(state);
	  
    // BEGIN state = _fastcalc.custom_expm_multiply_simple_core(
    //     A.data, A.indptr, A.indices, state, mu, m_star, s, tol, eta)
    INT N = _dim; // = length(_A_indptr)-1
    if(_s == 0) { // nothing to do - just copy input to output
      out_state->copy_from(init_state);
      delete init_state;
      return out_state;
    }
    
    double* F = out_state->_dataptr; //will be output
    double* scratch = new double[N];
    double* B = new double[N];
    for(INT i=0; i<N; i++) B[i] = init_state->_dataptr[i];
    //can't do: double* B = init_state->_dataptr; because B is modified by expm_... below
        
    double tol = 1e-16; // 2^-53 (=Scipy default) -- TODO: make into an arg...
    
    // F = expm(A)*B
    expm_multiply_simple_core_rep(_errgen_rep, B, N, _mu,
				  _m_star, _s, tol, _eta, F, scratch);
    
    //cleanup what we allocated
    delete [] scratch;
    delete [] B;
    delete init_state;
    
    DEBUG(out_state->print("OUTPUT"));
    return out_state;
  }

  StateCRep* OpCRep_ExpErrorgen::adjoint_acton(StateCRep* state, StateCRep* out_state) {
    assert(false); //ajoint_acton not implemented yet for Lindblad gates (TODO LATER)
    return NULL; //to avoid compiler warning
  }


  /****************************************************************************\
  |* OpCRep_Sparse                                                        *|
  \****************************************************************************/

  OpCRep_Sparse::OpCRep_Sparse(double* A_data, INT* A_indices, INT* A_indptr,
				       INT nnz, INT dim)
    :OpCRep(dim)
  {
    _A_data = A_data;
    _A_indices = A_indices;
    _A_indptr = A_indptr;
    _A_nnz = nnz;
  }
      
  OpCRep_Sparse::~OpCRep_Sparse() { }

  StateCRep* OpCRep_Sparse::acton(StateCRep* state, StateCRep* out_state)
  {
    // csr_matvec: implements out_state = A * state
    int r,k;
    INT N = _dim; // = length(_A_indptr)-1
    double* indata = state->_dataptr;
    double* outdata = out_state->_dataptr;
    for(r=0; r<N; r++) {
      outdata[r] = 0;
      for(k=_A_indptr[r]; k<_A_indptr[r+1]; k++)
	    outdata[r] += _A_data[k] * indata[ _A_indices[k]];
    }
    return out_state;
  }

  StateCRep* OpCRep_Sparse::adjoint_acton(StateCRep* state, StateCRep* out_state) {
    //Need to take transpose of a CSR matrix then mult by vector...
    assert(false); //ajoint_acton not implemented yet for Sparse gates (TODO LATER)
    return NULL; //to avoid compiler warning
  }



  // ADDITIONAL HELPER FUNCTIONS
  
  void expm_multiply_simple_core_sparsemx(double* Adata, INT* Aindptr,
					  INT* Aindices, double* B,
					  INT N, double mu, INT m_star,
					  INT s, double tol, double eta,
					  double* F, double* scratch) {
    INT i;
    INT j;
    INT r;
    INT k;

    double a;
    double c1;
    double c2;
    double coeff;
    double normF;

    //DEBUG printout
    //std::cout << "expm_multiply_simple_core: N=" << N << ", s="<<s<<", eta="<<eta<<std::endl;
    //std::cout << "B=";
    //for(i=0; i<N; i++) std::cout << B[i] << " ";
    //std::cout << std::endl;
    //std::cout << "A_indptr=";
    //for(i=0; i<N+1; i++) std::cout << Aindptr[i] << " ";
    //std::cout << std::endl;
    //std::cout << "A_indices=";
    //for(i=0; i<Aindptr[N]; i++) std::cout << Aindices[i] << " ";
    //std::cout << std::endl;
    //std::cout << "A_data=";
    //for(i=0; i<Aindptr[N]; i++) std::cout << Adata[i] << " ";
    //std::cout << std::endl << std::endl;

    for(i=0; i<N; i++) F[i] = B[i];
    
    for(i=0; i<s; i++) {
      if(m_star > 0) { //added by EGN
	//c1 = vec_inf_norm(B) #_exact_inf_norm(B)
	c1 = 0.0;
	for(k=0; k<N; k++) {
	  a = (B[k] >= 0) ? B[k] : -B[k]; // abs(B[k])
	  if(a > c1) c1 = a;
	}
      }
      
      for(j=0; j<m_star; j++) {
	coeff = 1.0 / (s*(j+1)); // t == 1.0
	
	// B = coeff * A.dot(B)
        // inline csr_matvec: implements scratch = A * B
	for(r=0; r<N; r++) {
	  scratch[r] = 0;
	  for(k=Aindptr[r]; k<Aindptr[r+1]; k++)
	    scratch[r] += Adata[k] * B[ Aindices[k]];
	}

	// if(j % 3 == 0) {...   // every == 3 TODO: work on this?
	c2 = 0.0;
	normF = 0.0;
	for(k=0; k<N; k++) {
	  B[k] = coeff * scratch[k]; //finishes B = coeff * A.dot(B) 
	  F[k] += B[k]; //F += B
        
	  a = (B[k] >= 0)? B[k] : -B[k]; //abs(B[k])
	  if(a > c2) c2 = a; // c2 = vec_inf_norm(B) // _exact_inf_norm(B)
	  a = (F[k] >= 0)? F[k] : -F[k]; //abs(F[k])
	  if(a > normF) normF = a; // normF = vec_inf_norm(F) // _exact_inf_norm(F)
	}

        // print("Iter %d,%d of %d,%d: %g+%g=%g < %g?" % (i,j,s,m_star,c1,c2,c1+c2,tol*normF))
	if(c1 + c2 <= tol * normF) {
	  // print(" --> YES - break early at %d of %d" % (i+1,s))
	  break;
	}
	c1 = c2;
      }

      // F *= eta
      // B = F
      for(k=0; k<N; k++) {
	F[k] *= eta;
	B[k] = F[k];
      }
    }
    // output value is in F upon returning
  }


  void expm_multiply_simple_core_rep(OpCRep* A_rep, double* B,
				     INT N, double mu, INT m_star,
				     INT s, double tol, double eta,
				     double* F, double* scratch) {
    INT i;
    INT j;
    INT k;

    double a;
    double c1;
    double c2;
    double coeff;
    double normF;

    StateCRep B_st(B, N); // just a wrapper
    StateCRep scratch_st(scratch, N); // just a wrapper

    for(i=0; i<N; i++) F[i] = B[i];
    
    for(i=0; i<s; i++) {
      if(m_star > 0) { //added by EGN
          //c1 = vec_inf_norm(B) #_exact_inf_norm(B)
          c1 = 0.0;
          for(k=0; k<N; k++) {
              a = (B[k] >= 0) ? B[k] : -B[k]; // abs(B[k])
              if(a > c1) c1 = a;
          }
      }
      
      for(j=0; j<m_star; j++) {
          coeff = 1.0 / (s*(j+1)); // t == 1.0
	
          // B = coeff * A.dot(B)
          A_rep->acton(&B_st,&scratch_st); // scratch = A.dot(B)
    
          // if(j % 3 == 0) {...   // every == 3 TODO: work on this?
          c2 = 0.0;
          normF = 0.0;
          for(k=0; k<N; k++) {
              B[k] = coeff * scratch[k]; //finishes B = coeff * A.dot(B) 
              F[k] += B[k]; //F += B
              
              a = (B[k] >= 0)? B[k] : -B[k]; //abs(B[k])
              if(a > c2) c2 = a; // c2 = vec_inf_norm(B) // _exact_inf_norm(B)
              a = (F[k] >= 0)? F[k] : -F[k]; //abs(F[k])
              if(a > normF) normF = a; // normF = vec_inf_norm(F) // _exact_inf_norm(F)
          }
          
          // print("Iter %d,%d of %d,%d: %g+%g=%g < %g?" % (i,j,s,m_star,c1,c2,c1+c2,tol*normF))
          if(c1 + c2 <= tol * normF) {
              // print(" --> YES - break early at %d of %d" % (i+1,s))
              break;
          }
          c1 = c2;
      }
      
      // F *= eta
      // B = F
      for(k=0; k<N; k++) {
          F[k] *= eta;
          B[k] = F[k];
      }
    }
    // output value is in F upon returning
  }

}
