#define NULL 0

#include <iostream>
#include <complex>
#include <assert.h>
#include <algorithm>    // std::find
#include "fastreps.h"

//using namespace std::complex_literals;

//#define DEBUG(x) x
#define DEBUG(x) 

namespace CReps {

  // DENSE MATRIX (DM) propagation
  
  /****************************************************************************\
  |* DMStateCRep                                                              *|
  \****************************************************************************/
  DMStateCRep::DMStateCRep(int dim) {
    _dataptr = new double[dim];
    for(int i=0; i<dim; i++) _dataptr[i] = 0;
    _dim = dim;
    _ownmem = true;
  }
  
  DMStateCRep::DMStateCRep(double* data, int dim, bool copy=false) {
    //DEGUG std::cout << "DMStateCRep initialized w/dim = " << dim << std::endl;
    if(copy) {
      _dataptr = new double[dim];
      for(int i=0; i<dim; i++) {
	_dataptr[i] = data[i];
      }
    } else {
      _dataptr = data;
    }
    _dim = dim;
    _ownmem = copy;
  }

  DMStateCRep::~DMStateCRep() {
    if(_ownmem && _dataptr != NULL)
      delete [] _dataptr;
  }

  void DMStateCRep::print(const char* label) {
    std::cout << label << " = [";
    for(int i=0; i<_dim; i++) std::cout << _dataptr[i] << " ";
    std::cout << "]" << std::endl;
  }

  void DMStateCRep::copy_from(DMStateCRep* st) {
    assert(_dim == st->_dim);
    for(int i=0; i<_dim; i++)
      _dataptr[i] = st->_dataptr[i];
  }

  /****************************************************************************\
  |* DMEffectCRep                                                             *|
  \****************************************************************************/
  DMEffectCRep::DMEffectCRep(int dim) {
    _dim = dim;
  }
  DMEffectCRep::~DMEffectCRep() { }


  /****************************************************************************\
  |* DMEffectCRep_Dense                                                       *|
  \****************************************************************************/
  DMEffectCRep_Dense::DMEffectCRep_Dense(double* data, int dim)
    :DMEffectCRep(dim)
  {
    _dataptr = data;
  }

  DMEffectCRep_Dense::~DMEffectCRep_Dense() { }

  double DMEffectCRep_Dense::probability(DMStateCRep* state) {
    double ret = 0.0;
    for(int i=0; i< _dim; i++) {
      ret += _dataptr[i] * state->_dataptr[i];
    }
    return ret;
  }

  
  /****************************************************************************\
  |* DMEffectCRep_TensorProd                                                  *|
  \****************************************************************************/

  DMEffectCRep_TensorProd::DMEffectCRep_TensorProd(double* kron_array,
						   int* factordims, int nfactors,
						   int max_factor_dim, int dim) 
    :DMEffectCRep(dim)
  {
    _kron_array = kron_array;
    _max_factor_dim = max_factor_dim;
    _factordims = factordims;
    _nfactors = nfactors;
  }

  DMEffectCRep_TensorProd::~DMEffectCRep_TensorProd() { }
    
  double DMEffectCRep_TensorProd::probability(DMStateCRep* state) {
    //future: add scratch buffer as argument? or compute in place somehow?
    double ret = 0.0;
    double* scratch = new double[_dim];

    // BEGIN _fastcalc.fast_kron(scratch, _kron_array, _factordims)
    // - TODO: make this into seprate function & reuse in fastcals.pyx?
    int N = _dim;
    int i, j, k, sz, off, endoff, krow;
    double mult;
    double* array = _kron_array;
    int* arraysizes = _factordims;

    // Put last factor at end of scratch
    k = _nfactors-1;  //last factor
    off = N - arraysizes[k]; //offset into scratch
    krow = k * _max_factor_dim; //offset to k-th row of `array`
    for(i=0; i < arraysizes[k]; i++)
      scratch[off+i] = array[krow+i];
    sz = arraysizes[k];

    // Repeatedly scale&copy last "sz" elements of outputvec forward
    // (as many times as there are elements in the current factor array)
    //  - but multiply *in-place* the last "sz" elements.
    for(k=_nfactors-2; k >= 0; k--) { //for all but the last factor
      off = N - sz*arraysizes[k];
      endoff = N - sz;
      krow = k * _max_factor_dim; //offset to k-th row of `array`

      // For all but the final element of array[k,:],
      // mult&copy final sz elements of scratch into position
      for(j=0; j< arraysizes[k]-1; j++) {
	mult = array[krow+j];
	for(i=0; i<sz; i++) scratch[off+i] = mult*scratch[endoff+i];
        off += sz;
      }

      // Last element: in-place mult
      // assert(off == endoff)
      mult = array[krow + arraysizes[k]-1];
      for(i=0; i<sz; i++) scratch[endoff+i] *= mult;
      sz *= arraysizes[k];
    }
    //assert(sz == N)
    // END _fastcalc.fast_kron (output in `scratch`)

    for(int i=0; i< _dim; i++) {
      ret += scratch[i] * state->_dataptr[i];
    }
    delete [] scratch;
    return ret;
  }

  /****************************************************************************\
  |* DMGateCRep                                                               *|
  \****************************************************************************/

  DMGateCRep::DMGateCRep(int dim) {
    _dim = dim;
  }
  DMGateCRep::~DMGateCRep() { }


  /****************************************************************************\
  |* DMGateCRep_Dense                                                         *|
  \****************************************************************************/

  DMGateCRep_Dense::DMGateCRep_Dense(double* data, int dim)
    :DMGateCRep(dim)
  {
    _dataptr = data;
  }
  DMGateCRep_Dense::~DMGateCRep_Dense() { }

  DMStateCRep* DMGateCRep_Dense::acton(DMStateCRep* state,
				       DMStateCRep* outstate) {
    DEBUG(std::cout << "Dense acton called!" << std::endl);
    DEBUG(state->print("INPUT"));
    int k;
    for(int i=0; i< _dim; i++) {
      outstate->_dataptr[i] = 0.0;
      k = i*_dim; // "row" offset into _dataptr, so dataptr[k+j] ~= dataptr[i,j]
      for(int j=0; j< _dim; j++) {
	outstate->_dataptr[i] += _dataptr[k+j] * state->_dataptr[j];
      }
    }
    DEBUG(outstate->print("OUTPUT"));
    return outstate;
  }

  DMStateCRep* DMGateCRep_Dense::adjoint_acton(DMStateCRep* state,
					       DMStateCRep* outstate) {
    DEBUG(std::cout << "Dense adjoint_acton called!" << std::endl);
    DEBUG(state->print("INPUT"));
    for(int i=0; i< _dim; i++) {
      outstate->_dataptr[i] = 0.0;
      for(int j=0; j< _dim; j++) {
	outstate->_dataptr[i] += _dataptr[j*_dim+i] * state->_dataptr[j];
      }
    }
    DEBUG(outstate->print("OUTPUT"));
    return outstate;
  }


  /****************************************************************************\
  |* DMGateCRep_Embedded                                                      *|
  \****************************************************************************/

  DMGateCRep_Embedded::DMGateCRep_Embedded(DMGateCRep* embedded_gate_crep, int* noop_incrementers,
					   int* numBasisEls_noop_blankaction, int* baseinds, int* blocksizes,
					   int embeddedDim, int nComponents, int iActiveBlock, int nBlocks, int dim)
    :DMGateCRep(dim)
  {
    _embedded_gate_crep = embedded_gate_crep;
    _noop_incrementers = noop_incrementers;
    _numBasisEls_noop_blankaction = numBasisEls_noop_blankaction;
    _baseinds = baseinds;
    _blocksizes = blocksizes;
    _nComponents = nComponents;
    _embeddedDim = embeddedDim;
    _iActiveBlock = iActiveBlock;
    _nBlocks = nBlocks;
  }
  DMGateCRep_Embedded::~DMGateCRep_Embedded() { }
  
  DMStateCRep* DMGateCRep_Embedded::acton(DMStateCRep* state, DMStateCRep* out_state) {

    DEBUG(std::cout << "Emedded acton called!" << std::endl);
    DEBUG(state->print("INPUT"));
    //_fastcalc.embedded_fast_acton_sparse(self.embedded_gate.acton,
    //                                         output_state, state,
    //                                         self.noop_incrementers,
    //                                         self.numBasisEls_noop_blankaction,
    //                                         self.baseinds)
    int i, j, k, vec_index_noop = 0;
    int nParts = _nComponents;
    int nActionIndices = _embeddedDim;
    int offset;

    double* state_data = state->_dataptr;
    double* outstate_data = out_state->_dataptr;

    //zero-out output state initially
    for(i=0; i<_dim; i++) outstate_data[i] = 0.0;

    int b[100]; // could alloc dynamically (LATER?)
    assert(nParts <= 100); // need to increase size of static arrays above
    for(i=0; i<nParts; i++) b[i] = 0;

    // Temporary states for acting the embedded gate on a subset of the whole
    DMStateCRep subState1(nActionIndices);
    DMStateCRep subState2(nActionIndices);
    
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
	  offset += _blocksizes[i];
	}
      }
    }

    DEBUG(out_state->print("OUTPUT"));
    return out_state;
  }


  DMStateCRep* DMGateCRep_Embedded::adjoint_acton(DMStateCRep* state, DMStateCRep* out_state) {

    //Note: exactly the same as acton(...) but calls embedded gate's adjoint_acton
    DEBUG(std::cout << "Emedded adjoint_acton called!" << std::endl);
    DEBUG(state->print("INPUT"));
    int i, j, k, vec_index_noop = 0;
    int nParts = _nComponents;
    int nActionIndices = _embeddedDim;
    int offset;

    double* state_data = state->_dataptr;
    double* outstate_data = out_state->_dataptr;

    //zero-out output state initially
    for(i=0; i<_dim; i++) outstate_data[i] = 0.0;

    int b[100]; // could alloc dynamically (LATER?)
    assert(nParts <= 100); // need to increase size of static arrays above
    for(i=0; i<nParts; i++) b[i] = 0;

    // Temporary states for acting the embedded gate on a subset of the whole
    DMStateCRep subState1(nActionIndices);
    DMStateCRep subState2(nActionIndices);
    
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
  |* DMGateCRep_Composed                                                      *|
  \****************************************************************************/
  DMGateCRep_Composed::DMGateCRep_Composed(std::vector<DMGateCRep*> factor_gate_creps, int dim)
    :DMGateCRep(dim),_factor_gate_creps(factor_gate_creps)
  {
  }
  DMGateCRep_Composed::~DMGateCRep_Composed() { }
  
  DMStateCRep* DMGateCRep_Composed::acton(DMStateCRep* state, DMStateCRep* out_state) {

    DEBUG(std::cout << "Composed acton called!" << std::endl);
    DEBUG(state->print("INPUT"));
    std::size_t nfactors = _factor_gate_creps.size();
    DMStateCRep *tmp2, *tmp1 = out_state; //tmp1 already alloc'd
    DMStateCRep* t; // for swapping

    //Act with first gate: output in tmp1
    _factor_gate_creps[0]->acton(state, tmp1);
    
    if(nfactors > 1) {
      DMStateCRep temp_state(_dim); tmp2 = &temp_state;

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

  DMStateCRep* DMGateCRep_Composed::adjoint_acton(DMStateCRep* state, DMStateCRep* out_state) {

    DEBUG(std::cout << "Composed adjoint_acton called!" << std::endl);
    DEBUG(state->print("INPUT"));
    std::size_t nfactors = _factor_gate_creps.size();
    DMStateCRep *tmp2, *tmp1 = out_state; //tmp1 already alloc'd
    DMStateCRep* t; // for swapping

    //Note: same as acton(...) but reverse order of gates and perform adjoint_acton
    //Act with last gate: output in tmp1
    _factor_gate_creps[nfactors-1]->adjoint_acton(state, tmp1);
    
    if(nfactors > 1) {
      DMStateCRep temp_state(_dim); tmp2 = &temp_state;

      //Act with additional gates: tmp1 -> tmp2 then swap, so output in tmp1
      for(int i=nfactors-2; i >= 0; i--) {
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
  |* DMGateCRep_Lindblad                                                      *|
  \****************************************************************************/
  DMGateCRep_Lindblad::DMGateCRep_Lindblad(double* A_data, int* A_indices, int* A_indptr, int nnz,
			double mu, double eta, int m_star, int s, int dim,
			double* unitarypost_data, int* unitarypost_indices,
			int* unitarypost_indptr, int unitarypost_nnz)
    :DMGateCRep(dim)
  {
    _U_data = unitarypost_data;
    _U_indices = unitarypost_indices;
    _U_indptr = unitarypost_indptr;
    _U_nnz = unitarypost_nnz;
    
    _A_data = A_data;
    _A_indices = A_indices;
    _A_indptr = A_indptr;
    _A_nnz = nnz;
    
    _mu = mu;
    _eta = eta;
    _m_star = m_star;
    _s = s;
  }
      
  DMGateCRep_Lindblad::~DMGateCRep_Lindblad() { }

  DMStateCRep* DMGateCRep_Lindblad::acton(DMStateCRep* state, DMStateCRep* out_state)
  {
    int i, j;
    DMStateCRep* init_state = new DMStateCRep(_dim);
    DEBUG(std::cout << "Lindblad acton called!" << _U_nnz << ", " << _A_nnz << std::endl);
    DEBUG(state->print("INPUT"));

    if(_U_nnz > 0) {
      // do init_state = dot(unitary_postfactor, state) via *sparse* dotprod
      for(i=0; i<_dim; i++) {
	init_state->_dataptr[i] = 0;
	for(j=_U_indptr[i]; j< _U_indptr[i+1]; j++)
	  init_state->_dataptr[i] += _U_data[j] * state->_dataptr[_U_indices[j]];
      }
    } else {
      // in this case, we still need a *copy* of state, since
      // expm_multiply_simple_core modifies its `B` argument.
      init_state->copy_from(state);
    }
	  
    // BEGIN state = _fastcalc.custom_expm_multiply_simple_core(
    //     A.data, A.indptr, A.indices, state, mu, m_star, s, tol, eta)
    int N = _dim; // = length(_A_indptr)-1
    if(_s == 0) { // nothing to do - just copy input to output
      out_state->copy_from(init_state);
      delete init_state;
      return out_state;
    }

    double* F = out_state->_dataptr; //will be output
    double* scratch = new double[N];
    double* B = init_state->_dataptr;
    double tol = 1e-16; // 2^-53 (=Scipy default) -- TODO: make into an arg...

    // F = expm(A)*B
    expm_multiply_simple_core(_A_data, _A_indptr, _A_indices, B,
			      N, _mu, _m_star, _s, tol, _eta, F, scratch);

    //cleanup what we allocated
    delete [] scratch;
    delete init_state;

    DEBUG(out_state->print("OUTPUT"));
    return out_state;
  }

  DMStateCRep* DMGateCRep_Lindblad::adjoint_acton(DMStateCRep* state, DMStateCRep* out_state) {
    assert(false); //ajoint_acton not implemented yet for Lindblad gates (TODO LATER)
    return NULL; //to avoid compiler warning
  }

  
  // STATE VECTOR (SV) propagation

  /****************************************************************************\
  |* SVStateCRep                                                              *|
  \****************************************************************************/
  SVStateCRep::SVStateCRep(int dim) {
    _dataptr = new dcomplex[dim];
    for(int i=0; i<dim; i++) _dataptr[i] = 0;
    _dim = dim;
    _ownmem = true;
  }
  
  SVStateCRep::SVStateCRep(dcomplex* data, int dim, bool copy=false) {
    //DEGUG std::cout << "SVStateCRep initialized w/dim = " << dim << std::endl;
    if(copy) {
      _dataptr = new dcomplex[dim];
      for(int i=0; i<dim; i++) {
	_dataptr[i] = data[i];
      }
    } else {
      _dataptr = data;
    }
    _dim = dim;
    _ownmem = copy;
  }

  SVStateCRep::~SVStateCRep() {
    if(_ownmem && _dataptr != NULL)
      delete [] _dataptr;
  }

  void SVStateCRep::print(const char* label) {
    std::cout << label << " = [";
    for(int i=0; i<_dim; i++) std::cout << _dataptr[i] << " ";
    std::cout << "]" << std::endl;
  }

  void SVStateCRep::copy_from(SVStateCRep* st) {
    assert(_dim == st->_dim);
    for(int i=0; i<_dim; i++)
      _dataptr[i] = st->_dataptr[i];
  }

  /****************************************************************************\
  |* SVEffectCRep                                                             *|
  \****************************************************************************/
  SVEffectCRep::SVEffectCRep(int dim) {
    _dim = dim;
  }
  SVEffectCRep::~SVEffectCRep() { }


  /****************************************************************************\
  |* SVEffectCRep_Dense                                                       *|
  \****************************************************************************/
  SVEffectCRep_Dense::SVEffectCRep_Dense(dcomplex* data, int dim)
    :SVEffectCRep(dim)
  {
    _dataptr = data;
  }

  SVEffectCRep_Dense::~SVEffectCRep_Dense() { }

  double SVEffectCRep_Dense::probability(SVStateCRep* state) {
    return (double)pow(std::abs(amplitude(state)),2);
  }

  dcomplex SVEffectCRep_Dense::amplitude(SVStateCRep* state) {
    dcomplex ret = 0.0;
    for(int i=0; i< _dim; i++) {
      ret += std::conj(_dataptr[i]) * state->_dataptr[i];
    }
    return ret;
  }

  
  /****************************************************************************\
  |* SVEffectCRep_TensorProd                                                  *|
  \****************************************************************************/

  SVEffectCRep_TensorProd::SVEffectCRep_TensorProd(dcomplex* kron_array,
						   int* factordims, int nfactors,
						   int max_factor_dim, int dim) 
    :SVEffectCRep(dim)
  {
    _kron_array = kron_array;
    _max_factor_dim = max_factor_dim;
    _factordims = factordims;
    _nfactors = nfactors;
  }

  SVEffectCRep_TensorProd::~SVEffectCRep_TensorProd() { }

  double SVEffectCRep_TensorProd::probability(SVStateCRep* state) {
    return (double)pow(std::abs(amplitude(state)),2);
  }
  
  dcomplex SVEffectCRep_TensorProd::amplitude(SVStateCRep* state) {
    //future: add scratch buffer as argument? or compute in place somehow?
    dcomplex ret = 0.0;
    dcomplex* scratch = new dcomplex[_dim];

    // BEGIN _fastcalc.fast_kron(scratch, _kron_array, _factordims)
    // - TODO: make this into seprate function & reuse in fastcals.pyx?
    int N = _dim;
    int i, j, k, sz, off, endoff, krow;
    dcomplex mult;
    dcomplex* array = _kron_array;
    int* arraysizes = _factordims;

    // Put last factor at end of scratch
    k = _nfactors-1;  //last factor
    off = N - arraysizes[k]; //offset into scratch
    krow = k * _max_factor_dim; //offset to k-th row of `array`
    for(i=0; i < arraysizes[k]; i++)
      scratch[off+i] = array[krow+i];
    sz = arraysizes[k];

    // Repeatedly scale&copy last "sz" elements of outputvec forward
    // (as many times as there are elements in the current factor array)
    //  - but multiply *in-place* the last "sz" elements.
    for(k=_nfactors-2; k >= 0; k--) { //for all but the last factor
      off = N - sz*arraysizes[k];
      endoff = N - sz;
      krow = k * _max_factor_dim; //offset to k-th row of `array`

      // For all but the final element of array[k,:],
      // mult&copy final sz elements of scratch into position
      for(j=0; j< arraysizes[k]-1; j++) {
	mult = array[krow+j];
	for(i=0; i<sz; i++) scratch[off+i] = mult*scratch[endoff+i];
        off += sz;
      }

      // Last element: in-place mult
      // assert(off == endoff)
      mult = array[krow + arraysizes[k]-1];
      for(i=0; i<sz; i++) scratch[endoff+i] *= mult;
      sz *= arraysizes[k];
    }
    //assert(sz == N)
    // END _fastcalc.fast_kron (output in `scratch`)

    for(int i=0; i< _dim; i++) {
      ret += std::conj(scratch[i]) * state->_dataptr[i];
        //conjugate scratch b/c we assume _kron_array contains
        // info for building up the *column* "effect vector" E
        // s.t. amplitudes are computed as dot(E.T.conj,state_col_vec)
    }
    delete [] scratch;
    return ret;
  }

  /****************************************************************************\
  |* SVGateCRep                                                               *|
  \****************************************************************************/

  SVGateCRep::SVGateCRep(int dim) {
    _dim = dim;
  }
  SVGateCRep::~SVGateCRep() { }


  /****************************************************************************\
  |* SVGateCRep_Dense                                                         *|
  \****************************************************************************/

  SVGateCRep_Dense::SVGateCRep_Dense(dcomplex* data, int dim)
    :SVGateCRep(dim)
  {
    _dataptr = data;
  }
  SVGateCRep_Dense::~SVGateCRep_Dense() { }

  SVStateCRep* SVGateCRep_Dense::acton(SVStateCRep* state,
				       SVStateCRep* outstate) {
    DEBUG(std::cout << "Dense acton called!" << std::endl);
    DEBUG(state->print("INPUT"));
    int k;
    for(int i=0; i< _dim; i++) {
      outstate->_dataptr[i] = 0.0;
      k = i*_dim; // "row" offset into _dataptr, so dataptr[k+j] ~= dataptr[i,j]
      for(int j=0; j< _dim; j++) {
	outstate->_dataptr[i] += _dataptr[k+j] * state->_dataptr[j];
      }
    }
    DEBUG(outstate->print("OUTPUT"));
    return outstate;
  }

  SVStateCRep* SVGateCRep_Dense::adjoint_acton(SVStateCRep* state,
					       SVStateCRep* outstate) {
    DEBUG(std::cout << "Dense adjoint_acton called!" << std::endl);
    DEBUG(state->print("INPUT"));
    for(int i=0; i< _dim; i++) {
      outstate->_dataptr[i] = 0.0;
      for(int j=0; j< _dim; j++) {
	outstate->_dataptr[i] += std::conj(_dataptr[j*_dim+i]) * state->_dataptr[j];
      }
    }
    DEBUG(outstate->print("OUTPUT"));
    return outstate;
  }


  /****************************************************************************\
  |* SVGateCRep_Embedded                                                      *|
  \****************************************************************************/

  SVGateCRep_Embedded::SVGateCRep_Embedded(SVGateCRep* embedded_gate_crep, int* noop_incrementers,
					   int* numBasisEls_noop_blankaction, int* baseinds, int* blocksizes,
					   int embeddedDim, int nComponents, int iActiveBlock, int nBlocks, int dim)
    :SVGateCRep(dim)
  {
    _embedded_gate_crep = embedded_gate_crep;
    _noop_incrementers = noop_incrementers;
    _numBasisEls_noop_blankaction = numBasisEls_noop_blankaction;
    _baseinds = baseinds;
    _blocksizes = blocksizes;
    _nComponents = nComponents;
    _embeddedDim = embeddedDim;
    _iActiveBlock = iActiveBlock;
    _nBlocks = nBlocks;
  }
  SVGateCRep_Embedded::~SVGateCRep_Embedded() { }
  
  SVStateCRep* SVGateCRep_Embedded::acton(SVStateCRep* state, SVStateCRep* out_state) {

    DEBUG(std::cout << "Emedded acton called!" << std::endl);
    DEBUG(state->print("INPUT"));
    //_fastcalc.embedded_fast_acton_sparse(self.embedded_gate.acton,
    //                                         output_state, state,
    //                                         self.noop_incrementers,
    //                                         self.numBasisEls_noop_blankaction,
    //                                         self.baseinds)
    int i, j, k, vec_index_noop = 0;
    int nParts = _nComponents;
    int nActionIndices = _embeddedDim;
    int offset;

    dcomplex* state_data = state->_dataptr;
    dcomplex* outstate_data = out_state->_dataptr;

    //zero-out output state initially
    for(i=0; i<_dim; i++) outstate_data[i] = 0.0;

    int b[100]; // could alloc dynamically (LATER?)
    assert(nParts <= 100); // need to increase size of static arrays above
    for(i=0; i<nParts; i++) b[i] = 0;

    // Temporary states for acting the embedded gate on a subset of the whole
    SVStateCRep subState1(nActionIndices);
    SVStateCRep subState2(nActionIndices);
    
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
	  offset += _blocksizes[i];
	}
      }
    }

    DEBUG(out_state->print("OUTPUT"));
    return out_state;
  }


  SVStateCRep* SVGateCRep_Embedded::adjoint_acton(SVStateCRep* state, SVStateCRep* out_state) {

    //Note: exactly the same as acton(...) but calls embedded gate's adjoint_acton
    DEBUG(std::cout << "Emedded adjoint_acton called!" << std::endl);
    DEBUG(state->print("INPUT"));
    int i, j, k, vec_index_noop = 0;
    int nParts = _nComponents;
    int nActionIndices = _embeddedDim;
    int offset;

    dcomplex* state_data = state->_dataptr;
    dcomplex* outstate_data = out_state->_dataptr;

    //zero-out output state initially
    for(i=0; i<_dim; i++) outstate_data[i] = 0.0;

    int b[100]; // could alloc dynamically (LATER?)
    assert(nParts <= 100); // need to increase size of static arrays above
    for(i=0; i<nParts; i++) b[i] = 0;

    // Temporary states for acting the embedded gate on a subset of the whole
    SVStateCRep subState1(nActionIndices);
    SVStateCRep subState2(nActionIndices);
    
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
  |* SVGateCRep_Composed                                                      *|
  \****************************************************************************/
  SVGateCRep_Composed::SVGateCRep_Composed(std::vector<SVGateCRep*> factor_gate_creps, int dim)
    :SVGateCRep(dim),_factor_gate_creps(factor_gate_creps)
  {
  }
  SVGateCRep_Composed::~SVGateCRep_Composed() { }
  
  SVStateCRep* SVGateCRep_Composed::acton(SVStateCRep* state, SVStateCRep* out_state) {

    DEBUG(std::cout << "Composed acton called!" << std::endl);
    DEBUG(state->print("INPUT"));
    std::size_t nfactors = _factor_gate_creps.size();
    SVStateCRep *tmp2, *tmp1 = out_state; //tmp1 already alloc'd
    SVStateCRep* t; // for swapping

    //Act with first gate: output in tmp1
    _factor_gate_creps[0]->acton(state, tmp1);
    
    if(nfactors > 1) {
      SVStateCRep temp_state(_dim); tmp2 = &temp_state;

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

  SVStateCRep* SVGateCRep_Composed::adjoint_acton(SVStateCRep* state, SVStateCRep* out_state) {

    DEBUG(std::cout << "Composed adjoint_acton called!" << std::endl);
    DEBUG(state->print("INPUT"));
    std::size_t nfactors = _factor_gate_creps.size();
    SVStateCRep *tmp2, *tmp1 = out_state; //tmp1 already alloc'd
    SVStateCRep* t; // for swapping

    //Note: same as acton(...) but reverse order of gates and perform adjoint_acton
    //Act with last gate: output in tmp1
    _factor_gate_creps[nfactors-1]->adjoint_acton(state, tmp1);
    
    if(nfactors > 1) {
      SVStateCRep temp_state(_dim); tmp2 = &temp_state;

      //Act with additional gates: tmp1 -> tmp2 then swap, so output in tmp1
      for(int i=nfactors-2; i >= 0; i--) {
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


  // STABILIZER propagation

  /****************************************************************************\
  |* SBStateCRep                                                              *|
  \****************************************************************************/

  SBStateCRep::SBStateCRep(int* smatrix, int* pvectors, dcomplex* amps, int namps, int n) {
    _smatrix = smatrix;
    _pvectors = pvectors;
    _amps = amps;
    _namps = namps;
    _n = n;
    _2n = 2*n;
    _ownmem = false;
    rref(); // initializes _zblock_start
  }

  SBStateCRep::SBStateCRep(int namps, int n) {
    _n = n;
    _2n = 2*n;
    _namps = namps;
    _smatrix = new int[_2n*_2n];
    _pvectors = new int[_namps*_2n];
    _amps = new dcomplex[_namps];
    _ownmem = true;
    _zblock_start = -1;
  }
  
  SBStateCRep::~SBStateCRep() {
    if(_ownmem) {
      delete [] _smatrix;
      delete [] _pvectors;
      delete [] _amps;
    }
  }
  
  void SBStateCRep::push_view(std::vector<int>& view) {
    _view_filters.push_back(view);
  }
  
  void SBStateCRep::pop_view() {
    _view_filters.pop_back();
  }
    
  void SBStateCRep::clifford_update(int* smatrix, int* svector, dcomplex* Umx) {
    //vs = (_np.array([1,0],complex), _np.array([0,1],complex)) # (v0,v1)
    DEBUG(std::cout << "Clifford Update BEGIN" << std::endl);
    int i,k,ip;
    std::vector<std::vector<int> >::iterator it;
    std::vector<int>::iterator it2;

    std::vector<int> qubits(_n);
    for(i=0; i<_n; i++) qubits[i] = i; // start with all qubits being acted on
    for(it=_view_filters.begin(); it != _view_filters.end(); ++it) {
      std::vector<int>& qfilter = *it;
      std::vector<int> new_qubits(qfilter.size());
      for(i=0; i < (int)qfilter.size(); i++)
	new_qubits[i] = qubits[ qfilter[i] ]; // apply each filter
      qubits.resize( new_qubits.size() );
      for(i=0; i < (int)qfilter.size(); i++)
	qubits[i] = new_qubits[i]; //copy new_qubits -> qubits (maybe a faster way?)
    }

    int nQ = qubits.size(); //number of qubits being acted on (<= n in general)
    std::vector<std::vector<int> > sampled_states(_namps);
    std::vector<std::vector<dcomplex> > sampled_amplitudes(_namps);

    int action_size = pow(2,qubits.size());
    std::vector<dcomplex> outstate(action_size);

    // Step1: Update global amplitudes - Part A
    DEBUG(std::cout << "UPDATE GLOBAL AMPS: zstart=" << _zblock_start << std::endl);
    for(ip=0; ip<_namps; ip++) {
      sampled_states[ip].resize(_n);
      sampled_amplitudes[ip].resize( action_size );
      canonical_amplitudes_sample(ip,qubits, sampled_states[ip], sampled_amplitudes[ip]);
    }

    // Step2: Apply clifford to stabilizer reps in _smatrix, _pvectors
    DEBUG(std::cout << "APPLY CLIFFORD TO FRAME" << std::endl);
    apply_clifford_to_frame(smatrix, svector, qubits);
    rref();

    //DEBUG!!!
    //std::cout << "S = ";
    //for(i=0; i<_2n*_2n; i++) std::cout << _smatrix[i] << " ";
    //std::cout << std::endl;
    //std::cout << "PS = ";
    //for(i=0; i<_namps*_2n; i++) std::cout << _pvectors[i] << " ";
    //std::cout << std::endl;

    // Step3: Update global amplitudes - Part B
    for(ip=0; ip<_namps; ip++) {
      const std::vector<int> & base_state = sampled_states[ip];
      const std::vector<dcomplex> & ampls = sampled_amplitudes[ip];

      // APPLYING U to instate = ampls, i.e. outstate = _np.dot(Umx,ampls)
      DEBUG(std::cout << "APPLYING U to instate = ");
      DEBUG(for(i=0; i<action_size; i++) std::cout << ampls[i] << " ");
      DEBUG(std::cout << std::endl);
      for(i=0; i<action_size; i++) {
	outstate[i] = 0.0;
	for(k=0; k<action_size; k++)
	  outstate[i] += Umx[i*action_size+k] * ampls[k]; // state-vector propagation
      }
      DEBUG(std::cout << "outstate = ");
      DEBUG(for(i=0; i<action_size; i++) std::cout << outstate[i] << " ");
      DEBUG(std::cout << std::endl);


      //Look for nonzero output component and figure out how
      // phase *actually* changed as per state-vector propagation, then
      // update self.a (global amplitudes) to account for this.
      for(k=0; k<action_size; k++) {
	dcomplex comp = outstate[k]; // component of output state
	if(std::abs(comp) > 1e-6) {
	  std::vector<int> zvals(base_state);
	  std::vector<int> k_zvals(nQ);
	  for(i=0; i<nQ; i++) k_zvals[i] = int( (k >> (nQ-1-i)) & 1);  // hack to extract binary(k)
	  for(i=0; i<nQ; i++) zvals[qubits[i]] = k_zvals[i];
	  
	  DEBUG(std::cout << "GETTING CANONICAL AMPLITUDE for B' = " << zvals[0] << " actual=" << comp << std::endl);
	  dcomplex camp = canonical_amplitude_of_target(ip, zvals);
	  DEBUG(std::cout << "GOT CANONICAL AMPLITUDE =" << camp << " updating global amp w/" << comp/camp << std::endl);
	  assert(std::abs(camp) > 1e-6); // Canonical amplitude zero when actual isn't!!
	  _amps[ip] *= comp / camp; // "what we want" / "what stab. frame gives"
	    // this essentially updates a "global phase adjustment factor"
	  break; // move on to next stabilizer state & global amplitude
	}
      }
      if(k == action_size)  assert(false); // Outstate was completely zero!
                                           // (this shouldn't happen if Umx is unitary!)
    }
    DEBUG(std::cout << "Clifford Update END" << std::endl);
  }

  
  dcomplex SBStateCRep::extract_amplitude(std::vector<int>& zvals) {
    dcomplex ampl = 0;
    for(int ip=0; ip < _namps; ip++) {
      ampl += _amps[ip] * canonical_amplitude_of_target(ip, zvals);
    }
    return ampl;
  }

  double SBStateCRep::measurement_probability(std::vector<int>& zvals) {
    // Could make this faster in the future by using anticommutator?
    // - maybe could use a _canonical_probability for each ip that is
    //   essentially the 'stabilizer_measurement_prob' fn? -- but need to
    //   preserve *amplitudes* upon measuring & getting output state, which
    //   isn't quite done in the 'pauli_z_meaurement' function.
    dcomplex amp = extract_amplitude(zvals);
    return pow(std::abs(amp),2);
    // Note: don't currently implement the 2nd method using the anticomm in C++... (maybe later)
  }

  void SBStateCRep::copy_from(SBStateCRep* other) {
    assert(_n == other->_n && _namps == other->_namps); //make sure we don't need to allocate anything
    int i;
    for(i=0;i<_2n*_2n;i++) _smatrix[i] = other->_smatrix[i];
    for(i=0;i<_namps*_2n;i++) _pvectors[i] = other->_pvectors[i];
    for(i=0;i<_namps;i++) _amps[i] = other->_amps[i];
    _zblock_start = other->_zblock_start;
    _view_filters.clear();
    for(i=0; i<(int)other->_view_filters.size(); i++)
      _view_filters.push_back( other->_view_filters[i] );
  }
    

  int SBStateCRep::udot1(int i, int j) {
    // dot(smatrix[:,i].T, U, smatrix[:,j])
    int ret = 0;
    for(int k=0; k < _n; k++)
      ret += _smatrix[(k+_n)*_2n+i] * _smatrix[k*_2n+j];
    return ret;
  }

  void SBStateCRep::udot2(int* out, int* smatrix1, int* smatrix2) {
    // out = dot(smatrix1.T, U, smatrix2)
    int tmp;
    for(int i=0; i<_2n; i++) {
      for(int j=0; j<_2n; j++) {
	tmp = 0;
	for(int k=0; k < _n; k++)
	  tmp += smatrix1[(k+_n)*_2n+i] * smatrix2[k*_2n+j];
	out[i*_2n+j] = tmp;
      }
    }
  }
  
  void SBStateCRep::colsum(int i, int j) {
    int k,row;
    int* pvec;
    int* s = _smatrix;
    for(int p=0; p<_namps; p++) {
      pvec = &_pvectors[ _2n*p ]; // p-th vector
      pvec[i] = (pvec[i] + pvec[j] + 2* udot1(i,j)) % 4;
      for(k=0; k<_n; k++) {
	row = k*_2n;
	s[row+i] = s[row+j] ^ s[row+i];
	row = (k+_n)*_2n;
	s[row+i] = s[row+j] ^ s[row+i];
      }
    }
  }
  
  void SBStateCRep::colswap(int i, int j) {
    int tmp;
    int* pvec;
    for(int k=0; k<_2n; k++) {
      tmp = _smatrix[k*_2n+i];
      _smatrix[k*_2n+i] = _smatrix[k*_2n+j];
      _smatrix[k*_2n+j] = tmp;
    }
    for(int p=0; p<_namps; p++) {
      pvec = &_pvectors[ _2n*p ]; // p-th vector
      tmp = pvec[i];
      pvec[i] = pvec[j];
      pvec[j] = tmp;
    }
  }
  
  void SBStateCRep::rref() {
    //Pass1: form X-block (of *columns*)
    int i=0, j,k,m; // current *column* (to match ref, but our rep is transposed!)
    for(j=0; j<_n; j++) { // current *row*
      for(k=i; k<_n; k++) { // set k = column with X/Y in j-th position
	if(_smatrix[j*_2n+k] == 1) break; // X or Y check
      }
      if(k == _n) continue; // no k found => next column
      colswap(i,k);
      colswap(i+_n,k+_n); // mirror in antistabilizer
      for(m=0; m<_n; m++) {
	if(m != i && _smatrix[j*_2n+m] == 1) { // j-th literal of column m(!=i) is X/Y
	  colsum(m,i);
	  colsum(i+_n,m+_n); // reverse-mirror in antistabilizer (preserves relations)
	}
      }
      i++;
    }
    _zblock_start = i; // first column of Z-block

    //Pass2: form Z-block (of *columns*)
    for(j=0; j<_n; j++) { // current *row*
      for(k=i; k<_n; k++) { // set k = column with Z in j-th position
	if(_smatrix[j*_2n+k] == 0 && _smatrix[(j+_n)*_2n+k] == 1) break; // Z check
      }
      if(k == _n) continue; // no k found => next column
      colswap(i,k);
      colswap(i+_n,k+_n); // mirror in antistabilizer
      for(m=0; m<_n; m++) {
	if(m != i && _smatrix[(j+_n)*_2n+m] == 1) { // j-th literal of column m(!=i) is Z/Y
	  colsum(m,i);
	  colsum(i+_n,m+_n); // reverse-mirror in antistabilizer (preserves relations)
	}
      }
      i++;
    }
  }


  //result = _np.array(zvals_to_acton,int);
  dcomplex SBStateCRep::apply_xgen(int igen, int pgen, std::vector<int>& zvals_to_acton,
				   dcomplex ampl, std::vector<int>& result) {

    dcomplex new_amp = (pgen/2 == 1) ? -ampl : ampl;
    //DEBUG std::cout << "new_amp = "<<new_amp<<std::endl;
    for(std::size_t i=0; i<result.size(); i++)
      result[i] = zvals_to_acton[i];
    
    for(int j=0; j<_n; j++) { // for each element (literal) in generator
      if(_smatrix[j*_2n+igen] == 1) { // # X or Y
	result[j] = 1-result[j]; //flip!
	// X => a' == a constraint on new/old amplitudes, so nothing to do
	// Y => a' == i*a constraint, so:
	if(_smatrix[(j+_n)*_2n + igen] == 1) { // Y
	  if(result[j] == 1) new_amp *= dcomplex(0,1.0); //+1i; // |0> -> i|1> (but "== 1" b/c result is already flipped)
	  else new_amp *= dcomplex(0,-1.0); //-1i;              // |1> -> -i|0>
	  //DEBUG std::cout << "new_amp2 = "<<new_amp<<std::endl;
	}
      }
      else if(_smatrix[(j+_n)*_2n + igen] == 1) { // Z
	// Z => a' == -a constraint if basis[j] == |1> (otherwise a == a)
	if(result[j] == 1) new_amp *= -1;
	//DEBUG std::cout << "new_amp3 = "<<new_amp<<std::endl;
      }
    }
    //DEBUG std::cout << "new_amp4 = "<<new_amp<<std::endl;
    return new_amp;
  }
        
  dcomplex SBStateCRep::get_target_ampl(std::vector<int>& tgt, std::vector<int>& anchor, dcomplex anchor_amp, int ip) {
    // requires just a single pass through X-block
    std::vector<int> zvals(anchor);
    dcomplex amp = anchor_amp; //start with anchor state
    int i,j,k, lead = -1;
    DEBUG(std::cout << "BEGIN get_target_ampl" << std::endl);
    
    for(i=0; i<_zblock_start; i++) { // index of current generator
      int gen_p = _pvectors[ip*_2n + i]; //phase of generator
      gen_p = (gen_p + 3*udot1(i,i)) % 4;  //counts number of Y's => -i's
      assert(gen_p == 0 || gen_p == 2); //Logic error: phase should be +/- only!
      
      // Get leading flipped qubit (lowest # qubit which will flip when we apply this)
      for(j=0; j<_n; j++) {
	if(_smatrix[j*_2n+i] == 1) { // check for X/Y literal in qubit pos j
	  assert(j > lead); // lead should be strictly increasing as we iterate due to rref structure
	  lead = j; break;
	}
      }
      if(j == _n) assert(false); //Should always break loop!

      DEBUG(std::cout << "GETTGT iter " << i << " lead=" << lead << " genp=" << gen_p << " amp=" << amp << std::endl);

      //Check whether we should apply this generator to zvals
      if(zvals[lead] != tgt[lead]) {
	// then applying this generator is productive - do it!
	DEBUG(std::cout << "Applying XGEN amp=" << amp << std::endl);
	std::vector<int> zvals_copy(zvals);
	amp = apply_xgen(i, gen_p, zvals, amp, zvals_copy);
	zvals = zvals_copy; //will this work (copy)?
	DEBUG(std::cout << "Resulting amp = " << amp << " zvals=");
        DEBUG(for(std::size_t z=0; z<zvals.size(); z++) std::cout << zvals[z]);
	DEBUG(std::cout << std::endl);
                    
	// Check if we've found target
	for(k=0; k<_n; k++) {
	  if(zvals[k] != tgt[k]) break;
	}
	if(k == _n) {
	  DEBUG(std::cout << "FOUND!" << std::endl);
	  return amp; // no break => (zvals == tgt)
	}
      }
    }
    assert(false); //Falied to find amplitude of target! (tgt)
    return 0; // just to avoid warning
  }
  
  dcomplex SBStateCRep::canonical_amplitude_of_target(int ip, std::vector<int>& target) {
    rref(); // ensure we're in reduced row echelon form
        
    // Stage1: go through Z-block columns and find an "anchor" - the first
    // basis state that is allowed given the Z-block parity constraints.
    // (In Z-block, cols can have only Z,I literals)
    int i,j;
    DEBUG(std::cout << "CanonicalAmps STAGE1: zblock_start = " << _zblock_start << std::endl);
    std::vector<int> anchor(_n); // "anchor" basis state (zvals), which gets amplitude 1.0 by definition
    for(i=0; i<_n; i++) anchor[i] = 0;
    
    int lead = _n;
    for(i=_n-1; i >= _zblock_start; i--) { //index of current generator
      int gen_p = _pvectors[ip*_2n + i]; //phase of generator
      gen_p = (gen_p + 3*udot1(i,i)) % 4;  //counts number of Y's => -i's
      assert(gen_p == 0 || gen_p == 2);
      std::cout << "STARTING LOOP!" << std::endl;
            
      // get positions of Zs
      std::vector<int> zpos;
      for(j=0; j<_n; j++) {
	if(_smatrix[(j+_n)*_2n+i] == 1) zpos.push_back(j);
      }

      // set values of anchor between zpos[0] and lead
      // (between current leading-Z position and the last iteration's,
      //  which marks the point at which anchor has been initialized to)
      int fixed1s = 0; // relevant number of 1s fixed by the already-initialized part of 'anchor'
      int target1s = 0; // number of 1s in target state, which we want to check for Z-block compatibility
      std::vector<int> zpos_to_fill;
      std::vector<int>::iterator it;
      for(it=zpos.begin(); it!=zpos.end(); ++it) {
	j = *it;
	if(j >= lead) {
	  if(anchor[j] == 1) fixed1s += 1;
	}
	else zpos_to_fill.push_back(j);

	if(target[j] == 1) target1s += 1;
      }
	
      assert(zpos_to_fill.size() > 0); // structure of rref Z-block should ensure this
      int parity = gen_p/2;
      int eff_parity = (parity - (fixed1s % 2)) % 2; // effective parity for zpos_to_fill

      DEBUG(std::cout << "  Current gen = "<<i<<" phase = "<<gen_p<<" zpos="<<zpos.size()<<" fixed1s="<<fixed1s<<" tofill="<<zpos_to_fill.size()<<" eff_parity="<<eff_parity<<"lead="<<lead << std::endl);
      DEBUG(std::cout << "   -anchor: ");
      DEBUG(for(int dbi=0; dbi<_n; dbi++) std::cout << anchor[dbi] << "  ");
      
      if((target1s % 2) != parity)
	return dcomplex(0.0); // target fails this parity check -> it's amplitude == 0 (OK)

      if(eff_parity == 0) { // even parity - fill with all 0s
	// BUT already initalized to 0s, so don't need to do anything for anchor
      }
      else { // odd parity (= 1 or -1) - fill with all 0s except final zpos_to_fill = 1
	anchor[zpos_to_fill[zpos_to_fill.size()-1]] = 1; // BUT just need to fill in the final 1
      }
      lead = zpos_to_fill[0]; // update the leading-Z index
      DEBUG(std::cout << "   ==> ");
      DEBUG(for(int dbi=0; dbi<_n; dbi++) std::cout << anchor[dbi] << "  ");
      DEBUG(std::cout << std::endl);
    }
    
    //Set anchor amplitude to appropriate 1.0/sqrt(2)^s 
    // (by definition - serves as a reference pt)
    // Note: 's' equals the minimum number of generators that are *different*
    // between this state and the basis state we're extracting and ampl for.
    // Since any/all comp. basis state generators can form all and only the
    // Z-literal only (Z-block) generators 's' is simplly the number of
    // X-block generators (= self.zblock_start).
    int s = _zblock_start;
    dcomplex anchor_amp = 1/(pow(sqrt(2.0),s));
    
    //STAGE 2b - for sampling a set
    // Check exit conditions
    DEBUG(std::cout << "CanonicalAmps STAGE2" << std::endl);
    for(i=0; i<_n; i++) {
      if(anchor[i] != target[i]) break;
    }
    if(i == _n) return anchor_amp; // no break => (anchor == target)
    
    // Stage2: move through X-block processing existing amplitudes
    // (or processing only to move toward a target state?)
    DEBUG(std::cout << "Getting target ampl" << std::endl);
    return get_target_ampl(target,anchor,anchor_amp,ip);
  }
    
  void SBStateCRep::canonical_amplitudes_sample(int ip, std::vector<int> qs_to_sample,
						std::vector<int>& anchor, std::vector<dcomplex>& amp_samples) {
    rref(); // ensure we're in reduced row echelon form

    int i,j,k;
    int remaining = pow(2,qs_to_sample.size()); //number we still need to find
    assert(amp_samples.size() == remaining);
    for(i=0; i<remaining; i++) amp_samples[i]= std::nan("empty slot");
    // what we'll eventually return - holds amplitudes of all
    //  variations of qs_to_sample starting from anchor.
        
    // Stage1: go through Z-block columns and find an "anchor" - the first
    // basis state that is allowed given the Z-block parity constraints.
    // (In Z-block, cols can have only Z,I literals)
    assert(anchor.size() == _n); // "anchor" basis state (zvals), which gets amplitude 1.0 by definition
    for(i=0; i<_n; i++) anchor[i] = 0;
    
    int lead = _n;
    for(i=_n-1; i >= _zblock_start; i--) { //index of current generator
      int gen_p = _pvectors[ip*_2n + i]; //phase of generator
      gen_p = (gen_p + 3*udot1(i,i)) % 4;  //counts number of Y's => -i's
      assert(gen_p == 0 || gen_p == 2);
            
      // get positions of Zs
      std::vector<int> zpos;
      for(j=0; j<_n; j++) {
	if(_smatrix[(j+_n)*_2n+i] == 1) zpos.push_back(j);
      }

      // set values of anchor between zpos[0] and lead
      // (between current leading-Z position and the last iteration's,
      //  which marks the point at which anchor has been initialized to)
      int fixed1s = 0; // relevant number of 1s fixed by the already-initialized part of 'anchor'
      std::vector<int> zpos_to_fill;
      std::vector<int>::iterator it;
      for(it=zpos.begin(); it!=zpos.end(); ++it) {
	j = *it;
	if(j >= lead) {
	  if(anchor[j] == 1) fixed1s += 1;
	}
	else zpos_to_fill.push_back(j);
      }
	
      assert(zpos_to_fill.size() > 0); // structure of rref Z-block should ensure this
      int parity = gen_p/2;
      int eff_parity = (parity - (fixed1s % 2)) % 2; // effective parity for zpos_to_fill

      if(eff_parity == 0) { // even parity - fill with all 0s
	// BUT already initalized to 0s, so don't need to do anything for anchor
      }
      else { // odd parity (= 1 or -1) - fill with all 0s except final zpos_to_fill = 1
	anchor[zpos_to_fill[zpos_to_fill.size()-1]] = 1; // BUT just need to fill in the final 1
      }
      lead = zpos_to_fill[0]; // update the leading-Z index
    }
    
    //Set anchor amplitude to appropriate 1.0/sqrt(2)^s 
    // (by definition - serves as a reference pt)
    // Note: 's' equals the minimum number of generators that are *different*
    // between this state and the basis state we're extracting and ampl for.
    // Since any/all comp. basis state generators can form all and only the
    // Z-literal only (Z-block) generators 's' is simplly the number of
    // X-block generators (= self.zblock_start).
    int s = _zblock_start;
    dcomplex anchor_amp = 1/(pow(sqrt(2.0),s));
    
    remaining -= 1;
    int nk = qs_to_sample.size();
    int anchor_indx = 0;
    for(k=0; k<nk; k++) anchor_indx += anchor[qs_to_sample[k]]*pow(2,(nk-1-k));
    amp_samples[ anchor_indx ] = anchor_amp;
    
    
    //STAGE 2b - for sampling a set
    
    //If we're trying to sample a set, check if any of the amplitudes
    // we're looking for are zero by the Z-block checks.  That is,
    // consider whether anchor with qs_to_sample indices updated
    // passes or fails each check
    for(i=_n-1; i >= _zblock_start; i--) { // index of current generator
      int gen_p = _pvectors[ip*_2n + i]; //phase of generator
      gen_p = (gen_p + 3*udot1(i,i)) % 4;  //counts number of Y's => -i's
      
      std::vector<int> zpos;
      for(j=0; j<_n; j++) {
	if(_smatrix[(j+_n)*_2n+i] == 1) zpos.push_back(j);
      }
      
      std::vector<int> inds;
      std::vector<int>::iterator it, it2;
      int fixed1s = 0; // number of 1s in target state, which we want to check for Z-block compatibility
      for(it=zpos.begin(); it!=zpos.end(); ++it) {
	j = *it;
	it2 = std::find(qs_to_sample.begin(),qs_to_sample.end(),j);
	if(it2 != qs_to_sample.end()) { // if j in qs_to_sample
	  int jpos = it2 - qs_to_sample.begin();
	  inds.push_back( jpos ); // "sample" indices in parity check
	}
	else if(anchor[j] == 1) {
	  fixed1s += 1;
	}
      }
      if(inds.size() > 0) {
	int parity = (gen_p/2 - (fixed1s % 2)) % 2; // effective parity
	int* b = new int[qs_to_sample.size()]; //els are just 0 or 1
	int bi;
	for(bi=0; bi<(int)qs_to_sample.size(); bi++) b[bi] = 0;
	k = 0;
	while(true) {
	  // tup == b
	  int tup_parity = 0;
	  for(int kk=0; kk<(int)inds.size(); kk++) tup_parity += b[inds[kk]];
	  if(tup_parity != parity) { // parity among inds is NOT allowed => set ampl to zero
	    if(std::isnan(amp_samples[k].real())) remaining -= 1; //need NAN check here -- TODO replace -1 sentinels
	    amp_samples[k] = 0.0;
	  }
	  
	  k++; // increment k
	  
	  // increment b ~ itertools.product
	  for(bi=qs_to_sample.size()-1; bi >= 0; bi--) {
	    if(b[bi]+1 < 2) { // 2 == number of indices, i.e. [0,1]
	      b[bi] += 1;
	      break;
	    }
	    else {
	      b[bi] = 0;
	    }
	  }
	  if(bi < 0) break;  // if didn't break out of loop above, then can't
	}                    // increment anything - break while(true) loop.
      }
    }
    
    // Check exit conditions
    if(remaining == 0) return;
    
    // Stage2: move through X-block processing existing amplitudes
    // (or processing only to move toward a target state?)

    std::vector<int> target(anchor);
    int* b = new int[qs_to_sample.size()]; //els are just 0 or 1
    int bi;
    for(bi=0; bi<(int)qs_to_sample.size(); bi++) b[bi] = 0;
    k = 0;
    while(true) {
      // tup == b
      if(std::isnan(amp_samples[k].real())) {
	for(int kk=0; kk<(int)qs_to_sample.size(); kk++)
	  target[qs_to_sample[kk]] = b[kk];
	amp_samples[k] = get_target_ampl(target,anchor,anchor_amp,ip);
      }
      
      k++; // increment k
      
      // increment b ~ itertools.product
      for(bi=qs_to_sample.size()-1; bi >= 0; bi--) {
	if(b[bi]+1 < 2) { // 2 == number of indices, i.e. [0,1]
	  b[bi] += 1;
	  break;
	}
	else {
	  b[bi] = 0;
	}
      }
      if(bi < 0) break;  // if didn't break out of loop above, then can't
    }
    return;
  }

  void SBStateCRep::apply_clifford_to_frame(int* s, int* p, std::vector<int> qubit_filter) {
    //for now, just embed s,p inside full-size s,p: (TODO: make this function more efficient!)
    int* full_s = new int[_2n*_2n];
    int* full_p = new int[_2n];

    // Embed s,p inside full_s and full_p
    int i,j,ne = qubit_filter.size();
    int two_ne = 2*ne;
    for(i=0; i<_2n; i++) {
      for(j=0; j<_2n; j++) full_s[i] = (i==j) ? 1 : 0; // full_s == identity
    }
    for(i=0; i<_2n; i++) full_p[i] = 0; // full_p = zero
    
    for(int ii=0; ii<ne; ii++) {
      i = qubit_filter[ii];
      full_p[i] = p[ii];
      full_p[i+_n] = p[ii+ne];
      
      for(int jj=0; jj<ne; jj++) {
	j = qubit_filter[jj];
	full_s[i*_2n+j] = s[ii*two_ne+jj];
	full_s[(i+_n)*_2n+j] = s[(ii+ne)*two_ne+jj];
	full_s[i*_2n+(j+_n)] = s[ii*two_ne+(jj+ne)];
	full_s[(i+_n)*_2n+(j+_n)] = s[(ii+ne)*two_ne+(jj+ne)];
      }
    }

    apply_clifford_to_frame(full_s, full_p);

    delete [] full_s;
    delete [] full_p;
  }
    
  
  void SBStateCRep::apply_clifford_to_frame(int* s, int* p) {
    int i,j,k,tmp;
    
    // Below we calculate the s and p for the output state using the formulas from
    // Hostens and De Moor PRA 71, 042315 (2005).

    // out_s = _mtx.dotmod2(s,self.s)
    int* out_s = new int[_2n*_2n];
    //if(qubit_filter.size() == 0) {
    for(i=0; i<_2n; i++) {
      for(j=0; j<_2n; j++) {
	tmp = 0;
	for(k=0; k<_2n; k++) // row(s, i) * col(_smatrix,j)
	  tmp += s[i*_2n+k] * _smatrix[k*_2n+j];
	out_s[i*_2n+j] = tmp % 2; // all els are mod(2)
      }
    }
    //} else {
    //  int ii;
    //  int ne = qubit_filter.size(); // number of qubits s,p act on
    //  
    //  //use qubit_filter - only rows & cols of "full s" corresponding to qubit_filter are non-identity
    //  for(i=0; i<_2n*_2n; i++) out_s[i] = _smatrix[i]; // copy out_s = _smatrix
    //
    //  for(ii=0; ii<qubit_filter.size(); ii++) { // only non-identity rows of "full s"
    //	i = qubit_filter[ii];
    //    for(j=0; j<_2n; j++) {
    //	  tmp = 0;
    //	  for(int kk=0; kk<qubit_filter.size(); kk++) { // only non-zero cols of non-identity i-th row of "full s"
    //	    k = qubit_filter[kk];
    //	    tmp += s[ii*_2n+kk] * _smatrix[k*_2n+j];
    //	    tmp += s[ii*_2n+(kk+ne)] * _smatrix[(k+_n)*_2n+j];
    //	  }
    //	  out_s[i*_2n+j] = tmp % 2; // all els are mod(2)
    //    }
    //
    //	// part2, for (i+n)-th row of "full s"
    //	i = qubit_filter[ii] + _n;
    //	int iin = ii + ne;
    //	for(j=0; j<_2n; j++) {
    //	  tmp = 0;
    //	  for(int kk=0; kk<qubit_filter.size(); kk++) { // only non-zero cols of non-identity i-th row of "full s"
    //	    k = qubit_filter[kk];
    //	    tmp += s[iin*_2n+kk] * _smatrix[k*_2n+j];
    //	    tmp += s[iin*_2n+(kk+ne)] * _smatrix[(k+_n)*_2n+j];
    //	  }
    //	  out_s[i*_2n+j] = tmp % 2; // all els are mod(2)
    //    }
    //  }
    //}

    int* inner = new int[_2n*_2n];
    int* tmp1 = new int[_2n];
    int* tmp2 = new int[_2n*_2n];
    int* vec = new int[_2n];
    udot2(inner, s, s);

    // vec = _np.dot(_np.transpose(_smatrix),p - _mtx.diagonal_as_vec(inner))
    for(i=0; i<_2n; i++) tmp1[i] = p[i] - inner[i*_2n+i];
    for(i=0; i<_2n; i++) {
      vec[i] = 0; 
      for(k=0; k<_2n; k++)
	vec[i] += _smatrix[k*_2n+i] * tmp1[k];
    }
	  
    //matrix = 2*_mtx.strictly_upper_triangle(inner)+_mtx.diagonal_as_matrix(inner)
    int* matrix = inner; //just modify inner in place since we don't need it anymore
    for(i=0; i<_2n; i++) {
      for(j=0; j<i; j++) matrix[i*_2n+j] = 0; //lower triangle
      for(j=i+1; j<_2n; j++) matrix[i*_2n+j] *= 2; //strict upper triangle
    }

    //vec += _mtx.diagonal_as_vec(_np.dot(_np.dot(_np.transpose(self.s),matrix),self.s))
    for(i=0; i<_2n; i++) {
      for(j=0; j<_2n; j++) {
	tmp2[i*_2n+j] = 0;
	for(k=0; k<_2n; k++)
	  tmp2[i*_2n+j] += _smatrix[k*_2n+i]*matrix[k*_2n+j];
      }
    }
    for(i=0; i<_2n; i++) {  //TODO - could put this within i-loop above and only use tmp1...
      for(k=0; k<_2n; k++)
	vec[i] += tmp2[i*_2n+k]*_smatrix[k*_2n+i];
    }

    // _smatrix = out_s (don't set this until we're done using _smatrix)
    for(i=0; i<_2n*_2n; i++) _smatrix[i] = out_s[i];
    for(i=0; i<_namps; i++) {
      int* pvec = &_pvectors[ _2n*i ]; // i-th vector
      for(k=0; k<_2n; k++) pvec[k] = (pvec[k] + vec[k]) % 4;
    }

    delete [] out_s;
    delete [] inner;
    delete [] tmp1;
    delete [] tmp2;
    delete [] vec;
  }
  

  /****************************************************************************\
  |* SBEffectCRep                                                             *|
  \****************************************************************************/
  SBEffectCRep::SBEffectCRep(int* zvals, int n)
    : _zvals(n)
  {
    for(int i=0; i<n; i++)
      _zvals[i] = zvals[i];
    _n = n;
  }
  SBEffectCRep::~SBEffectCRep() { }
  
  dcomplex SBEffectCRep::amplitude(SBStateCRep* state) {
    DEBUG(std::cout << "SBEffectCRep::amplitude called! zvals = " << _zvals[0] << std::endl);

    //DEBUG!!!
    //int i;
    //int _2n = state->_2n;
    //int _namps = state->_namps;
    //
    //std::cout << "S = ";
    //for(i=0; i<_2n*_2n; i++) std::cout << state->_smatrix[i] << " ";
    //std::cout << std::endl;
    //std::cout << "PS = ";
    //for(i=0; i<_namps*_2n; i++) std::cout << state->_pvectors[i] << " ";
    //std::cout << std::endl;
    //std::cout << "AMPs = ";
    //for(i=0; i<_namps; i++) std::cout << state->_amps[i] << " ";
    //std::cout << std::endl;
    
    return state->extract_amplitude(_zvals);
    //DEBUG std::cout << "AMP = " << amp << std::endl;
    //DEBUG return amp;
  }

  double SBEffectCRep::probability(SBStateCRep* state) {
    DEBUG(std::cout << "SBEffectCRep::probability called! zvals = " << _zvals[0] << std::endl);
    return pow(std::abs(amplitude(state)),2);
  }


  /****************************************************************************\
  |* SBGateCRep                                                               *|
  \****************************************************************************/
  SBGateCRep::SBGateCRep(int n) {
    _n = n;
  }
  SBGateCRep::~SBGateCRep() { }

  /****************************************************************************\
  |* SBGateCRep_Embedded                                                      *|
  \****************************************************************************/
  SBGateCRep_Embedded::SBGateCRep_Embedded(SBGateCRep* embedded_gate_crep, int n, int* qubits, int nqubits)
    :SBGateCRep(n),_qubits(nqubits)
  {
    _embedded_gate_crep = embedded_gate_crep;
    for(int i=0; i<nqubits; i++)
      _qubits[i] = qubits[i];
  }
  SBGateCRep_Embedded::~SBGateCRep_Embedded() { }
  
  SBStateCRep* SBGateCRep_Embedded::acton(SBStateCRep* state, SBStateCRep* out_state) {
    DEBUG(std::cout << "Stabilizer Embedded acton called!" << std::endl);
    state->push_view(_qubits);
    _embedded_gate_crep->acton(state, out_state);
    state->pop_view();
    out_state->pop_view(); //should have same view as input state
    return out_state;
  }

  SBStateCRep* SBGateCRep_Embedded::adjoint_acton(SBStateCRep* state, SBStateCRep* out_state) {
    DEBUG(std::cout << "Stabilizer Embedded adjoint_acton called!" << std::endl);
    state->push_view(_qubits);
    _embedded_gate_crep->adjoint_acton(state, out_state);
    state->pop_view();
    out_state->pop_view(); //should have same view as input state
    return out_state;
  }


  /****************************************************************************\
  |* SBGateCRep_Composed                                                      *|
  \****************************************************************************/
  SBGateCRep_Composed::SBGateCRep_Composed(std::vector<SBGateCRep*> factor_gate_creps, int n)
    :SBGateCRep(n),_factor_gate_creps(factor_gate_creps)
  {
  }
  SBGateCRep_Composed::~SBGateCRep_Composed() { }

  SBStateCRep* SBGateCRep_Composed::acton(SBStateCRep* state, SBStateCRep* out_state) {
    DEBUG(std::cout << "Stabilizer Composed acton called!" << std::endl);
    std::size_t nfactors = _factor_gate_creps.size();
    SBStateCRep *tmp2, *tmp1 = out_state; //tmp1 already alloc'd
    SBStateCRep* t; // for swapping

    //Act with first gate: output in tmp1
    _factor_gate_creps[0]->acton(state, tmp1);
    
    if(nfactors > 1) {
      SBStateCRep temp_state(tmp1->_namps,_n); tmp2 = &temp_state;

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
    return out_state;
  }

  SBStateCRep* SBGateCRep_Composed::adjoint_acton(SBStateCRep* state, SBStateCRep* out_state) {
    DEBUG(std::cout << "Stabilizer Composed adjoint_acton called!" << std::endl);
    std::size_t nfactors = _factor_gate_creps.size();
    SBStateCRep *tmp2, *tmp1 = out_state; //tmp1 already alloc'd
    SBStateCRep* t; // for swapping

    //Act with first gate: output in tmp1
    _factor_gate_creps[nfactors-1]->adjoint_acton(state, tmp1);

    if(nfactors > 1) {
      SBStateCRep temp_state(tmp1->_namps,_n); tmp2 = &temp_state;

      //Act with additional gates: tmp1 -> tmp2 then swap, so output in tmp1
      for(int i=nfactors-2; i >= 0; i--) {
	_factor_gate_creps[i]->adjoint_acton(tmp1,tmp2);
	t = tmp1; tmp1 = tmp2; tmp2 = t;
      }
      
      //tmp1 holds the output state now; if tmp1 == out_state
      // we're in luck, otherwise we need to copy it into out_state.
      if(tmp1 != out_state) {
	out_state->copy_from(tmp1);
      }
    }
    return out_state;
  }


  /****************************************************************************\
  |* SBGateCRep_Clifford                                                      *|
  \****************************************************************************/
  SBGateCRep_Clifford::SBGateCRep_Clifford(int* smatrix, int* svector, dcomplex* unitary,
					   int* smatrix_inv, int* svector_inv, dcomplex* unitary_adj, int n)
    :SBGateCRep(n)
  {
    _smatrix = smatrix;
    _svector = svector;
    _smatrix_inv = smatrix_inv;
    _svector_inv = svector_inv;
    _unitary = unitary;
    _unitary_adj = unitary_adj;
  }
  SBGateCRep_Clifford::~SBGateCRep_Clifford() { }
  
  SBStateCRep* SBGateCRep_Clifford::acton(SBStateCRep* state, SBStateCRep* out_state) {
    DEBUG(std::cout << "Stabilizer Clifford acton called!" << std::endl);
    out_state->copy_from(state);
    out_state->clifford_update(_smatrix, _svector, _unitary);
    return out_state;
  }

  SBStateCRep* SBGateCRep_Clifford::adjoint_acton(SBStateCRep* state, SBStateCRep* out_state) {
    DEBUG(std::cout << "Stabilizer Clifford adjoint_acton called!" << std::endl);
    out_state->copy_from(state);
    out_state->clifford_update(_smatrix_inv, _svector_inv, _unitary_adj);
    return out_state;
  }


  
  //TODO - implement in C

  //_fastcalc.fast_kron(scratch, _fast_kron_array, _fast_kron_factordims);

  //        state = _mt.expm_multiply_fast(self.err_gen_prep, state) # (N,) -> (N,) shape mapping

  //_fastcalc.embedded_fast_acton_sparse(self.embedded_gate.acton,
  //                                         output_state, state,
  //                                         self.noop_incrementers,
  //                                         self.numBasisEls_noop_blankaction,
  //                                         self.baseinds)


  void expm_multiply_simple_core(double* Adata, int* Aindptr,
				 int* Aindices, double* B,
				 int N, double mu, int m_star,
				 int s, double tol, double eta,
				 double* F, double* scratch) {
    int i;
    int j;
    int r;
    int k;

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


  // OTHER Classes

  /****************************************************************************\
  |* PolyCRep                                                                 *|
  \****************************************************************************/

  //std::unordered_map[int, dcomplex] _coeffs;
  //int _max_order;
  //int _max_num_vars;
  PolyCRep::PolyCRep() {
    _coeffs = std::unordered_map<int, dcomplex>();
    _max_order = 0;
    _max_num_vars = 0;
  }
  
  PolyCRep::PolyCRep(std::unordered_map<int, dcomplex> coeffs, int max_order, int max_num_vars) {
    _coeffs = coeffs;
    _max_order = max_order;
    _max_num_vars = max_num_vars;
  }

  PolyCRep::PolyCRep(const PolyCRep& other) {
    _coeffs = other._coeffs;
    _max_order = other._max_order;
    _max_num_vars = other._max_num_vars;
  }

  PolyCRep::~PolyCRep() { }

  PolyCRep PolyCRep::mult(const PolyCRep& other) {
    std::unordered_map<int, dcomplex>::iterator it1, itk;
    std::unordered_map<int, dcomplex>::const_iterator it2;
    std::unordered_map<int, dcomplex> result;
    dcomplex val;
    int k;

    for(it1 = _coeffs.begin(); it1 != _coeffs.end(); ++it1) {
      for(it2 = other._coeffs.begin(); it2 != other._coeffs.end(); ++it2) {
	k = mult_vinds_ints(it1->first, it2->first); //key to add
	itk = result.find(k);
	val = it1->second * it2->second;
	if(itk != result.end())
	  itk->second = itk->second + val;
	else result[k] = val;
      }
    }
    PolyCRep ret(result, _max_order, _max_num_vars);
    return ret; // need a copy constructor?
  }

  void PolyCRep::add_inplace(const PolyCRep& other) {
    std::unordered_map<int, dcomplex>::const_iterator it2;
      std::unordered_map<int, dcomplex>::iterator itk;
    dcomplex val, newval;
    int k;

    for(it2 = other._coeffs.begin(); it2 != other._coeffs.end(); ++it2) {
      k = it2->first; // key
      val = it2->second; // value
      itk = _coeffs.find(k);
      if(itk != _coeffs.end()) {
	newval = itk->second + val;
	if(std::abs(newval) > 1e-12) {
	  itk->second = newval; // note: += doens't work here (complex Cython?)
	} else {
	  _coeffs.erase(itk);
	}
      }
      else if(std::abs(val) > 1e-12) {
	_coeffs[k] = val;
      }
    }
  }

  void PolyCRep::scale(dcomplex scale) {
    std::unordered_map<int, dcomplex>::iterator it;
    for(it = _coeffs.begin(); it != _coeffs.end(); ++it) {
      it->second = it->second * scale; // note: *= doesn't work here (complex Cython?)
    }
  }

  int PolyCRep::vinds_to_int(std::vector<int> vinds) {
    int ret = 0;
    int m = 1;
    for(std::size_t i=0; i<vinds.size(); i++) { // last tuple index is most significant
      ret += (vinds[i]+1)*m;
      m *= _max_num_vars+1;
    }
    return ret;
  }
  
  std::vector<int> PolyCRep::int_to_vinds(int indx) {
    std::vector<int> ret;
    int nxt, i;
    while(indx != 0) {
      nxt = indx / (_max_num_vars+1);
      i = indx - nxt*(_max_num_vars+1);
      ret.push_back(i-1);
      indx = nxt;
      //assert(indx >= 0);
    }
    std::sort(ret.begin(),ret.end());
    return ret;
  }
  
  int PolyCRep::mult_vinds_ints(int i1, int i2) {
    // multiply vinds corresponding to i1 & i2 and return resulting integer
    std::vector<int> vinds1 = int_to_vinds(i1);
    std::vector<int> vinds2 = int_to_vinds(i2);
    vinds1.insert( vinds1.end(), vinds2.begin(), vinds2.end() );
    std::sort(vinds1.begin(),vinds1.end());
    return vinds_to_int(vinds1);
  }
  
  /****************************************************************************\
  |* SVTermCRep                                                               *|
  \****************************************************************************/
    
  SVTermCRep::SVTermCRep(PolyCRep* coeff, SVStateCRep* pre_state, SVStateCRep* post_state,
			 std::vector<SVGateCRep*> pre_ops, std::vector<SVGateCRep*> post_ops) {
    _coeff = coeff;
    _pre_state = pre_state;
    _post_state = post_state;
    _pre_effect = NULL;
    _post_effect = NULL;
    _pre_ops = pre_ops;
    _post_ops = post_ops;
  }
  
  SVTermCRep::SVTermCRep(PolyCRep* coeff, SVEffectCRep* pre_effect, SVEffectCRep* post_effect,
			 std::vector<SVGateCRep*> pre_ops, std::vector<SVGateCRep*> post_ops) {
    _coeff = coeff;
    _pre_state = NULL;
    _post_state = NULL;
    _pre_effect = pre_effect;
    _post_effect = post_effect;
    _pre_ops = pre_ops;
    _post_ops = post_ops;
  }
  
  SVTermCRep::SVTermCRep(PolyCRep* coeff, std::vector<SVGateCRep*> pre_ops,
			 std::vector<SVGateCRep*> post_ops) {
    _coeff = coeff;
    _pre_state = NULL;
    _post_state = NULL;
    _pre_effect = NULL;
    _post_effect = NULL;
    _pre_ops = pre_ops;
    _post_ops = post_ops;
  }

}
