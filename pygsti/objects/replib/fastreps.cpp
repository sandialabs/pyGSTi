#define NULL 0

#include <iostream>
#include <complex>
#include <assert.h>
#include <algorithm>    // std::find
#include "fastreps.h"
//#include <pthread.h>

//using namespace std::complex_literals;

//#define DEBUG(x) x
#define DEBUG(x) 

namespace CReps {

  // DENSE MATRIX (DM) propagation
  
  /****************************************************************************\
  |* DMStateCRep                                                              *|
  \****************************************************************************/
  DMStateCRep::DMStateCRep(INT dim) {
    _dataptr = new double[dim];
    for(INT i=0; i<dim; i++) _dataptr[i] = 0;
    _dim = dim;
    _ownmem = true;
  }
  
  DMStateCRep::DMStateCRep(double* data, INT dim, bool copy=false) {
    //DEGUG std::cout << "DMStateCRep initialized w/dim = " << dim << std::endl;
    if(copy) {
      _dataptr = new double[dim];
      for(INT i=0; i<dim; i++) {
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
    for(INT i=0; i<_dim; i++) std::cout << _dataptr[i] << " ";
    std::cout << "]" << std::endl;
  }

  void DMStateCRep::copy_from(DMStateCRep* st) {
    assert(_dim == st->_dim);
    for(INT i=0; i<_dim; i++)
      _dataptr[i] = st->_dataptr[i];
  }

  /****************************************************************************\
  |* DMEffectCRep                                                             *|
  \****************************************************************************/
  DMEffectCRep::DMEffectCRep(INT dim) {
    _dim = dim;
  }
  DMEffectCRep::~DMEffectCRep() { }


  /****************************************************************************\
  |* DMEffectCRep_Dense                                                       *|
  \****************************************************************************/
  DMEffectCRep_Dense::DMEffectCRep_Dense(double* data, INT dim)
    :DMEffectCRep(dim)
  {
    _dataptr = data;
  }

  DMEffectCRep_Dense::~DMEffectCRep_Dense() { }

  double DMEffectCRep_Dense::probability(DMStateCRep* state) {
    double ret = 0.0;
    for(INT i=0; i< _dim; i++) {
      ret += _dataptr[i] * state->_dataptr[i];
    }
    return ret;
  }

  
  /****************************************************************************\
  |* DMEffectCRep_TensorProd                                                  *|
  \****************************************************************************/

  DMEffectCRep_TensorProd::DMEffectCRep_TensorProd(double* kron_array,
						   INT* factordims, INT nfactors,
						   INT max_factor_dim, INT dim) 
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
    INT N = _dim;
    INT i, j, k, sz, off, endoff, krow;
    double mult;
    double* array = _kron_array;
    INT* arraysizes = _factordims;

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

    for(INT i=0; i< _dim; i++) {
      ret += scratch[i] * state->_dataptr[i];
    }
    delete [] scratch;
    return ret;
  }


  /****************************************************************************\
  |* DMEffectCRep_Computational                                               *|
  \****************************************************************************/

  DMEffectCRep_Computational::DMEffectCRep_Computational(INT nfactors, INT zvals_int, double abs_elval, INT dim)
    :DMEffectCRep(dim)
  {
    _nfactors = nfactors;
    _zvals_int = zvals_int;
    _abs_elval = abs_elval;
  }

  DMEffectCRep_Computational::~DMEffectCRep_Computational() { }
    
  double DMEffectCRep_Computational::probability(DMStateCRep* state) {
    // The logic here is very similar to the todense method in the Python rep version
    // Here we don't bother to compute the dense vector - we just perform the
    // dot product using only the nonzero vector elements.
    INT& N = _nfactors;
    INT nNonzero = 1 << N; // 2**N
    INT finalIndx, k, base;
    double ret = 0.0;
    
    for(INT finds=0; finds < nNonzero; finds++) {

      //Compute finalIndx
      finalIndx = 0; base = 1 << (2*N-2); //4**(N-1) = 2**(2N-2)
      for(k=0; k<N; k++) {
	finalIndx += ((finds >> k) & 1) * 3 * base;
	base = base >> 2; // /= 4 so base == 4**(N-1-k)
      }

      //Apply result
      if(parity(finds & _zvals_int))
	ret -= _abs_elval * state->_dataptr[finalIndx]; // minus sign
      else
	ret += _abs_elval * state->_dataptr[finalIndx];
    }
    return ret;
  }

  INT DMEffectCRep_Computational::parity(INT x) {
    // int64-bit specific
    x = (x & 0x00000000FFFFFFFF)^(x >> 32);
    x = (x & 0x000000000000FFFF)^(x >> 16);
    x = (x & 0x00000000000000FF)^(x >> 8);
    x = (x & 0x000000000000000F)^(x >> 4);
    x = (x & 0x0000000000000003)^(x >> 2);
    x = (x & 0x0000000000000001)^(x >> 1);
    return x & 1; // return the last bit (0 or 1)
  }


  /****************************************************************************\
  |* DMEffectCRep_Errgen                                                      *|
  \****************************************************************************/

  DMEffectCRep_Errgen::DMEffectCRep_Errgen(DMOpCRep* errgen_oprep,
					   DMEffectCRep* effect_rep,
					   INT errgen_id, INT dim)
    :DMEffectCRep(dim)
  {
    _errgen_ptr = errgen_oprep;
    _effect_ptr = effect_rep;
    _errgen_id = errgen_id;
  }
  
  DMEffectCRep_Errgen::~DMEffectCRep_Errgen() { }
  
  double DMEffectCRep_Errgen::probability(DMStateCRep* state) {
    DMStateCRep outState(_dim);
    _errgen_ptr->acton(state, &outState);
    return _effect_ptr->probability(&outState);
  }

  double DMEffectCRep_Errgen::probability_using_cache(DMStateCRep* state, DMStateCRep* errgen_on_state, INT& errgen_id) {
    if(errgen_id != _errgen_id) {
      _errgen_ptr->acton(state, errgen_on_state);
      errgen_id = _errgen_id;
    }
    return _effect_ptr->probability(errgen_on_state);
  }


  /****************************************************************************\
  |* DMOpCRep                                                               *|
  \****************************************************************************/

  DMOpCRep::DMOpCRep(INT dim) {
    _dim = dim;
  }
  DMOpCRep::~DMOpCRep() { }


  /****************************************************************************\
  |* DMOpCRep_Dense                                                         *|
  \****************************************************************************/

  DMOpCRep_Dense::DMOpCRep_Dense(double* data, INT dim)
    :DMOpCRep(dim)
  {
    _dataptr = data;
  }
  DMOpCRep_Dense::~DMOpCRep_Dense() { }

  DMStateCRep* DMOpCRep_Dense::acton(DMStateCRep* state,
				       DMStateCRep* outstate) {
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

  DMStateCRep* DMOpCRep_Dense::adjoint_acton(DMStateCRep* state,
					       DMStateCRep* outstate) {
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
  |* DMOpCRep_Embedded                                                      *|
  \****************************************************************************/

  DMOpCRep_Embedded::DMOpCRep_Embedded(DMOpCRep* embedded_gate_crep, INT* noop_incrementers,
				       INT* numBasisEls_noop_blankaction, INT* baseinds, INT* blocksizes,
				       INT embedded_dim, INT nComponentsInActiveBlock, INT iActiveBlock,
				       INT nBlocks, INT dim)
    :DMOpCRep(dim)
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
  DMOpCRep_Embedded::~DMOpCRep_Embedded() { }
  
  DMStateCRep* DMOpCRep_Embedded::acton(DMStateCRep* state, DMStateCRep* out_state) {

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
	}
	offset += _blocksizes[i];
      }
    }

    DEBUG(out_state->print("OUTPUT"));
    return out_state;
  }


  DMStateCRep* DMOpCRep_Embedded::adjoint_acton(DMStateCRep* state, DMStateCRep* out_state) {

    //Note: exactly the same as acton(...) but calls embedded gate's adjoint_acton
    DEBUG(std::cout << "DM Embedded adjoint_acton called!" << std::endl);
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
  |* DMOpCRep_Composed                                                      *|
  \****************************************************************************/
  DMOpCRep_Composed::DMOpCRep_Composed(std::vector<DMOpCRep*> factor_gate_creps, INT dim)
    :DMOpCRep(dim),_factor_gate_creps(factor_gate_creps)
  {
  }
  DMOpCRep_Composed::~DMOpCRep_Composed() { }

  void DMOpCRep_Composed::reinit_factor_op_creps(std::vector<DMOpCRep*> new_factor_gate_creps) {
    _factor_gate_creps.clear(); //removes all elements
    _factor_gate_creps.insert(_factor_gate_creps.end(),
			      new_factor_gate_creps.begin(),
			      new_factor_gate_creps.end());  //inserts contents of new array
  }
  
  DMStateCRep* DMOpCRep_Composed::acton(DMStateCRep* state, DMStateCRep* out_state) {

    DEBUG(std::cout << "Composed acton called!" << std::endl);
    DEBUG(state->print("INPUT"));
    std::size_t nfactors = _factor_gate_creps.size();
    DMStateCRep *tmp2, *tmp1 = out_state; //tmp1 already alloc'd
    DMStateCRep* t; // for swapping

    //if length is 0 just copy state --> outstate
    if(nfactors == 0) {
      out_state->copy_from(state);
      return out_state;
    }

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

  DMStateCRep* DMOpCRep_Composed::adjoint_acton(DMStateCRep* state, DMStateCRep* out_state) {

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
  |* DMOpCRep_Sum                                                           *|
  \****************************************************************************/
  DMOpCRep_Sum::DMOpCRep_Sum(std::vector<DMOpCRep*> factor_creps, INT dim)
    :DMOpCRep(dim),_factor_creps(factor_creps)
  {
  }
  DMOpCRep_Sum::~DMOpCRep_Sum() { }
  
  DMStateCRep* DMOpCRep_Sum::acton(DMStateCRep* state, DMStateCRep* out_state) {

    DEBUG(std::cout << "Sum acton called!" << std::endl);
    DEBUG(state->print("INPUT"));
    std::size_t nfactors = _factor_creps.size();
    DMStateCRep temp_state(_dim);

    //zero-out output state
    for(INT k=0; k<_dim; k++)
      out_state->_dataptr[k] = 0.0;

    //if length is 0 just return "0" state --> outstate
    if(nfactors == 0) return out_state;

    //Act with factors and accumulate into out_state
    for(std::size_t i=0; i < nfactors; i++) {
      _factor_creps[i]->acton(state,&temp_state);
      for(INT k=0; k<_dim; k++)
	out_state->_dataptr[k] += temp_state._dataptr[k];
    }
    DEBUG(out_state->print("OUTPUT"));
    return out_state;
  }

  DMStateCRep* DMOpCRep_Sum::adjoint_acton(DMStateCRep* state, DMStateCRep* out_state) {

    //Note: same as acton(...) but perform adjoint_acton
    DEBUG(std::cout << "Sum adjoint_acton called!" << std::endl);
    DEBUG(state->print("INPUT"));
    std::size_t nfactors = _factor_creps.size();
    DMStateCRep temp_state(_dim);

    //zero-out output state
    for(INT k=0; k<_dim; k++)
      out_state->_dataptr[k] = 0.0;

    //if length is 0 just return "0" state --> outstate
    if(nfactors == 0) return out_state;

    //Act with factors and accumulate into out_state
    for(std::size_t i=0; i < nfactors; i++) {
      _factor_creps[i]->adjoint_acton(state,&temp_state);
      for(INT k=0; k<_dim; k++)
	out_state->_dataptr[k] += temp_state._dataptr[k];
    }
    DEBUG(out_state->print("OUTPUT"));
    return out_state;
  }

  /****************************************************************************\
  |* DMOpCRep_Exponentiated                                                   *|
  \****************************************************************************/

  DMOpCRep_Exponentiated::DMOpCRep_Exponentiated(DMOpCRep* exponentiated_gate_crep, INT power, INT dim)
    :DMOpCRep(dim)
  {
    _exponentiated_gate_crep = exponentiated_gate_crep;
    _power = power;
  }

  DMOpCRep_Exponentiated::~DMOpCRep_Exponentiated() { }

  DMStateCRep* DMOpCRep_Exponentiated::acton(DMStateCRep* state, DMStateCRep* out_state) {

    DEBUG(std::cout << "Exponentiated acton called!" << std::endl);
    DEBUG(state->print("INPUT"));
    DMStateCRep *tmp2, *tmp1 = out_state; //tmp1 already alloc'd
    DMStateCRep* t; // for swapping

    //if power is 0 just copy state --> outstate
    if(_power == 0) {
      out_state->copy_from(state);
      return out_state;
    }

    //Act with first gate: output in tmp1
    _exponentiated_gate_crep->acton(state, tmp1);
    
    if(_power > 1) {
      DMStateCRep temp_state(_dim); tmp2 = &temp_state;

      //Act with additional gates: tmp1 -> tmp2 then swap, so output in tmp1
      for(INT i=1; i < _power; i++) {
	_exponentiated_gate_crep->acton(tmp1,tmp2);
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

  DMStateCRep* DMOpCRep_Exponentiated::adjoint_acton(DMStateCRep* state, DMStateCRep* out_state) {

    DEBUG(std::cout << "Exponentiated adjoint_acton called!" << std::endl);
    DEBUG(state->print("INPUT"));
    DMStateCRep *tmp2, *tmp1 = out_state; //tmp1 already alloc'd
    DMStateCRep* t; // for swapping

    //Note: same as acton(...) but perform adjoint_acton
    //if power is 0 just copy state --> outstate
    if(_power == 0) {
      out_state->copy_from(state);
      return out_state;
    }

    //Act with first gate: output in tmp1
    _exponentiated_gate_crep->adjoint_acton(state, tmp1);
    
    if(_power > 1) {
      DMStateCRep temp_state(_dim); tmp2 = &temp_state;

      //Act with additional gates: tmp1 -> tmp2 then swap, so output in tmp1
      for(INT i=1; i < _power; i++) {
	_exponentiated_gate_crep->adjoint_acton(tmp1,tmp2);
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
  |* DMOpCRep_Lindblad                                                      *|
  \****************************************************************************/

  DMOpCRep_Lindblad::DMOpCRep_Lindblad(DMOpCRep* errgen_rep,
			double mu, double eta, INT m_star, INT s, INT dim,
			double* unitarypost_data, INT* unitarypost_indices,
			INT* unitarypost_indptr, INT unitarypost_nnz)
    :DMOpCRep(dim)
  {
    _U_data = unitarypost_data;
    _U_indices = unitarypost_indices;
    _U_indptr = unitarypost_indptr;
    _U_nnz = unitarypost_nnz;
    
    _errgen_rep = errgen_rep;
    
    _mu = mu;
    _eta = eta;
    _m_star = m_star;
    _s = s;
  }
      
  DMOpCRep_Lindblad::~DMOpCRep_Lindblad() { }

  DMStateCRep* DMOpCRep_Lindblad::acton(DMStateCRep* state, DMStateCRep* out_state)
  {
    INT i, j;
    DMStateCRep* init_state = new DMStateCRep(_dim);
    DEBUG(std::cout << "Lindblad acton called!" << _U_nnz << std::endl);
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
    INT N = _dim; // = length(_A_indptr)-1
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
    expm_multiply_simple_core_rep(_errgen_rep, B, N, _mu,
				  _m_star, _s, tol, _eta, F, scratch);

    //cleanup what we allocated
    delete [] scratch;
    delete init_state;

    DEBUG(out_state->print("OUTPUT"));
    return out_state;
  }

  DMStateCRep* DMOpCRep_Lindblad::adjoint_acton(DMStateCRep* state, DMStateCRep* out_state) {
    assert(false); //ajoint_acton not implemented yet for Lindblad gates (TODO LATER)
    return NULL; //to avoid compiler warning
  }


  /****************************************************************************\
  |* DMOpCRep_Sparse                                                        *|
  \****************************************************************************/

  DMOpCRep_Sparse::DMOpCRep_Sparse(double* A_data, INT* A_indices, INT* A_indptr,
				       INT nnz, INT dim)
    :DMOpCRep(dim)
  {
    _A_data = A_data;
    _A_indices = A_indices;
    _A_indptr = A_indptr;
    _A_nnz = nnz;
  }
      
  DMOpCRep_Sparse::~DMOpCRep_Sparse() { }

  DMStateCRep* DMOpCRep_Sparse::acton(DMStateCRep* state, DMStateCRep* out_state)
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

  DMStateCRep* DMOpCRep_Sparse::adjoint_acton(DMStateCRep* state, DMStateCRep* out_state) {
    //Need to take transpose of a CSR matrix then mult by vector...
    assert(false); //ajoint_acton not implemented yet for Sparse gates (TODO LATER)
    return NULL; //to avoid compiler warning
  }

  
  // STATE VECTOR (SV) propagation

  /****************************************************************************\
  |* SVStateCRep                                                              *|
  \****************************************************************************/
  SVStateCRep::SVStateCRep(INT dim) {
    _dataptr = new dcomplex[dim];
    for(INT i=0; i<dim; i++) _dataptr[i] = 0;
    _dim = dim;
    _ownmem = true;
  }
  
  SVStateCRep::SVStateCRep(dcomplex* data, INT dim, bool copy=false) {
    //DEGUG std::cout << "SVStateCRep initialized w/dim = " << dim << std::endl;
    if(copy) {
      _dataptr = new dcomplex[dim];
      for(INT i=0; i<dim; i++) {
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
    for(INT i=0; i<_dim; i++) std::cout << _dataptr[i] << " ";
    std::cout << "]" << std::endl;
  }

  void SVStateCRep::copy_from(SVStateCRep* st) {
    assert(_dim == st->_dim);
    for(INT i=0; i<_dim; i++)
      _dataptr[i] = st->_dataptr[i];
  }

  /****************************************************************************\
  |* SVEffectCRep                                                             *|
  \****************************************************************************/
  SVEffectCRep::SVEffectCRep(INT dim) {
    _dim = dim;
  }
  SVEffectCRep::~SVEffectCRep() { }


  /****************************************************************************\
  |* SVEffectCRep_Dense                                                       *|
  \****************************************************************************/
  SVEffectCRep_Dense::SVEffectCRep_Dense(dcomplex* data, INT dim)
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
    for(INT i=0; i< _dim; i++) {
      ret += std::conj(_dataptr[i]) * state->_dataptr[i];
    }
    return ret;
  }

  
  /****************************************************************************\
  |* SVEffectCRep_TensorProd                                                  *|
  \****************************************************************************/

  SVEffectCRep_TensorProd::SVEffectCRep_TensorProd(dcomplex* kron_array,
						   INT* factordims, INT nfactors,
						   INT max_factor_dim, INT dim) 
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
    INT N = _dim;
    INT i, j, k, sz, off, endoff, krow;
    dcomplex mult;
    dcomplex* array = _kron_array;
    INT* arraysizes = _factordims;

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

    for(INT i=0; i< _dim; i++) {
      ret += std::conj(scratch[i]) * state->_dataptr[i];
        //conjugate scratch b/c we assume _kron_array contains
        // info for building up the *column* "effect vector" E
        // s.t. amplitudes are computed as dot(E.T.conj,state_col_vec)
    }
    delete [] scratch;
    return ret;
  }

  
  /****************************************************************************	\
  |* SVEffectCRep_Computational                                               *|
  \****************************************************************************/

  SVEffectCRep_Computational::SVEffectCRep_Computational(INT nfactors, INT zvals_int, INT dim)
    :SVEffectCRep(dim)
  {
    _nfactors = nfactors;
    _zvals_int = zvals_int;


    _nonzero_index = 0;
    INT base = 1 << (nfactors-1); // == pow(2,nfactors-1)
    for(INT i=0; i < nfactors; i++) {
      if((zvals_int >> i) & 1) // if i-th bit (i-th zval) is a 1 (it's either 0 or 1)
	_nonzero_index += base;
      base = base >> 1; // same as /= 2
    }
    
  }

  SVEffectCRep_Computational::~SVEffectCRep_Computational() { }

  double SVEffectCRep_Computational::probability(SVStateCRep* state) {
    return (double)pow(std::abs(amplitude(state)),2);
  }
  
  dcomplex SVEffectCRep_Computational::amplitude(SVStateCRep* state) {
    //There's only a single nonzero index with element == 1.0, so dotprod is trivial
    return state->_dataptr[_nonzero_index];
  }
    

  /****************************************************************************\
  |* SVOpCRep                                                               *|
  \****************************************************************************/

  SVOpCRep::SVOpCRep(INT dim) {
    _dim = dim;
  }
  SVOpCRep::~SVOpCRep() { }


  /****************************************************************************\
  |* SVOpCRep_Dense                                                         *|
  \****************************************************************************/

  SVOpCRep_Dense::SVOpCRep_Dense(dcomplex* data, INT dim)
    :SVOpCRep(dim)
  {
    _dataptr = data;
  }
  SVOpCRep_Dense::~SVOpCRep_Dense() { }

  SVStateCRep* SVOpCRep_Dense::acton(SVStateCRep* state,
				       SVStateCRep* outstate) {
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

  SVStateCRep* SVOpCRep_Dense::adjoint_acton(SVStateCRep* state,
					       SVStateCRep* outstate) {
    DEBUG(std::cout << "Dense adjoint_acton called!" << std::endl);
    DEBUG(state->print("INPUT"));
    for(INT i=0; i< _dim; i++) {
      outstate->_dataptr[i] = 0.0;
      for(INT j=0; j< _dim; j++) {
	outstate->_dataptr[i] += std::conj(_dataptr[j*_dim+i]) * state->_dataptr[j];
      }
    }
    DEBUG(outstate->print("OUTPUT"));
    return outstate;
  }


  /****************************************************************************\
  |* SVOpCRep_Embedded                                                      *|
  \****************************************************************************/

  SVOpCRep_Embedded::SVOpCRep_Embedded(SVOpCRep* embedded_gate_crep, INT* noop_incrementers,
				       INT* numBasisEls_noop_blankaction, INT* baseinds, INT* blocksizes,
				       INT embedded_dim, INT nComponentsInActiveBlock, INT iActiveBlock,
				       INT nBlocks, INT dim)
    :SVOpCRep(dim)
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
  SVOpCRep_Embedded::~SVOpCRep_Embedded() { }
  
  SVStateCRep* SVOpCRep_Embedded::acton(SVStateCRep* state, SVStateCRep* out_state) {

    DEBUG(std::cout << "SV Embedded acton called!" << std::endl);
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

    dcomplex* state_data = state->_dataptr;
    dcomplex* outstate_data = out_state->_dataptr;

    //zero-out output state initially
    for(i=0; i<_dim; i++) outstate_data[i] = 0.0;

    INT b[100]; // could alloc dynamically (LATER?)
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


  SVStateCRep* SVOpCRep_Embedded::adjoint_acton(SVStateCRep* state, SVStateCRep* out_state) {

    //Note: exactly the same as acton(...) but calls embedded gate's adjoint_acton
    DEBUG(std::cout << "SV Embedded adjoint_acton called!" << std::endl);
    DEBUG(state->print("INPUT"));
    INT i, j, k, vec_index_noop = 0;
    INT nParts = _nComponents;
    INT nActionIndices = _embeddedDim;
    INT offset;

    dcomplex* state_data = state->_dataptr;
    dcomplex* outstate_data = out_state->_dataptr;

    //zero-out output state initially
    for(i=0; i<_dim; i++) outstate_data[i] = 0.0;

    INT b[100]; // could alloc dynamically (LATER?)
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
  |* SVOpCRep_Composed                                                      *|
  \****************************************************************************/
  SVOpCRep_Composed::SVOpCRep_Composed(std::vector<SVOpCRep*> factor_gate_creps, INT dim)
    :SVOpCRep(dim),_factor_gate_creps(factor_gate_creps)
  {
  }
  SVOpCRep_Composed::~SVOpCRep_Composed() { }

  void SVOpCRep_Composed::reinit_factor_op_creps(std::vector<SVOpCRep*> new_factor_gate_creps) {
    _factor_gate_creps.clear(); //removes all elements
    _factor_gate_creps.insert(_factor_gate_creps.end(),
			      new_factor_gate_creps.begin(),
			      new_factor_gate_creps.end());  //inserts contents of new array
  }
  
  SVStateCRep* SVOpCRep_Composed::acton(SVStateCRep* state, SVStateCRep* out_state) {

    DEBUG(std::cout << "Composed acton called!" << std::endl);
    DEBUG(state->print("INPUT"));
    std::size_t nfactors = _factor_gate_creps.size();
    SVStateCRep *tmp2, *tmp1 = out_state; //tmp1 already alloc'd
    SVStateCRep* t; // for swapping

    //if length is 0 just copy state --> outstate
    if(nfactors == 0) {
      out_state->copy_from(state);
      return out_state;
    }

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

  SVStateCRep* SVOpCRep_Composed::adjoint_acton(SVStateCRep* state, SVStateCRep* out_state) {

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
  |* SVOpCRep_Sum                                                           *|
  \****************************************************************************/
  SVOpCRep_Sum::SVOpCRep_Sum(std::vector<SVOpCRep*> factor_creps, INT dim)
    :SVOpCRep(dim),_factor_creps(factor_creps)
  {
  }
  SVOpCRep_Sum::~SVOpCRep_Sum() { }
  
  SVStateCRep* SVOpCRep_Sum::acton(SVStateCRep* state, SVStateCRep* out_state) {

    DEBUG(std::cout << "Sum acton called!" << std::endl);
    DEBUG(state->print("INPUT"));
    std::size_t nfactors = _factor_creps.size();
    SVStateCRep temp_state(_dim);

    //zero-out output state
    for(INT k=0; k<_dim; k++)
      out_state->_dataptr[k] = 0.0;

    //if length is 0 just return "0" state --> outstate
    if(nfactors == 0) return out_state;

    //Act with factors and accumulate into out_state
    for(std::size_t i=0; i < nfactors; i++) {
      _factor_creps[i]->acton(state,&temp_state);
      for(INT k=0; k<_dim; k++)
	out_state->_dataptr[k] += temp_state._dataptr[k];
    }
    DEBUG(out_state->print("OUTPUT"));
    return out_state;
  }

  SVStateCRep* SVOpCRep_Sum::adjoint_acton(SVStateCRep* state, SVStateCRep* out_state) {

    //Note: same as acton(...) but perform adjoint_acton
    DEBUG(std::cout << "Sum adjoint_acton called!" << std::endl);
    DEBUG(state->print("INPUT"));
    std::size_t nfactors = _factor_creps.size();
    SVStateCRep temp_state(_dim);

    //zero-out output state
    for(INT k=0; k<_dim; k++)
      out_state->_dataptr[k] = 0.0;

    //if length is 0 just return "0" state --> outstate
    if(nfactors == 0) return out_state;

    //Act with factors and accumulate into out_state
    for(std::size_t i=0; i < nfactors; i++) {
      _factor_creps[i]->adjoint_acton(state,&temp_state);
      for(INT k=0; k<_dim; k++)
	out_state->_dataptr[k] += temp_state._dataptr[k];
    }
    DEBUG(out_state->print("OUTPUT"));
    return out_state;
  }

  /****************************************************************************	\
  |* SVOpCRep_Exponentiated                                                   *|
  \****************************************************************************/

  SVOpCRep_Exponentiated::SVOpCRep_Exponentiated(SVOpCRep* exponentiated_gate_crep, INT power, INT dim)
    :SVOpCRep(dim)
  {
    _exponentiated_gate_crep = exponentiated_gate_crep;
    _power = power;
  }

  SVOpCRep_Exponentiated::~SVOpCRep_Exponentiated() { }

  SVStateCRep* SVOpCRep_Exponentiated::acton(SVStateCRep* state, SVStateCRep* out_state) {

    DEBUG(std::cout << "Exponentiated acton called!" << std::endl);
    DEBUG(state->print("INPUT"));
    SVStateCRep *tmp2, *tmp1 = out_state; //tmp1 already alloc'd
    SVStateCRep* t; // for swapping

    //if power is 0 just copy state --> outstate
    if(_power == 0) {
      out_state->copy_from(state);
      return out_state;
    }

    //Act with first gate: output in tmp1
    _exponentiated_gate_crep->acton(state, tmp1);
    
    if(_power > 1) {
      SVStateCRep temp_state(_dim); tmp2 = &temp_state;

      //Act with additional gates: tmp1 -> tmp2 then swap, so output in tmp1
      for(INT i=1; i < _power; i++) {
	_exponentiated_gate_crep->acton(tmp1,tmp2);
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

  SVStateCRep* SVOpCRep_Exponentiated::adjoint_acton(SVStateCRep* state, SVStateCRep* out_state) {

    DEBUG(std::cout << "Exponentiated adjoint_acton called!" << std::endl);
    DEBUG(state->print("INPUT"));
    SVStateCRep *tmp2, *tmp1 = out_state; //tmp1 already alloc'd
    SVStateCRep* t; // for swapping

    //Note: same as acton(...) but perform adjoint_acton
    //if power is 0 just copy state --> outstate
    if(_power == 0) {
      out_state->copy_from(state);
      return out_state;
    }

    //Act with first gate: output in tmp1
    _exponentiated_gate_crep->adjoint_acton(state, tmp1);
    
    if(_power > 1) {
      SVStateCRep temp_state(_dim); tmp2 = &temp_state;

      //Act with additional gates: tmp1 -> tmp2 then swap, so output in tmp1
      for(INT i=1; i < _power; i++) {
	_exponentiated_gate_crep->adjoint_acton(tmp1,tmp2);
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

  SBStateCRep::SBStateCRep(INT* smatrix, INT* pvectors, dcomplex* amps, INT namps, INT n) {
    _smatrix = smatrix;
    _pvectors = pvectors;
    _amps = amps;
    _namps = namps;
    _n = n;
    _2n = 2*n;
    _ownmem = false;
    rref(); // initializes _zblock_start
  }

  SBStateCRep::SBStateCRep(INT namps, INT n) {
    _n = n;
    _2n = 2*n;
    _namps = namps;
    _smatrix = new INT[_2n*_2n];
    _pvectors = new INT[_namps*_2n];
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
  
  void SBStateCRep::push_view(std::vector<INT>& view) {
    _view_filters.push_back(view);
  }
  
  void SBStateCRep::pop_view() {
    _view_filters.pop_back();
  }
    
  void SBStateCRep::clifford_update(INT* smatrix, INT* svector, dcomplex* Umx) {
    //vs = (_np.array([1,0],complex), _np.array([0,1],complex)) # (v0,v1)
    DEBUG(std::cout << "Clifford Update BEGIN" << std::endl);
    INT i,k,ip;
    std::vector<std::vector<INT> >::iterator it;
    std::vector<INT>::iterator it2;

    std::vector<INT> qubits(_n);
    for(i=0; i<_n; i++) qubits[i] = i; // start with all qubits being acted on
    for(it=_view_filters.begin(); it != _view_filters.end(); ++it) {
      std::vector<INT>& qfilter = *it;
      std::vector<INT> new_qubits(qfilter.size());
      for(i=0; i < (INT)qfilter.size(); i++)
	new_qubits[i] = qubits[ qfilter[i] ]; // apply each filter
      qubits.resize( new_qubits.size() );
      for(i=0; i < (INT)qfilter.size(); i++)
	qubits[i] = new_qubits[i]; //copy new_qubits -> qubits (maybe a faster way?)
    }

    INT nQ = qubits.size(); //number of qubits being acted on (<= n in general)
    std::vector<std::vector<INT> > sampled_states(_namps);
    std::vector<std::vector<dcomplex> > sampled_amplitudes(_namps);

    INT action_size = pow(2,qubits.size());
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

    //DEBUG!!! - print smx and pvecs
    //std::cout << "S = ";
    //for(i=0; i<_2n*_2n; i++) std::cout << _smatrix[i] << " ";
    //std::cout << std::endl;
    //std::cout << "PS = ";
    //for(i=0; i<_namps*_2n; i++) std::cout << _pvectors[i] << " ";
    //std::cout << std::endl;

    // Step3: Update global amplitudes - Part B
    for(ip=0; ip<_namps; ip++) {
      const std::vector<INT> & base_state = sampled_states[ip];
      const std::vector<dcomplex> & ampls = sampled_amplitudes[ip];

      //DEBUG!!! - print Umx
      //std::cout << "U = ";
      //for(i=0; i<action_size*action_size; i++) std::cout << Umx[i] << " ";
      //std::cout << std::endl;

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
	  std::vector<INT> zvals(base_state);
	  std::vector<INT> k_zvals(nQ);
	  for(i=0; i<nQ; i++) k_zvals[i] = INT( (k >> (nQ-1-i)) & 1);  // hack to extract binary(k)
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

  
  dcomplex SBStateCRep::extract_amplitude(std::vector<INT>& zvals) {
    dcomplex ampl = 0;
    for(INT ip=0; ip < _namps; ip++) {
      ampl += _amps[ip] * canonical_amplitude_of_target(ip, zvals);
    }
    return ampl;
  }

  double SBStateCRep::measurement_probability(std::vector<INT>& zvals) {
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
    INT i;
    for(i=0;i<_2n*_2n;i++) _smatrix[i] = other->_smatrix[i];
    for(i=0;i<_namps*_2n;i++) _pvectors[i] = other->_pvectors[i];
    for(i=0;i<_namps;i++) _amps[i] = other->_amps[i];
    _zblock_start = other->_zblock_start;
    _view_filters.clear();
    for(i=0; i<(INT)other->_view_filters.size(); i++)
      _view_filters.push_back( other->_view_filters[i] );
  }
    

  INT SBStateCRep::udot1(INT i, INT j) {
    // dot(smatrix[:,i].T, U, smatrix[:,j])
    INT ret = 0;
    for(INT k=0; k < _n; k++)
      ret += _smatrix[(k+_n)*_2n+i] * _smatrix[k*_2n+j];
    return ret;
  }

  void SBStateCRep::udot2(INT* out, INT* smatrix1, INT* smatrix2) {
    // out = dot(smatrix1.T, U, smatrix2)
    INT tmp;
    for(INT i=0; i<_2n; i++) {
      for(INT j=0; j<_2n; j++) {
	tmp = 0;
	for(INT k=0; k < _n; k++)
	  tmp += smatrix1[(k+_n)*_2n+i] * smatrix2[k*_2n+j];
	out[i*_2n+j] = tmp;
      }
    }
  }
  
  void SBStateCRep::colsum(INT i, INT j) {
    INT k,row;
    INT* pvec;
    INT* s = _smatrix;
    for(INT p=0; p<_namps; p++) {
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
  
  void SBStateCRep::colswap(INT i, INT j) {
    INT tmp;
    INT* pvec;
    for(INT k=0; k<_2n; k++) {
      tmp = _smatrix[k*_2n+i];
      _smatrix[k*_2n+i] = _smatrix[k*_2n+j];
      _smatrix[k*_2n+j] = tmp;
    }
    for(INT p=0; p<_namps; p++) {
      pvec = &_pvectors[ _2n*p ]; // p-th vector
      tmp = pvec[i];
      pvec[i] = pvec[j];
      pvec[j] = tmp;
    }
  }
  
  void SBStateCRep::rref() {
    //Pass1: form X-block (of *columns*)
    INT i=0, j,k,m; // current *column* (to match ref, but our rep is transposed!)
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


  //result = _np.array(zvals_to_acton,INT);
  dcomplex SBStateCRep::apply_xgen(INT igen, INT pgen, std::vector<INT>& zvals_to_acton,
				   dcomplex ampl, std::vector<INT>& result) {

    dcomplex new_amp = (pgen/2 == 1) ? -ampl : ampl;
    //DEBUG std::cout << "new_amp = "<<new_amp<<std::endl;
    for(std::size_t i=0; i<result.size(); i++)
      result[i] = zvals_to_acton[i];
    
    for(INT j=0; j<_n; j++) { // for each element (literal) in generator
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
	if(result[j] == 1) new_amp *= -1.0;
	//DEBUG std::cout << "new_amp3 = "<<new_amp<<std::endl;
      }
    }
    //DEBUG std::cout << "new_amp4 = "<<new_amp<<std::endl;
    return new_amp;
  }
        
  dcomplex SBStateCRep::get_target_ampl(std::vector<INT>& tgt, std::vector<INT>& anchor, dcomplex anchor_amp, INT ip) {
    // requires just a single pass through X-block
    std::vector<INT> zvals(anchor);
    dcomplex amp = anchor_amp; //start with anchor state
    INT i,j,k, lead = -1;
    DEBUG(std::cout << "BEGIN get_target_ampl" << std::endl);
    
    for(i=0; i<_zblock_start; i++) { // index of current generator
      INT gen_p = _pvectors[ip*_2n + i]; //phase of generator
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
	std::vector<INT> zvals_copy(zvals);
	amp = apply_xgen(i, gen_p, zvals, amp, zvals_copy);
	zvals = zvals_copy; //will this work (copy)?

	//DEBUG!!! - print XGEN return val
	//std::cout << "Resulting amp = " << amp << " zvals=";
        //for(std::size_t z=0; z<zvals.size(); z++) std::cout << zvals[z];
	//std::cout << std::endl;
                    
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
  
  dcomplex SBStateCRep::canonical_amplitude_of_target(INT ip, std::vector<INT>& target) {
    rref(); // ensure we're in reduced row echelon form
        
    // Stage1: go through Z-block columns and find an "anchor" - the first
    // basis state that is allowed given the Z-block parity constraints.
    // (In Z-block, cols can have only Z,I literals)
    INT i,j;
    DEBUG(std::cout << "CanonicalAmps STAGE1: zblock_start = " << _zblock_start << std::endl);
    std::vector<INT> anchor(_n); // "anchor" basis state (zvals), which gets amplitude 1.0 by definition
    for(i=0; i<_n; i++) anchor[i] = 0;
    
    INT lead = _n;
    for(i=_n-1; i >= _zblock_start; i--) { //index of current generator
      INT gen_p = _pvectors[ip*_2n + i]; //phase of generator
      gen_p = (gen_p + 3*udot1(i,i)) % 4;  //counts number of Y's => -i's
      assert(gen_p == 0 || gen_p == 2);
      DEBUG(std::cout << "STARTING LOOP!" << std::endl);
            
      // get positions of Zs
      std::vector<INT> zpos;
      for(j=0; j<_n; j++) {
	if(_smatrix[(j+_n)*_2n+i] == 1) zpos.push_back(j);
      }      

      // set values of anchor between zpos[0] and lead
      // (between current leading-Z position and the last iteration's,
      //  which marks the point at which anchor has been initialized to)
      INT fixed1s = 0; // relevant number of 1s fixed by the already-initialized part of 'anchor'
      INT target1s = 0; // number of 1s in target state, which we want to check for Z-block compatibility
      std::vector<INT> zpos_to_fill;
      std::vector<INT>::iterator it;
      for(it=zpos.begin(); it!=zpos.end(); ++it) {
	j = *it;
	if(j >= lead) {
	  if(anchor[j] == 1) fixed1s += 1;
	}
	else zpos_to_fill.push_back(j);

	if(target[j] == 1) target1s += 1;
      }

      assert(zpos_to_fill.size() > 0); // structure of rref Z-block should ensure this
      INT parity = gen_p/2;
      INT eff_parity = (parity - (fixed1s % 2)) % 2; // effective parity for zpos_to_fill

      DEBUG(std::cout << "  Current gen = "<<i<<" phase = "<<gen_p<<" zpos="<<zpos.size()<<" fixed1s="<<fixed1s<<" tofill="<<zpos_to_fill.size()<<" eff_parity="<<eff_parity<<" lead="<<lead << std::endl);
      DEBUG(std::cout << "   -anchor: ");
      DEBUG(for(INT dbi=0; dbi<_n; dbi++) std::cout << anchor[dbi] << "  ");
      
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
      DEBUG(for(INT dbi=0; dbi<_n; dbi++) std::cout << anchor[dbi] << "  ");
      DEBUG(std::cout << std::endl);
    }
    
    //Set anchor amplitude to appropriate 1.0/sqrt(2)^s 
    // (by definition - serves as a reference pt)
    // Note: 's' equals the minimum number of generators that are *different*
    // between this state and the basis state we're extracting and ampl for.
    // Since any/all comp. basis state generators can form all and only the
    // Z-literal only (Z-block) generators 's' is simplly the number of
    // X-block generators (= self.zblock_start).
    INT s = _zblock_start;
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
    
  void SBStateCRep::canonical_amplitudes_sample(INT ip, std::vector<INT> qs_to_sample,
						std::vector<INT>& anchor, std::vector<dcomplex>& amp_samples) {
    rref(); // ensure we're in reduced row echelon form

    INT i,j,k;
    INT remaining = pow(2,qs_to_sample.size()); //number we still need to find
    assert(amp_samples.size() == remaining);
    for(i=0; i<remaining; i++) amp_samples[i]= std::nan("empty slot");
    // what we'll eventually return - holds amplitudes of all
    //  variations of qs_to_sample starting from anchor.
        
    // Stage1: go through Z-block columns and find an "anchor" - the first
    // basis state that is allowed given the Z-block parity constraints.
    // (In Z-block, cols can have only Z,I literals)
    assert(anchor.size() == _n); // "anchor" basis state (zvals), which gets amplitude 1.0 by definition
    for(i=0; i<_n; i++) anchor[i] = 0;
    
    INT lead = _n;
    for(i=_n-1; i >= _zblock_start; i--) { //index of current generator
      INT gen_p = _pvectors[ip*_2n + i]; //phase of generator
      gen_p = (gen_p + 3*udot1(i,i)) % 4;  //counts number of Y's => -i's
      assert(gen_p == 0 || gen_p == 2);
            
      // get positions of Zs
      std::vector<INT> zpos;
      for(j=0; j<_n; j++) {
	if(_smatrix[(j+_n)*_2n+i] == 1) zpos.push_back(j);
      }

      // set values of anchor between zpos[0] and lead
      // (between current leading-Z position and the last iteration's,
      //  which marks the point at which anchor has been initialized to)
      INT fixed1s = 0; // relevant number of 1s fixed by the already-initialized part of 'anchor'
      std::vector<INT> zpos_to_fill;
      std::vector<INT>::iterator it;
      for(it=zpos.begin(); it!=zpos.end(); ++it) {
	j = *it;
	if(j >= lead) {
	  if(anchor[j] == 1) fixed1s += 1;
	}
	else zpos_to_fill.push_back(j);
      }
	
      assert(zpos_to_fill.size() > 0); // structure of rref Z-block should ensure this
      INT parity = gen_p/2;
      INT eff_parity = (parity - (fixed1s % 2)) % 2; // effective parity for zpos_to_fill

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
    INT s = _zblock_start;
    dcomplex anchor_amp = 1/(pow(sqrt(2.0),s));
    
    remaining -= 1;
    INT nk = qs_to_sample.size();
    INT anchor_indx = 0;
    for(k=0; k<nk; k++) anchor_indx += anchor[qs_to_sample[k]]*pow(2,(nk-1-k));
    amp_samples[ anchor_indx ] = anchor_amp;
    
    
    //STAGE 2b - for sampling a set
    
    //If we're trying to sample a set, check if any of the amplitudes
    // we're looking for are zero by the Z-block checks.  That is,
    // consider whether anchor with qs_to_sample indices updated
    // passes or fails each check
    for(i=_n-1; i >= _zblock_start; i--) { // index of current generator
      INT gen_p = _pvectors[ip*_2n + i]; //phase of generator
      gen_p = (gen_p + 3*udot1(i,i)) % 4;  //counts number of Y's => -i's
      
      std::vector<INT> zpos;
      for(j=0; j<_n; j++) {
	if(_smatrix[(j+_n)*_2n+i] == 1) zpos.push_back(j);
      }
      
      std::vector<INT> inds;
      std::vector<INT>::iterator it, it2;
      INT fixed1s = 0; // number of 1s in target state, which we want to check for Z-block compatibility
      for(it=zpos.begin(); it!=zpos.end(); ++it) {
	j = *it;
	it2 = std::find(qs_to_sample.begin(),qs_to_sample.end(),j);
	if(it2 != qs_to_sample.end()) { // if j in qs_to_sample
	  INT jpos = it2 - qs_to_sample.begin();
	  inds.push_back( jpos ); // "sample" indices in parity check
	}
	else if(anchor[j] == 1) {
	  fixed1s += 1;
	}
      }
      if(inds.size() > 0) {
	INT parity = (gen_p/2 - (fixed1s % 2)) % 2; // effective parity
	INT* b = new INT[qs_to_sample.size()]; //els are just 0 or 1
	INT bi;
	for(bi=0; bi<(INT)qs_to_sample.size(); bi++) b[bi] = 0;
	k = 0;
	while(true) {
	  // tup == b
	  INT tup_parity = 0;
	  for(INT kk=0; kk<(INT)inds.size(); kk++) tup_parity += b[inds[kk]];
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
	delete [] b;
      }
    }
    
    // Check exit conditions
    if(remaining == 0) return;
    
    // Stage2: move through X-block processing existing amplitudes
    // (or processing only to move toward a target state?)

    std::vector<INT> target(anchor);
    INT* b = new INT[qs_to_sample.size()]; //els are just 0 or 1
    INT bi;
    for(bi=0; bi<(INT)qs_to_sample.size(); bi++) b[bi] = 0;
    k = 0;
    while(true) {
      // tup == b
      if(std::isnan(amp_samples[k].real())) {
	for(INT kk=0; kk<(INT)qs_to_sample.size(); kk++)
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
    delete [] b;
    return;
  }

  void SBStateCRep::apply_clifford_to_frame(INT* s, INT* p, std::vector<INT> qubit_filter) {
    //for now, just embed s,p inside full-size s,p: (TODO: make this function more efficient!)
    INT* full_s = new INT[_2n*_2n];
    INT* full_p = new INT[_2n];

    // Embed s,p inside full_s and full_p
    INT i,j,ne = qubit_filter.size();
    INT two_ne = 2*ne;
    for(i=0; i<_2n; i++) {
      for(j=0; j<_2n; j++) full_s[i*_2n+j] = (i==j) ? 1 : 0; // full_s == identity
    }
    for(i=0; i<_2n; i++) full_p[i] = 0; // full_p = zero
    
    for(INT ii=0; ii<ne; ii++) {
      i = qubit_filter[ii];
      full_p[i] = p[ii];
      full_p[i+_n] = p[ii+ne];
      
      for(INT jj=0; jj<ne; jj++) {
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
    
  
  void SBStateCRep::apply_clifford_to_frame(INT* s, INT* p) {
    INT i,j,k,tmp;
    
    // Below we calculate the s and p for the output state using the formulas from
    // Hostens and De Moor PRA 71, 042315 (2005).

    // out_s = _mtx.dotmod2(s,self.s)
    INT* out_s = new INT[_2n*_2n];
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
    //  INT ii;
    //  INT ne = qubit_filter.size(); // number of qubits s,p act on
    //  
    //  //use qubit_filter - only rows & cols of "full s" corresponding to qubit_filter are non-identity
    //  for(i=0; i<_2n*_2n; i++) out_s[i] = _smatrix[i]; // copy out_s = _smatrix
    //
    //  for(ii=0; ii<qubit_filter.size(); ii++) { // only non-identity rows of "full s"
    //	i = qubit_filter[ii];
    //    for(j=0; j<_2n; j++) {
    //	  tmp = 0;
    //	  for(INT kk=0; kk<qubit_filter.size(); kk++) { // only non-zero cols of non-identity i-th row of "full s"
    //	    k = qubit_filter[kk];
    //	    tmp += s[ii*_2n+kk] * _smatrix[k*_2n+j];
    //	    tmp += s[ii*_2n+(kk+ne)] * _smatrix[(k+_n)*_2n+j];
    //	  }
    //	  out_s[i*_2n+j] = tmp % 2; // all els are mod(2)
    //    }
    //
    //	// part2, for (i+n)-th row of "full s"
    //	i = qubit_filter[ii] + _n;
    //	INT iin = ii + ne;
    //	for(j=0; j<_2n; j++) {
    //	  tmp = 0;
    //	  for(INT kk=0; kk<qubit_filter.size(); kk++) { // only non-zero cols of non-identity i-th row of "full s"
    //	    k = qubit_filter[kk];
    //	    tmp += s[iin*_2n+kk] * _smatrix[k*_2n+j];
    //	    tmp += s[iin*_2n+(kk+ne)] * _smatrix[(k+_n)*_2n+j];
    //	  }
    //	  out_s[i*_2n+j] = tmp % 2; // all els are mod(2)
    //    }
    //  }
    //}

    INT* inner = new INT[_2n*_2n];
    INT* tmp1 = new INT[_2n];
    INT* tmp2 = new INT[_2n*_2n];
    INT* vec = new INT[_2n];
    udot2(inner, s, s);

    // vec = _np.dot(_np.transpose(_smatrix),p - _mtx.diagonal_as_vec(inner))
    for(i=0; i<_2n; i++) tmp1[i] = p[i] - inner[i*_2n+i];
    for(i=0; i<_2n; i++) {
      vec[i] = 0; 
      for(k=0; k<_2n; k++)
	vec[i] += _smatrix[k*_2n+i] * tmp1[k];
    }
	  
    //matrix = 2*_mtx.strictly_upper_triangle(inner)+_mtx.diagonal_as_matrix(inner)
    INT* matrix = inner; //just modify inner in place since we don't need it anymore
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
      INT* pvec = &_pvectors[ _2n*i ]; // i-th vector
      for(k=0; k<_2n; k++) pvec[k] = (pvec[k] + vec[k]) % 4;
    }

    delete [] out_s;
    delete [] inner;
    delete [] tmp1;
    delete [] tmp2;
    delete [] vec;
  }

  void SBStateCRep::print(const char* label) {
    std::cout << "<" << label << " (SBStateCRep - TODO print)>" << std::endl;
  }

  

  /****************************************************************************\
  |* SBEffectCRep                                                             *|
  \****************************************************************************/
  SBEffectCRep::SBEffectCRep(INT* zvals, INT n)
    : _zvals(n)
  {
    for(INT i=0; i<n; i++)
      _zvals[i] = zvals[i];
    _n = n;
  }
  SBEffectCRep::~SBEffectCRep() { }
  
  dcomplex SBEffectCRep::amplitude(SBStateCRep* state) {
    DEBUG(std::cout << "SBEffectCRep::amplitude called! zvals = " << _zvals[0] << std::endl);

    //DEBUG!!!
    //INT i;
    //INT _2n = state->_2n;
    //INT _namps = state->_namps;
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
  |* SBOpCRep                                                               *|
  \****************************************************************************/
  SBOpCRep::SBOpCRep(INT n) {
    _n = n;
  }
  SBOpCRep::~SBOpCRep() { }

  /****************************************************************************\
  |* SBOpCRep_Embedded                                                      *|
  \****************************************************************************/
  SBOpCRep_Embedded::SBOpCRep_Embedded(SBOpCRep* embedded_gate_crep, INT n, INT* qubits, INT nqubits)
    :SBOpCRep(n),_qubits(nqubits)
  {
    _embedded_gate_crep = embedded_gate_crep;
    for(INT i=0; i<nqubits; i++)
      _qubits[i] = qubits[i];
  }
  SBOpCRep_Embedded::~SBOpCRep_Embedded() { }
  
  SBStateCRep* SBOpCRep_Embedded::acton(SBStateCRep* state, SBStateCRep* out_state) {
    DEBUG(std::cout << "Stabilizer Embedded acton called!" << std::endl);
    state->push_view(_qubits);
    _embedded_gate_crep->acton(state, out_state);
    state->pop_view();
    out_state->pop_view(); //should have same view as input state
    return out_state;
  }

  SBStateCRep* SBOpCRep_Embedded::adjoint_acton(SBStateCRep* state, SBStateCRep* out_state) {
    DEBUG(std::cout << "Stabilizer Embedded adjoint_acton called!" << std::endl);
    state->push_view(_qubits);
    _embedded_gate_crep->adjoint_acton(state, out_state);
    state->pop_view();
    out_state->pop_view(); //should have same view as input state
    return out_state;
  }


  /****************************************************************************\
  |* SBOpCRep_Composed                                                      *|
  \****************************************************************************/
  SBOpCRep_Composed::SBOpCRep_Composed(std::vector<SBOpCRep*> factor_gate_creps, INT n)
    :SBOpCRep(n),_factor_gate_creps(factor_gate_creps)
  {
  }
  SBOpCRep_Composed::~SBOpCRep_Composed() { }

  SBStateCRep* SBOpCRep_Composed::acton(SBStateCRep* state, SBStateCRep* out_state) {
    DEBUG(std::cout << "Stabilizer Composed acton called!" << std::endl);
    std::size_t nfactors = _factor_gate_creps.size();
    SBStateCRep *tmp2, *tmp1 = out_state; //tmp1 already alloc'd
    SBStateCRep* t; // for swapping

    //if length is 0 just copy state --> outstate
    if(nfactors == 0) {
      out_state->copy_from(state);
      return out_state;
    }

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

  SBStateCRep* SBOpCRep_Composed::adjoint_acton(SBStateCRep* state, SBStateCRep* out_state) {
    DEBUG(std::cout << "Stabilizer Composed adjoint_acton called!" << std::endl);
    std::size_t nfactors = _factor_gate_creps.size();
    SBStateCRep *tmp2, *tmp1 = out_state; //tmp1 already alloc'd
    SBStateCRep* t; // for swapping

    //Act with first gate: output in tmp1
    _factor_gate_creps[nfactors-1]->adjoint_acton(state, tmp1);

    if(nfactors > 1) {
      SBStateCRep temp_state(tmp1->_namps,_n); tmp2 = &temp_state;

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
    return out_state;
  }



  /****************************************************************************\
  |* SBOpCRep_Sum                                                           *|
  \****************************************************************************/
  SBOpCRep_Sum::SBOpCRep_Sum(std::vector<SBOpCRep*> factor_creps, INT n)
    :SBOpCRep(n),_factor_creps(factor_creps)
  {
  }
  SBOpCRep_Sum::~SBOpCRep_Sum() { }

  SBStateCRep* SBOpCRep_Sum::acton(SBStateCRep* state, SBStateCRep* out_state) {
    assert(false); // need further stabilizer frame support to represent the sum of stabilizer states
    return NULL; //to avoid compiler warning
  }

  SBStateCRep* SBOpCRep_Sum::adjoint_acton(SBStateCRep* state, SBStateCRep* out_state) {
    assert(false); // need further stabilizer frame support to represent the sum of stabilizer states
    return NULL; //to avoid compiler warning
  }

  
  /****************************************************************************	\
  |* SBOpCRep_Exponentiated                                                   *|
  \****************************************************************************/

  SBOpCRep_Exponentiated::SBOpCRep_Exponentiated(SBOpCRep* exponentiated_gate_crep, INT power, INT n)
    :SBOpCRep(n)
  {
    _exponentiated_gate_crep = exponentiated_gate_crep;
    _power = power;
  }

  SBOpCRep_Exponentiated::~SBOpCRep_Exponentiated() { }

  SBStateCRep* SBOpCRep_Exponentiated::acton(SBStateCRep* state, SBStateCRep* out_state) {

    DEBUG(std::cout << "Exponentiated acton called!" << std::endl);
    DEBUG(state->print("INPUT"));
    SBStateCRep *tmp2, *tmp1 = out_state; //tmp1 already alloc'd
    SBStateCRep* t; // for swapping

    //if power is 0 just copy state --> outstate
    if(_power == 0) {
      out_state->copy_from(state);
      return out_state;
    }

    //Act with first gate: output in tmp1
    _exponentiated_gate_crep->acton(state, tmp1);
    
    if(_power > 1) {
      SBStateCRep temp_state(tmp1->_namps,_n); tmp2 = &temp_state;

      //Act with additional gates: tmp1 -> tmp2 then swap, so output in tmp1
      for(INT i=1; i < _power; i++) {
	_exponentiated_gate_crep->acton(tmp1,tmp2);
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

  SBStateCRep* SBOpCRep_Exponentiated::adjoint_acton(SBStateCRep* state, SBStateCRep* out_state) {

    DEBUG(std::cout << "Exponentiated adjoint_acton called!" << std::endl);
    DEBUG(state->print("INPUT"));
    SBStateCRep *tmp2, *tmp1 = out_state; //tmp1 already alloc'd
    SBStateCRep* t; // for swapping

    //Note: same as acton(...) but perform adjoint_acton
    //if power is 0 just copy state --> outstate
    if(_power == 0) {
      out_state->copy_from(state);
      return out_state;
    }

    //Act with first gate: output in tmp1
    _exponentiated_gate_crep->adjoint_acton(state, tmp1);
    
    if(_power > 1) {
      SBStateCRep temp_state(tmp1->_namps,_n); tmp2 = &temp_state;

      //Act with additional gates: tmp1 -> tmp2 then swap, so output in tmp1
      for(INT i=1; i < _power; i++) {
	_exponentiated_gate_crep->adjoint_acton(tmp1,tmp2);
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
  |* SBOpCRep_Clifford                                                      *|
  \****************************************************************************/
  SBOpCRep_Clifford::SBOpCRep_Clifford(INT* smatrix, INT* svector, dcomplex* unitary,
					   INT* smatrix_inv, INT* svector_inv, dcomplex* unitary_adj, INT n)
    :SBOpCRep(n)
  {
    _smatrix = smatrix;
    _svector = svector;
    _smatrix_inv = smatrix_inv;
    _svector_inv = svector_inv;
    _unitary = unitary;
    _unitary_adj = unitary_adj;

    //DEBUG!!!
    //std::cout << "IN SBOpCRep_Clifford CONSTRUCTOR U = ";
    //for(int i=0; i<2*2; i++) std::cout << _unitary_adj[i] << " ";
    //std::cout << std::endl;

  }
  SBOpCRep_Clifford::~SBOpCRep_Clifford() { }
  
  SBStateCRep* SBOpCRep_Clifford::acton(SBStateCRep* state, SBStateCRep* out_state) {
    DEBUG(std::cout << "Stabilizer Clifford acton called!" << std::endl);
    out_state->copy_from(state);
    out_state->clifford_update(_smatrix, _svector, _unitary);
    return out_state;
  }

  SBStateCRep* SBOpCRep_Clifford::adjoint_acton(SBStateCRep* state, SBStateCRep* out_state) {
    DEBUG(std::cout << "Stabilizer Clifford adjoint_acton called!" << std::endl);

    //DEBUG!!!
    //std::cout << "AT SBOpCRep_Clifford::adjoint_acton U = ";
    //for(INT i=0; i<2*2; i++) std::cout << _unitary_adj[i] << " ";
    //std::cout << std::endl;
    
    out_state->copy_from(state);
    out_state->clifford_update(_smatrix_inv, _svector_inv, _unitary_adj);
    return out_state;
  


  
  //TODO - implement in C

  //_fastcalc.fast_kron(scratch, _fast_kron_array, _fast_kron_factordims);

  //        state = _mt.expm_multiply_fast(self.err_gen_prep, state) # (N,) -> (N,) shape mapping

  //_fastcalc.embedded_fast_acton_sparse(self.embedded_op.acton,
  //                                         output_state, state,
  //                                         self.noop_incrementers,
  //                                         self.numBasisEls_noop_blankaction,
  //                                         self.baseinds)


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


  void expm_multiply_simple_core_rep(DMOpCRep* A_rep, double* B,
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

    DMStateCRep B_st(B, N); // just a wrapper
    DMStateCRep scratch_st(scratch, N); // just a wrapper

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


  // OTHER Classes

  /****************************************************************************\
  |* PolynomialCRep                                                                 *|
  \****************************************************************************/
  
  PolynomialCRep::PolynomialCRep() {
    _coeffs = std::unordered_map<PolynomialVarsIndex, dcomplex>();
    _max_num_vars = 0;
    _vindices_per_int = 0;
  }
  
  PolynomialCRep::PolynomialCRep(std::unordered_map<PolynomialVarsIndex, dcomplex> coeffs, INT max_num_vars, INT vindices_per_int) {
    _coeffs = coeffs;
    _max_num_vars = max_num_vars;
    _vindices_per_int = vindices_per_int;
  }

  PolynomialCRep::PolynomialCRep(const PolynomialCRep& other) {
    _coeffs = other._coeffs;
    _max_num_vars = other._max_num_vars;
  }

  PolynomialCRep::~PolynomialCRep() { }

  PolynomialCRep PolynomialCRep::mult(const PolynomialCRep& other) {
    std::unordered_map<PolynomialVarsIndex, dcomplex>::iterator it1, itk;
    std::unordered_map<PolynomialVarsIndex, dcomplex>::const_iterator it2;
    std::unordered_map<PolynomialVarsIndex, dcomplex> result;
    dcomplex val;
    PolynomialVarsIndex k;

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
    PolynomialCRep ret(result, _max_num_vars, _vindices_per_int);
    return ret; // need a copy constructor?
  }

  PolynomialCRep PolynomialCRep::abs_mult(const PolynomialCRep& other) {
    std::unordered_map<PolynomialVarsIndex, dcomplex>::iterator it1, itk;
    std::unordered_map<PolynomialVarsIndex, dcomplex>::const_iterator it2;
    std::unordered_map<PolynomialVarsIndex, dcomplex> result;
    dcomplex val;
    PolynomialVarsIndex k;

    for(it1 = _coeffs.begin(); it1 != _coeffs.end(); ++it1) {
      for(it2 = other._coeffs.begin(); it2 != other._coeffs.end(); ++it2) {
	k = mult_vinds_ints(it1->first, it2->first); //key to add
	itk = result.find(k);
	val = std::abs(it1->second * it2->second);
	if(itk != result.end())
	  itk->second = itk->second + val;
	else result[k] = val;
      }
    }
    PolynomialCRep ret(result, _max_num_vars, _vindices_per_int);
    return ret; // need a copy constructor?
  }


  void PolynomialCRep::add_inplace(const PolynomialCRep& other) {
    std::unordered_map<PolynomialVarsIndex, dcomplex>::const_iterator it2;
      std::unordered_map<PolynomialVarsIndex, dcomplex>::iterator itk;
    dcomplex val, newval;
    PolynomialVarsIndex k;

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

  void PolynomialCRep::add_abs_inplace(const PolynomialCRep& other) {
    std::unordered_map<PolynomialVarsIndex, dcomplex>::const_iterator it2;
      std::unordered_map<PolynomialVarsIndex, dcomplex>::iterator itk;
    double val;
    PolynomialVarsIndex k;

    for(it2 = other._coeffs.begin(); it2 != other._coeffs.end(); ++it2) {
      k = it2->first; // key
      val = std::abs(it2->second); // value
      if(val > 1e-12) {
          itk = _coeffs.find(k);
          if(itk != _coeffs.end()) {
              itk->second = itk->second + (dcomplex)val; // note: += doens't work here (complex Cython?)
          }
          else {
              _coeffs[k] = (dcomplex)val;
          }
      }
    }
  }

  PolynomialCRep PolynomialCRep::abs() {
    std::unordered_map<PolynomialVarsIndex, dcomplex> result;
    std::unordered_map<PolynomialVarsIndex, dcomplex>::iterator it;
    for(it = _coeffs.begin(); it != _coeffs.end(); ++it) {
        result[it->first] = std::abs(it->second);
    }
    
    PolynomialCRep ret(result, _max_num_vars, _vindices_per_int);
    return ret; // need a copy constructor?
  }

  void PolynomialCRep::scale(dcomplex scale) {
    std::unordered_map<PolynomialVarsIndex, dcomplex>::iterator it;
    for(it = _coeffs.begin(); it != _coeffs.end(); ++it) {
      it->second = it->second * scale; // note: *= doesn't work here (complex Cython?)
    }
  }

  void PolynomialCRep::add_scalar_to_all_coeffs_inplace(dcomplex offset) {
    std::unordered_map<PolynomialVarsIndex, dcomplex>::iterator it;
    for(it = _coeffs.begin(); it != _coeffs.end(); ++it) {
      it->second = it->second + offset; // note: += doesn't work here (complex Cython?)
    }
  }


  PolynomialVarsIndex PolynomialCRep::vinds_to_int(std::vector<INT> vinds) {
    INT ret, end, m;
    INT sz = ceil(1.0 * vinds.size() / _vindices_per_int);
    PolynomialVarsIndex ret_tup(sz);
    for(INT k=0; k<sz-1; k++) {
      ret = 0; m = 1;
      for(std::size_t i=_vindices_per_int*k; i<_vindices_per_int*(k+1); i++) { // last tuple index is most significant
	ret += (vinds[i]+1)*m;
	m *= _max_num_vars+1;
      }
      ret_tup._parts[k] = ret;
    }
    if(sz > 0) { //final iteration has different uppper limit
      ret = 0; m = 1;
      for(std::size_t i=_vindices_per_int*(sz-1); i<vinds.size(); i++) { // last tuple index is most significant
	ret += (vinds[i]+1)*m;
	m *= _max_num_vars+1;
      }
      ret_tup._parts[sz-1] = ret;
    }
    return ret_tup;
  }
  
  std::vector<INT> PolynomialCRep::int_to_vinds(PolynomialVarsIndex indx_tup) {
    std::vector<INT> ret;
    INT nxt, i, indx;
    for(std::size_t i=0; i < indx_tup._parts.size(); i++) {
      indx = indx_tup._parts[i];
      while(indx != 0) {
        nxt = indx / (_max_num_vars+1);
	i = indx - nxt*(_max_num_vars+1);
	ret.push_back(i-1);
	indx = nxt;
	//assert(indx >= 0);
      }
    }
    std::sort(ret.begin(),ret.end());
    return ret;
  }
  
  PolynomialVarsIndex PolynomialCRep::mult_vinds_ints(PolynomialVarsIndex i1, PolynomialVarsIndex i2) {
    // multiply vinds corresponding to i1 & i2 and return resulting integer
    std::vector<INT> vinds1 = int_to_vinds(i1);
    std::vector<INT> vinds2 = int_to_vinds(i2);
    vinds1.insert( vinds1.end(), vinds2.begin(), vinds2.end() );
    std::sort(vinds1.begin(),vinds1.end());
    return vinds_to_int(vinds1);
  }
  
  /****************************************************************************\
  |* SVTermCRep                                                               *|
  \****************************************************************************/
    
  SVTermCRep::SVTermCRep(PolynomialCRep* coeff, double magnitude, double logmagnitude,
			 SVStateCRep* pre_state, SVStateCRep* post_state,
			 std::vector<SVOpCRep*> pre_ops, std::vector<SVOpCRep*> post_ops) {
    _coeff = coeff;
    _magnitude = magnitude;
    _logmagnitude = logmagnitude;
    _pre_state = pre_state;
    _post_state = post_state;
    _pre_effect = NULL;
    _post_effect = NULL;
    _pre_ops = pre_ops;
    _post_ops = post_ops;
  }
  
  SVTermCRep::SVTermCRep(PolynomialCRep* coeff, double magnitude, double logmagnitude,
			 SVEffectCRep* pre_effect, SVEffectCRep* post_effect,
			 std::vector<SVOpCRep*> pre_ops, std::vector<SVOpCRep*> post_ops) {
    _coeff = coeff;
    _magnitude = magnitude;
    _logmagnitude = logmagnitude;
    _pre_state = NULL;
    _post_state = NULL;
    _pre_effect = pre_effect;
    _post_effect = post_effect;
    _pre_ops = pre_ops;
    _post_ops = post_ops;
  }
  
  SVTermCRep::SVTermCRep(PolynomialCRep* coeff, double magnitude, double logmagnitude,
			 std::vector<SVOpCRep*> pre_ops, std::vector<SVOpCRep*> post_ops) {
    _coeff = coeff;
    _magnitude = magnitude;
    _logmagnitude = logmagnitude;
    _pre_state = NULL;
    _post_state = NULL;
    _pre_effect = NULL;
    _post_effect = NULL;
    _pre_ops = pre_ops;
    _post_ops = post_ops;
  }

  /****************************************************************************\
  |* SVTermDirectCRep                                                         *|
  \****************************************************************************/
    
  SVTermDirectCRep::SVTermDirectCRep(dcomplex coeff, double magnitude, double logmagnitude,
				     SVStateCRep* pre_state, SVStateCRep* post_state,
				     std::vector<SVOpCRep*> pre_ops, std::vector<SVOpCRep*> post_ops) {
    _coeff = coeff;
    _magnitude = magnitude;
    _logmagnitude = logmagnitude;
    _pre_state = pre_state;
    _post_state = post_state;
    _pre_effect = NULL;
    _post_effect = NULL;
    _pre_ops = pre_ops;
    _post_ops = post_ops;
  }
  
  SVTermDirectCRep::SVTermDirectCRep(dcomplex coeff, double magnitude, double logmagnitude,
				     SVEffectCRep* pre_effect, SVEffectCRep* post_effect,
				     std::vector<SVOpCRep*> pre_ops, std::vector<SVOpCRep*> post_ops) {
    _coeff = coeff;
    _magnitude = magnitude;
    _logmagnitude = logmagnitude;
    _pre_state = NULL;
    _post_state = NULL;
    _pre_effect = pre_effect;
    _post_effect = post_effect;
    _pre_ops = pre_ops;
    _post_ops = post_ops;
  }
  
  SVTermDirectCRep::SVTermDirectCRep(dcomplex coeff, double magnitude, double logmagnitude,
				     std::vector<SVOpCRep*> pre_ops,
				     std::vector<SVOpCRep*> post_ops) {
    _coeff = coeff;
    _magnitude = magnitude;
    _logmagnitude = logmagnitude;
    _pre_state = NULL;
    _post_state = NULL;
    _pre_effect = NULL;
    _post_effect = NULL;
    _pre_ops = pre_ops;
    _post_ops = post_ops;
  }

  /****************************************************************************\
  |* SBTermCRep                                                               *|
  \****************************************************************************/
    
  SBTermCRep::SBTermCRep(PolynomialCRep* coeff, double magnitude, double logmagnitude,
			 SBStateCRep* pre_state, SBStateCRep* post_state,
			 std::vector<SBOpCRep*> pre_ops, std::vector<SBOpCRep*> post_ops) {
    _coeff = coeff;
    _magnitude = magnitude;
    _logmagnitude = logmagnitude;
    _pre_state = pre_state;
    _post_state = post_state;
    _pre_effect = NULL;
    _post_effect = NULL;
    _pre_ops = pre_ops;
    _post_ops = post_ops;
  }
  
  SBTermCRep::SBTermCRep(PolynomialCRep* coeff, double magnitude, double logmagnitude,
			 SBEffectCRep* pre_effect, SBEffectCRep* post_effect,
			 std::vector<SBOpCRep*> pre_ops, std::vector<SBOpCRep*> post_ops) {
    _coeff = coeff;
    _magnitude = magnitude;
    _logmagnitude = logmagnitude;
    _pre_state = NULL;
    _post_state = NULL;
    _pre_effect = pre_effect;
    _post_effect = post_effect;
    _pre_ops = pre_ops;
    _post_ops = post_ops;
  }
  
  SBTermCRep::SBTermCRep(PolynomialCRep* coeff, double magnitude, double logmagnitude,
			 std::vector<SBOpCRep*> pre_ops,
			 std::vector<SBOpCRep*> post_ops) {
    _coeff = coeff;
    _magnitude = magnitude;
    _logmagnitude = logmagnitude;
    _pre_state = NULL;
    _post_state = NULL;
    _pre_effect = NULL;
    _post_effect = NULL;
    _pre_ops = pre_ops;
    _post_ops = post_ops;
  }

  
}
