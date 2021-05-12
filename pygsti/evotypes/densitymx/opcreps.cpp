#define NULL 0

#include <iostream>
#include <complex>
#include <assert.h>
#include <algorithm>    // std::find
#include "opcreps.h"
//#include <pthread.h>

//using namespace std::complex_literals;

//#define DEBUG(x) x
#define DEBUG(x) 

namespace CReps {
  // DENSE MATRIX (DM) propagation
  
  /****************************************************************************\
  |* StateCRep                                                              *|
  \****************************************************************************/
  StateCRep::StateCRep(INT dim) {
    _dataptr = new double[dim];
    for(INT i=0; i<dim; i++) _dataptr[i] = 0;
    _dim = dim;
    _ownmem = true;
  }
  
  StateCRep::StateCRep(double* data, INT dim, bool copy=false) {
    //DEGUG std::cout << "StateCRep initialized w/dim = " << dim << std::endl;
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

  StateCRep::~StateCRep() {
    if(_ownmem && _dataptr != NULL)
      delete [] _dataptr;
  }

  void StateCRep::print(const char* label) {
    std::cout << label << " = [";
    for(INT i=0; i<_dim; i++) std::cout << _dataptr[i] << " ";
    std::cout << "]" << std::endl;
  }

  void StateCRep::copy_from(StateCRep* st) {
    assert(_dim == st->_dim);
    for(INT i=0; i<_dim; i++)
      _dataptr[i] = st->_dataptr[i];
  }

  /****************************************************************************\
  |* EffectCRep                                                             *|
  \****************************************************************************/
  EffectCRep::EffectCRep(INT dim) {
    _dim = dim;
  }
  EffectCRep::~EffectCRep() { }


  /****************************************************************************\
  |* EffectCRep_Dense                                                       *|
  \****************************************************************************/
  EffectCRep_Dense::EffectCRep_Dense(double* data, INT dim)
    :EffectCRep(dim)
  {
    _dataptr = data;
  }

  EffectCRep_Dense::~EffectCRep_Dense() { }

  double EffectCRep_Dense::probability(StateCRep* state) {
    double ret = 0.0;
    for(INT i=0; i< _dim; i++) {
      ret += _dataptr[i] * state->_dataptr[i];
    }
    return ret;
  }

  
  /****************************************************************************\
  |* EffectCRep_TensorProd                                                  *|
  \****************************************************************************/

  EffectCRep_TensorProd::EffectCRep_TensorProd(double* kron_array,
						   INT* factordims, INT nfactors,
						   INT max_factor_dim, INT dim) 
    :EffectCRep(dim)
  {
    _kron_array = kron_array;
    _max_factor_dim = max_factor_dim;
    _factordims = factordims;
    _nfactors = nfactors;
  }

  EffectCRep_TensorProd::~EffectCRep_TensorProd() { }
    
  double EffectCRep_TensorProd::probability(StateCRep* state) {
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
  |* EffectCRep_Computational                                               *|
  \****************************************************************************/

  EffectCRep_Computational::EffectCRep_Computational(INT nfactors, INT zvals_int, double abs_elval, INT dim)
    :EffectCRep(dim)
  {
    _nfactors = nfactors;
    _zvals_int = zvals_int;
    _abs_elval = abs_elval;
  }

  EffectCRep_Computational::~EffectCRep_Computational() { }
    
  double EffectCRep_Computational::probability(StateCRep* state) {
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

  INT EffectCRep_Computational::parity(INT x) {
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
  |* EffectCRep_Errgen                                                      *|
  \****************************************************************************/

  EffectCRep_Errgen::EffectCRep_Errgen(OpCRep* errgen_oprep,
					   EffectCRep* effect_rep,
					   INT errgen_id, INT dim)
    :EffectCRep(dim)
  {
    _errgen_ptr = errgen_oprep;
    _effect_ptr = effect_rep;
    _errgen_id = errgen_id;
  }
  
  EffectCRep_Errgen::~EffectCRep_Errgen() { }
  
  double EffectCRep_Errgen::probability(StateCRep* state) {
    StateCRep outState(_dim);
    _errgen_ptr->acton(state, &outState);
    return _effect_ptr->probability(&outState);
  }

  double EffectCRep_Errgen::probability_using_cache(StateCRep* state, StateCRep* errgen_on_state, INT& errgen_id) {
    if(errgen_id != _errgen_id) {
      _errgen_ptr->acton(state, errgen_on_state);
      errgen_id = _errgen_id;
    }
    return _effect_ptr->probability(errgen_on_state);
  }


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
  OpCRep_Sum::OpCRep_Sum(std::vector<OpCRep*> factor_creps, INT dim)
    :OpCRep(dim),_factor_creps(factor_creps)
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
    for(std::size_t i=0; i < nfactors; i++) {
      _factor_creps[i]->acton(state,&temp_state);
      for(INT k=0; k<_dim; k++)
	out_state->_dataptr[k] += temp_state._dataptr[k];
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
    for(std::size_t i=0; i < nfactors; i++) {
      _factor_creps[i]->adjoint_acton(state,&temp_state);
      for(INT k=0; k<_dim; k++)
	out_state->_dataptr[k] += temp_state._dataptr[k];
    }
    DEBUG(out_state->print("OUTPUT"));
    return out_state;
  }

  /****************************************************************************\
  |* OpCRep_Repeated                                                   *|
  \****************************************************************************/

  OpCRep_Repeated::OpCRep_Repeated(OpCRep* exponentiated_gate_crep, INT num_repetitions, INT dim)
    :OpCRep(dim)
  {
    _exponentiated_gate_crep = exponentiated_gate_crep;
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
	_repreated_crep->adjoint_acton(tmp1,tmp2);
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
			double mu, double eta, INT m_star, INT s, INT dim,
			double* unitarypost_data, INT* unitarypost_indices,
			INT* unitarypost_indptr, INT unitarypost_nnz)
    :OpCRep(dim)
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
      
  OpCRep_ExpErrorgen::~OpCRep_ExpErrorgen() { }

  StateCRep* OpCRep_ExpErrorgen::acton(StateCRep* state, StateCRep* out_state)
  {
    INT i, j;
    StateCRep* init_state = new StateCRep(_dim);
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
}
