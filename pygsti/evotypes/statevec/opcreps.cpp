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

namespace CReps {

  /****************************************************************************\
  |* OpCRep                                                                   *|
  \****************************************************************************/

  OpCRep::OpCRep(INT dim) {
    _dim = dim;
  }
  OpCRep::~OpCRep() { }


  /****************************************************************************\
  |* OpCRep_DenseUnitary                                                      *|
  \****************************************************************************/

  OpCRep_DenseUnitary::OpCRep_DenseUnitary(dcomplex* data, INT dim)
    :OpCRep(dim)
  {
    _dataptr = data;
  }
  OpCRep_DenseUnitary::~OpCRep_DenseUnitary() { }

  StateCRep* OpCRep_DenseUnitary::acton(StateCRep* state,
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

  StateCRep* OpCRep_DenseUnitary::adjoint_acton(StateCRep* state,
                                                StateCRep* outstate) {
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
  |* OpCRep_Embedded                                                          *|
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

    DEBUG(std::cout << " Embedded acton called!" << std::endl);
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
	  offset += _blocksizes[i];
	}
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

    dcomplex* state_data = state->_dataptr;
    dcomplex* outstate_data = out_state->_dataptr;

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
  |* OpCRep_Composed                                                          *|
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
  |* OpCRep_Sum                                                               *|
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
  |* OpCRep_Repeated                                                          *|
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

}
