#define NULL 0

#include <iostream>
#include <complex>
#include <assert.h>
#include <algorithm>    // std::find

#include "statecreps.h"
#include "opcreps.h"
#include "effectcreps.h"
//#include <pthread.h>

//using namespace std::complex_literals;

//#define DEBUG(x) x
#define DEBUG(x) 

namespace CReps {
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
  EffectCRep_Dense::EffectCRep_Dense(dcomplex* data, INT dim)
    :EffectCRep(dim)
  {
    _dataptr = data;
  }

  EffectCRep_Dense::~EffectCRep_Dense() { }

  double EffectCRep_Dense::probability(StateCRep* state) {
    return (double)pow(std::abs(amplitude(state)),2);
  }

  dcomplex EffectCRep_Dense::amplitude(StateCRep* state) {
    dcomplex ret = 0.0;
    for(INT i=0; i< _dim; i++) {
      ret += std::conj(_dataptr[i]) * state->_dataptr[i];
    }
    return ret;
  }

  
  /****************************************************************************\
  |* EffectCRep_TensorProd                                                  *|
  \****************************************************************************/

  EffectCRep_TensorProd::EffectCRep_TensorProd(dcomplex* kron_array,
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
    return (double)pow(std::abs(amplitude(state)),2);
  }
  
  dcomplex EffectCRep_TensorProd::amplitude(StateCRep* state) {
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
  |* EffectCRep_Computational                                               *|
  \****************************************************************************/

  EffectCRep_Computational::EffectCRep_Computational(INT nfactors, INT zvals_int, INT dim)
    :EffectCRep(dim)
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

  EffectCRep_Computational::~EffectCRep_Computational() { }

  double EffectCRep_Computational::probability(StateCRep* state) {
    return (double)pow(std::abs(amplitude(state)),2);
  }
  
  dcomplex EffectCRep_Computational::amplitude(StateCRep* state) {
    //There's only a single nonzero index with element == 1.0, so dotprod is trivial
    return state->_dataptr[_nonzero_index];
  }
}
