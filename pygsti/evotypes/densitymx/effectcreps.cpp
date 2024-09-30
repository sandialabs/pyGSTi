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

namespace CReps_densitymx {
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

//  INT EffectCRep_Computational::parity(INT x) {
//    // int64-bit specific
//    x = (x & 0x00000000FFFFFFFF)^(x >> 32);
//    x = (x & 0x000000000000FFFF)^(x >> 16);
//    x = (x & 0x00000000000000FF)^(x >> 8);
//    x = (x & 0x000000000000000F)^(x >> 4);
//    x = (x & 0x0000000000000003)^(x >> 2);
//    x = (x & 0x0000000000000001)^(x >> 1);
//    return x & 1; // return the last bit (0 or 1)
//  }

  inline INT EffectCRep_Computational::parity(INT x) {
    x ^= (x >> 32);
    x ^= (x >> 16);
    x ^= (x >> 8);
    x ^= (x >> 4);
    x ^= (x >> 2);
    x ^= (x >> 1);
    return x & 1; // Return the last bit
  }


  /****************************************************************************\
  |* EffectCRep_Composed                                                    *|
  \****************************************************************************/

  EffectCRep_Composed::EffectCRep_Composed(OpCRep* errgen_oprep,
					   EffectCRep* effect_rep,
					   INT errgen_id, INT dim)
    :EffectCRep(dim)
  {
    _errgen_ptr = errgen_oprep;
    _effect_ptr = effect_rep;
    _errgen_id = errgen_id;
  }
  
  EffectCRep_Composed::~EffectCRep_Composed() { }
  
  double EffectCRep_Composed::probability(StateCRep* state) {
    StateCRep outState(_dim);
    _errgen_ptr->acton(state, &outState);
    return _effect_ptr->probability(&outState);
  }

  double EffectCRep_Composed::probability_using_cache(StateCRep* state, StateCRep* errgen_on_state, INT& errgen_id) {
    if(errgen_id != _errgen_id) {
      _errgen_ptr->acton(state, errgen_on_state);
      errgen_id = _errgen_id;
    }
    return _effect_ptr->probability(errgen_on_state);
  }
}
