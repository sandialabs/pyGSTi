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

namespace CReps_stabilizer {

  /**************************************************************************** \
  |* EffectCRep                                                             *|
  \****************************************************************************/
  EffectCRep::EffectCRep(INT n)  {
    _n = n;
  }
  EffectCRep::~EffectCRep() { }
  

  /****************************************************************************\
  |* EffectCRep_Computational                                                 *|
  \****************************************************************************/

  EffectCRep_Computational::EffectCRep_Computational(INT* zvals, INT n)
    : EffectCRep(n), _zvals(n)
  {
    for(INT i=0; i<n; i++)
      _zvals[i] = zvals[i];
    _n = n;
  }
  EffectCRep_Computational::~EffectCRep_Computational() { }
  
  dcomplex EffectCRep_Computational::amplitude(StateCRep* state) {
    DEBUG(std::cout << "EffectCRep::amplitude called! zvals = " << _zvals[0] << std::endl);

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

  double EffectCRep_Computational::probability(StateCRep* state) {
    DEBUG(std::cout << "EffectCRep::probability called! zvals = " << _zvals[0] << std::endl);
    return pow(std::abs(amplitude(state)),2);
  }

    
  /**************************************************************************** \
  |* EffectCRep_Composed                                                      *|
  \****************************************************************************/

  EffectCRep_Composed::EffectCRep_Composed(OpCRep* errgen_oprep,
					   EffectCRep* effect_rep,
					   INT errgen_id, INT n)
    :EffectCRep(n)
  {
    _errgen_ptr = errgen_oprep;
    _effect_ptr = effect_rep;
    _errgen_id = errgen_id;
  }
  
  EffectCRep_Composed::~EffectCRep_Composed() { }

  double EffectCRep_Composed::probability(StateCRep* state) {
      StateCRep outState(state->_namps, _n);
    _errgen_ptr->acton(state, &outState);
    return _effect_ptr->probability(&outState);
  }

  dcomplex EffectCRep_Composed::amplitude(StateCRep* state) {
      StateCRep outState(state->_namps, _n);
      _errgen_ptr->acton(state, &outState);
      return _effect_ptr->amplitude(&outState);
  }
}
