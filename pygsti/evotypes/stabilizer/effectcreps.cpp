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

  /**************************************************************************** \
  |* EffectCRep                                                             *|
  \****************************************************************************/
  EffectCRep::EffectCRep(INT* zvals, INT n)
    : _zvals(n)
  {
    for(INT i=0; i<n; i++)
      _zvals[i] = zvals[i];
    _n = n;
  }
  EffectCRep::~EffectCRep() { }
  
  dcomplex EffectCRep::amplitude(StateCRep* state) {
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

  double EffectCRep::probability(StateCRep* state) {
    DEBUG(std::cout << "EffectCRep::probability called! zvals = " << _zvals[0] << std::endl);
    return pow(std::abs(amplitude(state)),2);
  }

}
