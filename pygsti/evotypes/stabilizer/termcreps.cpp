

#include <iostream>
#include <complex>
#include <assert.h>

#include "basecreps.h"
#include "statecreps.h"
#include "opcreps.h"
#include "effectcreps.h"
#include "termcreps.h"
//#include <pthread.h>

//using namespace std::complex_literals;

//#define DEBUG(x) x
#define DEBUG(x) 

namespace CReps_stabilizer {

  /****************************************************************************\
  |* TermCRep                                                                 *|
  \****************************************************************************/
    
  TermCRep::TermCRep(CReps::PolynomialCRep* coeff, double magnitude, double logmagnitude,
			 StateCRep* pre_state, StateCRep* post_state,
			 std::vector<OpCRep*> pre_ops, std::vector<OpCRep*> post_ops) {
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
  
  TermCRep::TermCRep(CReps::PolynomialCRep* coeff, double magnitude, double logmagnitude,
			 EffectCRep* pre_effect, EffectCRep* post_effect,
			 std::vector<OpCRep*> pre_ops, std::vector<OpCRep*> post_ops) {
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
  
  TermCRep::TermCRep(CReps::PolynomialCRep* coeff, double magnitude, double logmagnitude,
			 std::vector<OpCRep*> pre_ops,
			 std::vector<OpCRep*> post_ops) {
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
