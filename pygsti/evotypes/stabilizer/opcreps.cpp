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
  OpCRep::OpCRep(INT n) {
    _n = n;
  }
  OpCRep::~OpCRep() { }

  /****************************************************************************\
  |* OpCRep_Embedded                                                          *|
  \****************************************************************************/
  OpCRep_Embedded::OpCRep_Embedded(OpCRep* embedded_gate_crep, INT n, INT* qubits, INT nqubits)
    :OpCRep(n),_qubits(nqubits)
  {
    _embedded_gate_crep = embedded_gate_crep;
    for(INT i=0; i<nqubits; i++)
      _qubits[i] = qubits[i];
  }
  OpCRep_Embedded::~OpCRep_Embedded() { }
  
  StateCRep* OpCRep_Embedded::acton(StateCRep* state, StateCRep* out_state) {
    DEBUG(std::cout << "Stabilizer Embedded acton called!" << std::endl);
    state->push_view(_qubits);
    _embedded_gate_crep->acton(state, out_state);
    state->pop_view();
    out_state->pop_view(); //should have same view as input state
    return out_state;
  }

  StateCRep* OpCRep_Embedded::adjoint_acton(StateCRep* state, StateCRep* out_state) {
    DEBUG(std::cout << "Stabilizer Embedded adjoint_acton called!" << std::endl);
    state->push_view(_qubits);
    _embedded_gate_crep->adjoint_acton(state, out_state);
    state->pop_view();
    out_state->pop_view(); //should have same view as input state
    return out_state;
  }


  /****************************************************************************\
  |* OpCRep_Composed                                                          *|
  \****************************************************************************/
  OpCRep_Composed::OpCRep_Composed(std::vector<OpCRep*> factor_gate_creps, INT n)
    :OpCRep(n),_factor_gate_creps(factor_gate_creps)
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
    DEBUG(std::cout << "Stabilizer Composed acton called!" << std::endl);
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
      StateCRep temp_state(tmp1->_namps,_n); tmp2 = &temp_state;

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

  StateCRep* OpCRep_Composed::adjoint_acton(StateCRep* state, StateCRep* out_state) {
    DEBUG(std::cout << "Stabilizer Composed adjoint_acton called!" << std::endl);
    std::size_t nfactors = _factor_gate_creps.size();
    StateCRep *tmp2, *tmp1 = out_state; //tmp1 already alloc'd
    StateCRep* t; // for swapping

    //Act with first gate: output in tmp1
    _factor_gate_creps[nfactors-1]->adjoint_acton(state, tmp1);

    if(nfactors > 1) {
      StateCRep temp_state(tmp1->_namps,_n); tmp2 = &temp_state;

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
  |* OpCRep_Sum                                                               *|
  \****************************************************************************/
  OpCRep_Sum::OpCRep_Sum(std::vector<OpCRep*> factor_creps, INT n)
    :OpCRep(n),_factor_creps(factor_creps)
  {
  }
  OpCRep_Sum::~OpCRep_Sum() { }

  StateCRep* OpCRep_Sum::acton(StateCRep* state, StateCRep* out_state) {
    assert(false); // need further stabilizer frame support to represent the sum of stabilizer states
    return NULL; //to avoid compiler warning
  }

  StateCRep* OpCRep_Sum::adjoint_acton(StateCRep* state, StateCRep* out_state) {
    assert(false); // need further stabilizer frame support to represent the sum of stabilizer states
    return NULL; //to avoid compiler warning
  }

  
  /****************************************************************************\
  |* OpCRep_Repeated                                                          *|
  \****************************************************************************/

  OpCRep_Repeated::OpCRep_Repeated(OpCRep* repeated_crep, INT num_repetitions, INT n)
    :OpCRep(n)
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
      StateCRep temp_state(tmp1->_namps,_n); tmp2 = &temp_state;

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
      StateCRep temp_state(tmp1->_namps,_n); tmp2 = &temp_state;

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
  |* OpCRep_Clifford                                                          *|
  \****************************************************************************/
  OpCRep_Clifford::OpCRep_Clifford(INT* smatrix, INT* svector, dcomplex* unitary,
					   INT* smatrix_inv, INT* svector_inv, dcomplex* unitary_adj, INT n)
    :OpCRep(n)
  {
    _smatrix = smatrix;
    _svector = svector;
    _smatrix_inv = smatrix_inv;
    _svector_inv = svector_inv;
    _unitary = unitary;
    _unitary_adj = unitary_adj;

    //DEBUG!!!
    //std::cout << "IN OpCRep_Clifford CONSTRUCTOR U = ";
    //for(int i=0; i<2*2; i++) std::cout << _unitary_adj[i] << " ";
    //std::cout << std::endl;

  }
  OpCRep_Clifford::~OpCRep_Clifford() { }
  
  StateCRep* OpCRep_Clifford::acton(StateCRep* state, StateCRep* out_state) {
    DEBUG(std::cout << "Stabilizer Clifford acton called!" << std::endl);
    out_state->copy_from(state);
    out_state->clifford_update(_smatrix, _svector, _unitary);
    return out_state;
  }

  StateCRep* OpCRep_Clifford::adjoint_acton(StateCRep* state, StateCRep* out_state) {
    DEBUG(std::cout << "Stabilizer Clifford adjoint_acton called!" << std::endl);

    //DEBUG!!!
    //std::cout << "AT OpCRep_Clifford::adjoint_acton U = ";
    //for(INT i=0; i<2*2; i++) std::cout << _unitary_adj[i] << " ";
    //std::cout << std::endl;
    
    out_state->copy_from(state);
    out_state->clifford_update(_smatrix_inv, _svector_inv, _unitary_adj);
    return out_state;
  }

}
