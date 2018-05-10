#define NULL 0

#include <iostream>
#include "fastreps.h"


namespace CReps {

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

  double DMEffectCRep_Dense::amplitude(DMStateCRep* state) {
    double ret = 0.0;
    for(int i=0; i< _dim; i++) {
      ret += _dataptr[i] * state->_dataptr[i];
    }
    return ret;
  }

  /****************************************************************************\
  |* DMEffectCRep_TensorProd                                                  *|
  \****************************************************************************/

  DMEffectCRep_TensorProd::DMEffectCRep_TensorProd() 
    :DMEffectCRep(dim)
  {
    _fast_kron_array = ;
    _fast_kron_factordims = ?;
  }

  DMEffectCRep_TensorProd::~DMEffectCRep_TensorProd() { }
    
  double DMEffectCRep_TensorProd::amplitude(DMStateCRep* state) {
    //future: add scratch buffer as argument? or compute in place somehow?
    double ret = 0.0;
    double* scratch = new double[dim];
    //TODO _fastcalc.fast_kron(scratch, _fast_kron_array, _fast_kron_factordims);

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
    _dim = 0;
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
    int k;
    for(int i=0; i< _dim; i++) {
      outstate->_dataptr[i] = 0.0;
      k = i*_dim; // "row" offset into _dataptr, so dataptr[k+j] ~= dataptr[i,j]
      for(int j=0; j< _dim; j++) {
	outstate->_dataptr[i] += _dataptr[k+j] * state->_dataptr[j];
      }
    }
    return outstate;
  }

  /****************************************************************************\
  |* DMGateCRep_Embedded                                                      *|
  \****************************************************************************/

  DMGateCRep_Embedded::DMGateCRep_Embedded(dim)
    :DMGateCRep(dim)
  {    
  }
  DMGateCRep_Embedded::~DMGateCRep_Embedded() { }
  
  DMStateCRep* DMGateCRep_Embedded::acton(DMStateCRep* state, DMStateCRep* out_state) {
    //_fastcalc.embedded_fast_acton_sparse(self.embedded_gate.acton,
    //                                         output_state, state,
    //                                         self.noop_incrementers,
    //                                         self.numBasisEls_noop_blankaction,
    //                                         self.baseinds)
                                 
    //act on other blocks trivially:
    //if self.nBasisBlocks > 1:
    //  self._acton_other_blocks_trivially(output_state,state)
  }
      


  /****************************************************************************\
  |* DMGateCRep_Composed                                                      *|
  \****************************************************************************/
  DMGateCRep_Composed::DMGateCRep_Composed(DMGateCRep* factor_gate_creps, int nfactors, int dim)
    :DMGateCRep(dim)
  {
    _factor_gate_creps = factor_gate_creps;
    _nfactors = nfactors;
    //assert(_nfactors > 1) // we assume we always have at least one factor
  }
  DMGateCRep_Composed::~DMGateCRep_Composed() { }
  
  DMStateCRep* DMGateCRep_Composed::acton(DMStateCRep* state, DMStateCRep* out_state) {

    DMStateCRep* tmp1 = out_state; //already alloc'd
    DMStateCRep* t; // for swapping
    _factor_gate_creps[0].acton(state, tmp1);
    if(_nfactors > 1) {
      DMStateCRep* temp = new DMStateCRep(_dim);
      DMStateCRep* tmp2;
      for(int i=1; i<_nfactors; i++) {
	_factor_gate_creps[i].acton(tmp1,tmp2);
	t = tmp1; tmp1 = tmp2; tmp2 = t;
      }
      //tmp1 holds the output state now; if tmp1 == output_state
      // we're in luck, otherwise we need to copy it into output_state.
      if(tmp1 != output_state) {
	for(int j=0; j<_dim; j++) output_state[j] = tmp1[j];
      }
      delete temp; // cleanup the temporary state we made
    }
  }

  /****************************************************************************\
  |* DMGateCRep_Lindblad                                                      *|
  \****************************************************************************/
  DMGateCRep_Lindblad::DMGateCRep_Lindblad(double* data, int dim)
  {
  }
      
  DMGateCRep_Lindblad::~DMGateCRep_Linblad() { }
  DMStateCRep* DMGateCRep_Lindblad::acton(DMStateCRep* state, DMStateCRep* out_state)
  {
    if self.unitary_postfactor is not None:
            state = self.unitary_postfactor.dot(state) #works for sparse or dense matrices
            
        if self.sparse:
            //state = _spsl.expm_multiply( self.err_gen, state) #SLOW
            state = _mt.expm_multiply_fast(self.err_gen_prep, state) # (N,) -> (N,) shape mapping
        else:
            state = _np.dot(self.exp_err_gen, state)
  }


  // SIMILAR for STATE VECS (SV) propagation...
  
  // <<TODO>>


  // STABILIZER propagation

  /****************************************************************************\
  |* SBStateCRep                                                              *|
  \****************************************************************************/
  class SBStateCRep {  // a stabilizer frame (perhaps optionally w/out phase info, so just a state?)
    public:
    int _n;
    double* _smatrix, _pvector;
    SBStateCRep(int dim);    
    SBStateCRep(double* data, int dim, bool copy);
    ~SBStateCRep();
  };

  /****************************************************************************\
  |* SBEffectCRep                                                             *|
  \****************************************************************************/
  class SBEffectCRep { // a stabilizer measurement - just implement z-basis meas here for now(?)
    public:
    int _n;
    int* _zvals;
    SBEffectCRep(int dim);
    ~SBEffectCRep();
    virtual double amplitude(SBStateCRep* state);
  };


  /****************************************************************************\
  |* SBGateCRep                                                               *|
  \****************************************************************************/
  class SBGateCRep {
    public:
    SBGateCRep();
    SBGateCRep(double* data, int dim);
    ~SBGateCRep();
    virtual SBStateCRep* acton(SBStateCRep* state, SBStateCRep* out_state);
  };

  /****************************************************************************\
  |* SBGateCRep_Embedded                                                      *|
  \****************************************************************************/
  class SBGateCRep_Embedded {
    public:
    int _n;
    int _targetQs;
    SBGateCRep* _embedded_gate_crep;
    SBGateCRep_Embedded();
    SBGateCRep_Embedded(double* data, int dim);
    ~SBGateCRep_Embedded();
    virtual SBStateCRep* acton(SBStateCRep* state, SBStateCRep* out_state);
  };

  /****************************************************************************\
  |* SBGateCRep_Composed                                                      *|
  \****************************************************************************/
  class SBGateCRep_Composed {
    SBGateCRep* _factor_gate_creps;
    public:
    SBGateCRep_Composed();
    SBGateCRep_Composed(double* data, int dim);
    ~SBGateCRep_Composed();
    virtual SBStateCRep* acton(SBStateCRep* state, SBStateCRep* out_state);
  };

  /****************************************************************************\
  |* SBGateCRep_Clifford                                                      *|
  \****************************************************************************/
  class SBGateCRep_Clifford {
    public:
    int _n;
    double* _smatrix, _svector; //symplectic rep
    SBGateCRep_Clifford();
    SBGateCRep_Clifford(double* data, int dim);
    ~SBGateCRep_Clifford();
    virtual SBStateCRep* acton(SBStateCRep* state, SBStateCRep* out_state);
  };



  //--------------------------------------------------------------------------------------------

  

  // DMGateCRep ------------------------------
  
  
}
