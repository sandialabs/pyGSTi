#define NULL 0

#include <iostream>
#include "fastreps.h"


namespace CReps {

  // DMStateCRep ------------------------------
  DMStateCRep::DMStateCRep() {
    std::cout << "Creating new uninit!" << std::endl;
    _dataptr = NULL;
    _dim = 0;
    _ownmem = false;
  }

  DMStateCRep::DMStateCRep(int dim) {
    _dataptr = new double[dim];
    for(int i=0; i<dim; i++) _dataptr[i] = 0;
    _dim = dim;
    _ownmem = true;
  }

  
  DMStateCRep::DMStateCRep(double* data, int dim, bool copy=false) {
    if(copy) {
      _dataptr = new double[dim];
      std::cout << "Creating new initialized! " << _dataptr << std::endl;
      for(int i=0; i<dim; i++) _dataptr[i] = data[i];
    } else {
      _dataptr = data;
    }
    _dim = dim;
    _ownmem = copy;
  }

  DMStateCRep::~DMStateCRep() {
    std::cout << "DELETING " << _dataptr << std::endl;
    if(_ownmem && _dataptr != NULL)
      delete [] _dataptr;
    std::cout << "Done deconstr." << std::endl;
  }


  // DMEffectCRep ------------------------------
  DMEffectCRep::DMEffectCRep() {
    _dataptr = NULL;
    _dim = 0;
  }

  DMEffectCRep::DMEffectCRep(double* data, int dim) {
    _dataptr = data;
    _dim = dim;
  }

  DMEffectCRep::~DMEffectCRep() { }

  double DMEffectCRep::amplitude(DMStateCRep* state) {
    double ret = 0.0;
    for(int i=0; i< _dim; i++) {
      ret += _dataptr[i] * state->_dataptr[i];
    }
    return ret;
  }


  // DMGateCRep ------------------------------
  DMGateCRep::DMGateCRep() {
    _dataptr = NULL;
    _dim = 0;
  }

  DMGateCRep::DMGateCRep(double* data, int dim) {
    _dataptr = data;
    _dim = dim;
  }

  DMGateCRep::~DMGateCRep() { }
  
  DMStateCRep* DMGateCRep::acton(DMStateCRep* state,
				 DMStateCRep* outstate) {
    //if(outstate == NULL)
    //  outstate = new DMStateCRep(new double[_dim], _dim);

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
  
  
}
