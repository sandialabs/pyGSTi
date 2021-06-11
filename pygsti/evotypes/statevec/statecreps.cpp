#define NULL 0

#include <iostream>
#include <complex>
#include <assert.h>
#include <algorithm>    // std::find
#include "statecreps.h"
//#include <pthread.h>

//using namespace std::complex_literals;

//#define DEBUG(x) x
#define DEBUG(x) 

namespace CReps {

  /****************************************************************************\
  |* StateCRep                                                              *|
  \****************************************************************************/
  StateCRep::StateCRep(INT dim) {
    _dataptr = new dcomplex[dim];
    for(INT i=0; i<dim; i++) _dataptr[i] = 0;
    _dim = dim;
    _ownmem = true;
  }
  
  StateCRep::StateCRep(dcomplex* data, INT dim, bool copy) {
    //DEGUG std::cout << "StateCRep initialized w/dim = " << dim << std::endl;
    if(copy) {
      _dataptr = new dcomplex[dim];
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
}
