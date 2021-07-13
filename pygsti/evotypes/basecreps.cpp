#define NULL 0

#include <algorithm>
#include <iostream>
#include <complex>
#include <assert.h>
#include "basecreps.h"

namespace CReps {

  /**************************************************************************** \
  |* PolynomialCRep                                                                 *|
  \****************************************************************************/
  
  PolynomialCRep::PolynomialCRep() {
    _coeffs = std::unordered_map<PolynomialVarsIndex, dcomplex>();
    _max_num_vars = 0;
    _vindices_per_int = 0;
  }
  
  PolynomialCRep::PolynomialCRep(std::unordered_map<PolynomialVarsIndex, dcomplex> coeffs, INT max_num_vars, INT vindices_per_int) {
    _coeffs = coeffs;
    _max_num_vars = max_num_vars;
    _vindices_per_int = vindices_per_int;
  }

  PolynomialCRep::PolynomialCRep(const PolynomialCRep& other) {
    _coeffs = other._coeffs;
    _max_num_vars = other._max_num_vars;
  }

  PolynomialCRep::~PolynomialCRep() { }

  PolynomialCRep PolynomialCRep::mult(const PolynomialCRep& other) {
    std::unordered_map<PolynomialVarsIndex, dcomplex>::iterator it1, itk;
    std::unordered_map<PolynomialVarsIndex, dcomplex>::const_iterator it2;
    std::unordered_map<PolynomialVarsIndex, dcomplex> result;
    dcomplex val;
    PolynomialVarsIndex k;

    for(it1 = _coeffs.begin(); it1 != _coeffs.end(); ++it1) {
      for(it2 = other._coeffs.begin(); it2 != other._coeffs.end(); ++it2) {
	k = mult_vinds_ints(it1->first, it2->first); //key to add
	itk = result.find(k);
	val = it1->second * it2->second;
	if(itk != result.end())
	  itk->second = itk->second + val;
	else result[k] = val;
      }
    }
    PolynomialCRep ret(result, _max_num_vars, _vindices_per_int);
    return ret; // need a copy constructor?
  }

  PolynomialCRep PolynomialCRep::abs_mult(const PolynomialCRep& other) {
    std::unordered_map<PolynomialVarsIndex, dcomplex>::iterator it1, itk;
    std::unordered_map<PolynomialVarsIndex, dcomplex>::const_iterator it2;
    std::unordered_map<PolynomialVarsIndex, dcomplex> result;
    dcomplex val;
    PolynomialVarsIndex k;

    for(it1 = _coeffs.begin(); it1 != _coeffs.end(); ++it1) {
      for(it2 = other._coeffs.begin(); it2 != other._coeffs.end(); ++it2) {
	k = mult_vinds_ints(it1->first, it2->first); //key to add
	itk = result.find(k);
	val = std::abs(it1->second * it2->second);
	if(itk != result.end())
	  itk->second = itk->second + val;
	else result[k] = val;
      }
    }
    PolynomialCRep ret(result, _max_num_vars, _vindices_per_int);
    return ret; // need a copy constructor?
  }


  void PolynomialCRep::add_inplace(const PolynomialCRep& other) {
    std::unordered_map<PolynomialVarsIndex, dcomplex>::const_iterator it2;
      std::unordered_map<PolynomialVarsIndex, dcomplex>::iterator itk;
    dcomplex val, newval;
    PolynomialVarsIndex k;

    for(it2 = other._coeffs.begin(); it2 != other._coeffs.end(); ++it2) {
      k = it2->first; // key
      val = it2->second; // value
      itk = _coeffs.find(k);
      if(itk != _coeffs.end()) {
	newval = itk->second + val;
	if(std::abs(newval) > 1e-12) {
	  itk->second = newval; // note: += doens't work here (complex Cython?)
	} else {
	  _coeffs.erase(itk);
	}
      }
      else if(std::abs(val) > 1e-12) {
	_coeffs[k] = val;
      }
    }
  }

  void PolynomialCRep::add_abs_inplace(const PolynomialCRep& other) {
    std::unordered_map<PolynomialVarsIndex, dcomplex>::const_iterator it2;
      std::unordered_map<PolynomialVarsIndex, dcomplex>::iterator itk;
    double val;
    PolynomialVarsIndex k;

    for(it2 = other._coeffs.begin(); it2 != other._coeffs.end(); ++it2) {
      k = it2->first; // key
      val = std::abs(it2->second); // value
      if(val > 1e-12) {
          itk = _coeffs.find(k);
          if(itk != _coeffs.end()) {
              itk->second = itk->second + (dcomplex)val; // note: += doens't work here (complex Cython?)
          }
          else {
              _coeffs[k] = (dcomplex)val;
          }
      }
    }
  }

  PolynomialCRep PolynomialCRep::abs() {
    std::unordered_map<PolynomialVarsIndex, dcomplex> result;
    std::unordered_map<PolynomialVarsIndex, dcomplex>::iterator it;
    for(it = _coeffs.begin(); it != _coeffs.end(); ++it) {
        result[it->first] = std::abs(it->second);
    }
    
    PolynomialCRep ret(result, _max_num_vars, _vindices_per_int);
    return ret; // need a copy constructor?
  }

  void PolynomialCRep::scale(dcomplex scale) {
    std::unordered_map<PolynomialVarsIndex, dcomplex>::iterator it;
    for(it = _coeffs.begin(); it != _coeffs.end(); ++it) {
      it->second = it->second * scale; // note: *= doesn't work here (complex Cython?)
    }
  }

  void PolynomialCRep::add_scalar_to_all_coeffs_inplace(dcomplex offset) {
    std::unordered_map<PolynomialVarsIndex, dcomplex>::iterator it;
    for(it = _coeffs.begin(); it != _coeffs.end(); ++it) {
      it->second = it->second + offset; // note: += doesn't work here (complex Cython?)
    }
  }


  PolynomialVarsIndex PolynomialCRep::vinds_to_int(std::vector<INT> vinds) {
    INT ret, end, m;
    INT sz = ceil(1.0 * vinds.size() / _vindices_per_int);
    PolynomialVarsIndex ret_tup(sz);
    for(INT k=0; k<sz-1; k++) {
      ret = 0; m = 1;
      for(std::size_t i=_vindices_per_int*k; i<_vindices_per_int*(k+1); i++) { // last tuple index is most significant
	ret += (vinds[i]+1)*m;
	m *= _max_num_vars+1;
      }
      ret_tup._parts[k] = ret;
    }
    if(sz > 0) { //final iteration has different uppper limit
      ret = 0; m = 1;
      for(std::size_t i=_vindices_per_int*(sz-1); i<vinds.size(); i++) { // last tuple index is most significant
	ret += (vinds[i]+1)*m;
	m *= _max_num_vars+1;
      }
      ret_tup._parts[sz-1] = ret;
    }
    return ret_tup;
  }
  
  std::vector<INT> PolynomialCRep::int_to_vinds(PolynomialVarsIndex indx_tup) {
    std::vector<INT> ret;
    INT nxt, i, indx;
    for(std::size_t i=0; i < indx_tup._parts.size(); i++) {
      indx = indx_tup._parts[i];
      while(indx != 0) {
        nxt = indx / (_max_num_vars+1);
	i = indx - nxt*(_max_num_vars+1);
	ret.push_back(i-1);
	indx = nxt;
	//assert(indx >= 0);
      }
    }
    std::sort(ret.begin(),ret.end());
    return ret;
  }
  
  PolynomialVarsIndex PolynomialCRep::mult_vinds_ints(PolynomialVarsIndex i1, PolynomialVarsIndex i2) {
    // multiply vinds corresponding to i1 & i2 and return resulting integer
    std::vector<INT> vinds1 = int_to_vinds(i1);
    std::vector<INT> vinds2 = int_to_vinds(i2);
    vinds1.insert( vinds1.end(), vinds2.begin(), vinds2.end() );
    std::sort(vinds1.begin(),vinds1.end());
    return vinds_to_int(vinds1);
  }
}
