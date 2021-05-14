#include <complex>
#include <vector>
#include <unordered_map>
#include <cmath>

typedef std::complex<double> dcomplex;
typedef long long INT;


namespace CReps {

  //Polynomial key class - essentially a huge INT for holding many variable indices compactly
  class PolynomialVarsIndex {
    public:
    std::vector<INT> _parts;
    PolynomialVarsIndex() {}
    PolynomialVarsIndex(INT size) :_parts(size) {}

    bool operator==(PolynomialVarsIndex i) const {
      std::vector<INT>::const_iterator it, it2;
      if( i._parts.size() != this->_parts.size() ) return false;
      for(it=i._parts.begin(), it2=this->_parts.begin();
	  it != i._parts.end() && it2 != this->_parts.end();
	  ++it, ++it2) { //zip
	if(*it != *it2) return false;
      }
      return true;
    }

    bool operator<(PolynomialVarsIndex i) const {
      std::vector<INT>::const_iterator it, it2;
      if( i._parts.size() != this->_parts.size() )
	return this->_parts.size() < i._parts.size();
				     
      for(it=i._parts.begin(), it2=this->_parts.begin();
	  it != i._parts.end() && it2 != this->_parts.end();
	  ++it, ++it2) { //zip
	if(*it != *it2) return (*it2) < (*it);
      }
      return false; //equal, so not "<"
    }
  };
}

namespace std {

  //struct PolyVarsIndexHasher
  template <>
  struct hash<CReps::PolynomialVarsIndex>
  {
    std::size_t operator()(const CReps::PolynomialVarsIndex& k) const
    {
      using std::size_t;
      using std::hash;
      using std::string;

      std::size_t ret = 0;
      std::vector<INT>::const_iterator it;
      
      if(k._parts.size() == 0) return 0;
      for(it=k._parts.begin(); it != k._parts.end(); ++it) {
	ret = ret ^ hash<INT>()(*it); //TODO: rotate/shift bits to make a better hash here
      }
      return ret;
    }
  };
}


namespace CReps {

  //Polynomial class
  class PolynomialCRep {
    public:
    std::unordered_map<PolynomialVarsIndex, dcomplex> _coeffs;
    INT _max_num_vars;
    INT _vindices_per_int;
    PolynomialCRep();
    PolynomialCRep(std::unordered_map<PolynomialVarsIndex, dcomplex> coeffs, INT max_num_vars, INT vindices_per_int);
    PolynomialCRep(const PolynomialCRep& other);
    ~PolynomialCRep();
    PolynomialCRep abs();
    PolynomialCRep mult(const PolynomialCRep& other);
    PolynomialCRep abs_mult(const PolynomialCRep& other);
    void add_inplace(const PolynomialCRep& other);
    void add_abs_inplace(const PolynomialCRep& other);
    void add_scalar_to_all_coeffs_inplace(dcomplex offset);
    void scale(dcomplex scale);
    PolynomialVarsIndex vinds_to_int(std::vector<INT> vinds);
    std::vector<INT> int_to_vinds(PolynomialVarsIndex indx);
    private:
    PolynomialVarsIndex mult_vinds_ints(PolynomialVarsIndex i1, PolynomialVarsIndex i2); // multiply vinds corresponding to i1 & i2 and return resulting integer
  };
}
