#include <complex>
#include <vector>
#include <unordered_map>
#include <cmath>

typedef std::complex<double> dcomplex;
typedef long long INT;


namespace CReps_densitymx {

  class StateCRep {
    public:
    double* _dataptr;
    INT _dim;
    bool _ownmem;
    StateCRep(INT dim);    
    StateCRep(double* data, INT dim, bool copy=false);
    ~StateCRep();
    void print(const char* label);
    void copy_from(StateCRep* st);
  };

}
