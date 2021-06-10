#include <complex>
#include <vector>
#include <unordered_map>
#include <cmath>

typedef std::complex<double> dcomplex;
typedef long long INT;

namespace CReps {

  class EffectCRep { // a stabilizer measurement - just implement z-basis meas here for now(?)
    public:
    INT _n;
    std::vector<INT> _zvals;
    EffectCRep(INT* zvals, INT n);
    ~EffectCRep();
    dcomplex amplitude(StateCRep* state); //make virtual if we turn this into
    double probability(StateCRep* state); // a base class & derive from it.
  };

}
