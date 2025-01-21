#include <complex>
#include <vector>
#include <unordered_map>
#include <cmath>

typedef std::complex<double> dcomplex;
typedef long long INT;

namespace CReps_stabilizer {

  class EffectCRep {
    public:
    INT _n;
    EffectCRep(INT n);
    virtual ~EffectCRep();
    virtual double probability(StateCRep* state) = 0;
    virtual dcomplex amplitude(StateCRep* state) = 0;
  };

  class EffectCRep_Computational :public EffectCRep { // a stabilizer measurement - just implement z-basis meas here for now(?)
    public:
    std::vector<INT> _zvals;
    EffectCRep_Computational(INT* zvals, INT n);
    ~EffectCRep_Computational();
    dcomplex amplitude(StateCRep* state);
    double probability(StateCRep* state);
  };
    
  class EffectCRep_Composed :public EffectCRep {
    public:
    OpCRep* _errgen_ptr;
    EffectCRep* _effect_ptr;
    INT _errgen_id;

    EffectCRep_Composed(OpCRep* errgen_oprep, EffectCRep* effect_rep, INT errgen_id, INT n);
    virtual ~EffectCRep_Composed();
    virtual double probability(StateCRep* state);
    virtual dcomplex amplitude(StateCRep* state);
  };
}
