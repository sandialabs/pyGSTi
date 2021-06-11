#include <complex>
#include <vector>
#include <unordered_map>
#include <cmath>

typedef std::complex<double> dcomplex;
typedef long long INT;

namespace CReps {
    
  class EffectCRep {
    public:
    INT _dim;
    EffectCRep(INT dim);
    virtual ~EffectCRep();
    virtual double probability(StateCRep* state) = 0;
    virtual double probability_using_cache(StateCRep* state, StateCRep* precomp_state, INT& precomp_id) {
      return this->probability(state);
    }
  };


  class EffectCRep_Dense :public EffectCRep {
    public:
    double* _dataptr;
    EffectCRep_Dense(double* data, INT dim);
    virtual ~EffectCRep_Dense();
    virtual double probability(StateCRep* state);
  };

  class EffectCRep_TensorProd :public EffectCRep {
    public:
    double* _kron_array;
    INT _max_factor_dim;
    INT* _factordims;
    INT _nfactors;

    EffectCRep_TensorProd(double* kron_array, INT* factordims, INT nfactors, INT max_factor_dim, INT dim);
    virtual ~EffectCRep_TensorProd();
    virtual double probability(StateCRep* state);    
  };

  class EffectCRep_Computational :public EffectCRep {
    public:
    INT _nfactors;
    INT _zvals_int;
    double _abs_elval;

    EffectCRep_Computational(INT nfactors, INT zvals_int, double abs_elval, INT dim);
    virtual ~EffectCRep_Computational();
    virtual double probability(StateCRep* state);
    INT parity(INT x);
  };


  class EffectCRep_Composed :public EffectCRep {
    public:
    OpCRep* _errgen_ptr;
    EffectCRep* _effect_ptr;
    INT _errgen_id;

    EffectCRep_Composed(OpCRep* errgen_oprep, EffectCRep* effect_rep, INT errgen_id, INT dim);
    virtual ~EffectCRep_Composed();
    virtual double probability(StateCRep* state);
    virtual double probability_using_cache(StateCRep* state, StateCRep* errgen_on_state, INT& errgen_id);
  };
}
