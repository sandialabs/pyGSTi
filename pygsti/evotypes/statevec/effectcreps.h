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
    virtual dcomplex amplitude(StateCRep* state) = 0;
  };


  class EffectCRep_Dense :public EffectCRep {
    public:
    dcomplex* _dataptr;
    EffectCRep_Dense(dcomplex* data, INT dim);
    virtual ~EffectCRep_Dense();
    virtual double probability(StateCRep* state);
    virtual dcomplex amplitude(StateCRep* state);;
  };

  class EffectCRep_TensorProd :public EffectCRep {
    public:
    dcomplex* _kron_array;
    INT _max_factor_dim;
    INT* _factordims;
    INT _nfactors;

    EffectCRep_TensorProd(dcomplex* kron_array, INT* factordims, INT nfactors, INT max_factor_dim, INT dim);
    virtual ~EffectCRep_TensorProd();
    virtual double probability(StateCRep* state);
    virtual dcomplex amplitude(StateCRep* state);
  };

  class EffectCRep_Computational :public EffectCRep {
    public:
    INT _nfactors;
    INT _zvals_int;
    INT _nonzero_index;

    EffectCRep_Computational(INT nfactors, INT zvals_int, INT dim);
    virtual ~EffectCRep_Computational();
    virtual double probability(StateCRep* state);
    virtual dcomplex amplitude(StateCRep* state);
  };

}
