#include <complex>
#include <vector>
#include <unordered_map>
#include <cmath>

typedef std::complex<double> dcomplex;
typedef long long INT;


namespace CReps_statevec {

  class OpCRep {
    public:
    INT _dim;

    OpCRep(INT dim);
    virtual ~OpCRep();
    virtual StateCRep* acton(StateCRep* state, StateCRep* out_state) = 0;
    virtual StateCRep* adjoint_acton(StateCRep* state, StateCRep* out_state) = 0;
  };

  class OpCRep_DenseUnitary :public OpCRep {
    public:
    dcomplex* _dataptr;
    OpCRep_DenseUnitary(dcomplex* data, INT dim);
    virtual ~OpCRep_DenseUnitary();
    virtual StateCRep* acton(StateCRep* state, StateCRep* out_state);
    virtual StateCRep* adjoint_acton(StateCRep* state, StateCRep* out_state);
  };

  class OpCRep_Embedded :public OpCRep{
    public:
    OpCRep* _embedded_gate_crep;
    INT* _noop_incrementers;
    INT* _numBasisEls_noop_blankaction;
    INT* _baseinds;
    INT* _blocksizes; // basis blockdim**2 elements
    INT _nComponents, _embeddedDim, _iActiveBlock, _nBlocks;
    
    
    OpCRep_Embedded(OpCRep* embedded_gate_crep, INT* noop_incrementers,
		      INT* numBasisEls_noop_blankaction, INT* baseinds, INT* blocksizes,
		      INT embedded_dim, INT nComponentsInActiveBlock, INT iActiveBlock,
		      INT nBlocks, INT dim);
    virtual ~OpCRep_Embedded();
    virtual StateCRep* acton(StateCRep* state, StateCRep* out_state);
    virtual StateCRep* adjoint_acton(StateCRep* state, StateCRep* out_state);
  };

  class OpCRep_Composed :public OpCRep{
    public:
    std::vector<OpCRep*> _factor_gate_creps;
    OpCRep_Composed(std::vector<OpCRep*> factor_gate_creps, INT dim);
    void reinit_factor_op_creps(std::vector<OpCRep*> new_factor_gate_creps);
    virtual ~OpCRep_Composed();
    virtual StateCRep* acton(StateCRep* state, StateCRep* out_state);
    virtual StateCRep* adjoint_acton(StateCRep* state, StateCRep* out_state);
  };

  class OpCRep_Sum :public OpCRep{
    public:
    std::vector<OpCRep*> _factor_creps;
    OpCRep_Sum(std::vector<OpCRep*> factor_creps, INT dim);
    virtual ~OpCRep_Sum();
    virtual StateCRep* acton(StateCRep* state, StateCRep* out_state);
    virtual StateCRep* adjoint_acton(StateCRep* state, StateCRep* out_state);
  };

  class OpCRep_Repeated :public OpCRep{
    public:
    OpCRep* _repeated_crep;
    INT _num_repetitions;
    
    OpCRep_Repeated(OpCRep* repeated_crep, INT num_repetitions, INT dim);
    virtual ~OpCRep_Repeated();
    virtual StateCRep* acton(StateCRep* state, StateCRep* out_state);
    virtual StateCRep* adjoint_acton(StateCRep* state, StateCRep* out_state);
  };

}
