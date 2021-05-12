#include <complex>
#include <vector>
#include <unordered_map>
#include <cmath>

typedef std::complex<double> dcomplex;
typedef long long INT;


namespace CReps {
  //Forward declarations (as necessary)
  class OpCRep;

  //Helper functions
  void expm_multiply_simple_core(double* Adata, INT* Aindptr,
				 INT* Aindices, double* B,
				 INT N, double mu, INT m_star,
				 INT s, double tol, double eta,
				 double* F, double* scratch);
  void expm_multiply_simple_core_rep(OpCRep* A_rep, double* B,
				     INT N, double mu, INT m_star,
				     INT s, double tol, double eta,
				     double* F, double* scratch);

  // DENSE MATRIX () propagation

  // STATEs
  class StateCRep {
    public:
    double* _dataptr;
    INT _dim;
    bool _ownmem;
    StateCRep(INT dim);    
    StateCRep(double* data, INT dim, bool copy);
    ~StateCRep();
    void print(const char* label);
    void copy_from(StateCRep* st);
  };

  // EFFECTSs
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


  class EffectCRep_Errgen :public EffectCRep {
    public:
    OpCRep* _errgen_ptr;
    EffectCRep* _effect_ptr;
    INT _errgen_id;

    EffectCRep_Errgen(OpCRep* errgen_oprep, EffectCRep* effect_rep, INT errgen_id, INT dim);
    virtual ~EffectCRep_Errgen();
    virtual double probability(StateCRep* state);
    virtual double probability_using_cache(StateCRep* state, StateCRep* errgen_on_state, INT& errgen_id);
  };


  // GATEs
  class OpCRep {
    public:
    INT _dim;

    OpCRep(INT dim);
    virtual ~OpCRep();
    virtual StateCRep* acton(StateCRep* state, StateCRep* out_state) = 0;
    virtual StateCRep* adjoint_acton(StateCRep* state, StateCRep* out_state) = 0;
  };

  class OpCRep_Dense :public OpCRep {
    public:
    double* _dataptr;
    OpCRep_Dense(double* data, INT dim);
    virtual ~OpCRep_Dense();
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


  class OpCRep_ExpErrorgen :public OpCRep{
    public:
    OpCRep* _errgen_rep;
    double _mu;
    double _eta;
    INT _m_star;
    INT _s;

    OpCRep_ExpErrorgen(OpCRep* errgen_rep,
			double mu, double eta, INT m_star, INT s, INT dim);
    virtual ~OpCRep_ExpErrorgen();
    virtual StateCRep* acton(StateCRep* state, StateCRep* out_state);
    virtual StateCRep* adjoint_acton(StateCRep* state, StateCRep* out_state);
  };

  class OpCRep_Sparse :public OpCRep{
    public:
    double* _A_data;
    INT* _A_indices;
    INT* _A_indptr;
    INT _A_nnz;

    OpCRep_Sparse(double* A_data, INT* A_indices, INT* A_indptr,
		      INT nnz, INT dim);
    virtual ~OpCRep_Sparse();
    virtual StateCRep* acton(StateCRep* state, StateCRep* out_state);
    virtual StateCRep* adjoint_acton(StateCRep* state, StateCRep* out_state);
  };
}
