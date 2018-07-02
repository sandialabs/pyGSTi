#include <complex>
#include <vector>
#include <unordered_map>
#include <cmath>

typedef std::complex<double> dcomplex;
typedef long long INT;

namespace CReps {

  //Helper functions
  void expm_multiply_simple_core(double* Adata, INT* Aindptr,
				 INT* Aindices, double* B,
				 INT N, double mu, INT m_star,
				 INT s, double tol, double eta,
				 double* F, double* scratch);

  // DENSE MATRIX (DM) propagation

  // STATEs
  class DMStateCRep {
    public:
    double* _dataptr;
    INT _dim;
    bool _ownmem;
    DMStateCRep(INT dim);    
    DMStateCRep(double* data, INT dim, bool copy);
    ~DMStateCRep();
    void print(const char* label);
    void copy_from(DMStateCRep* st);
  };

  // EFFECTSs
  class DMEffectCRep {
    public:
    INT _dim;
    DMEffectCRep(INT dim);
    virtual ~DMEffectCRep();
    virtual double probability(DMStateCRep* state) = 0;
  };


  class DMEffectCRep_Dense :public DMEffectCRep {
    public:
    double* _dataptr;
    DMEffectCRep_Dense(double* data, INT dim);
    virtual ~DMEffectCRep_Dense();
    virtual double probability(DMStateCRep* state);
  };

  class DMEffectCRep_TensorProd :public DMEffectCRep {
    public:
    double* _kron_array;
    INT _max_factor_dim;
    INT* _factordims;
    INT _nfactors;

    DMEffectCRep_TensorProd(double* kron_array, INT* factordims, INT nfactors, INT max_factor_dim, INT dim);
    virtual ~DMEffectCRep_TensorProd();
    virtual double probability(DMStateCRep* state);    
  };

  class DMEffectCRep_Computational :public DMEffectCRep {
    public:
    INT _nfactors;
    INT _zvals_int;
    double _abs_elval;

    DMEffectCRep_Computational(INT nfactors, INT zvals_int, double abs_elval, INT dim);
    virtual ~DMEffectCRep_Computational();
    virtual double probability(DMStateCRep* state);
    INT parity(INT x);
  };


  // GATEs
  class DMGateCRep {
    public:
    INT _dim;

    DMGateCRep(INT dim);
    virtual ~DMGateCRep();
    virtual DMStateCRep* acton(DMStateCRep* state, DMStateCRep* out_state) = 0;
    virtual DMStateCRep* adjoint_acton(DMStateCRep* state, DMStateCRep* out_state) = 0;
  };

  class DMGateCRep_Dense :public DMGateCRep {
    public:
    double* _dataptr;
    DMGateCRep_Dense(double* data, INT dim);
    virtual ~DMGateCRep_Dense();
    virtual DMStateCRep* acton(DMStateCRep* state, DMStateCRep* out_state);
    virtual DMStateCRep* adjoint_acton(DMStateCRep* state, DMStateCRep* out_state);
  };

  class DMGateCRep_Embedded :public DMGateCRep{
    public:
    DMGateCRep* _embedded_gate_crep;
    INT* _noop_incrementers;
    INT* _numBasisEls_noop_blankaction;
    INT* _baseinds;
    INT* _blocksizes; // basis blockdim**2 elements
    INT _nComponents, _embeddedDim, _iActiveBlock, _nBlocks;
    
    
    DMGateCRep_Embedded(DMGateCRep* embedded_gate_crep, INT* noop_incrementers,
			INT* numBasisEls_noop_blankaction, INT* baseinds, INT* blocksizes,
			INT embedded_dim, INT nComponentsInActiveBlock, INT iActiveBlock,
			INT nBlocks, INT dim);
    virtual ~DMGateCRep_Embedded();
    virtual DMStateCRep* acton(DMStateCRep* state, DMStateCRep* out_state);
    virtual DMStateCRep* adjoint_acton(DMStateCRep* state, DMStateCRep* out_state);
  };

  class DMGateCRep_Composed :public DMGateCRep{
    public:
    std::vector<DMGateCRep*> _factor_gate_creps;
    DMGateCRep_Composed(std::vector<DMGateCRep*> factor_gate_creps);
    virtual ~DMGateCRep_Composed();
    virtual DMStateCRep* acton(DMStateCRep* state, DMStateCRep* out_state);
    virtual DMStateCRep* adjoint_acton(DMStateCRep* state, DMStateCRep* out_state);
  };

  class DMGateCRep_Lindblad :public DMGateCRep{
    public:
    double* _U_data; //unitary postfactor 
    INT* _U_indices; // as a CSR sparse
    INT* _U_indptr;  // matrix
    INT _U_nnz;
    double* _A_data;
    INT* _A_indices;
    INT* _A_indptr;
    INT _A_nnz;
    double _mu, _eta; // tol?
    INT _m_star, _s;

    DMGateCRep_Lindblad(double* A_data, INT* A_indices, INT* A_indptr, INT nnz,
			double mu, double eta, INT m_star, INT s, INT dim,
		        double* unitarypost_data, INT* unitarypost_indices,
			INT* unitarypost_indptr, INT unitarypost_nnz);
    virtual ~DMGateCRep_Lindblad();
    virtual DMStateCRep* acton(DMStateCRep* state, DMStateCRep* out_state);
    virtual DMStateCRep* adjoint_acton(DMStateCRep* state, DMStateCRep* out_state);
  };



  // STATE VECTOR (SV) propagation

  // STATEs
  class SVStateCRep {
    public:
    dcomplex* _dataptr;
    INT _dim;
    bool _ownmem;
    SVStateCRep(INT dim);    
    SVStateCRep(dcomplex* data, INT dim, bool copy);
    ~SVStateCRep();
    void print(const char* label);
    void copy_from(SVStateCRep* st);
  };

  // EFFECTSs
  class SVEffectCRep {
    public:
    INT _dim;
    SVEffectCRep(INT dim);
    virtual ~SVEffectCRep();
    virtual double probability(SVStateCRep* state) = 0;
    virtual dcomplex amplitude(SVStateCRep* state) = 0;
  };


  class SVEffectCRep_Dense :public SVEffectCRep {
    public:
    dcomplex* _dataptr;
    SVEffectCRep_Dense(dcomplex* data, INT dim);
    virtual ~SVEffectCRep_Dense();
    virtual double probability(SVStateCRep* state);
    virtual dcomplex amplitude(SVStateCRep* state);;
  };

  class SVEffectCRep_TensorProd :public SVEffectCRep {
    public:
    dcomplex* _kron_array;
    INT _max_factor_dim;
    INT* _factordims;
    INT _nfactors;

    SVEffectCRep_TensorProd(dcomplex* kron_array, INT* factordims, INT nfactors, INT max_factor_dim, INT dim);
    virtual ~SVEffectCRep_TensorProd();
    virtual double probability(SVStateCRep* state);
    virtual dcomplex amplitude(SVStateCRep* state);
  };


  // GATEs
  class SVGateCRep {
    public:
    INT _dim;

    SVGateCRep(INT dim);
    virtual ~SVGateCRep();
    virtual SVStateCRep* acton(SVStateCRep* state, SVStateCRep* out_state) = 0;
    virtual SVStateCRep* adjoint_acton(SVStateCRep* state, SVStateCRep* out_state) = 0;
  };

  class SVGateCRep_Dense :public SVGateCRep {
    public:
    dcomplex* _dataptr;
    SVGateCRep_Dense(dcomplex* data, INT dim);
    virtual ~SVGateCRep_Dense();
    virtual SVStateCRep* acton(SVStateCRep* state, SVStateCRep* out_state);
    virtual SVStateCRep* adjoint_acton(SVStateCRep* state, SVStateCRep* out_state);
  };

  class SVGateCRep_Embedded :public SVGateCRep{
    public:
    SVGateCRep* _embedded_gate_crep;
    INT* _noop_incrementers;
    INT* _numBasisEls_noop_blankaction;
    INT* _baseinds;
    INT* _blocksizes; // basis blockdim**2 elements
    INT _nComponents, _embeddedDim, _iActiveBlock, _nBlocks;
    
    
    SVGateCRep_Embedded(SVGateCRep* embedded_gate_crep, INT* noop_incrementers,
			INT* numBasisEls_noop_blankaction, INT* baseinds, INT* blocksizes,
			INT embedded_dim, INT nComponentsInActiveBlock, INT iActiveBlock,
			INT nBlocks, INT dim);
    virtual ~SVGateCRep_Embedded();
    virtual SVStateCRep* acton(SVStateCRep* state, SVStateCRep* out_state);
    virtual SVStateCRep* adjoint_acton(SVStateCRep* state, SVStateCRep* out_state);
  };

  class SVGateCRep_Composed :public SVGateCRep{
    public:
    std::vector<SVGateCRep*> _factor_gate_creps;
    SVGateCRep_Composed(std::vector<SVGateCRep*> factor_gate_creps);
    virtual ~SVGateCRep_Composed();
    virtual SVStateCRep* acton(SVStateCRep* state, SVStateCRep* out_state);
    virtual SVStateCRep* adjoint_acton(SVStateCRep* state, SVStateCRep* out_state);
  };


  // STABILIZER propagation

    // STATEs
  class SBStateCRep {  // a stabilizer frame (perhaps optionally w/out phase info, so just a state?)
    public:
    INT _n, _2n, _namps;
    INT* _smatrix;
    INT* _pvectors;
    INT _zblock_start;
    dcomplex* _amps;
    std::vector<std::vector<INT> > _view_filters;
    bool _ownmem;
    
    SBStateCRep(INT* smatrix, INT* pvectors, dcomplex* amps, INT namps, INT n);
    SBStateCRep(INT namps, INT n);
    ~SBStateCRep();
    void push_view(std::vector<INT>& view);
    void pop_view();
    void clifford_update(INT* smatrix, INT* svector, dcomplex* Umx);
    dcomplex extract_amplitude(std::vector<INT>& zvals);
    double measurement_probability(std::vector<INT>& zvals); //, qubit_filter);
    void copy_from(SBStateCRep* other);

    private:
    INT udot1(INT i, INT j);
    void udot2(INT* out, INT* smatrix1, INT* smatrix2);
    void colsum(INT i, INT j);
    void colswap(INT i, INT j);
    void rref();
    dcomplex apply_xgen(INT igen, INT pgen, std::vector<INT>& zvals_to_acton,
			dcomplex ampl, std::vector<INT>& result);
    dcomplex get_target_ampl(std::vector<INT>& tgt, std::vector<INT>& anchor,
			     dcomplex anchor_amp, INT ip);
    dcomplex canonical_amplitude_of_target(INT ip, std::vector<INT>& target);
    void canonical_amplitudes_sample(INT ip, std::vector<INT> qs_to_sample,
				     std::vector<INT>& anchor, std::vector<dcomplex>& amp_samples);

    void canonical_amplitudes(INT ip, INT* target, INT qs_to_sample=1);
    void apply_clifford_to_frame(INT* s, INT* p, std::vector<INT> qubit_filter);
    void apply_clifford_to_frame(INT* s, INT* p);
  };

  // EFFECTSs
  class SBEffectCRep { // a stabilizer measurement - just implement z-basis meas here for now(?)
    public:
    INT _n;
    std::vector<INT> _zvals;
    SBEffectCRep(INT* zvals, INT n);
    ~SBEffectCRep();
    dcomplex amplitude(SBStateCRep* state); //make virtual if we turn this into
    double probability(SBStateCRep* state); // a base class & derive from it.
  };


  // GATEs
  class SBGateCRep {
    public:
    INT _n;
    SBGateCRep(INT dim);
    virtual ~SBGateCRep();
    virtual SBStateCRep* acton(SBStateCRep* state, SBStateCRep* out_state) = 0;
    virtual SBStateCRep* adjoint_acton(SBStateCRep* state, SBStateCRep* out_state) = 0;
  };

  class SBGateCRep_Embedded :public SBGateCRep {
    public:
    std::vector<INT> _qubits;
    SBGateCRep* _embedded_gate_crep;
    SBGateCRep_Embedded(SBGateCRep* embedded_gate_crep, INT n, INT* qubits, INT nqubits);
    virtual ~SBGateCRep_Embedded();
    virtual SBStateCRep* acton(SBStateCRep* state, SBStateCRep* out_state);
    virtual SBStateCRep* adjoint_acton(SBStateCRep* state, SBStateCRep* out_state);
  };

  class SBGateCRep_Composed :public SBGateCRep {
    std::vector<SBGateCRep*> _factor_gate_creps;
    public:
    SBGateCRep_Composed(std::vector<SBGateCRep*> factor_gate_creps);
    virtual ~SBGateCRep_Composed();
    virtual SBStateCRep* acton(SBStateCRep* state, SBStateCRep* out_state);
    virtual SBStateCRep* adjoint_acton(SBStateCRep* state, SBStateCRep* out_state);
  };

  class SBGateCRep_Clifford :public SBGateCRep{
    public:
    INT _n;
    INT *_smatrix, *_svector; //symplectic rep
    INT *_smatrix_inv, *_svector_inv; //of the inverse, for adjoint ops
    dcomplex *_unitary, *_unitary_adj;
    SBGateCRep_Clifford(INT* smatrix, INT* svector, dcomplex* unitary,
			INT* smatrix_inv, INT* svector_inv, dcomplex* unitary_adj, INT n);
    virtual ~SBGateCRep_Clifford();
    virtual SBStateCRep* acton(SBStateCRep* state, SBStateCRep* out_state);
    virtual SBStateCRep* adjoint_acton(SBStateCRep* state, SBStateCRep* out_state);
  };

  //Polynomial class
  class PolyCRep {
    public:
    std::unordered_map<INT, dcomplex> _coeffs;
    INT _max_order;
    INT _max_num_vars;
    PolyCRep();
    PolyCRep(std::unordered_map<INT, dcomplex> coeffs, INT max_order, INT max_num_vars);
    PolyCRep(const PolyCRep& other);
    ~PolyCRep();
    PolyCRep mult(const PolyCRep& other);
    void add_inplace(const PolyCRep& other);
    void scale(dcomplex scale);
    private:
    INT vinds_to_int(std::vector<INT> vinds);
    std::vector<INT> int_to_vinds(INT indx);
    INT mult_vinds_ints(INT i1, INT i2); // multiply vinds corresponding to i1 & i2 and return resulting integer
  };
  
  //TERMS
  class SVTermCRep {
    public:
    PolyCRep* _coeff;
    SVStateCRep* _pre_state;
    SVEffectCRep* _pre_effect;
    std::vector<SVGateCRep*> _pre_ops;
    SVStateCRep* _post_state;
    SVEffectCRep* _post_effect;
    std::vector<SVGateCRep*> _post_ops;
    SVTermCRep(PolyCRep* coeff, SVStateCRep* pre_state, SVStateCRep* post_state,
	       std::vector<SVGateCRep*> pre_ops, std::vector<SVGateCRep*> post_ops);
    SVTermCRep(PolyCRep* coeff, SVEffectCRep* pre_effect, SVEffectCRep* post_effect,
	       std::vector<SVGateCRep*> pre_ops, std::vector<SVGateCRep*> post_ops);
    SVTermCRep(PolyCRep* coeff, std::vector<SVGateCRep*> pre_ops, std::vector<SVGateCRep*> post_ops);
  };

  class SBTermCRep {
    public:
    PolyCRep* _coeff;
    SBStateCRep* _pre_state;
    SBEffectCRep* _pre_effect;
    std::vector<SBGateCRep*> _pre_ops;
    SBStateCRep* _post_state;
    SBEffectCRep* _post_effect;
    std::vector<SBGateCRep*> _post_ops;
    SBTermCRep(PolyCRep* coeff, SBStateCRep* pre_state, SBStateCRep* post_state,
	       std::vector<SBGateCRep*> pre_ops, std::vector<SBGateCRep*> post_ops);
    SBTermCRep(PolyCRep* coeff, SBEffectCRep* pre_effect, SBEffectCRep* post_effect,
	       std::vector<SBGateCRep*> pre_ops, std::vector<SBGateCRep*> post_ops);
    SBTermCRep(PolyCRep* coeff, std::vector<SBGateCRep*> pre_ops, std::vector<SBGateCRep*> post_ops);
  };

  

}
