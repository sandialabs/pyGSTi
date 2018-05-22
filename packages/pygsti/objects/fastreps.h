#include <complex>
#include <vector>
#include <unordered_map>
#include <cmath>

typedef std::complex<double> dcomplex;

namespace CReps {

  //Helper functions
  void expm_multiply_simple_core(double* Adata, int* Aindptr,
				 int* Aindices, double* B,
				 int N, double mu, int m_star,
				 int s, double tol, double eta,
				 double* F, double* scratch);

  // DENSE MATRIX (DM) propagation

  // STATEs
  class DMStateCRep {
    public:
    double* _dataptr;
    int _dim;
    bool _ownmem;
    DMStateCRep(int dim);    
    DMStateCRep(double* data, int dim, bool copy);
    ~DMStateCRep();
    void print(const char* label);
    void copy_from(DMStateCRep* st);
  };

  // EFFECTSs
  class DMEffectCRep {
    public:
    int _dim;
    DMEffectCRep(int dim);
    virtual ~DMEffectCRep();
    virtual double probability(DMStateCRep* state) = 0;
  };


  class DMEffectCRep_Dense :public DMEffectCRep {
    public:
    double* _dataptr;
    DMEffectCRep_Dense(double* data, int dim);
    virtual ~DMEffectCRep_Dense();
    virtual double probability(DMStateCRep* state);
  };

  class DMEffectCRep_TensorProd :public DMEffectCRep {
    public:
    double* _kron_array;
    int _max_factor_dim;
    int* _factordims;
    int _nfactors;

    DMEffectCRep_TensorProd(double* kron_array, int* factordims, int nfactors, int max_factor_dim, int dim);
    virtual ~DMEffectCRep_TensorProd();
    virtual double probability(DMStateCRep* state);    
  };


  // GATEs
  class DMGateCRep {
    public:
    int _dim;

    DMGateCRep(int dim);
    virtual ~DMGateCRep();
    virtual DMStateCRep* acton(DMStateCRep* state, DMStateCRep* out_state) = 0;
    virtual DMStateCRep* adjoint_acton(DMStateCRep* state, DMStateCRep* out_state) = 0;
  };

  class DMGateCRep_Dense :public DMGateCRep {
    public:
    double* _dataptr;
    DMGateCRep_Dense(double* data, int dim);
    virtual ~DMGateCRep_Dense();
    virtual DMStateCRep* acton(DMStateCRep* state, DMStateCRep* out_state);
    virtual DMStateCRep* adjoint_acton(DMStateCRep* state, DMStateCRep* out_state);
  };

  class DMGateCRep_Embedded :public DMGateCRep{
    public:
    DMGateCRep* _embedded_gate_crep;
    int* _noop_incrementers;
    int* _numBasisEls_noop_blankaction;
    int* _baseinds;
    int* _blocksizes; // basis blockdim**2 elements
    int _nComponents, _embeddedDim, _iActiveBlock, _nBlocks;
    
    
    DMGateCRep_Embedded(DMGateCRep* embedded_gate_crep, int* noop_incrementers,
			int* numBasisEls_noop_blankaction, int* baseinds, int* blocksizes,
			int embedded_dim, int nComponentsInActiveBlock, int iActiveBlock,
			int nBlocks, int dim);
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
    int* _U_indices; // as a CSR sparse
    int* _U_indptr;  // matrix
    int _U_nnz;
    double* _A_data;
    int* _A_indices;
    int* _A_indptr;
    int _A_nnz;
    double _mu, _eta; // tol?
    int _m_star, _s;

    DMGateCRep_Lindblad(double* A_data, int* A_indices, int* A_indptr, int nnz,
			double mu, double eta, int m_star, int s, int dim,
		        double* unitarypost_data, int* unitarypost_indices,
			int* unitarypost_indptr, int unitarypost_nnz);
    virtual ~DMGateCRep_Lindblad();
    virtual DMStateCRep* acton(DMStateCRep* state, DMStateCRep* out_state);
    virtual DMStateCRep* adjoint_acton(DMStateCRep* state, DMStateCRep* out_state);
  };



  // STATE VECTOR (SV) propagation

  // STATEs
  class SVStateCRep {
    public:
    dcomplex* _dataptr;
    int _dim;
    bool _ownmem;
    SVStateCRep(int dim);    
    SVStateCRep(dcomplex* data, int dim, bool copy);
    ~SVStateCRep();
    void print(const char* label);
    void copy_from(SVStateCRep* st);
  };

  // EFFECTSs
  class SVEffectCRep {
    public:
    int _dim;
    SVEffectCRep(int dim);
    virtual ~SVEffectCRep();
    virtual double probability(SVStateCRep* state) = 0;
    virtual dcomplex amplitude(SVStateCRep* state) = 0;
  };


  class SVEffectCRep_Dense :public SVEffectCRep {
    public:
    dcomplex* _dataptr;
    SVEffectCRep_Dense(dcomplex* data, int dim);
    virtual ~SVEffectCRep_Dense();
    virtual double probability(SVStateCRep* state);
    virtual dcomplex amplitude(SVStateCRep* state);;
  };

  class SVEffectCRep_TensorProd :public SVEffectCRep {
    public:
    dcomplex* _kron_array;
    int _max_factor_dim;
    int* _factordims;
    int _nfactors;

    SVEffectCRep_TensorProd(dcomplex* kron_array, int* factordims, int nfactors, int max_factor_dim, int dim);
    virtual ~SVEffectCRep_TensorProd();
    virtual double probability(SVStateCRep* state);
    virtual dcomplex amplitude(SVStateCRep* state);
  };


  // GATEs
  class SVGateCRep {
    public:
    int _dim;

    SVGateCRep(int dim);
    virtual ~SVGateCRep();
    virtual SVStateCRep* acton(SVStateCRep* state, SVStateCRep* out_state) = 0;
    virtual SVStateCRep* adjoint_acton(SVStateCRep* state, SVStateCRep* out_state) = 0;
  };

  class SVGateCRep_Dense :public SVGateCRep {
    public:
    dcomplex* _dataptr;
    SVGateCRep_Dense(dcomplex* data, int dim);
    virtual ~SVGateCRep_Dense();
    virtual SVStateCRep* acton(SVStateCRep* state, SVStateCRep* out_state);
    virtual SVStateCRep* adjoint_acton(SVStateCRep* state, SVStateCRep* out_state);
  };

  class SVGateCRep_Embedded :public SVGateCRep{
    public:
    SVGateCRep* _embedded_gate_crep;
    int* _noop_incrementers;
    int* _numBasisEls_noop_blankaction;
    int* _baseinds;
    int* _blocksizes; // basis blockdim**2 elements
    int _nComponents, _embeddedDim, _iActiveBlock, _nBlocks;
    
    
    SVGateCRep_Embedded(SVGateCRep* embedded_gate_crep, int* noop_incrementers,
			int* numBasisEls_noop_blankaction, int* baseinds, int* blocksizes,
			int embedded_dim, int nComponentsInActiveBlock, int iActiveBlock,
			int nBlocks, int dim);
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
    int _n, _2n, _namps;
    int* _smatrix;
    int* _pvectors;
    int _zblock_start;
    dcomplex* _amps;
    std::vector<std::vector<int> > _view_filters;
    bool _ownmem;
    
    SBStateCRep(int* smatrix, int* pvectors, dcomplex* amps, int namps, int n);
    SBStateCRep(int namps, int n);
    ~SBStateCRep();
    void push_view(std::vector<int>& view);
    void pop_view();
    void clifford_update(int* smatrix, int* svector, dcomplex* Umx);
    dcomplex extract_amplitude(std::vector<int>& zvals);
    double measurement_probability(std::vector<int>& zvals); //, qubit_filter);
    void copy_from(SBStateCRep* other);

    private:
    int udot1(int i, int j);
    void udot2(int* out, int* smatrix1, int* smatrix2);
    void colsum(int i, int j);
    void colswap(int i, int j);
    void rref();
    dcomplex apply_xgen(int igen, int pgen, std::vector<int>& zvals_to_acton,
			dcomplex ampl, std::vector<int>& result);
    dcomplex get_target_ampl(std::vector<int>& tgt, std::vector<int>& anchor,
			     dcomplex anchor_amp, int ip);
    dcomplex canonical_amplitude_of_target(int ip, std::vector<int>& target);
    void canonical_amplitudes_sample(int ip, std::vector<int> qs_to_sample,
				     std::vector<int>& anchor, std::vector<dcomplex>& amp_samples);

    void canonical_amplitudes(int ip, int* target, int qs_to_sample=1);
    void apply_clifford_to_frame(int* s, int* p, std::vector<int> qubit_filter);
    void apply_clifford_to_frame(int* s, int* p);
  };

  // EFFECTSs
  class SBEffectCRep { // a stabilizer measurement - just implement z-basis meas here for now(?)
    public:
    int _n;
    std::vector<int> _zvals;
    SBEffectCRep(int* zvals, int n);
    ~SBEffectCRep();
    dcomplex amplitude(SBStateCRep* state); //make virtual if we turn this into
    double probability(SBStateCRep* state); // a base class & derive from it.
  };


  // GATEs
  class SBGateCRep {
    public:
    int _n;
    SBGateCRep(int dim);
    virtual ~SBGateCRep();
    virtual SBStateCRep* acton(SBStateCRep* state, SBStateCRep* out_state) = 0;
    virtual SBStateCRep* adjoint_acton(SBStateCRep* state, SBStateCRep* out_state) = 0;
  };

  class SBGateCRep_Embedded :public SBGateCRep {
    public:
    std::vector<int> _qubits;
    SBGateCRep* _embedded_gate_crep;
    SBGateCRep_Embedded(SBGateCRep* embedded_gate_crep, int n, int* qubits, int nqubits);
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
    int _n;
    int *_smatrix, *_svector; //symplectic rep
    int *_smatrix_inv, *_svector_inv; //of the inverse, for adjoint ops
    dcomplex *_unitary, *_unitary_adj;
    SBGateCRep_Clifford(int* smatrix, int* svector, dcomplex* unitary,
			int* smatrix_inv, int* svector_inv, dcomplex* unitary_adj, int n);
    virtual ~SBGateCRep_Clifford();
    virtual SBStateCRep* acton(SBStateCRep* state, SBStateCRep* out_state);
    virtual SBStateCRep* adjoint_acton(SBStateCRep* state, SBStateCRep* out_state);
  };

  //Polynomial class
  class PolyCRep {
    public:
    std::unordered_map<int, dcomplex> _coeffs;
    int _max_order;
    int _max_num_vars;
    PolyCRep();
    PolyCRep(std::unordered_map<int, dcomplex> coeffs, int max_order, int max_num_vars);
    PolyCRep(const PolyCRep& other);
    ~PolyCRep();
    PolyCRep mult(const PolyCRep& other);
    void add_inplace(const PolyCRep& other);
    void scale(dcomplex scale);
    private:
    int vinds_to_int(std::vector<int> vinds);
    std::vector<int> int_to_vinds(int indx);
    int mult_vinds_ints(int i1, int i2); // multiply vinds corresponding to i1 & i2 and return resulting integer
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
