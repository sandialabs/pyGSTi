#include <complex>
#include <vector>
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
    virtual double amplitude(DMStateCRep* state) = 0;
  };


  class DMEffectCRep_Dense :public DMEffectCRep {
    public:
    double* _dataptr;
    DMEffectCRep_Dense(double* data, int dim);
    virtual ~DMEffectCRep_Dense();
    virtual double amplitude(DMStateCRep* state);
  };

  class DMEffectCRep_TensorProd :public DMEffectCRep {
    public:
    double* _kron_array;
    int _max_factor_dim;
    int* _factordims;
    int _nfactors;

    DMEffectCRep_TensorProd(double* kron_array, int* factordims, int nfactors, int max_factor_dim, int dim);
    virtual ~DMEffectCRep_TensorProd();
    virtual double amplitude(DMStateCRep* state);
  };


  // GATEs
  class DMGateCRep {
    public:
    int _dim;

    DMGateCRep(int dim);
    virtual ~DMGateCRep();
    virtual DMStateCRep* acton(DMStateCRep* state, DMStateCRep* out_state) = 0;
  };

  class DMGateCRep_Dense :public DMGateCRep {
    public:
    double* _dataptr;
    DMGateCRep_Dense(double* data, int dim);
    virtual ~DMGateCRep_Dense();
    virtual DMStateCRep* acton(DMStateCRep* state, DMStateCRep* out_state);
  };

  class DMGateCRep_Embedded :public DMGateCRep{
    public:
    DMGateCRep* _embedded_gate_crep;
    int* _noop_incrementers;
    int* _numBasisEls_noop_blankaction;
    int* _baseinds;
    int* _blocksizes; // basis blockdim**2 elements
    int _nComponents, _nActive, _iActiveBlock, _nBlocks;
    
    
    DMGateCRep_Embedded(DMGateCRep* embedded_gate_crep, int* noop_incrementers,
			int* numBasisEls_noop_blankaction, int* baseinds, int* blocksizes,
			int nActive, int nComponents, int iActiveBlock, int nBlocks, int dim);
    virtual ~DMGateCRep_Embedded();
    virtual DMStateCRep* acton(DMStateCRep* state, DMStateCRep* out_state);
  };

  class DMGateCRep_Composed :public DMGateCRep{
    public:
    std::vector<DMGateCRep*> _factor_gate_creps;
    DMGateCRep_Composed(std::vector<DMGateCRep*> factor_gate_creps, int nfactors, int dim); //TODO: remove nfactors?
    virtual ~DMGateCRep_Composed();
    virtual DMStateCRep* acton(DMStateCRep* state, DMStateCRep* out_state);
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
  };


  // SIMILAR for STATE VECS (SV) propagation...
  
  // <<TODO>>


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
    
    SBStateCRep(int* smatrix, int* pvectors, dcomplex* amps, int namps, int n);
    ~SBStateCRep();
    void push_view(std::vector<int>& view);
    void pop_view();
    void clifford_update(int* smatrix, int* svector, dcomplex* Umx);
    dcomplex extract_amplitude(std::vector<int>& zvals);
    double measurement_probability(std::vector<int>& zvals); //, qubit_filter);

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
  };

  // EFFECTSs
  class SBEffectCRep { // a stabilizer measurement - just implement z-basis meas here for now(?)
    public:
    int _n;
    int* _zvals;
    SBEffectCRep(int* zvals, int dim);
    ~SBEffectCRep();
    virtual double amplitude(SBStateCRep* state);
  };


  // GATEs
  class SBGateCRep {
    public:
    int _n;
    SBGateCRep(int dim);
    ~SBGateCRep();
    virtual SBStateCRep* acton(SBStateCRep* state, SBStateCRep* out_state) = 0;
  };

  class SBGateCRep_Embedded :public SBGateCRep {
    public:
    int _n;
    int _targetQs;
    SBGateCRep* _embedded_gate_crep;
    SBGateCRep_Embedded(int n); //TODO
    ~SBGateCRep_Embedded();
    virtual SBStateCRep* acton(SBStateCRep* state, SBStateCRep* out_state);
  };

  class SBGateCRep_Composed :public SBGateCRep {
    SBGateCRep* _factor_gate_creps;
    int _nfactors;
    public:
    SBGateCRep_Composed(SBGateCRep* factor_gate_creps, int nfactors, int n);
    ~SBGateCRep_Composed();
    virtual SBStateCRep* acton(SBStateCRep* state, SBStateCRep* out_state);
  };

  class SBGateCRep_Clifford :public SBGateCRep{
    public:
    int _n;
    int *_smatrix, *_svector; //symplectic rep
    dcomplex* _unitary;
    SBGateCRep_Clifford(int* smatrix, int* svector, dcomplex* unitary, int n);
    ~SBGateCRep_Clifford();
    virtual SBStateCRep* acton(SBStateCRep* state, SBStateCRep* out_state);
  };


}
