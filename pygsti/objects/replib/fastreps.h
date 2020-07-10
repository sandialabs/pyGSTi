#include <complex>
#include <vector>
#include <unordered_map>
#include <cmath>

typedef std::complex<double> dcomplex;
typedef long long INT;


namespace CReps {
  
  //Polynomial key class - essentially a huge INT for holding many variable indices compactly
  class PolynomialVarsIndex {
    public:
    std::vector<INT> _parts;
    PolynomialVarsIndex() {}
    PolynomialVarsIndex(INT size) :_parts(size) {}

    bool operator==(PolynomialVarsIndex i) const {
      std::vector<INT>::const_iterator it, it2;
      if( i._parts.size() != this->_parts.size() ) return false;
      for(it=i._parts.begin(), it2=this->_parts.begin();
	  it != i._parts.end() && it2 != this->_parts.end();
	  ++it, ++it2) { //zip
	if(*it != *it2) return false;
      }
      return true;
    }

    bool operator<(PolynomialVarsIndex i) const {
      std::vector<INT>::const_iterator it, it2;
      if( i._parts.size() != this->_parts.size() )
	return this->_parts.size() < i._parts.size();
				     
      for(it=i._parts.begin(), it2=this->_parts.begin();
	  it != i._parts.end() && it2 != this->_parts.end();
	  ++it, ++it2) { //zip
	if(*it != *it2) return (*it2) < (*it);
      }
      return false; //equal, so not "<"
    }
  };
}

namespace std {

  //struct PolyVarsIndexHasher
  template <>
  struct hash<CReps::PolynomialVarsIndex>
  {
    std::size_t operator()(const CReps::PolynomialVarsIndex& k) const
    {
      using std::size_t;
      using std::hash;
      using std::string;

      std::size_t ret = 0;
      std::vector<INT>::const_iterator it;
      
      if(k._parts.size() == 0) return 0;
      for(it=k._parts.begin(); it != k._parts.end(); ++it) {
	ret = ret ^ hash<INT>()(*it); //TODO: rotate/shift bits to make a better hash here
      }
      return ret;
    }
  };
}

namespace CReps {

  //Forward declarations (as necessary)
  class DMOpCRep;

  //Helper functions
  void expm_multiply_simple_core(double* Adata, INT* Aindptr,
				 INT* Aindices, double* B,
				 INT N, double mu, INT m_star,
				 INT s, double tol, double eta,
				 double* F, double* scratch);
  void expm_multiply_simple_core_rep(DMOpCRep* A_rep, double* B,
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


  class DMEffectCRep_Errgen :public DMEffectCRep {
    public:
    DMOpCRep* _errgen_ptr;
    DMEffectCRep* _effect_ptr;
    INT _errgen_id;

    DMEffectCRep_Errgen(DMOpCRep* errgen_oprep, DMEffectCRep* effect_rep, INT errgen_id, INT dim);
    virtual ~DMEffectCRep_Errgen();
    virtual double probability(DMStateCRep* state);
  };


  // GATEs
  class DMOpCRep {
    public:
    INT _dim;

    DMOpCRep(INT dim);
    virtual ~DMOpCRep();
    virtual DMStateCRep* acton(DMStateCRep* state, DMStateCRep* out_state) = 0;
    virtual DMStateCRep* adjoint_acton(DMStateCRep* state, DMStateCRep* out_state) = 0;
  };

  class DMOpCRep_Dense :public DMOpCRep {
    public:
    double* _dataptr;
    DMOpCRep_Dense(double* data, INT dim);
    virtual ~DMOpCRep_Dense();
    virtual DMStateCRep* acton(DMStateCRep* state, DMStateCRep* out_state);
    virtual DMStateCRep* adjoint_acton(DMStateCRep* state, DMStateCRep* out_state);
  };

  class DMOpCRep_Embedded :public DMOpCRep{
    public:
    DMOpCRep* _embedded_gate_crep;
    INT* _noop_incrementers;
    INT* _numBasisEls_noop_blankaction;
    INT* _baseinds;
    INT* _blocksizes; // basis blockdim**2 elements
    INT _nComponents, _embeddedDim, _iActiveBlock, _nBlocks;
    
    
    DMOpCRep_Embedded(DMOpCRep* embedded_gate_crep, INT* noop_incrementers,
		      INT* numBasisEls_noop_blankaction, INT* baseinds, INT* blocksizes,
		      INT embedded_dim, INT nComponentsInActiveBlock, INT iActiveBlock,
		      INT nBlocks, INT dim);
    virtual ~DMOpCRep_Embedded();
    virtual DMStateCRep* acton(DMStateCRep* state, DMStateCRep* out_state);
    virtual DMStateCRep* adjoint_acton(DMStateCRep* state, DMStateCRep* out_state);
  };

  class DMOpCRep_Composed :public DMOpCRep{
    public:
    std::vector<DMOpCRep*> _factor_gate_creps;
    DMOpCRep_Composed(std::vector<DMOpCRep*> factor_gate_creps, INT dim);
    void reinit_factor_op_creps(std::vector<DMOpCRep*> new_factor_gate_creps);
    virtual ~DMOpCRep_Composed();
    virtual DMStateCRep* acton(DMStateCRep* state, DMStateCRep* out_state);
    virtual DMStateCRep* adjoint_acton(DMStateCRep* state, DMStateCRep* out_state);
  };

  class DMOpCRep_Sum :public DMOpCRep{
    public:
    std::vector<DMOpCRep*> _factor_creps;
    DMOpCRep_Sum(std::vector<DMOpCRep*> factor_creps, INT dim);
    virtual ~DMOpCRep_Sum();
    virtual DMStateCRep* acton(DMStateCRep* state, DMStateCRep* out_state);
    virtual DMStateCRep* adjoint_acton(DMStateCRep* state, DMStateCRep* out_state);
  };

  class DMOpCRep_Exponentiated :public DMOpCRep{
    public:
    DMOpCRep* _exponentiated_gate_crep;
    INT _power;
    
    DMOpCRep_Exponentiated(DMOpCRep* exponentiated_gate_crep, INT power, INT dim);
    virtual ~DMOpCRep_Exponentiated();
    virtual DMStateCRep* acton(DMStateCRep* state, DMStateCRep* out_state);
    virtual DMStateCRep* adjoint_acton(DMStateCRep* state, DMStateCRep* out_state);
  };


  class DMOpCRep_Lindblad :public DMOpCRep{
    public:
    DMOpCRep* _errgen_rep;
    double* _U_data;
    INT* _U_indices;
    INT* _U_indptr;
    INT _U_nnz;
    double _mu;
    double _eta;
    INT _m_star;
    INT _s;

    DMOpCRep_Lindblad(DMOpCRep* errgen_rep,
			double mu, double eta, INT m_star, INT s, INT dim,
		        double* unitarypost_data, INT* unitarypost_indices,
			INT* unitarypost_indptr, INT unitarypost_nnz);
    virtual ~DMOpCRep_Lindblad();
    virtual DMStateCRep* acton(DMStateCRep* state, DMStateCRep* out_state);
    virtual DMStateCRep* adjoint_acton(DMStateCRep* state, DMStateCRep* out_state);
  };

  class DMOpCRep_Sparse :public DMOpCRep{
    public:
    double* _A_data;
    INT* _A_indices;
    INT* _A_indptr;
    INT _A_nnz;

    DMOpCRep_Sparse(double* A_data, INT* A_indices, INT* A_indptr,
		      INT nnz, INT dim);
    virtual ~DMOpCRep_Sparse();
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

  class SVEffectCRep_Computational :public SVEffectCRep {
    public:
    INT _nfactors;
    INT _zvals_int;
    INT _nonzero_index;

    SVEffectCRep_Computational(INT nfactors, INT zvals_int, INT dim);
    virtual ~SVEffectCRep_Computational();
    virtual double probability(SVStateCRep* state);
    virtual dcomplex amplitude(SVStateCRep* state);
  };


  // GATEs
  class SVOpCRep {
    public:
    INT _dim;

    SVOpCRep(INT dim);
    virtual ~SVOpCRep();
    virtual SVStateCRep* acton(SVStateCRep* state, SVStateCRep* out_state) = 0;
    virtual SVStateCRep* adjoint_acton(SVStateCRep* state, SVStateCRep* out_state) = 0;
  };

  class SVOpCRep_Dense :public SVOpCRep {
    public:
    dcomplex* _dataptr;
    SVOpCRep_Dense(dcomplex* data, INT dim);
    virtual ~SVOpCRep_Dense();
    virtual SVStateCRep* acton(SVStateCRep* state, SVStateCRep* out_state);
    virtual SVStateCRep* adjoint_acton(SVStateCRep* state, SVStateCRep* out_state);
  };

  class SVOpCRep_Embedded :public SVOpCRep{
    public:
    SVOpCRep* _embedded_gate_crep;
    INT* _noop_incrementers;
    INT* _numBasisEls_noop_blankaction;
    INT* _baseinds;
    INT* _blocksizes; // basis blockdim**2 elements
    INT _nComponents, _embeddedDim, _iActiveBlock, _nBlocks;
    
    
    SVOpCRep_Embedded(SVOpCRep* embedded_gate_crep, INT* noop_incrementers,
		      INT* numBasisEls_noop_blankaction, INT* baseinds, INT* blocksizes,
		      INT embedded_dim, INT nComponentsInActiveBlock, INT iActiveBlock,
		      INT nBlocks, INT dim);
    virtual ~SVOpCRep_Embedded();
    virtual SVStateCRep* acton(SVStateCRep* state, SVStateCRep* out_state);
    virtual SVStateCRep* adjoint_acton(SVStateCRep* state, SVStateCRep* out_state);
  };

  class SVOpCRep_Composed :public SVOpCRep{
    public:
    std::vector<SVOpCRep*> _factor_gate_creps;
    SVOpCRep_Composed(std::vector<SVOpCRep*> factor_gate_creps, INT dim);
    void reinit_factor_op_creps(std::vector<SVOpCRep*> new_factor_gate_creps);
    virtual ~SVOpCRep_Composed();
    virtual SVStateCRep* acton(SVStateCRep* state, SVStateCRep* out_state);
    virtual SVStateCRep* adjoint_acton(SVStateCRep* state, SVStateCRep* out_state);
  };

  class SVOpCRep_Sum :public SVOpCRep{
    public:
    std::vector<SVOpCRep*> _factor_creps;
    SVOpCRep_Sum(std::vector<SVOpCRep*> factor_creps, INT dim);
    virtual ~SVOpCRep_Sum();
    virtual SVStateCRep* acton(SVStateCRep* state, SVStateCRep* out_state);
    virtual SVStateCRep* adjoint_acton(SVStateCRep* state, SVStateCRep* out_state);
  };

  class SVOpCRep_Exponentiated :public SVOpCRep{
    public:
    SVOpCRep* _exponentiated_gate_crep;
    INT _power;
    
    SVOpCRep_Exponentiated(SVOpCRep* exponentiated_gate_crep, INT power, INT dim);
    virtual ~SVOpCRep_Exponentiated();
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
    void print(const char* label);

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
  class SBOpCRep {
    public:
    INT _n;
    SBOpCRep(INT dim);
    virtual ~SBOpCRep();
    virtual SBStateCRep* acton(SBStateCRep* state, SBStateCRep* out_state) = 0;
    virtual SBStateCRep* adjoint_acton(SBStateCRep* state, SBStateCRep* out_state) = 0;
  };

  class SBOpCRep_Embedded :public SBOpCRep {
    public:
    std::vector<INT> _qubits;
    SBOpCRep* _embedded_gate_crep;
    SBOpCRep_Embedded(SBOpCRep* embedded_gate_crep, INT n, INT* qubits, INT nqubits);
    virtual ~SBOpCRep_Embedded();
    virtual SBStateCRep* acton(SBStateCRep* state, SBStateCRep* out_state);
    virtual SBStateCRep* adjoint_acton(SBStateCRep* state, SBStateCRep* out_state);
  };

  class SBOpCRep_Composed :public SBOpCRep {
    std::vector<SBOpCRep*> _factor_gate_creps;
    public:
    SBOpCRep_Composed(std::vector<SBOpCRep*> factor_gate_creps, INT n);
    virtual ~SBOpCRep_Composed();
    virtual SBStateCRep* acton(SBStateCRep* state, SBStateCRep* out_state);
    virtual SBStateCRep* adjoint_acton(SBStateCRep* state, SBStateCRep* out_state);
  };

  class SBOpCRep_Sum :public SBOpCRep {
    std::vector<SBOpCRep*> _factor_creps;
    public:
    SBOpCRep_Sum(std::vector<SBOpCRep*> factor_creps, INT n);
    virtual ~SBOpCRep_Sum();
    virtual SBStateCRep* acton(SBStateCRep* state, SBStateCRep* out_state);
    virtual SBStateCRep* adjoint_acton(SBStateCRep* state, SBStateCRep* out_state);
  };

  class SBOpCRep_Exponentiated :public SBOpCRep{
    public:
    SBOpCRep* _exponentiated_gate_crep;
    INT _power;
    
    SBOpCRep_Exponentiated(SBOpCRep* exponentiated_gate_crep, INT power, INT n);
    virtual ~SBOpCRep_Exponentiated();
    virtual SBStateCRep* acton(SBStateCRep* state, SBStateCRep* out_state);
    virtual SBStateCRep* adjoint_acton(SBStateCRep* state, SBStateCRep* out_state);
  };

  class SBOpCRep_Clifford :public SBOpCRep{
    public:
    INT _n;
    INT *_smatrix, *_svector; //symplectic rep
    INT *_smatrix_inv, *_svector_inv; //of the inverse, for adjoint ops
    dcomplex *_unitary, *_unitary_adj;
    SBOpCRep_Clifford(INT* smatrix, INT* svector, dcomplex* unitary,
			INT* smatrix_inv, INT* svector_inv, dcomplex* unitary_adj, INT n);
    virtual ~SBOpCRep_Clifford();
    virtual SBStateCRep* acton(SBStateCRep* state, SBStateCRep* out_state);
    virtual SBStateCRep* adjoint_acton(SBStateCRep* state, SBStateCRep* out_state);
  };

  
  //Polynomial class
  class PolynomialCRep {
    public:
    std::unordered_map<PolynomialVarsIndex, dcomplex> _coeffs;
    INT _max_num_vars;
    INT _vindices_per_int;
    PolynomialCRep();
    PolynomialCRep(std::unordered_map<PolynomialVarsIndex, dcomplex> coeffs, INT max_num_vars, INT vindices_per_int);
    PolynomialCRep(const PolynomialCRep& other);
    ~PolynomialCRep();
    PolynomialCRep abs();
    PolynomialCRep mult(const PolynomialCRep& other);
    PolynomialCRep abs_mult(const PolynomialCRep& other);
    void add_inplace(const PolynomialCRep& other);
    void add_abs_inplace(const PolynomialCRep& other);
    void add_scalar_to_all_coeffs_inplace(dcomplex offset);
    void scale(dcomplex scale);
    PolynomialVarsIndex vinds_to_int(std::vector<INT> vinds);
    std::vector<INT> int_to_vinds(PolynomialVarsIndex indx);
    private:
    PolynomialVarsIndex mult_vinds_ints(PolynomialVarsIndex i1, PolynomialVarsIndex i2); // multiply vinds corresponding to i1 & i2 and return resulting integer
  };
  
  //TERMS
  class SVTermCRep {
    public:
    PolynomialCRep* _coeff;
    double _magnitude;
    double _logmagnitude;
    SVStateCRep* _pre_state;
    SVEffectCRep* _pre_effect;
    std::vector<SVOpCRep*> _pre_ops;
    SVStateCRep* _post_state;
    SVEffectCRep* _post_effect;
    std::vector<SVOpCRep*> _post_ops;
    SVTermCRep(PolynomialCRep* coeff, double magnitude, double logmagnitude,
	       SVStateCRep* pre_state, SVStateCRep* post_state,
	       std::vector<SVOpCRep*> pre_ops, std::vector<SVOpCRep*> post_ops);
    SVTermCRep(PolynomialCRep* coeff, double magnitude, double logmagnitude,
	       SVEffectCRep* pre_effect, SVEffectCRep* post_effect,
	       std::vector<SVOpCRep*> pre_ops, std::vector<SVOpCRep*> post_ops);
    SVTermCRep(PolynomialCRep* coeff, double magnitude, double logmagnitude,
	       std::vector<SVOpCRep*> pre_ops, std::vector<SVOpCRep*> post_ops);
  };

  class SVTermDirectCRep {
    public:
    dcomplex _coeff;
    double _magnitude;
    double _logmagnitude;
    SVStateCRep* _pre_state;
    SVEffectCRep* _pre_effect;
    std::vector<SVOpCRep*> _pre_ops;
    SVStateCRep* _post_state;
    SVEffectCRep* _post_effect;
    std::vector<SVOpCRep*> _post_ops;
    SVTermDirectCRep(dcomplex coeff, double magnitude, double logmagnitude,
		     SVStateCRep* pre_state, SVStateCRep* post_state,
		     std::vector<SVOpCRep*> pre_ops, std::vector<SVOpCRep*> post_ops);
    SVTermDirectCRep(dcomplex coeff, double magnitude, double logmagnitude,
		     SVEffectCRep* pre_effect, SVEffectCRep* post_effect,
		     std::vector<SVOpCRep*> pre_ops, std::vector<SVOpCRep*> post_ops);
    SVTermDirectCRep(dcomplex coeff, double magnitude, double logmagnitude,
		     std::vector<SVOpCRep*> pre_ops, std::vector<SVOpCRep*> post_ops);
  };

  class SBTermCRep {
    public:
    PolynomialCRep* _coeff;
    double _magnitude;
    double _logmagnitude;
    SBStateCRep* _pre_state;
    SBEffectCRep* _pre_effect;
    std::vector<SBOpCRep*> _pre_ops;
    SBStateCRep* _post_state;
    SBEffectCRep* _post_effect;
    std::vector<SBOpCRep*> _post_ops;
    SBTermCRep(PolynomialCRep* coeff, double magnitude, double logmagnitude,
	       SBStateCRep* pre_state, SBStateCRep* post_state,
	       std::vector<SBOpCRep*> pre_ops, std::vector<SBOpCRep*> post_ops);
    SBTermCRep(PolynomialCRep* coeff, double magnitude, double logmagnitude,
	       SBEffectCRep* pre_effect, SBEffectCRep* post_effect,
	       std::vector<SBOpCRep*> pre_ops, std::vector<SBOpCRep*> post_ops);
    SBTermCRep(PolynomialCRep* coeff, double magnitude, double logmagnitude,
	       std::vector<SBOpCRep*> pre_ops, std::vector<SBOpCRep*> post_ops);
  };

  

}
