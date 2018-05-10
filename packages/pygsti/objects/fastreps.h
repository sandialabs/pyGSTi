namespace CReps {


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
  };

  // EFFECTSs
  class DMEffectCRep {
    public:
    int _dim;
    DMEffectCRep(int dim);
    ~DMEffectCRep();
    virtual double amplitude(DMStateCRep* state) = 0;
  };


  class DMEffectCRep_Dense :public DMEffectCRep {
    public:
    double* _dataptr;
    DMEffectCRep_Dense(double* data, int dim);
    ~DMEffectCRep_Dense();
    virtual double amplitude(DMStateCRep* state);
  };

  class DMEffectCRep_TensorProd :public DMEffectCRep {
    public:
    double* _kron_array;
    int _kron_array_dim1;
    int* _factordims;
    int _nfactors;

    DMEffectCRep_TensorProd();
    //Other constructor...
    ~DMEffectCRep_TensorProd();
    virtual double amplitude(DMStateCRep* state);
  };


  // GATEs
  class DMGateCRep {
    public:
    int _dim;

    DMGateCRep(int dim);
    ~DMGateCRep();
    virtual DMStateCRep* acton(DMStateCRep* state, DMStateCRep* out_state) = 0;
  };

  class DMGateCRep_Dense :public DMGateCRep{
    public:
    double* _dataptr;
    DMGateCRep(double* data, int dim);
    ~DMGateCRep();
    virtual DMStateCRep* acton(DMStateCRep* state, DMStateCRep* out_state);
  };

  class DMGateCRep_Embedded :public DMGateCRep{
    public:
    DMGateCRep* _embedded_gate_crep;
    int noop_incrementers; //TODO
    int numBasisEls_noop_blankaction; //TODO
    int baseinds; //TODO
    
    DMGateCRep_Embedded(int dim);
    ~DMGateCRep_Embedded();
    virtual DMStateCRep* acton(DMStateCRep* state, DMStateCRep* out_state);
  };

  class DMGateCRep_Composed :public DMGateCRep{
    public:
    DMGateCRep* _factor_gate_creps;
    int _nfactors;
    DMGateCRep_Composed(DMGateCRep* factor_gate_creps, int nfactors, int dim);
    ~DMGateCRep_Composed();
    virtual DMStateCRep* acton(DMStateCRep* state, DMStateCRep* out_state);
  };

  class DMGateCRep_Lindblad :public DMGateCRep{
    public:
    double* unitary_postfactor; // might need to be a sparse mx?  <different rep case?>
    double* A, mu, m_star, s, eta; //prep for matrix mult...

    DMGateCRep_Lindblad(double* data, int dim);
    ~DMGateCRep_Linblad();
    virtual DMStateCRep* acton(DMStateCRep* state, DMStateCRep* out_state);
  };


  // SIMILAR for STATE VECS (SV) propagation...
  
  // <<TODO>>


  // STABILIZER propagation

    // STATEs
  class SBStateCRep {  // a stabilizer frame (perhaps optionally w/out phase info, so just a state?)
    public:
    int _n;
    double* _smatrix, _pvector;
    SBStateCRep(int dim);    
    SBStateCRep(double* data, int dim, bool copy);
    ~SBStateCRep();
  };

  // EFFECTSs
  class SBEffectCRep { // a stabilizer measurement - just implement z-basis meas here for now(?)
    public:
    int _n;
    int* _zvals;
    SBEffectCRep(int dim);
    ~SBEffectCRep();
    virtual double amplitude(SBStateCRep* state);
  };


  // GATEs
  class SBGateCRep {
    public:
    SBGateCRep();
    SBGateCRep(double* data, int dim);
    ~SBGateCRep();
    virtual SBStateCRep* acton(SBStateCRep* state, SBStateCRep* out_state);
  };

  class SBGateCRep_Embedded {
    public:
    int _n;
    int _targetQs;
    SBGateCRep* _embedded_gate_crep;
    SBGateCRep_Embedded();
    SBGateCRep_Embedded(double* data, int dim);
    ~SBGateCRep_Embedded();
    virtual SBStateCRep* acton(SBStateCRep* state, SBStateCRep* out_state);
  };

  class SBGateCRep_Composed {
    SBGateCRep* _factor_gate_creps;
    public:
    SBGateCRep_Composed();
    SBGateCRep_Composed(double* data, int dim);
    ~SBGateCRep_Composed();
    virtual SBStateCRep* acton(SBStateCRep* state, SBStateCRep* out_state);
  };

  class SBGateCRep_Clifford {
    public:
    int _n;
    double* _smatrix, _svector; //symplectic rep
    SBGateCRep_Clifford();
    SBGateCRep_Clifford(double* data, int dim);
    ~SBGateCRep_Clifford();
    virtual SBStateCRep* acton(SBStateCRep* state, SBStateCRep* out_state);
  };


}
