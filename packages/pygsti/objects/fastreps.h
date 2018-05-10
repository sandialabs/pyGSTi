namespace CReps {

  //class StateCRep {
  //};
  
  class DMStateCRep { //:public StateCRep {
    public:
    double* _dataptr;
    int _dim;
    bool _ownmem;

    DMStateCRep();
    DMStateCRep(int dim);    
    DMStateCRep(double* data, int dim, bool copy);
    ~DMStateCRep();
  };


  //class EffectCRep {
  //};
  
  class DMEffectCRep { //: public EffectCRep {
    public:
    double* _dataptr;
    int _dim;

    DMEffectCRep();
    DMEffectCRep(double* data, int dim);
    ~DMEffectCRep();
    double amplitude(DMStateCRep* state);
  };


  class DMGateCRep {
    public:
    double* _dataptr;
    int _dim;

    DMGateCRep();
    DMGateCRep(double* data, int dim);
    ~DMGateCRep();
    DMStateCRep* acton(DMStateCRep* state, DMStateCRep* out_state);
  };

}
