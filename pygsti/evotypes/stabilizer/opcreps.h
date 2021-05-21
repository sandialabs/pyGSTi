#include <complex>
#include <vector>
#include <unordered_map>
#include <cmath>

typedef std::complex<double> dcomplex;
typedef long long INT;


namespace CReps {

  class OpCRep {
    public:
    INT _n;
    OpCRep(INT dim);
    virtual ~OpCRep();
    virtual StateCRep* acton(StateCRep* state, StateCRep* out_state) = 0;
    virtual StateCRep* adjoint_acton(StateCRep* state, StateCRep* out_state) = 0;
  };

  class OpCRep_Embedded :public OpCRep {
    public:
    std::vector<INT> _qubits;
    OpCRep* _embedded_gate_crep;
    OpCRep_Embedded(OpCRep* embedded_gate_crep, INT n, INT* qubits, INT nqubits);
    virtual ~OpCRep_Embedded();
    virtual StateCRep* acton(StateCRep* state, StateCRep* out_state);
    virtual StateCRep* adjoint_acton(StateCRep* state, StateCRep* out_state);
  };

  class OpCRep_Composed :public OpCRep {
    std::vector<OpCRep*> _factor_gate_creps;
    public:
    OpCRep_Composed(std::vector<OpCRep*> factor_gate_creps, INT n);
    void reinit_factor_op_creps(std::vector<OpCRep*> new_factor_gate_creps);
    virtual ~OpCRep_Composed();
    virtual StateCRep* acton(StateCRep* state, StateCRep* out_state);
    virtual StateCRep* adjoint_acton(StateCRep* state, StateCRep* out_state);
  };

  class OpCRep_Sum :public OpCRep {
    std::vector<OpCRep*> _factor_creps;
    public:
    OpCRep_Sum(std::vector<OpCRep*> factor_creps, INT n);
    virtual ~OpCRep_Sum();
    virtual StateCRep* acton(StateCRep* state, StateCRep* out_state);
    virtual StateCRep* adjoint_acton(StateCRep* state, StateCRep* out_state);
  };

  class OpCRep_Repeated :public OpCRep{
    public:
    OpCRep* _repeated_crep;
    INT _num_repetitions;
    
    OpCRep_Repeated(OpCRep* repeated_crep, INT num_repetitions, INT n);
    virtual ~OpCRep_Repeated();
    virtual StateCRep* acton(StateCRep* state, StateCRep* out_state);
    virtual StateCRep* adjoint_acton(StateCRep* state, StateCRep* out_state);
  };

  class OpCRep_Clifford :public OpCRep{
    public:
    INT _n;
    INT *_smatrix, *_svector; //symplectic rep
    INT *_smatrix_inv, *_svector_inv; //of the inverse, for adjoint ops
    dcomplex *_unitary, *_unitary_adj;
    OpCRep_Clifford(INT* smatrix, INT* svector, dcomplex* unitary,
			INT* smatrix_inv, INT* svector_inv, dcomplex* unitary_adj, INT n);
    virtual ~OpCRep_Clifford();
    virtual StateCRep* acton(StateCRep* state, StateCRep* out_state);
    virtual StateCRep* adjoint_acton(StateCRep* state, StateCRep* out_state);
  };

}
