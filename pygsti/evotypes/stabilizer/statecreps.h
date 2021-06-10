#include <complex>
#include <vector>
#include <unordered_map>
#include <cmath>

typedef std::complex<double> dcomplex;
typedef long long INT;


namespace CReps {

  class StateCRep {
    public:
        public:
    INT _n, _2n, _namps;
    INT* _smatrix;
    INT* _pvectors;
    INT _zblock_start;
    dcomplex* _amps;
    std::vector<std::vector<INT> > _view_filters;
    bool _ownmem;
    
    StateCRep(INT* smatrix, INT* pvectors, dcomplex* amps, INT namps, INT n);
    StateCRep(INT namps, INT n);
    ~StateCRep();
    void push_view(std::vector<INT>& view);
    void pop_view();
    void clifford_update(INT* smatrix, INT* svector, dcomplex* Umx);
    dcomplex extract_amplitude(std::vector<INT>& zvals);
    double measurement_probability(std::vector<INT>& zvals); //, qubit_filter);
    void copy_from(StateCRep* other);
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

}
