#include <complex>
#include <vector>
#include <cmath>

typedef std::complex<double> dcomplex;
typedef long long INT;


namespace CReps {

  class TermCRep {
    public:
    PolynomialCRep* _coeff;
    double _magnitude;
    double _logmagnitude;
    StateCRep* _pre_state;
    EffectCRep* _pre_effect;
    std::vector<OpCRep*> _pre_ops;
    StateCRep* _post_state;
    EffectCRep* _post_effect;
    std::vector<OpCRep*> _post_ops;
    TermCRep(PolynomialCRep* coeff, double magnitude, double logmagnitude,
	       StateCRep* pre_state, StateCRep* post_state,
	       std::vector<OpCRep*> pre_ops, std::vector<OpCRep*> post_ops);
    TermCRep(PolynomialCRep* coeff, double magnitude, double logmagnitude,
	       EffectCRep* pre_effect, EffectCRep* post_effect,
	       std::vector<OpCRep*> pre_ops, std::vector<OpCRep*> post_ops);
    TermCRep(PolynomialCRep* coeff, double magnitude, double logmagnitude,
	       std::vector<OpCRep*> pre_ops, std::vector<OpCRep*> post_ops);
  };
}
