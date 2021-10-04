from pygsti.algorithms.contract import contract
from pygsti.modelmembers.povms import UnconstrainedPOVM
from . import fixtures
from ..util import BaseCase


class ContractFunctionBase(object):
    def setUp(self):
        super(ContractFunctionBase, self).setUp()
        self.options = dict(
            verbosity=10,
            tol=10.0
        )

    def test_contract(self):
        result = contract(self.model, self.target, **self.options)
        # TODO assert correctness


class ContractFunctionWithDataset(ContractFunctionBase):
    def setUp(self):
        super(ContractFunctionWithDataset, self).setUp()
        self.options.update(
            dataset=fixtures.ds_lgst
        )


class ContractLGSTModelBase(object):
    def setUp(self):
        super(ContractLGSTModelBase, self).setUp()
        self.model = fixtures.mdl_lgst.copy()


class ContractBigKickModelBase(ContractLGSTModelBase):
    def setUp(self):
        # big kick that should land it outside XP, TP, etc, so contraction
        # routines are more tested
        # XXX Are these useful test cases and if so, are the LGST model cases also useful?
        # By applying a kick, we ensure the model *needs* to be contracted, and provides
        # a more realistic & strenous test of the contraction routine.  The LGST cases are
        # alternatives, which often need contraction as well - we can probably just keep one
        # or the other (more important to assert correctness)
        super(ContractBigKickModelBase, self).setUp()
        self.model.kick(absmag=1.0)


class CPContractLGSTTester(ContractFunctionBase, ContractLGSTModelBase, BaseCase):
    target = "CP"


class NonDirectCPContractLGSTTester(CPContractLGSTTester):
    def setUp(self):
        super(NonDirectCPContractLGSTTester, self).setUp()
        self.options.update(
            use_direct_cp=False
        )


class CPContractBigKickTester(CPContractLGSTTester, ContractBigKickModelBase):
    pass


class TPContractLGSTTester(ContractFunctionBase, ContractLGSTModelBase, BaseCase):
    target = "TP"


class TPContractBigKickTester(TPContractLGSTTester, ContractBigKickModelBase):
    pass


class CPTPContractLGSTTester(ContractFunctionBase, ContractLGSTModelBase, BaseCase):
    target = "CPTP"


class NonDirectCPTPContractLGSTTester(CPTPContractLGSTTester):
    def setUp(self):
        super(NonDirectCPTPContractLGSTTester, self).setUp()
        self.options.update(
            use_direct_cp=False
        )


class ZeroMaxIterCPTPContractLGSTTester(CPTPContractLGSTTester):
    def setUp(self):
        super(ZeroMaxIterCPTPContractLGSTTester, self).setUp()
        self.options.update(
            maxiter=0
        )


class CPTPContractBigKickTester(CPTPContractLGSTTester, ContractBigKickModelBase):
    pass


#Removed these after removing the unused "forbidden_prob" function from tools/likelihoodfns.py
#class XPContractLGSTTester(ContractFunctionWithDataset, ContractLGSTModelBase, BaseCase):
#    target = "XP"
#
#
#class XPContractBigKickTester(XPContractLGSTTester, ContractBigKickModelBase):
#    pass
#
#
#class XPTPContractLGSTTester(ContractFunctionWithDataset, ContractLGSTModelBase, BaseCase):
#    target = "XPTP"
#
#
#class XPTPContractBigKickTester(XPTPContractLGSTTester, ContractBigKickModelBase):
#    pass


class VSPAMContractLGSTTester(ContractFunctionBase, ContractLGSTModelBase, BaseCase):
    target = "vSPAM"

    def test_contract_with_bad_effect(self):
        self.model.povms['Mdefault'] = UnconstrainedPOVM([('0', [100.0, 0, 0, 0])], evotype='default')  # E eigvals all > 1.0
        result = contract(self.model, self.target, **self.options)
        # TODO assert correctness
        self.model.povms['Mdefault'] = UnconstrainedPOVM([('0', [-100.0, 0, 0, 0])], evotype='default')  # E eigvals all < 0
        result = contract(self.model, self.target, **self.options)
        # TODO assert correctness


class VSPAMContractBigKickTester(VSPAMContractLGSTTester, ContractBigKickModelBase):
    pass


class NothingContractLGSTTester(ContractFunctionBase, ContractLGSTModelBase, BaseCase):
    target = "nothing"


class NothingContractBigKickTester(NothingContractLGSTTester, ContractBigKickModelBase):
    pass


class ContractExceptionTester(ContractLGSTModelBase, BaseCase):
    def test_contract_raises_on_bad_target(self):
        with self.assertRaises(ValueError):
            contract(self.model, "foobar")
