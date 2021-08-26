import importlib
from unittest import mock

from ..util import BaseCase

warn_message = 'This might go badly'
error_message = 'Something terrible happened'
log_message = 'Data received'


class ModuleTestInstance(BaseCase):

    def test_basereps_module_exists(self):
        importlib.import_module("pygsti.evotypes.basereps_cython")

    def test_densitymx_modules_exist(self):
        importlib.import_module("pygsti.evotypes.densitymx.statereps")
        importlib.import_module("pygsti.evotypes.densitymx.opreps")
        importlib.import_module("pygsti.evotypes.densitymx.effectreps")

    def test_statevec_modules_exist(self):
        importlib.import_module("pygsti.evotypes.statevec.statereps")
        importlib.import_module("pygsti.evotypes.statevec.opreps")
        importlib.import_module("pygsti.evotypes.statevec.effectreps")
        importlib.import_module("pygsti.evotypes.statevec.termreps")

    def test_stabilizer_modules_exist(self):
        importlib.import_module("pygsti.evotypes.stabilizer.statereps")
        importlib.import_module("pygsti.evotypes.stabilizer.opreps")
        importlib.import_module("pygsti.evotypes.stabilizer.effectreps")
        importlib.import_module("pygsti.evotypes.stabilizer.termreps")

    def test_fastcalc_module_exists(self):
        importlib.import_module("pygsti.tools.fastcalc")

    def test_fastopcalc_module_exists(self):
        importlib.import_module("pygsti.baseobjs.opcalc.fastopcalc")

    def test_mapforwardsim_calc_modules_exist(self):
        importlib.import_module("pygsti.forwardsims.mapforwardsim_calc_densitymx")

    def test_termforwardsim_calc_modules_exist(self):
        importlib.import_module("pygsti.forwardsims.termforwardsim_calc_statevec")
        importlib.import_module("pygsti.forwardsims.termforwardsim_calc_stabilizer")

    def test_fastcircuitparser_module_exists(self):
        importlib.import_module("pygsti.circuits.circuitparser.fastcircuitparser")
