import tempfile

from ..util import BaseCase, with_temp_path

from pygsti.io import loaders


class LoadersTester(BaseCase):
    @with_temp_path
    def test_load_model(self, tmp_path):
        gateset4_txt = """
# Test text file describing a model

# State prepared, specified as a state in the Pauli basis (I,X,Y,Z)
PREP: rho
LiouvilleVec
1/sqrt(2) 0 0 1/sqrt(2)

# State measured as yes outcome, also specified as a state in the Pauli basis
POVM: Mdefault

EFFECT: 0
LiouvilleVec
1/sqrt(2) 0 0 1/sqrt(2)

EFFECT: 1
LiouvilleVec
1/sqrt(2) 0 0 -1/sqrt(2)

END POVM

GATE: Gi
LiouvilleMx
1 0 0 0
0 1 0 0
0 0 1 0
0 0 0 1

GATE: Gx
LiouvilleMx
1 0 0 0
0 1 0 0
0 0 0 1
0 0 -1 0

GATE: Gy
LiouvilleMx
1 0 0 0
0 0 0 -1
0 0 1 0
0 1 0 0

BASIS: pp 4
GAUGEGROUP: Full
"""
        with open(tmp_path, "w") as output:
            output.write(gateset4_txt)
        gateset4 = loaders.load_model(tmp_path)
        # TODO assert correctness
