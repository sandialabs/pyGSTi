from . import io
from . import core
from . import utils
from . import lattice
from . import estimation
from . import highrank
from . import highrank_sampling
from . import compressed_sensing
from . import singleton
from . import model_selection
from . import logical_decoration
from . import validation
from . import circuit_noise

# `report` (validation battery + HTML report generation) is deliberately not
# imported here: it pulls in matplotlib.pyplot, which the rest of the package
# does not need. Import it explicitly:
#
#     from pygsti.extras.sparsedem import report
