"""
Padding ExperimentDesigns out with idle lines
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

def pad_edesign_with_idle_lines(edesign, line_labels):
    """Utility to explicitly pad out ExperimentDesigns with idle lines.

    Parameters
    ----------
    edesign: ExperimentDesign
        The edesign to be padded.

    line_labels: tuple of int or str
        Full line labels for the padded edesign.

    Returns
    -------
    ExperimentDesign
        An edesign where all circuits have been padded out with missing idle lines
    """
    from pygsti.protocols import CombinedExperimentDesign as _CombinedDesign
    from pygsti.protocols import SimultaneousExperimentDesign as _SimulDesign

    if set(edesign.qubit_labels) == set(line_labels):
        return edesign
    
    if isinstance(edesign, _CombinedDesign):
        new_designs = {}
        for subkey, subdesign in edesign.items():
            new_designs[subkey] = pad_edesign_with_idle_lines(subdesign, line_labels)
        
        return _CombinedDesign(new_designs, qubit_labels=line_labels)

    # SimultaneousDesign with single design + full qubit labels tensors out the circuits with idle lines
    return _SimulDesign([edesign], qubit_labels=line_labels)
