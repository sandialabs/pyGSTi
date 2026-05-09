import pygsti.models.modelconstruction as mc
from pygsti.processors.processorspec import QubitProcessorSpec as _ProcessorSpec


#Mimics a function that used to be in pyGSTi, replaced with create_cloudnoise_model_from_hops_and_weights
def build_XYCNOT_cloudnoise_model(nQubits, geometry="line", cnot_edges=None,
                                  maxIdleWeight=1, maxSpamWeight=1, maxhops=0,
                                  extraWeight1Hops=0, extraGateWeight=0,
                                  roughNoise=None, simulator="matrix", parameterization="H+S",
                                  spamtype="lindblad", addIdleNoiseToAllGates=True,
                                  errcomp_type="gates", evotype="default", return_clouds=False, verbosity=0,
                                  xy_gate_names=('Gx', 'Gy')):

    availability = {}; nonstd_gate_unitaries = {}
    gatenames = ['{idle}', *xy_gate_names]
    if nQubits > 1:
        gatenames.append('Gcnot')
        if cnot_edges is not None:
            availability['Gcnot'] = cnot_edges
    pspec = _ProcessorSpec(nQubits, gatenames, nonstd_gate_unitaries, availability, geometry)
    assert(spamtype == "lindblad")  # unused and should remove this arg, but should always be "lindblad"
    mdl = mc.create_cloud_crosstalk_model_from_hops_and_weights(
        pspec, None,
        maxIdleWeight, maxSpamWeight, maxhops,
        extraWeight1Hops, extraGateWeight,
        simulator, evotype, parameterization, parameterization,
        "add_global" if addIdleNoiseToAllGates else "none",
        errcomp_type, True, True, True, 'pp', verbosity)

    if return_clouds:
        #FUTURE - just return cloud *keys*? (operation label values are never used
        # downstream, but may still be useful for debugging, so keep for now)
        return mdl, mdl.clouds
    else:
        return mdl
