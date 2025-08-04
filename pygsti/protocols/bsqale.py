from mirror_edesign import (qiskit_circuits_to_mirror_edesign,
                            qiskit_circuits_to_fullstack_mirror_edesign,
                            qiskit_circuits_to_svb_mirror_edesign)

# right now this file is functioning as a wrapper for a bunch of functionality that exists in pygsti.protocols.mirror_edesign. A refactor is probably in order.

import mirror_edesign

def qiskit_circuits_to_mirror_edesign(qk_circs, # yes transpiled
                                          mirroring_kwargs_dict={}
                                          ):

    return mirror_edesign.qiskit_circuits_to_mirror_edesign(qk_circs, mirroring_kwargs_dict)


def qiskit_circuits_to_fullstack_mirror_edesign(qk_circs, #not transpiled
                                                    qk_backend=None,
                                                    coupling_map=None,
                                                    basis_gates=None,
                                                    transpiler_kwargs_dict={},
                                                    mirroring_kwargs_dict={},
                                                    num_transpilation_attempts=100,
                                                    return_qiskit_time=False
                                                    ):

    return mirror_edesign.qiskit_circuits_to_fullstack_mirror_edesign(qk_circs, #not transpiled
                                                    qk_backend,
                                                    coupling_map,
                                                    basis_gates,
                                                    transpiler_kwargs_dict,
                                                    mirroring_kwargs_dict,
                                                    num_transpilation_attempts,
                                                    return_qiskit_time
                                                    )

def qiskit_circuits_to_svb_mirror_edesign(qk_circs,
                                              width_depth_dict,
                                              backend_num_qubits,
                                              coupling_map,
                                              instruction_durations,
                                              aggregate_subcircs,
                                              subcirc_kwargs_dict={},
                                              mirroring_kwargs_dict={}
                                              ): # qk_circs must already be transpiled to the device

    return mirror_edesign.qiskit_circuits_to_svb_mirror_edesign(qk_circs,
                                              width_depth_dict,
                                              backend_num_qubits,
                                              coupling_map,
                                              instruction_durations,
                                              aggregate_subcircs,
                                              subcirc_kwargs_dict,
                                              mirroring_kwargs_dict
                                              ) # qk_circs must already be transpiled to the device