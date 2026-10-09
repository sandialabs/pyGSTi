# Which guide do I need?

This part is the practitioner layer. It assumes you have been through [Start here](../start/Index), or that you already know what you want to run, and it covers each protocol at full width — the options, the failure modes, and the variants that the guided path leaves out.

The chapters split into two kinds, and knowing which is which saves a lot of scrolling.

## Pick the chapter for your protocol

- Gate set tomography — a complete, self-consistent description of every gate, preparation and measurement in a small gate set. Four chapters, because GST has the most ways to go wrong: [designing the circuits](gst/GSTDesigns), [running the fit](gst/RunningGST), [judging the fit](gst/JudgingTheFit) and what to do when it is bad, and [richer device models](gst/RicherDeviceModels) for mid-circuit measurement, qutrits, leakage and time dependence.
- [Randomized benchmarking](rb/HowRBWorks) — error rates from random circuits, in several flavours. Start with this chapter rather than a specific flavour; it explains the shared workflow and then points at Clifford, direct, mirror and binary RB, the simultaneous and qudit variants, and how to simulate RB data.
- [Volumetric benchmarking](benchmarks/VolumetricBenchmarks) — how wide and how deep your circuits can get before the signal dies, as a map over circuit shapes rather than a single number. Its one section, [mirror circuit fidelity estimation](benchmarks/MirrorFidelityEstimation), estimates the process fidelity of a circuit you supply by mirroring it.
- [Drift characterization](drift/DriftCharacterization) — detecting and characterizing instability from time-stamped data, on any circuits and any number of qubits.

## Chapters every protocol draws on

- [Running QCVV protocols](workflow/Workflow) — the experiment design → data → protocol → results pattern that all of the above share. Read it once and the rest of this part reads faster.
- [Modeling a noisy device](models/DeviceModels) — which kind of model to feed a protocol, where its noise-free target comes from (a model pack or a processor spec), and the noise you put on it. Assembling an [explicit model](../internals/models/ExplicitModels) matrix by matrix is covered under Internals.
- [Results](analysis/Results) — what a protocol hands back, and how to get numbers, error bars and reports out of it. Also where [gauge freedom in practice](analysis/GaugeFreedom) lives.

## Getting off simulated data

[Getting your own data in](../start/YourOwnData) is the general route, whatever the hardware. On IBM Q hardware pyGSTi can also [submit the design and collect the counts itself](../advanced/interop/IBMQ).

If none of this fits, [Advanced capabilities](../advanced/Index) holds what only some readers need (specialist protocols, machine-learned error models, other platforms' hardware and tooling), and [Internals](../internals/Index) is the machinery underneath: operators, conventions, simulators, the extension points beneath the `Protocol` layer, and the figures that reports are assembled from. A specific problem may already be answered in the [FAQ](../reference/Troubleshooting).
