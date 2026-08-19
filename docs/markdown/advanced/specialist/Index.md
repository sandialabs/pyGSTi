# Specialist protocols

These pages cover protocols with a narrow audience: each one answers a question that matters a great deal to some people and not at all to most. They are grouped here so the main [characterization guides](../../guides/Index) stay short, not because they are unfinished or unsupported.

Read them individually — there is no reading order, and nothing here depends on anything else here.

- [Robust phase estimation](RobustPhaseEstimation) — a small number of circuits gives a precise estimate of a few rotation angles. Much cheaper than GST when those angles are all you want.
- [Idle tomography](IdleTomography) — characterizes the errors in a multi-qubit idle from a small, intuitive circuit set. **Read its warning first:** the reported intrinsic stochastic rates currently disagree with injected rates by a factor of two, and the discrepancy is unresolved.
- [Parity benchmarking](ParityBenchmarking) — the weight-$X$ disturbance metric between a reference and a test dataset, applied to a four-qubit parity check.

Two neighbouring chapters carry material that might otherwise look specialist. [Mastering GST](../../guides/gst/RunningGST) covers [qutrit GST](../../guides/gst/QutritGST), [time-dependent GST](../../guides/gst/TimeDependentGST) and [the TVD objective](../../guides/gst/RobustGST-TVD); [randomized benchmarking](../../guides/rb/HowRBWorks) covers [SU(2) qudit RB](../../guides/rb/SU2QuditRB).
