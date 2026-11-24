# Running GST

GST fits a model to the outcome counts from an experiment design. If you haven't run a fit before, [Your first GST run](../../../start/FirstGST) takes you from a model pack to a report.

- [GST protocol options](ProtocolOptions) shows how to use `GateSetTomography` and `StandardGST`, choose model parameterizations, configure gauge optimization, and save results. Start here when you need more control over a fit.
- [2Q-GST fitting](TwoQubitGST) runs a two-qubit analysis, including circuit reduction and memory settings.
- [Running GST in parallel with MPI](Parallelism) covers distributed fits, checkpointing, and restarting from saved progress.

If you still need circuits, see [GST experiment designs](../designs/Index). After fitting, use [Judging GST fits](../fit/Index) to decide whether the model describes your data.
