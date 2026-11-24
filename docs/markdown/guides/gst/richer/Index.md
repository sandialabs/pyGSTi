# GST with richer device models

pyGSTi supports GST models with intermediate measurements, additional energy levels, and time-dependent operations. Choose the notebook that matches the features you want to characterize:

- [Instruments and intermediate measurements](Instruments) constructs and parameterizes instruments, which represent operations with classical outcomes, and runs GST on a model containing them.
- [Qutrit GST](QutritGST) demonstrates an analysis with a three-level computational system.
- [Leakage](Leakage) adds a level outside the computational subspace to a target model and fits that model to data.
- [Time-dependent GST](TimeDependentGST) uses timestamped data and time-dependent operations to fit changes during an experiment.

For gate names and device connectivity, see [Describing your device](../../../models/ProcessorSpecs). [Custom operators](../../../core/operators/CustomOperators) shows how to define an operation class for your model.
