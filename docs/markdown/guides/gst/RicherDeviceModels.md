---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.3
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# GST with richer device models

Standard GST assumes your device is a fixed set of gates acting on some qubits, with one state preparation and one measurement at the ends. That assumption is doing a lot of work, and real devices break it in several distinct ways. This chapter is about what to do when yours does.

The four pages here are not variations on a theme. Each one names a different place where the standard model runs out, and the fix is different in each case.

**Your device has operations that are not gates.** A mid-circuit measurement is not a gate: it produces a classical outcome partway through the circuit and steers what happens afterwards. pyGSTi represents these as *instruments*, which are collections of trace-non-preserving maps that sum to something trace-preserving. [Instruments and intermediate measurements](MidCircuitMeasurement) covers constructing them, how they are parameterized, and running GST on a model that contains them.

**Your device is not made of qubits.** If the computational unit has three levels rather than two, the whole state space changes, and so do the fiducials and germs you need to explore it. [Qutrit GST](QutritGST) works an example end to end.

**Your device leaves the computational subspace.** Leakage is the case where your qubits are qubits most of the time, and the model has to account for population escaping to a level you did not intend to use. [Leakage](Leakage) shows how to build a target model with the extra level and fit to it.

**Your device does not hold still.** Standard GST fits one set of gate parameters to all your data at once, which is exactly wrong if the device drifts over the course of the experiment. [Time-dependent GST](TimeDependentGST) covers time-aware circuits, timestamped data, and objective functions that re-simulate at each time.

If your device is straightforward and you are looking for how to describe it to pyGSTi in the first place, that is [describing your device](../workflow/DescribeYourDevice) rather than anything here. If the model you need is unusual in ways these pages do not cover, [custom operators](../../advanced/models/CustomOperators) is where the extension machinery lives.
