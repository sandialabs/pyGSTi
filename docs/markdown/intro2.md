---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.3
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# pyGSTi
Welcome! This notebook serves as a top-level "readme" document for pyGSTi.  It and the other notebooks in this directory describe and demonstrate how you can use the `pygsti` package to accomplish various tasks related to quantum characterization.  We'll start with a brief general introduction and then move on to specifics.

## What does pyGSTi do?
PyGSTi is a **Python framework for modeling and analyzing collections of qubits**, more affectionately called quantum information processors (QIPs).  We use "**QIP**" generously to include systems of just 1 or 2 qubits in addition to larger systems.  Initially pyGSTi was developed to perform a single type of data analysis called Gate Set Tomography (GST), from where pyGSTi derives it's name.  While GST is still a central capability of the package, pyGSTi is capable of modeling QIP behavior and analyzing QIP data much more generally.

For example, some things you might use pyGSTi for are:

- constructing and manipulating quantum circuits.
- constructing a model for a QIP.
- computing the outcome probabilities of a circuit predicted by a QIP model.
- simulating observed data based on a QIP model. 
- testing how a particular QIP model agrees with real data taken.
- running high-level quantum characterization protocols such as:
    - Gate Set Tomography (GST) on 1 or 2 qubits
    - Clifford Randomized Benchmarking (RB)
    - Direct Randomized Benchmarking (DRB)
    - Robust Phase Estimation (RPE)
    - Multi-qubit reduced-model tomography
- performing "drift-detection" tests on multiple passes of ideally identical data.
- computing the process fidelity or diamond distance between two gate matrices.
- making sweet-looking figures that compare QIP models and data.
- creating integrated reports which explain characterization results (especially GST).


## Getting Started
This documentation facilitates **two basic approaches** to getting started with pyGSTi.  

The first approach is a **top-down** approach that shows how to use pyGSTi as a stand-alone QCVV tool that runs protocols on data.  An overview of what protocols pyGSTi can run and how to invoke them is given in the [pyGSTi protocols overview](Tutorials/00-Protocols.ipynb) notebook.

The second is a **bottom-up** approach that starts by introducing the main components of pyGSTi and how they work together, and then explains what you can do with them.  If you like this approach or are ambivalent, you should read through these two notebooks to get a high-level overview of what pyGSTi can do:
- [Part1: pyGSTi's essential objects](Tutorials/01-Essential-Objects.ipynb)
- [Part2: Using these objects](Tutorials/02-Using-Essential-Objects.ipynb)

(These notebooks contain links to other notebooks covering more advanced functionality that you are free to explore as desired.)

If you have a particular problem you need solved and you want to know what 5 lines of Python code will get it done for you, then you should visit the [FAQ](FAQ.ipynb) and with links to associated example notebooks.  If your question isn't answered there, you should email us at pygsti@sandia.gov or add an issue to the pyGSTi github page.
