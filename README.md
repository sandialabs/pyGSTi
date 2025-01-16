********************************************************************************
  pyGSTi 0.9.13
********************************************************************************

[![master build](https://github.com/sandialabs/pyGSTi/actions/workflows/beta-master.yml/badge.svg?branch=master)](https://github.com/sandialabs/pyGSTi/actions/workflows/beta-master.yml)
[![develop build](https://github.com/sandialabs/pyGSTi/actions/workflows/develop.yml/badge.svg?branch=develop)](https://github.com/sandialabs/pyGSTi/actions/workflows/develop.yml)
[![beta build](https://github.com/sandialabs/pyGSTi/actions/workflows/beta-master.yml/badge.svg?branch=beta)](https://github.com/sandialabs/pyGSTi/actions/workflows/beta-master.yml)

pyGSTi
------
**pyGSTi** is an open-source software for *modeling and characterizing noisy quantum information processors*
(QIPs), i.e., systems of one or more qubits.  It is licensed under the Apache License, Version 2.0.
Copyright information can be found in ``NOTICE``, and the license itself in ``LICENSE``.

There are three main objects in pyGSTi:
- `Circuit`: a quantum circuit (can have many qubits).
- `Model`: a description of a QIP's gate and SPAM operations (a noise model).
- `DataSet`: a dictionary-like container holding experimental data.

You can do various things by with these objects:

- **Circuit simulation**: compute a the outcome probabilities of a `Circuit` using a `Model`.
- **Data simulation**: simulate experimental data (a `DataSet`) using a `Model`.
- **Model testing**: Test whether a given `Model` fits the data in a `DataSet`.
- **Model estimation**: Estimate a `Model` from a `DataSet` (e.g. using GST).
- **Model-less characterization**: Perform Randomized Benchmarking on a `DataSet`.

In particular, there are a number of characterization protocols currently implemented in pyGSTi:
- **Gate Set Tomography (GST)** is the most complex and is where the software derives its name
 (a "python GST implementation").
- **Randomized Benchmarking (RB)** is a well-known method for assessing the
 quality of a QIP in an average sense.  PyGSTi implements standard "Clifford" RB
 as well as the more scalable "Direct" RB methods.
- **Robust Phase Estimation (RPE)** is a method designed for quickly learning
 a few noise parameters of a QIP that particularly useful for tuning up qubits.

PyGSTi is designed with a modular structure so as to be highly customizable
and easily integrated to new or existing python software.  It runs using
python 3.8 or higher.  To faclilitate integration with software for running
cloud-QIP experiments, pyGSTi `Circuit` objects can be converted to IBM's
**OpenQASM** and Rigetti Quantum Computing's **Quil** circuit description languages.

Installation
------------
Apart from several optional Cython modules, pyGSTi is written entirely in Python.
To install pyGSTi and only its required dependencies run:

``pip install pygsti``

**Or**, to install pyGSTi with all its optional dependencies too, run:

``pip install pygsti[complete]``

The disadvantage to these approaches is that the numerous tutorials
included in the package will then be buried within your Python's
`site_packages` directory, which you'll likely want to access later on.
**Alternatively**, you can **locally install** pyGSTi using the following commands:

~~~
cd <install_directory>
git clone https://github.com/pyGSTio/pyGSTi.git
cd pyGSTi
pip install -e .[complete]
~~~

As above, you can leave off the `.[complete]` if you only went the minimal
set of dependencies installed.  You could also replace the `git clone ...`
command with `unzip pygsti-0.9.x.zip` where the latter file is a downloaded
pyGSTi source archive.  Any of the above installations *should* build
the set of optional Cython extension modules if a working C/C++ compiler
and the `Cython` package are present.  If, however, compilation fails or
you later decided to add Cython support, you can rebuild the extension
modules (without reinstalling) if you've followed the local installation
approach above using the command:

`python setup.py build_ext --inplace`

Finally, [Jupyter notebook](http://jupyter.org/) is highly recommended as
it is generally convenient and the format of the included tutorials and
examples.  It is installed automatically when `[complete]` is used, otherwise
it can be installed separately.

Getting Started
---------------
Here's a couple of simple examples to get you started.

#### Circuit simulation
To compute the outcome probabilities of a circuit, you just need to create
a `Circuit` object (describing your circuit) and a `Model` object containing
the operations contained in your circuit.  Here we use a "stock" single-qubit `Model`
containing *Idle*, *X(&pi;/2)*, and *Y(&pi;/2)* gates labelled `Gi`, `Gx`,
and `Gy`, respectively:
~~~
import pygsti
from pygsti.modelpacks import smq1Q_XYI

mycircuit = pygsti.circuits.Circuit([('Gxpi2',0), ('Gypi2',0), ('Gxpi2',0)])
model = smq1Q_XYI.target_model()
outcome_probabilities = model.probabilities(mycircuit)
~~~


#### Gate Set Tomography
Gate Set Tomography is used to characterize the operations performed by
hardware designed to implement a (small) system of quantum bits (qubits).
Here's the basic idea:

  1. you tell pyGSTi what gates you'd ideally like to perform
  2. pyGSTi tells you what circuits it want's data for
  3. you perform the requested experiments and place the resulting
     data (outcome counts) into a text file that looks something like:

     ```
     ## Columns = 0 count, 1 count
     {} 0 100  # the empty sequence (just prep then measure)
     Gx 10 90  # prep, do a X(pi/2) gate, then measure
     GxGy 40 60  # prep, do a X(pi/2) gate followed by a Y(pi/2), then measure
     Gx^4 20 80  # etc...
     ```

  4. pyGSTi takes the data file and outputs a "report" - currently a
     HTML web page.

In code, running GST looks something like this:
~~~
import pygsti
from pygsti.modelpacks import smq1Q_XYI

# 1) get the ideal "target" Model (a "stock" model in this case)
mdl_ideal = smq1Q_XYI.target_model()

# 2) generate a GST experiment design
edesign = smq1Q_XYI.create_gst_experiment_design(4) # user-defined: how long do you want the longest circuits?

# 3) write a data-set template
pygsti.io.write_empty_dataset("MyData.txt", edesign.all_circuits_needing_data, "## Columns = 0 count, 1 count")

# STOP! "MyData.txt" now has columns of zeros where actual data should go.
# REPLACE THE ZEROS WITH ACTUAL DATA, then proceed with:
ds = pygsti.io.load_dataset("MyData.txt") # load data -> DataSet object

# OR: Create a simulated dataset with:
# ds = pygsti.data.simulate_data(mdl_ideal, edesign, num_samples=1000)

# 4) run GST (now using the modern object-based interface)
data = pygsti.protocols.ProtocolData(edesign, ds) # Step 1: Bundle up the dataset and circuits into a ProtocolData object
protocol = pygsti.protocols.StandardGST() # Step 2: Select a Protocol to run
results = protocol.run(data) # Step 3: Run the protocol!

# 5) Create a nice HTML report detailing the results
report = pygsti.report.construct_standard_report(results, title="My Report", verbosity=1)
report.write_html("myReport", auto_open=True, verbosity=1) # Can also write out Jupyter notebooks!
~~~

Tutorials and Examples
----------------------
There are numerous tutorials (meant to be pedagogical) and examples (meant to be demonstrate
how to do some particular thing) in the form of Jupyter notebooks beneath the `pyGSTi/jupyter_notebooks`
directory.  The root "START HERE" notebook will direct you where to go based on what you're most
interested in learning about.  You can view the
[read-only GitHub version of this notebook](https://github.com/pyGSTio/pyGSTi/blob/master/jupyter_notebooks/START_HERE.ipynb)
or you can [explore the tutorials interactively](https://mybinder.org/v2/gh/pyGSTio/pyGSTi/master)
using JupyterHub via Binder.  Note the existence of a
[FAQ](https://github.com/pyGSTio/pyGSTi/blob/master/jupyter_notebooks/FAQ.ipynb), which
addresses common issues.


#### Running notebooks *locally*
While it's possible to view the notebooks on GitHub using the links above, it's
usually nicer to run them *locally* so you can mess around with the code as
you step through it.  To do this, you'll need to start up a Jupyter notebook
server using the following steps (this assumes you've followed the *local
installation* directions above):

* Changing to the notebook directory, by running:
    ``cd jupyter_notebooks/Tutorials/``

* Start up the Jupyter notebook server by running:
  ``jupyter notebook``

The Jupyter server should open up your web browser to the server root, from
where you can start the first "START_HERE.ipynb" notebook.  Note that the key
command to execute a cell within the Jupyter notebook is ``Shift+Enter``, not
just ``Enter``.


Documentation
-------------
Online documentation is hosted on [Read the Docs](http://pygsti.readthedocs.io).

License
-------
PyGSTi is licensed under the [Apache License Version 2.0](https://github.com/pyGSTio/pyGSTi/blob/master/LICENSE).


Questions?
----------
For help and support with pyGSTi, please contact the authors at
pygsti@sandia.gov.
