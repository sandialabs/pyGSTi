# Installing `pyGSTi`

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
git clone https://github.com/sandialabs/pyGSTi.git
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