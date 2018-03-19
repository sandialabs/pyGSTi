"""A python implementation of Gate Set Tomography"""

from distutils.core import setup
from distutils.extension import Extension

#execfile("packages/pygsti/_version.py")

# 3.0 changes the way exec has to be called
with open("packages/pygsti/_version.py") as f:
    code = compile(f.read(), "packages/pygsti/_version.py", 'exec')
    exec(code)

try:
    import numpy as np
    from Cython.Build import cythonize
    ext_modules = [
        Extension("pygsti.tools.fastcalc",
                  sources=["packages/pygsti/tools/fastcalc.pyx"], # , "fastcalc.c
                  # Cython docs on NumPy usage should mention this!
                  #define_macros = [('NPY_NO_DEPRECATED_API','NPY_1_7_API_VERSION')],
                  #leave above commented, see http://docs.cython.org/en/latest/src/reference/compilation.html#configuring-the-c-build
                  #define_macros = [('CYTHON_TRACE','1')], #for profiling
                  include_dirs=['.', np.get_include()]
                  #libraries=['m'] #math lib?
                  )
        ]
    ext_modules = cythonize(ext_modules)
except ImportError: # if Cython isn't available (e.g. in readthedocs) just skip
    #print warning??
    ext_modules = []
    

classifiers = """\
Development Status :: 4 - Beta
Intended Audience :: Science/Research
License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)
Programming Language :: Python
Topic :: Scientific/Engineering :: Physics
Operating System :: Microsoft :: Windows
Operating System :: MacOS :: MacOS X
Operating System :: Unix
"""

descriptionTxt = """\
Gate set tomography (GST) is a quantum tomography protocol that provides full characterization of a quantum logic device (e.g. a qubit).  GST estimates a set of quantum logic gates and (simultaneously) the associated state preparation and measurement (SPAM) operations.  GST is self-calibrating.  This eliminates a key limitation of traditional quantum state and process tomography, which characterize either states (assuming perfect processes) or processes (assuming perfect state preparation and measurement), but not both together.  Compared with benchmarking protocols such as randomized benchmarking, GST provides much more detailed and accurate information about the gates, but demands more data.  The primary downside of GST has been its complexity.  Whereas benchmarking and state/process tomography data can be analyzed with relatively simple algorithms, GST requires more complex algorithms and more fine-tuning (linear GST is an exception that can be implemented easily).  pyGSTi addresses and eliminates this obstacle by providing a fully-featured, publicly available implementation of GST in the Python programming language.

The primary goals of the pyGSTi project are to:

- provide efficient and robust implementations of Gate Set Tomography algorithms;
- allow straightforward interoperability with other software;
- provide a powerful high-level interface suited to inexperienced programmers, so that
  common GST tasks can be performed using just one or two lines of code;
- use modular design to make it easy for users to modify, customize, and extend GST functionality.
"""

setup(name='pyGSTi',
      version=__version__,
      description='A python implementation of Gate Set Tomography',
      long_description=descriptionTxt,
      author='Erik Nielsen, Kenneth Rudinger, John Gamble, Robin Blume-Kohout',
      author_email='pygsti@sandia.gov',
      packages=['pygsti', 'pygsti.algorithms', 'pygsti.construction', 'pygsti.drivers', 'pygsti.io', 'pygsti.objects', 'pygsti.optimize', 'pygsti.report', 'pygsti.tools'],
      package_dir={'': 'packages'},
      package_data={'pygsti.tools': ['fastcalc.pyx'],
                    'pygsti.report': ['templates/*.tex', 'templates/*.html', 'templates/*.json',
                                      'templates/report_notebook/*.txt',
                                      'templates/standard_html_report/*.html',
                                      'templates/offline/README.txt',
                                      'templates/offline/*.js',
                                      'templates/offline/*.css',
                                      'templates/offline/fonts/*',
                                      'templates/offline/images/*']},
      requires=['numpy','scipy','plotly','ply'],
      extras_require = {
           'diamond norm computation':  ['cvxpy', 'cvxopt'],
           'nose testing' : ['nose'],
           'image comparison' : ['Pillow'],
           'accurate memory profiling' : ['psutil'],
           'multi-processor support' : ['mpi4py'],
           'evolutionary optimization algorithm': ['deap'],
           'pickling report tables': ['pandas'],
           'generating PDFs of report figures': ['matplotlib'],
           'generating report notebooks': ['ipython','notebook'],
           'read/write message pack format': ['msgpack-python'],
	   'extension modules': ['cython']
      },
      platforms = ["any"],      
      url = 'http://www.pygsti.info',
      download_url = 'https://github.com/pyGSTio/pyGSTi/tarball/master',
      keywords = ['pygsti', 'tomography', 'gate set', 'pigsty', 'pig', 'quantum', 'qubit'],
      classifiers = list(filter(None, classifiers.split("\n"))),
      ext_modules=ext_modules,
     )
