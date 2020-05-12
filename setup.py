"""A python implementation of Gate Set Tomography"""

from warnings import warn

try:
    from setuptools import setup
    from setuptools import Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension

descriptionTxt = """\
Gate set tomography (GST) is a quantum tomography protocol that provides full characterization of a quantum logic device
(e.g. a qubit).  GST estimates a set of quantum logic gates and (simultaneously) the associated state preparation and
measurement (SPAM) operations.  GST is self-calibrating.  This eliminates a key limitation of traditional quantum state
and process tomography, which characterize either states (assuming perfect processes) or processes (assuming perfect
state preparation and measurement), but not both together.  Compared with benchmarking protocols such as randomized
benchmarking, GST provides much more detailed and accurate information about the gates, but demands more data.  The
primary downside of GST has been its complexity.  Whereas benchmarking and state/process tomography data can be analyzed
with relatively simple algorithms, GST requires more complex algorithms and more fine-tuning (linear GST is an exception
that can be implemented easily).  pyGSTi addresses and eliminates this obstacle by providing a fully-featured, publicly
available implementation of GST in the Python programming language.

The primary goals of the pyGSTi project are to:

- provide efficient and robust implementations of Gate Set Tomography algorithms;
- allow straightforward interoperability with other software;
- provide a powerful high-level interface suited to inexperienced programmers, so that
  common GST tasks can be performed using just one or two lines of code;
- use modular design to make it easy for users to modify, customize, and extend GST functionality.
"""

# Extra requirements
extras = {
    'diamond_norm': [
        'cvxopt',
        'cvxpy'
    ],
    'memory_profiling': ['psutil'],
    'multiprocessor': ['mpi4py'],
    'evolutionary_optimization': ['deap'],
    'report_pickling': ['pandas'],
    'report_pdf_figures': ['matplotlib'],
    'html_reports': ['jinja2'],
    'notebooks': [
        'ipython',
        'notebook'
    ],
    'msgpack': ['msgpack'],
    'extensions': ['cython'],
    'linting': [
        'autopep8',
        'flake8'
    ],
    'testing': [
        'coverage',
        'cvxopt',
        'cvxpy',
        'cython',
        'matplotlib',
        'mpi4py',
        'msgpack',
        'nose',
        'nose-timer',
        'pandas',
        'psutil',
        'rednose',
        'zmq',
        'jinja2'
    ]
}

# Add `complete' target, which will install all extras listed above
extras['complete'] = list({pkg for req in extras.values() for pkg in req})

# Add `no_mpi' target, identical to `complete' target but without mpi4py,
# which is unavailable in some common environments.
extras['no_mpi'] = [e for e in extras['complete'] if e != 'mpi4py']


# Configure setuptools_scm to build the post-release version number
def custom_version():
    from setuptools_scm.version import postrelease_version

    return {'version_scheme': postrelease_version}


def setup_with_extensions(extensions=None):
    setup(
        name='pyGSTi',
        use_scm_version=custom_version,
        description='A python implementation of Gate Set Tomography',
        long_description=descriptionTxt,
        author='Erik Nielsen, Kenneth Rudinger, Timothy Proctor, John Gamble, Robin Blume-Kohout',
        author_email='pygsti@sandia.gov',
        packages=[
            'pygsti',
            'pygsti.algorithms',
            'pygsti.construction',
            'pygsti.drivers',
            'pygsti.extras',
            'pygsti.extras.rb',
            'pygsti.extras.rpe',
            'pygsti.extras.drift',
            'pygsti.extras.idletomography',
            'pygsti.extras.crosstalk',
            'pygsti.extras.devices',
            'pygsti.io',
            'pygsti.io.circuitparser',
            'pygsti.modelpacks',
            'pygsti.modelpacks.legacy',
            'pygsti.objects',
            'pygsti.objects.replib',
            'pygsti.objects.opcalc',
            'pygsti.optimize',
            'pygsti.protocols',
            'pygsti.report',
            'pygsti.report.section',
            'pygsti.tools'
        ],
        package_dir={'': '.'},
        package_data={
            'pygsti.tools': ['fastcalc.pyx'],
            'pygsti.objects.replib': [
                'fastreplib.pyx',
                'fastreps.cpp',
                'fastreps.h'
            ],
            'pygsti.objects.opcalc': ['fastopcalc.pyx'],
            'pygsti.io.circuitparser': ['fastcircuitparser.pyx'],
            'pygsti.report': [
                'templates/*.tex',
                'templates/*.html',
                'templates/*.json',
                'templates/*.ipynb',
                'templates/report_notebook/*.txt',
                'templates/standard_html_report/*.html',
                'templates/standard_html_report/tabs/*.html',
                'templates/idletomography_html_report/*.html',
                'templates/idletomography_html_report/tabs/*.html',
                'templates/drift_html_report/*.html',
                'templates/drift_html_report/tabs/*.html',
                'templates/offline/README.txt',
                'templates/offline/*.js',
                'templates/offline/*.css',
                'templates/offline/fonts/*',
                'templates/offline/images/*'
            ]
        },
        setup_requires=['setuptools_scm'],
        install_requires=[
            'numpy>=1.15.0',
            'scipy',
            'plotly',
            'ply'
        ],
        extras_require=extras,
        python_requires='>=3.5',
        platforms=["any"],
        url='http://www.pygsti.info',
        download_url='https://github.com/pyGSTio/pyGSTi/tarball/master',
        keywords=[
            'pygsti',
            'tomography',
            'gate set',
            'pigsty',
            'pig',
            'quantum',
            'qubit'
        ],
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: Apache Software License",
            "Programming Language :: Python",
            "Topic :: Scientific/Engineering :: Physics",
            "Operating System :: Microsoft :: Windows",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Unix"
        ],
        ext_modules=extensions or [],
    )


try:
    # Try to compile extensions first

    import numpy as np
    from Cython.Build import cythonize
    ext_modules = [
        Extension(
            "pygsti.tools.fastcalc",
            sources=["pygsti/tools/fastcalc.pyx"],  # , "fastcalc.c
            # # Cython docs on NumPy usage should mention this!
            # define_macros = [('NPY_NO_DEPRECATED_API','NPY_1_7_API_VERSION')],
            # # leave above commented
            # # see http://docs.cython.org/en/latest/src/reference/compilation.html#configuring-the-c-build
            # define_macros = [('CYTHON_TRACE','1')], #for profiling
            include_dirs=['.', np.get_include()]
            # libraries=['m'] #math lib?
        ),
        Extension(
            "pygsti.objects.opcalc.fastopcalc",
            sources=["pygsti/objects/opcalc/fastopcalc.pyx"],
            include_dirs=['.', np.get_include()],
            language="c++",
            extra_compile_args=["-std=c++11"],  # ,"-stdlib=libc++"
            extra_link_args=["-std=c++11"]
        ),
        Extension(
            "pygsti.objects.replib.fastreplib",
            sources=[
                "pygsti/objects/replib/fastreplib.pyx",
                "pygsti/objects/replib/fastreps.cpp"
            ],
            include_dirs=['.', np.get_include()],
            language="c++",
            extra_compile_args=["-std=c++11"],  # ,"-stdlib=libc++"
            extra_link_args=["-std=c++11"]
        ),
        Extension(
            "pygsti.io.circuitparser.fastcircuitparser",
            sources=["pygsti/io/circuitparser/fastcircuitparser.pyx"],
            include_dirs=['.', np.get_include()],
            language="c++",
            extra_compile_args=["-std=c++11"],  # ,"-stdlib=libc++"
            extra_link_args=["-std=c++11"]
        )
    ]
    setup_with_extensions(cythonize(ext_modules, exclude_failures=True))
except ImportError:
    # Cython or numpy is not available
    warn("Extensions build tools are not available. Installing without Cython extensions...")
    setup_with_extensions()
except SystemExit:
    # Extension compilation failed
    warn("Error in extension compilation. Installing without Cython extensions...")
    setup_with_extensions()
