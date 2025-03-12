"""A python implementation of Gate Set Tomography"""

from warnings import warn
from collections import defaultdict

try:
    from setuptools import setup, find_packages
    from setuptools import Extension
    from setuptools.command.build_ext import build_ext
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension
    from distutils.command.build_ext import build_ext


    
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
    'pytorch' : ['torch'],
    'diamond_norm': [
        'cvxopt',
        'cvxpy'
    ],
    'memory_profiling': ['psutil'],
    'multiprocessor': ['mpi4py'],
    'evolutionary_optimization': ['deap'],
    'report_pickling': ['pandas'],
    'report_pdf_figures': ['matplotlib'],
    'html_reports': ['jinja2', 'MarkupSafe'],
    'notebooks': [
        'ipython',
        'notebook',
        'jupyter_server'
    ],
    'mongodb': ['pymongo'],
    'msgpack': ['msgpack'],
    'extensions': ['cython'],
    'linting': [
        'autopep8',
        'flake8'
    ],
    'interpygate': ['csaps'],
    'testing': [
        'pytest',
        'pytest-xdist',
        'pytest-cov',
        'nbval',
        'csaps',
        'cvxopt',
        'cvxpy',
        'cython',
        'matplotlib',
        'mpi4py',
        'msgpack',
        'packaging',
        'pandas',
        'psutil',
        'zmq',
        'jinja2',
        'seaborn',
        'scipy',
        'ply',
        'cirq-core',
        'notebook',
        'ipython',
        'jupyter_server',
        'torch'
    ]
}

# Add `complete' target, which will install all extras listed above
extras['complete'] = list({pkg for req in extras.values() for pkg in req})

# Add `no_mpi' target, identical to `complete' target but without mpi4py,
# which is unavailable in some common environments.
extras['no_mpi'] = [e for e in extras['complete'] if e != 'mpi4py']

# Add testing_no_cython target, identical to `testing` but no cython
extras['testing_no_cython'] = [e for e in extras['testing'] if e != 'cython']


# Configure setuptools_scm to build a custom version (for more info,
# see https://stackoverflow.com/a/78657279 and https://setuptools-scm.readthedocs.io/en/latest/extending)
# If on a clean release, it uses no local scheme
# If not on master, appends branch to the local scheme
# If not clean/, uses default local scheme (node-and-date)
def custom_version(version):
    from setuptools_scm.version import get_local_node_and_date

    b = version.branch if version.branch and version.branch != "master" else None

    local_scheme = "no-local-version"
    if version.dirty or version.distance:
        local_scheme = get_local_node_and_date(version) + (f'.{b}' if b else '')
    elif b:
        local_scheme = f"+{b}"

    return local_scheme


#Create a custom command class that allows us to specify different compiler flags
# based on the compiler (~platform) being used (see
# https://stackoverflow.com/questions/30985862/how-to-identify-compiler-before-defining-cython-extensions)
BUILD_ARGS = defaultdict(lambda: ["-std=c++11"])  # ,"-stdlib=libc++", '-O3', '-g0'
for compiler, args in [
        ('msvc', []),
        ('gcc', ["-std=c++11", "-Wno-deprecated"])]:
    BUILD_ARGS[compiler] = args


class build_ext_compiler_check(build_ext):
    def build_extensions(self):
        compiler = self.compiler.compiler_type
        args = BUILD_ARGS[compiler]
        print("\n\nCompiler: ",compiler,"\n")
        for ext in self.extensions:
            if ext.language == "c++":  # only do this for c++ files, so we can specify -std=c++11, etc.
                ext.extra_compile_args = args
        build_ext.build_extensions(self)


def setup_with_extensions(extensions=None):
    setup(
        long_description=descriptionTxt,
        use_scm_version={'version_scheme': 'no-guess-dev', 'version_file': "pygsti/_version.py", 'local_scheme': custom_version},
        cmdclass={'build_ext': build_ext_compiler_check},
        extras_require=extras,
        ext_modules=extensions or [],
        packages=find_packages(),
        package_data={
            'pygsti.tools': ['fastcalc.pyx'],
            'pygsti.evotypes': [
                'basereps_cython.pxd',
                'basereps_cython.pyx',
                'basecreps.cpp',
                'basecreps.h'],
            'pygsti.evotypes.densitymx': [
                'opreps.pxd',
                'opreps.pyx',
                'opcreps.cpp',
                'opcreps.h',
                'statereps.pxd',
                'statereps.pyx',
                'statecreps.cpp',
                'statecreps.h',
                'effectreps.pxd',
                'effectreps.pyx',
                'effectcreps.cpp',
                'effectcreps.h'
            ],
            'pygsti.evotypes.statevec': [
                'opreps.pxd',
                'opreps.pyx',
                'opcreps.cpp',
                'opcreps.h',
                'statereps.pxd',
                'statereps.pyx',
                'statecreps.cpp',
                'statecreps.h',
                'effectreps.pxd',
                'effectreps.pyx',
                'effectcreps.cpp',
                'effectcreps.h',
                'termreps.pxd',
                'termreps.pyx',
                'termcreps.cpp',
                'termcreps.h',
            ],
            'pygsti.evotypes.stabilizer': [
                'opreps.pxd',
                'opreps.pyx',
                'opcreps.cpp',
                'opcreps.h',
                'statereps.pxd',
                'statereps.pyx',
                'statecreps.cpp',
                'statecreps.h',
                'effectreps.pxd',
                'effectreps.pyx',
                'effectcreps.cpp',
                'effectcreps.h',
                'termreps.pxd',
                'termreps.pyx',
                'termcreps.cpp',
                'termcreps.h',
            ],
            'pygsti.forwardsims': [
                'mapforwardsim_calc_densitymx.pyx',
                'termforwardsim_calc_statevec.pyx',
                'termforwardsim_calc_stabilizer.pyx'
            ],
            'pygsti.baseobjs.opcalc': ['fastopcalc.pyx'],
            'pygsti.circuits.circuitparser': ['fastcircuitparser.pyx'],
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
        }
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
            "pygsti.baseobjs.opcalc.fastopcalc",
            sources=["pygsti/baseobjs/opcalc/fastopcalc.pyx"],
            include_dirs=['.', np.get_include()],
            language="c++",
            extra_link_args=["-std=c++11"]
        ),
        Extension(
            "pygsti.evotypes.basereps_cython",
            sources=[
                "pygsti/evotypes/basereps_cython.pyx",
                "pygsti/evotypes/basecreps.cpp"
            ],
            include_dirs=['.', np.get_include()],
            language="c++",
            extra_link_args=["-std=c++11"]
        ),
        Extension(
            "pygsti.evotypes.densitymx.statereps",
            sources=[
                "pygsti/evotypes/densitymx/statereps.pyx",
                "pygsti/evotypes/densitymx/statecreps.cpp"
            ],
            include_dirs=['.', 'pygsti/evotypes', np.get_include()],
            language="c++",
            extra_link_args=["-std=c++11"]
        ),
        Extension(
            "pygsti.evotypes.densitymx.opreps",
            sources=[
                "pygsti/evotypes/densitymx/opreps.pyx",
                "pygsti/evotypes/densitymx/opcreps.cpp",
                "pygsti/evotypes/densitymx/statecreps.cpp"
            ],
            include_dirs=['.', 'pygsti/evotypes', np.get_include()],
            language="c++",
            extra_link_args=["-std=c++11"]
        ),
        Extension(
            "pygsti.evotypes.densitymx.effectreps",
            sources=[
                "pygsti/evotypes/densitymx/effectreps.pyx",
                "pygsti/evotypes/densitymx/effectcreps.cpp",
                "pygsti/evotypes/densitymx/statecreps.cpp",
                "pygsti/evotypes/densitymx/opcreps.cpp"
            ],
            include_dirs=['.', 'pygsti/evotypes', np.get_include()],
            language="c++",
            extra_link_args=["-std=c++11"]
        ),
        Extension(
            "pygsti.evotypes.statevec.statereps",
            sources=[
                "pygsti/evotypes/statevec/statereps.pyx",
                "pygsti/evotypes/statevec/statecreps.cpp"
            ],
            include_dirs=['.', 'pygsti/evotypes', np.get_include()],
            language="c++",
            extra_link_args=["-std=c++11"]
        ),
        Extension(
            "pygsti.evotypes.statevec.opreps",
            sources=[
                "pygsti/evotypes/statevec/opreps.pyx",
                "pygsti/evotypes/statevec/opcreps.cpp",
                "pygsti/evotypes/statevec/statecreps.cpp"
            ],
            include_dirs=['.', 'pygsti/evotypes', np.get_include()],
            language="c++",
            extra_link_args=["-std=c++11"]
        ),
        Extension(
            "pygsti.evotypes.statevec.effectreps",
            sources=[
                "pygsti/evotypes/statevec/effectreps.pyx",
                "pygsti/evotypes/statevec/effectcreps.cpp",
                "pygsti/evotypes/statevec/statecreps.cpp",
                "pygsti/evotypes/statevec/opcreps.cpp"
            ],
            include_dirs=['.', 'pygsti/evotypes', np.get_include()],
            language="c++",
            extra_link_args=["-std=c++11"]
        ),
        Extension(
            "pygsti.evotypes.statevec.termreps",
            sources=[
                "pygsti/evotypes/statevec/termreps.pyx",
                "pygsti/evotypes/statevec/termcreps.cpp",
                "pygsti/evotypes/statevec/statecreps.cpp",
                "pygsti/evotypes/statevec/opcreps.cpp",
                "pygsti/evotypes/statevec/effectcreps.cpp"
            ],
            include_dirs=['.', 'pygsti/evotypes', np.get_include()],
            language="c++",
            extra_link_args=["-std=c++11"]
        ),
        Extension(
            "pygsti.evotypes.stabilizer.statereps",
            sources=[
                "pygsti/evotypes/stabilizer/statereps.pyx",
                "pygsti/evotypes/stabilizer/statecreps.cpp"
            ],
            include_dirs=['.', 'pygsti/evotypes', np.get_include()],
            language="c++",
            extra_link_args=["-std=c++11"]
        ),
        Extension(
            "pygsti.evotypes.stabilizer.opreps",
            sources=[
                "pygsti/evotypes/stabilizer/opreps.pyx",
                "pygsti/evotypes/stabilizer/opcreps.cpp",
                "pygsti/evotypes/stabilizer/statecreps.cpp"
            ],
            include_dirs=['.', 'pygsti/evotypes', np.get_include()],
            language="c++",
            extra_link_args=["-std=c++11"]
        ),
        Extension(
            "pygsti.evotypes.stabilizer.effectreps",
            sources=[
                "pygsti/evotypes/stabilizer/effectreps.pyx",
                "pygsti/evotypes/stabilizer/effectcreps.cpp",
                "pygsti/evotypes/stabilizer/statecreps.cpp",
                "pygsti/evotypes/stabilizer/opcreps.cpp"
            ],
            include_dirs=['.', 'pygsti/evotypes', np.get_include()],
            language="c++",
            extra_link_args=["-std=c++11"]
        ),
        Extension(
            "pygsti.evotypes.stabilizer.termreps",
            sources=[
                "pygsti/evotypes/stabilizer/termreps.pyx",
                "pygsti/evotypes/stabilizer/termcreps.cpp",
                "pygsti/evotypes/stabilizer/statecreps.cpp",
                "pygsti/evotypes/stabilizer/opcreps.cpp",
                "pygsti/evotypes/stabilizer/effectcreps.cpp"
            ],
            include_dirs=['.', 'pygsti/evotypes', np.get_include()],
            language="c++",
            extra_link_args=["-std=c++11"]
        ),
        Extension(
            "pygsti.forwardsims.mapforwardsim_calc_densitymx",
            sources=[
                "pygsti/forwardsims/mapforwardsim_calc_densitymx.pyx",
                "pygsti/evotypes/densitymx/statecreps.cpp",
            ],
            include_dirs=['.', 'pygsti/evotypes', 'pygsti/evotypes/densitymx', np.get_include()],
            language="c++",
            extra_link_args=["-std=c++11"]
        ),
        Extension(
            "pygsti.forwardsims.termforwardsim_calc_statevec",
            sources=[
                "pygsti/forwardsims/termforwardsim_calc_statevec.pyx",
                "pygsti/evotypes/statevec/statecreps.cpp",
                "pygsti/evotypes/basecreps.cpp"
            ],
            include_dirs=['.', 'pygsti/evotypes', 'pygsti/evotypes/statevec', np.get_include()],
            language="c++",
            extra_link_args=["-std=c++11"]
        ),
        Extension(
            "pygsti.forwardsims.termforwardsim_calc_stabilizer",
            sources=[
                "pygsti/forwardsims/termforwardsim_calc_stabilizer.pyx",
                "pygsti/evotypes/stabilizer/statecreps.cpp",
                "pygsti/evotypes/basecreps.cpp"
            ],
            include_dirs=['.', 'pygsti/evotypes', 'pygsti/evotypes/stabilizer', np.get_include()],
            language="c++",
            extra_link_args=["-std=c++11"]
        ),
        Extension(
            "pygsti.circuits.circuitparser.fastcircuitparser",
            sources=["pygsti/circuits/circuitparser/fastcircuitparser.pyx"],
            include_dirs=['.', np.get_include()],
            language="c++",
            extra_link_args=["-std=c++11"]
        )
    ]
    setup_with_extensions(cythonize(ext_modules, compiler_directives={'language_level': "3"}, exclude_failures=True))
except ImportError:
    # Cython or numpy is not available
    warn("Extensions build tools are not available. Installing without Cython extensions...")
    setup_with_extensions()
except SystemExit:
    # Extension compilation failed
    warn("Error in extension compilation. Installing without Cython extensions...")
    setup_with_extensions()
