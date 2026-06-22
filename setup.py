"""A python implementation of Gate Set Tomography"""

from warnings import warn
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
import sys


try:
    from setuptools import setup, find_packages
    from setuptools import Extension
    from setuptools.command.build_ext import build_ext
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension
    from distutils.command.build_ext import build_ext


# Configure setuptools_scm to build a custom version (for more info,
# see https://stackoverflow.com/a/78657279 and https://setuptools-scm.readthedocs.io/en/latest/extending)
# If on a clean release, it uses no local scheme
# Otherwise, it uses g{commit hash}.{branch}.[CLEAN | d{date}] for the local scheme,
# where the last entry is "CLEAN" if everything is committed and otherwise the date of last commit
def custom_version(version):
    from setuptools_scm.version import get_local_node_and_date

    local_scheme = ""
    if version.dirty or version.distance:
        node_and_date = get_local_node_and_date(version)

        if version.dirty:
            node, date = node_and_date.split('.')
        else:
            node = node_and_date
            date = "CLEAN"
        
        local_scheme = node + f'.{version.branch}.' + date

    return local_scheme


#Create a custom command class that allows us to specify different compiler flags
# based on the compiler (~platform) being used (see
# https://stackoverflow.com/questions/30985862/how-to-identify-compiler-before-defining-cython-extensions)
BUILD_ARGS = defaultdict(lambda: ["-std=c++11"])  # ,"-stdlib=libc++", '-O3', '-g0'
for compiler, args in [
        ('msvc', []),
        ('gcc', ["-std=c++11", "-Wno-deprecated"])]:
    BUILD_ARGS[compiler] = args


class our_build_ext(build_ext):
    """Two-phase, race-free parallel build of the Cython/C++ extensions.

    Several extensions share the same hand-written ``.cpp`` sources (e.g.
    ``statecreps.cpp`` is a source of four stabilizer extensions). setuptools'
    built-in ``parallel`` builds *whole extensions* concurrently and names each
    object file purely from its source path, so two extensions sharing a source
    write the *same* ``.o`` at the same time -- a file-collision race that links a
    truncated object and yields ``undefined symbol`` import errors (issue #791).

    Instead we:
      1. compile each distinct *compile job* exactly once, in parallel; then
      2. link each extension sequentially from the already-built objects.

    A "compile job" is a ``(source, compile-settings)`` pair, where the settings
    are the macros, include dirs and compile flags that affect the emitted object.
    Extensions that compile a shared source with **identical** settings share a
    single object file (the common case: compiled once, not once-per-extension).
    Extensions that ask for **different** settings on the same source each get
    their own object, built with exactly their own flags -- so per-extension
    compile flags are always respected, and a single (default) set of flags is
    simply used for every job. Distinct settings are isolated in their own object
    subdirectory so their object files never collide (and thus never race).
    """

    def _ext_compile_settings(self, ext):
        """The object-affecting compile settings for ``ext`` (matches distutils)."""
        macros = ext.define_macros[:]
        for undef in ext.undef_macros:
            macros.append((undef,))
        return {
            'macros': macros,
            'include_dirs': list(ext.include_dirs or []),
            'extra_postargs': list(ext.extra_compile_args or []),
        }

    def _plan_compiles(self):
        """Group compiles by (settings, source) and assign each group an output dir.

        Returns ``(jobs, ext_output_dir)`` where ``jobs`` maps each unique
        ``(output_dir, source)`` to its compile parameters, and ``ext_output_dir``
        maps each extension to the directory holding its objects. Extensions whose
        settings are byte-for-byte identical map to the same output dir, so a
        source they share is compiled exactly once; extensions with differing
        settings get distinct dirs, so the same source can be compiled separately
        with each extension's own flags without the two objects colliding.
        """
        jobs = {}
        ext_output_dir = {}
        sig_to_dir = {}
        for ext in self.extensions:
            settings = self._ext_compile_settings(ext)
            # A stable key for the settings; identical settings -> identical key.
            sig = repr((settings['macros'], settings['include_dirs'], settings['extra_postargs']))
            if sig not in sig_to_dir:
                if len(sig_to_dir) == 0:
                    # First/most-common settings live directly in build_temp so the
                    # usual single-flag-set build keeps the conventional layout.
                    sig_to_dir[sig] = self.build_temp
                else:
                    digest = hashlib.sha1(sig.encode('utf-8')).hexdigest()[:12]
                    sig_to_dir[sig] = os.path.join(self.build_temp, 'flags-' + digest)
            output_dir = sig_to_dir[sig]
            ext_output_dir[id(ext)] = output_dir
            for src in ext.sources:
                key = (output_dir, src)
                if key not in jobs:
                    jobs[key] = {
                        'src': src,
                        'output_dir': output_dir,
                        'macros': list(settings['macros']),
                        'include_dirs': list(settings['include_dirs']),
                        'extra_postargs': list(settings['extra_postargs']),
                        'depends': list(ext.depends or []),
                    }
                else:
                    # Same source + same settings -> same object. Settings already
                    # match (they define the key); only union the rebuild triggers.
                    for d in (ext.depends or []):
                        if d not in jobs[key]['depends']:
                            jobs[key]['depends'].append(d)
        return jobs, ext_output_dir

    def build_extensions(self):
        compiler = self.compiler.compiler_type
        args = BUILD_ARGS[compiler]
        print("\n\nCompiler: ", compiler, "\n")
        for ext in self.extensions:
            if ext.language == "c++":  # only do this for c++ files, so we can specify -std=c++11, etc.
                ext.extra_compile_args = args

        jobs, ext_output_dir = self._plan_compiles()

        # Pre-create every object output directory up front; distutils' mkpath is
        # not safe to call concurrently from the compile workers below.
        for job in jobs.values():
            obj = self.compiler.object_filenames([job['src']], output_dir=job['output_dir'])[0]
            os.makedirs(os.path.dirname(obj), exist_ok=True)

        # ---- Phase 1: compile each distinct (source, settings) job once ----
        def _compile_one(job):
            self.compiler.compile(
                [job['src']],
                output_dir=job['output_dir'],
                macros=job['macros'],
                include_dirs=job['include_dirs'],
                debug=self.debug,
                extra_postargs=job['extra_postargs'],
                depends=job['depends'],
            )

        # Keep Windows serial: concurrent cl.exe invocations have their own
        # documented collision modes (e.g. shared PDBs). The compile-once dedup
        # still makes the serial Windows build faster than before.
        if sys.platform == "win32":
            max_workers = 1
        else:
            max_workers = int(os.environ.get("BUILD_JOBS", os.cpu_count() or 1))

        job_list = list(jobs.values())
        if max_workers > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                # list() forces iteration so exceptions from workers propagate.
                list(pool.map(_compile_one, job_list))
        else:
            for job in job_list:
                _compile_one(job)

        # ---- Phase 2: link each extension sequentially from prebuilt objects ----
        for ext in self.extensions:
            output_dir = ext_output_dir[id(ext)]
            ext_objects = self.compiler.object_filenames(ext.sources, output_dir=output_dir)
            if ext.extra_objects:
                ext_objects = ext_objects + list(ext.extra_objects)
            ext_path = self.get_ext_fullpath(ext.name)
            os.makedirs(os.path.dirname(ext_path), exist_ok=True)
            language = ext.language or self.compiler.detect_language(ext.sources)
            self.compiler.link_shared_object(
                ext_objects,
                ext_path,
                libraries=self.get_libraries(ext),
                library_dirs=ext.library_dirs,
                runtime_library_dirs=ext.runtime_library_dirs,
                extra_postargs=ext.extra_link_args or [],
                export_symbols=self.get_export_symbols(ext),
                debug=self.debug,
                build_temp=self.build_temp,
                target_lang=language,
            )


# Check if environment can try to build extensions
try:
    import numpy as np
    from Cython.Build import cythonize

    if "PYGSTI_SKIP_CYTHON" in os.environ:
        warn("PYGSTI_SKIP_CYTHON env variable defined. Installing without Cython extensions...")
        extensions = None
    else:
        common_evotypes_kwargs = dict(
            include_dirs=['.', 'pygsti/evotypes', np.get_include()],
            language="c++",
            extra_link_args=["-std=c++11"],
            extra_compile_args=['-O3']
        )
        ext_modules = [
            Extension(
                "pygsti.tools.fastcalc",
                sources=["pygsti/tools/fastcalc.pyx"],  # , "fastcalc.c
                # # Cython docs on NumPy usage should mention this!
                # define_macros = [('NPY_NO_DEPRECATED_API','NPY_1_7_API_VERSION')],
                # # leave above commented
                # # see http://docs.cython.org/en/latest/src/reference/compilation.html#configuring-the-c-build
                # define_macros = [('CYTHON_TRACE','1')], #for profiling
                include_dirs=['.', np.get_include()],
                # libraries=['m'] #math lib?
            ),
            Extension(
                "pygsti.tools.fasterrgencalc",
                sources=["pygsti/tools/fasterrgencalc.pyx"],  # , "fasterrgencalc.c
                # # Cython docs on NumPy usage should mention this!
                # define_macros = [('NPY_NO_DEPRECATED_API','NPY_1_7_API_VERSION')],
                # # leave above commented
                # # see http://docs.cython.org/en/latest/src/reference/compilation.html#configuring-the-c-build
                # define_macros = [('CYTHON_TRACE','1')], #for profiling
                include_dirs=['.', np.get_include()],
                extra_compile_args=['-O3']
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
                **common_evotypes_kwargs
            ),
            Extension(
                "pygsti.evotypes.densitymx.opreps",
                sources=[
                    "pygsti/evotypes/densitymx/opreps.pyx",
                    "pygsti/evotypes/densitymx/opcreps.cpp",
                    "pygsti/evotypes/densitymx/statecreps.cpp"
                ],
                **common_evotypes_kwargs
            ),
            Extension(
                "pygsti.evotypes.densitymx.effectreps",
                sources=[
                    "pygsti/evotypes/densitymx/effectreps.pyx",
                    "pygsti/evotypes/densitymx/effectcreps.cpp",
                    "pygsti/evotypes/densitymx/statecreps.cpp",
                    "pygsti/evotypes/densitymx/opcreps.cpp"
                ],
                **common_evotypes_kwargs
            ),
            Extension(
                "pygsti.evotypes.statevec.statereps",
                sources=[
                    "pygsti/evotypes/statevec/statereps.pyx",
                    "pygsti/evotypes/statevec/statecreps.cpp"
                ],
                **common_evotypes_kwargs
            ),
            Extension(
                "pygsti.evotypes.statevec.opreps",
                sources=[
                    "pygsti/evotypes/statevec/opreps.pyx",
                    "pygsti/evotypes/statevec/opcreps.cpp",
                    "pygsti/evotypes/statevec/statecreps.cpp"
                ],
                **common_evotypes_kwargs
            ),
            Extension(
                "pygsti.evotypes.statevec.effectreps",
                sources=[
                    "pygsti/evotypes/statevec/effectreps.pyx",
                    "pygsti/evotypes/statevec/effectcreps.cpp",
                    "pygsti/evotypes/statevec/statecreps.cpp",
                    "pygsti/evotypes/statevec/opcreps.cpp"
                ],
                **common_evotypes_kwargs
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
                **common_evotypes_kwargs
            ),
            Extension(
                "pygsti.evotypes.stabilizer.statereps",
                sources=[
                    "pygsti/evotypes/stabilizer/statereps.pyx",
                    "pygsti/evotypes/stabilizer/statecreps.cpp"
                ],
                **common_evotypes_kwargs
            ),
            Extension(
                "pygsti.evotypes.stabilizer.opreps",
                sources=[
                    "pygsti/evotypes/stabilizer/opreps.pyx",
                    "pygsti/evotypes/stabilizer/opcreps.cpp",
                    "pygsti/evotypes/stabilizer/statecreps.cpp"
                ],
                **common_evotypes_kwargs
            ),
            Extension(
                "pygsti.evotypes.stabilizer.effectreps",
                sources=[
                    "pygsti/evotypes/stabilizer/effectreps.pyx",
                    "pygsti/evotypes/stabilizer/effectcreps.cpp",
                    "pygsti/evotypes/stabilizer/statecreps.cpp",
                    "pygsti/evotypes/stabilizer/opcreps.cpp"
                ],
                **common_evotypes_kwargs
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
                **common_evotypes_kwargs
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
        extensions = cythonize(ext_modules, compiler_directives={'language_level': "3"}, exclude_failures="PYGSTI_CYTHON_EXCLUDE_FAILURES" in os.environ)
except ImportError:
    warn("Extensions build tools are not available. Installing without Cython extensions...")
    extensions = None

try:
    setup(
        use_scm_version={'version_scheme': 'no-guess-dev', 'version_file': "pygsti/_version.py", 'local_scheme': custom_version},
        cmdclass={'build_ext': our_build_ext},
        ext_modules=extensions,
        packages=find_packages(where='.', include=['pygsti']),
        package_data={
            'pygsti.tools': ['fastcalc.pyx', 'fasterrgencalc.pyx'],
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
except SystemExit as e:
    print("\nAn error occurred while compiling Cython extensions.")
    print("Either fix the compilation issue or use the PYGSTI_CYTHON_SKIP to skip compilation,",)
    print('e.g. PYGSTI_CYTHON_SKIP=1 pip install pygsti')
    print("To enable partial Cython failures (i.e. the exclude_failures=True flag of cythonize), use PYGSTI_CYTHON_EXCLUDE_FAILURES instead.\n")
    raise e
