# Travis CI Notes - Rob

(updated and continued from Lucas's previous notes)

### Overview:

When new history is pushed to `pyGSTio/pyGSTi`, Travis CI begins a new
build in a Xenial-based environment. The configuration of this build
depends on the branch, but consists of four sequential phases:

1. [**Linting**](#linting),
2. [**Unit testing**](#unit-testing),
3. [**Extra tests**](#extra-tests) (on beta, master, and tags), and
4. [**Deployment**](#deployment) (on develop and tags)

Phases are run sequentially, but jobs within a phase are run
concurrently. Once all jobs in a phase have finished _without error_,
the next phase begins. Should a job end with an error, it _fails_;
subsequent phases will be cancelled and the build is marked as
_failing._

### Linting

[`flake8`][flake8] is used for static linting. We use two strategies
for linting:

- _General linting_ checks for all linting errors we care about. It
  uses the configuration defined in [`.flake8`][.flake8]
- _Critical linting_ checks only for major errors, including syntax
  errors, runtime errors, and undefined names. It uses the
  configuration defined in [`.flake8-critical`][.flake8-critical]

Critical linting is less strict, in that it only checks for a strict
subset of the errors checked by general linting.

The behavior of a build's **Linting** phase differs by branch:

- On `beta`, `master`, and for tags, only _general linting_ is
  run. Any errors will cause the build to fail.
- On all other branches, linting is split into two jobs. _Critical
  linting_ errors will cause the build to fail. _General linting_ is
  also run and will display errors on the build page, but will not
  cause the build to fail.

The net result is that developers can make a pull request with
less-serious style errors and merge into `develop` without issue, but
releases should be stylistically correct. Implicitly, that means a
maintainer (e.g. me or possibly even you) is needed to fix up style
errors before a build can be released. Furthermore, developers then
implicitly accept that a maintainer may need to re-format their code
in ways they might not like.

### Unit testing

Unit tests are included in [`test/unit`][unit-tests]. The
**Unit testing** build phase is run the same regardless of the branch,
and uses Travis CI's build matrix expansion. The unit test suite is
run with [`nose`][nose] using our [configuration](#nose-configuration)
_concurrently_ for each supported python version.

As of writing, we support python versions:

- 3.5
- 3.6
- 3.7
- 3.8

### Extra tests

In addition to [unit tests](#unit-testing), a number of additional
tests are included in [`test/test_packages`][test-packages]. These
additional tests cover various functionality not covered by unit
tests, including file I/O and multiprocessor tests. They also
typically take longer to run.

The **Extra tests** build phase is run for the `beta` and `master`
branches, as well as tags. Like [unit tests](#unit-testing), these
tests are run with [`nose`][nose], but only for the earliest supported
python version (as of writing, this is 3.5).

Because Travis CI jobs time out after 50 minutes, these tests are
split into several different jobs. The exact split is subject to
change.

### Deployment

At the end of a successful build, **Deployment** tasks are run. On
`develop`, the branch is pushed to `beta`, triggering a build for that
branch. On tags, the tag is packaged and deployed to [pyPI][pypi].

#### `develop`

When a build on `develop` completes without error, the branch is
automatically pushed to `beta` using the script in
[`CI/push.sh`][push.sh]. Specifically, the `beta` branch is applied on
top of the new history in `develop` and pushed to origin; if `beta`
can't be automatically fast-forwarded, the job fails, and the issue
must be manually resolved. There are only a few circumstances in which
this merge can fail:

- Something was (manually) pushed to `beta`. Don't do this. If it
  happens, manually merge `beta` into `develop` and push both.
- History on `develop` was changed since the last automatic push to
  `beta`. Don't do this either. If it happens, manually merge
  `develop` into `beta` and push both.

The push to github is authenticated using the encrypted private key at
[CI/github_deploy_key.enc][github_deploy_key.enc]. This key is
decrypted in the push stage's `before_install` script using secure
environment variables defined in the Travis CI repo settings. The
merge is performed by an automatic "Travis CI" user, but apparently,
because I made the key, Travis will show the builds as being triggered
by me. Don't worry about it.


#### tags

When a build for a tag completes without error, the build is packaged
and deployed to [PyPI][pypi]. This uses Travis's built-in `deploy`
directive, but only after running a final job to build Cython extensions.

As of writing, only a source distribution is published to PyPI. Wheels
can be published manually if desired. It would be nice if we could
publish wheels from our CI builds, but unfortunately for us, PyPI has
[strict requirements][manylinux] for wheels built in Linux
environments. Workarounds may be implemented in the future.

Automatic deployments to PyPI are made using an automatic "pygsti-ci"
user. The password for this user is defined as an encrypted variable
in [`.travis.yml`][.travis.yml].

### Addendum

#### Nose configuration

We use a number of plugins and configuration options for
[`nose`][nose]. These are applied as environment variables in our
[`.travis.yml`][.travis.yml] configuration, but may be manually set by
developers via command-line arguments or under the `[nosetests]`
heading in `setup.cfg`

- IDs (`--with-id`, `NOSE_WITH_ID=1`): gives a persistent ID number to
  each test. These numbers are stored in `.noseids`, which is
  gitignored, so IDs are local to each developer (but are still useful
  for personal reference).
- Timer (`--with-timer`, `NOSE_WITH_TIMER=1`): shows a summary of the
  time taken by each individual test at the end of a run.
- Coverage (`--with-coverage`, `NOSE_WITH_COVERAGE=1`): Shows a
  coverage report after running tests. Currently, nothing is done with
  this coverage report. You can also configure `nose` to generate an
  HTML coverage report, which is useful.
- Rednose (`--rednose`, `NOSE_REDNOSE=1`): Adds color and readability
  to test output. Currently has spotty support in the Travis build
  logs, for unknown reasons.
- Verbosity 2 (`-v`, `NOSE_VERBOSE=2`): The default nose output is
  much less useful. However, builds can get pretty big, so consider
  disabling this in the future.
- Multiprocess (`--processes=-1`, `NOSE_PROCESSES=-1`): Run tests
  concurrently. Faster builds, but may potentially cause issues.


[.flake8]: https://github.com/pyGSTio/pyGSTi/blob/master/.flake8
[.flake8-critical]: https://github.com/pyGSTio/pyGSTi/blob/master/.flake8-critical
[.travis.yml]: https://github.com/pyGSTio/pyGSTi/blob/master/.travis.yml
[unit-tests]: https://github.com/pyGSTio/pyGSTi/tree/develop/test/unit
[test-packages]: https://github.com/pyGSTio/pyGSTi/tree/develop/test/test_packages
[push.sh]: https://github.com/pyGSTio/pyGSTi/blob/develop/CI/push.sh
[github_deploy_key.enc]: https://github.com/pyGSTio/pyGSTi/blob/develop/CI/github_deploy_key.enc

[flake8]: http://flake8.pycqa.org/en/latest/
[nose]: https://nose.readthedocs.io/en/latest/
[pypi]: https://pypi.org/project/pyGSTi/
[manylinux]: https://github.com/pypa/manylinux
