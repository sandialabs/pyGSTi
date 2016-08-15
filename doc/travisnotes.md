# Travis CI Notes - Lucas

### Overview:

  - Travis CI clones pygsti, and executes the `test/travisTests.py` script in different environments
  - A build matrix is created, testing almost everything it can in the least amount of time
  - If the tests pass on develop, an automatic push to the beta branch is made. This seems to be buggy.
  - On beta, coverage info is generated and should be visible in the job log
  - When a build fails, the author of the latest commit to that branch is notified by github-registered e-mail. There is also custom configuration in `.travis.yml`


### The build matrix:

Currently, our tests run both python 2.7 and 3.5:

  - Default: test `objects`, `tools`, `io`, `construction`, `optimize`, and `algorithms` packages.
  - ReportA: run `test_reports_logL_TP_wCIs` only (77% coverage of `report` package in ~7 minutes)
  - ReportB: run all report tests except for `testReport.py`. This includes formatter, table, analysis, metrics, and plotting tests.
  - Drivers: run only the `drivers` test package

This results in a 2x4 matrix, which often finishes completely in about thirty-forty minutes.

So, this means that in the test `testReport.py`, only `test_reports_logL_TP_wCIs` is run, and *none of the other* tests in `testReport.py`.

Other than that, only the `mpi` tests aren't being run:

Recently, I removed the `mpi` tests (they take too long), but they can be added back in by modifying `.travis.yml`'s env field:

```yaml
env:
  - Default=True
  - ReportA=True
  ...
  - MPI=True # Add this if mpi tests are modified to take less than forty minutes
```

### Optimization

  - There is a script `CI/install.sh`, which only downloads packages required for the current test environment
    (For example, `texlive-full` is only installed for `ReportA` and `drivers` environments, since they need pdflatex)

  - Additionally, all pip installs are cached, generally making them run in under a few seconds. However, the cache size is enormous, at around 2Gb
    (The most recent beta build installed all pip dependencies in under 10 seconds - caching is worth keeping :) )

  - By default, all tests besides mpi tests run in parallel through the `nose` `--processes=-1` (all cores) flag.

  - Coverage is only turned on when testing on `beta`

### Automatic pushing:

This is definitely an experimental feature. <s> It hardly works. </s>

Ideally, when a build is *completely* successful, develop should be merged into beta.

However, if all of a build's jobs pass, except for an *errored* job (*not* failed), the deploy hook will still trigger.  
(Errored jobs are marked with an exclamation `!`, and failed jobs are marked with an `x`)

Errored jobs occur when, for example, a download doesn't work

Failed jobs occur when one of our tests exits with non-zero status

This is just what I've noticed, so I'm not sure that it's entirely correct. The travis CI documentation describes the deploy step:

> The following example runs scripts/deploy.sh on the develop branch of your repository if the *build* is successful.

Which is distinguished from the `after_success` step, which runs if the *job* is successful.

That being pointed out, I'm fairly sure that deployment wont happen if the build is a failure (our fault), meaning the deployment step *sortof* works


