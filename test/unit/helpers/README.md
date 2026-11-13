# Tests of test-support tooling

This directory tests the machinery that other tests use to construct experiments, run them, and judge their results. Its subject is test-support code, primarily the modules under `test/helpers/`, rather than pyGSTi's production APIs.

Examples include:

- Checking that experiment builders create the intended models, noise injections, training circuits and held-out circuits.
- Checking statistical reference calculations against independent numerical calculations, so acceptance criteria rest on trustworthy results.
- Checking runner configuration, artifact paths, persistence delegation and subprocess import environments.

Classify a test by the behavior its assertions check. Using production models, circuits or datasets as inputs does not make a test a production-API test when the assertions concern a helper's construction or calculations. Conversely, a test that directly checks an existing production API belongs with that API's tests, even if its setup previously used a helper.

Keep production protocol tests under `test/unit/protocols/` and end-to-end scientific recovery/detection tests under `test/integration/`. For example, a direct `simulate_data` sampling-contract test belongs with the dataset-construction tests under `test/unit/construction/`, whereas a test of the harness's predictive-error reference belongs here.

Keep reusable support implementations under `test/helpers/`; this directory contains their tests and local test setup. Avoid importing collected test modules as shared libraries. Prefer small, focused checks and independent references where a defect could silently invalidate other tests. Full GST fits belong in the integration tests rather than being repeated to check runner plumbing.

These tests are discovered by the existing `test/unit` CI invocation. The directory does not require a separate test job or a special marker merely because it tests test-support code.
