```
Sandia National Laboratories is a multimission laboratory managed
and operated by National Technology and Engineering Solutions of Sandia,
LLC, a wholly owned subsidiary of Honeywell International, Inc., for the
U.S. Department of Energy's National Nuclear Security Administration under
contract DE-NA0003525.

Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
(NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
Government retains certain rights in this software.
```

# SparTen Contribution Guide

## Checklist

- [ ] **Issue**
  Before the merge request, submit an issue for the change, providing
  as much detailed information as possible. For bug reports, please
  provide enough information to reproduce the problem.

- [ ] **Fork**
  Create a branch or fork of the code and make your changes.

- [ ] **Documentation**
  For any changes, add sufficient documentation in Doxygen-style comments.

- [ ] **Unit Tests**
  Create or update tests in the `unit_test` directory, especially for 
  new code.

- [ ] **Regression Tests**
  Create or update tests in the `regrression_test` directory, especially for 
  bugs/issues.

- [ ] **Release Notes**
  Update `CHANGELOG` with any significant bug fixes or
  additions.

- [ ] **Contributors List**
  Update `CONTRIBUTORS.md` with your name and a brief description of
  the contributions.

- [ ] **Pass All Tests**
  - [ ] Confirm that all unit tests pass: `./bin/Sparten_unit_test`.
  - [ ] Confirm that all regression tests pass: `./bin/Sparten_regression_test`.


- [ ] **Merge Request**
  At any point, create a work-in-progress merge request, referencing
  the issue number and with this checklist and ```WIP``` in the header.


