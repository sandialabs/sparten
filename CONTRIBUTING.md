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
 
- [ ] **Submit an issue.**
  Use a descriptive title `<title>` and provide as much detailed information as possible in the comments. For bug reports, please provide a minimal working example to reproduce the problem. The issue will be numbered `<num>`.
 
- [ ] **Create a branch and make your changes.**
  Name the branch `<num>-<title>`. This is done automatically when using the web interface to Gitlab.
 
- [ ] **Create a merge request.**
  This can be done at the time the branch is created, which is preferred.
 
- [ ] **Unit Tests.**
  Create or update tests in the `unit_test` directory, especially for
  new code.
 
- [ ] **Regression Tests.**
  Create or update tests in the `regression_test` directory, especially for
  bugs/issues.
 
- [ ] **Pass All Tests.**
  - [ ] Confirm that all unit tests pass: `./bin/Sparten_unit_test`.
  - [ ] Confirm that all regression tests pass: `./bin/Sparten_regression_test`.
 
- [ ] **Documentation.**
  A sufficient documentation in Doxygen-style comments.
 
- [ ] **Release Notes**
  Update `CHANGELOG` with any significant bug fixes or additions.
 
- [ ] **Contributors List.**
  Update `CONTRIBUTORS.md` with your name and a brief description of
  the contributions.
 
 
## Review
- [ ] **Mark merge request as _ready_.** Iterate further if necessary.

- [ ] **Final step: merge branch**
