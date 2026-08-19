## PR Description

Please describe the changes proposed by this PR, and reference any issues
that it addresses:

## Checklist

### General
1. [ ] I have made the PR off a new branch from my fork 
   (`<your_username>`:`<feature-branch_name>`), *not* 
   `<your_username>`:`master`.
2. [ ] I have added my changes to the `CHANGELOG.md` file at the top.
3. [ ] I have made any necessary changes to the documentation in the `README`.

### Code checks
1. [ ] If there are new features implemented, add suitable tests
   in the `tests` directory.
2. [ ] If any new dependencies are introduced through the new features,
   add the packages to `dependencies` (or to the `dev` extra)
   in `pyproject.toml`.
3. [ ] Run `make fasttest` in a console in the top level directory
   to make sure all the tests pass.
4. [ ] Run `make format` in a console in the top level directory
   to make the code comply with the formatting standards.
