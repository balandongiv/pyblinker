# 024 Validation Repo Restructure

1. Title
Restructure `blinker_pyblinker_validation` into a reproducible research repository with code under `src`, tutorial entry points under `tutorial`, and canonical validation runners under `src/validation`

2. Date/time
2026-03-15 12:00:00 +08:00

3. Hypothesis
If the legacy validation package and the root-level tutorial scripts are consolidated into a single source tree, then the repository will become easier to share publicly without breaking the editable `pyblinker` development loop.

4. Files inspected
- `C:\Users\balan\IdeaProjects\blinker_pyblinker_validation\README.md`
- `C:\Users\balan\IdeaProjects\blinker_pyblinker_validation\good_practice.md`
- `C:\Users\balan\IdeaProjects\blinker_pyblinker_validation\pyblinker\blinker_pyblinker_validation\*`
- `C:\Users\balan\IdeaProjects\blinker_pyblinker_validation\murat_sequence\*`
- `C:\Users\balan\IdeaProjects\blinker_pyblinker_validation\raja_sequence\*`
- `C:\Users\balan\IdeaProjects\blinker_pyblinker_validation\src\*`

5. Files changed
- pending implementation

6. Exact change made
Started a full repository-structure cleanup so that canonical validation code moves into `src/validation`, tutorial/pipeline entry points move under `tutorial`, and documentation is rewritten around the current post-reset workflow.

7. Why the change was made
The repository needs to be shareable as a public research codebase, which requires a clear separation between core validation code, tutorial entry points, generated artifacts, and historical findings.

8. MATLAB reference used
- `murat_sequence/step2_run_blinker.py`
- `raja_sequence/step3_run_blinker.py`
- MATLAB helper code under `src/matlab_runner`

9. Validation scope
- Which subjects:
  - structural refactor only
- How many subjects:
  - 0
- Which tests:
  - pending verification after implementation

10. Before/after metrics
Before:
- repository contains a split structure:
  - partially migrated modules under `src`
  - validation runners still under `pyblinker/blinker_pyblinker_validation`
  - tutorial entry points at the repository root

After:
- pending implementation

11. Whether the change was kept or reverted
Pending.

12. Next recommended step
Move the canonical validation modules into `src/validation`, relocate tutorial entry points into `tutorial`, update imports and output paths, then rewrite the README and reproducibility notes around the new structure.
