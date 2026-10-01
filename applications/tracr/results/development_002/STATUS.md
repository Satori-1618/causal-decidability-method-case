Preflight stopped before collecting outcomes. The expanded upstream clean-source
check incorrectly treated generated Python bytecode as source changes. The check
was narrowed to Python source files (tracked changes and untracked additions).
No model-outcome records were produced; the directory is preserved rather than
silently reused. No confirmation cases had been generated.
