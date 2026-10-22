# Intake protocol — sendability work (U04)

Internal working document for the sendability revision. It is not part of the
scientific release and may be removed before a merge to `main`.

Recorded 2026-10-22 in a cloud container (Linux x86_64, Python 3.13.16).

## Starting states

| Item | Value | How confirmed |
|---|---|---|
| Method repository | `https://github.com/Satori-1618/causal-decidability-method-case` | `git remote -v` |
| Public `main` | `ea4a3a52756210f7635a4962105d1633de91d54f` | `git ls-remote --heads origin` |
| Li/Saphra branch `codex/saphra-ablation-preflight` | `24f3e2639db276c72414385a398c5b9d135e4059` | `git ls-remote --heads origin` |
| Working branch | `claude/brave-thompson-9vr9bn`, created locally at `ea4a3a5` | `git rev-parse HEAD`; the branch was **not** on the remote before the first push of this work |
| Local changes before work | none | `git status --porcelain` empty |
| Website repository | `Sapphire-Bridge/sapphire-bridge.github.io` @ `4d082688d84ad166938022b55d4ae933428fd988` | anonymous shallow clone; read-only in this session |
| Other public branches | 8 heads (listed by `git ls-remote`) | no changes made to them |

Nothing was reset or overwritten. A detached read-only worktree of `24f3e26` was used for
review only.

## Inputs available

- Public method repository (full history) and the public website clone.
- Anonymous GitHub REST reads (public events API) for push chronology.
- An isolated verification environment: Python 3.13.16, NumPy 2.4.6, SciPy 1.15.3, and
  this repository installed with `pip install --no-deps -e .`. No PyTorch, JAX,
  Transformers or model weights are installed.

## Inputs missing (they block only the tasks that depend on them)

| Missing input | Blocks |
|---|---|
| Private source repository `Satori-1618/causal-decidability` (access request refused in this session) | TaxiGPT and Causal Parrots verification packages (plan 2.6.3, 2.7.4–2.7.5) |
| The locally prepared package candidates (TaxiGPT, Causal Parrots) | same |
| The counter-audit matrix (64 items for v1) and feedback items U01–U02 | plan 0.3 |
| arXiv access (DNS blocked here) | wording checks against the original papers. The counter-audit reports having read the central originals; those results are not re-derived here. |
| Write access to the website repository | publishing website changes; local patches only |

No missing input is reconstructed from summaries.

## Baseline verification of public `main`

A fresh anonymous clone of `ea4a3a5` was checked with the isolated environment:

```
cd <fresh clone>
<venv>/bin/pip install --no-deps -e .
<venv>/bin/python scripts/verify_release.py
```

All eight checks passed. The log is in `baseline_verify_release_ea4a3a5.log`. The working
tree stayed clean; only the ignored `src/causal_decidability.egg-info/` was created
by the editable install.
