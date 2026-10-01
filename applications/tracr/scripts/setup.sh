#!/usr/bin/env bash
set -euo pipefail
task_app_dir="$(cd "$(dirname "$0")/.." && pwd)"
task_repo_root="$(cd "$task_app_dir/../.." && pwd)"
task_upstream="$task_app_dir/artifacts/upstream"
task_revision="9ce2b8c82b6ba10e62e86cf6f390e7536d4fd2cd"
if [[ ! -d "$task_upstream/.git" ]]; then
  git clone https://github.com/google-deepmind/tracr.git "$task_upstream"
  git -C "$task_upstream" checkout "$task_revision"
fi
[[ "$(git -C "$task_upstream" rev-parse HEAD)" == "$task_revision" ]] || {
  echo 'Existing Tracr checkout has the wrong revision; refusing to replace it.' >&2
  exit 1
}
if [[ ! -x "$task_repo_root/.venv/bin/python" ]]; then
  python3 -m venv "$task_repo_root/.venv"
fi
if command -v uv >/dev/null 2>&1; then
  uv pip install --python "$task_repo_root/.venv/bin/python" -r "$task_app_dir/requirements.lock.txt"
  uv pip install --python "$task_repo_root/.venv/bin/python" --no-deps -e "$task_upstream" -e "$task_repo_root"
else
  "$task_repo_root/.venv/bin/python" -m ensurepip
  "$task_repo_root/.venv/bin/python" -m pip install -r "$task_app_dir/requirements.lock.txt"
  "$task_repo_root/.venv/bin/python" -m pip install --no-deps -e "$task_upstream" -e "$task_repo_root"
fi
