#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SGLANG_VERSION=$(python -c 'from importlib.metadata import version; print(version("sglang"))')
case "$SGLANG_VERSION" in
    0.5.12.post1) PATCH_FILE="$SCRIPT_DIR/sglang_${SGLANG_VERSION}_student_topk.patch" ;;
    *) exit 0 ;;
esac
SGLANG_PACKAGE=$(python -c 'from importlib.util import find_spec; print(next(iter(find_spec("sglang").submodule_search_locations)))')

# Run outside the checkout so Git does not filter paths under a local .venv.
if git -C / apply --unsafe-paths --directory="$SGLANG_PACKAGE" --check --reverse "$PATCH_FILE" >/dev/null 2>&1; then
    echo "[install.sh] SGLang per-position scoring patch is already applied."
    exit 0
fi
git -C / apply --unsafe-paths --directory="$SGLANG_PACKAGE" --check "$PATCH_FILE"
git -C / apply --unsafe-paths --directory="$SGLANG_PACKAGE" "$PATCH_FILE"
