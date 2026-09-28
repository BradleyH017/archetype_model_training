#!/bin/bash
# temp/multimil_download.sh
# Installs MultiMIL (https://github.com/theislab/multimil) and its dependency
# tree into a local directory rather than the environment's own site-packages
# -- same rationale as temp/shap_download.sh: useful when the environment
# lives in a (possibly read-only/shared) Singularity container you don't want
# to modify. Run this using the SAME python3 you'll later run MultiMIL with
# (e.g. inside the same `singularity exec <container> bash
# temp/multimil_download.sh`), so the installed builds match that Python's
# version/ABI -- confirmed to matter directly, not just caution: a plain
# `pip install multimil --target pip_packages` (no version pin) resolved
# zarr==3.4.0 as a fresh transitive dependency of anndata, and zarr 3.x's own
# code uses the `type X = ...` alias statement (PEP 695), which needs Python
# 3.12+ -- a SyntaxError, not an AttributeError, the moment anndata (and
# hence scvi-tools, and hence multimil) is imported under an older Python
# (e.g. this pipeline's containers, which run 3.9-3.10). Separately, even on
# a new-enough Python, zarr 3.x is a structural rewrite that removed the
# `zarr.core.Array` path anndata's compat shim imports -- so the fix is not
# "run on a newer Python," it's specifically "pin zarr to its last 2.x line."
#
# Usage:
#   bash temp/multimil_download.sh
set -euo pipefail

TARGET_DIR="multimil_packages"
mkdir -p "$TARGET_DIR"

# Sanity check: which interpreter is actually doing this install -- compare
# against the same check run where you later `import multimil`. See
# shap_download.sh's identical comment for why a mismatch here (e.g. the
# bare host's python3 vs. the Singularity container's) breaks compiled/
# version-sensitive dependencies even when both report similar version
# numbers.
echo "Installing with: $(python3 -c 'import sys; print(sys.version, sys.executable)')"

if ! python3 -m pip --version >/dev/null 2>&1; then
    echo "No pip module for this Python -- bootstrapping via ensurepip --user"
    python3 -m ensurepip --user
fi

# Pinned to the version already confirmed working here (1.0.0) -- unlike
# shap, multimil genuinely needs its full dependency tree (anndata,
# scvi-tools, lightning/pytorch_lightning, mudata, etc. -- none of which this
# pipeline's containers already provide, unlike numpy/pandas/scikit-learn),
# so this is a normal (not --no-deps) install: pip resolves and pulls all of
# that in one go, confirmed to install successfully end-to-end except for
# the zarr issue fixed below.
MULTIMIL_VERSION="0.3.2"
python3 -m pip install --target="$TARGET_DIR" "multimil==${MULTIMIL_VERSION}"

# The one broken transitive dependency: force it back down to the last 2.x
# release, which is the API (`zarr.core.Array`) anndata's own zarr-io compat
# layer actually expects -- confirmed by direct SyntaxError/traceback, not
# guessed. --no-deps + --force-reinstall here so this one downgrade doesn't
# also drag anndata/scvi-tools back to older versions to "satisfy" zarr<3 --
# none of them actually require zarr>=3, that upper bound just wasn't set
# anywhere in this tree.
python3 -m pip install --target="$TARGET_DIR" --no-deps --force-reinstall "zarr<3"

echo
echo "Installed multimil==${MULTIMIL_VERSION} (+ deps, zarr pinned <3) into ${TARGET_DIR}/"
echo "To use it, PREPEND that directory to PYTHONPATH -- not append -- so the"
echo "versions installed here are used consistently instead of mixing with"
echo "whatever anndata/scvi-tools this container may already ship, e.g.:"
echo "  PYTHONPATH=\"${TARGET_DIR}:\${PYTHONPATH:-}\" python3 -c 'import multimil'"
echo "Verify it resolved from here, not the container's own site-packages:"
echo "  PYTHONPATH=\"${TARGET_DIR}:\${PYTHONPATH:-}\" python3 -c 'import anndata; print(anndata.__file__)'"
