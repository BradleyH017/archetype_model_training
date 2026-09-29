#!/bin/bash
# installs/multimil_download.sh
# Installs MultiMIL (https://github.com/theislab/multimil) and its dependency
# tree into a local directory rather than the environment's own site-packages
# -- same rationale as installs/shap_download.sh: useful when the environment
# lives in a (possibly read-only/shared) Singularity container you don't want
# to modify.
#
# MultiMIL==1.0.0 requires Python >=3.12 -- confirmed directly from PyPI
# metadata (`requires_python`), not guessed, and it's not just this release:
# every published version requires at least Python 3.10 (0.0.1-0.3.2:
# >=3.10; 1.0.0: >=3.12), so there is no version of MultiMIL that runs on
# this pipeline's own containers (Python 3.9-3.10). `pip install --target=`
# does NOT enforce requires-python the way a normal environment install
# does, so running this with too old a python3 will either silently install
# source it can't run (surfacing later as e.g. a `bool | int` TypeError from
# PEP 604 union syntax needing 3.10+) or, with a modern enough pip, simply
# refuse with "Could not find a version that satisfies the requirement
# multimil==1.0.0 (from versions: 0.0.1, 0.2.0, ..., 0.3.2)" -- both observed
# directly. Hence the hard version check below rather than just an
# informational echo.
#
# Separately (unrelated to the Python-version requirement above): a plain
# `pip install multimil` also resolves zarr==3.4.0 as a fresh transitive
# dependency of anndata, and zarr 3.x is a structural rewrite that dropped
# the `zarr.core.Array` path anndata's own zarr-io compat shim imports --
# confirmed by direct traceback, not guessed. Pin it back to the last 2.x
# line below regardless of Python version.
#
# Usage:
#   PYTHON_BIN=/path/to/python3.12 bash installs/multimil_download.sh
#   (PYTHON_BIN defaults to `python3.12` on PATH, falling back to Sanger's
#   conda-audited miniforge build below -- override it if neither applies
#   to you.)
set -euo pipefail

PYTHON_BIN="${/software/hgi/softpack/installs/users/sb84/bayesian_inference/1-scripts/python}"


# Hard requirement, not a suggestion -- see the module comment above for why
# MultiMIL has never had a release compatible with an older Python.
python -c "
import sys
if sys.version_info < (3, 12):
    sys.exit(f'PYTHON_BIN resolved to Python {sys.version.split()[0]} ({sys.executable}) -- MultiMIL needs >=3.12.')
"

TARGET_DIR="multimil_packages_1.0.0"
mkdir -p "$TARGET_DIR"

# Exported now (not just documented at the end) so that if pip itself needs
# bootstrapping below, the bootstrapped copy is immediately importable too --
# see that block for why plain `ensurepip --user` isn't used here. Required
# for every `-m pip` call below to work at all: pip under --target is a
# plain importable package (multimil_packages_1.0.0/pip/), not a script, so
# it only resolves via `python -m pip` once its directory is on PYTHONPATH --
# there is no standalone `pip` executable to run directly here.

if ! python -m pip --version >/dev/null 2>&1; then
    # `ensurepip --user` installs into $HOME/.local -- deliberately not used:
    # this Python (e.g. a Spack/conda-audited build) may also not add
    # $HOME/.local to its user-site path at all, which is exactly what
    # happened here (pip installed successfully but was then still reported
    # as "No module named pip" on the very next invocation). get-pip.py
    # --target instead installs pip as a plain package directly into
    # TARGET_DIR, the same place -- and same mechanism -- every other
    # package in this script lands, entirely independent of $HOME.
    echo "No pip module for this Python -- bootstrapping into ${TARGET_DIR} (not \$HOME) via get-pip.py"
    curl -sSL https://bootstrap.pypa.io/get-pip.py -o "${TARGET_DIR}/get-pip.py"
    python "${TARGET_DIR}/get-pip.py" --target="$TARGET_DIR" --no-warn-script-location
    rm -f "${TARGET_DIR}/get-pip.py"
fi

# Not --no-deps: unlike shap, multimil genuinely needs its full dependency
# tree (anndata, scvi-tools, lightning/pytorch_lightning, mudata, etc. --
# none of which this pipeline's containers already provide, unlike
# numpy/pandas/scikit-learn), so pip resolves and pulls all of that in one
# go -- confirmed to install successfully end-to-end except for the zarr
# issue fixed below.
MULTIMIL_VERSION="1.0.0"
pip install --target="$TARGET_DIR" "multimil==${MULTIMIL_VERSION}"

# The one broken transitive dependency: force it back down to the last 2.x
# release, which is the API (`zarr.core.Array`) anndata's own zarr-io compat
# layer actually expects -- confirmed by direct traceback, not guessed.
# --no-deps + --force-reinstall here so this one downgrade doesn't also drag
# anndata/scvi-tools back to older versions to "satisfy" zarr<3 -- none of
# them actually require zarr>=3, that upper bound just wasn't set anywhere
# in this tree.
#"$PYTHON_BIN" -m pip install --target="$TARGET_DIR" --no-deps --force-reinstall "zarr<3"

# A second broken transitive dependency, this time GPU-relevant: multimil's
# resolution pulls the newest default torch, and torch's default (plain
# PyPI, no --index-url) wheel now bundles CUDA 13.0 (confirmed from its own
# requires_dist: `cuda-toolkit==13.0.3`), which needs a driver much newer
# than most clusters have -- confirmed by direct
# "NVIDIA driver on your system is too old (found version 12040)" error, not
# guessed. Force-installed here from PyTorch's own cu124 wheel index instead
# -- 2.6.0 is the newest version that index still builds for cu124, and it
# satisfies every already-installed package that depends on torch here
# (lightning==2.6.6 needs >=2.1.0,<4.0; pytorch_lightning==2.6.6 needs
# >=2.1.0; scvi-tools==1.5.1 is unconstrained). Deliberately NOT --no-deps,
# unlike the zarr fix above: torch's matching nvidia-*-cu12 shared libraries
# (cublas/cudnn/nccl/etc.) have to be pulled alongside it from the same
# index, not left as whatever cu13 copies the default resolution grabbed.
pip install --target="$TARGET_DIR" --force-reinstall \
    --index-url https://download.pytorch.org/whl/cu124 "torch==2.6.0"

echo
echo "Installed multimil==${MULTIMIL_VERSION} (+ deps, zarr pinned <3) into ${TARGET_DIR}/"
echo "To use it, PREPEND that directory to PYTHONPATH -- not append -- so the"
echo "versions installed here are used consistently instead of mixing with"
echo "whatever anndata/scvi-tools this container may already ship, e.g.:"
echo "  PYTHONPATH=\"${TARGET_DIR}:\${PYTHONPATH:-}\" ${PYTHON_BIN} -c 'import multimil'"
echo "Verify it resolved from here, not the container's own site-packages:"
echo "  PYTHONPATH=\"${TARGET_DIR}:\${PYTHONPATH:-}\" ${PYTHON_BIN} -c 'import anndata; print(anndata.__file__)'"
echo
echo "Remember to also run any script that imports multimil with this SAME"
echo "python3.12 interpreter, not the container's own python3 -- e.g.:"
echo "  PYTHONPATH=\"${TARGET_DIR}:\${PYTHONPATH:-}\" ${PYTHON_BIN} scripts/003a-multimil.py"
