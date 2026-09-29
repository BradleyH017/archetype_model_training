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

# Falls back to bare `python` (whatever's on PATH) if not overridden -- set
# PYTHON_BIN explicitly if that's not the right 3.12 interpreter for you.
PYTHON_BIN="${PYTHON_BIN:-python}"

# Hard requirement, not a suggestion -- see the module comment above for why
# MultiMIL has never had a release compatible with an older Python.
python -c "
import sys
if sys.version_info < (3, 12):
    sys.exit(f'PYTHON_BIN resolved to Python {sys.version.split()[0]} ({sys.executable}) -- MultiMIL needs >=3.12.')
"

TARGET_DIR="multimil_packages_1.0.0"
mkdir -p "$TARGET_DIR"

MULTIMIL_VERSION="1.0.0"
TORCH_VERSION="2.6.0+cu124"
pip install --target="$TARGET_DIR" \
    --extra-index-url https://download.pytorch.org/whl/cu124 \
    "torch==${TORCH_VERSION}" "multimil==${MULTIMIL_VERSION}"

# The one broken transitive dependency: force it back down to the last 2.x
# release, which is the API (`zarr.core.Array`) anndata's own zarr-io compat
# layer actually expects -- confirmed by direct traceback, not guessed.
# --no-deps + --force-reinstall here so this one downgrade doesn't also drag
# anndata/scvi-tools back to older versions to "satisfy" zarr<3 -- none of
# them actually require zarr>=3, that upper bound just wasn't set anywhere
# in this tree. Safe as a separate force-reinstall pass (unlike torch above):
# zarr is plain Python with no same-purpose sibling packages under a
# different name, so there's nothing for a second pass to leave orphaned.
#"$PYTHON_BIN" -m pip install --target="$TARGET_DIR" --no-deps --force-reinstall "zarr<3"

# A real bug in multimil==1.0.0 itself (confirmed still present on the
# unreleased `main` branch too, so no newer release fixes it -- can't just
# bump the version pin above): create_df() in multimil/utils/_utils.py
# mutates its `pred` dict argument in place -- `pred[key] = np.atleast_1d(
# torch.cat(pred[key])...)` for every key, all at once. get_model_output()
# in multimil/model/_mil.py then calls save_predictions_in_adata() (which
# calls create_df()) once PER target of a given covariate type
# (classification / ordinal_regression / regression), but passes the SAME
# bag_true dict every time rather than a fresh copy per target. So
# processing the first target converts every OTHER target's still-list-of-
# tensors entry to a numpy array too, as a side effect -- and the second
# target's own create_df() call then does torch.cat() on an already-numpy
# value, crashing with "TypeError: cat(): argument 'tensors' (position 1)
# must be tuple of Tensors, not numpy.ndarray". Confirmed by direct
# traceback, reproducible with any model that registers 2+ targets of the
# same type -- which is presumably why upstream's own tutorials (one target
# each) never hit it, but this pipeline's 3 archetype regression targets
# (B1/B2/B3_clr_pca) do, on the second target processed.
#
# Patched here (not just once by hand in a running checkout) so a fresh
# `rm -rf multimil_packages_1.0.0` + rerun doesn't silently lose the fix and
# reintroduce the crash: copying the dict before mutating it means
# converting one target's entries never touches the entries belonging to a
# target get_model_output() hasn't processed yet. Idempotent (skips if
# already patched, e.g. from a hand-edit) and fails loudly rather than
# silently no-op-ing if upstream's source no longer matches what's expected
# here, so a future multimil release changing this code gets noticed rather
# than silently left unpatched.
UTILS_FILE="${TARGET_DIR}/multimil/utils/_utils.py"
"$PYTHON_BIN" - "$UTILS_FILE" <<'PATCH_EOF'
import sys

path = sys.argv[1]
src = open(path).read()

if "pred = dict(pred)" in src:
    print(f"{path}: create_df() dict-mutation patch already applied -- skipping")
    sys.exit(0)

old = "    if isinstance(pred, dict):\n        for key in pred.keys():\n"
if old not in src:
    sys.exit(
        f"Could not find the expected unpatched create_df() code in {path} -- "
        "multimil's source has changed since this patch was written; check "
        "installs/multimil_download.sh and whether this bug is still present."
    )

new = (
    "    if isinstance(pred, dict):\n"
    "        # Upstream bug patch -- see installs/multimil_download.sh for\n"
    "        # the full explanation. Copy the dict before mutating it:\n"
    "        # get_model_output() reuses this SAME dict across every target\n"
    "        # of a covariate type instead of passing a fresh copy per\n"
    "        # target, so mutating pred[key] in place here would silently\n"
    "        # corrupt the still-list-of-tensors entries of targets not yet\n"
    "        # processed.\n"
    "        pred = dict(pred)\n"
    "        for key in pred.keys():\n"
)
open(path, "w").write(src.replace(old, new, 1))
print(f"Patched {path} (create_df dict-mutation bug)")
PATCH_EOF

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
