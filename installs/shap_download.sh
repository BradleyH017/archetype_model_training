#!/bin/bash
# temp/shap_download.sh
# Installs a specific shap version into a local directory rather than the
# environment's own site-packages -- useful when the environment lives in a
# (possibly read-only/shared) Singularity container you don't want to modify.
# Run this using the SAME python3 you'll later run 008_shap_explain.py with
# (e.g. inside the same `singularity exec <container> bash temp/shap_download.sh`),
# so the installed build matches that Python's version/ABI.
#
# Usage:
#   bash temp/shap_download.sh
set -euo pipefail

# 0.47.0 minimum, not 0.46.0: this container's ambient numpy is 2.4.6, and
# shap<0.47.0's vendored shap/plots/colors/_colorconv.py (a copy of an old
# scikit-image module) uses a dtype-construction pattern NumPy 2.0 removed --
# confirmed fixed in 0.47.0 specifically ("Re-vendor colorconv from skimage
# 0.24.0", https://github.com/shap/shap/pull/3785), not just guessed.
SHAP_VERSION="0.47.0"
TARGET_DIR="pip_packages"

mkdir -p "$TARGET_DIR"

# Sanity check: which interpreter is actually doing this install. Compare
# this against the same check run where you later `import shap` -- if they
# differ, that mismatch is almost certainly why a compiled dependency
# (e.g. numba) fails to import afterward, even if both report the same
# version number (two "3.10.12"s built against different system libraries,
# e.g. one on the bare host and one inside a Singularity container, are not
# guaranteed to be interchangeable for compiled extensions).
echo "Installing with: $(python3 -c 'import sys; print(sys.version, sys.executable)')"

# This Python has no pip module at all (not just missing from PATH) -- seen
# on Spack-built containers, which manage packages their own way and don't
# bundle pip. ensurepip is the standard offline bootstrap for exactly this
# (a small vendored pip wheel that ships in CPython's stdlib on most
# builds) -- --user installs it into $HOME's site-packages rather than the
# container's own (likely read-only) one. If ensurepip ALSO isn't available,
# this container's Python was built without it entirely (some minimal/Spack
# builds strip it too) -- at that point you'd need either network access for
# https://bootstrap.pypa.io/get-pip.py, or a different environment that
# already has a matching Python + pip.
if ! python3 -m pip --version >/dev/null 2>&1; then
    echo "No pip module for this Python -- bootstrapping via ensurepip --user"
    python3 -m ensurepip --user
fi

# --no-deps matters, confirmed by direct experience, not just caution: without
# it, pip installed its own numpy into $TARGET_DIR, which then shadows the
# container's own via PYTHONPATH -- and pip's freshly-resolved numpy didn't
# match the container's ambient one (2.4.6) closely enough to avoid breaking
# shap<0.47.0's vendored color-conversion code (see the SHAP_VERSION comment
# above -- fixed upstream in 0.47.0, not something to work around here).
# --no-deps leaves numpy/scipy/pandas/scikit-learn/etc. for the container's
# own copies -- already proven working with everything else in this
# pipeline (xgboost, sklearn) -- to satisfy instead, rather than risking a
# second, possibly-inconsistent version of any of them.
python3 -m pip install --target="$TARGET_DIR" --no-deps "shap==${SHAP_VERSION}"

# shap also declares several SMALL, shap-specific dependencies this container
# has no other reason to already have -- unlike numpy/scipy/pandas/
# scikit-learn, which it does. Confirmed by direct ModuleNotFoundErrors, not
# assumed.
python3 -m pip install --target="$TARGET_DIR" --no-deps slicer cloudpickle tqdm packaging

# numba turned out to be missing too, despite being part of the general
# scientific-Python ecosystem -- it's more specialized (its own LLVM
# toolchain, via llvmlite) than the others and evidently wasn't included in
# this container's build. numba/llvmlite are tightly co-versioned (numba's
# own dependency metadata pins a matching llvmlite range for a reason -- its
# C-API usage of LLVM is version-sensitive), so pin BOTH explicitly rather
# than let each resolve independently. Per numba's own compatibility table
# (numba.readthedocs.io/en/stable/user/installing.html): numba==0.67.0
# supports numpy 2.0 <= version < 2.6 (covers this container's 2.4.6) and
# requires llvmlite 0.49.x specifically -- both confirmed, not guessed.
# Still --no-deps: numba/llvmlite only need SOME compatible numpy already
# present, not to install their own.
python3 -m pip install --target="$TARGET_DIR" --no-deps "numba==0.67.0" "llvmlite~=0.49.0"

echo
echo "Installed shap==${SHAP_VERSION} into ${TARGET_DIR}/"
echo "To use it, add that directory to PYTHONPATH before running your script, e.g.:"
echo "  PYTHONPATH=\"${TARGET_DIR}:\${PYTHONPATH:-}\" python3 scripts/008_shap_explain.py ..."
