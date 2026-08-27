#!/usr/bin/env python3
"""
999-parse_training_config.py
Utility, not a pipeline step: extracts the handful of scalar params
run_finalize_model.sh needs from a Nextflow-style conf/training.config,
without depending on Nextflow itself (`nextflow config` would be the more
"proper" way to resolve params, but adds a JVM-startup dependency to what is
otherwise a plain bash wrapper -- this targets exactly this project's config
style instead: single-line `key = value` assignments inside `params { ... }`,
optionally followed by a `// comment`).

NOT a general Groovy config parser -- multi-line values, `${...}`
interpolation, or a param overridden inside a `profiles { ... }` block won't
be picked up correctly. Good enough for conf/training.config as it's
actually written; re-check by hand if that file's structure changes
significantly.

Prints shell-assignable `KEY=value` lines (values shell-quoted) to stdout,
meant to be consumed via `eval "$(python3 scripts/999-parse_training_config.py --config <path>)"`.

Usage:
    python scripts/999-parse_training_config.py --config conf/training.config
"""
import argparse
import re
import shlex
import sys


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    return p.parse_args()


def extract_scalar(text, key):
    """First `key = value` assignment, value either a quoted string or a bare
    (non-whitespace) token -- stops before any trailing `// comment` either
    way. Returns None if not found."""
    m = re.search(
        r'^\s*' + re.escape(key) + r'\s*=\s*("[^"]*"|\'[^\']*\'|\S+)',
        text, re.MULTILINE,
    )
    if not m:
        return None
    val = m.group(1)
    if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
        val = val[1:-1]
    return val


def extract_container(text, process_name):
    """container = "..." specifically inside a withName: '<process_name>' { ... }
    block -- NOT the same as a top-level params.container_* value, since the
    actual container used by a process can be overridden per-process."""
    block_start = re.search(
        r"withName:\s*['\"][^'\"]*" + re.escape(process_name) + r"[^'\"]*['\"]\s*\{",
        text,
    )
    if not block_start:
        return None
    # Search forward from the block start to the next closing brace at the
    # same nesting level -- approximated here by just the next "}" on its own
    # line, which matches how conf/training.config actually formats these
    # blocks (no nested braces inside a withName block currently).
    rest = text[block_start.end():]
    end = rest.find("\n    }")
    block = rest[:end] if end != -1 else rest
    m = re.search(r'container\s*=\s*"([^"]+)"', block)
    return m.group(1) if m else None


def main():
    args = parse_args()
    with open(args.config) as f:
        text = f.read()

    out = {}
    for key in ["hvg_pct", "hvg_max", "hvg_select_method", "meta_file",
                "train_only_covs", "train_test_covs",
                "n_outer_folds", "random_seed"]:
        val = extract_scalar(text, key)
        if val is None:
            print(f"# WARNING: '{key}' not found in {args.config}", file=sys.stderr)
            val = ""
        out[key] = val

    # label_mode isn't a single scalar -- it's implied by whichever
    # run_label_* flag is "true" (the pipeline only ever runs one at a time
    # in practice; see conf/training.config's own comment on this).
    label_flags = {
        "full": extract_scalar(text, "run_label_full"),
        "argmax": extract_scalar(text, "run_label_argmax"),
        "argmax_filtered": extract_scalar(text, "run_label_argmax_filtered"),
    }
    active = [mode for mode, v in label_flags.items() if v == "true"]
    if len(active) != 1:
        print(f"# WARNING: expected exactly one run_label_* flag set to \"true\", "
              f"found {active or 'none'} -- defaulting label_mode to 'full'", file=sys.stderr)
        out["label_mode"] = "full"
    else:
        out["label_mode"] = active[0]

    container = extract_container(text, "TRAIN_TREE_MODELS")
    if container is None:
        print("# WARNING: could not find TRAIN_TREE_MODELS's container override "
              "in the process{} block -- falling back to params.xgboost2_env",
              file=sys.stderr)
        container = extract_scalar(text, "xgboost2_env") or ""
    out["container"] = container

    for key, val in out.items():
        print(f"{key.upper()}={shlex.quote(val)}")


if __name__ == "__main__":
    main()
