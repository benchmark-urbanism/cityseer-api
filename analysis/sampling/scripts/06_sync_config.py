#!/usr/bin/env python
"""
06_sync_config.py - Sync fitted model parameters to cityseer config.

Updates pysrc/cityseer/config.py with the fitted sampling model constants:
- SAMPLING_PROPORTIONAL_K: The proportional constant k
- SAMPLING_MIN_EFF_N: The minimum effective sample size floor

Also updates the baseline model constants if needed.

Outputs:
    - Modified pysrc/cityseer/config.py
"""

import json
import re
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
SAMPLING_DIR = SCRIPT_DIR.parent  # analysis/sampling
OUTPUT_DIR = SAMPLING_DIR / "output"
CONFIG_PATH = SAMPLING_DIR.parent.parent / "pysrc" / "cityseer" / "config.py"


# =============================================================================
# LOADING
# =============================================================================

def load_model() -> dict:
    """Load the fitted model parameters."""
    model_path = OUTPUT_DIR / "sampling_model.json"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Run 03_combined_model.py first.")

    with open(model_path) as f:
        return json.load(f)


def load_config() -> str:
    """Load current config.py content."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config not found at {CONFIG_PATH}")

    with open(CONFIG_PATH) as f:
        return f.read()


# =============================================================================
# UPDATING
# =============================================================================

def update_constant(content: str, name: str, value: float, comment: str = "") -> str:
    """
    Update a constant in the config file.

    Handles both existing constants (updates value) and new constants (adds them).
    """
    # Pattern to match: CONSTANT_NAME: float = value
    pattern = rf"^({name}:\s*float\s*=\s*)[\d.+-e]+(.*)$"

    if re.search(pattern, content, re.MULTILINE):
        # Update existing constant
        replacement = rf"\g<1>{value}\g<2>"
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
        print(f"  Updated: {name} = {value}")
    else:
        # Constant doesn't exist - we'll need to add it
        print(f"  WARNING: {name} not found in config.py")
        print(f"           Please add manually: {name}: float = {value}")

    return content


def sync_to_config(model: dict, dry_run: bool = False) -> bool:
    """
    Sync model parameters to config.py.

    Parameters
    ----------
    model : dict
        The sampling model with k and min_eff_n
    dry_run : bool
        If True, print changes without writing

    Returns
    -------
    bool
        True if successful
    """
    k = model["model"]["k"]
    min_eff_n = model["model"]["min_eff_n"]

    print(f"\nSyncing to {CONFIG_PATH}:")
    print(f"  k = {k}")
    print(f"  min_eff_n = {min_eff_n}")

    content = load_config()
    original_content = content

    # Update SAMPLING_PROPORTIONAL_K
    content = update_constant(content, "SAMPLING_PROPORTIONAL_K", k)

    # Check if SAMPLING_MIN_EFF_N exists, if not it needs to be added
    if "SAMPLING_MIN_EFF_N" in content:
        content = update_constant(content, "SAMPLING_MIN_EFF_N", float(min_eff_n))
    else:
        print(f"\n  NOTE: SAMPLING_MIN_EFF_N not found in config.py")
        print(f"        This is a new constant for the inverted model.")
        print(f"        Please add it manually after SAMPLING_PROPORTIONAL_K:")
        print(f"        SAMPLING_MIN_EFF_N: float = {float(min_eff_n)}")

    if dry_run:
        print("\n  [DRY RUN - no changes written]")
        if content != original_content:
            print("\n  Changes that would be made:")
            # Show diff
            for i, (old, new) in enumerate(zip(original_content.split("\n"), content.split("\n"))):
                if old != new:
                    print(f"    Line {i+1}:")
                    print(f"      - {old}")
                    print(f"      + {new}")
        return True

    if content == original_content:
        print("\n  No changes needed - config already up to date")
        return True

    # Write updated content
    with open(CONFIG_PATH, "w") as f:
        f.write(content)

    print("\n  Config updated successfully")
    return True


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("06_sync_config.py - Syncing model to cityseer config")
    print("=" * 70)

    # Load model
    model = load_model()
    print(f"\nLoaded model:")
    print(f"  k = {model['model']['k']}")
    print(f"  min_eff_n = {model['model']['min_eff_n']}")

    # Sync to config
    success = sync_to_config(model, dry_run=False)

    if success:
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"\nUpdated: {CONFIG_PATH}")
        print(f"\nThe cityseer library now uses:")
        print(f"  SAMPLING_PROPORTIONAL_K = {model['model']['k']}")
        print(f"  (SAMPLING_MIN_EFF_N = {model['model']['min_eff_n']} - add manually if new)")

        print("\n" + "-" * 70)
        print("IMPORTANT: If SAMPLING_MIN_EFF_N is new, add it to config.py:")
        print("-" * 70)
        print(f"""
# === MINIMUM EFFECTIVE SAMPLE SIZE FLOOR ===
# When reach is low, proportional sampling (k/sqrt(reach)) gives too few
# samples for reliable estimates. This floor ensures we always sample
# at least min_eff_n nodes.
# Fitted on synthetic networks (01_fit_model.py, 02_fit_floor.py)
SAMPLING_MIN_EFF_N: float = {float(model['model']['min_eff_n'])}
""")

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
