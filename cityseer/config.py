import os

QUIET_MODE: bool = False
if "GCP_PROJECT" in os.environ:
    QUIET_MODE = True

if "CITYSEER_QUIET_MODE" in os.environ:
    if os.environ["CITYSEER_QUIET_MODE"].lower() in ["true", "1"]:
        QUIET_MODE = True

DEBUG_MODE: bool = False
if "CITYSEER_DEBUG_MODE" in os.environ:
    if os.environ["CITYSEER_DEBUG_MODE"].lower() in ["true", "1"]:
        DEBUG_MODE = True

# for calculating default betas vs. distances
MIN_THRESH_WT: float = 0.01831563888873418
# for all_close equality checks
ATOL: float = 0.001
RTOL: float = 0.0001
# fastmath flags
FASTMATH: set[str] = {"ninf", "nsz", "arcp", "contract", "afn", "reassoc"}
