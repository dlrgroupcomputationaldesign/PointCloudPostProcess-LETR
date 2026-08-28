"""Derive SURVEY_BASIS from a point cloud. Re-export shim.

The implementation moved into the package
(``post_process.utils.survey_basis_util``) so the pipeline can compute a basis
itself when none is supplied, rather than depending on a script at the repo
root. This module stays so existing imports keep working:

    from survey_basis import survey_basis_from_df, survey_basis_from_e57

Keeping one implementation matters more than the convenience of a root-level
file: two copies of an estimator whose convention is easy to get backwards
(see yaw_matrix) would drift, and the failure is silent -- a transposed basis is
still a valid rotation.
"""

import sys
from pathlib import Path

_PKG = Path(__file__).resolve().parent / "post_process_src"
if _PKG.is_dir() and str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from post_process.utils.survey_basis_util import (  # noqa: E402,F401
    WALL_LABEL,
    estimate_yaw,
    survey_basis_from_df,
    survey_basis_from_e57,
    yaw_matrix,
)

__all__ = [
    "WALL_LABEL",
    "estimate_yaw",
    "survey_basis_from_df",
    "survey_basis_from_e57",
    "yaw_matrix",
]
