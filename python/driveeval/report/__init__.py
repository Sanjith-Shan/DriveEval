"""HTML reporting for DriveEval.

`build.py` turns the results store into one self-contained file. The section
order is fixed and starts with what the harness cannot measure, because a
reviewer who reads the limits first can calibrate everything after them; a
reviewer who meets them at the end has already formed a view.
"""

from driveeval.report.assets import CSS, PALETTE, SEQUENTIAL
from driveeval.report.build import LIMITATIONS_FALLBACK, build_report

__all__ = ["CSS", "LIMITATIONS_FALLBACK", "PALETTE", "SEQUENTIAL", "build_report"]
