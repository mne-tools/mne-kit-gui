# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD-3-Clause
import json
from pathlib import Path
from textwrap import TextWrapper


def read_tooltips(gui_name: str) -> dict[str, str]:
    """Read and format tooltips, return a dict."""
    help_path = Path(__file__).parent / "help" / (gui_name + ".json")
    with open(help_path) as fid:
        raw_tooltips = json.load(fid)
    format_ = TextWrapper(width=60, fix_sentence_endings=True).fill
    return {key: format_(text) for key, text in raw_tooltips.items()}
