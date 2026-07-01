"""Convenience functions for opening GUIs."""

# Authors: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD-3-Clause

from importlib.metadata import version

from mne.viz.backends._utils import _init_mne_qtapp, _qt_app_exec

try:
    __version__ = version("mne-kit-gui")
except Exception:
    __version__ = "0.0.0"
del version


def fiducials(*args, **kwargs):
    """Set the fiducials for an MRI subject (removed).

    .. deprecated::
        This GUI has been removed. Use the MRI fiducials functionality in
        ``mne coreg`` instead.
    """
    raise RuntimeError(
        "The mne_kit_gui.fiducials GUI has been removed. Use the coregistration "
        "GUI instead, e.g. by running `mne coreg` from the command line or "
        "`mne.gui.coregistration()` from Python."
    )


def kit2fiff(*, block=True):
    """Convert KIT files to the fiff format.

    The recommended way to use the GUI is through bash with::

        $ mne kit2fiff

    Parameters
    ----------
    block : bool
        If True (default), enter the Qt event loop and block until the
        GUI is closed. Set to False (e.g. in tests) to show the GUI
        without blocking.

    Returns
    -------
    frame : instance of Kit2FiffFrame
        The GUI frame.
    """
    from ._kit2fiff_gui import Kit2FiffFrame

    app = _init_mne_qtapp()
    frame = Kit2FiffFrame()
    frame.show()
    if block:
        _qt_app_exec(app)
    return frame
