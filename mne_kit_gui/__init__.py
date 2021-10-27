"""Convenience functions for opening GUIs."""

# Authors: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD-3-Clause

import os

from mne.utils import verbose, get_config
from ._utils import _check_mayavi_version
from ._backend import _testing_mode


__version__ = '1.0.1'


def _initialize_gui(frame, view=None):
    """Initialize GUI depending on testing mode."""
    if _testing_mode():  # open without entering mainloop
        return frame.edit_traits(view=view), frame
    else:
        frame.configure_traits(view=view)
        return frame


def fiducials(subject=None, fid_file=None, subjects_dir=None):
    """Set the fiducials for an MRI subject.

    Parameters
    ----------
    subject : str
        Name of the mri subject.
    fid_file : None | str
        Load a fiducials file different form the subject's default
        ("{subjects_dir}/{subject}/bem/{subject}-fiducials.fif").
    subjects_dir : None | str
        Overrule the subjects_dir environment variable.

    Returns
    -------
    frame : instance of FiducialsFrame
        The GUI frame.

    Notes
    -----
    All parameters are optional, since they can be set through the GUI.
    The functionality in this GUI is also part of :func:`coregistration`.
    """
    _check_mayavi_version()
    from ._backend import _check_backend
    _check_backend()
    from ._fiducials_gui import FiducialsFrame
    frame = FiducialsFrame(subject, subjects_dir, fid_file=fid_file)
    return _initialize_gui(frame)


def kit2fiff():
    """Convert KIT files to the fiff format.

    The recommended way to use the GUI is through bash with::

        $ mne kit2fiff

    Returns
    -------
    frame : instance of Kit2FiffFrame
        The GUI frame.
    """
    _check_mayavi_version()
    from ._backend import _check_backend
    _check_backend()
    from ._kit2fiff_gui import Kit2FiffFrame
    frame = Kit2FiffFrame()
    return _initialize_gui(frame)
