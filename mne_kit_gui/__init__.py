"""Convenience functions for opening GUIs."""

# Authors: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD-3-Clause

from mne.viz.backends._utils import _init_mne_qtapp, _qt_app_exec


__version__ = '1.3.0'


def fiducials(subject=None, fid_file=None, subjects_dir=None, *, block=True):
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
    block : bool
        If True (default), enter the Qt event loop and block until the
        GUI is closed. Set to False (e.g. in tests) to show the GUI
        without blocking.

    Returns
    -------
    frame : instance of FiducialsFrame
        The GUI frame.

    Notes
    -----
    All parameters are optional, since they can be set through the GUI.
    The functionality in this GUI is also part of :func:`coregistration`.
    """
    from ._fiducials_gui import FiducialsFrame
    app = _init_mne_qtapp()
    frame = FiducialsFrame(subject=subject, subjects_dir=subjects_dir)
    if fid_file is not None:
        frame.model.fid.file = fid_file
    frame.show()
    if block:
        _qt_app_exec(app)
    return frame


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
