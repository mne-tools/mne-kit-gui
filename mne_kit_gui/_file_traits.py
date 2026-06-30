# -*- coding: utf-8 -*-
"""File data sources."""

# Authors: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD-3-Clause

# Path handling convention used throughout mne_kit_gui: traitlets has no
# native Path trait, and Qt widgets (QFileDialog, QLineEdit) only deal in
# str, so file/directory paths are stored as `Unicode()` traits. Internally,
# wrap them in `pathlib.Path` for any actual path manipulation (joining,
# checking existence, getting the basename/parent/suffix, etc.), then
# convert back to `str` at the point where a Path would otherwise flow into
# a trait assignment or a Qt call.

import os
from pathlib import Path

import numpy as np

from traitlets import Bool, HasTraits, Any, Int, List, Unicode, observe

from qtpy.QtWidgets import QFileDialog, QMessageBox, QProgressDialog
from qtpy.QtCore import Qt

from mne.bem import read_bem_surfaces
from mne.io.constants import FIFF
from mne.io import read_info, read_fiducials, read_raw
from mne import create_info
from mne.surface import read_surface, complete_surface_info
from mne.coreg import (_is_mri_subject, _mri_subject_has_bem,
                       create_default_subject)
from mne.utils import get_config, set_config
from mne.viz._3d import _fiducial_coords
from mne.channels import read_dig_fif

try:
    from mne.io._read_raw import _get_supported
except ImportError:  # MNE < 1.6
    def _get_supported():
        from mne.io._read_raw import supported
        return supported


fid_wildcard = "*.fif"
trans_wildcard = "*.fif"


def _expand_path(p):
    return str(Path(os.path.expandvars(p)).expanduser().absolute())


def get_fs_home(parent=None):
    """Get the FREESURFER_HOME directory.

    Parameters
    ----------
    parent : QWidget | None
        Parent widget for any dialogs shown to the user.

    Returns
    -------
    fs_home : None | str
        The FREESURFER_HOME path or None if the user cancels.

    Notes
    -----
    If FREESURFER_HOME can't be found, the user is prompted with a file dialog.
    If specified successfully, the resulting path is stored with
    mne.set_config().
    """
    return _get_root_home('FREESURFER_HOME', 'freesurfer', _fs_home_problem,
                          parent=parent)


def _get_root_home(cfg, name, check_fun, parent=None):
    root = get_config(cfg)
    problem = check_fun(root)
    while problem:
        info = ("Please select the %s directory. This is the root "
                "directory of the %s installation." % (cfg, name))
        QMessageBox.information(
            parent, "Select the %s Directory" % cfg,
            '\n\n'.join((problem, info)))
        msg = "Please select the %s Directory" % cfg
        path = QFileDialog.getExistingDirectory(parent, msg)
        if path:
            root = path
            problem = check_fun(root)
            if problem is None:
                set_config(cfg, root, set_env=False)
        else:
            return None
    return root


def set_fs_home():
    """Set the FREESURFER_HOME environment variable.

    Returns
    -------
    success : bool
        True if the environment variable could be set, False if FREESURFER_HOME
        could not be found.

    Notes
    -----
    If FREESURFER_HOME can't be found, the user is prompted with a file dialog.
    If specified successfully, the resulting path is stored with
    mne.set_config().
    """
    fs_home = get_fs_home()
    if fs_home is None:
        return False
    else:
        os.environ['FREESURFER_HOME'] = fs_home
        return True


def _fs_home_problem(fs_home):
    """Check FREESURFER_HOME path.

    Return str describing problem or None if the path is okay.
    """
    if fs_home is None:
        return "FREESURFER_HOME is not set."
    fs_home = Path(fs_home)
    if not fs_home.exists():
        return "FREESURFER_HOME (%s) does not exist." % fs_home
    if not (fs_home / 'subjects' / 'fsaverage').exists():
        return ("FREESURFER_HOME (%s) does not contain the fsaverage "
                "subject." % fs_home)


def _mne_root_problem(mne_root):
    """Check MNE_ROOT path.

    Return str describing problem or None if the path is okay.
    """
    if mne_root is None:
        return "MNE_ROOT is not set."
    mne_root = Path(mne_root)
    if not mne_root.exists():
        return "MNE_ROOT (%s) does not exist." % mne_root
    if not (mne_root / 'share' / 'mne' / 'mne_analyze').exists():
        return ("MNE_ROOT (%s) is missing files. If this is your MNE "
                "installation, consider reinstalling." % mne_root)


class Surf:
    """Expose a surface similar to the ones used elsewhere in MNE."""

    def __init__(self, rr=None, nn=None, tris=None):
        self.rr = np.empty((0, 3)) if rr is None else rr
        self.nn = np.empty((0, 3)) if nn is None else nn
        self.tris = np.empty((0, 3), int) if tris is None else tris


class SurfaceSource(HasTraits):
    """Expose points and tris of a file storing a surface.

    Parameters
    ----------
    file : str
        Path to a *-bem.fif file or a surface containing a Freesurfer surface.

    Attributes
    ----------
    surf : Surf | None
        Surface object with rr, nn, tris attributes.

    Notes
    -----
    surf is updated whenever file changes.
    """

    file = Unicode()
    surf = Any()
    parent = Any()  # QWidget | None, for parenting dialogs

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.surf = Surf()

    @observe('file')
    def _file_changed(self, change):
        """Read the file."""
        path = Path(change['new']) if change['new'] else None
        if path is not None and path.exists():
            if path.suffix == '.fif':
                bem = read_bem_surfaces(
                    path, on_defects='warn', verbose=False
                )[0]
            else:
                try:
                    bem = read_surface(path, return_dict=True)[2]
                    bem['rr'] *= 1e-3
                    complete_surface_info(bem, copy=False)
                except Exception:
                    QMessageBox.critical(
                        self.parent, "Error Loading Surface",
                        "Error loading surface from %s (see "
                        "Terminal for details)." % path)
                    self.file = ''
                    raise
            self.surf = Surf(rr=bem['rr'], tris=bem['tris'], nn=bem['nn'])
        else:
            self.surf = Surf()


class FiducialsSource(HasTraits):
    """Expose points of a given fiducials fif file.

    Parameters
    ----------
    file : str
        Path to a fif file with fiducials (*.fif).

    Attributes
    ----------
    fname : str
        Basename of the file.
    points : ndarray, shape (3, 3) | None
        Fiducial coordinates, or None if not loaded.
    """

    file = Unicode()
    fname = Unicode()
    points = Any()  # ndarray (3, 3) or None
    mni_points = Any()  # ndarray (3, 3) or None, set externally
    parent = Any()  # QWidget | None, for parenting dialogs

    @observe('file')
    def _file_changed(self, change):
        path = Path(change['new']) if change['new'] else None
        self.fname = path.name if path else ''
        if path is not None and path.exists():
            try:
                self.points = _fiducial_coords(*read_fiducials(path))
            except Exception as err:
                QMessageBox.critical(
                    self.parent, "Error Reading Fiducials",
                    "Error reading fiducials from %s: %s (See terminal "
                    "for more information)" % (self.fname, str(err)))
                self.file = ''
                raise
        else:
            self.points = self.mni_points  # can be None


class DigSource(HasTraits):
    """Expose digitization information from a file.

    Parameters
    ----------
    file : str
        Path to the raw/epochs/evoked or DigMontage file.

    Attributes
    ----------
    lpa, nasion, rpa : ndarray, shape (1, 3)
        Cardinal fiducial coordinates.
    points : ndarray, shape (n, 3)
        Head shape points (filtered by points_filter).
    eeg_points : ndarray, shape (n, 3)
        EEG sensor coordinates.
    hpi_points : ndarray, shape (n, 3)
        HPI coil coordinates.
    """

    supported = _get_supported()
    file = Unicode()

    inst_fname = Unicode()
    inst_dir = Unicode()
    _info = Any()  # mne.Info or None

    points_filter = Any()  # boolean index array or None
    n_omitted = Int()

    # head shape
    _hsp_points = Any()  # ndarray (n, 3)
    points = Any()       # ndarray (n, 3), filtered by points_filter

    # fiducials
    lpa = Any()     # ndarray (1, 3)
    nasion = Any()  # ndarray (1, 3)
    rpa = Any()     # ndarray (1, 3)

    # EEG / HPI
    eeg_points = Any()  # ndarray (n, 3)
    hpi_points = Any()  # ndarray (n, 3)

    parent = Any()  # QWidget | None, for parenting dialogs

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._reset_derived()

    def _reset_derived(self):
        empty = np.empty((0, 3))
        zeros = np.zeros((1, 3))
        self._info = None
        self.inst_fname = '-'
        self.inst_dir = ''
        self._hsp_points = empty
        self.points = empty
        self.lpa = zeros
        self.nasion = zeros
        self.rpa = zeros
        self.eeg_points = empty
        self.hpi_points = np.zeros((0, 3))
        self.n_omitted = 0

    @observe('file')
    def _file_changed(self, change):
        path = Path(change['new']) if change['new'] else None
        self.points_filter = None  # reset filter first

        if path is None:
            self._reset_derived()
            return

        self.inst_fname = path.name
        self.inst_dir = str(path.parent)

        info = self._read_info(path)
        self._info = info
        self._update_from_info()

    def _read_info(self, path):
        info = None
        if path.name.endswith(('.fif', '.fif.gz')):
            try:
                info = read_info(path, verbose=False)
            except Exception:
                try:
                    dig = read_dig_fif(path).dig
                except Exception:
                    pass
                else:
                    info = create_info(1, 1000., 'mag')
                    with info._unlock():
                        info['dig'] = dig
        else:
            info = read_raw(path).info

        if info is None or info['dig'] is None:
            QMessageBox.critical(
                self.parent, "Error Reading Digitization File",
                "The selected file does not contain digitization "
                "information. Please select a different file.")
            self.file = ''
            return None

        point_kinds = {d['kind'] for d in info['dig']}
        missing = [key for key in ('LPA', 'Nasion', 'RPA') if
                   getattr(FIFF, f'FIFFV_POINT_{key.upper()}') not in
                   point_kinds]
        if missing:
            pts = _fiducial_coords(info['dig'])
            if len(pts) == 3:
                _append_fiducials(info['dig'], *pts.T)
            else:
                QMessageBox.critical(
                    self.parent, "Error Reading Digitization File",
                    "The selected digitization file does not contain "
                    f"all cardinal points (missing: {', '.join(missing)}). "
                    "Please select a different file.")
                self.file = ''
                return None
        return info

    def _update_from_info(self):
        info = self._info
        empty = np.empty((0, 3))
        zeros = np.zeros((1, 3))

        if not info or not info['dig']:
            self._hsp_points = empty
            self.lpa = zeros
            self.nasion = zeros
            self.rpa = zeros
            self.eeg_points = empty
            self.hpi_points = np.zeros((0, 3))
        else:
            dig = info['dig']
            hsp = np.array([d['r'] for d in dig
                            if d['kind'] == FIFF.FIFFV_POINT_EXTRA])
            self._hsp_points = hsp if len(hsp) else empty

            self.lpa = self._cardinal_point(FIFF.FIFFV_POINT_LPA)
            self.nasion = self._cardinal_point(FIFF.FIFFV_POINT_NASION)
            self.rpa = self._cardinal_point(FIFF.FIFFV_POINT_RPA)

            eeg = [d['r'] for d in dig
                   if d['kind'] == FIFF.FIFFV_POINT_EEG and
                   d['coord_frame'] == FIFF.FIFFV_COORD_HEAD]
            self.eeg_points = np.array(eeg) if eeg else empty

            hpi = [d['r'] for d in dig
                   if d['kind'] == FIFF.FIFFV_POINT_HPI and
                   d['coord_frame'] == FIFF.FIFFV_COORD_HEAD]
            self.hpi_points = np.array(hpi) if hpi else np.zeros((0, 3))

        self._update_points()

    def _update_points(self):
        filt = self.points_filter
        hsp = self._hsp_points
        if filt is None:
            self.points = hsp
            self.n_omitted = 0
        else:
            self.points = hsp[filt]
            self.n_omitted = int(np.sum(filt == False))  # noqa: E712

    @observe('points_filter')
    def _points_filter_changed(self, change):
        self._update_points()

    def _cardinal_point(self, ident):
        if not self._info or not self._info['dig']:
            return np.zeros((1, 3))
        for d in self._info['dig']:
            if (d['kind'] == FIFF.FIFFV_POINT_CARDINAL and
                    d['ident'] == ident):
                return d['r'][None, :]
        return np.zeros((1, 3))


def _append_fiducials(dig, lpa, nasion, rpa):
    dig.append({'coord_frame': FIFF.FIFFV_COORD_HEAD,
                'ident': FIFF.FIFFV_POINT_LPA,
                'kind': FIFF.FIFFV_POINT_CARDINAL,
                'r': lpa})
    dig.append({'coord_frame': FIFF.FIFFV_COORD_HEAD,
                'ident': FIFF.FIFFV_POINT_NASION,
                'kind': FIFF.FIFFV_POINT_CARDINAL,
                'r': nasion})
    dig.append({'coord_frame': FIFF.FIFFV_COORD_HEAD,
                'ident': FIFF.FIFFV_POINT_RPA,
                'kind': FIFF.FIFFV_POINT_CARDINAL,
                'r': rpa})


class MRISubjectSource(HasTraits):
    """Find subjects in SUBJECTS_DIR and select one.

    Parameters
    ----------
    subjects_dir : str
        SUBJECTS_DIR path.
    subject : str
        Subject, corresponding to a folder in SUBJECTS_DIR.
    """

    subjects_dir = Unicode()
    subject = Unicode()
    subjects = List()  # list of str

    can_create_fsaverage = Bool()
    subject_has_bem = Bool()
    mri_dir = Unicode()
    parent = Any()  # QWidget | None, for parenting dialogs

    @observe('subjects_dir')
    def _subjects_dir_changed(self, change):
        self._update_subjects()
        # Re-emit subject so downstream observers on 'subject' fire even if
        # the value string didn't change (e.g. same name, new dir).
        self.subject = self.subject

    def _update_subjects(self):
        sdir = Path(self.subjects_dir) if self.subjects_dir else None
        if sdir is not None and sdir.is_dir():
            subjects = [s.name for s in sdir.iterdir()
                       if _is_mri_subject(s.name, sdir)]
            if not subjects:
                subjects = ['']
        else:
            subjects = ['']
        self.subjects = sorted(subjects)
        self.can_create_fsaverage = (
            sdir is not None and sdir.exists() and
            'fsaverage' not in self.subjects
        )

    @observe('subjects_dir', 'subject')
    def _update_bem_and_mri_dir(self, change):
        sdir = self.subjects_dir
        sub = self.subject
        self.mri_dir = str(Path(sdir) / sub) if sdir and sub else ''
        self.subject_has_bem = (
            bool(sub) and _mri_subject_has_bem(sub, sdir)
        )

    def refresh(self):
        """Refresh the subject list based on subjects_dir contents."""
        self._update_subjects()

    def create_fsaverage(self):  # noqa: D102
        if not self.subjects_dir:
            raise RuntimeError(
                "No subjects directory is selected. Please specify "
                "subjects_dir first.")

        fs_home = get_fs_home(parent=self.parent)
        if fs_home is None:
            raise RuntimeError(
                "FreeSurfer contains files that are needed for copying the "
                "fsaverage brain. Please install FreeSurfer and try again.")

        create_default_subject(fs_home=fs_home, update=True,
                               subjects_dir=self.subjects_dir)
        self.refresh()
        self.subject = 'fsaverage'


class SubjectSelectorPanel:
    """Subject selector (model-facing helper, not a Qt widget).

    Wraps an :class:`MRISubjectSource` and exposes its properties directly.
    The Qt widget that drives these controls lives in the GUI layer.
    """

    def __init__(self, model, parent=None):
        self.model = model
        self.parent = parent
        self.model.parent = parent

    # -- passthrough properties -------------------------------------------------

    @property
    def subjects_dir(self):
        return self.model.subjects_dir

    @subjects_dir.setter
    def subjects_dir(self, value):
        self.model.subjects_dir = value
        if value and self.model.subjects == ['']:
            QMessageBox.information(
                self.parent, "No Subjects Found",
                "The directory selected as subjects-directory "
                "(%s) does not contain any valid MRI subjects. If "
                "this is not expected make sure all MRI subjects have "
                "head surface model files which "
                "can be created by running:\n\n    $ mne "
                "make_scalp_surfaces" % value)

    @property
    def subject(self):
        return self.model.subject

    @subject.setter
    def subject(self, value):
        self.model.subject = value

    @property
    def subjects(self):
        return self.model.subjects

    @property
    def can_create_fsaverage(self):
        return self.model.can_create_fsaverage

    def create_fsaverage(self):
        """Copy fsaverage to SUBJECTS_DIR with a progress dialog."""
        prog = QProgressDialog("Copying fsaverage files ...",
                               None, 0, 0, self.parent)
        prog.setWindowTitle("Creating FsAverage ...")
        prog.setWindowModality(Qt.WindowModal)
        prog.setMinimumDuration(0)
        prog.show()
        try:
            self.model.create_fsaverage()
        except Exception as err:
            QMessageBox.critical(
                self.parent, "Error Creating FsAverage", str(err))
            raise
        finally:
            prog.close()
