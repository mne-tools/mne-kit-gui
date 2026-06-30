"""Traitlets/Qt GUI for averaging two sets of KIT marker points."""

# Authors: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD-3-Clause

import datetime
from pathlib import Path

import numpy as np

from qtpy.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QVBoxLayout,
)

from traitlets import Bool, Float, HasTraits, Any, List, Unicode, observe

from mne.transforms import apply_trans, rotation, translation
from mne.coreg import fit_matched_points
from mne.io.kit import read_mrk

from ._viewer import PointObject


mrk_wildcard = (
    "Supported Files (*.sqd *.mrk *.txt *.pickled);;"
    "Sqd marker file (*.sqd *.mrk);;"
    "Text marker file (*.txt);;"
    "Pickled markers (*.pickled)"
)
mrk_out_wildcard = "Tab separated values file (*.txt)"
out_ext = ".txt"


class ReorderDialog(QDialog):
    """Dialog for entering a new marker point order."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Reorder Marker Points")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("New order (five space-delimited numbers 0–4):"))
        self._edit = QLineEdit("0 1 2 3 4")
        layout.addWidget(self._edit)
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=self
        )
        buttons.accepted.connect(self._try_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _try_accept(self):
        if self.index is not None:
            self.accept()
        else:
            QMessageBox.warning(
                self, "Invalid Input", "Enter exactly five distinct numbers 0–4."
            )

    @property
    def index(self):
        try:
            idx = [int(i) for i in self._edit.text().split()]
        except ValueError:
            return None
        if sorted(idx) == [0, 1, 2, 3, 4]:
            return idx
        return None


class MarkerPoints(HasTraits):
    """Represent 5 marker points."""

    points = Any()  # ndarray (5, 3)
    name = Unicode()
    dir = Unicode()
    parent = Any()  # QWidget | None, for parenting dialogs

    can_save = Bool()

    def __init__(self, **kwargs):
        if "points" not in kwargs:
            kwargs["points"] = np.zeros((5, 3))
        super().__init__(**kwargs)

    @observe("points")
    def _points_changed(self, change):
        pts = change["new"]
        self.can_save = bool(pts is not None and np.any(pts))

    def save_as(self, parent=None):
        """Prompt user for a save path and save the marker points."""
        path, _ = QFileDialog.getSaveFileName(
            parent, "Save Markers", self.name or "", mrk_out_wildcard
        )
        if not path:
            return

        path = Path(path)
        if path.suffix != out_ext:
            path = path.with_suffix(out_ext)

        if path.exists():
            reply = QMessageBox.question(
                parent,
                "Overwrite File?",
                "The file %r already exists. Should it be replaced?" % str(path),
            )
            if reply != QMessageBox.Yes:
                return
        self.save(path)

    def save(self, path):
        """Save the marker points.

        Parameters
        ----------
        path : str
            Path to the file. Extension '.txt' writes a tab-separated file.
        """
        _write_dig_points(path, self.points)


class MarkerPointSource(MarkerPoints):  # noqa: D401
    """MarkerPoints subclass for source files."""

    file = Unicode()
    use = List()  # list of ints 0-4
    enabled = Bool()

    def __init__(self, **kwargs):
        if "use" not in kwargs:
            kwargs["use"] = list(range(5))
        super().__init__(**kwargs)

    @observe("file")
    def _file_changed(self, change):
        fname = change["new"]
        if fname:
            path = Path(fname)
            self.name = path.name
            self.dir = str(path.parent)
        else:
            self.name = ""
            self.dir = ""
        self._load(fname)

    def _load(self, fname):
        if not fname:
            self.points = np.zeros((5, 3))
            return
        try:
            pts = read_mrk(fname)
        except Exception as err:
            QMessageBox.critical(self.parent, "Error Reading mrk", str(err))
            self.points = np.zeros((5, 3))
        else:
            self.points = pts

    @observe("points", "use")
    def _update_enabled(self, change):
        self.enabled = bool(np.any(self.points))

    def clear(self):
        """Clear all marker data."""
        self.file = ""
        self.points = np.zeros((5, 3))
        self.use = list(range(5))

    def edit(self, parent=None):
        """Open an edit dialog for manual coordinate entry."""
        # Phase 1: placeholder — will be a proper QDialog in the Qt layer
        pass

    def reorder(self, parent=None):
        """Prompt for a new point order and apply it."""
        dlg = ReorderDialog(parent)
        if dlg.exec_() == QDialog.Accepted and dlg.index is not None:
            self.points = self.points[dlg.index]

    def switch_left_right(self):
        """Swap left and right marker points."""
        self.points = self.points[[1, 0, 2, 4, 3]]


class MarkerPointDest(MarkerPoints):  # noqa: D401
    """MarkerPoints subclass that serves for derived/interpolated points."""

    src1 = Any()  # MarkerPointSource
    src2 = Any()  # MarkerPointSource
    method = Unicode("Transform")
    enabled = Bool()

    def __init__(self, src1=None, src2=None, **kwargs):
        super().__init__(**kwargs)
        self.src1 = src1
        self.src2 = src2

    @observe("src1")
    def _src1_changed(self, change):
        old = change["old"]
        new = change["new"]
        if old is not None:
            old.unobserve(
                self._src_attr_changed,
                names=["points", "use", "name", "dir", "enabled"],
            )
        if new is not None:
            new.observe(
                self._src_attr_changed,
                names=["points", "use", "name", "dir", "enabled"],
            )
        self._recompute()

    @observe("src2")
    def _src2_changed(self, change):
        old = change["old"]
        new = change["new"]
        if old is not None:
            old.unobserve(
                self._src_attr_changed,
                names=["points", "use", "name", "dir", "enabled"],
            )
        if new is not None:
            new.observe(
                self._src_attr_changed,
                names=["points", "use", "name", "dir", "enabled"],
            )
        self._recompute()

    def _src_attr_changed(self, change):
        """React to any change on src1 or src2."""
        if change["name"] in ("name", "dir"):
            self._update_name_dir()
        else:
            self._recompute()

    def _update_name_dir(self):
        n1 = self.src1.name if self.src1 else ""
        n2 = self.src2.name if self.src2 else ""
        self.dir = self.src1.dir if self.src1 else ""

        if not n1:
            self.name = n2
        elif not n2:
            self.name = n1
        elif n1 == n2:
            self.name = n1
        else:
            i = 0
            while i < min(len(n1), len(n2)) and n1[i] == n2[i]:
                i += 1
            self.name = n1[:i]

    @observe("method")
    def _method_changed(self, change):
        self._recompute()

    def _recompute(self):
        self.points = self._compute_points()
        self.enabled = bool(self.points is not None and np.any(self.points))

    def _compute_points(self):
        src1, src2 = self.src1, self.src2

        if not (src1 and src1.enabled):
            if src2 and src2.enabled:
                return src2.points
            return np.zeros((5, 3))
        elif not (src2 and src2.enabled):
            return src1.points

        if self.method == "Average":
            return self._compute_points_average(src1, src2)
        return self._compute_points_transform(src1, src2)

    def _compute_points_average(self, src1, src2):
        if len(np.union1d(src1.use, src2.use)) < 5:
            QMessageBox.critical(
                self.parent,
                "Marker Average Error",
                "Need at least one source for each point.",
            )
            return np.zeros((5, 3))
        pts = (src1.points + src2.points) / 2.0
        for i in np.setdiff1d(src1.use, src2.use):
            pts[i] = src1.points[i]
        for i in np.setdiff1d(src2.use, src1.use):
            pts[i] = src2.points[i]
        return pts

    def _compute_points_transform(self, src1, src2):
        idx = np.intersect1d(np.array(src1.use), np.array(src2.use), assume_unique=True)
        if len(idx) < 3:
            QMessageBox.critical(
                self.parent,
                "Marker Interpolation Error",
                "Need at least three shared points for transformation.",
            )
            return np.zeros((5, 3))

        src_pts = src1.points[idx]
        tgt_pts = src2.points[idx]
        est = fit_matched_points(src_pts, tgt_pts, out="params")
        rot = np.array(est[:3]) / 2.0
        tra = np.array(est[3:]) / 2.0

        if len(src1.use) == 5:
            trans = np.dot(translation(*tra), rotation(*rot))
            return apply_trans(trans, src1.points)
        elif len(src2.use) == 5:
            trans = np.dot(translation(*-tra), rotation(*-rot))
            return apply_trans(trans, src2.points)
        else:
            trans1 = np.dot(translation(*tra), rotation(*rot))
            pts = apply_trans(trans1, src1.points)
            trans2 = np.dot(translation(*-tra), rotation(*-rot))
            for i in np.setdiff1d(src2.use, src1.use):
                pts[i] = apply_trans(trans2, src2.points[i])
            return pts


class CombineMarkersModel(HasTraits):
    """Combine markers model."""

    mrk1 = Any()  # MarkerPointSource
    mrk2 = Any()  # MarkerPointSource
    mrk3 = Any()  # MarkerPointDest
    distance = Unicode()
    parent = Any()  # QWidget | None, for parenting dialogs

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.mrk1 is None:
            self.mrk1 = MarkerPointSource()
        if self.mrk2 is None:
            self.mrk2 = MarkerPointSource()
        if self.mrk3 is None:
            self.mrk3 = MarkerPointDest(src1=self.mrk1, src2=self.mrk2)
        for sub_model in (self.mrk1, self.mrk2, self.mrk3):
            sub_model.parent = self.parent
        self.mrk1.observe(self._update_distance, names=["points"])
        self.mrk2.observe(self._update_distance, names=["points"])

    @observe("parent")
    def _parent_changed(self, change):
        for sub_model in (self.mrk1, self.mrk2, self.mrk3):
            sub_model.parent = change["new"]

    def _update_distance(self, change=None):
        if not np.any(self.mrk1.points) or not np.any(self.mrk2.points):
            self.distance = ""
            return
        ds = np.sqrt(np.sum((self.mrk1.points - self.mrk2.points) ** 2, 1))
        self.distance = "\t".join("%.1f mm" % (d * 1000) for d in ds)

    def clear(self):
        """Clear all marker data."""
        self.mrk1.clear()
        self.mrk2.clear()
        self.mrk3.method = "Transform"


class CombineMarkersPanel(HasTraits):  # noqa: D401
    """Has two marker point sources and interpolates to a third one."""

    model = Any()  # CombineMarkersModel

    # model mirrors for convenient access
    mrk1 = Any()
    mrk2 = Any()
    mrk3 = Any()
    distance = Unicode()

    # Visualization
    scene = Any()  # pyvistaqt.QtInteractor
    scale = Float(5e-3)
    mrk1_obj = Any()  # PointObject
    mrk2_obj = Any()  # PointObject
    mrk3_obj = Any()  # PointObject
    trans = Any()  # ndarray (4, 4)
    parent = Any()  # QWidget | None, for parenting dialogs

    def __init__(self, **kwargs):
        if "model" not in kwargs:
            kwargs["model"] = CombineMarkersModel()
        if "trans" not in kwargs:
            kwargs["trans"] = np.eye(4)
        super().__init__(**kwargs)

        model = self.model
        model.parent = self.parent
        self.mrk1 = model.mrk1
        self.mrk2 = model.mrk2
        self.mrk3 = model.mrk3

        # Sync distance from model
        model.observe(
            lambda ch: setattr(self, "distance", ch["new"]), names=["distance"]
        )

        # Visualization objects
        self.mrk1_obj = PointObject(
            scene=self.scene, color=(0.608, 0.216, 0.216), point_scale=self.scale
        )
        self.mrk2_obj = PointObject(
            scene=self.scene, color=(0.216, 0.608, 0.216), point_scale=self.scale
        )
        self.mrk3_obj = PointObject(
            scene=self.scene, color=(0.588, 0.784, 1.0), point_scale=self.scale
        )

        # Wire visibility from 'enabled' on each source
        model.mrk1.observe(
            lambda ch: setattr(self.mrk1_obj, "visible", ch["new"]), names=["enabled"]
        )
        model.mrk2.observe(
            lambda ch: setattr(self.mrk2_obj, "visible", ch["new"]), names=["enabled"]
        )
        model.mrk3.observe(
            lambda ch: setattr(self.mrk3_obj, "visible", ch["new"]), names=["enabled"]
        )

        # Wire point updates
        model.mrk1.observe(self._update_mrk1, names=["points"])
        model.mrk2.observe(self._update_mrk2, names=["points"])
        model.mrk3.observe(self._update_mrk3, names=["points"])

    @observe("trans")
    def _trans_changed(self, change):
        self._update_mrk1()
        self._update_mrk2()
        self._update_mrk3()

    @observe("parent")
    def _parent_changed(self, change):
        self.model.parent = change["new"]

    def _update_mrk1(self, change=None):
        if self.mrk1_obj is not None:
            self.mrk1_obj.points = apply_trans(self.trans, self.model.mrk1.points)

    def _update_mrk2(self, change=None):
        if self.mrk2_obj is not None:
            self.mrk2_obj.points = apply_trans(self.trans, self.model.mrk2.points)

    def _update_mrk3(self, change=None):
        if self.mrk3_obj is not None:
            self.mrk3_obj.points = apply_trans(self.trans, self.model.mrk3.points)


def _write_dig_points(fname, dig_points):
    """Write points to text file.

    Parameters
    ----------
    fname : path-like
        Path to the file to write. The kind of file to write is determined
        based on the extension: '.txt' for tab separated text file.
    dig_points : numpy.ndarray, shape (n_points, 3)
        Points.
    """
    from mne import __version__

    ext = Path(fname).suffix
    dig_points = np.asarray(dig_points)
    if (dig_points.ndim != 2) or (dig_points.shape[1] != 3):
        err = "Points must be of shape (n_points, 3), not %s" % (dig_points.shape,)
        raise ValueError(err)

    if ext == ".txt":
        with open(fname, "wb") as fid:
            version = __version__
            now = datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")
            fid.write(
                b"%% Ascii 3D points file created by mne-python version"
                b" %s at %s\n" % (version.encode(), now.encode())
            )
            fid.write(b"%% %d 3D points, x y z per line\n" % len(dig_points))
            np.savetxt(fid, dig_points, delimiter="\t", newline="\n")
    else:
        msg = "Unrecognized extension: %r. Need '.txt'." % ext
        raise ValueError(msg)
