# -*- coding: utf-8 -*-
"""Traitlets/Qt GUI for setting MRI fiducials."""

# Authors: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD-3-Clause

from pathlib import Path

import numpy as np

from qtpy.QtWidgets import (
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)
from qtpy.QtCore import Qt

from traitlets import Bool, HasTraits, Any, Unicode, observe

from mne.coreg import (
    fid_fname,
    _find_fiducials_files,
    _find_head_bem,
    get_mni_fiducials,
)
from mne.defaults import DEFAULTS
from mne.io import write_fiducials
from mne.io.constants import FIFF
from mne.surface import complete_surface_info, decimate_surface
from mne.utils import get_subjects_dir, logger, warn

from ._file_traits import (
    SurfaceSource,
    FiducialsSource,
    MRISubjectSource,
    SubjectSelectorPanel,
)
from ._viewer import (
    HeadViewController,
    PointObject,
    SurfaceObject,
    build_head_view_group,
    embed_pyvista_scene,
)

defaults = DEFAULTS["coreg"]

_VIEW_DICT = {"lpa": "left", "nasion": "front", "rpa": "right"}

_SET_TOOLTIP = "Click on the MRI image to set the position, or enter values below"


class MRIHeadWithFiducialsModel(HasTraits):
    """Represent an MRI head shape (high and low res) with fiducials.

    Attributes
    ----------
    lpa, nasion, rpa : ndarray (1, 3)
        Fiducial coordinates in MRI space (meters).
    """

    subject_source = Any()  # MRISubjectSource
    bem_low_res = Any()  # SurfaceSource
    bem_high_res = Any()  # SurfaceSource
    fid = Any()  # FiducialsSource

    # delegated for convenience
    subjects_dir = Unicode()
    subject = Unicode()
    subject_has_bem = Bool()
    fid_file = Unicode()
    fid_fname = Unicode()
    fid_points = Any()  # ndarray (3, 3) or None

    lpa = Any()  # ndarray (1, 3)
    nasion = Any()  # ndarray (1, 3)
    rpa = Any()  # ndarray (1, 3)

    can_save = Bool()
    can_save_as = Bool()
    can_reset = Bool()
    fid_ok = Bool()
    default_fid_fname = Unicode()
    lock_fiducials = Bool(False)

    parent = Any()  # QWidget | None, for parenting dialogs

    def __init__(self, **kwargs):
        # subjects_dir/subject/parent trigger observers that need the
        # sub-models to already exist -- defer applying them until after
        # the defaults below are set up.
        subjects_dir = kwargs.pop("subjects_dir", None)
        subject = kwargs.pop("subject", None)
        parent = kwargs.pop("parent", None)
        super().__init__(**kwargs)
        self._init_sub_models()
        self._wire_observers()
        self._apply_deferred(subjects_dir, subject, parent)

    def _init_sub_models(self):
        # Sub-models must exist before lpa/nasion/rpa are assigned below,
        # since those assignments fire observers that read self.fid etc.
        if self.subject_source is None:
            self.subject_source = MRISubjectSource()
        if self.bem_low_res is None:
            self.bem_low_res = SurfaceSource()
        if self.bem_high_res is None:
            self.bem_high_res = SurfaceSource()
        if self.fid is None:
            self.fid = FiducialsSource()
        zeros = np.zeros((1, 3))
        if self.lpa is None:
            self.lpa = zeros.copy()
        if self.nasion is None:
            self.nasion = zeros.copy()
        if self.rpa is None:
            self.rpa = zeros.copy()

    def _wire_observers(self):
        # Wire subject_source delegations
        self.subject_source.observe(self._src_dir_changed, names=["subjects_dir"])
        self.subject_source.observe(self._src_subject_changed, names=["subject"])
        self.subject_source.observe(
            lambda ch: setattr(self, "subject_has_bem", ch["new"]),
            names=["subject_has_bem"],
        )

        # Wire fid delegations (bidirectional: fid.file <-> fid_file)
        self.fid.observe(self._on_fid_file_changed, names=["file"])
        self.fid.observe(
            lambda ch: setattr(self, "fid_fname", ch["new"]), names=["fname"]
        )
        self.fid.observe(self._on_fid_points_changed, names=["points"])

    def _apply_deferred(self, subjects_dir, subject, parent):
        if subjects_dir is not None:
            self.subjects_dir = subjects_dir
        if subject is not None:
            self.subject = subject
        if parent is not None:
            self.parent = parent

    @observe("parent")
    def _parent_changed(self, change):
        for sub_model in (
            self.subject_source,
            self.bem_low_res,
            self.bem_high_res,
            self.fid,
        ):
            sub_model.parent = change["new"]

    def _on_fid_file_changed(self, change):
        self.fid_file = change["new"]

    @observe("fid_file")
    def _fid_file_changed(self, change):
        if self.fid.file != change["new"]:
            self.fid.file = change["new"]

    def _src_dir_changed(self, change):
        self.subjects_dir = change["new"]
        self._update_default_fid_fname()

    def _src_subject_changed(self, change):
        self.subject = change["new"]
        self._update_default_fid_fname()

    def _update_default_fid_fname(self):
        sdir = self.subjects_dir
        sub = self.subject
        if sdir and sub:
            self.default_fid_fname = fid_fname.format(subjects_dir=sdir, subject=sub)
        else:
            self.default_fid_fname = ""

    @observe("lpa", "nasion", "rpa")
    def _fiducials_changed(self, change):
        self._update_can_flags()

    def _update_can_flags(self):
        nas, lpa, rpa = self.nasion, self.lpa, self.rpa
        can_save_as = not (
            np.all(nas == lpa) or np.all(nas == rpa) or np.all(lpa == rpa)
        )
        self.can_save_as = can_save_as
        self.can_save = can_save_as and bool(
            self.fid_file or (self.subjects_dir and self.subject)
        )
        self.fid_ok = all(np.any(pt) for pt in (nas, lpa, rpa))
        self._update_can_reset()

    def _update_can_reset(self):
        fp = self.fid.points
        if not self.fid_file or fp is None:
            self.can_reset = False
            return
        self.can_reset = bool(
            np.any(self.lpa != fp[0:1])
            or np.any(self.nasion != fp[1:2])
            or np.any(self.rpa != fp[2:3])
        )

    def _on_fid_points_changed(self, change):
        self.fid_points = change["new"]
        self.reset_fiducials()

    def reset_fiducials(self):
        """Reset fiducial positions from the loaded fid file."""
        fp = self.fid_points
        if fp is not None:
            self.lpa = fp[0:1]
            self.nasion = fp[1:2]
            self.rpa = fp[2:3]

    def save(self, fname=None):
        """Save the current fiducials to a file."""
        if fname is None:
            fname = self.fid_file
        if not fname:
            fname = self.default_fid_fname

        dig = [
            {
                "kind": FIFF.FIFFV_POINT_CARDINAL,
                "ident": FIFF.FIFFV_POINT_LPA,
                "r": np.array(self.lpa[0]),
            },
            {
                "kind": FIFF.FIFFV_POINT_CARDINAL,
                "ident": FIFF.FIFFV_POINT_NASION,
                "r": np.array(self.nasion[0]),
            },
            {
                "kind": FIFF.FIFFV_POINT_CARDINAL,
                "ident": FIFF.FIFFV_POINT_RPA,
                "r": np.array(self.rpa[0]),
            },
        ]
        write_fiducials(fname, dig, FIFF.FIFFV_COORD_MRI)
        self.fid.file = str(fname)

    def load_subject(self, subject=None, subjects_dir=None):
        """Load head surface and fiducial info for the given subject."""
        if subject is None:
            subject = self.subject
        if subjects_dir is None:
            subjects_dir = self.subjects_dir
        if not subjects_dir or not subject:
            return

        high_res_path = _find_head_bem(subject, subjects_dir, high_res=True)
        low_res_path = _find_head_bem(subject, subjects_dir, high_res=False)
        if high_res_path is None and low_res_path is None:
            msg = "No standard head model was found for subject %s" % subject
            QMessageBox.critical(self.parent, "No head surfaces found", msg)
            raise RuntimeError(msg)
        if high_res_path is not None:
            self.bem_high_res.file = high_res_path
        else:
            self.bem_high_res.file = low_res_path

        if low_res_path is None:
            warn(
                "No low-resolution head found, decimating high resolution "
                "mesh (%d vertices): %s"
                % (len(self.bem_high_res.surf["rr"]), high_res_path)
            )
            rr, tris = decimate_surface(
                self.bem_high_res.surf["rr"],
                self.bem_high_res.surf["tris"],
                n_triangles=5120,
            )
            surf = complete_surface_info(
                {"rr": rr, "tris": tris}, copy=False, verbose=False
            )
            self.bem_low_res.surf = {"rr": surf["rr"], "tris": surf["tris"]}
        else:
            self.bem_low_res.file = low_res_path

        try:
            fids = get_mni_fiducials(subject, subjects_dir)
        except Exception:
            self.fid.mni_points = None
        else:
            self.fid.mni_points = np.array([f["r"] for f in fids], float)

        fid_files = _find_fiducials_files(subject, subjects_dir)
        if len(fid_files) == 0:
            self.fid.file = ""
            self.lock_fiducials = False
        else:
            self.fid.file = fid_files[0].format(
                subjects_dir=subjects_dir, subject=subject
            )
            self.lock_fiducials = True

        self.reset_fiducials()

    @observe("subjects_dir", "subject")
    def _on_subject_changed(self, change):
        """React to subject/subjects_dir changes coming from this object."""
        # Also forward to subject_source so it stays in sync
        ss = self.subject_source
        if change["name"] == "subjects_dir" and ss.subjects_dir != change["new"]:
            ss.subjects_dir = change["new"]
        elif change["name"] == "subject" and ss.subject != change["new"]:
            ss.subject = change["new"]
        if self.subjects_dir and self.subject:
            try:
                self.load_subject()
            except RuntimeError:
                pass
            self._update_default_fid_fname()


class FiducialsPanel(HasTraits):
    """Set fiducials on an MRI surface (model/controller layer)."""

    model = Any()  # MRIHeadWithFiducialsModel
    headview = Any()  # HeadViewController
    hsp_obj = Any()  # SurfaceObject

    set = Unicode("LPA")  # 'LPA' | 'Nasion' | 'RPA'
    current_pos_mm = Any()  # ndarray (1, 3) in mm

    picker = Any()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.current_pos_mm is None:
            self.current_pos_mm = np.zeros((1, 3))
        if self.model is not None:
            self.model.observe(
                self._on_model_fid_changed, names=["lpa", "nasion", "rpa"]
            )

    def _on_model_fid_changed(self, change):
        """Update current_pos_mm when the active fiducial changes."""
        attr = self.set.lower()
        val = getattr(self.model, attr)
        if val is not None:
            self.current_pos_mm = val * 1000

    @observe("current_pos_mm")
    def _pos_mm_changed(self, change):
        attr = self.set.lower()
        new_m = change["new"] * 1e-3
        if not np.allclose(getattr(self.model, attr), new_m):
            setattr(self.model, attr, new_m)

    @observe("set")
    def _set_changed(self, change):
        new = change["new"].lower()
        self._on_model_fid_changed(None)
        if self.headview is not None:
            self.headview.on_set_view(_VIEW_DICT[new])

    def save_as(self):
        """Prompt for a path and save fiducials."""
        parent = self.model.parent
        default = self.model.fid_file or self.model.default_fid_fname
        path, _ = QFileDialog.getSaveFileName(
            parent, "Save Fiducials", default, "Fiducials (*.fif)"
        )
        if not path:
            return
        path = Path(path)
        if path.suffix != ".fif":
            path = path.with_name(path.name + ".fif")
        if path.exists():
            reply = QMessageBox.question(
                parent,
                "Overwrite File?",
                "The file %r already exists. Replace it?" % str(path),
            )
            if reply != QMessageBox.Yes:
                return
        self.model.save(str(path))

    def _on_pick(self, point, picker):
        """Handle a pyvista pick event — position the active fiducial."""
        if self.model.lock_fiducials:
            return

        self.picker = picker
        if point is None:
            logger.debug("GUI: picked empty location")
            return

        hsp_surf = self.hsp_obj.surf if self.hsp_obj else None
        if hsp_surf is None or picker.GetActor() is not hsp_surf:
            logger.debug("GUI: picked object other than MRI")
            return

        set_ = self.set.lower()
        setattr(self.model, set_, np.atleast_2d(point))


class FiducialsFrame(QMainWindow):
    """Qt window for setting MRI fiducials with a pyvista 3D scene.

    Parameters
    ----------
    subject : None | str
        Subject to select initially.
    subjects_dir : None | str
        Override the SUBJECTS_DIR environment variable.
    """

    def __init__(self, subject=None, subjects_dir=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Set MRI Fiducials")
        self.setAttribute(Qt.WA_DeleteOnClose, True)

        # --- model layer ---
        self.model = MRIHeadWithFiducialsModel(parent=self)

        # pyvista scene embedded directly as a QWidget.
        self._scene_widget = QWidget(self)
        QVBoxLayout(self._scene_widget)
        self.scene = embed_pyvista_scene(self._scene_widget)

        self.headview = HeadViewController(scene=self.scene, system="RAS")
        self.panel = FiducialsPanel(model=self.model, headview=self.headview)
        self.spanel = SubjectSelectorPanel(model=self.model.subject_source, parent=self)

        point_scale = float(defaults["mri_fid_scale"])
        self.point_scale = point_scale
        self.mri_obj = None
        self.lpa_obj = None
        self.nasion_obj = None
        self.rpa_obj = None

        # Build Qt UI
        self._build_ui()

        # Load subjects_dir / subject if provided
        subjects_dir = get_subjects_dir(subjects_dir)
        if subjects_dir:
            self.spanel.subjects_dir = str(subjects_dir)
        if subject and subject in self.spanel.subjects:
            self.spanel.subject = subject

        self._init_plot()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        central = QWidget(self)
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # Left: 3D scene
        main_layout.addWidget(self._scene_widget, stretch=3)

        # Right: controls panel
        ctrl = QWidget()
        ctrl_layout = QVBoxLayout(ctrl)
        main_layout.addWidget(ctrl, stretch=1)

        # Head view controls (compass buttons + scale + interaction)
        ctrl_layout.addWidget(build_head_view_group(self.headview))

        # Subject selector
        sg = QGroupBox("Subject")
        sg_layout = QVBoxLayout(sg)
        from qtpy.QtWidgets import QComboBox, QLineEdit

        source = self.model.subject_source

        # subjects_dir row: a path field (kept in sync with the model) + browse
        dir_row = QHBoxLayout()
        self._subjects_dir_edit = QLineEdit(source.subjects_dir)
        self._subjects_dir_edit.setPlaceholderText("SUBJECTS_DIR")
        self._subjects_dir_edit.editingFinished.connect(
            lambda: setattr(self.spanel, "subjects_dir", self._subjects_dir_edit.text())
        )
        source.observe(
            lambda ch: self._subjects_dir_edit.setText(ch["new"]),
            names=["subjects_dir"],
        )
        dir_browse = QPushButton("Browse")
        dir_browse.setObjectName("subjects_dir_browse")
        dir_browse.clicked.connect(self._browse_subjects_dir)
        dir_row.addWidget(self._subjects_dir_edit)
        dir_row.addWidget(dir_browse)
        sg_layout.addLayout(dir_row)

        self._subject_combo = QComboBox()
        self._subject_combo.currentTextChanged.connect(
            lambda s: setattr(self.spanel, "subject", s)
        )
        sg_layout.addWidget(self._subject_combo)

        # fsaverage creation, enabled only when fsaverage is not already present
        self._fsaverage_btn = QPushButton("fsaverage⇨SUBJECTS_DIR")
        self._fsaverage_btn.setObjectName("create_fsaverage")
        self._fsaverage_btn.clicked.connect(self._create_fsaverage)
        self._fsaverage_btn.setEnabled(source.can_create_fsaverage)
        source.observe(
            lambda ch: self._fsaverage_btn.setEnabled(ch["new"]),
            names=["can_create_fsaverage"],
        )
        sg_layout.addWidget(self._fsaverage_btn)

        ctrl_layout.addWidget(sg)
        # Keep combo populated when subjects list changes
        source.observe(self._update_subject_combo, names=["subjects"])
        source.observe(
            lambda ch: self._subject_combo.setCurrentText(ch["new"]), names=["subject"]
        )

        # Fiducial set picker
        fg = QGroupBox("Set fiducial")
        fg_layout = QVBoxLayout(fg)
        self._set_radios = {}
        for label in ("LPA", "Nasion", "RPA"):
            rb = QRadioButton(label)
            rb.toggled.connect(
                lambda checked, lbl=label: (
                    setattr(self.panel, "set", lbl) if checked else None
                )
            )
            fg_layout.addWidget(rb)
            self._set_radios[label] = rb
        self._set_radios["LPA"].setChecked(True)
        ctrl_layout.addWidget(fg)

        # Current position spinboxes (mm)
        pg = QGroupBox("Position (mm)")
        pg_layout = QHBoxLayout(pg)
        self._pos_spins = []
        for i, axis in enumerate("XYZ"):
            spin = QDoubleSpinBox()
            spin.setRange(-500, 500)
            spin.setDecimals(1)
            spin.setSingleStep(0.1)
            spin.valueChanged.connect(
                lambda v, idx=i: self._on_pos_spin_changed(idx, v)
            )
            pg_layout.addWidget(QLabel(axis))
            pg_layout.addWidget(spin)
            self._pos_spins.append(spin)
        ctrl_layout.addWidget(pg)
        self.panel.observe(self._update_pos_spins, names=["current_pos_mm"])

        # Save / reset buttons
        bg = QGroupBox("File")
        bg_layout = QVBoxLayout(bg)

        # fiducials file path (read-only, mirrors the model) + browse
        fid_row = QHBoxLayout()
        self._fid_file_edit = QLineEdit(self.model.fid_file)
        self._fid_file_edit.setObjectName("fid_file")
        self._fid_file_edit.setReadOnly(True)
        self._fid_file_edit.setPlaceholderText("Fiducials file...")
        self.model.observe(
            lambda ch: self._fid_file_edit.setText(ch["new"]), names=["fid_file"]
        )
        fid_browse = QPushButton("Browse")
        fid_browse.setObjectName("fid_file_browse")
        fid_browse.clicked.connect(self._browse_fid_file)
        fid_row.addWidget(self._fid_file_edit)
        fid_row.addWidget(fid_browse)
        bg_layout.addLayout(fid_row)

        self._save_btn = QPushButton("Save")
        self._save_btn.clicked.connect(lambda: self.model.save())
        self._save_as_btn = QPushButton("Save As...")
        self._save_as_btn.clicked.connect(lambda: self.panel.save_as())
        self._reset_btn = QPushButton("↻ Reset")
        self._reset_btn.clicked.connect(lambda: self.model.reset_fiducials())
        bg_layout.addWidget(self._save_btn)
        bg_layout.addWidget(self._save_as_btn)
        bg_layout.addWidget(self._reset_btn)
        ctrl_layout.addWidget(bg)

        # Update button enabled states
        self.model.observe(
            self._update_buttons, names=["can_save", "can_save_as", "can_reset"]
        )
        self._update_buttons()

        # The editing controls are disabled while the fiducials are locked
        self._lockable_groups = [fg, pg, bg]
        self.model.observe(self._update_lock_state, names=["lock_fiducials"])
        self._update_lock_state()

        ctrl_layout.addStretch()
        self.resize(900, 700)

    def _update_subject_combo(self, change):
        subjects = change["new"]
        self._subject_combo.blockSignals(True)
        self._subject_combo.clear()
        self._subject_combo.addItems(subjects)
        cur = self.spanel.subject
        if cur in subjects:
            self._subject_combo.setCurrentText(cur)
        self._subject_combo.blockSignals(False)

    def _browse_subjects_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Select SUBJECTS_DIR")
        if path:
            self.spanel.subjects_dir = path

    def _browse_fid_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Fiducials File", "", "Fiducials (*.fif)"
        )
        if path:
            self.model.fid_file = path

    def _update_lock_state(self, change=None):
        enabled = not self.model.lock_fiducials
        for group in self._lockable_groups:
            group.setEnabled(enabled)

    def _create_fsaverage(self):
        try:
            self.spanel.create_fsaverage()
        except Exception:
            pass  # spanel.create_fsaverage already reported the error dialog

    def _on_pos_spin_changed(self, idx, value):
        pos = self.panel.current_pos_mm.copy()
        pos[0, idx] = value
        self.panel.current_pos_mm = pos

    def _update_pos_spins(self, change):
        pos = change["new"]
        for i, spin in enumerate(self._pos_spins):
            spin.blockSignals(True)
            spin.setValue(float(pos[0, i]))
            spin.blockSignals(False)

    def _update_buttons(self, change=None):
        self._save_btn.setEnabled(bool(self.model.can_save))
        self._save_as_btn.setEnabled(bool(self.model.can_save_as))
        self._reset_btn.setEnabled(bool(self.model.can_reset))

    # ------------------------------------------------------------------
    # 3D scene initialisation
    # ------------------------------------------------------------------

    def _init_plot(self):
        color = defaults["head_color"]
        src = self.model.bem_low_res
        self.mri_obj = SurfaceObject(
            points=src.surf["rr"] if src.surf else np.empty((0, 3)),
            color=color,
            tris=src.surf["tris"] if src.surf else np.empty((0, 3), int),
            scene=self.scene,
        )
        self.mri_obj.plot()
        self.panel.hsp_obj = self.mri_obj

        # Watch for BEM surface changes
        self.model.bem_low_res.observe(self._on_mri_src_change, names=["surf"])

        # Fiducial point objects
        for key in ("lpa", "nasion", "rpa"):
            obj = PointObject(
                scene=self.scene,
                color=defaults[f"{key}_color"],
                has_norm=True,
                point_scale=self.point_scale,
            )
            setattr(self, f"{key}_obj", obj)

            # Update point object when model fiducial changes
            def _make_updater(o):
                def _upd(ch):
                    v = ch["new"]
                    if v is not None:
                        o.points = v

                return _upd

            self.model.observe(_make_updater(obj), names=[key])
            # Set initial position
            v = getattr(self.model, key)
            if v is not None:
                obj.points = v

        self.headview.on_set_view("left")

        # Mouse picking for fiducial placement: only the head surface
        # (added with pickable=True in SurfaceObject.plot) is pickable.
        self.scene.enable_surface_point_picking(
            callback=self.panel._on_pick,
            show_message=False,
            show_point=False,
            left_clicking=True,
            use_picker=True,
            pickable_window=False,
        )

    def _on_mri_src_change(self, change):
        surf = change["new"]
        if surf is None or not np.any(surf["tris"]):
            self.mri_obj.clear()
            return
        self.mri_obj.points = surf["rr"]
        self.mri_obj.tris = surf["tris"]
        self.mri_obj.plot()

    def closeEvent(self, event):
        """Prompt to save unsaved fiducial changes before closing."""
        if self.model.can_save:
            reply = QMessageBox.question(
                self,
                "Unsaved changes",
                "There are unsaved fiducial changes. Save before closing?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            )
            if reply == QMessageBox.Save:
                self.model.save()
            elif reply == QMessageBox.Cancel:
                event.ignore()
                return
        event.accept()
        self.scene.close()
