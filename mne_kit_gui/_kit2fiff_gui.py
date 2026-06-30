"""Traitlets/Qt GUI for converting data from KIT systems."""

# Authors: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD-3-Clause

from collections import Counter
import os
import queue
from pathlib import Path
from threading import Thread

import numpy as np

from qtpy.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from qtpy.QtCore import Qt

from traitlets import Bool, Float, HasTraits, Any, Int, List, Unicode, observe

from mne.channels import make_dig_montage
from mne.event import _find_events
from mne.io.constants import FIFF
from mne.io.kit.coreg import _read_dig_kit
from mne.io.kit.kit import (
    RawKIT,
    KIT,
    _make_stim_channel,
    _default_stim_chs,
    UnsupportedKITFormat,
)
from mne.transforms import (
    apply_trans,
    als_ras_trans,
    get_ras_to_neuromag_trans,
    Transform,
)
from mne.coreg import _decimate_points, fit_matched_points
from mne.utils import get_config, set_config, logger, warn

from ._marker_gui import CombineMarkersPanel, CombineMarkersModel
from ._help import read_tooltips
from ._viewer import HeadViewController, PointObject, embed_pyvista_scene


hsp_wildcard = "Head Shape Points (*.hsp *.txt)"
elp_wildcard = "Head Shape Fiducials (*.elp *.txt)"
kit_con_wildcard = "Continuous KIT Files (*.sqd *.con)"

tooltips = read_tooltips("kit2fiff")


class Kit2FiffModel(HasTraits):
    """Data model for Kit2Fiff conversion.

    Markers are transformed into RAS coordinate system (as are the sensor
    coordinates). Head shape digitizer data is transformed into neuromag-like
    space.
    """

    # --- inputs ---
    markers = Any()  # CombineMarkersModel
    sqd_file = Unicode()
    allow_unknown_format = Bool(False)
    hsp_file = Unicode()
    fid_file = Unicode()
    stim_coding = Unicode(">")
    stim_chs = Unicode("")
    stim_slope = Unicode("-")
    stim_threshold = Float(1.0)
    use_mrk = List()  # list of int 0-4
    show_gui = Bool(False)

    # --- level-1 computed (depend on one input) ---
    raw = Any()
    elp_raw = Any()  # ndarray (n, 3) or None
    hsp_raw = Any()  # ndarray (n, 3) or None
    mrk = Any()  # ndarray (5, 3)

    # --- level-2 computed ---
    misc_chs = Any()  # list of int
    misc_chs_desc = Unicode()
    misc_data = Any()  # ndarray or None
    can_test_stim = Bool()
    polhemus_neuromag_trans = Any()  # ndarray (4, 4) or None
    stim_chs_array = Any()  # ndarray or None
    stim_chs_ok = Bool()
    stim_chs_comment = Unicode()

    # --- level-3 computed ---
    elp = Any()  # ndarray (5, 3) or empty
    fid = Any()  # ndarray (3, 3) or empty
    hsp = Any()  # ndarray (n, 3) or empty

    # --- level-4 computed ---
    dev_head_trans = Any()  # ndarray (4, 4)
    head_dev_trans = Any()  # ndarray (4, 4)

    # --- filenames ---
    sqd_fname = Unicode("-")
    hsp_fname = Unicode("-")
    fid_fname = Unicode("-")

    # --- overall ---
    can_save = Bool()

    parent = Any()  # QWidget | None, for parenting dialogs

    def __init__(self, **kwargs):
        if "markers" not in kwargs:
            kwargs["markers"] = CombineMarkersModel()
        if "use_mrk" not in kwargs:
            kwargs["use_mrk"] = list(range(5))
        super().__init__(**kwargs)
        empty = np.empty((0, 3))
        eye = np.eye(4)
        self.mrk = np.zeros((5, 3))
        self.elp = empty
        self.fid = empty
        self.hsp = empty
        self.dev_head_trans = eye
        self.head_dev_trans = eye
        self.markers.parent = self.parent
        # Wire markers
        self.markers.mrk3.observe(self._mrk3_points_changed, names=["points"])
        self._recompute_misc()
        self._recompute_stim_chs_array()

    @observe("parent")
    def _parent_changed(self, change):
        self.markers.parent = change["new"]

    # ------------------------------------------------------------------
    # Observers
    # ------------------------------------------------------------------

    def _mrk3_points_changed(self, change):
        self.mrk = apply_trans(als_ras_trans, change["new"])
        self._recompute_dev_head_trans()
        self._update_can_save()

    @observe("sqd_file")
    def _sqd_file_changed(self, change):
        fname = change["new"]
        self.sqd_fname = Path(fname).name if fname else "-"
        self._recompute_raw()

    def _recompute_raw(self):
        fname = self.sqd_file
        if not fname:
            self.raw = None
        else:
            try:
                self.raw = RawKIT(
                    fname, stim=None, allow_unknown_format=self.allow_unknown_format
                )
            except UnsupportedKITFormat as exc:
                if self.show_gui:
                    QMessageBox.warning(
                        self.parent,
                        "Unsupported SQD File Format",
                        "The selected SQD file is written in an old file "
                        "format (%s) that is not officially supported. "
                        "Confirm that the results are as expected." % exc.sqd_version,
                    )
                self.allow_unknown_format = True
                self._recompute_raw()
                return
            except Exception as err:
                self.sqd_file = ""
                if self.show_gui:
                    QMessageBox.critical(
                        self.parent,
                        "Error Reading SQD File",
                        "Error reading SQD data file: %s" % str(err),
                    )
                raise
        self._recompute_misc()
        self._recompute_stim_chs_array()
        self._update_can_save()

    def _recompute_misc(self):
        raw = self.raw
        if raw is None:
            self.misc_chs = None
            self.misc_chs_desc = "No SQD file selected..."
            self.can_test_stim = False
            self.misc_data = None
        else:
            chs = [
                i
                for i, ch in enumerate(raw.info["chs"])
                if ch["kind"] == FIFF.FIFFV_MISC_CH
            ]
            self.misc_chs = chs
            if not chs:
                self.misc_chs_desc = "0 MISC channels"
            elif np.all(np.diff(chs) == 1):
                self.misc_chs_desc = "%i:%i" % (chs[0], chs[-1] + 1)
            else:
                self.misc_chs_desc = "%i... (discontinuous)" % chs[0]
            self.can_test_stim = True
            self.misc_data = None  # load on demand

    @observe("fid_file")
    def _fid_file_changed(self, change):
        fname = change["new"]
        self.fid_fname = Path(fname).name if fname else "-"
        self._recompute_elp_raw()

    def _recompute_elp_raw(self):
        fname = self.fid_file
        if not fname:
            self.elp_raw = None
        else:
            try:
                pts = _read_dig_kit(fname)
                if len(pts) < 8:
                    raise ValueError("File contains %i points, need 8" % len(pts))
            except Exception as err:
                if self.show_gui:
                    QMessageBox.critical(
                        self.parent, "Error Reading Fiducials", str(err)
                    )
                self.fid_file = ""
                raise
            else:
                self.elp_raw = pts
        self._recompute_polhemus_trans()

    def _recompute_polhemus_trans(self):
        elp_raw = self.elp_raw
        if elp_raw is None:
            self.polhemus_neuromag_trans = None
        else:
            nasion, lpa, rpa = apply_trans(als_ras_trans, elp_raw[:3])
            trans = get_ras_to_neuromag_trans(nasion, lpa, rpa)
            self.polhemus_neuromag_trans = np.dot(trans, als_ras_trans)
        self._recompute_elp_fid()

    def _recompute_elp_fid(self):
        empty = np.empty((0, 3))
        elp_raw = self.elp_raw
        trans = self.polhemus_neuromag_trans
        if elp_raw is None or trans is None:
            self.elp = empty
            self.fid = empty
        else:
            self.elp = apply_trans(trans, elp_raw[3:8])
            self.fid = apply_trans(trans, elp_raw[:3])
        self._recompute_dev_head_trans()
        self._recompute_hsp()
        self._update_can_save()

    @observe("hsp_file")
    def _hsp_file_changed(self, change):
        fname = change["new"]
        self.hsp_fname = Path(fname).name if fname else "-"
        self._recompute_hsp_raw()

    def _recompute_hsp_raw(self):
        fname = self.hsp_file
        if not fname:
            self.hsp_raw = None
        else:
            try:
                pts = _read_dig_kit(fname)
                n_pts = len(pts)
                if n_pts > KIT.DIG_POINTS:
                    msg = (
                        "The selected head shape contains {n} points, "
                        "which is more than the recommended maximum "
                        "({rec}). The file will be automatically "
                        "downsampled.".format(n=n_pts, rec=KIT.DIG_POINTS)
                    )
                    if self.show_gui:
                        QMessageBox.information(
                            self.parent, "Too Many Head Shape Points", msg
                        )
                    pts = _decimate_points(pts, 5)
            except Exception as err:
                if self.show_gui:
                    QMessageBox.critical(
                        self.parent, "Error Reading Head Shape", str(err)
                    )
                self.hsp_file = ""
                raise
            self.hsp_raw = pts
        self._recompute_hsp()
        self._update_can_save()

    def _recompute_hsp(self):
        hsp_raw = self.hsp_raw
        trans = self.polhemus_neuromag_trans
        if hsp_raw is None or trans is None:
            self.hsp = np.empty((0, 3))
        else:
            self.hsp = apply_trans(trans, hsp_raw)

    @observe("use_mrk")
    def _use_mrk_changed(self, change):
        self._recompute_dev_head_trans()
        self._update_can_save()

    def _recompute_dev_head_trans(self):
        mrk = self.mrk
        elp = self.elp
        if mrk is None or not np.any(elp):
            self.dev_head_trans = np.eye(4)
            self.head_dev_trans = np.eye(4)
            return

        src_pts = mrk
        dst_pts = elp
        n_use = len(self.use_mrk)

        if n_use < 3:
            if self.show_gui:
                QMessageBox.critical(
                    self.parent,
                    "Not Enough Marker Points",
                    "Estimating the device head transform requires at "
                    "least 3 marker points. Please adjust the markers used.",
                )
            self.dev_head_trans = np.eye(4)
            self.head_dev_trans = np.eye(4)
            return
        elif n_use < 5:
            src_pts = src_pts[self.use_mrk]
            dst_pts = dst_pts[self.use_mrk]

        trans = fit_matched_points(src_pts, dst_pts, out="trans")
        self.dev_head_trans = trans
        self.head_dev_trans = np.linalg.inv(trans)

    @observe("stim_chs", "stim_coding")
    def _stim_params_changed(self, change):
        self._recompute_stim_chs_array()

    def _recompute_stim_chs_array(self):
        raw = self.raw
        if raw is None:
            self.stim_chs_array = None
            self.stim_chs_ok = False
            self.stim_chs_comment = ""
            self._update_can_save()
            return

        chs = self.stim_chs.strip()
        if not chs:
            picks = _default_stim_chs(raw.info)
        else:
            try:
                picks = eval("r_[%s]" % chs, vars(np))
                if picks.dtype.kind != "i":
                    raise TypeError("Need array of int")
            except Exception:
                self.stim_chs_array = None
                self.stim_chs_ok = False
                self.stim_chs_comment = "Invalid!"
                self._update_can_save()
                return

        if self.stim_coding == "<":
            picks = picks[::-1]
        self.stim_chs_array = picks
        self.stim_chs_ok = True
        self.stim_chs_comment = (
            "Default:  The first 8 MISC channels"
            if not chs
            else "Ok:  %i channels" % len(picks)
        )
        self._update_can_save()

    def _update_can_save(self):
        if not self.stim_chs_ok:
            self.can_save = False
            return
        has_all = (
            np.any(self.dev_head_trans)
            and np.any(self.hsp)
            and np.any(self.elp)
            and np.any(self.fid)
        )
        if has_all:
            self.can_save = True
            return
        has_any = self.hsp_file or self.fid_file or np.any(self.mrk)
        self.can_save = not bool(has_any)

    def clear_all(self):
        """Clear all specified input parameters."""
        self.markers.clear()
        self.sqd_file = ""
        self.hsp_file = ""
        self.fid_file = ""
        self.use_mrk = list(range(5))

    def get_misc_data(self, parent=None):
        """Load misc channel data from the SQD file, with progress dialog."""
        if self.misc_data is not None:
            return self.misc_data
        if self.raw is None:
            return None
        if parent is None:
            parent = self.parent

        from qtpy.QtWidgets import QProgressDialog

        prog = QProgressDialog(
            "Loading stim channel data from SQD file ...", None, 0, 0, parent
        )
        prog.setWindowTitle("Loading SQD data...")
        prog.setWindowModality(Qt.WindowModal)
        prog.setMinimumDuration(0)
        prog.show()
        try:
            data, _ = self.raw[self.misc_chs]
        except Exception as err:
            if self.show_gui:
                QMessageBox.critical(
                    parent,
                    "Error Reading SQD File",
                    "Error reading SQD data file: %s" % str(err),
                )
            raise
        finally:
            prog.close()
        self.misc_data = data
        return data

    def get_event_info(self, parent=None):
        """Count events with current stim channel settings."""
        data = self.get_misc_data(parent)
        if data is None:
            return None
        idx = [self.misc_chs.index(ch) for ch in self.stim_chs_array]
        data = data[idx]
        coding = "channel" if self.stim_coding == "channel" else "binary"
        stim_ch = _make_stim_channel(
            data, self.stim_slope, self.stim_threshold, coding, self.stim_chs_array
        )
        events = _find_events(
            stim_ch, self.raw.first_samp, consecutive=True, min_samples=3
        )
        return Counter(events[:, 2])

    def get_raw(self, preload=False):
        """Create a raw object based on the current model settings."""
        if not self.can_save:
            raise ValueError("Not all necessary parameters are set")

        stim_code = "channel" if self.stim_coding == "channel" else "binary"
        logger.info(
            "Creating raw with stim=%r, slope=%r, stim_code=%r, stimthresh=%r",
            self.stim_chs_array,
            self.stim_slope,
            stim_code,
            self.stim_threshold,
        )
        raw = RawKIT(
            self.sqd_file,
            preload=preload,
            stim=self.stim_chs_array,
            slope=self.stim_slope,
            stim_code=stim_code,
            stimthresh=self.stim_threshold,
            allow_unknown_format=self.allow_unknown_format,
        )

        if np.any(self.fid):
            mon = make_dig_montage(
                nasion=self.fid[0],
                lpa=self.fid[1],
                rpa=self.fid[2],
                hpi=self.elp,
                hsp=self.hsp,
            )
            with raw.info._unlock():
                raw.info["dig"] = mon.dig
                raw.info["dev_head_t"] = Transform("meg", "head", self.dev_head_trans)
        return raw


def _load_model_config():
    """Load saved configuration values and return validated Kit2FiffModel."""
    config = get_config(home_dir=os.environ.get("_MNE_FAKE_HOME_DIR"))
    stim_threshold = 1.0
    if "MNE_KIT2FIFF_STIM_CHANNEL_THRESHOLD" in config:
        try:
            stim_threshold = float(config["MNE_KIT2FIFF_STIM_CHANNEL_THRESHOLD"])
        except ValueError:
            warn(
                "Ignoring invalid configuration value for "
                "MNE_KIT2FIFF_STIM_CHANNEL_THRESHOLD: %r (expected float)"
                % (config["MNE_KIT2FIFF_STIM_CHANNEL_THRESHOLD"],)
            )
    stim_slope = config.get("MNE_KIT2FIFF_STIM_CHANNEL_SLOPE", "-")
    if stim_slope not in "+-":
        warn(
            "Ignoring invalid configuration value for "
            "MNE_KIT2FIFF_STIM_CHANNEL_SLOPE: %s (expected + or -)" % stim_slope
        )
        stim_slope = "-"
    stim_coding = config.get("MNE_KIT2FIFF_STIM_CHANNEL_CODING", ">")
    if stim_coding not in ("<", ">", "channel"):
        warn(
            "Ignoring invalid configuration value for "
            "MNE_KIT2FIFF_STIM_CHANNEL_CODING: %s (expected <, > or "
            "channel)" % stim_coding
        )
        stim_coding = ">"
    return Kit2FiffModel(
        stim_chs=config.get("MNE_KIT2FIFF_STIM_CHANNELS", ""),
        stim_coding=stim_coding,
        stim_slope=stim_slope,
        stim_threshold=stim_threshold,
        show_gui=True,
    )


class Kit2FiffPanel(HasTraits):
    """Controller for kit2fiff conversion panel state and save queue."""

    model = Any()  # Kit2FiffModel

    # Visualization
    scene = Any()  # pyvistaqt.QtInteractor
    fid_obj = Any()  # PointObject
    elp_obj = Any()  # PointObject
    hsp_obj = Any()  # PointObject

    # Save queue state (for UI feedback)
    queue = Any()
    queue_feedback = Unicode()
    queue_current = Unicode()
    queue_len = Int(0)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.queue is None:
            self.queue = queue.Queue()
        self._start_save_worker()

        self.fid_obj = PointObject(
            scene=self.scene, color=(0.1, 1.0, 0.1), point_scale=5e-3, name="Fiducials"
        )
        self.elp_obj = PointObject(
            scene=self.scene,
            color=(0.196, 0.196, 0.863),
            point_scale=1e-2,
            opacity=0.2,
            name="ELP",
        )
        self.hsp_obj = PointObject(
            scene=self.scene, color=(0.784,) * 3, point_scale=2e-3, name="HSP"
        )

        model = self.model
        if model is not None:
            model.observe(self._update_fid, names=["fid", "head_dev_trans"])
            model.observe(self._update_hsp, names=["hsp", "head_dev_trans"])
            model.observe(self._update_elp, names=["elp", "head_dev_trans"])
            self._update_fid()
            self._update_elp()
            self._update_hsp()

    def _start_save_worker(self):
        def worker():
            while True:
                raw, fname = self.queue.get()
                basename = Path(fname).name
                self.queue_len -= 1
                self.queue_current = "Processing: %s" % basename
                try:
                    raw.save(fname, overwrite=True)
                    res = "Saved: %s"
                except Exception as err:
                    logger.warning("Error saving %s: %s", basename, err)
                    res = "Error saving: %s"
                self.queue_current = ""
                self.queue_feedback = res % basename
                self.queue.task_done()

        t = Thread(target=worker, daemon=True)
        t.start()

    def _update_fid(self, change=None):
        if self.fid_obj is not None and self.model is not None:
            self.fid_obj.points = apply_trans(self.model.head_dev_trans, self.model.fid)

    def _update_hsp(self, change=None):
        if self.hsp_obj is not None and self.model is not None:
            self.hsp_obj.points = apply_trans(self.model.head_dev_trans, self.model.hsp)

    def _update_elp(self, change=None):
        if self.elp_obj is not None and self.model is not None:
            self.elp_obj.points = apply_trans(self.model.head_dev_trans, self.model.elp)

    def save_as(self, parent=None):
        """Prompt for a save path and queue the raw file for saving."""
        model = self.model
        if parent is None:
            parent = model.parent
        try:
            raw = model.get_raw()
        except Exception as err:
            QMessageBox.critical(parent, "Error Creating KIT Raw", str(err))
            raise

        sqd_path = Path(model.sqd_file)
        stem = sqd_path.stem
        if not stem.endswith("raw"):
            stem += "-raw"
        default_path = sqd_path.with_name(stem + ".fif")

        path, _ = QFileDialog.getSaveFileName(
            parent, "Save FIFF", str(default_path), "FIFF raw file (*.fif)"
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

        self.queue.put((raw, path))
        self.queue_len += 1

    def test_stim(self, parent=None):
        """Show a count of events with current stim settings."""
        if parent is None:
            parent = self.model.parent
        try:
            events = self.model.get_event_info(parent)
        except Exception as err:
            QMessageBox.critical(
                parent,
                "Error Reading Events from SQD File",
                "Error reading events: %s" % str(err),
            )
            raise

        if not events:
            QMessageBox.information(
                parent,
                "No Events Found",
                "No events were found with the current settings.",
            )
        else:
            lines = ["Events found (ID: n events):"]
            for id_ in sorted(events):
                lines.append("%3i: \t%i" % (id_, events[id_]))
            QMessageBox.information(parent, "Events in SQD File", "\n".join(lines))


class Kit2FiffFrame(QMainWindow):
    """Qt window for KIT to FIF conversion."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("KIT to FIFF Conversion")
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.resize(1100, 700)

        self.model = _load_model_config()
        self.model.parent = self

        # pyvista scene embedded directly as a QWidget.
        self._scene_widget = QWidget(self)
        QVBoxLayout(self._scene_widget)
        self.scene = embed_pyvista_scene(self._scene_widget)

        self.headview = HeadViewController(scene=self.scene, scale=160, system="RAS")
        self.kit2fiff_panel = Kit2FiffPanel(scene=self.scene, model=self.model)
        self.marker_panel = CombineMarkersPanel(
            scene=self.scene, model=self.model.markers, trans=als_ras_trans, parent=self
        )

        self._build_ui()

        # Update feedback labels when queue state changes
        self.kit2fiff_panel.observe(
            lambda ch: self._queue_label.setText(ch["new"]), names=["queue_feedback"]
        )
        self.kit2fiff_panel.observe(
            lambda ch: self._queue_current_label.setText(ch["new"]),
            names=["queue_current"],
        )
        self.kit2fiff_panel.observe(
            lambda ch: self._queue_len_label.setText(
                "Queue: %i" % ch["new"] if ch["new"] else ""
            ),
            names=["queue_len"],
        )

        # Update button states from model
        self.model.observe(
            self._update_ui_state,
            names=[
                "can_save",
                "can_test_stim",
                "stim_chs_ok",
                "stim_chs_comment",
                "misc_chs_desc",
                "sqd_fname",
                "hsp_fname",
                "fid_fname",
            ],
        )
        self._update_ui_state()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        central = QWidget(self)
        self.setCentralWidget(central)
        outer = QHBoxLayout(central)

        # ---- left column: marker panel ----
        outer.addLayout(self._build_marker_column(), stretch=1)

        # ---- center: scene + head view buttons ----
        center = QVBoxLayout()
        center.addWidget(self._scene_widget, stretch=4)
        for label, view in [
            ("Front", "front"),
            ("Left", "left"),
            ("Right", "right"),
            ("Top", "top"),
        ]:
            btn = QPushButton(label)
            btn.clicked.connect(lambda _, v=view: self.headview.on_set_view(v))
            center.addWidget(btn)
        outer.addLayout(center, stretch=2)

        # ---- right column: kit2fiff panel ----
        outer.addLayout(self._build_kit2fiff_column(), stretch=1)

    def _build_marker_column(self):
        layout = QVBoxLayout()
        g1 = QGroupBox("Source Marker 1")
        self._mrk1_file_edit = self._make_file_row(
            g1,
            "mrk1",
            "*.sqd *.mrk *.txt *.pickled",
            lambda p: setattr(self.model.markers.mrk1, "file", p),
        )
        layout.addWidget(g1)

        g2 = QGroupBox("Source Marker 2")
        self._mrk2_file_edit = self._make_file_row(
            g2,
            "mrk2",
            "*.sqd *.mrk *.txt *.pickled",
            lambda p: setattr(self.model.markers.mrk2, "file", p),
        )
        layout.addWidget(g2)

        g3 = QGroupBox("Stats")
        g3_layout = QVBoxLayout(g3)
        self._distance_label = QLabel("")
        g3_layout.addWidget(self._distance_label)
        self.model.markers.observe(
            lambda ch: self._distance_label.setText(ch["new"]), names=["distance"]
        )
        layout.addWidget(g3)

        g4 = QGroupBox("New Marker")
        method_combo = QComboBox()
        method_combo.addItems(["Transform", "Average"])
        method_combo.currentTextChanged.connect(
            lambda t: setattr(self.model.markers.mrk3, "method", t)
        )
        self._make_file_row_layout(g4, QVBoxLayout(g4), method_combo)
        layout.addWidget(g4)
        return layout

    def _make_file_row(self, parent_widget, label, wildcard, callback):
        layout = QVBoxLayout(parent_widget)
        row = QHBoxLayout()
        edit = QLineEdit()
        edit.setPlaceholderText("File path...")
        btn = QPushButton("Browse")
        btn.clicked.connect(lambda: self._browse_file(edit, wildcard, callback))
        row.addWidget(edit)
        row.addWidget(btn)
        layout.addLayout(row)
        return edit

    def _make_file_row_layout(self, parent_widget, layout, extra_widget):
        row = QHBoxLayout()
        row.addWidget(extra_widget)
        layout.addLayout(row)

    def _browse_file(self, edit, wildcard, callback):
        path, _ = QFileDialog.getOpenFileName(self, "Select File", "", wildcard)
        if path:
            edit.setText(path)
            callback(path)

    def _build_kit2fiff_column(self):
        layout = QVBoxLayout()

        # Sources group
        sg = QGroupBox("Sources")
        sg_layout = QVBoxLayout(sg)

        row = QHBoxLayout()
        self._sqd_edit = QLineEdit()
        self._sqd_edit.setPlaceholderText("KIT data file...")
        sqd_btn = QPushButton("Browse")
        sqd_btn.clicked.connect(
            lambda: self._browse_file(
                self._sqd_edit,
                kit_con_wildcard,
                lambda p: setattr(self.model, "sqd_file", p),
            )
        )
        row.addWidget(QLabel("Data:"))
        row.addWidget(self._sqd_edit)
        row.addWidget(sqd_btn)
        sg_layout.addLayout(row)
        self._sqd_fname_label = QLabel("-")
        sg_layout.addWidget(self._sqd_fname_label)

        row2 = QHBoxLayout()
        self._hsp_edit = QLineEdit()
        hsp_btn = QPushButton("Browse")
        hsp_btn.clicked.connect(
            lambda: self._browse_file(
                self._hsp_edit,
                hsp_wildcard,
                lambda p: setattr(self.model, "hsp_file", p),
            )
        )
        row2.addWidget(QLabel("HSP:"))
        row2.addWidget(self._hsp_edit)
        row2.addWidget(hsp_btn)
        sg_layout.addLayout(row2)

        row3 = QHBoxLayout()
        self._fid_edit = QLineEdit()
        fid_btn = QPushButton("Browse")
        fid_btn.clicked.connect(
            lambda: self._browse_file(
                self._fid_edit,
                elp_wildcard,
                lambda p: setattr(self.model, "fid_file", p),
            )
        )
        row3.addWidget(QLabel("FID:"))
        row3.addWidget(self._fid_edit)
        row3.addWidget(fid_btn)
        sg_layout.addLayout(row3)

        clear_dig_btn = QPushButton("Clear Digitizer Files")
        clear_dig_btn.clicked.connect(
            lambda: [
                setattr(self.model, "hsp_file", ""),
                setattr(self.model, "fid_file", ""),
            ]
        )
        sg_layout.addWidget(clear_dig_btn)
        layout.addWidget(sg)

        # Events group
        eg = QGroupBox("Events")
        eg_layout = QVBoxLayout(eg)
        self._misc_chs_label = QLabel("No SQD file selected...")
        eg_layout.addWidget(self._misc_chs_label)

        slope_combo = QComboBox()
        slope_combo.addItems(["- (Trough: 5→0 V)", "+ (Peak: 0→5 V)"])
        slope_combo.currentIndexChanged.connect(
            lambda i: setattr(self.model, "stim_slope", "-" if i == 0 else "+")
        )
        eg_layout.addWidget(QLabel("Event Onset:"))
        eg_layout.addWidget(slope_combo)

        coding_combo = QComboBox()
        coding_combo.addItems(["little-endian (>)", "big-endian (<)", "Channel#"])
        coding_combo.currentIndexChanged.connect(
            lambda i: setattr(self.model, "stim_coding", [">", "<", "channel"][i])
        )
        eg_layout.addWidget(QLabel("Value Coding:"))
        eg_layout.addWidget(coding_combo)

        self._stim_chs_edit = QLineEdit()
        self._stim_chs_edit.setPlaceholderText("Default (first 8 MISC)")
        self._stim_chs_edit.editingFinished.connect(
            lambda: setattr(self.model, "stim_chs", self._stim_chs_edit.text())
        )
        eg_layout.addWidget(QLabel("Channels:"))
        eg_layout.addWidget(self._stim_chs_edit)

        self._stim_comment_label = QLabel("")
        eg_layout.addWidget(self._stim_comment_label)

        thr_spin = QDoubleSpinBox()
        thr_spin.setRange(0, 100)
        thr_spin.setValue(1.0)
        thr_spin.valueChanged.connect(
            lambda v: setattr(self.model, "stim_threshold", v)
        )
        eg_layout.addWidget(QLabel("Threshold:"))
        eg_layout.addWidget(thr_spin)

        btn_row = QHBoxLayout()
        self._test_stim_btn = QPushButton("Find Events")
        self._test_stim_btn.clicked.connect(lambda: self.kit2fiff_panel.test_stim(self))
        self._plot_raw_btn = QPushButton("Plot Raw")
        self._plot_raw_btn.clicked.connect(
            lambda: self.model.raw.plot() if self.model.raw else None
        )
        btn_row.addWidget(self._test_stim_btn)
        btn_row.addWidget(self._plot_raw_btn)
        eg_layout.addLayout(btn_row)
        layout.addWidget(eg)

        # Save/clear row
        save_row = QHBoxLayout()
        self._save_btn = QPushButton("Save FIFF...")
        self._save_btn.clicked.connect(lambda: self.kit2fiff_panel.save_as(self))
        self._clear_btn = QPushButton("Clear All")
        self._clear_btn.clicked.connect(lambda: self.model.clear_all())
        save_row.addWidget(self._save_btn)
        save_row.addWidget(self._clear_btn)
        layout.addLayout(save_row)

        # Queue feedback
        self._queue_label = QLabel("")
        self._queue_current_label = QLabel("")
        self._queue_len_label = QLabel("")
        layout.addWidget(self._queue_label)
        layout.addWidget(self._queue_current_label)
        layout.addWidget(self._queue_len_label)

        layout.addStretch()
        return layout

    # ------------------------------------------------------------------
    # UI state updates
    # ------------------------------------------------------------------

    def _update_ui_state(self, change=None):
        model = self.model
        self._save_btn.setEnabled(bool(model.can_save))
        self._test_stim_btn.setEnabled(bool(model.can_test_stim))
        self._plot_raw_btn.setEnabled(model.raw is not None)
        self._misc_chs_label.setText(model.misc_chs_desc)
        self._stim_comment_label.setText(model.stim_chs_comment)
        if hasattr(self, "_sqd_fname_label"):
            self._sqd_fname_label.setText(model.sqd_fname)

    # ------------------------------------------------------------------
    # Config persistence / window lifecycle
    # ------------------------------------------------------------------

    def save_config(self, home_dir=None):
        """Write configuration values to mne config."""
        model = self.model
        set_config(
            "MNE_KIT2FIFF_STIM_CHANNELS", model.stim_chs, home_dir, set_env=False
        )
        set_config(
            "MNE_KIT2FIFF_STIM_CHANNEL_CODING",
            model.stim_coding,
            home_dir,
            set_env=False,
        )
        set_config(
            "MNE_KIT2FIFF_STIM_CHANNEL_SLOPE", model.stim_slope, home_dir, set_env=False
        )
        set_config(
            "MNE_KIT2FIFF_STIM_CHANNEL_THRESHOLD",
            str(model.stim_threshold),
            home_dir,
            set_env=False,
        )

    def closeEvent(self, event):
        """Veto closing while files are still being saved."""
        if self.kit2fiff_panel.queue.unfinished_tasks:
            QMessageBox.information(
                self,
                "Saving Still in Progress",
                "Can not close the window while saving is still in progress. "
                "Please wait until all files are processed.",
            )
            event.ignore()
            return
        try:
            self.save_config()
        except Exception as exc:
            warn("Error saving GUI configuration:\n%s" % exc)
        event.accept()
        self.scene.close()
