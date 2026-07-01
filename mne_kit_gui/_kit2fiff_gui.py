"""Traitlets/Qt GUI for converting data from KIT systems."""

# Authors: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD-3-Clause

from collections import Counter
from collections.abc import Callable, Sequence
import os
import queue
from pathlib import Path
from threading import Thread

import numpy as np
from pyvistaqt import QtInteractor  # ty: ignore[unresolved-import]

from qtpy.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QColorDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)
from qtpy.QtCore import Qt
from qtpy.QtGui import QCloseEvent, QColor

from traitlets import Bool, Bunch, Float, HasTraits, Any, Int, List, Unicode, observe

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

from ._marker_gui import CombineMarkersPanel, CombineMarkersModel, MarkerPointSource
from ._help import read_tooltips
from ._viewer import (
    HeadViewController,
    PointObject,
    build_head_view_group,
    embed_pyvista_scene,
)


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

    def __init__(
        self,
        *,
        stim_chs: str = "",
        stim_coding: str = ">",
        stim_slope: str = "-",
        stim_threshold: float = 1.0,
        show_gui: bool = False,
    ) -> None:
        super().__init__(
            markers=CombineMarkersModel(),
            use_mrk=list(range(5)),
            stim_chs=stim_chs,
            stim_coding=stim_coding,
            stim_slope=stim_slope,
            stim_threshold=stim_threshold,
            show_gui=show_gui,
        )
        self.mrk = np.zeros((5, 3))
        self.elp = np.empty((0, 3))
        self.fid = np.empty((0, 3))
        self.hsp = np.empty((0, 3))
        self.dev_head_trans = np.eye(4)
        self.head_dev_trans = np.eye(4)
        self.markers.parent = self.parent
        # Wire markers
        self.markers.mrk3.observe(self._mrk3_points_changed, names=["points"])
        self._recompute_misc()
        self._recompute_stim_chs_array()

    @observe("parent")
    def _parent_changed(self, change: Bunch) -> None:
        self.markers.parent = change["new"]

    # ------------------------------------------------------------------
    # Observers
    # ------------------------------------------------------------------

    def _mrk3_points_changed(self, change: Bunch) -> None:
        self.mrk = apply_trans(als_ras_trans, change["new"])
        self._recompute_dev_head_trans()
        self._update_can_save()

    @observe("sqd_file")
    def _sqd_file_changed(self, change: Bunch) -> None:
        fname = change["new"]
        self.sqd_fname = Path(fname).name if fname else "-"
        self._recompute_raw()

    def _recompute_raw(self) -> None:
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

    def _recompute_misc(self) -> None:
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
                if ch["kind"] == FIFF.FIFFV_MISC_CH  # ty: ignore[unresolved-attribute]
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
    def _fid_file_changed(self, change: Bunch) -> None:
        fname = change["new"]
        self.fid_fname = Path(fname).name if fname else "-"
        self._recompute_elp_raw()

    def _recompute_elp_raw(self) -> None:
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

    def _recompute_polhemus_trans(self) -> None:
        elp_raw = self.elp_raw
        if elp_raw is None:
            self.polhemus_neuromag_trans = None
        else:
            nasion, lpa, rpa = apply_trans(als_ras_trans, elp_raw[:3])
            trans = get_ras_to_neuromag_trans(nasion, lpa, rpa)
            self.polhemus_neuromag_trans = np.dot(trans, als_ras_trans)
        self._recompute_elp_fid()

    def _recompute_elp_fid(self) -> None:
        elp_raw = self.elp_raw
        trans = self.polhemus_neuromag_trans
        if elp_raw is None or trans is None:
            self.elp = np.empty((0, 3))
            self.fid = np.empty((0, 3))
        else:
            self.elp = apply_trans(trans, elp_raw[3:8])
            self.fid = apply_trans(trans, elp_raw[:3])
        self._recompute_dev_head_trans()
        self._recompute_hsp()
        self._update_can_save()

    @observe("hsp_file")
    def _hsp_file_changed(self, change: Bunch) -> None:
        fname = change["new"]
        self.hsp_fname = Path(fname).name if fname else "-"
        self._recompute_hsp_raw()

    def _recompute_hsp_raw(self) -> None:
        fname = self.hsp_file
        if not fname:
            self.hsp_raw = None
        else:
            try:
                pts = _read_dig_kit(fname)
                n_pts = len(pts)
                if n_pts > KIT.DIG_POINTS:  # ty: ignore[unresolved-attribute]
                    msg = (
                        "The selected head shape contains {n} points, "
                        "which is more than the recommended maximum "
                        "({rec}). The file will be automatically "
                        "downsampled.".format(
                            n=n_pts,
                            rec=KIT.DIG_POINTS,  # ty: ignore[unresolved-attribute]
                        )
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

    def _recompute_hsp(self) -> None:
        hsp_raw = self.hsp_raw
        trans = self.polhemus_neuromag_trans
        if hsp_raw is None or trans is None:
            self.hsp = np.empty((0, 3))
        else:
            self.hsp = apply_trans(trans, hsp_raw)

    @observe("use_mrk")
    def _use_mrk_changed(self, change: Bunch) -> None:
        self._recompute_dev_head_trans()
        self._update_can_save()

    def _recompute_dev_head_trans(self) -> None:
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
    def _stim_params_changed(self, change: Bunch) -> None:
        self._recompute_stim_chs_array()

    def _recompute_stim_chs_array(self) -> None:
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

    def _update_can_save(self) -> None:
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

    def clear_all(self) -> None:
        """Clear all specified input parameters."""
        self.markers.clear()
        self.sqd_file = ""
        self.hsp_file = ""
        self.fid_file = ""
        self.use_mrk = list(range(5))

    def get_misc_data(self) -> np.ndarray | None:
        """Load misc channel data from the SQD file, with progress dialog."""
        if self.misc_data is not None:
            return self.misc_data
        if self.raw is None:
            return None
        parent = self.parent

        prog = QProgressDialog(
            # a None cancel-button label makes the dialog uncancelable
            "Loading stim channel data from SQD file ...",
            None,  # ty: ignore[invalid-argument-type]
            0,
            0,
            parent,
        )
        prog.setWindowTitle("Loading SQD data...")
        prog.setWindowModality(Qt.WindowModal)  # ty: ignore[unresolved-attribute]
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

    def get_event_info(self) -> Counter | None:
        """Count events with current stim channel settings."""
        data = self.get_misc_data()
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

    def get_raw(self, preload: bool = False) -> RawKIT:
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


def _load_model_config() -> Kit2FiffModel:
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

    def __init__(
        self,
        *,
        scene: QtInteractor | None = None,
        model: Kit2FiffModel | None = None,
    ) -> None:
        super().__init__(scene=scene, model=model)
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
            scene=self.scene, color=(0.784, 0.784, 0.784), point_scale=2e-3, name="HSP"
        )

        model = self.model
        if model is not None:
            model.observe(self._update_fid, names=["fid", "head_dev_trans"])
            model.observe(self._update_hsp, names=["hsp", "head_dev_trans"])
            model.observe(self._update_elp, names=["elp", "head_dev_trans"])
            self._update_fid()
            self._update_elp()
            self._update_hsp()

    def _start_save_worker(self) -> None:
        def worker() -> None:
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

    def _update_fid(self, change: Bunch | None = None) -> None:
        if self.fid_obj is not None and self.model is not None:
            self.fid_obj.points = apply_trans(self.model.head_dev_trans, self.model.fid)

    def _update_hsp(self, change: Bunch | None = None) -> None:
        if self.hsp_obj is not None and self.model is not None:
            self.hsp_obj.points = apply_trans(self.model.head_dev_trans, self.model.hsp)

    def _update_elp(self, change: Bunch | None = None) -> None:
        if self.elp_obj is not None and self.model is not None:
            self.elp_obj.points = apply_trans(self.model.head_dev_trans, self.model.elp)

    def save_as(self) -> None:
        """Prompt for a save path and queue the raw file for saving."""
        model = self.model
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
            if reply != QMessageBox.Yes:  # ty: ignore[unresolved-attribute]
                return

        self.queue.put((raw, path))
        self.queue_len += 1

    def test_stim(self) -> None:
        """Show a count of events with current stim settings."""
        parent = self.model.parent
        try:
            events = self.model.get_event_info()
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

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("KIT to FIFF Conversion")
        self.setAttribute(Qt.WA_DeleteOnClose, True)  # ty: ignore[unresolved-attribute]
        self.resize(1100, 700)

        self.model = _load_model_config()
        self.model.parent = self

        # pyvista scene embedded directly as a QWidget.
        self._scene_widget = QWidget(self)
        QVBoxLayout(self._scene_widget)
        self.scene = embed_pyvista_scene(self._scene_widget)

        self.headview = HeadViewController(scene=self.scene, scale=0.16, system="RAS")
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
                "sqd_file",
                "hsp_file",
                "fid_file",
                "sqd_fname",
                "hsp_fname",
                "fid_fname",
                "use_mrk",
            ],
        )
        self._update_ui_state()

        # start from a top-down view, matching the mayavi GUI's default
        self.headview.on_set_view("top")

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)
        outer = QHBoxLayout(central)

        # Give the 3D scene the lion's share of the width: unlike the original
        # mayavi layout, the plot is useful to watch during every adjustment.
        # ---- left column: marker panel ----
        outer.addLayout(self._build_marker_column(), stretch=1)

        # ---- center: scene + head view controls ----
        center = QVBoxLayout()
        center.addWidget(self._scene_widget, stretch=4)
        center.addWidget(build_head_view_group(self.headview))
        outer.addLayout(center, stretch=2)

        # ---- right column: kit2fiff panel ----
        outer.addLayout(self._build_kit2fiff_column(), stretch=1)

    def _build_marker_column(self) -> QVBoxLayout:
        layout = QVBoxLayout()
        markers = self.model.markers
        mp = self.marker_panel
        wildcard = "*.sqd *.mrk *.txt *.pickled"
        layout.addWidget(
            self._build_source_marker_group(
                "Source Marker 1", "mrk1", markers.mrk1, mp.mrk1_obj, wildcard
            )
        )
        layout.addWidget(
            self._build_source_marker_group(
                "Source Marker 2", "mrk2", markers.mrk2, mp.mrk2_obj, wildcard
            )
        )

        g3 = QGroupBox("Stats")
        g3_layout = QHBoxLayout(g3)
        g3_layout.addWidget(QLabel("Distance:"))
        self._distance_label = QLabel("")
        g3_layout.addWidget(self._distance_label, stretch=1)
        markers.observe(
            lambda ch: self._distance_label.setText(ch["new"]), names=["distance"]
        )
        layout.addWidget(g3)

        g4 = QGroupBox("New Marker")
        g4_layout = QVBoxLayout(g4)
        # "Method" as a pair of radio buttons, matching the original mayavi GUI
        method_row = QHBoxLayout()
        method_row.addWidget(QLabel("Method:"))
        method_group = QButtonGroup(g4)
        for text in ("Transform", "Average"):
            rb = QRadioButton(text)
            rb.setObjectName("mrk3_method_%s" % text.lower())
            rb.setChecked(markers.mrk3.method == text)
            rb.toggled.connect(
                lambda checked, t=text: (
                    setattr(markers.mrk3, "method", t) if checked else None
                )
            )
            method_group.addButton(rb)
            method_row.addWidget(rb)
        method_row.addStretch()

        def _sync_method(change: Bunch) -> None:
            for rb in method_group.buttons():
                rb.setChecked(rb.text() == change["new"])

        markers.mrk3.observe(_sync_method, names=["method"])
        g4_layout.addLayout(method_row)

        save3 = QPushButton("Save as")
        save3.setObjectName("mrk3_save")
        save3.clicked.connect(lambda *_: markers.mrk3.save_as())
        save3.setEnabled(markers.mrk3.can_save)
        markers.mrk3.observe(lambda ch: save3.setEnabled(ch["new"]), names=["can_save"])
        g4_layout.addWidget(save3)
        g4_layout.addLayout(self._build_point_object_row("mrk3", mp.mrk3_obj))
        layout.addWidget(g4)
        return layout

    def _build_source_marker_group(
        self,
        title: str,
        name: str,
        mrk: MarkerPointSource,
        obj: PointObject,
        wildcard: str,
    ) -> QGroupBox:
        """Build a group box with the full set of controls for one source."""
        g = QGroupBox(title)
        layout = QVBoxLayout(g)
        layout.addLayout(self._build_marker_file_row(name, mrk, wildcard))
        layout.addLayout(self._build_marker_use_row(name, mrk))
        layout.addLayout(self._build_marker_button_row(name, mrk))
        layout.addLayout(self._build_point_object_row(name, obj))
        return g

    def _build_marker_file_row(
        self, name: str, mrk: MarkerPointSource, wildcard: str
    ) -> QVBoxLayout:
        # the (read-only) line edit mirrors the model's file path, with the
        # basename shown separately so it stays visible for long paths
        outer = QVBoxLayout()
        row = QHBoxLayout()
        edit = QLineEdit()
        edit.setObjectName("%s_file" % name)
        edit.setReadOnly(True)
        edit.setPlaceholderText("File path...")
        browse = QPushButton("Browse")
        browse.setObjectName("%s_browse" % name)
        browse.clicked.connect(
            lambda: self._browse_file(edit, wildcard, lambda p: setattr(mrk, "file", p))
        )
        mrk.observe(lambda ch: edit.setText(ch["new"]), names=["file"])
        row.addWidget(edit)
        row.addWidget(browse)
        outer.addLayout(row)

        basename = QLabel(mrk.name)
        basename.setObjectName("%s_name" % name)
        mrk.observe(lambda ch: basename.setText(ch["new"]), names=["name"])
        outer.addWidget(basename)
        return outer

    def _build_marker_use_row(self, name: str, mrk: MarkerPointSource) -> QHBoxLayout:
        # checkboxes selecting which points (0-4) feed the interpolation
        row = QHBoxLayout()
        row.addWidget(QLabel("Use:"))
        checks = []
        for i in range(5):
            cb = QCheckBox(str(i))
            cb.setObjectName("%s_use_%i" % (name, i))
            cb.setChecked(i in mrk.use)
            cb.setEnabled(mrk.enabled)
            cb.toggled.connect(
                lambda checked, idx=i: self._toggle_use(mrk, idx, checked)
            )
            row.addWidget(cb)
            checks.append(cb)

        def _sync_use(change):
            for idx, cb in enumerate(checks):
                cb.blockSignals(True)
                cb.setChecked(idx in change["new"])
                cb.blockSignals(False)

        mrk.observe(_sync_use, names=["use"])
        mrk.observe(
            lambda ch: [cb.setEnabled(ch["new"]) for cb in checks], names=["enabled"]
        )
        return row

    def _build_marker_button_row(
        self, name: str, mrk: MarkerPointSource
    ) -> QVBoxLayout:
        # Two compact rows keep the marker panel narrow (leaving width for the
        # 3D scene): file lifecycle on top, point-order ops below. Clear and
        # Save as are gated on there being data (can_save).
        outer = QVBoxLayout()
        rows = {"a": QHBoxLayout(), "b": QHBoxLayout()}
        gated = []
        for key, label, handler, r in (
            ("clear", "Clear", mrk.clear, "a"),
            ("edit", "Edit", mrk.edit, "a"),
            ("save", "Save as", mrk.save_as, "a"),
            ("switch", "Switch Left/Right", mrk.switch_left_right, "b"),
            ("reorder", "Reorder", mrk.reorder, "b"),
        ):
            btn = QPushButton(label)
            btn.setObjectName("%s_%s" % (name, key))
            btn.clicked.connect(lambda *_, h=handler: h())
            btn.setEnabled(mrk.can_save if key in ("clear", "save") else True)
            rows[r].addWidget(btn)
            if key in ("clear", "save"):
                gated.append(btn)
        outer.addLayout(rows["a"])
        outer.addLayout(rows["b"])
        mrk.observe(
            lambda ch: [btn.setEnabled(ch["new"]) for btn in gated], names=["can_save"]
        )
        return outer

    def _build_point_object_row(self, name: str, obj: PointObject) -> QHBoxLayout:
        """Build visualization controls (Show/color/Size/Label) for a glyph."""
        row = QHBoxLayout()

        show = QCheckBox("Show")
        show.setObjectName("%s_show" % name)
        show.setChecked(obj.visible)
        show.toggled.connect(lambda checked: setattr(obj, "visible", checked))
        obj.observe(lambda ch: show.setChecked(ch["new"]), names=["visible"])
        row.addWidget(show)

        color = QPushButton()
        color.setObjectName("%s_color" % name)
        color.clicked.connect(lambda: self._pick_color(obj))
        self._set_color_swatch(color, obj.color)
        obj.observe(
            lambda ch: self._set_color_swatch(color, ch["new"]), names=["color"]
        )
        row.addWidget(color)

        row.addWidget(QLabel("Size:"))
        size = QDoubleSpinBox()
        size.setObjectName("%s_size" % name)
        size.setDecimals(4)
        size.setRange(0.0, 1.0)
        size.setSingleStep(1e-3)
        size.setValue(obj.point_scale)
        size.valueChanged.connect(lambda v: setattr(obj, "point_scale", v))
        obj.observe(lambda ch: size.setValue(ch["new"]), names=["point_scale"])
        row.addWidget(size)

        label = QCheckBox("Label")
        label.setObjectName("%s_label" % name)
        label.setChecked(obj.label)
        label.toggled.connect(lambda checked: setattr(obj, "label", checked))
        obj.observe(lambda ch: label.setChecked(ch["new"]), names=["label"])
        row.addWidget(label)

        return row

    @staticmethod
    def _select_radio(group: QButtonGroup, obj_name: str) -> None:
        """Check the radio in ``group`` with the given objectName, if not already."""
        for rb in group.buttons():
            if rb.objectName() == obj_name and not rb.isChecked():
                rb.setChecked(True)
                break

    @staticmethod
    def _set_color_swatch(button: QPushButton, color: Sequence[float]) -> None:
        qcolor = QColor.fromRgbF(*color[:3])
        button.setText("(%d,%d,%d)" % (qcolor.red(), qcolor.green(), qcolor.blue()))
        # pick a readable text color based on the swatch's luminance
        luminance = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
        fg = "black" if luminance > 0.5 else "white"
        button.setStyleSheet("background-color: %s; color: %s" % (qcolor.name(), fg))

    def _pick_color(self, obj: PointObject) -> None:
        qcolor = QColorDialog.getColor(QColor.fromRgbF(*obj.color[:3]), self)
        if qcolor.isValid():
            obj.color = (qcolor.redF(), qcolor.greenF(), qcolor.blueF())

    def _toggle_use(self, mrk: MarkerPointSource, idx: int, checked: bool) -> None:
        use = set(mrk.use)
        if checked:
            use.add(idx)
        else:
            use.discard(idx)
        mrk.use = sorted(use)

    def _browse_file(
        self, edit: QLineEdit, wildcard: str, callback: Callable[[str], None]
    ) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select File", "", wildcard)
        if path:
            edit.setText(path)
            callback(path)

    def _build_kit2fiff_column(self) -> QVBoxLayout:
        layout = QVBoxLayout()

        # Sources group
        sg = QGroupBox("Sources")
        sg_layout = QVBoxLayout(sg)

        row = QHBoxLayout()
        self._sqd_edit = QLineEdit()
        self._sqd_edit.setReadOnly(True)
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
        self._hsp_edit.setReadOnly(True)
        self._hsp_edit.setPlaceholderText("Digitizer head shape...")
        hsp_btn = QPushButton("Browse")
        hsp_btn.clicked.connect(
            lambda: self._browse_file(
                self._hsp_edit,
                hsp_wildcard,
                lambda p: setattr(self.model, "hsp_file", p),
            )
        )
        row2.addWidget(QLabel("Digitizer\nHead Shape:"))
        row2.addWidget(self._hsp_edit)
        row2.addWidget(hsp_btn)
        sg_layout.addLayout(row2)
        self._hsp_fname_label = QLabel("-")
        sg_layout.addWidget(self._hsp_fname_label)

        row3 = QHBoxLayout()
        self._fid_edit = QLineEdit()
        self._fid_edit.setReadOnly(True)
        self._fid_edit.setPlaceholderText("Digitizer fiducials...")
        fid_btn = QPushButton("Browse")
        fid_btn.clicked.connect(
            lambda: self._browse_file(
                self._fid_edit,
                elp_wildcard,
                lambda p: setattr(self.model, "fid_file", p),
            )
        )
        row3.addWidget(QLabel("Digitizer\nFiducials:"))
        row3.addWidget(self._fid_edit)
        row3.addWidget(fid_btn)
        sg_layout.addLayout(row3)
        self._fid_fname_label = QLabel("-")
        sg_layout.addWidget(self._fid_fname_label)

        clear_dig_btn = QPushButton("Clear Digitizer Files")
        clear_dig_btn.clicked.connect(
            lambda: [
                setattr(self.model, "hsp_file", ""),
                setattr(self.model, "fid_file", ""),
            ]
        )
        sg_layout.addWidget(clear_dig_btn)

        # "Use mrk" checkboxes: which marker points feed the dev->head transform
        use_row = QHBoxLayout()
        use_row.addWidget(QLabel("Use mrk:"))
        self._use_mrk_checks = []
        for i in range(5):
            cb = QCheckBox(str(i))
            cb.setObjectName("use_mrk_%i" % i)
            cb.setChecked(i in self.model.use_mrk)
            cb.toggled.connect(
                lambda checked, idx=i: self._toggle_use_mrk(idx, checked)
            )
            use_row.addWidget(cb)
            self._use_mrk_checks.append(cb)
        sg_layout.addLayout(use_row)
        layout.addWidget(sg)

        # Events group -- label-left form layout matching the original mayavi GUI
        eg = QGroupBox("Events")
        eg_form = QFormLayout(eg)
        eg_form.setLabelAlignment(Qt.AlignRight)  # ty: ignore[unresolved-attribute]
        eg_form.setFieldGrowthPolicy(
            QFormLayout.AllNonFixedFieldsGrow  # ty: ignore[unresolved-attribute]
        )

        self._misc_chs_label = QLabel("No SQD file selected...")
        eg_form.addRow("MISC Channels:", self._misc_chs_label)

        # Event onset: trough (falling, "-") vs peak (rising, "+")
        self._slope_group = QButtonGroup(eg)
        slope_row = QHBoxLayout()
        for text, slope, tag in (
            ("Trough (5 to 0 V)", "-", "trough"),
            ("Peak (0 to 5 V)", "+", "peak"),
        ):
            rb = QRadioButton(text)
            rb.setObjectName("stim_slope_%s" % tag)
            rb.setChecked(self.model.stim_slope == slope)
            rb.toggled.connect(
                lambda checked, s=slope: (
                    setattr(self.model, "stim_slope", s) if checked else None
                )
            )
            self._slope_group.addButton(rb)
            slope_row.addWidget(rb)
        slope_row.addStretch()
        eg_form.addRow("Event Onset:", slope_row)
        self.model.observe(
            lambda ch: self._select_radio(
                self._slope_group,
                "stim_slope_%s" % {"-": "trough", "+": "peak"}[ch["new"]],
            ),
            names=["stim_slope"],
        )

        # Value coding: binary little-/big-endian, or per-channel
        self._coding_group = QButtonGroup(eg)
        coding_row = QHBoxLayout()
        for text, code, tag in (
            ("Little-endian", ">", "little"),
            ("Big-endian", "<", "big"),
            ("Channel#", "channel", "channel"),
        ):
            rb = QRadioButton(text)
            rb.setObjectName("stim_coding_%s" % tag)
            rb.setChecked(self.model.stim_coding == code)
            rb.toggled.connect(
                lambda checked, c=code: (
                    setattr(self.model, "stim_coding", c) if checked else None
                )
            )
            self._coding_group.addButton(rb)
            coding_row.addWidget(rb)
        coding_row.addStretch()
        eg_form.addRow("Value Coding:", coding_row)
        self.model.observe(
            lambda ch: self._select_radio(
                self._coding_group,
                "stim_coding_%s"
                % {">": "little", "<": "big", "channel": "channel"}[ch["new"]],
            ),
            names=["stim_coding"],
        )

        self._stim_chs_edit = QLineEdit(self.model.stim_chs)
        self._stim_chs_edit.setPlaceholderText("Default (first 8 MISC)")
        self._stim_chs_edit.editingFinished.connect(
            lambda: setattr(self.model, "stim_chs", self._stim_chs_edit.text())
        )
        eg_form.addRow("Channels:", self._stim_chs_edit)

        self._stim_comment_label = QLabel("")
        eg_form.addRow("", self._stim_comment_label)

        thr_spin = QDoubleSpinBox()
        thr_spin.setObjectName("stim_threshold")
        thr_spin.setRange(0, 100)
        thr_spin.setValue(self.model.stim_threshold)
        thr_spin.valueChanged.connect(
            lambda v: setattr(self.model, "stim_threshold", v)
        )
        eg_form.addRow("Threshold:", thr_spin)

        btn_row = QHBoxLayout()
        self._test_stim_btn = QPushButton("Find Events")
        self._test_stim_btn.clicked.connect(lambda: self.kit2fiff_panel.test_stim())
        self._plot_raw_btn = QPushButton("Plot Raw")
        self._plot_raw_btn.clicked.connect(
            lambda: self.model.raw.plot() if self.model.raw else None
        )
        btn_row.addWidget(self._test_stim_btn)
        btn_row.addWidget(self._plot_raw_btn)
        eg_form.addRow(btn_row)
        layout.addWidget(eg)

        # Save/clear row
        save_row = QHBoxLayout()
        self._save_btn = QPushButton("Save FIFF...")
        self._save_btn.clicked.connect(lambda: self.kit2fiff_panel.save_as())
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

    def _update_ui_state(self, change: Bunch | None = None) -> None:
        model = self.model
        self._save_btn.setEnabled(bool(model.can_save))
        self._test_stim_btn.setEnabled(bool(model.can_test_stim))
        self._plot_raw_btn.setEnabled(model.raw is not None)
        self._misc_chs_label.setText(model.misc_chs_desc)
        self._stim_comment_label.setText(model.stim_chs_comment)
        if hasattr(self, "_sqd_fname_label"):
            # keep the path fields, basename labels, and marker checkboxes in
            # sync with the model (e.g. after loading files or clear_all)
            self._sqd_edit.setText(model.sqd_file)
            self._hsp_edit.setText(model.hsp_file)
            self._fid_edit.setText(model.fid_file)
            self._sqd_fname_label.setText(model.sqd_fname)
            self._hsp_fname_label.setText(model.hsp_fname)
            self._fid_fname_label.setText(model.fid_fname)
            for idx, cb in enumerate(self._use_mrk_checks):
                cb.blockSignals(True)
                cb.setChecked(idx in model.use_mrk)
                cb.blockSignals(False)

    def _toggle_use_mrk(self, idx: int, checked: bool) -> None:
        use = set(self.model.use_mrk)
        if checked:
            use.add(idx)
        else:
            use.discard(idx)
        self.model.use_mrk = sorted(use)

    # ------------------------------------------------------------------
    # Config persistence / window lifecycle
    # ------------------------------------------------------------------

    def save_config(self, home_dir: str | None = None) -> None:
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

    def closeEvent(self, event: QCloseEvent) -> None:
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
