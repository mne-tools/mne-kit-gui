# Authors: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD-3-Clause

from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QLabel,
    QLineEdit,
    QPushButton,
)

import mne
from mne.io import read_raw_fif
import mne_kit_gui

from mne.io.kit.constants import KIT

from mne_kit_gui._kit2fiff_gui import (
    Kit2FiffModel,
    Kit2FiffPanel,
    _load_model_config,
)

kit_data_dir = Path(__file__).parent / "data"
mrk_pre_path = kit_data_dir / "test_mrk_pre.sqd"
mrk_post_path = kit_data_dir / "test_mrk_post.sqd"
sqd_path = kit_data_dir / "test.sqd"
hsp_path = kit_data_dir / "test_hsp.txt"
fid_path = kit_data_dir / "test_elp.txt"
fif_path = kit_data_dir / "test_bin_raw.fif"


def test_kit2fiff_model(tmp_path, mocker):
    """Test Kit2Fiff model."""
    tgt_fname = tmp_path / "test-raw.fif"

    model = Kit2FiffModel()
    assert not model.can_save
    assert model.misc_chs_desc == "No SQD file selected..."
    assert model.stim_chs_comment == ""
    model.markers.mrk1.file = str(mrk_pre_path)
    model.markers.mrk2.file = str(mrk_post_path)
    model.sqd_file = str(sqd_path)
    assert model.misc_chs_desc == "160:192"
    model.hsp_file = str(hsp_path)
    assert not model.can_save
    model.fid_file = str(fid_path)
    assert model.can_save

    # events
    model.stim_slope = "+"
    assert model.get_event_info() == {1: 2}
    model.stim_slope = "-"
    assert model.get_event_info() == {254: 2, 255: 2}

    # stim channels
    model.stim_chs = "181:184, 186"
    assert_array_equal(model.stim_chs_array, [181, 182, 183, 186])
    assert model.stim_chs_ok
    assert model.get_event_info() == {}
    model.stim_chs = "181:184, bad"
    assert not model.stim_chs_ok
    assert not model.can_save
    model.stim_chs = ""
    assert model.can_save

    # export raw
    raw_out = model.get_raw()
    raw_out.save(tgt_fname)
    raw = read_raw_fif(tgt_fname)

    # Compare exported raw with the original binary conversion
    raw_bin = read_raw_fif(fif_path)
    trans_bin = raw.info["dev_head_t"]["trans"]
    want_keys = list(raw_bin.info.keys())
    assert sorted(want_keys) == sorted(raw.info.keys())
    trans_transform = raw_bin.info["dev_head_t"]["trans"]
    assert_allclose(trans_transform, trans_bin, 0.1)

    # Averaging markers
    model.markers.mrk3.method = "Average"
    trans_avg = model.dev_head_trans
    assert not np.all(trans_avg == trans_transform)
    assert_allclose(trans_avg, trans_bin, 0.1)

    # Test exclusion of one marker
    model.markers.mrk3.method = "Transform"
    model.use_mrk = [1, 2, 3, 4]
    assert not np.all(model.dev_head_trans == trans_transform)
    assert not np.all(model.dev_head_trans == trans_avg)
    assert not np.all(model.dev_head_trans == np.eye(4))

    # test setting stim channels
    model.stim_slope = "+"
    events_bin = mne.find_events(raw_bin, stim_channel="STI 014")

    model.stim_coding = "<"
    raw = model.get_raw()
    events = mne.find_events(raw, stim_channel="STI 014")
    assert_array_equal(events, events_bin)

    events_rev = events_bin.copy()
    events_rev[:, 2] = 1
    model.stim_coding = ">"
    raw = model.get_raw()
    events = mne.find_events(raw, stim_channel="STI 014")
    assert_array_equal(events, events_rev)

    model.stim_coding = "channel"
    model.stim_chs = "160:161"
    raw = model.get_raw()
    events = mne.find_events(raw, stim_channel="STI 014")
    assert_array_equal(events, events_bin + [0, 0, 32])

    # fewer than three markers cannot estimate a transform
    mock_msgbox = mocker.patch("mne_kit_gui._kit2fiff_gui.QMessageBox")
    model.show_gui = True
    model.use_mrk = [0, 1]
    assert_array_equal(model.dev_head_trans, np.eye(4))
    assert_array_equal(model.head_dev_trans, np.eye(4))
    mock_msgbox.critical.assert_called_once()
    model.show_gui = False
    model.use_mrk = [0, 1, 2, 3, 4]

    # test reset
    model.clear_all()
    assert model.use_mrk == [0, 1, 2, 3, 4]
    assert model.sqd_file == ""


def test_save_worker(tmp_path, monkeypatch, mocker):
    """Test the background save-queue worker without spawning a real thread."""
    captured = {}

    class FakeThread:
        def __init__(self, target, daemon=None):
            captured["target"] = target

        def start(self):
            pass  # don't actually run the worker in a background thread

    monkeypatch.setattr("mne_kit_gui._kit2fiff_gui.Thread", FakeThread)
    panel = Kit2FiffPanel()
    worker = captured["target"]

    # A one-shot queue yields a single item, then breaks the worker's loop.
    class _Stop(Exception):
        pass

    class OneShotQueue:
        def __init__(self, item):
            self._item = item

        def get(self):
            if self._item is None:
                raise _Stop
            item, self._item = self._item, None
            return item

        def task_done(self):
            pass

    fname = tmp_path / "out-raw.fif"

    # successful save
    raw = mocker.Mock()
    panel.queue = OneShotQueue((raw, fname))
    with pytest.raises(_Stop):
        worker()
    raw.save.assert_called_once_with(fname, overwrite=True)
    assert panel.queue_feedback == "Saved: out-raw.fif"
    assert panel.queue_current == ""

    # failed save is caught and reported via the feedback string
    raw_bad = mocker.Mock()
    raw_bad.save.side_effect = RuntimeError("boom")
    panel.queue = OneShotQueue((raw_bad, fname))
    with pytest.raises(_Stop):
        worker()
    assert panel.queue_feedback == "Error saving: out-raw.fif"


def test_save_as(tmp_path, monkeypatch, mocker):
    """Test Kit2FiffPanel.save_as queueing and dialogs (no real thread)."""
    monkeypatch.setattr(
        "mne_kit_gui._kit2fiff_gui.Thread", lambda *a, **k: mocker.Mock()
    )
    mock_fd = mocker.patch("mne_kit_gui._kit2fiff_gui.QFileDialog")
    mock_qmb = mocker.patch("mne_kit_gui._kit2fiff_gui.QMessageBox")

    panel = Kit2FiffPanel()
    panel.model = mocker.Mock()
    panel.model.sqd_file = str(tmp_path / "test.sqd")

    # an error creating the raw object is reported and re-raised
    panel.model.get_raw.side_effect = RuntimeError("nope")
    with pytest.raises(RuntimeError, match="nope"):
        panel.save_as()
    mock_qmb.critical.assert_called_once()

    # a successful raw is queued and the ".fif" suffix is appended
    raw = mocker.Mock()
    panel.model.get_raw.side_effect = None
    panel.model.get_raw.return_value = raw
    no_suffix = tmp_path / "out"  # missing extension, does not yet exist
    mock_fd.getSaveFileName.return_value = (str(no_suffix), "")
    panel.save_as()
    assert panel.queue_len == 1
    queued_raw, queued_path = panel.queue.get_nowait()
    assert queued_raw is raw
    assert queued_path == tmp_path / "out.fif"

    # an existing target prompts to overwrite; answering "No" cancels the queue
    existing = tmp_path / "exists.fif"
    existing.write_text("x")
    mock_fd.getSaveFileName.return_value = (str(existing), "")
    mock_qmb.question.return_value = mock_qmb.No
    panel.save_as()
    assert panel.queue_len == 1  # unchanged
    mock_qmb.question.assert_called_once()

    # an empty selection is a no-op
    mock_fd.getSaveFileName.return_value = ("", "")
    panel.save_as()
    assert panel.queue_len == 1


def test_panel_test_stim(monkeypatch, mocker):
    """Test Kit2FiffPanel.test_stim event reporting (no real thread)."""
    monkeypatch.setattr(
        "mne_kit_gui._kit2fiff_gui.Thread", lambda *a, **k: mocker.Mock()
    )
    mock_qmb = mocker.patch("mne_kit_gui._kit2fiff_gui.QMessageBox")

    panel = Kit2FiffPanel()
    panel.model = mocker.Mock()

    # reading events can fail -> critical dialog and re-raise
    panel.model.get_event_info.side_effect = RuntimeError("bad")
    with pytest.raises(RuntimeError, match="bad"):
        panel.test_stim()
    mock_qmb.critical.assert_called_once()

    # no events found
    panel.model.get_event_info.side_effect = None
    panel.model.get_event_info.return_value = {}
    panel.test_stim()

    # events found are listed
    panel.model.get_event_info.return_value = {1: 2, 5: 3}
    panel.test_stim()
    assert mock_qmb.information.call_count == 2


@pytest.mark.filterwarnings("ignore:.*:RuntimeWarning")
def test_kit2fiff_model_read_errors(tmp_path, mocker):
    """Test that malformed dig/SQD files raise and report via QMessageBox."""
    mock_msgbox = mocker.patch("mne_kit_gui._kit2fiff_gui.QMessageBox")
    model = Kit2FiffModel(show_gui=True)

    # a fiducials file with fewer than the required 8 points
    too_few = tmp_path / "few.txt"
    np.savetxt(too_few, np.arange(9, dtype=float).reshape(3, 3))
    with pytest.raises(ValueError, match="need 8"):
        model.fid_file = str(too_few)
    assert model.fid_file == ""  # reset to empty on error

    # an unparsable head-shape file
    bad_hsp = tmp_path / "bad.txt"
    bad_hsp.write_text("not\tnumbers\there\n")
    with pytest.raises(Exception):
        model.hsp_file = str(bad_hsp)
    assert model.hsp_file == ""

    # a head-shape file with too many points triggers automatic downsampling
    big_hsp = tmp_path / "big.txt"
    pts = np.random.RandomState(0).randn(KIT.DIG_POINTS + 1, 3)
    np.savetxt(big_hsp, pts)
    model.hsp_file = str(big_hsp)
    assert 0 < len(model.hsp_raw) <= KIT.DIG_POINTS
    mock_msgbox.information.assert_called_once()

    # a corrupt SQD file
    bad_sqd = tmp_path / "bad.sqd"
    bad_sqd.write_bytes(b"not a valid sqd file")
    with pytest.raises(Exception):
        model.sqd_file = str(bad_sqd)
    assert model.sqd_file == ""

    # each failing read reported an error dialog
    assert mock_msgbox.critical.call_count == 3


def test_load_model_config_invalid(tmp_path, monkeypatch):
    """Test that _load_model_config ignores invalid saved config values."""
    monkeypatch.setenv("_MNE_FAKE_HOME_DIR", str(tmp_path))
    # write invalid values to the config file (set_env=False avoids polluting
    # os.environ for other tests)
    for key, val in (
        ("MNE_KIT2FIFF_STIM_CHANNEL_THRESHOLD", "not-a-float"),
        ("MNE_KIT2FIFF_STIM_CHANNEL_SLOPE", "?"),
        ("MNE_KIT2FIFF_STIM_CHANNEL_CODING", "?"),
    ):
        mne.set_config(key, val, home_dir=str(tmp_path), set_env=False)
    with pytest.warns(RuntimeWarning, match="Ignoring invalid"):
        model = _load_model_config()
    # invalid values fall back to the defaults
    assert model.stim_threshold == 1.0
    assert model.stim_slope == "-"
    assert model.stim_coding == ">"


def test_kit2fiff_gui(qtbot, check_gc, tmp_path, monkeypatch, mocker):
    """Test Kit2Fiff GUI."""
    monkeypatch.setenv("_MNE_FAKE_HOME_DIR", str(tmp_path))

    # WA_DeleteOnClose means this frame's underlying C++ object is gone
    # once we close it below, so don't also register it with qtbot for
    # auto-close at teardown -- that would double-close it.
    frame = mne_kit_gui.kit2fiff(block=False)

    assert not frame.model.can_save
    assert frame.model.stim_threshold == 1.0
    frame.model.stim_threshold = 10.0
    frame.model.stim_chs = "save this!"
    frame.save_config(str(tmp_path))
    frame.close()

    # test setting persistence
    frame = mne_kit_gui.kit2fiff(block=False)
    with qtbot.wait_exposed(frame):
        pass
    assert frame.model.stim_threshold == 10.0
    assert frame.model.stim_chs == "save this!"

    # set and reset marker file
    points = [
        [-0.084612, 0.021582, -0.056144],
        [0.080425, 0.021995, -0.061171],
        [-0.000787, 0.105530, 0.014168],
        [-0.047943, 0.091835, 0.010240],
        [0.042976, 0.094380, 0.010807],
    ]
    assert_array_equal(frame.marker_panel.mrk1_obj.points, 0)
    assert_array_equal(frame.marker_panel.mrk3_obj.points, 0)
    frame.model.markers.mrk1.file = str(mrk_pre_path)
    assert_allclose(frame.marker_panel.mrk1_obj.points, points, atol=1e-6)
    assert_allclose(frame.marker_panel.mrk3_obj.points, points, atol=1e-6)
    frame.marker_panel.mrk1_obj.label = True
    frame.marker_panel.mrk1_obj.label = False

    # --- exercise the marker-panel controls (dialogs mocked) ---
    mrk1 = frame.model.markers.mrk1
    # the file path field mirrors the loaded file, with the basename shown too
    assert frame.findChild(QLineEdit, "mrk1_file").text() == str(mrk_pre_path)
    assert frame.findChild(QLabel, "mrk1_name").text() == mrk_pre_path.name
    # Clear/Save As are enabled once data is present
    clear_btn = frame.findChild(QPushButton, "mrk1_clear")
    assert clear_btn.isEnabled()
    assert frame.findChild(QPushButton, "mrk1_save").isEnabled()

    # toggling a "use" checkbox updates the model, and model changes sync back
    cb0 = frame.findChild(QCheckBox, "mrk1_use_0")
    assert cb0.isChecked()
    cb0.setChecked(False)
    assert 0 not in mrk1.use
    cb0.setChecked(True)
    assert 0 in mrk1.use
    mrk1.use = [1, 2, 3, 4]  # model change syncs the checkbox back
    assert not cb0.isChecked()
    mrk1.use = [0, 1, 2, 3, 4]

    # the Browse button routes the chosen path into the model
    mock_open = mocker.patch("mne_kit_gui._kit2fiff_gui.QFileDialog")
    mock_open.getOpenFileName.return_value = (str(mrk_post_path), "")
    frame.findChild(QPushButton, "mrk1_browse").click()
    assert mrk1.file == str(mrk_post_path)
    mocker.stopall()

    # Switch L/R swaps the points
    before = mrk1.points.copy()
    frame.findChild(QPushButton, "mrk1_switch").click()
    assert_array_equal(mrk1.points, before[[1, 0, 2, 4, 3]])

    # per-object visualization controls (Show / Size / Label / Color)
    mrk1_obj = frame.marker_panel.mrk1_obj
    show = frame.findChild(QCheckBox, "mrk1_show")
    show.setChecked(not show.isChecked())
    assert mrk1_obj.visible == show.isChecked()
    mrk1_obj.visible = True  # model change syncs the checkbox back
    assert show.isChecked()

    size = frame.findChild(QDoubleSpinBox, "mrk1_size")
    size.setValue(0.02)
    assert mrk1_obj.point_scale == 0.02

    lbl = frame.findChild(QCheckBox, "mrk1_label")
    lbl.setChecked(True)
    assert mrk1_obj.label
    lbl.setChecked(False)

    # the Color button shows the int RGB triplet (like the old TraitsUI swatch)
    color_btn = frame.findChild(QPushButton, "mrk1_color")
    assert color_btn.text() == "(155,55,55)"  # mrk1's default color
    # ... and routes a newly-picked color into the object, updating the label
    mock_color = mocker.patch("mne_kit_gui._kit2fiff_gui.QColorDialog")
    picked = QColor.fromRgbF(0.1, 0.2, 0.3)
    mock_color.getColor.return_value = picked
    color_btn.click()
    assert_allclose(mrk1_obj.color, (0.1, 0.2, 0.3), atol=1e-3)
    assert color_btn.text() == "(%d,%d,%d)" % (
        picked.red(),
        picked.green(),
        picked.blue(),
    )
    # a cancelled (invalid) color is ignored
    mock_color.getColor.return_value = QColor()
    color_btn.click()
    assert_allclose(mrk1_obj.color, (0.1, 0.2, 0.3), atol=1e-3)
    mocker.stopall()

    # Reorder / Edit / Save As open dialogs -> mock them out
    fake_reorder = mocker.Mock()
    fake_reorder.exec_.return_value = QDialog.Accepted
    fake_reorder.index = [0, 1, 2, 3, 4]
    mocker.patch("mne_kit_gui._marker_gui.ReorderDialog", return_value=fake_reorder)
    frame.findChild(QPushButton, "mrk1_reorder").click()

    fake_edit = mocker.Mock()
    fake_edit.exec_.return_value = QDialog.Rejected
    mocker.patch("mne_kit_gui._marker_gui.EditPointsDialog", return_value=fake_edit)
    frame.findChild(QPushButton, "mrk1_edit").click()

    mock_fd = mocker.patch("mne_kit_gui._marker_gui.QFileDialog")
    mock_fd.getSaveFileName.return_value = ("", "")
    frame.findChild(QPushButton, "mrk1_save").click()
    frame.findChild(QPushButton, "mrk3_save").click()

    # Clear empties the source and disables the gated buttons
    frame.findChild(QPushButton, "mrk1_clear").click()
    assert_array_equal(mrk1.points, np.zeros((5, 3)))
    assert not clear_btn.isEnabled()

    # --- kit2fiff Sources: file fields, basename labels, Use-mrk checkboxes ---
    frame.model.sqd_file = str(sqd_path)
    frame.model.hsp_file = str(hsp_path)
    frame.model.fid_file = str(fid_path)
    # path fields and basename labels now mirror the model (were blank before)
    assert frame._sqd_edit.text() == str(sqd_path)
    assert frame._hsp_edit.text() == str(hsp_path)
    assert frame._fid_edit.text() == str(fid_path)
    assert frame._sqd_fname_label.text() == sqd_path.name
    assert frame._hsp_fname_label.text() == hsp_path.name
    assert frame._fid_fname_label.text() == fid_path.name
    # a Use-mrk checkbox toggles the model (kept >=3 to avoid the marker warning)
    cb2 = frame.findChild(QCheckBox, "use_mrk_2")
    assert cb2.isChecked()
    cb2.setChecked(False)
    assert 2 not in frame.model.use_mrk
    frame.model.use_mrk = [0, 1, 2, 3, 4]  # model change syncs the checkbox back
    assert cb2.isChecked()

    frame.model.clear_all()
    assert_array_equal(frame.marker_panel.mrk1_obj.points, 0)
    assert_array_equal(frame.marker_panel.mrk3_obj.points, 0)
    # clearing also empties the Sources path fields and basename labels
    assert frame._sqd_edit.text() == ""
    assert frame._fid_fname_label.text() == "-"
    frame.close()
