# Authors: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD-3-Clause

from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from qtpy.QtWidgets import QPushButton

from mne.datasets import testing

import mne_kit_gui
from mne_kit_gui._fiducials_gui import MRIHeadWithFiducialsModel


def test_mri_model(subjects_dir_tmp):
    """Test MRIHeadWithFiducialsModel Traits Model."""
    tgt_fname = subjects_dir_tmp / "test-fiducials.fif"

    # Remove the two files that will make the fiducials okay via MNI estimation
    (subjects_dir_tmp / "sample" / "bem" / "sample-fiducials.fif").unlink()
    (subjects_dir_tmp / "sample" / "mri" / "transforms" / "talairach.xfm").unlink()

    model = MRIHeadWithFiducialsModel(subjects_dir=str(subjects_dir_tmp))
    model.subject = "sample"
    assert model.default_fid_fname[-20:] == "sample-fiducials.fif"
    assert not model.can_reset
    assert not model.can_save
    model.lpa = [[-1, 0, 0]]
    model.nasion = [[0, 1, 0]]
    model.rpa = [[1, 0, 0]]
    assert not model.can_reset
    assert model.can_save

    bem_fname = Path(model.bem_high_res.file).name
    assert not model.can_reset
    assert bem_fname == "sample-head-dense.fif"

    model.save(tgt_fname)
    assert model.fid_file == str(tgt_fname)

    # resetting the file should not affect the model's fiducials
    model.fid_file = ""
    assert_array_equal(model.lpa, [[-1, 0, 0]])
    assert_array_equal(model.nasion, [[0, 1, 0]])
    assert_array_equal(model.rpa, [[1, 0, 0]])

    # reset model
    model.lpa = [[0, 0, 0]]
    model.nasion = [[0, 0, 0]]
    model.rpa = [[0, 0, 0]]
    assert_array_equal(model.lpa, [[0, 0, 0]])
    assert_array_equal(model.nasion, [[0, 0, 0]])
    assert_array_equal(model.rpa, [[0, 0, 0]])

    # loading the file should assign the model's fiducials
    model.fid_file = str(tgt_fname)
    assert_array_equal(model.lpa, [[-1, 0, 0]])
    assert_array_equal(model.nasion, [[0, 1, 0]])
    assert_array_equal(model.rpa, [[1, 0, 0]])

    # after changing from file model should be able to reset
    model.nasion = [[1, 1, 1]]
    assert model.can_reset
    model.reset_fiducials()
    assert_array_equal(model.nasion, [[0, 1, 0]])


class _FakePicker:
    def __init__(self, frame):
        self._frame = frame

    def GetActor(self):
        return self._frame.mri_obj.surf


class _OtherPicker:
    def GetActor(self):
        return None


@testing.requires_testing_data
def test_fiducials_frame(qtbot, check_gc, mocker, tmp_path):
    """Test FiducialsFrame GUI, including the 3D scene and picking."""
    subjects_dir = testing.data_path(download=False) / "subjects"

    # WA_DeleteOnClose means this frame's underlying C++ object is gone
    # once we close it below, so don't also register it with qtbot for
    # auto-close at teardown -- that would double-close it. Build it through
    # the public ``fiducials`` convenience function to exercise that wrapper.
    frame = mne_kit_gui.fiducials(
        subject="sample", subjects_dir=str(subjects_dir), block=False
    )

    # the head surface and fiducial point glyphs should be plotted
    assert frame.mri_obj.surf is not None
    assert frame.mri_obj.points.shape[1] == 3
    assert frame.lpa_obj.glyph is not None

    # --- subject selector controls ---
    # the subjects_dir field mirrors the model (previously it stayed blank)
    assert frame._subjects_dir_edit.text() == str(subjects_dir)
    # the fsaverage button's enabled state tracks can_create_fsaverage
    fs_btn = frame.findChild(QPushButton, "create_fsaverage")
    source = frame.model.subject_source
    source.can_create_fsaverage = False
    assert not fs_btn.isEnabled()
    source.can_create_fsaverage = True
    assert fs_btn.isEnabled()
    # clicking invokes create_fsaverage; errors are swallowed after reporting
    mocker.patch.object(frame.spanel, "create_fsaverage")
    fs_btn.click()
    frame.spanel.create_fsaverage.assert_called_once()
    frame.spanel.create_fsaverage.side_effect = RuntimeError("boom")
    fs_btn.click()  # the panel already showed a dialog; no traceback escapes
    # the subjects_dir Browse button routes a chosen directory into the model
    mock_fd = mocker.patch("mne_kit_gui._fiducials_gui.QFileDialog")
    mock_fd.getExistingDirectory.return_value = str(subjects_dir)  # same dir
    frame.findChild(QPushButton, "subjects_dir_browse").click()
    assert frame.spanel.subjects_dir == str(subjects_dir)
    mock_fd.getExistingDirectory.return_value = ""  # cancelled -> no-op
    frame.findChild(QPushButton, "subjects_dir_browse").click()
    mocker.stopall()

    # head views should not raise and should move the camera
    for view in ("front", "left", "right", "top"):
        frame.headview.on_set_view(view)
    # invalid views/systems raise
    with pytest.raises(ValueError, match="Invalid view"):
        frame.headview.on_set_view("nope")

    for interaction in ("trackball", "terrain"):
        frame.headview.interaction = interaction

    # changing the parallel scale re-renders without raising
    frame.headview.scale = 0.2

    # toggling point-object styling exercises the various trait observers
    lpa_obj = frame.lpa_obj
    lpa_obj.point_scale = lpa_obj.point_scale * 2
    lpa_obj.color = (0.0, 1.0, 0.0)
    lpa_obj.opacity = 0.5
    lpa_obj.resolution = 6
    lpa_obj.label = True
    lpa_obj.visible = False  # also hides the labels via _on_hide
    lpa_obj.visible = True

    # ... and the same for the head surface object
    mri_obj = frame.mri_obj
    mri_obj.color = (0.5, 0.5, 0.5)
    mri_obj.opacity = 0.8
    mri_obj.visible = False
    mri_obj.visible = True
    mri_obj.points = mri_obj.points  # live point update (same topology)
    mri_obj.rep = "Wireframe"
    mri_obj.plot()

    pt = frame.mri_obj.points[100]

    # picking while fiducials are locked should be ignored, and the editing
    # groups should be disabled
    before = frame.model.lpa.copy()
    frame.model.lock_fiducials = True
    assert all(not g.isEnabled() for g in frame._lockable_groups)
    fake_picker = _FakePicker(frame)
    frame.panel._on_pick(pt, fake_picker)
    assert_array_equal(frame.model.lpa, before)

    # an empty pick (no intersection) should be ignored without raising, and
    # unlocking re-enables the editing groups
    frame.model.lock_fiducials = False
    assert all(g.isEnabled() for g in frame._lockable_groups)
    frame.panel._on_pick(None, fake_picker)
    assert_array_equal(frame.model.lpa, before)

    # picking on the head surface should move the active fiducial
    frame.panel._on_pick(pt, fake_picker)
    assert_array_equal(np.asarray(frame.model.lpa), [pt])

    # picking something other than the head surface should be ignored
    before = frame.model.nasion.copy()
    frame.panel.set = "Nasion"
    frame.panel._on_pick(pt, _OtherPicker())
    assert_array_equal(frame.model.nasion, before)

    # a change to the BEM source surface propagates to the rendered object
    orig_surf = frame.model.bem_low_res.surf
    frame.model.bem_low_res.surf = {
        "rr": orig_surf["rr"],
        "tris": np.empty((0, 3), int),  # no faces -> clear the object
    }
    assert frame.mri_obj.surf is None
    frame.model.bem_low_res.surf = orig_surf  # restore and re-plot
    assert frame.mri_obj.surf is not None

    # save_as: exercise the file dialog, the ".fif" suffix, and overwrite prompt
    mock_fd = mocker.patch("mne_kit_gui._fiducials_gui.QFileDialog")
    mock_msgbox = mocker.patch("mne_kit_gui._fiducials_gui.QMessageBox")
    save_path = tmp_path / "out_fids"
    mock_fd.getSaveFileName.return_value = (str(save_path), "")
    frame.panel.save_as()
    assert (tmp_path / "out_fids.fif").exists()  # suffix appended
    # file now exists -> overwrite prompt; answer "No" so nothing is rewritten
    mock_msgbox.question.return_value = mock_msgbox.No
    frame.panel.save_as()
    mock_msgbox.question.assert_called_once()
    mock_fd.getSaveFileName.return_value = ("", "")
    frame.panel.save_as()  # empty selection is a no-op

    # the fiducials-file field mirrors the model, and Browse loads a file
    assert frame._fid_file_edit.text() == frame.model.fid_file
    saved_fif = str(tmp_path / "out_fids.fif")
    mock_fd.getOpenFileName.return_value = (saved_fif, "")
    frame.findChild(QPushButton, "fid_file_browse").click()
    assert frame.model.fid_file == saved_fif
    assert frame._fid_file_edit.text() == saved_fif
    mock_fd.getOpenFileName.return_value = ("", "")  # cancelled -> no-op
    frame.findChild(QPushButton, "fid_file_browse").click()
    assert frame.model.fid_file == saved_fif

    # closing with unsaved changes prompts; answer "Discard" so it still closes
    mock_msgbox.question.reset_mock()
    mock_msgbox.question.return_value = mock_msgbox.Discard
    assert frame.model.can_save
    frame.close()
    mock_msgbox.question.assert_called_once()
