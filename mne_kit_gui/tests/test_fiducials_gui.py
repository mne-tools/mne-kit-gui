# Authors: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD-3-Clause

from pathlib import Path

import numpy as np
from numpy.testing import assert_array_equal

from mne.datasets import testing

from mne_kit_gui._fiducials_gui import MRIHeadWithFiducialsModel
from mne_kit_gui._fiducials_gui import FiducialsFrame


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


@testing.requires_testing_data
def test_fiducials_frame(qtbot):
    """Test FiducialsFrame GUI, including the 3D scene and picking."""
    subjects_dir = testing.data_path(download=False) / "subjects"

    # WA_DeleteOnClose means this frame's underlying C++ object is gone
    # once we close it below, so don't also register it with qtbot for
    # auto-close at teardown -- that would double-close it.
    frame = FiducialsFrame(subject="sample", subjects_dir=str(subjects_dir))

    # the head surface and fiducial point glyphs should be plotted
    assert frame.mri_obj.surf is not None
    assert frame.mri_obj.points.shape[1] == 3
    assert frame.lpa_obj.glyph is not None

    # head views should not raise and should move the camera
    for view in ("front", "left", "right", "top"):
        frame.headview.on_set_view(view)

    for interaction in ("trackball", "terrain"):
        frame.headview.interaction = interaction

    pt = frame.mri_obj.points[100]

    class _FakePicker:
        def GetActor(self):
            return frame.mri_obj.surf

    class _OtherPicker:
        def GetActor(self):
            return None

    # picking while fiducials are locked should be ignored
    before = frame.model.lpa.copy()
    frame.model.lock_fiducials = True
    frame.panel._on_pick(pt, _FakePicker())
    assert_array_equal(frame.model.lpa, before)

    # an empty pick (no intersection) should be ignored without raising
    frame.model.lock_fiducials = False
    frame.panel._on_pick(None, _FakePicker())
    assert_array_equal(frame.model.lpa, before)

    # picking on the head surface should move the active fiducial
    frame.panel._on_pick(pt, _FakePicker())
    assert_array_equal(np.asarray(frame.model.lpa), [pt])

    # picking something other than the head surface should be ignored
    before = frame.model.nasion.copy()
    frame.panel.set = "Nasion"
    frame.panel._on_pick(pt, _OtherPicker())
    assert_array_equal(frame.model.nasion, before)

    frame.model.can_save = False  # avoid the unsaved-changes dialog on close
    frame.close()
