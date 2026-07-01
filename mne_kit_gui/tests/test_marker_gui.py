# Authors: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD-3-Clause

from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from pytest_mock import MockerFixture
from pytestqt.qtbot import QtBot

from qtpy.QtWidgets import QDialog

from mne.io.kit import read_mrk

from mne_kit_gui._marker_gui import (
    CombineMarkersModel,
    CombineMarkersPanel,
    EditPointsDialog,
    MarkerPointSource,
    ReorderDialog,
    _write_dig_points,
)

kit_data_dir = Path(__file__).parent / "data"
mrk_pre_path = kit_data_dir / "test_mrk_pre.sqd"
mrk_post_path = kit_data_dir / "test_mrk_post.sqd"
mrk_avg_path = kit_data_dir / "test_mrk.sqd"


def test_combine_markers_model(tmp_path: Path, mocker: MockerFixture) -> None:
    """Test CombineMarkersModel Traits Model."""
    tgt_fname = tmp_path / "test.txt"

    model = CombineMarkersModel()

    # set one marker file
    assert not model.mrk3.can_save
    model.mrk1.file = str(mrk_pre_path)
    assert model.mrk3.can_save
    assert_array_equal(model.mrk3.points, model.mrk1.points)

    # setting second marker file
    model.mrk2.file = str(mrk_pre_path)
    assert_array_equal(model.mrk3.points, model.mrk1.points)

    # set second marker
    model.mrk2.clear()
    model.mrk2.file = str(mrk_post_path)
    assert np.any(model.mrk3.points)
    points_interpolate_mrk1_mrk2 = model.mrk3.points

    # change interpolation method
    model.mrk3.method = "Average"
    mrk_avg = read_mrk(mrk_avg_path)
    assert_array_equal(model.mrk3.points, mrk_avg)

    # clear second marker
    model.mrk2.clear()
    assert_array_equal(model.mrk1.points, model.mrk3.points)

    # I/O
    model.mrk2.file = str(mrk_post_path)
    model.mrk3.save(tgt_fname)
    mrk_io = read_mrk(tgt_fname)
    assert_array_equal(mrk_io, model.mrk3.points)

    # exclude an individual marker
    model.mrk1.use = [1, 2, 3, 4]
    assert_array_equal(model.mrk3.points[0], model.mrk2.points[0])
    assert_array_equal(model.mrk3.points[1:], mrk_avg[1:])

    # reset model
    model.clear()
    model.mrk1.file = str(mrk_pre_path)
    model.mrk2.file = str(mrk_post_path)
    assert_array_equal(model.mrk3.points, points_interpolate_mrk1_mrk2)

    # swap left/right marker points
    swapped = model.mrk1.points[[1, 0, 2, 4, 3]]
    model.mrk1.switch_left_right()
    assert_array_equal(model.mrk1.points, swapped)
    model.mrk1.switch_left_right()

    # transform with only src2 fully used (src1 partial)
    model.mrk3.method = "Transform"
    model.mrk1.use = [0, 1, 2, 3]
    model.mrk2.use = [0, 1, 2, 3, 4]
    assert np.any(model.mrk3.points)
    # ... and the both-partial branch (>=3 shared, each has a unique point)
    model.mrk2.use = [1, 2, 3, 4]
    assert np.any(model.mrk3.points)

    # averaging with each source contributing a unique point
    model.mrk3.method = "Average"
    assert np.any(model.mrk3.points)
    model.mrk3.method = "Transform"
    model.mrk1.use = [0, 1, 2, 3, 4]
    model.mrk2.use = [0, 1, 2, 3, 4]

    # save_as: a missing extension is appended and an existing file prompts
    save_path = tmp_path / "saved_markers"
    mock_fd = mocker.patch("mne_kit_gui._marker_gui.QFileDialog")
    mock_qmb = mocker.patch("mne_kit_gui._marker_gui.QMessageBox")
    mock_fd.getSaveFileName.return_value = (str(save_path), "")
    model.mrk1.save_as()
    assert (tmp_path / "saved_markers.txt").exists()  # ".txt" appended
    # second save: file exists -> overwrite prompt, answered "Yes"
    mock_qmb.question.return_value = mock_qmb.Yes
    model.mrk1.save_as()
    mock_qmb.question.assert_called_once()
    # an empty selection is a no-op
    mock_fd.getSaveFileName.return_value = ("", "")
    model.mrk1.save_as()
    mocker.stopall()

    # error paths report via QMessageBox.critical and fall back to zeros
    mock_msgbox = mocker.patch("mne_kit_gui._marker_gui.QMessageBox")
    # transform needs >=3 shared points
    model.mrk1.use = [0, 1]
    model.mrk2.use = [2, 3]
    assert_array_equal(model.mrk3.points, np.zeros((5, 3)))
    # average needs every point covered by at least one source
    model.mrk3.method = "Average"
    assert_array_equal(model.mrk3.points, np.zeros((5, 3)))
    assert mock_msgbox.critical.call_count >= 2

    # an unreadable marker file resets points to zero and warns
    bad = tmp_path / "bad.sqd"
    bad.write_bytes(b"not a marker file")
    model.mrk1.file = str(bad)
    assert_array_equal(model.mrk1.points, np.zeros((5, 3)))


def test_write_dig_points_errors(tmp_path: Path) -> None:
    """Test _write_dig_points input validation."""
    with pytest.raises(ValueError, match="Points must be of shape"):
        _write_dig_points(tmp_path / "out.txt", np.zeros((5, 4)))
    with pytest.raises(ValueError, match="Unrecognized extension"):
        _write_dig_points(tmp_path / "out.dat", np.zeros((5, 3)))


def test_reorder_dialog(qtbot: QtBot, mocker: MockerFixture) -> None:
    """Test ReorderDialog parsing and MarkerPointSource.reorder/edit."""
    dlg = ReorderDialog()
    qtbot.addWidget(dlg)
    # default order is the identity
    assert dlg.index == [0, 1, 2, 3, 4]
    # non-integer and duplicate inputs are rejected
    dlg._edit.setText("a b c d e")
    assert dlg.index is None
    dlg._edit.setText("0 1 2 3 3")
    assert dlg.index is None
    # invalid input shows a warning when the dialog is accepted
    mock_msgbox = mocker.patch("mne_kit_gui._marker_gui.QMessageBox")
    dlg._try_accept()
    mock_msgbox.warning.assert_called_once()

    # reorder() applies an accepted dialog's permutation to the points
    src = MarkerPointSource()
    src.points = np.arange(15).reshape(5, 3).astype(float)
    fake_dlg = mocker.Mock()
    fake_dlg.exec_.return_value = QDialog.Accepted  # ty: ignore[unresolved-attribute]
    fake_dlg.index = [4, 3, 2, 1, 0]
    mocker.patch("mne_kit_gui._marker_gui.ReorderDialog", return_value=fake_dlg)
    src.reorder()
    assert_array_equal(src.points[0], [12, 13, 14])


def test_edit_points_dialog(qtbot: QtBot) -> None:
    """Test EditPointsDialog reads back the (possibly edited) coordinates."""
    points = np.arange(15).reshape(5, 3).astype(float) / 100
    dlg = EditPointsDialog(points)
    qtbot.addWidget(dlg)
    assert_allclose(dlg.points, points, atol=1e-6)
    # editing a spin box is reflected in the returned points
    dlg._spins[0][0].setValue(0.123456)
    assert_allclose(dlg.points[0, 0], 0.123456, atol=1e-9)


def test_marker_source_edit(mocker: MockerFixture) -> None:
    """Test MarkerPointSource.edit applies an accepted dialog's points."""
    src = MarkerPointSource()
    new_points = np.arange(15).reshape(5, 3).astype(float) / 10

    fake_dlg = mocker.Mock()
    fake_dlg.exec_.return_value = QDialog.Accepted  # ty: ignore[unresolved-attribute]
    fake_dlg.points = new_points
    mocker.patch("mne_kit_gui._marker_gui.EditPointsDialog", return_value=fake_dlg)
    src.edit()
    assert_array_equal(src.points, new_points)

    # a cancelled dialog leaves the points unchanged
    fake_dlg.exec_.return_value = QDialog.Rejected  # ty: ignore[unresolved-attribute]
    src.edit()
    assert_array_equal(src.points, new_points)


def test_combine_markers_panel() -> None:
    """Test CombineMarkersPanel."""
    CombineMarkersPanel()
