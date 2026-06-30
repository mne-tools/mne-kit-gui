# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD-3-Clause

from pathlib import Path
from numpy import array
from numpy.testing import assert_allclose, assert_array_equal
import numpy as np
import pytest

import mne
from mne.datasets import testing
from mne.channels import read_dig_fif
from mne.io.constants import FIFF

from mne_kit_gui._file_traits import (
    _expand_path,
    _fs_home_problem,
    _mne_root_problem,
    SurfaceSource,
    FiducialsSource,
    DigSource,
    MRISubjectSource,
    SubjectSelectorPanel,
)


data_path = testing.data_path(download=False)
subjects_dir = data_path / "subjects"
bem_path = subjects_dir / "sample" / "bem" / "sample-1280-bem.fif"
inst_path = data_path / "MEG" / "sample" / "sample_audvis_trunc_raw.fif"
fid_path = Path(mne.__file__).parent / "data" / "fsaverage" / "fsaverage-fiducials.fif"


def test_path_utils(tmp_path):
    """Test _expand_path, _fs_home_problem, and _mne_root_problem."""
    result = _expand_path("~")
    assert Path(result).is_absolute() and "~" not in result

    assert _fs_home_problem(None) is not None
    assert _fs_home_problem("/nonexistent") is not None
    assert _fs_home_problem(str(tmp_path)) is not None  # no fsaverage subdir

    assert _mne_root_problem(None) is not None
    assert _mne_root_problem("/nonexistent") is not None
    assert _mne_root_problem(str(tmp_path)) is not None  # no mne_analyze dir


@testing.requires_testing_data
def test_bem_source(tmp_path, mocker):
    """Test SurfaceSource: normal load, nonexistent path, and invalid file."""
    bem = SurfaceSource()
    assert bem.surf["rr"].shape == (0, 3)
    assert bem.surf["tris"].shape == (0, 3)

    bem.file = str(bem_path)
    assert bem.surf["rr"].shape == (642, 3)
    assert bem.surf["tris"].shape == (1280, 3)

    bem.file = "/nonexistent/path/to/surface.fif"
    assert bem.surf["rr"].shape == (0, 3)

    mocker.patch("mne_kit_gui._file_traits.QMessageBox")
    bad_file = tmp_path / "bad.surf"
    bad_file.write_bytes(b"not a valid surface")
    with pytest.raises(Exception):
        bem.file = str(bad_file)


@testing.requires_testing_data
def test_fiducials_source(tmp_path, mocker):
    """Test FiducialsSource: normal load, empty reset, and bad file."""
    fid = FiducialsSource()
    fid.file = str(fid_path)

    points = array(
        [
            [-0.08061612, -0.02908875, -0.04131077],
            [0.00146763, 0.08506715, -0.03483611],
            [0.08436285, -0.02850276, -0.04127743],
        ]
    )
    assert_allclose(fid.points, points, 1e-6)

    fid.file = ""
    assert fid.points is None

    mocker.patch("mne_kit_gui._file_traits.QMessageBox")
    bad_fif = tmp_path / "bad.fif"
    bad_fif.write_bytes(b"not a valid fif file")
    with pytest.raises(Exception):
        fid.file = str(bad_fif)


@pytest.mark.filterwarnings("ignore:.*does not conform to MNE naming.*:RuntimeWarning")
@testing.requires_testing_data
def test_digitization_source(tmp_path, mocker):
    """Test DigSource: load, points_filter, no-dig file, and reset."""
    inst = DigSource()
    assert inst.inst_fname == "-"

    inst.file = str(inst_path)
    assert inst.inst_dir == str(inst_path.parent)

    # FIFF
    lpa = array([[-7.13766068e-02, 0.00000000e00, 5.12227416e-09]])
    nasion = array([[3.72529030e-09, 1.02605611e-01, 4.19095159e-09]])
    rpa = array([[7.52676800e-02, 0.00000000e00, 5.58793545e-09]])
    assert_allclose(inst.lpa, lpa)
    assert_allclose(inst.nasion, nasion)
    assert_allclose(inst.rpa, rpa)

    # points_filter
    hsp = inst._hsp_points
    assert len(hsp) > 1
    filt = np.ones(len(hsp), dtype=bool)
    filt[0] = False
    inst.points_filter = filt
    assert len(inst.points) == len(hsp) - 1
    assert inst.n_omitted == 1
    inst.points_filter = None

    # DigMontage
    montage = read_dig_fif(inst_path)
    montage_path = tmp_path / "temp_montage.fif"
    montage.save(montage_path)
    inst.file = str(montage_path)
    assert_allclose(inst.lpa, lpa)
    assert_allclose(inst.nasion, nasion)
    assert_allclose(inst.rpa, rpa)

    # EGI MFF
    inst.file = str(data_path / "EGI" / "test_egi.mff")
    assert len(inst.points) == 0
    assert len(inst.eeg_points) in (129, 130)  # old vs new MNE
    assert_allclose(inst.lpa * 1000, [[-67.1, 0, 0]], atol=0.1)
    assert_allclose(inst.nasion * 1000, [[0.0, 103.6, 0]], atol=0.1)
    assert_allclose(inst.rpa * 1000, [[67.1, 0, 0]], atol=0.1)

    # CTF
    inst.file = str(data_path / "CTF" / "testdata_ctf.ds")
    assert len(inst.points) == 0
    assert len(inst.eeg_points) == 8
    assert_allclose(inst.lpa * 1000, [[-74.3, 0.0, 0.0]], atol=0.1)
    assert_allclose(inst.nasion * 1000, [[0.0, 117.7, 0.0]], atol=0.1)
    assert_allclose(inst.rpa * 1000, [[84.9, -0.0, 0.0]], atol=0.1)

    # no-dig .fif → resets to default state
    mocker.patch("mne_kit_gui._file_traits.QMessageBox")
    bad_fif = tmp_path / "nodig.fif"
    bad_fif.write_bytes(b"not a valid fif file")
    inst.file = str(bad_fif)
    assert inst.inst_fname == "-"

    # explicit reset via empty file
    inst.file = str(inst_path)
    inst.file = ""
    assert inst.inst_fname == "-"
    assert inst.inst_dir == ""


def test_dig_source_cardinal_point():
    """Test DigSource._cardinal_point edge cases."""
    inst = DigSource()
    # no _info at all
    assert_array_equal(inst._cardinal_point(FIFF.FIFFV_POINT_LPA), [[0, 0, 0]])
    # _info present but no cardinal entries
    inst._info = {
        "dig": [
            {
                "kind": FIFF.FIFFV_POINT_EEG,
                "ident": 1,
                "r": np.zeros(3),
                "coord_frame": FIFF.FIFFV_COORD_HEAD,
            }
        ]
    }
    assert_array_equal(inst._cardinal_point(FIFF.FIFFV_POINT_LPA), [[0, 0, 0]])


@testing.requires_testing_data
def test_subject_source():
    """Test SubjectSelector."""
    mri = MRISubjectSource()
    mri.subjects_dir = str(subjects_dir)
    assert "sample" in mri.subjects
    mri.subject = "sample"


@testing.requires_testing_data
def test_subject_source_with_fsaverage(tmp_path, monkeypatch):
    """Test SubjectSelector."""
    mri = MRISubjectSource()
    assert not mri.can_create_fsaverage
    pytest.raises(RuntimeError, mri.create_fsaverage)

    mri.subjects_dir = str(tmp_path)
    assert mri.can_create_fsaverage
    assert not (tmp_path / "fsaverage").is_dir()
    # fake FREESURFER_HOME
    monkeypatch.setenv("FREESURFER_HOME", str(data_path))
    mri.create_fsaverage()
    assert (tmp_path / "fsaverage").is_dir()


def test_subject_selector_panel(tmp_path, mocker, monkeypatch):
    """Test SubjectSelectorPanel."""
    mock_msgbox = mocker.patch("mne_kit_gui._file_traits.QMessageBox")
    mocker.patch("mne_kit_gui._file_traits.QProgressDialog")

    model = MRISubjectSource()
    panel = SubjectSelectorPanel(model)

    assert panel.subjects_dir == ""
    assert not panel.can_create_fsaverage
    panel.subject = "test"
    assert panel.subject == "test"

    # setter shows QMessageBox.information when no subjects found
    panel.subjects_dir = str(tmp_path)
    assert panel.subjects_dir == str(tmp_path)
    assert panel.subjects == [""]
    assert panel.can_create_fsaverage
    mock_msgbox.information.assert_called_once()

    # create_fsaverage error path shows QMessageBox.critical and re-raises
    error = RuntimeError("no fs_home")
    monkeypatch.setattr(model, "create_fsaverage", lambda: (_ for _ in ()).throw(error))
    with pytest.raises(RuntimeError, match="no fs_home"):
        panel.create_fsaverage()
    mock_msgbox.critical.assert_called_once()
