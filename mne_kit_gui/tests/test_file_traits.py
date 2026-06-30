# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD-3-Clause

from pathlib import Path

from numpy import array
from numpy.testing import assert_allclose
import pytest

import mne
from mne.datasets import testing
from mne.channels import read_dig_fif

from mne_kit_gui._file_traits import (
    SurfaceSource,
    FiducialsSource,
    DigSource,
    MRISubjectSource,
)


data_path = testing.data_path(download=False)
subjects_dir = data_path / "subjects"
bem_path = subjects_dir / "sample" / "bem" / "sample-1280-bem.fif"
inst_path = data_path / "MEG" / "sample" / "sample_audvis_trunc_raw.fif"
fid_path = Path(mne.__file__).parent / "data" / "fsaverage" / "fsaverage-fiducials.fif"


@testing.requires_testing_data
def test_bem_source():
    """Test SurfaceSource."""
    bem = SurfaceSource()
    assert bem.surf["rr"].shape == (0, 3)
    assert bem.surf["tris"].shape == (0, 3)

    bem.file = str(bem_path)
    assert bem.surf["rr"].shape == (642, 3)
    assert bem.surf["tris"].shape == (1280, 3)


@testing.requires_testing_data
def test_fiducials_source():
    """Test FiducialsSource."""
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


@pytest.mark.filterwarnings("ignore:.*does not conform to MNE naming.*:RuntimeWarning")
@testing.requires_testing_data
def test_digitization_source(tmp_path):
    """Test DigSource."""
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
