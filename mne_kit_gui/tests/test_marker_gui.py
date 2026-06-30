# Authors: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD-3-Clause

from pathlib import Path

import numpy as np
from numpy.testing import assert_array_equal

from mne.io.kit import read_mrk

from mne_kit_gui._marker_gui import CombineMarkersModel, CombineMarkersPanel

kit_data_dir = Path(__file__).parent / 'data'
mrk_pre_path = kit_data_dir / 'test_mrk_pre.sqd'
mrk_post_path = kit_data_dir / 'test_mrk_post.sqd'
mrk_avg_path = kit_data_dir / 'test_mrk.sqd'


def test_combine_markers_model(tmp_path):
    """Test CombineMarkersModel Traits Model."""
    tgt_fname = tmp_path / 'test.txt'

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
    model.mrk3.method = 'Average'
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


def test_combine_markers_panel():
    """Test CombineMarkersPanel."""
    CombineMarkersPanel()
