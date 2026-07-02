# Authors: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD-3-Clause

import numpy as np
from numpy.testing import assert_allclose

from qtpy.QtWidgets import QComboBox, QDoubleSpinBox, QPushButton

from mne_kit_gui._viewer import (
    HeadViewController,
    Object,
    PointObject,
    _mm_fmt,
    _sph_to_cart_view,
    build_head_view_group,
)
from mne_kit_gui.conftest import FindChild


def test_build_head_view_group(qtbot, find_child: FindChild) -> None:
    """Test the head-view group's buttons, scale, and interaction controls."""
    hv = HeadViewController()  # no scene attached
    group = build_head_view_group(hv)
    qtbot.addWidget(group)

    # all four compass buttons exist and calling them is a no-op without a scene
    for name in ("top", "right", "front", "left"):
        find_child(group, QPushButton, "view_%s" % name).click()

    # scale spin box is two-way bound to headview.scale
    spin = find_child(group, QDoubleSpinBox, "view_scale")
    assert spin.value() == hv.scale
    spin.setValue(0.25)
    assert hv.scale == 0.25
    hv.scale = 0.5
    assert spin.value() == 0.5

    # interaction combo is two-way bound to headview.interaction
    combo = find_child(group, QComboBox, "view_interaction")
    assert combo.currentText() == hv.interaction == "trackball"
    combo.setCurrentText("terrain")
    assert hv.interaction == "terrain"
    hv.interaction = "trackball"
    assert combo.currentText() == "trackball"


def test_mm_fmt() -> None:
    """Test the mm value formatter."""
    assert _mm_fmt(1.234) == "1.2"


def test_objects_without_scene() -> None:
    """Toggling traits with no scene attached should be safe no-ops."""
    # HeadViewController: every observer/method short-circuits without a scene
    hvc = HeadViewController()
    hvc.scale = 0.2
    hvc.interaction = "terrain"
    hvc.on_set_view("front")  # returns early, no scene

    # Object base class: _update_points needs an existing src
    base = Object()
    assert base._update_points() is None

    # PointObject: toggling traits with no scene attached is a safe no-op
    p = PointObject()
    p.points = np.zeros((3, 3))
    p.color = (1.0, 0.0, 0.0)
    p.opacity = 0.5
    p.point_scale = 5
    p.resolution = 4
    p.label = True  # no scene -> nothing is added
    p.visible = False  # hides labels via _on_hide


def test_sph_to_cart_view() -> None:
    """Test camera position conversion and view-up degeneracy handling."""
    # looking along +z (elevation 0) should give a degenerate-avoiding view-up
    pos, view_up = _sph_to_cart_view(0.0, 0.0, 1.0, (0.0, 0.0, 0.0))
    assert_allclose(pos, [0.0, 0.0, 1.0], atol=1e-7)
    assert view_up == (0.0, 1.0, 0.0)

    # a non-degenerate elevation uses the z-axis as view-up
    pos, view_up = _sph_to_cart_view(0.0, 90.0, 2.0, (1.0, 0.0, 0.0))
    assert_allclose(pos, [3.0, 0.0, 0.0], atol=1e-7)
    assert view_up == (0.0, 0.0, 1.0)
