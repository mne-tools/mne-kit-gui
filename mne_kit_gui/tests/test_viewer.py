# Authors: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD-3-Clause

import numpy as np
from numpy.testing import assert_allclose

from mne.transforms import rotation

from mne_kit_gui._3d import _glyph_geom
from mne_kit_gui._viewer import (
    HeadViewController,
    Object,
    PointObject,
    SurfaceObject,
    _mm_fmt,
    _sph_to_cart_view,
    build_head_view_group,
)


def test_build_head_view_group(qtbot):
    """Test the head-view group's buttons, scale, and interaction controls."""
    from qtpy.QtWidgets import QComboBox, QDoubleSpinBox, QPushButton

    hv = HeadViewController()  # no scene attached
    group = build_head_view_group(hv)
    qtbot.addWidget(group)

    # all four compass buttons exist and calling them is a no-op without a scene
    for name in ("top", "right", "front", "left"):
        btn = group.findChild(QPushButton, "view_%s" % name)
        assert btn is not None
        btn.click()

    # scale spin box is two-way bound to headview.scale
    spin = group.findChild(QDoubleSpinBox, "view_scale")
    assert spin.value() == hv.scale
    spin.setValue(0.25)
    assert hv.scale == 0.25
    hv.scale = 0.5
    assert spin.value() == 0.5

    # interaction combo is two-way bound to headview.interaction
    combo = group.findChild(QComboBox, "view_interaction")
    assert combo.currentText() == hv.interaction == "trackball"
    combo.setCurrentText("terrain")
    assert hv.interaction == "terrain"
    hv.interaction = "trackball"
    assert combo.currentText() == "trackball"


def test_mm_fmt():
    """Test the mm value formatter."""
    assert _mm_fmt(1.234) == "1.2"


def test_glyph_geom():
    """Test _glyph_geom for all glyph modes and an optional transform."""
    # sphere
    sphere = _glyph_geom("sphere", resolution=6)
    assert sphere.GetNumberOfPoints() > 0

    # cylinder (with a height offset)
    cyl = _glyph_geom("cylinder", resolution=6, height=0.01)
    assert cyl.GetNumberOfPoints() > 0

    # octahedron, with and without a solid transform
    oct_plain = _glyph_geom("oct")
    assert oct_plain.GetNumberOfPoints() > 0
    oct_rot = _glyph_geom("oct", solid_transform=rotation(0, 0, np.pi / 4))
    assert oct_rot.GetNumberOfPoints() == oct_plain.GetNumberOfPoints()


def test_objects_without_scene():
    """Toggling traits with no scene attached should be safe no-ops."""
    # HeadViewController: every observer/method short-circuits without a scene
    hvc = HeadViewController()
    hvc.scale = 0.2
    hvc.interaction = "terrain"
    hvc.on_set_view("front")  # returns early, no scene

    # Object base class: _update_points needs an existing src
    base = Object()
    assert base._update_points() is None

    # PointObject: default geometry has a single nearest point -> not orientable
    p = PointObject()
    assert not p.orientable
    p.points = np.zeros((3, 3))
    p.color = (1.0, 0.0, 0.0)
    p.opacity = 0.5
    p.point_scale = 5
    p.resolution = 4
    p.label = True  # no scene -> nothing is added
    p.visible = False  # hides labels via _on_hide

    # SurfaceObject: plot and all the sync observers are no-ops without a scene
    s = SurfaceObject()
    s.plot()
    s.color = (1.0, 0.0, 0.0)
    s.opacity = 0.5
    s.rear_opacity = 0.5
    s.visible = False
    s.points = np.zeros((2, 3))


def test_sph_to_cart_view():
    """Test camera position conversion and view-up degeneracy handling."""
    # looking along +z (elevation 0) should give a degenerate-avoiding view-up
    pos, view_up = _sph_to_cart_view(0.0, 0.0, 1.0, (0.0, 0.0, 0.0))
    assert_allclose(pos, [0.0, 0.0, 1.0], atol=1e-7)
    assert view_up == (0.0, 1.0, 0.0)

    # a non-degenerate elevation uses the z-axis as view-up
    pos, view_up = _sph_to_cart_view(0.0, 90.0, 2.0, (1.0, 0.0, 0.0))
    assert_allclose(pos, [3.0, 0.0, 0.0], atol=1e-7)
    assert view_up == (0.0, 0.0, 1.0)
