# -*- coding: utf-8 -*-
"""PyVista/traitlets GUI visualization elements."""

# Authors: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD-3-Clause

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from vtkmodules.vtkFiltersSources import vtkSphereSource

from qtpy.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from traitlets import HasTraits, Any, Bool, Bunch, Float, Int, List, Unicode, observe


def _mm_fmt(x: float) -> str:
    """Format data in units of mm."""
    return "%0.1f" % x


def embed_pyvista_scene(parent_widget: QWidget) -> QtInteractor:
    """Embed a pyvistaqt scene in a Qt widget.

    Parameters
    ----------
    parent_widget : QWidget
        A widget with a layout already set, to which the plotter's Qt
        control will be added.

    Returns
    -------
    plotter : pyvistaqt.QtInteractor
        The plotter, which is itself a QWidget embedded in
        ``parent_widget`` and is used directly for all 3D plotting
        (``plotter.add_mesh``, ``plotter.camera``, etc.).
    """
    # "three lights" is pyvista's port of mayavi's default "raymond" rig
    # (three white camera lights at intensity 1.0/0.6/0.5); the default
    # vtkLightKit has a weaker (0.75), warm-tinted key light that renders
    # every color noticeably darker than the original mayavi GUI.
    plotter = QtInteractor(parent_widget, lighting="three lights")
    plotter.set_background((0.5, 0.5, 0.5))
    layout = parent_widget.layout()
    assert layout is not None  # the caller sets a layout before calling
    layout.addWidget(plotter)
    return plotter


def _sph_to_cart_view(
    azimuth: float,
    elevation: float,
    distance: float,
    focalpoint: tuple[float, float, float],
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Convert (azimuth, elevation) on a sphere to a camera position.

    Mirrors the convention used by ``mne.transforms._sph_to_cart``:
    azimuth is the angle in the x-y plane from the x-axis, elevation is
    the angle from the z-axis.
    """
    phi = np.deg2rad(azimuth)
    theta = np.deg2rad(elevation)
    position = np.asarray(focalpoint, float) + distance * np.array(
        [
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ]
    )
    # avoid a degenerate view-up vector when looking down/up the z-axis
    view_up = (0.0, 0.0, 1.0) if 5.0 <= abs(elevation) <= 175.0 else (0.0, 1.0, 0.0)
    return position, view_up


_BUTTON_WIDTH = -80
_DEG_WIDTH = -50  # radian floats
_MM_WIDTH = _DEG_WIDTH  # mm floats
_SCALE_WIDTH = _DEG_WIDTH  # scale floats
_INC_BUTTON_WIDTH = -25  # inc/dec buttons
_DEG_STEP_WIDTH = -50
_MM_STEP_WIDTH = _DEG_STEP_WIDTH
_SCALE_STEP_WIDTH = _DEG_STEP_WIDTH
_WEIGHT_WIDTH = -60  # weight floats
_VIEW_BUTTON_WIDTH = -60
# width is optimized for macOS and Linux avoid a horizontal scroll-bar
# even when a vertical one is present
_COREG_WIDTH = -290
_TEXT_WIDTH = -260
_REDUCED_TEXT_WIDTH = _TEXT_WIDTH - 40 * np.sign(_TEXT_WIDTH)
_DIG_SOURCE_WIDTH = _TEXT_WIDTH - 50 * np.sign(_TEXT_WIDTH)
_MRI_FIDUCIALS_WIDTH = _TEXT_WIDTH - 60 * np.sign(_TEXT_WIDTH)
_SHOW_BORDER = True
_RESET_LABEL = "↻"
_RESET_WIDTH = _INC_BUTTON_WIDTH


class HeadViewController(HasTraits):
    """Set head views for the given coordinate system.

    Parameters
    ----------
    system : 'RAS' | 'ALS' | 'ARI'
        Coordinate system described as initials for directions associated with
        the x, y, and z axes. Relevant terms are: Anterior, Right, Left,
        Superior, Inferior.
    """

    system = Unicode("RAS")
    interaction = Unicode("trackball")
    scale = Float(0.16)
    scene = Any()  # pyvistaqt.QtInteractor

    @observe("scene")
    def _scene_changed(self, change: Bunch) -> None:
        if change["new"] is not None:
            self._init_view()

    def _init_view(self) -> None:
        scene = self.scene
        if scene is None:
            return
        scene.enable_parallel_projection()
        scene.camera.parallel_scale = self.scale
        self.interaction = self.interaction  # apply deferred interaction

    @observe("scale")
    def _scale_changed(self, change: Bunch) -> None:
        scene = self.scene
        if scene is not None:
            scene.camera.parallel_scale = change["new"]
            scene.render()

    @observe("interaction")
    def _interaction_changed(self, change: Bunch) -> None:
        scene = self.scene
        if scene is None:
            return
        interaction = change["new"]
        kwargs = {"mouse_wheel_zooms": True} if interaction == "terrain" else {}
        getattr(scene, "enable_%s_style" % interaction)(**kwargs)

    def on_set_view(self, view: str, _: str = "") -> None:
        """Set a named head view ('front', 'left', 'right', 'top')."""
        if self.scene is None:
            return

        system = self.system
        kwargs = {
            "ALS": {
                "front": (0, 90, -90),
                "left": (90, 90, 180),
                "right": (-90, 90, 0),
                "top": (0, 0, -90),
            },
            "RAS": {
                "front": (90.0, 90.0, 180),
                "left": (180, 90, 90),
                "right": (0.0, 90, 270),
                "top": (90, 0, 180),
            },
            "ARI": {
                "front": (0, 90, 90),
                "left": (-90, 90, 180),
                "right": (90, 90, 0),
                "top": (0, 180, 90),
            },
        }
        if system not in kwargs:
            raise ValueError("Invalid system: %r" % system)
        if view not in kwargs[system]:
            raise ValueError("Invalid view: %r" % view)
        az, el, roll = kwargs[system][view]
        distance = self.scene.camera.distance
        position, view_up = _sph_to_cart_view(az, el, distance, (0.0, 0.0, 0.0))
        self.scene.camera_position = [tuple(position), (0.0, 0.0, 0.0), view_up]
        self.scene.camera.roll = roll
        # Fit the data along the chosen view direction, mirroring the auto-fit
        # that mayavi's ``mlab.view(distance=None)`` performs. Without this the
        # parallel-projection scale stays wherever it was set, which can leave
        # the (meter-scale) geometry an invisible speck.
        self.scene.reset_camera()
        self.scale = self.scene.camera.parallel_scale
        self.scene.render()


def build_head_view_group(headview: HeadViewController) -> QGroupBox:
    """Build a Qt "View" group for a :class:`HeadViewController`.

    Parameters
    ----------
    headview : HeadViewController
        The controller whose scene the returned controls drive.

    Returns
    -------
    group : QGroupBox
        A titled group box with the view buttons arranged as a compass (Top
        centered above Right / Front / Left), plus a scale field and a
        trackball/terrain interaction selector.
    """
    group = QGroupBox("View")
    layout = QVBoxLayout(group)

    # compass-style button grid: Top centered above Right / Front / Left
    grid = QGridLayout()
    for name, (r, c) in (
        ("top", (0, 1)),
        ("right", (1, 0)),
        ("front", (1, 1)),
        ("left", (1, 2)),
    ):
        btn = QPushButton(name.capitalize())
        btn.setObjectName("view_%s" % name)
        btn.clicked.connect(lambda *_, v=name: headview.on_set_view(v))
        grid.addWidget(btn, r, c)
    layout.addLayout(grid)

    # scale field + interaction selector
    row = QHBoxLayout()
    row.addWidget(QLabel("Scale:"))
    scale = QDoubleSpinBox()
    scale.setObjectName("view_scale")
    scale.setDecimals(3)
    scale.setRange(0.0, 1e6)
    scale.setSingleStep(max(headview.scale / 20.0, 1e-3))
    scale.setValue(headview.scale)
    scale.valueChanged.connect(lambda v: setattr(headview, "scale", v))
    headview.observe(lambda ch: scale.setValue(ch["new"]), names=["scale"])
    row.addWidget(scale)

    interaction = QComboBox()
    interaction.setObjectName("view_interaction")
    interaction.addItems(["trackball", "terrain"])
    interaction.setCurrentText(headview.interaction)
    interaction.currentTextChanged.connect(
        lambda t: setattr(headview, "interaction", t)
    )
    headview.observe(
        lambda ch: interaction.setCurrentText(ch["new"]), names=["interaction"]
    )
    row.addWidget(interaction)
    layout.addLayout(row)

    return group


class Object(HasTraits):
    """Represent a 3d object in a pyvista scene."""

    points = Any()  # ndarray (n, 3)
    name = Unicode()

    scene = Any()  # pyvistaqt.QtInteractor
    src = Any()  # pyvista.PolyData currently displayed

    color = Any((1.0, 1.0, 1.0))  # RGB tuple
    opacity = Float(0.99)
    visible = Bool(True)

    def __init__(
        self,
        *,
        points: np.ndarray | None = None,
        name: str = "",
        scene: QtInteractor | None = None,
        color: tuple[float, float, float] = (1.0, 1.0, 1.0),
        opacity: float = 0.99,
    ) -> None:
        if points is None:
            points = np.empty((0, 3))
        super().__init__(
            points=points,
            name=name,
            scene=scene,
            color=color,
            opacity=opacity,
        )

    def _update_points(self) -> bool | None:
        """Update the location of the plotted points."""
        if self.src is not None and len(self.points) == self.src.n_points:
            self.src.points = self.points
            return True


class PointObject(Object):
    """Represent a group of individual points in a pyvista scene."""

    label = Bool(False)
    label_scale = Float(0.01)
    text3d_labels = List()
    point_scale = Float(10)

    glyph = Any()  # pyvista Actor for the glyphed points
    resolution = Int(8)

    def __init__(
        self,
        *,
        points: np.ndarray | None = None,
        name: str = "",
        scene: QtInteractor | None = None,
        color: tuple[float, float, float] = (1.0, 1.0, 1.0),
        opacity: float = 0.99,
        point_scale: float = 10,
    ) -> None:
        super().__init__(
            points=points, name=name, scene=scene, color=color, opacity=opacity
        )
        self.point_scale = point_scale

    @observe("scene")
    def _scene_changed(self, change: Bunch) -> None:
        if change["new"] is not None:
            self._plot_points()

    @observe("label")
    def _show_labels(self, change: Bunch) -> None:
        show = change["new"]
        while self.text3d_labels:
            name = self.text3d_labels.pop()
            self.scene.remove_actor(name)

        if show and self.scene is not None and len(self.points) > 0:
            name = "%s-labels" % (self.name or id(self))
            pts = self.points
            labels = [" %i" % i for i in range(len(pts))]
            self.scene.add_point_labels(
                pts,
                labels,
                text_color=self.color,
                shape=None,
                show_points=False,
                font_size=int(self.label_scale * 1000),
                name=name,
                always_visible=True,
            )
            self.text3d_labels.append(name)

    @observe("visible")
    def _on_hide(self, change: Bunch) -> None:
        if not change["new"]:
            self.label = False

    def _plot_points(self) -> None:
        """(Re)build the glyphed points and push them to the scene."""
        if self.scene is None:
            return
        if self.glyph is not None:
            self.scene.remove_actor(self.glyph)
            self.glyph = None
        self.src = None

        pts = self.points
        if len(pts) == 0:
            return

        src = vtkSphereSource()
        src.SetThetaResolution(self.resolution)
        src.SetPhiResolution(self.resolution)
        src.Update()
        geom = pv.PolyData(src.GetOutput())
        geom.compute_normals(
            cell_normals=False,
            point_normals=True,
            split_vertices=False,
            consistent_normals=False,
            non_manifold_traversal=False,
        )

        cloud = pv.PolyData(np.asarray(pts, float))
        mesh = cloud.glyph(
            orient=False, scale=False, factor=self.point_scale, geom=geom
        )

        self.glyph = self.scene.add_mesh(
            mesh,
            opacity=self.opacity,
            culling="back",
            pickable=False,
            name=self.name or None,
            render=False,
            smooth_shading=True,
            color=self.color,
        )
        self.glyph.SetVisibility(self.visible)
        self.src = mesh
        self.scene.render()

    @observe("points", "resolution")
    def _update_projections(self, change: Bunch) -> None:
        """Rebuild the glyphed points after a style-affecting change."""
        self._plot_points()

    @observe("color")
    def _color_changed(self, change: Bunch) -> None:
        self._plot_points()

    @observe("point_scale")
    def _point_scale_changed(self, change: Bunch) -> None:
        self._plot_points()

    @observe("visible")
    def _visible_changed(self, change: Bunch) -> None:
        if self.glyph is not None:
            self.glyph.SetVisibility(change["new"])
            self.scene.render()

    @observe("opacity")
    def _opacity_changed(self, change: Bunch) -> None:
        if self.glyph is not None:
            self.glyph.prop.opacity = change["new"]
            self.scene.render()
