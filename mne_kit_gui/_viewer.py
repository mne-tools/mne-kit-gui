# -*- coding: utf-8 -*-
"""PyVista/traitlets GUI visualization elements."""

# Authors: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD-3-Clause

import numpy as np

from traitlets import HasTraits, Any, Bool, Float, Int, List, Unicode, observe

from mne.defaults import DEFAULTS
from mne.surface import _DistanceQuery
from mne.transforms import apply_trans, rotation

from ._utils import _create_mesh_surf, _glyph_geom


def _mm_fmt(x):
    """Format data in units of mm."""
    return "%0.1f" % x


def embed_pyvista_scene(parent_widget):
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
    from pyvistaqt import QtInteractor

    plotter = QtInteractor(parent_widget)
    parent_widget.layout().addWidget(plotter)
    return plotter


def _sph_to_cart_view(azimuth, elevation, distance, focalpoint):
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
    def _scene_changed(self, change):
        if change["new"] is not None:
            self._init_view()

    def _init_view(self):
        scene = self.scene
        if scene is None:
            return
        scene.enable_parallel_projection()
        scene.camera.parallel_scale = self.scale
        self.interaction = self.interaction  # apply deferred interaction

    @observe("scale")
    def _scale_changed(self, change):
        scene = self.scene
        if scene is not None:
            scene.camera.parallel_scale = change["new"]
            scene.render()

    @observe("interaction")
    def _interaction_changed(self, change):
        scene = self.scene
        if scene is None:
            return
        interaction = change["new"]
        kwargs = {"mouse_wheel_zooms": True} if interaction == "terrain" else {}
        getattr(scene, "enable_%s_style" % interaction)(**kwargs)

    def on_set_view(self, view, _=""):
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
        self.scene.render()


class Object(HasTraits):
    """Represent a 3d object in a pyvista scene."""

    points = Any()  # ndarray (n, 3)
    nn = Any()  # ndarray (n, 3)
    name = Unicode()

    scene = Any()  # pyvistaqt.QtInteractor
    src = Any()  # pyvista.PolyData currently displayed

    color = Any((1.0, 1.0, 1.0))  # RGB tuple
    opacity = Float(0.99)
    visible = Bool(True)

    def __init__(self, **kwargs):
        if "points" not in kwargs:
            kwargs["points"] = np.empty((0, 3))
        if "nn" not in kwargs:
            kwargs["nn"] = np.empty((0, 3))
        super().__init__(**kwargs)

    def _update_points(self):
        """Update the location of the plotted points."""
        if self.src is not None and len(self.points) == self.src.n_points:
            self.src.points = self.points
            return True


class PointObject(Object):
    """Represent a group of individual points in a pyvista scene."""

    label = Bool(False)
    label_scale = Float(0.01)
    projectable = Bool(False)
    text3d_labels = List()
    point_scale = Float(10)

    # projection onto a surface
    nearest = Any()  # _DistanceQuery instance
    check_inside = Any()  # _CheckInside instance
    project_to_trans = Any()  # ndarray (4, 4) or None
    project_to_surface = Bool(False)
    orient_to_surface = Bool(False)
    scale_by_distance = Bool(False)
    mark_inside = Bool(False)
    inside_color = Any((0.0, 0.0, 0.0))

    glyph = Any()  # pyvista Actor for the glyphed points
    resolution = Int(8)

    def __init__(self, view="points", has_norm=False, **kwargs):
        assert view in ("points", "cloud", "arrow", "oct")
        self._view = view
        self._has_norm = bool(has_norm)
        if "nearest" not in kwargs:
            kwargs["nearest"] = _DistanceQuery(np.zeros((1, 3)))
        super().__init__(**kwargs)

    @property
    def orientable(self):
        return self.nearest is not None and len(self.nearest.data) > 1

    @observe("scene")
    def _scene_changed(self, change):
        if change["new"] is not None:
            self._plot_points()

    @observe("label")
    def _show_labels(self, change):
        show = change["new"]
        while self.text3d_labels:
            name = self.text3d_labels.pop()
            self.scene.remove_actor(name)

        if show and self.scene is not None and len(self.points) > 0:
            name = "%s-labels" % (self.name or id(self))
            if self._view == "arrow":
                pts, labels = self.points[:1], [self.name]
            else:
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
    def _on_hide(self, change):
        if not change["new"]:
            self.label = False

    def _project(self):
        """Compute the points/normals/scalars to actually render."""
        pts, nn = self.points, self.nn
        scalars = None
        if self._view == "arrow":
            return pts, nn, scalars
        nearest = self.nearest
        if (
            nearest is None
            or len(nearest.data) <= 1
            or len(pts) == 0
            or self.project_to_trans is None
        ):
            return pts, None, scalars
        inv_trans = np.linalg.inv(self.project_to_trans)
        proj_rr = apply_trans(inv_trans, pts)
        idx = nearest.query(proj_rr)[1]
        proj_pts = apply_trans(self.project_to_trans, nearest.data[idx])
        proj_nn = apply_trans(
            self.project_to_trans, self.check_inside.surf["nn"][idx], move=False
        )
        if self.project_to_surface:
            pts = proj_pts
        if self.mark_inside and not self.project_to_surface:
            scalars = (~self.check_inside(proj_rr, verbose=False)).astype(float)
        return pts, proj_nn, scalars

    def _glyph_mode(self):
        if self._view == "oct":
            return "oct"
        if self.project_to_surface or self.orient_to_surface:
            return "cylinder"
        return "sphere"

    def _plot_points(self):
        """(Re)build the glyphed points and push them to the scene."""
        if self.scene is None:
            return
        if self.glyph is not None:
            self.scene.remove_actor(self.glyph)
            self.glyph = None
        self.src = None

        pts, nn, scalars = self._project()
        if len(pts) == 0:
            return

        import pyvista as pv

        cloud = pv.PolyData(np.asarray(pts, float))
        mode = self._glyph_mode()
        use_scalars = scalars is not None and mode != "arrow"
        if use_scalars:
            # vtkGlyph3D copies point_data through to every point of each
            # glyph instance, so this becomes a per-glyph-point scalar.
            cloud.point_data["mark_inside"] = scalars

        if self._view == "arrow":
            geom = pv.Arrow()
        else:
            transform = rotation(0, 0, np.pi / 4) if mode == "oct" else None
            height = DEFAULTS["coreg"]["eegp_height"] if mode == "cylinder" else None
            geom = _glyph_geom(
                mode,
                resolution=self.resolution,
                solid_transform=transform,
                height=height,
            )

        # Orient by normal for 'arrow' glyphs, or for cylinder glyphs (which
        # represent sensors sitting flat against the projected surface).
        orient_by_normal = self._view == "arrow" or mode == "cylinder"
        if orient_by_normal and nn is not None and len(nn) == len(pts):
            cloud.point_data["vec"] = nn
            orient = "vec"
        else:
            orient = False
        mesh = cloud.glyph(
            orient=orient, scale=False, factor=self.point_scale, geom=geom
        )

        kwargs = {"opacity": self.opacity, "culling": "back", "pickable": False}
        if use_scalars:
            from matplotlib.colors import ListedColormap

            kwargs.update(
                scalars="mark_inside",
                cmap=ListedColormap([self.inside_color, self.color]),
                clim=[0.0, 1.0],
                show_scalar_bar=False,
            )
        else:
            kwargs["color"] = self.color
        self.glyph = self.scene.add_mesh(
            mesh, name=self.name or None, render=False, **kwargs
        )
        self.glyph.SetVisibility(self.visible)
        self.src = mesh
        self.scene.render()

    @observe(
        "points",
        "project_to_trans",
        "project_to_surface",
        "mark_inside",
        "nearest",
        "orient_to_surface",
        "resolution",
        "scale_by_distance",
    )
    def _update_projections(self, change):
        """Rebuild the glyphed points after a style-affecting change."""
        self._plot_points()

    @observe("color", "inside_color")
    def _color_changed(self, change):
        self._plot_points()

    @observe("point_scale")
    def _point_scale_changed(self, change):
        self._plot_points()

    @observe("visible")
    def _visible_changed(self, change):
        if self.glyph is not None:
            self.glyph.SetVisibility(change["new"])
            self.scene.render()

    @observe("opacity")
    def _opacity_changed(self, change):
        if self.glyph is not None:
            self.glyph.prop.opacity = change["new"]
            self.scene.render()


class SurfaceObject(Object):
    """Represent a solid object in a pyvista scene.

    Notes
    -----
    Doesn't automatically update plot because update requires both
    :attr:`points` and :attr:`tris`. Call :meth:`plot` after updating both
    attributes.
    """

    rep = Unicode("Surface")  # "Surface" | "Wireframe"
    tris = Any()  # ndarray (n, 3) int

    surf = Any()  # pyvista Actor (front face)
    surf_rear = Any()  # pyvista Actor (back face, optional)
    rear_opacity = Float(1.0)

    def __init__(self, block_behind=False, **kwargs):
        self._block_behind = block_behind
        if "tris" not in kwargs:
            kwargs["tris"] = np.empty((0, 3), int)
        super().__init__(**kwargs)

    @observe("scene")
    def _scene_changed(self, change):
        if change["new"] is not None:
            self.plot()

    def clear(self):  # noqa: D102
        if self.surf is not None:
            self.scene.remove_actor(self.surf)
        if self.surf_rear is not None:
            self.scene.remove_actor(self.surf_rear)
        self.src = None
        self.surf = None
        self.surf_rear = None

    def plot(self):
        """Add the surface to the pyvista scene."""
        if self.scene is None:
            return
        self.clear()

        if self.tris is None or not np.any(self.tris):
            return

        surf_dict = {"rr": self.points, "tris": self.tris}
        mesh = _create_mesh_surf(surf_dict)
        self.src = mesh
        style = "wireframe" if self.rep == "Wireframe" else "surface"

        if self._block_behind:
            self.surf_rear = self.scene.add_mesh(
                mesh,
                color=self.color,
                style=style,
                line_width=1,
                opacity=self.rear_opacity,
                culling="front",
                pickable=False,
                render=False,
            )
            self.surf_rear.SetVisibility(self.visible)

        self.surf = self.scene.add_mesh(
            mesh,
            color=self.color,
            style=style,
            line_width=1,
            opacity=self.opacity,
            culling="back",
            pickable=True,
            render=False,
        )
        self.surf.SetVisibility(self.visible)
        self.scene.render()

    @observe("visible")
    def _sync_visible_surf(self, change):
        if self.surf is not None:
            self.surf.SetVisibility(change["new"])
        if self.surf_rear is not None:
            self.surf_rear.SetVisibility(change["new"])
        if self.scene is not None:
            self.scene.render()

    @observe("color")
    def _sync_color_surf(self, change):
        if self.surf is not None:
            self.surf.prop.color = change["new"]
        if self.surf_rear is not None:
            self.surf_rear.prop.color = change["new"]
        if self.scene is not None:
            self.scene.render()

    @observe("opacity")
    def _sync_opacity_surf(self, change):
        if self.surf is not None:
            self.surf.prop.opacity = change["new"]
            self.scene.render()

    @observe("rear_opacity")
    def _sync_rear_opacity(self, change):
        if self.surf_rear is not None:
            self.surf_rear.prop.opacity = change["new"]
            self.scene.render()

    @observe("points")
    def _points_changed(self, change):
        """Live-update point positions if the mesh topology is unchanged.

        Callers that also change ``tris`` should call :meth:`plot` instead,
        since that requires a full mesh rebuild.
        """
        if self._update_points() and self.src is not None:
            self.src.compute_normals(
                cell_normals=False,
                point_normals=True,
                split_vertices=False,
                consistent_normals=False,
                non_manifold_traversal=False,
                inplace=True,
            )
            if self.scene is not None:
                self.scene.render()
