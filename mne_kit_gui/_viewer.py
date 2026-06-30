# -*- coding: utf-8 -*-
"""Mayavi/traitlets GUI visualization elements."""

# Authors: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD-3-Clause

import numpy as np

from mayavi.core.ui.mayavi_scene import MayaviScene
from mayavi.mlab import pipeline, text3d
from mayavi.tools.mlab_scene_model import MlabSceneModel
from tvtk.api import tvtk

from traitlets import (HasTraits, Any, Bool, Float, Int, List, Unicode,
                       observe)

from mne.defaults import DEFAULTS
from mne.surface import _DistanceQuery
from mne.transforms import apply_trans, rotation

from ._utils import _create_mesh_surf, _oct_glyph, _toggle_mlab_render


def _mm_fmt(x):
    """Format data in units of mm."""
    return '%0.1f' % x


def embed_mayavi_scene(parent_widget, scene_class=MayaviScene):
    """Embed a mayavi scene in a Qt widget without using traitsui.

    Replicates what ``tvtk.pyface.ui.qt4.scene_editor._SceneEditor``
    does internally, since that machinery is normally only reachable
    through a traitsui ``View``/``Item(editor=SceneEditor(...))``.

    Parameters
    ----------
    parent_widget : QWidget
        A widget with a layout already set, to which the scene's Qt
        control will be added.
    scene_class : type
        The tvtk toolkit scene class to instantiate (e.g. ``MayaviScene``).

    Notes
    -----
    ``MlabSceneModel.activated`` is a traits ``Event`` (write-only, fires
    once, holds no state) so it cannot be queried later to check whether
    a scene is already active. Callers must instead set
    ``mlab_model.activated = True`` themselves once all objects that need
    to react to activation (e.g. via
    ``scene.on_trait_change(callback, 'activated')``) have been
    constructed -- and only after doing so does this helper's sibling
    state, ``mlab_model._gui_activated``, flip to ``True`` so that
    objects constructed *after* activation (see ``Object._scene_changed``
    in this module) can detect they missed the event and plot
    immediately instead.

    Returns
    -------
    mlab_model : MlabSceneModel
        The traits-based scene model (``mlab_model.scene`` is itself,
        per ``SceneModel.scene`` semantics); used to drive mayavi
        plotting via ``mlab_model.mlab`` and to observe
        ``'activated'``.
    """
    mlab_model = MlabSceneModel()
    mlab_model._gui_activated = False
    scene = scene_class(parent_widget)
    mlab_model.scene_editor = scene
    parent_widget.layout().addWidget(scene.control)
    scene.render()
    return mlab_model


def activate_mayavi_scene(mlab_model):
    """Activate a scene created by :func:`embed_mayavi_scene`.

    Fires the (write-only) ``activated`` event for any listeners already
    registered via ``on_trait_change(..., 'activated')``, then flips the
    ``_gui_activated`` flag so objects constructed afterwards can detect
    they missed the event (see ``Object._scene_changed``).
    """
    mlab_model.activated = True
    mlab_model._gui_activated = True


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
_RESET_LABEL = u"↻"
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
    interaction = Unicode('trackball')
    scale = Float(0.16)
    scene = Any()  # MlabSceneModel

    @observe('scene')
    def _scene_changed(self, change):
        scene = change['new']
        if scene is not None:
            # Hook into the traits-based MlabSceneModel activation event.
            scene.on_trait_change(self._init_view, 'activated')
            if getattr(scene, '_gui_activated', False):
                self._init_view()

    def _init_view(self):
        scene = self.scene
        if scene is None:
            return
        scene.parallel_projection = True
        if scene.renderer:
            scene.camera.parallel_scale = self.scale
            scene.on_trait_change(self._sync_scale_to_camera, 'activated')
            self.interaction = self.interaction  # apply deferred interaction

    @observe('scale')
    def _scale_changed(self, change):
        scene = self.scene
        if scene is not None and scene.renderer:
            scene.camera.parallel_scale = change['new']
            scene.render()

    @observe('interaction')
    def _interaction_changed(self, change):
        interaction = change['new']
        scene = self.scene
        if scene is None or scene.interactor is None:
            return
        self.on_set_view('front', '')
        scene.mlab.draw()
        scene.interactor.interactor_style = (
            tvtk.InteractorStyleTerrain() if interaction == 'terrain' else
            tvtk.InteractorStyleTrackballCamera()
        )
        self.on_set_view('front', '')
        scene.mlab.draw()

    def _sync_scale_to_camera(self):
        if self.scene and self.scene.renderer:
            self.scene.camera.parallel_scale = self.scale

    def on_set_view(self, view, _=''):
        """Set a named head view ('front', 'left', 'right', 'top')."""
        if self.scene is None:
            return

        system = self.system
        kwargs = dict(ALS=dict(front=(0, 90, -90),
                               left=(90, 90, 180),
                               right=(-90, 90, 0),
                               top=(0, 0, -90)),
                      RAS=dict(front=(90., 90., 180),
                               left=(180, 90, 90),
                               right=(0., 90, 270),
                               top=(90, 0, 180)),
                      ARI=dict(front=(0, 90, 90),
                               left=(-90, 90, 180),
                               right=(90, 90, 0),
                               top=(0, 180, 90)))
        if system not in kwargs:
            raise ValueError("Invalid system: %r" % system)
        if view not in kwargs[system]:
            raise ValueError("Invalid view: %r" % view)
        az, el, roll = kwargs[system][view]
        self.scene.mlab.view(
            azimuth=az,
            elevation=el,
            roll=roll,
            distance=None,
            reset_roll=True,
            focalpoint=(0., 0., 0.),
            figure=self.scene.mayavi_scene,
        )


class Object(HasTraits):
    """Represent a 3d object in a mayavi scene."""

    points = Any()  # ndarray (n, 3)
    nn = Any()      # ndarray (n, 3)
    name = Unicode()

    scene = Any()   # MlabSceneModel
    src = Any()     # VTKDataSource

    color = Any((1., 1., 1.))   # RGB tuple
    opacity = Float(0.99)
    visible = Bool(True)

    def __init__(self, **kwargs):
        if 'points' not in kwargs:
            kwargs['points'] = np.empty((0, 3))
        if 'nn' not in kwargs:
            kwargs['nn'] = np.empty((0, 3))
        super().__init__(**kwargs)

    def _update_points(self):
        """Update the location of the plotted points."""
        if hasattr(self.src, 'data'):
            self.src.data.points = self.points
            return True


class PointObject(Object):
    """Represent a group of individual points in a mayavi scene."""

    label = Bool(False)
    label_scale = Float(0.01)
    projectable = Bool(False)
    text3d_labels = List()
    point_scale = Float(10)

    # projection onto a surface
    nearest = Any()          # _DistanceQuery instance
    check_inside = Any()     # _CheckInside instance
    project_to_trans = Any()  # ndarray (4, 4) or None
    project_to_surface = Bool(False)
    orient_to_surface = Bool(False)
    scale_by_distance = Bool(False)
    mark_inside = Bool(False)
    inside_color = Any((0., 0., 0.))

    glyph = Any()   # mayavi Glyph module
    resolution = Int(8)

    def __init__(self, view='points', has_norm=False, **kwargs):
        assert view in ('points', 'cloud', 'arrow', 'oct')
        self._view = view
        self._has_norm = bool(has_norm)
        if 'nearest' not in kwargs:
            kwargs['nearest'] = _DistanceQuery(np.zeros((1, 3)))
        super().__init__(**kwargs)

    @property
    def orientable(self):
        return self.nearest is not None and len(self.nearest.data) > 1

    @observe('scene')
    def _scene_changed(self, change):
        scene = change['new']
        if scene is not None:
            scene.on_trait_change(self._plot_points, 'activated')
            if getattr(scene, '_gui_activated', False):
                self._plot_points()

    @observe('label')
    def _show_labels(self, change):
        show = change['new']
        _toggle_mlab_render(self, False)
        while self.text3d_labels:
            text = self.text3d_labels.pop()
            text.remove()

        if show and self.src is not None and len(self.src.data.points) > 0:
            fig = self.scene.mayavi_scene
            if self._view == 'arrow':
                x, y, z = self.src.data.points[0]
                self.text3d_labels.append(text3d(
                    x, y, z, self.name, scale=self.label_scale,
                    color=self.color, figure=fig))
            else:
                for i, (x, y, z) in enumerate(np.array(self.src.data.points)):
                    self.text3d_labels.append(text3d(
                        x, y, z, ' %i' % i, scale=self.label_scale,
                        color=self.color, figure=fig))
        _toggle_mlab_render(self, True)

    @observe('visible')
    def _on_hide(self, change):
        if not change['new']:
            self.label = False

    def _plot_points(self):
        """Add the points to the mayavi pipeline."""
        if self.scene is None:
            return
        if hasattr(self.glyph, 'remove'):
            self.glyph.remove()
        if hasattr(self.src, 'remove'):
            self.src.remove()

        _toggle_mlab_render(self, False)
        x, y, z = self.points.T if len(self.points) else ([], [], [])
        fig = self.scene.mayavi_scene
        scatter = pipeline.scalar_scatter(x, y, z, fig=fig)
        if not scatter.running:
            return
        mode = {'cloud': 'sphere', 'points': 'sphere', 'oct': 'sphere'}.get(
            self._view, self._view)
        assert mode in ('sphere', 'arrow')
        glyph = pipeline.glyph(scatter, color=self.color,
                               figure=fig, scale_factor=self.point_scale,
                               opacity=1., resolution=self.resolution,
                               mode=mode)
        if self._view == 'oct':
            _oct_glyph(glyph.glyph.glyph_source, rotation(0, 0, np.pi / 4))
        glyph.actor.property.backface_culling = True
        glyph.glyph.glyph.vector_mode = 'use_normal'
        glyph.glyph.glyph.clamping = False
        if mode == 'arrow':
            glyph.glyph.glyph_source.glyph_position = 'tail'

        glyph.actor.mapper.color_mode = 'map_scalars'
        glyph.actor.mapper.scalar_mode = 'use_point_data'
        glyph.actor.mapper.use_lookup_table_scalar_range = False

        self.src = scatter
        self.glyph = glyph

        # Apply current trait values to the mayavi objects
        glyph.glyph.glyph.scale_factor = self.point_scale
        glyph.actor.property.color = self.color
        glyph.visible = self.visible
        glyph.actor.property.opacity = self.opacity
        glyph.actor.mapper.scalar_visibility = self.mark_inside

        self.observe(self._sync_point_scale, names=['point_scale'])
        self.observe(self._sync_color, names=['color'])
        self.observe(self._sync_visible_glyph, names=['visible'])
        self.observe(self._sync_opacity, names=['opacity'])
        self.observe(self._sync_mark_inside, names=['mark_inside'])

        self._update_marker_scaling()
        self._update_marker_type()
        self._update_colors()
        _toggle_mlab_render(self, True)

    def _sync_point_scale(self, change):
        if self.glyph is not None:
            self.glyph.glyph.glyph.scale_factor = change['new']

    def _sync_color(self, change):
        if self.glyph is not None:
            self.glyph.actor.property.color = change['new']
        self._update_colors()

    def _sync_visible_glyph(self, change):
        if self.glyph is not None:
            self.glyph.visible = change['new']

    def _sync_opacity(self, change):
        if self.glyph is not None:
            self.glyph.actor.property.opacity = change['new']

    def _sync_mark_inside(self, change):
        if self.glyph is not None:
            self.glyph.actor.mapper.scalar_visibility = change['new']

    def _get_nearest(self, proj_rr):
        idx = self.nearest.query(proj_rr)[1]
        proj_pts = apply_trans(
            self.project_to_trans, self.nearest.data[idx])
        proj_nn = apply_trans(
            self.project_to_trans, self.check_inside.surf['nn'][idx],
            move=False)
        return proj_pts, proj_nn

    @observe('points', 'project_to_trans', 'project_to_surface',
             'mark_inside', 'nearest')
    def _update_projections(self, change):
        """Update the styles of the plotted points."""
        if not hasattr(self.src, 'data'):
            return
        if self._view == 'arrow':
            self.src.data.point_data.normals = self.nn
            self.src.data.point_data.update()
            return
        nearest = self.nearest
        if nearest is None or len(nearest.data) <= 1 or len(self.points) == 0:
            return

        pts = self.points
        inv_trans = np.linalg.inv(self.project_to_trans)
        proj_rr = apply_trans(inv_trans, self.points)
        proj_pts, proj_nn = self._get_nearest(proj_rr)
        vec = pts - proj_pts
        if self.project_to_surface:
            pts = proj_pts
        nn = proj_nn
        if self.mark_inside and not self.project_to_surface:
            scalars = (~self.check_inside(proj_rr, verbose=False)).astype(int)
        else:
            scalars = np.ones(len(pts))
        dist = np.linalg.norm(vec, axis=-1, keepdims=True)
        self.src.data.point_data.normals = (250 * dist + 1) * nn
        self.src.data.point_data.scalars = scalars
        self.glyph.actor.mapper.scalar_range = [0., 1.]
        self.src.data.points = pts
        self.src.data.point_data.update()

    @observe('color', 'inside_color')
    def _inside_color_changed(self, change):
        self._update_colors()

    def _update_colors(self):
        if self.glyph is None:
            return
        inside = np.array(self.inside_color)
        if np.mean(np.abs(inside - 0.5)) < 0.2:
            inside.fill(0.)
        else:
            inside = 1 - inside
        colors = np.array([tuple(inside) + (1,),
                           tuple(self.color) + (1,)]) * 255.
        self.glyph.module_manager.scalar_lut_manager.lut.table = colors

    @observe('project_to_surface', 'orient_to_surface')
    def _marker_type_changed(self, change):
        self._update_marker_type()

    def _update_marker_type(self):
        if self.glyph is None or self._view == 'arrow':
            return
        defaults = DEFAULTS['coreg']
        gs = self.glyph.glyph.glyph_source
        res = getattr(gs.glyph_source, 'theta_resolution',
                      getattr(gs.glyph_source, 'resolution', None))
        if res is None:
            return
        if self.project_to_surface or self.orient_to_surface:
            gs.glyph_source = tvtk.CylinderSource()
            gs.glyph_source.height = defaults['eegp_height']
            gs.glyph_source.center = (0., -defaults['eegp_height'], 0)
            gs.glyph_source.resolution = res
        else:
            gs.glyph_source = tvtk.SphereSource()
            gs.glyph_source.phi_resolution = res
            gs.glyph_source.theta_resolution = res

    @observe('scale_by_distance', 'project_to_surface')
    def _marker_scaling_changed(self, change):
        self._update_marker_scaling()

    def _update_marker_scaling(self):
        if self.glyph is None:
            return
        if self.scale_by_distance and not self.project_to_surface:
            self.glyph.glyph.scale_mode = 'scale_by_vector'
        else:
            self.glyph.glyph.scale_mode = 'data_scaling_off'

    @observe('resolution')
    def _resolution_changed(self, change):
        if not self.glyph:
            return
        new = change['new']
        gs = self.glyph.glyph.glyph_source.glyph_source
        if isinstance(gs, tvtk.SphereSource):
            gs.phi_resolution = new
            gs.theta_resolution = new
        elif isinstance(gs, tvtk.CylinderSource):
            gs.resolution = new
        else:  # ArrowSource
            gs.tip_resolution = new
            gs.shaft_resolution = new


class SurfaceObject(Object):
    """Represent a solid object in a mayavi scene.

    Notes
    -----
    Doesn't automatically update plot because update requires both
    :attr:`points` and :attr:`tris`. Call :meth:`plot` after updating both
    attributes.
    """

    rep = Unicode("Surface")  # "Surface" | "Wireframe"
    tris = Any()    # ndarray (n, 3) int

    surf = Any()       # mayavi Surface module
    surf_rear = Any()  # mayavi Surface module (back face, optional)
    rear_opacity = Float(1.)

    def __init__(self, block_behind=False, **kwargs):
        self._block_behind = block_behind
        self._deferred_tris_update = False
        if 'tris' not in kwargs:
            kwargs['tris'] = np.empty((0, 3), int)
        super().__init__(**kwargs)

    @observe('scene')
    def _scene_changed(self, change):
        scene = change['new']
        if scene is not None:
            scene.on_trait_change(self.plot, 'activated')
            if getattr(scene, '_gui_activated', False):
                self.plot()

    def clear(self):  # noqa: D102
        if hasattr(self.src, 'remove'):
            self.src.remove()
        if hasattr(self.surf, 'remove'):
            self.surf.remove()
        if hasattr(self.surf_rear, 'remove'):
            self.surf_rear.remove()
        self.src = None
        self.surf = None
        self.surf_rear = None

    def plot(self):
        """Add the surface to the mayavi pipeline."""
        _scale = self.scene.camera.parallel_scale
        self.clear()

        if self.tris is None or not np.any(self.tris):
            return

        fig = self.scene.mayavi_scene
        surf_dict = dict(rr=self.points, tris=self.tris)
        normals = _create_mesh_surf(surf_dict, fig=fig)
        self.src = normals.parent
        rep = 'wireframe' if self.rep == 'Wireframe' else 'surface'

        if self._block_behind:
            surf_rear = pipeline.surface(
                normals, figure=fig, color=self.color, representation=rep,
                line_width=1)
            surf_rear.actor.property.frontface_culling = True
            self.surf_rear = surf_rear
            surf_rear.actor.property.color = self.color
            surf_rear.visible = self.visible
            surf_rear.actor.property.opacity = self.rear_opacity

        surf = pipeline.surface(
            normals, figure=fig, color=self.color, representation=rep,
            line_width=1)
        surf.actor.property.backface_culling = True
        surf.visible = self.visible
        surf.actor.property.color = self.color
        surf.actor.property.opacity = self.opacity
        self.surf = surf

        self.observe(self._sync_visible_surf, names=['visible'])
        self.observe(self._sync_color_surf, names=['color'])
        self.observe(self._sync_opacity_surf, names=['opacity'])
        if self._block_behind:
            self.observe(self._sync_rear_opacity, names=['rear_opacity'])

        self.scene.camera.parallel_scale = _scale

    def _sync_visible_surf(self, change):
        if self.surf is not None:
            self.surf.visible = change['new']
        if self.surf_rear is not None:
            self.surf_rear.visible = change['new']

    def _sync_color_surf(self, change):
        if self.surf is not None:
            self.surf.actor.property.color = change['new']
        if self.surf_rear is not None:
            self.surf_rear.actor.property.color = change['new']

    def _sync_opacity_surf(self, change):
        if self.surf is not None:
            self.surf.actor.property.opacity = change['new']

    def _sync_rear_opacity(self, change):
        if self.surf_rear is not None:
            self.surf_rear.actor.property.opacity = change['new']

    @observe('tris')
    def _tris_changed(self, change):
        self._deferred_tris_update = True

    @observe('points')
    def _points_changed(self, change):
        if self._deferred_tris_update and self.src is not None:
            self.src.data.polys = None
        if Object._update_points(self):
            if self._deferred_tris_update:
                self.src.data.polys = self.tris
                self._deferred_tris_update = False
            self.src.update()
