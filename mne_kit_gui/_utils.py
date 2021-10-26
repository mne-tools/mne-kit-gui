"""Helpers."""

# Authors: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD-3-Clause

from distutils.version import LooseVersion
import sys
import warnings

from mne.surface import _normalize_vectors
from mne.utils import check_version, warn


def _check_mayavi_version(min_version='4.3.0'):
    """Check mayavi version."""
    if not check_version('mayavi', min_version):
        raise RuntimeError("Need mayavi >= %s" % min_version)


def _check_pyqt5_version():
    bad = True
    try:
        from PyQt5.Qt import PYQT_VERSION_STR as version
    except Exception:
        version = 'unknown'
    else:
        if LooseVersion(version) >= LooseVersion('5.10'):
            bad = False
    bad &= sys.platform == 'darwin'
    if bad:
        warn('macOS users should use PyQt5 >= 5.10 for GUIs, got %s. '
             'Please upgrade e.g. with:\n\n'
             '    pip install "PyQt5>=5.10,<5.14"\n'
             % (version,))

    return version


def _import_mlab():
    """Quietly import mlab."""
    with warnings.catch_warnings(record=True):
        from mayavi import mlab
    return mlab


def _toggle_mlab_render(fig, render):
    mlab = _import_mlab()
    if mlab.options.backend != 'test':
        fig.scene.disable_render = not render


def _create_mesh_surf(surf, fig=None, scalars=None, vtk_normals=True):
    """Create Mayavi mesh from MNE surf."""
    mlab = _import_mlab()
    x, y, z = surf['rr'].T
    with warnings.catch_warnings(record=True):  # traits
        mesh = mlab.pipeline.triangular_mesh_source(
            x, y, z, surf['tris'], scalars=scalars, figure=fig)
    if vtk_normals:
        mesh = mlab.pipeline.poly_data_normals(mesh)
        mesh.filter.compute_cell_normals = False
        mesh.filter.consistency = False
        mesh.filter.non_manifold_traversal = False
        mesh.filter.splitting = False
    else:
        # make absolutely sure these are normalized for Mayavi
        nn = surf['nn'].copy()
        _normalize_vectors(nn)
        mesh.data.point_data.normals = nn
        mesh.data.cell_data.normals = None
    return mesh


def _oct_glyph(glyph_source, transform):
    from tvtk.api import tvtk
    from tvtk.common import configure_input
    from traits.api import Array
    gs = tvtk.PlatonicSolidSource()

    # Workaround for:
    #  File "mayavi/components/glyph_source.py", line 231, in _glyph_position_changed  # noqa: E501
    #    g.center = 0.0, 0.0, 0.0
    # traits.trait_errors.TraitError: Cannot set the undefined 'center' attribute of a 'TransformPolyDataFilter' object.  # noqa: E501
    class SafeTransformPolyDataFilter(tvtk.TransformPolyDataFilter):
        center = Array(shape=(3,), value=np.zeros(3))

    gs.solid_type = 'octahedron'
    if transform is not None:
        # glyph:             mayavi.modules.vectors.Vectors
        # glyph.glyph:       vtkGlyph3D
        # glyph.glyph.glyph: mayavi.components.glyph.Glyph
        assert transform.shape == (4, 4)
        tr = tvtk.Transform()
        tr.set_matrix(transform.ravel())
        trp = SafeTransformPolyDataFilter()
        configure_input(trp, gs)
        trp.transform = tr
        trp.update()
        gs = trp
    glyph_source.glyph_source = gs


def requires_mayavi(function):
    """Skip a test if package is not available (decorator)."""
    import pytest
    reason = 'Test %s skipped, requires mayavi.' % (function.__name__,)
    try:
        with warnings.catch_warnings(record=True):  # traits
            from mayavi import mlab  # noqa
    except Exception as exc:
        reason += ' Got exception (%s)' % (exc,)
        skip = True
    else:
        skip = False
    return pytest.mark.skipif(skip, reason=reason)(function)
