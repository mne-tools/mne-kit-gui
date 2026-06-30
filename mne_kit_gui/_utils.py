"""Helpers."""

# Authors: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD-3-Clause

import numpy as np

from mne.surface import _normalize_vectors


def _create_mesh_surf(surf, compute_normals=True):
    """Create a pyvista PolyData mesh from an MNE surf dict.

    Parameters
    ----------
    surf : dict
        Dict with keys 'rr' (n, 3) and 'tris' (n, 3), and optionally 'nn'.
    compute_normals : bool
        If True, compute (smooth) normals with vtk. If False, use the
        normals in ``surf['nn']`` if present.

    Returns
    -------
    mesh : pyvista.PolyData
        The mesh.
    """
    import pyvista as pv

    vertices = np.asarray(surf["rr"], float)
    tris = np.asarray(surf["tris"], int)
    faces = np.c_[np.full(len(tris), 3), tris]
    mesh = pv.PolyData(vertices, faces)
    if compute_normals:
        mesh.compute_normals(
            cell_normals=False,
            point_normals=True,
            split_vertices=False,
            consistent_normals=False,
            non_manifold_traversal=False,
            inplace=True,
        )
    elif surf.get("nn") is not None:
        nn = np.array(surf["nn"], float)
        _normalize_vectors(nn)
        mesh.point_data["Normals"] = nn
        mesh.GetPointData().SetActiveNormals("Normals")
    return mesh


def _glyph_geom(mode, resolution=8, solid_transform=None, height=None):
    """Build unit glyph geometry (a vtkPolyData) for a given mode.

    Parameters
    ----------
    mode : 'sphere' | 'cylinder' | 'oct'
        The kind of glyph to build.
    resolution : int
        Resolution for 'sphere'/'cylinder' sources.
    solid_transform : ndarray, shape (4, 4) | None
        Optional transform applied to the (typically 'oct') source, e.g. to
        rotate it into a more pleasing orientation.
    height : float | None
        For 'cylinder', the cylinder height; the cylinder is also offset
        so that one face sits at the origin (so e.g. an EEG-sensor-like
        disc glyph appears to sit "on" the point rather than centered
        through it).

    Returns
    -------
    geom : vtkPolyData
        The unit glyph geometry, suitable for use as the ``geom`` argument
        to :meth:`pyvista.PolyDataFilters.glyph`.
    """
    from vtkmodules.vtkFiltersSources import (
        vtkSphereSource,
        vtkCylinderSource,
        vtkPlatonicSolidSource,
    )

    if mode == "sphere":
        src = vtkSphereSource()
        src.SetThetaResolution(resolution)
        src.SetPhiResolution(resolution)
    elif mode == "cylinder":
        src = vtkCylinderSource()
        src.SetResolution(resolution)
        if height is not None:
            src.SetHeight(height)
            src.SetCenter(0.0, -height / 2.0, 0.0)
    elif mode == "oct":
        src = vtkPlatonicSolidSource()
        src.SetSolidTypeToOctahedron()
    else:
        raise ValueError("mode must be sphere, cylinder, or oct, got %r" % (mode,))
    src.Update()
    geom = src.GetOutput()
    if solid_transform is not None:
        from vtkmodules.vtkCommonTransforms import vtkTransform
        from vtkmodules.vtkFiltersGeneral import vtkTransformFilter

        assert solid_transform.shape == (4, 4)
        tr = vtkTransform()
        tr.SetMatrix(solid_transform.astype(np.float64).ravel())
        trp = vtkTransformFilter()
        trp.SetInputData(geom)
        trp.SetTransform(tr)
        trp.Update()
        geom = trp.GetOutput()
    return geom
