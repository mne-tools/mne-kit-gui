"""Helpers."""

# Authors: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD-3-Clause

import numpy as np
import pyvista as pv


def _create_mesh_surf(surf):
    """Create a pyvista PolyData mesh from an MNE surf dict.

    Parameters
    ----------
    surf : dict
        Dict with keys 'rr' (n, 3) and 'tris' (n, 3).

    Returns
    -------
    mesh : pyvista.PolyData
        The mesh.
    """
    vertices = np.asarray(surf["rr"], float)
    tris = np.asarray(surf["tris"], int)
    faces = np.c_[np.full(len(tris), 3), tris]
    mesh = pv.PolyData(vertices, faces)
    mesh.compute_normals(
        cell_normals=False,
        point_normals=True,
        split_vertices=False,
        consistent_normals=False,
        non_manifold_traversal=False,
        inplace=True,
    )
    return mesh
