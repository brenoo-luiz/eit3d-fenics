from __future__ import annotations

from typing import List

import dolfinx
import ufl

from eit3d.config import LATERAL_TAG


def cosine_current(mesh: dolfinx.mesh.Mesh, k: int) -> ufl.core.expr.Expr:
    x = ufl.SpatialCoordinate(mesh)
    return ufl.cos(k * ufl.atan2(x[1], x[0]))


def cosine_currents(mesh: dolfinx.mesh.Mesh, ks) -> List[ufl.core.expr.Expr]:
    return [cosine_current(mesh, k) for k in ks]


def lateral_measure(mesh: dolfinx.mesh.Mesh, facet_tags: dolfinx.mesh.MeshTags) -> ufl.Measure:
    return ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)(LATERAL_TAG)