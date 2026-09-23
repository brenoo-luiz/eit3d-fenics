from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
from pathlib import Path
from typing import Optional, Tuple

import dolfinx
import dolfinx.io
import gmsh
import numpy as np
from dolfinx.io import gmsh as gmshio
from mpi4py import MPI

from eit3d.config import (
    CACHE_DIR, TOP_TAG, BOTTOM_TAG, LATERAL_TAG, VOLUME_TAG,
    MeshConfig,
)

logger = logging.getLogger(__name__)


def _mesh_cache_path(cfg: MeshConfig) -> Path:
    """
    Returns a unique cache path based on a hash of the MeshConfig.

    This ensures that if radius, height or mesh sizes change, the old
    cached mesh is never loaded silently — a new file is generated instead.
    """
    key = hashlib.sha1(
        json.dumps(dataclasses.asdict(cfg), sort_keys=True).encode()
    ).hexdigest()[:10]
    return CACHE_DIR / f"mesh_{key}.xdmf"


class CylinderMesh:
    """
    3D cylinder mesh generated with Gmsh, cached to disk as XDMF.

    The cache filename is derived from a hash of MeshConfig, so changing
    any parameter (radius, height, size_max, size_min) automatically
    triggers regeneration instead of silently loading a stale mesh.

    First run:  generates mesh (~90s) and saves to cache/mesh_<hash>.xdmf
    Next runs:  loads from disk (~1s)
    force=True: regenerates even if cache exists
    """

    def __init__(
        self,
        config   : MeshConfig,
        comm     : MPI.Comm = MPI.COMM_WORLD,
        force    : bool     = False,
    ) -> None:
        self._config    = config
        self._comm      = comm
        self._force     = force
        self._mesh_file = _mesh_cache_path(config)
        self._mesh       : Optional[dolfinx.mesh.Mesh]     = None
        self._facet_tags : Optional[dolfinx.mesh.MeshTags] = None
        self._mesh_file.parent.mkdir(parents=True, exist_ok=True)

    def get(self) -> Tuple[dolfinx.mesh.Mesh, dolfinx.mesh.MeshTags]:
        """Return (mesh, facet_tags), loading from cache if available."""
        if self._mesh is not None:
            return self._mesh, self._facet_tags
        if self._mesh_file.exists() and not self._force:
            self._load()
        else:
            self._generate()
            self._save()
        return self._mesh, self._facet_tags

    def is_cached(self) -> bool:
        return self._mesh_file.exists()

    @property
    def mesh_file(self) -> Path:
        return self._mesh_file

    @property
    def n_cells(self) -> int:
        self.get()
        return self._mesh.topology.index_map(3).size_global

    @property
    def n_vertices(self) -> int:
        self.get()
        return self._mesh.topology.index_map(0).size_global

    def _generate(self) -> None:
        logger.info(
            "Generating mesh (config hash: %s) ...",
            self._mesh_file.stem,
        )
        gmsh.finalize()
        gmsh.initialize()

        try:
            half_h = self._config.height / 2.0
            gmsh.model.occ.addCylinder(
                0, 0, -half_h,
                0, 0, self._config.height,
                self._config.radius,
            )
            gmsh.model.occ.synchronize()

            top_surfs, bot_surfs, lat_surfs = [], [], []
            tol = 0.1 * self._config.size_max
            for s in gmsh.model.getEntities(dim=2):
                bb = gmsh.model.getBoundingBox(s[0], s[1])
                z_min, z_max = bb[2], bb[5]
                if z_min > half_h - tol:
                    top_surfs.append(s[1])
                elif z_max < -half_h + tol:
                    bot_surfs.append(s[1])
                else:
                    lat_surfs.append(s[1])

            gmsh.model.addPhysicalGroup(2, top_surfs,  tag=TOP_TAG)
            gmsh.model.addPhysicalGroup(2, bot_surfs,  tag=BOTTOM_TAG)
            gmsh.model.addPhysicalGroup(2, lat_surfs,  tag=LATERAL_TAG)
            gmsh.model.addPhysicalGroup(
                3, [v[1] for v in gmsh.model.getEntities(dim=3)],
                tag=VOLUME_TAG,
            )

            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", self._config.size_max)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMin", self._config.size_min)
            gmsh.option.setNumber("Mesh.Algorithm",      6)
            gmsh.option.setNumber("Mesh.Algorithm3D",    1)
            gmsh.option.setNumber("Mesh.Optimize",       1)
            gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)

            gmsh.model.mesh.generate(3)
            gmsh.model.mesh.optimize("Netgen")
            gmsh.model.mesh.optimize("Relocate3D")

            mesh_data        = gmshio.model_to_mesh(gmsh.model, self._comm, 0, gdim=3)
            self._mesh       = mesh_data.mesh
            self._facet_tags = mesh_data.facet_tags
        finally:
            gmsh.finalize()

        logger.info(
            "Mesh generated: %d cells, %d vertices",
            self.n_cells, self.n_vertices,
        )

    def _save(self) -> None:
        logger.info("Saving mesh to %s ...", self._mesh_file)
        self._mesh.topology.create_connectivity(
            self._mesh.topology.dim - 1, self._mesh.topology.dim,
        )
        with dolfinx.io.XDMFFile(self._comm, str(self._mesh_file), "w") as xdmf:
            xdmf.write_mesh(self._mesh)
            xdmf.write_meshtags(self._facet_tags, self._mesh.geometry)

    def _load(self) -> None:
        logger.info("Loading mesh from %s ...", self._mesh_file)
        with dolfinx.io.XDMFFile(self._comm, str(self._mesh_file), "r") as xdmf:
            self._mesh = xdmf.read_mesh(name="mesh")
            self._mesh.topology.create_connectivity(
                self._mesh.topology.dim - 1, self._mesh.topology.dim,
            )
            self._facet_tags = xdmf.read_meshtags(self._mesh, name="facet_tags")
        logger.info(
            "Mesh loaded: %d cells, %d vertices",
            self.n_cells, self.n_vertices,
        )