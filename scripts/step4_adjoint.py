"""
adjoint F'(gamma)* h and its consistency test.

Follows the advisor's steps:
    1. Solve the forward problem with gamma and g:   find u,   int_dOmega u   = 0
    2. Solve the forward problem with gamma and h
        (h in place of the current, h = 2 on the caps, -1 on the lateral
        surface, check int_dOmega h = 0):             find psi, int_dOmega psi = 0
    3. F'(gamma)* h = -grad(u).grad(psi)

Consistency test (sigma = same direction eta used in step3):
    1. F'(gamma) sigma = omega|_dOmega
    2. check <F'(gamma)* h, sigma> = <h, F'(gamma) sigma>:
            a = |int_Omega (-grad u.grad psi) sigma - int_dOmega h omega|
                / |int_dOmega h omega|   ~ 0
"""

import logging
import sys
from pathlib import Path

import numpy as np
import ufl
from mpi4py import MPI

sys.path.insert(0, str(Path(__file__).parent.parent))

import dolfinx  # noqa: E402

from eit3d import EITConfig, EITPipeline  # noqa: E402
from eit3d.config import BOTTOM_TAG, LATERAL_TAG, TOP_TAG  # noqa: E402
from eit3d.solvers import AdjointSolver  # noqa: E402

H_CAPS    = 2.0    # value of h on the top and bottom caps
H_LATERAL = -1.0   # value of h on the lateral surface


def cap_lateral_pattern(mesh, height: float, caps: float, lateral: float):
    """h = caps on the caps (|z| = height/2), lateral elsewhere on the boundary."""
    z = ufl.SpatialCoordinate(mesh)[2]
    return ufl.conditional(ufl.gt(abs(z), height / 2.0 - 1e-8), caps, lateral)


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    comm = MPI.COMM_WORLD
    cfg  = EITConfig()
    pipe = EITPipeline(cfg)
    print(pipe.status())

    mesh, facet_tags = pipe.get_mesh()
    V      = pipe.get_function_space()
    ds     = ufl.Measure("ds", domain=mesh)
    ds_tag = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)
    one    = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(1.0))

    def integrate(expr) -> float:
        return comm.allreduce(
            dolfinx.fem.assemble_scalar(dolfinx.fem.form(expr)), op=MPI.SUM,
        )

    gamma = pipe.build_gamma()
    sigma = pipe.build_eta()          # sigma = same direction eta as in step3

    # 1. forward problem with gamma and g
    print("\n1. forward problem with gamma and g")
    u = pipe.solve_forward(pattern=0, gamma=gamma)
    print(f"   int u ds   = {integrate(u * ds):.2e}")

    # 2. forward problem with gamma and h
    print(f"\n2. forward problem with gamma and h  (h = {H_CAPS:g} caps / {H_LATERAL:g} lateral)")
    area_caps = integrate(one * ds_tag(TOP_TAG)) + integrate(one * ds_tag(BOTTOM_TAG))
    area_lat  = integrate(one * ds_tag(LATERAL_TAG))
    r, height = cfg.mesh.radius, cfg.mesh.height
    exact_h   = H_CAPS * 2 * np.pi * r**2 + H_LATERAL * 2 * np.pi * r * height
    h         = cap_lateral_pattern(mesh, height, H_CAPS, H_LATERAL)
    print(f"   int h ds   = {exact_h:.2e}  (exact geometry)")
    print(f"   int h ds   = {integrate(h * ds):.2e}  (mesh: caps {area_caps:.4f}, "
            f"lateral {area_lat:.4f}; removed by the solver)")

    adjoint = AdjointSolver(
        mesh=mesh, V=V, gamma=gamma, u_gamma=u, h=h,
        config=cfg.solver, comm=comm,
    )
    adj = adjoint.solve()
    print(f"   int psi ds = {integrate(adjoint.psi * ds):.2e}")

    # 3. F'(gamma)* h
    print("\n3. F'(gamma)* h = -grad(u).grad(psi)")
    print(f"   range      = [{adj.x.array.min():.4f}, {adj.x.array.max():.4f}]")

    # Consistency test
    print("\nconsistency test")
    omega = pipe.solve_derivative(u_gamma=u, gamma=gamma, eta=sigma)
    lhs   = integrate(adj * sigma * ufl.dx)
    rhs   = integrate(h * omega * ds)
    a     = abs(lhs - rhs) / abs(rhs)
    scale = np.sqrt(integrate(adj**2 * ufl.dx)) * np.sqrt(integrate(sigma**2 * ufl.dx))

    print(f"   <F'(gamma)* h, sigma> = {lhs: .10e}")
    print(f"   <h, F'(gamma) sigma>  = {rhs: .10e}")
    print(f"   a                     = {a:.2e}")
    print(f"   a (scaled by ||F'* h|| ||sigma|| = {scale:.3e}) = {abs(lhs - rhs) / scale:.2e}")

    # h above is even in z and omega is odd in z, so <h, omega> is zero for the
    # exact geometry and the denominator of a is only mesh asymmetry. Repeat with
    # an h without that symmetry to check a with a well-conditioned denominator.
    x      = ufl.SpatialCoordinate(mesh)
    h_asym = x[2] ** 3 + x[0] * x[2]
    adj_a  = AdjointSolver(mesh, V, gamma, u, h_asym, cfg.solver, comm).solve()
    lhs_a  = integrate(adj_a * sigma * ufl.dx)
    rhs_a  = integrate(h_asym * omega * ds)
    print("\ncomplementary check  (h = z^3 + x z, no parity cancellation)")
    print(f"   <F'(gamma)* h, sigma> = {lhs_a: .10e}")
    print(f"   <h, F'(gamma) sigma>  = {rhs_a: .10e}")
    print(f"   a                     = {abs(lhs_a - rhs_a) / abs(rhs_a):.2e}")


if __name__ == "__main__":
    main()