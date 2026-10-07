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

from eit3d import EITConfig, EITPipeline, report  # noqa: E402
from eit3d.solvers import AdjointSolver  # noqa: E402

H_CAPS = 2.0
H_LATERAL = -1.0
TOL_A = 1e-6


def cap_lateral_pattern(mesh, height: float, caps: float, lateral: float):
    z = ufl.SpatialCoordinate(mesh)[2]
    return ufl.conditional(ufl.gt(abs(z), height / 2.0 - 1e-8), caps, lateral)


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    comm = MPI.COMM_WORLD
    cfg = EITConfig()
    pipe = EITPipeline(cfg)

    report.title("TESTE DO ADJUNTO  F'(γ)*h")
    report.mesh(pipe)

    mesh, _ = pipe.get_mesh()
    V = pipe.get_function_space()
    ds = ufl.Measure("ds", domain=mesh)

    def integrate(expr) -> float:
        return comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(expr)), op=MPI.SUM)

    gamma = pipe.build_gamma()
    sigma = pipe.build_eta()
    g_top, g_bot = cfg.current.patterns[0]

    report.section(f"Passo 1: problema direto com γ e g  (g = {g_top:+g} topo, {g_bot:+g} base)")
    u = pipe.solve_forward(pattern=0, gamma=gamma)
    ok1 = report.check_zero("∫u na fronteira", integrate(u * ds))

    report.section(f"Passo 2: problema direto com γ e h  (h = {H_CAPS:g} nas tampas, {H_LATERAL:g} na lateral)")
    r, height = cfg.mesh.radius, cfg.mesh.height
    h = cap_lateral_pattern(mesh, height, H_CAPS, H_LATERAL)
    int_h = H_CAPS * 2 * np.pi * r**2 + H_LATERAL * 2 * np.pi * r * height
    int_h_msh = integrate(h * ds)
    int_abs_h = integrate(abs(h) * ds)
    ok_h = report.check_zero("∫h na fronteira (geometria exata)", int_h)
    report.check("∫h na fronteira (malha)", int_h_msh, abs(int_h_msh) < 1e-2 * int_abs_h, ideal="~0")

    adjoint = AdjointSolver(mesh=mesh, V=V, gamma=gamma, u_gamma=u, h=h, config=cfg.solver, comm=comm)
    adj = adjoint.solve()
    ok2 = report.check_zero("∫ψ na fronteira", integrate(adjoint.psi * ds))

    report.section("Passo 3: F'(γ)*h = -∇u·∇ψ")
    report.info("Calculado", f"valores entre {adj.x.array.min():.2f} e {adj.x.array.max():.2f}")

    report.section("Teste de consistência  (σ = mesma direção η do step3)")
    omega = pipe.solve_derivative(u_gamma=u, gamma=gamma, eta=sigma)
    lhs = integrate(adj * sigma * ufl.dx)
    rhs = integrate(h * omega * ds)
    a = abs(lhs - rhs) / abs(rhs)
    report.info("⟨F'(γ)*h, σ⟩", f"{lhs: .8e}")
    report.info("⟨h, F'(γ)σ⟩", f"{rhs: .8e}")
    ok_a = report.check("a", a, a < TOL_A)

    x = ufl.SpatialCoordinate(mesh)
    h_asym = x[2] ** 3 + x[0] * x[2]
    adj_a = AdjointSolver(mesh, V, gamma, u, h_asym, cfg.solver, comm).solve()
    lhs_a = integrate(adj_a * sigma * ufl.dx)
    rhs_a = integrate(h_asym * omega * ds)
    a_asym = abs(lhs_a - rhs_a) / abs(rhs_a)
    report.section("Verificação extra  (h = z³ + xz, sem a simetria do h acima)")
    ok_x = report.check("a", a_asym, a_asym < TOL_A)

    report.result(ok1 and ok_h and ok2 and ok_a and ok_x, "adjunto correto", "adjunto com problema")


if __name__ == "__main__":
    main()