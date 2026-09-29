"""
Eight electric currents g_k = cos(k theta) on the lateral surface, zero on the bases.
Forward problem, directional derivative, adjoint and consistency tests for the set G.
"""

import logging
import sys
from pathlib import Path

import dolfinx.plot
import numpy as np
import pyvista
import ufl
from mpi4py import MPI

sys.path.insert(0, str(Path(__file__).parent.parent))

import dolfinx  # noqa: E402

from eit3d import EITConfig, EITPipeline, report  # noqa: E402
from eit3d.config import OUTPUTS_DIR, ConsistencyTestConfig, MeshConfig  # noqa: E402
from eit3d.currents import cosine_currents, lateral_measure  # noqa: E402
from eit3d.solvers import MultiAdjointSolver, MultiDerivativeSolver, MultiForwardSolver  # noqa: E402
from eit3d.visualization.static import StaticRenderer, SurfacePanel  # noqa: E402

KS             = range(1, 9)
MESH           = MeshConfig(size_max=0.1, size_min=0.05)
CONSISTENCY    = ConsistencyTestConfig(n_iter=70, base=0.8)
PROGRESS_EVERY = 10
SLOPE_RANGE    = (0.95, 1.05)
FIT_T_MAX      = 1e-2
FIT_NOISE_GAP  = 30.0
TOL_A          = 1e-6
TOL_REL        = 1e-2


def fit_window(t_vals: np.ndarray, idx_min: int) -> np.ndarray:
    mask = (t_vals <= FIT_T_MAX) & (t_vals >= FIT_NOISE_GAP * t_vals[idx_min])
    if mask.sum() < 3:
        mask = np.arange(len(t_vals)) < max(idx_min, 3)
    return np.where(mask)[0]


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    comm = MPI.COMM_WORLD
    cfg  = EITConfig(mesh=MESH, consistency=CONSISTENCY)
    pipe = EITPipeline(cfg)
    n_g  = len(KS)

    report.title(f"TESTE COM {n_g} CORRENTES  g_k = cos(kθ)")
    report.mesh(pipe)
    report.info("Correntes", f"cos(kθ) na lateral, 0 nas bases, k = {KS[0]}, ..., {KS[-1]}")

    mesh, facet_tags = pipe.get_mesh()
    V      = pipe.get_function_space()
    ds     = ufl.Measure("ds", domain=mesh)
    ds_lat = lateral_measure(mesh, facet_tags)

    def integrate(expr) -> float:
        return comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(expr)), op=MPI.SUM)

    currents = cosine_currents(mesh, KS)
    gamma    = pipe.build_gamma()
    sigma    = pipe.build_eta()

    report.section("Conjunto de correntes G = {g_1, ..., g_8}")
    int_g   = max(abs(integrate(g * ds_lat)) for g in currents)
    scale_g = min(integrate(abs(g) * ds_lat) for g in currents)
    gram    = np.array([[integrate(gi * gj * ds_lat) for gj in currents] for gi in currents])
    norms   = np.sqrt(np.diag(gram))
    off     = np.abs(gram / np.outer(norms, norms) - np.eye(n_g)).max()
    rank    = int(np.linalg.matrix_rank(gram))
    ok_g = all([
        report.check("Maior |∫g_k| na fronteira", int_g, int_g < TOL_REL * scale_g, ideal="~0"),
        report.check("Posto da matriz de Gram", rank, rank == n_g, ideal=str(n_g), fmt="d"),
        report.check("Maior produto interno g_i·g_j (i≠j)", off, off < TOL_REL, ideal="~0"),
    ])

    report.section(f"Problema direto: {n_g} soluções u_k")
    u_list = MultiForwardSolver(mesh, V, gamma, currents, ds_lat, cfg.solver, comm).solve_all()
    ok_u = report.check_zero("Maior |∫u_k| na fronteira", max(abs(integrate(u * ds)) for u in u_list))

    report.section(f"Derivada: F'_G(γ)σ = (ω_1, ..., ω_{n_g})")
    w_list = MultiDerivativeSolver(mesh, V, gamma, sigma, u_list, cfg.solver, comm).solve_all()
    ok_w = report.check_zero("Maior |∫ω_k| na fronteira", max(abs(integrate(w * ds)) for w in w_list))

    base   = cfg.consistency.base
    n_iter = cfg.consistency.n_iter
    report.section(f"Teste de consistência da derivada: {n_iter} perturbações γ + tσ, com t = {base:g}ⁿ")
    print("  y = erro das 8 derivadas numéricas em relação a ω (norma do conjunto)")
    print(f"  {'n':>5}  {'t':>9}  {'y':>9}")

    gamma_n   = dolfinx.fem.Function(gamma.function_space)
    diffs     = [dolfinx.fem.Function(V) for _ in KS]
    err_forms = [dolfinx.fem.form((d - w) ** 2 * ds) for d, w in zip(diffs, w_list)]
    norm_w    = np.sqrt(sum(integrate(w ** 2 * ds) for w in w_list))
    t_vals    = cfg.consistency.t_values()
    y_vals    = np.empty(len(t_vals))

    for k, t_n in enumerate(t_vals):
        gamma_n.x.array[:] = gamma.x.array + t_n * sigma.x.array
        gamma_n.x.scatter_forward()
        un_list = MultiForwardSolver(mesh, V, gamma_n, currents, ds_lat, cfg.solver, comm).solve_all()

        err = 0.0
        for d, u0, u1, form in zip(diffs, u_list, un_list, err_forms):
            d.x.array[:] = (u1.x.array - u0.x.array) / t_n
            d.x.scatter_forward()
            err += comm.allreduce(dolfinx.fem.assemble_scalar(form), op=MPI.SUM)
        y_vals[k] = np.sqrt(err) / norm_w
        if k % PROGRESS_EVERY == 0 or k == n_iter - 1:
            print(f"  {k:>5}  {t_n:>9.1e}  {y_vals[k]:>9.1e}")

    idx_min  = int(np.argmin(y_vals))
    log_t    = np.log10(t_vals)
    log_y    = np.log10(y_vals)
    window   = fit_window(t_vals, idx_min)
    coeffs   = np.polyfit(log_t[window], log_y[window], 1)
    slope    = coeffs[0]
    fit_line = np.polyval(coeffs, log_t[:window[-1] + 1])

    report.info("Menor erro", f"y = {y_vals[idx_min]:.1e}  (em n = {idx_min}, t = {t_vals[idx_min]:.1e})")
    report.info("Queda do erro", f"de {y_vals[0]:.1e} para {y_vals[idx_min]:.1e}"
                f"  ({np.log10(y_vals[0] / y_vals[idx_min]):.0f} ordens de grandeza)")
    report.info("Ajuste da inclinação", f"n = {window[0]} a {window[-1]}  (t entre {t_vals[window[-1]]:.0e} e {t_vals[window[0]]:.0e})")
    ok_slope = report.check("Inclinação (log y  ×  log t)", slope,
                            SLOPE_RANGE[0] < slope < SLOPE_RANGE[1], ideal="1", fmt=".3f")

    report.section("Adjunto: F'_G(γ)*h = Σ -∇u_k·∇ψ_k,  com h_k = g_k")
    adjoint = MultiAdjointSolver(mesh, V, gamma, u_list, currents, ds_lat, cfg.solver, comm)
    adj     = adjoint.solve()
    ok_psi  = report.check_zero("Maior |∫ψ_k| na fronteira", max(abs(integrate(p * ds)) for p in adjoint.psis))

    lhs = integrate(adj * sigma * ufl.dx)
    rhs = sum(integrate(h * w * ds_lat) for h, w in zip(currents, w_list))
    a   = abs(lhs - rhs) / abs(rhs)
    report.info("⟨F'_G(γ)*h, σ⟩", f"{lhs: .8e}")
    report.info("⟨h, F'_G(γ)σ⟩", f"{rhs: .8e}")
    ok_a = report.check("a", a, a < TOL_A)

    z      = ufl.SpatialCoordinate(mesh)[2]
    h_ext  = [(1 + z ** 2) * g for g in currents]
    adj_x  = MultiAdjointSolver(mesh, V, gamma, u_list, h_ext, ds_lat, cfg.solver, comm).solve()
    lhs_x  = integrate(adj_x * sigma * ufl.dx)
    rhs_x  = sum(integrate(h * w * ds_lat) for h, w in zip(h_ext, w_list))
    a_x    = abs(lhs_x - rhs_x) / abs(rhs_x)
    report.section("Verificação extra  (h_k = (1 + z²) g_k, com ψ_k diferente de u_k)")
    ok_x = report.check("a", a_x, a_x < TOL_A)

    report.section("Contribuição de cada corrente  ⟨g_k, ω_k⟩")
    for k, g, w in zip(KS, currents, w_list):
        report.info(f"k = {k}", f"{integrate(g * w * ds_lat): .2e}")

    renderer = StaticRenderer(cfg, OUTPUTS_DIR)
    topo, ct, geo = dolfinx.plot.vtk_mesh(V)
    coords  = V.tabulate_dof_coordinates()
    on_base = np.abs(coords[:, 2]) >= cfg.mesh.height / 2 - 1e-8
    theta   = np.arctan2(coords[:, 1], coords[:, 0])

    def current_panel(k: int) -> SurfacePanel:
        values = np.cos(k * theta)
        values[on_base] = 0.0
        grid = pyvista.UnstructuredGrid(topo, ct, geo)
        grid[f"g_{k}"] = values
        return SurfacePanel(grid, f"g_{k}", f"g_{k} = cos({k}θ)", cmap="coolwarm", clim=(-1.0, 1.0))

    outputs = [
        renderer.render_surfaces([current_panel(k) for k in KS[:4]], "step5_currents_1-4.png"),
        renderer.render_surfaces([current_panel(k) for k in KS[4:]], "step5_currents_5-8.png"),
        renderer.render_consistency(y_vals, t_vals, slope, fit_line, idx_min, filename="step5_consistency.png"),
    ]

    report.result(
        all([ok_g, ok_u, ok_w, ok_slope, ok_psi, ok_a, ok_x]),
        f"{n_g} correntes: problema direto, derivada e adjunto corretos",
        "há verificações incorretas",
    )
    report.files(outputs)


if __name__ == "__main__":
    main()