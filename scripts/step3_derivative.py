import logging
import sys
from pathlib import Path

import dolfinx.plot
import numpy as np
import pyvista
import ufl
from mpi4py import MPI

sys.path.insert(0, str(Path(__file__).parent.parent))

from eit3d import EITConfig, EITPipeline, report
from eit3d.config import OUTPUTS_DIR, ConsistencyTestConfig
from eit3d.solvers.forward import ForwardSolver
from eit3d.visualization.static import StaticRenderer

PROGRESS_EVERY = 15
SLOPE_RANGE = (0.95, 1.05)
FIT_T_MAX = 1e-2
FIT_NOISE_GAP = 30.0


def fit_window(t_vals: np.ndarray, idx_min: int) -> np.ndarray:
    mask = (t_vals <= FIT_T_MAX) & (t_vals >= FIT_NOISE_GAP * t_vals[idx_min])
    if mask.sum() < 3:
        mask = np.arange(len(t_vals)) < max(idx_min, 3)
    return np.where(mask)[0]


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    comm = MPI.COMM_WORLD
    cfg = EITConfig(consistency=ConsistencyTestConfig(n_iter=150, base=0.9))
    pipe = EITPipeline(cfg)
    base = cfg.consistency.base

    report.title("TESTE DA DERIVADA  F'(γ)η")
    report.mesh(pipe)

    gamma = pipe.build_gamma()
    eta = pipe.build_eta()

    u_gamma = pipe.solve_forward(pattern=0, gamma=gamma)
    omega = pipe.solve_derivative(u_gamma=u_gamma, gamma=gamma, eta=eta)

    mesh, facet_tags = pipe.get_mesh()
    ds = ufl.Measure("ds", domain=mesh)

    def integrate(expr) -> float:
        return comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(expr)), op=MPI.SUM)

    integral_omega = integrate(omega * ds)
    norm_omega_bnd = np.sqrt(integrate(omega**2 * ds))

    report.section("Cálculo da derivada")
    ok_u = report.check_zero("∫u na fronteira", integrate(u_gamma * ds))
    ok_omega = report.check_zero("∫ω na fronteira  (ω = F'(γ)η)", integral_omega)

    n_iter = cfg.consistency.n_iter
    report.section(f"Teste de consistência: {n_iter} perturbações γ + tη, com t = {base:g}ⁿ")
    print("  y = erro entre a derivada numérica e ω (deve diminuir junto com t)")
    print(f"  {'n':>5}  {'t':>9}  {'y':>9}")

    V = pipe.get_function_space()
    V0 = dolfinx.fem.functionspace(mesh, ("DG", 0))
    gamma_n = dolfinx.fem.Function(V0)
    diff_fn = dolfinx.fem.Function(V)
    t_vals = cfg.consistency.t_values()
    y_vals = np.empty(len(t_vals))
    g_top, g_bot = cfg.current.patterns[0]
    err_form = dolfinx.fem.form((diff_fn - omega)**2 * ds)

    for k, t_n in enumerate(t_vals):
        gamma_n.x.array[:] = gamma.x.array + t_n * eta.x.array
        gamma_n.x.scatter_forward()

        u_n = ForwardSolver(
            mesh=mesh, facet_tags=facet_tags, V=V,
            gamma=gamma_n, config=cfg.solver, comm=comm,
            g_top=g_top, g_bot=g_bot,
        ).solve()

        diff_fn.x.array[:] = (u_n.x.array - u_gamma.x.array) / t_n
        diff_fn.x.scatter_forward()

        y_vals[k] = np.sqrt(comm.allreduce(dolfinx.fem.assemble_scalar(err_form), op=MPI.SUM)) / norm_omega_bnd
        if k % PROGRESS_EVERY == 0 or k == n_iter - 1:
            print(f"  {k:>5}  {t_n:>9.1e}  {y_vals[k]:>9.1e}")

    idx_min = int(np.argmin(y_vals))
    log_t = np.log10(t_vals)
    log_y = np.log10(y_vals)

    window = fit_window(t_vals, idx_min)
    coeffs = np.polyfit(log_t[window], log_y[window], 1)
    slope = coeffs[0]
    fit_line = np.polyval(coeffs, log_t[:window[-1] + 1])

    report.section("Resultado do teste")
    report.info("Menor erro", f"y = {y_vals[idx_min]:.1e}  (em n = {idx_min}, t = {t_vals[idx_min]:.1e})")
    report.info("Queda do erro", f"de {y_vals[0]:.1e} para {y_vals[idx_min]:.1e}"
                f"  ({np.log10(y_vals[0] / y_vals[idx_min]):.0f} ordens de grandeza)")
    report.info("Ajuste da inclinação", f"n = {window[0]} a {window[-1]}  (t entre {t_vals[window[-1]]:.0e} e {t_vals[window[0]]:.0e})")
    ok_slope = report.check(
        "Inclinação (log y  ×  log t)", slope, SLOPE_RANGE[0] < slope < SLOPE_RANGE[1],
        ideal="1", fmt=".3f",
    )

    renderer = StaticRenderer(cfg, OUTPUTS_DIR)
    topo, ct, geo = dolfinx.plot.vtk_mesh(V)

    grid_ug = pyvista.UnstructuredGrid(topo, ct, geo)
    grid_ug["u_gamma"] = u_gamma.x.array.real
    grid_om = pyvista.UnstructuredGrid(topo, ct, geo)
    grid_om["omega"] = omega.x.array.real

    outputs = [
        renderer.render_geometry(),
        renderer.render_forward(grid_ug, "u_gamma"),
        renderer.render_omega(grid_om, "omega", integral_omega),
        renderer.render_consistency(y_vals, t_vals, slope, fit_line, idx_min),
    ]

    report.result(
        ok_u and ok_omega and ok_slope,
        "derivada correta (o erro cai na mesma proporção que t)",
        "derivada com problema",
    )
    report.files(outputs)


if __name__ == "__main__":
    main()