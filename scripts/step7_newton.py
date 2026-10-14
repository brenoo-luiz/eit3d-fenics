import argparse
import gc
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from eit3d import report
from eit3d.benchmark import TwoSpheres, boundary_norm, transfer_all
from eit3d.config import OUTPUTS_DIR
from eit3d.currents import lateral_measure
from eit3d.inverse import InexactNewton
from eit3d.visualization.static import StaticRenderer

MU = 0.9
STEPS = (0.25, 0.5, 1.0)
N_OUTER = 100
N_INNER = 100
N_SNAPSHOTS = 10
TOL_TRANSFER = 1e-2


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outer", type=int, default=N_OUTER, help="outer iterations K_max")
    parser.add_argument("--inner", type=int, default=N_INNER, help="maximum inner iterations n_max")
    parser.add_argument("--mu", type=float, default=MU)
    parser.add_argument("--steps", type=float, nargs="+", default=list(STEPS),
                        help="values of lambda for the inner gradient iterations")
    parser.add_argument("--snapshot-every", type=int, default=None,
                        help=f"interval between plotted iterations (default: outer / {N_SNAPSHOTS})")
    args = parser.parse_args()
    if args.outer < 1 or args.inner < 1:
        parser.error("--outer and --inner must be at least 1")
    if not 0 < args.mu < 1:
        parser.error("--mu must be in (0, 1)")
    if any(step <= 0 for step in args.steps):
        parser.error("--steps must be positive")
    if args.snapshot_every is None:
        args.snapshot_every = max(1, args.outer // N_SNAPSHOTS)
    return args


def first_below(values: np.ndarray, level: float) -> str:
    below = np.flatnonzero(values < level)
    return str(int(below[0])) if below.size else "-"


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    args = parse_args()
    bench = TwoSpheres()

    report.title("PROBLEMA INVERSO  (Newton inexato)")
    report.info("Solução verdadeira γ⁺", f"{bench.background:g} no cilindro, {bench.value:g} em {len(bench.centers)} esferas")
    report.info("Esferas", ", ".join(f"({x:g}, {y:g}, {z:g})" for x, y, z in bench.centers) + f"  raio {bench.radius:g}")
    report.info("Correntes", f"cos(kθ) na lateral, 0 nas bases, k = {bench.ks[0]}, ..., {bench.ks[-1]}")
    report.info("Parâmetros", f"μ = {args.mu:g}, n_max = {args.inner}, K_max = {args.outer}, γ₀ ≡ {bench.background:g}")

    report.section("Dados ũ = F_G(γ⁺)  (malha fina)")
    fine = bench.fine_pipeline()
    report.mesh(fine)
    data_fine = bench.data(fine)
    norm_fine = boundary_norm(data_fine)

    report.section("Problema inverso  (malha grossa)")
    coarse = bench.coarse_pipeline()
    report.mesh(coarse)
    mesh, facet_tags = coarse.get_mesh()
    V = coarse.get_function_space()

    data = transfer_all(data_fine, V)
    del data_fine, fine
    gc.collect()

    diff_norm = abs(boundary_norm(data) - norm_fine) / norm_fine
    ok_transfer = report.check("Transferência dos dados entre malhas", diff_norm, diff_norm < TOL_TRANSFER, ideal="~0")

    gamma_true = bench.conductivity(mesh)
    gamma0 = bench.initial_guess(mesh)
    method = InexactNewton(
        mesh, V, bench.currents(mesh), lateral_measure(mesh, facet_tags), data, coarse.config.solver, gamma_true,
    )
    floor = method.residual(method.forward(gamma_true))
    report.info("Resíduo de γ⁺ na malha grossa", f"{floor:.2e}  (referência para o resíduo)")

    norm_sq = method.linearization_norm_squared(gamma0)
    report.info("‖F'(γ₀)‖²", f"{norm_sq:.3f}")
    report.info("λ estável (teoria)", f"0 < λ < 2/‖F'(γ₀)‖² = {2 / norm_sq:.3f}")

    results = {}
    for step in args.steps:
        report.section(f"λ = {step:g}  (K_max = {args.outer}, n_max = {args.inner})")
        print(f"  {'k':>5}  {'resíduo':>10}  {'erro':>10}  {'internas':>8}")

        def show(record):
            inner = "" if record.inner is None else f"{record.inner:>8d}"
            print(f"  {record.k:>5}  {record.residual:>10.3e}  {record.error:>10.3e}  {inner}", flush=True)

        result = method.run(
            gamma0, step, args.outer, mu=args.mu, n_inner=args.inner,
            snapshot_every=args.snapshot_every, callback=show,
        )
        results[step] = result
        if result.failure is not None:
            report.info("Interrompido", f"γ ficou ≤ 0 na iteração {result.failure.iteration} (λ grande demais)")
            continue
        ratio = result.residuals[-1] / result.residuals[0]
        report.check("Resíduo final / inicial", ratio, ratio < 1, ideal="< 1", fmt=".3f")
        report.info("Erro final", f"{result.errors[-1]:.3f}  (inicial {result.errors[0]:.3f})")
        report.info("γ final", f"de {result.gamma.x.array.min():.2f} a {result.gamma.x.array.max():.2f}"
                    f"  (verdadeira: {bench.background:g} a {bench.value:g})")

    report.section("Comparação entre os valores de λ")
    print(f"  {'λ':>6}  {'situação':>10}  {'k < γ⁺':>7}  {'internas':>9}  {'resíduo':>10}  {'erro':>7}")
    for step, result in results.items():
        status = "divergiu" if result.failure is not None else "estável"
        print(f"  {step:>6g}  {status:>10}  {first_below(result.residuals, floor):>7}  "
                f"{int(result.inner_iterations.sum()):>9d}  {result.residuals[-1]:>10.3e}  {result.errors[-1]:>7.3f}")
    print("  k < γ⁺: primeira iteração com resíduo abaixo do resíduo de γ⁺;  internas: total de iterações internas")

    renderer = StaticRenderer(coarse.config, OUTPUTS_DIR)
    outputs = []
    for step, result in results.items():
        tag = f"mu_{args.mu:g}_lambda_{step:g}_k{args.outer}"
        outputs.append(renderer.render_inversion_history(
            result.residuals, result.errors, step, f"step7_convergence_{tag}.png", floor=floor,
            title=f"Newton inexato  ·  μ = {args.mu:g}  ·  λ = {step:g}",
        ))
        snapshots = sorted(result.snapshots.items())
        if not snapshots:
            continue
        low = min(float(g.x.array.min()) for _, g in snapshots)
        high = max(float(g.x.array.max()) for _, g in snapshots)
        fields = [("γ⁺ (verdadeira)", gamma_true)] + [(f"γ_{k}", g) for k, g in snapshots]
        clims = [(bench.background, bench.value)] + [(low, high)] * len(snapshots)
        outputs.append(renderer.render_conductivity_sections(
            fields, f"step7_gamma_{tag}.png", spheres=bench.spheres, axis="y", plane=0.0, clims=clims,
        ))

    stable = {step: r.residuals for step, r in results.items() if r.failure is None}
    if len(stable) > 1:
        outputs.append(renderer.render_step_comparison(
            stable, f"step7_comparison_mu_{args.mu:g}_k{args.outer}.png", floor=floor,
        ))

    if not stable:
        report.result(False, "", "o método divergiu em todos os λ testados")
    else:
        ok = ok_transfer and all(r[-1] < r[0] for r in stable.values())
        report.result(ok, "o método de Newton inexato reduziu o resíduo em todos os λ estáveis",
                    "há verificações incorretas")
    report.files(outputs)


if __name__ == "__main__":
    main()