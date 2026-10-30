import argparse
import gc
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from eit3d import report
from eit3d.benchmark import TwoSpheres, boundary_norm, transfer_all
from eit3d.config import OUTPUTS_DIR
from eit3d.currents import lateral_measure
from eit3d.inverse import GradientMethod
from eit3d.visualization.grid import TRUE_FIELD, save_conductivity
from eit3d.visualization.static import StaticRenderer

STEPS = (1.0, 0.1, 0.01)
N_ITER = 100
N_SNAPSHOTS = 10
TOL_TRANSFER = 1e-2


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=N_ITER)
    parser.add_argument("--steps", type=float, nargs="+", default=list(STEPS))
    parser.add_argument(
        "--snapshot-every",
        type=int,
        default=None,
        help=f"interval between plotted iterations (default: iterations / {N_SNAPSHOTS})",
    )
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be at least 1")
    if any(step <= 0 for step in args.steps):
        parser.error("--steps must be positive")
    if args.snapshot_every is None:
        args.snapshot_every = max(1, args.iterations // N_SNAPSHOTS)
    return args


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    args = parse_args()
    bench = TwoSpheres()

    report.title("PROBLEMA INVERSO  (método do gradiente)")
    report.info(
        "Solução verdadeira γ⁺",
        f"{bench.background:g} no cilindro, {bench.value:g} em {len(bench.centers)} esferas",
    )
    report.info(
        "Esferas",
        ", ".join(f"({x:g}, {y:g}, {z:g})" for x, y, z in bench.centers)
        + f"  raio {bench.radius:g}",
    )
    report.info(
        "Correntes",
        f"cos(kθ) na lateral, 0 nas bases, k = {bench.ks[0]}, ..., {bench.ks[-1]}",
    )

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
    ok_transfer = report.check(
        "Transferência dos dados entre malhas",
        diff_norm,
        diff_norm < TOL_TRANSFER,
        ideal="~0",
    )

    gamma_true = bench.conductivity(mesh)
    method = GradientMethod(
        mesh,
        V,
        bench.currents(mesh),
        lateral_measure(mesh, facet_tags),
        data,
        coarse.config.solver,
        gamma_true,
    )
    floor = method.residual(method.forward(gamma_true))
    report.info(
        "Resíduo de γ⁺ na malha grossa", f"{floor:.2e}  (referência para o resíduo)"
    )

    gamma0 = bench.initial_guess(mesh)
    results = {}
    for step in args.steps:
        report.section(
            f"λ = {step:g}  ({args.iterations} iterações, γ₀ ≡ {bench.background:g})"
        )
        print(f"  {'k':>5}  {'resíduo':>10}  {'erro':>10}")

        def show(record, n=args.iterations):
            if record.k % args.snapshot_every == 0 or record.k == n:
                print(
                    f"  {record.k:>5}  {record.residual:>10.3e}  {record.error:>10.3e}"
                )

        result = method.run(
            gamma0,
            step,
            args.iterations,
            snapshot_every=args.snapshot_every,
            callback=show,
        )
        results[step] = result

        if result.failure is not None:
            report.info(
                "Interrompido",
                f"γ ficou ≤ 0 na iteração {result.failure.iteration} (passo grande demais)",
            )
            continue
        ratio = result.residuals[-1] / result.residuals[0]
        report.check(
            "Resíduo final / inicial", ratio, ratio < 1, ideal="< 1", fmt=".3f"
        )
        report.info(
            "Erro final", f"{result.errors[-1]:.3f}  (inicial {result.errors[0]:.3f})"
        )
        report.info(
            "γ final",
            f"de {result.gamma.x.array.min():.2f} a {result.gamma.x.array.max():.2f}"
            f"  (verdadeira: {bench.background:g} a {bench.value:g})",
        )

    renderer = StaticRenderer(coarse.config, OUTPUTS_DIR)
    outputs = [
        renderer.render_inclusions(
            bench.centers,
            bench.radius,
            bench.value,
            bench.background,
            "step6_gamma_true.png",
        )
    ]
    for step, result in results.items():
        outputs.append(
            renderer.render_inversion_history(
                result.residuals,
                result.errors,
                step,
                f"step6_convergence_lambda_{step:g}_k{args.iterations}.png",
                floor=floor,
            )
        )
        snapshots = sorted(result.snapshots.items())
        if not snapshots:
            continue
        low = min(float(g.x.array.min()) for _, g in snapshots)
        high = max(float(g.x.array.max()) for _, g in snapshots)
        fields = [("γ⁺ (verdadeira)", gamma_true)] + [
            (f"γ_{k}", g) for k, g in snapshots
        ]
        clims = [(bench.background, bench.value)] + [(low, high)] * len(snapshots)
        outputs.append(
            renderer.render_conductivity_sections(
                fields,
                f"step6_gamma_lambda_{step:g}_k{args.iterations}.png",
                spheres=bench.spheres,
                axis="y",
                plane=0.0,
                clims=clims,
            )
        )
        saved = {TRUE_FIELD: gamma_true} | {f"k{k}": g for k, g in snapshots}
        outputs.append(
            save_conductivity(
                saved,
                OUTPUTS_DIR / f"step6_gamma_lambda_{step:g}_k{args.iterations}.vtu",
            )
        )

    stable_steps = {
        step: r.residuals for step, r in results.items() if r.failure is None
    }
    if len(stable_steps) > 1:
        outputs.append(
            renderer.render_step_comparison(
                stable_steps,
                f"step6_comparison_k{args.iterations}.png",
                floor=floor,
            )
        )

    stable = [r for r in results.values() if r.failure is None]
    if not stable:
        report.result(
            False, "", "o método divergiu em todos os λ testados (passo grande demais)"
        )
    else:
        ok = ok_transfer and all(r.residuals[-1] < r.residuals[0] for r in stable)
        report.result(
            ok,
            "o método do gradiente reduziu o resíduo em todos os λ estáveis",
            "há verificações incorretas",
        )
    report.files(outputs)


if __name__ == "__main__":
    main()
