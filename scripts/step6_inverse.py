import argparse
import gc
import logging
import sys
from pathlib import Path

import numpy as np
import ufl
from mpi4py import MPI

sys.path.insert(0, str(Path(__file__).parent.parent))

import dolfinx

from eit3d import EITConfig, EITPipeline, report
from eit3d.config import OUTPUTS_DIR, MeshConfig
from eit3d.currents import cosine_currents, lateral_measure
from eit3d.fields import SpheresField
from eit3d.inverse import GradientMethod, data_from, transfer
from eit3d.visualization.static import StaticRenderer

FINE_MESH = MeshConfig(size_max=0.05, size_min=0.02)
COARSE_MESH = MeshConfig(size_max=0.1, size_min=0.05)
KS = range(1, 9)
TRUE_CENTERS = ((0.4, 0.0, 0.4), (-0.4, 0.0, -0.4))
TRUE_RADIUS = 0.25
TRUE_VALUE = 10.0
BACKGROUND = 1.0
STEPS = (1.0, 0.1, 0.01)
N_ITER = 100
N_SNAPSHOTS = 10
TOL_TRANSFER = 1e-2


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=N_ITER)
    parser.add_argument("--steps", type=float, nargs="+", default=list(STEPS))
    parser.add_argument("--snapshot-every", type=int, default=None,
                        help=f"interval between plotted iterations (default: iterations / {N_SNAPSHOTS})")
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be at least 1")
    if any(step <= 0 for step in args.steps):
        parser.error("--steps must be positive")
    if args.snapshot_every is None:
        args.snapshot_every = max(1, args.iterations // N_SNAPSHOTS)
    return args


def true_conductivity(mesh):
    return SpheresField(mesh, TRUE_CENTERS, TRUE_RADIUS, TRUE_VALUE, BACKGROUND, name="gamma+").build()


def boundary_norm(functions, ds, comm) -> float:
    return float(np.sqrt(sum(
        comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(f ** 2 * ds)), op=MPI.SUM)
        for f in functions
    )))


def generate_data(cfg, comm):
    pipe = EITPipeline(cfg)
    report.mesh(pipe)
    mesh, facet_tags = pipe.get_mesh()
    V = pipe.get_function_space()
    currents = cosine_currents(mesh, KS)
    data = data_from(mesh, V, true_conductivity(mesh), currents, lateral_measure(mesh, facet_tags), cfg.solver, comm)
    norm = boundary_norm(data, ufl.Measure("ds", domain=mesh), comm)
    return data, norm


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    args = parse_args()
    comm = MPI.COMM_WORLD

    report.title("PROBLEMA INVERSO  (método do gradiente)")
    report.info("Solução verdadeira γ⁺", f"{BACKGROUND:g} no cilindro, {TRUE_VALUE:g} em {len(TRUE_CENTERS)} esferas")
    report.info("Esferas", ", ".join(f"({x:g}, {y:g}, {z:g})" for x, y, z in TRUE_CENTERS) + f"  raio {TRUE_RADIUS:g}")
    report.info("Correntes", f"cos(kθ) na lateral, 0 nas bases, k = {KS[0]}, ..., {KS[-1]}")

    report.section("Dados ũ = F_G(γ⁺)  (malha fina)")
    data_fine, norm_fine = generate_data(EITConfig(mesh=FINE_MESH), comm)

    cfg = EITConfig(mesh=COARSE_MESH)
    pipe = EITPipeline(cfg)
    report.section("Problema inverso  (malha grossa)")
    report.mesh(pipe)
    mesh, facet_tags = pipe.get_mesh()
    V = pipe.get_function_space()
    ds = ufl.Measure("ds", domain=mesh)

    data = [transfer(d, V) for d in data_fine]
    del data_fine
    gc.collect()

    norm_coarse = boundary_norm(data, ds, comm)
    diff_norm = abs(norm_coarse - norm_fine) / norm_fine
    ok_transfer = report.check("Transferência dos dados entre malhas", diff_norm, diff_norm < TOL_TRANSFER, ideal="~0")

    currents = cosine_currents(mesh, KS)
    gamma_true = true_conductivity(mesh)
    method = GradientMethod(mesh, V, currents, lateral_measure(mesh, facet_tags), data, cfg.solver, gamma_true, comm)

    floor = method.residual(method.forward(gamma_true))
    report.info("Resíduo de γ⁺ na malha grossa", f"{floor:.2e}  (referência para o resíduo)")

    gamma0 = SpheresField(mesh, (), TRUE_RADIUS, BACKGROUND, BACKGROUND, name="gamma0").build()
    results = {}
    for step in args.steps:
        report.section(f"λ = {step:g}  ({args.iterations} iterações, γ₀ ≡ {BACKGROUND:g})")
        print(f"  {'k':>5}  {'resíduo':>10}  {'erro':>10}")

        def show(record, n=args.iterations):
            if record.k % args.snapshot_every == 0 or record.k == n:
                print(f"  {record.k:>5}  {record.residual:>10.3e}  {record.error:>10.3e}")

        result = method.run(gamma0, step, args.iterations, snapshot_every=args.snapshot_every, callback=show)
        results[step] = result

        if result.failure is not None:
            report.info("Interrompido", f"γ ficou ≤ 0 na iteração {result.failure.iteration} (passo grande demais)")
            continue
        ratio = result.residuals[-1] / result.residuals[0]
        report.check("Resíduo final / inicial", ratio, ratio < 1, ideal="< 1", fmt=".3f")
        report.info("Erro final", f"{result.errors[-1]:.3f}  (inicial {result.errors[0]:.3f})")
        report.info("γ final", f"de {result.gamma.x.array.min():.2f} a {result.gamma.x.array.max():.2f}"
                    f"  (verdadeira: {BACKGROUND:g} a {TRUE_VALUE:g})")

    renderer = StaticRenderer(cfg, OUTPUTS_DIR)
    spheres = [(c, TRUE_RADIUS) for c in TRUE_CENTERS]
    outputs = []
    for step, result in results.items():
        outputs.append(renderer.render_inversion_history(
            result.residuals, result.errors, step, f"step6_convergence_lambda_{step:g}_k{args.iterations}.png", floor=floor,
        ))
        snapshots = sorted(result.snapshots.items())
        if not snapshots:
            continue
        low = min(float(g.x.array.min()) for _, g in snapshots)
        high = max(float(g.x.array.max()) for _, g in snapshots)
        fields = [("γ⁺ (verdadeira)", gamma_true)] + [(f"γ_{k}", g) for k, g in snapshots]
        clims = [(BACKGROUND, TRUE_VALUE)] + [(low, high)] * len(snapshots)
        outputs.append(renderer.render_conductivity_sections(
            fields, f"step6_gamma_lambda_{step:g}_k{args.iterations}.png", spheres=spheres, axis="y", plane=0.0, clims=clims,
        ))

    stable = [r for r in results.values() if r.failure is None]
    if not stable:
        report.result(False, "", "o método divergiu em todos os λ testados (passo grande demais)")
    else:
        ok = ok_transfer and all(r.residuals[-1] < r.residuals[0] for r in stable)
        report.result(ok, "o método do gradiente reduziu o resíduo em todos os λ estáveis", "há verificações incorretas")
    report.files(outputs)


if __name__ == "__main__":
    main()