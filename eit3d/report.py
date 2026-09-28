from __future__ import annotations

from pathlib import Path
from typing import Iterable

WIDTH    = 66
TOL_ZERO = 1e-10


def _mark(ok: bool) -> str:
    return "✓" if ok else "✗"


def _thousands(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def title(text: str) -> None:
    print("\n" + "=" * WIDTH)
    print(text)
    print("=" * WIDTH)


def section(text: str) -> None:
    print(f"\n{text}")


def info(label: str, value: str) -> None:
    print(f"  {label:<34} {value}")


def check(label: str, value: float, ok: bool, ideal: str = "0", fmt: str = ".1e") -> bool:
    print(f"  {label:<34} {value:>10{fmt}}   ideal: {ideal:<22} {_mark(ok)}")
    return ok


def below(tol: float) -> str:
    return f"0 (aceitável < {tol:.0e})".replace("e-0", "e-")


def check_zero(label: str, value: float, tol: float = TOL_ZERO) -> bool:
    return check(label, value, abs(value) < tol, ideal="0")


def mesh(pipe) -> None:
    from eit3d.mesh.cylinder import mesh_cache_path

    if not mesh_cache_path(pipe.config.mesh).exists():
        print("  Gerando a malha (só na primeira vez, cerca de 1,5 min)...")
    mesh_, _ = pipe.get_mesh()
    cells = mesh_.topology.index_map(3).size_global
    dofs  = pipe.get_function_space().dofmap.index_map.size_global
    info("Malha", f"{_thousands(cells)} elementos, {_thousands(dofs)} incógnitas")


def result(ok: bool, text_ok: str, text_fail: str) -> None:
    print("\n" + "-" * WIDTH)
    print(f"RESULTADO: {text_ok if ok else text_fail}  {_mark(ok)}")
    print("-" * WIDTH)


def files(paths: Iterable[Path]) -> None:
    print("\nArquivos gerados:")
    for p in paths:
        print(f"  {Path(p)}")