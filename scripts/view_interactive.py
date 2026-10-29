import argparse
import logging
import sys
from pathlib import Path

import pyvista

sys.path.insert(0, str(Path(__file__).parent.parent))

from eit3d.benchmark import TwoSpheres
from eit3d.config import OUTPUTS_DIR, EITConfig
from eit3d.visualization import InversionViewer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive 3D view of a reconstruction saved by step6 or step7"
    )
    parser.add_argument(
        "file",
        type=Path,
        nargs="?",
        help="saved reconstruction (default: newest outputs/*.vtu)",
    )
    args = parser.parse_args()
    if args.file is None:
        saved = sorted(OUTPUTS_DIR.glob("*.vtu"), key=lambda p: p.stat().st_mtime)
        if not saved:
            parser.error("no .vtu in outputs/; run step6 or step7 first")
        args.file = saved[-1]
    if not args.file.exists():
        parser.error(f"{args.file} not found")
    return args


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    bench = TwoSpheres()
    print(f"Abrindo {args.file}")
    grid = pyvista.read(args.file)
    InversionViewer(EITConfig(mesh=bench.coarse_mesh), grid, bench.spheres).show()


if __name__ == "__main__":
    main()
