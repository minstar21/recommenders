"""Execute selected quickstarts using the currently installed Python environment.

Usage: python tools/validate_quickstarts.py --output /absolute/new/output
"""

import argparse
import asyncio
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import nbformat
from nbclient import NotebookClient
from jupyter_client.kernelspec import KernelSpecManager
from jupyter_client import AsyncKernelManager
from tempfile import TemporaryDirectory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--notebooks",
        nargs="+",
        default=[
            "sar_movielens",
            "als_movielens",
            "lightgbm_tinycriteo",
            "cpu_models_offline",
        ],
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    folder = root / "examples" / "00_quick_start"
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    with TemporaryDirectory() as kernels:
        spec = Path(kernels) / "minstar-validation"
        spec.mkdir()
        (spec / "kernel.json").write_text(
            json.dumps(
                {
                    "argv": [
                        sys.executable,
                        "-m",
                        "ipykernel_launcher",
                        "-f",
                        "{connection_file}",
                    ],
                    "display_name": "Local validation",
                    "language": "python",
                }
            )
        )
        manager = KernelSpecManager(kernel_dirs=[kernels])
        for name in args.notebooks:
            if Path(name).name != name:
                parser.error("Notebook names must be basenames")
            notebook = nbformat.read(folder / (name + ".ipynb"), as_version=4)
            started = time.monotonic()
            result = {
                "notebook": name,
                "python": sys.version,
                "status": "failed",
                "source_sha256_lf": hashlib.sha256(
                    (folder / (name + ".ipynb")).read_bytes().replace(b"\r\n", b"\n")
                ).hexdigest(),
            }
            km = AsyncKernelManager(
                kernel_name="minstar-validation", kernel_spec_manager=manager
            )
            client = None
            try:
                client = NotebookClient(
                    notebook,
                    km=km,
                    timeout=600,
                    resources={"metadata": {"path": str(folder)}},
                )
                client.execute(cleanup_kc=True)
                metrics = {}
                for cell in notebook.cells:
                    for output in cell.get("outputs", []):
                        value = output.get("data", {}).get(
                            "application/notebook_utils.json+json"
                        )
                        if value:
                            number = value["data"]
                            if isinstance(number, (int, float)) and not math.isfinite(
                                number
                            ):
                                raise ValueError("Nonfinite metric: " + value["name"])
                            metrics[value["name"]] = number
                result["metrics"] = metrics
                result["status"] = "passed"
            except Exception as exc:
                result["error"] = str(exc)
            finally:
                if client is not None and client.kc is not None:
                    client.kc.stop_channels()
                if km.has_kernel:
                    asyncio.run(km.shutdown_kernel(now=True))
            result["seconds"] = round(time.monotonic() - started, 2)
            nbformat.write(notebook, args.output / (name + ".ipynb"))
            results.append(result)
            (args.output / "results.json").write_text(json.dumps(results, indent=2))
            print(name, result["status"], result["seconds"], flush=True)
    return int(any(r["status"] != "passed" for r in results))


if __name__ == "__main__":
    raise SystemExit(main())
