"""Restartable acquisition and inspection; fail closed before unverified training."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import shutil
import subprocess
import time

import yaml

from . import DOI, DATASET_PAGE
from .acquisition import (AcquisitionError, atomic_json, fetch_metadata, metadata_manifest,
                          acquire, safe_extract, hashes, verify_file, METADATA_URL)
from .adapter import I24MSDAdapter


def preserve(root=Path(".")):
    spec = json.loads((root / "outputs/i24_msd/preservation.json").read_text())
    for name, expected in spec["files"].items():
        data = (root / name).read_bytes()
        if name in spec["append_only"]:
            data = data.split(b"\n<!-- I24_MSD -->")[0] if name.endswith(".md") else data.split(b"\n# I24_MSD")[0]
        if hashlib.sha256(data).hexdigest() != expected:
            raise AcquisitionError(f"Preserved Phase 1/2/3 file changed: {name}")
    return len(spec["files"])


def resources(root):
    import torch
    free = shutil.disk_usage(root)
    mem = Path("/proc/meminfo").read_text().splitlines() if Path("/proc/meminfo").exists() else []
    jobs = subprocess.run(["ps", "-eo", "pid,comm,pcpu,pmem", "--sort=-pcpu"], text=True, capture_output=True, check=True).stdout.splitlines()[:13]
    return {"python": platform.python_version(), "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
            "cpu": next((line.split(":",1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")), platform.machine()), "disk_free_bytes": free.free,
            "memory": [line for line in mem if line.startswith(("MemTotal:", "MemAvailable:"))], "active_jobs": jobs}


def run(config, data_root=None, stage="pilot", metadata_json=None, file_ids=(), import_dir=None,
        max_download_bytes=None, max_disk_bytes=None, extract=False):
    cfg = yaml.safe_load(Path(config).read_text())
    out = Path(cfg["output"])
    if not out.resolve().is_relative_to(Path("outputs/i24_msd").resolve()):
        raise AcquisitionError("Outputs must stay in the separate outputs/i24_msd namespace")
    out.mkdir(parents=True, exist_ok=True)
    preserved = preserve()
    started = time.perf_counter()
    manifest_path = out / "DOWNLOAD_MANIFEST.json"
    metadata_path = Path(metadata_json) if metadata_json else out / "dataverse_metadata.json"
    raw = Path(data_root or "data/i24_msd/raw")
    max_download_bytes = max_download_bytes if max_download_bytes is not None else cfg["max_download_bytes"]
    max_disk_bytes = max_disk_bytes if max_disk_bytes is not None else cfg["max_disk_bytes"]
    audit = resources(out)
    atomic_json(out / "resource_audit.json", audit)
    manifest = {"doi": DOI, "dataset_page": DATASET_PAGE, "metadata_url": METADATA_URL,
                "status": "BLOCKED", "files": [], "total_downloaded_bytes": 0, "real_scenarios_parsed": 0,
                "dataset_id": None, "version": None}
    try:
        if cfg.get("research_gate_required"):
            raise AcquisitionError("Research remains gated until an actual real pilot and resource review pass; never auto-run")
        if metadata_path.exists():
            manifest = metadata_manifest(json.loads(metadata_path.read_text()))
        else:
            manifest = fetch_metadata(metadata_path)
        manifest["metadata_sha256"] = hashes(metadata_path)["SHA-256"]
        if metadata_json and metadata_path.resolve() != (out / "dataverse_metadata.json").resolve():
            (out / "dataverse_metadata.json").write_bytes(metadata_path.read_bytes())
        manifest["metadata_origin"] = "user_supplied_authorized_export" if metadata_json else "public_api"
        manifest["budget"] = {"max_download_bytes": max_download_bytes, "max_disk_bytes": max_disk_bytes}
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text())
            if previous.get("metadata_sha256") not in (None, manifest["metadata_sha256"]):
                raise AcquisitionError("Immutable release metadata changed; use a new output directory")
            for item in manifest["files"]:
                old = next((f for f in previous["files"] if f["id"] == item["id"]), None)
                if old and old.get("downloaded_bytes"):
                    item.update(old)
            manifest["total_downloaded_bytes"] = sum(f["downloaded_bytes"] for f in manifest["files"])
            if manifest["total_downloaded_bytes"]:
                manifest["status"] = "BYTES_VERIFIED_NOT_PARSED"
        atomic_json(manifest_path, manifest)
        if stage == "discover":
            return manifest
        # A smoke/pilot namespace may share the exact already verified raw ZIP.
        # Rehash it locally; never redownload because an output directory changed.
        for item in manifest["files"]:
            path = raw / f'{item["id"]}-{item["name"]}'
            if not item.get("downloaded_bytes") and path.is_file():
                measured = verify_file(path, item)
                item.update(local_path=str(path), hashes=measured, downloaded_bytes=path.stat().st_size,
                            accessibility="BYTES_VERIFIED_NOT_PARSED", acquisition="verified_local_cache")
        manifest["total_downloaded_bytes"] = sum(f["downloaded_bytes"] for f in manifest["files"])
        if manifest["total_downloaded_bytes"]:
            manifest["status"] = "BYTES_VERIFIED_NOT_PARSED"
            atomic_json(manifest_path, manifest)
        if not file_ids and not manifest["total_downloaded_bytes"]:
            file_ids = cfg.get("source_file_ids", [])
        if file_ids:
            acquire(manifest, raw, list(file_ids), max_download_bytes, max_disk_bytes, import_dir,
                    on_progress=lambda value: atomic_json(manifest_path, value))
        elif not manifest["total_downloaded_bytes"]:
            raise AcquisitionError("Metadata available: inspect files/descriptions, then select actual --file-id values within budget")
        adapter = I24MSDAdapter(manifest)
        inspection = adapter.inspect()
        atomic_json(out / "inspection.json", inspection)
        if extract:
            for report in inspection:
                if report["container"] in ("ZIP", "TAR"):
                    archive = Path(report["path"])
                    dest = archive.with_name(archive.name + ".extracted")
                    if dest.exists():
                        saved = json.loads((dest / "EXTRACTION_MANIFEST.json").read_text())
                        if saved["archive_sha256"] != hashes(archive)["SHA-256"]:
                            raise AcquisitionError("Extraction source changed")
                        for entry in saved["files"]:
                            if hashes(dest / entry["path"]) != entry["hashes"]:
                                raise AcquisitionError("Extracted content changed")
                    else:
                        safe_extract(archive, dest, max_disk_bytes)
        if stage in ("acquire", "inspect"):
            return manifest
        from .campaign import run_pilot
        checkpoints_before = len(list((out / "checkpoints").glob("*.pt")))
        result = run_pilot(cfg, manifest, raw, out)
        history_path = out / "execution_history.json"
        history = json.loads(history_path.read_text()) if history_path.exists() else []
        history.append({"completed": True, "elapsed_seconds": time.perf_counter()-started,
                        "checkpoints_at_start": checkpoints_before, "source_kind": "REAL_I24_MSD",
                        "config": str(config), "real_bytes_reused": result["total_downloaded_bytes"]})
        atomic_json(history_path, history)
        return result
    except (AcquisitionError, OSError, ValueError, KeyError) as error:
        # Never relabel verified bytes as fully parsed real scenarios.
        manifest["last_error"] = str(error)
        atomic_json(manifest_path, manifest)
        atomic_json(out / "BLOCKED.json", {"status": "BLOCKED", "stage": stage, "reason": str(error),
                    "real_scenarios": 0, "gradient_steps": 0, "checkpoint": None, "demo": None,
                    "elapsed_seconds": time.perf_counter() - started, "preserved_files": preserve(),
                    "resume": f"uv run python scripts/run_i24_msd_pipeline.py --config {config} --data-root {raw}"})
        raise AcquisitionError(str(error)) from error
    finally:
        preserve()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/i24_msd_pilot.yaml")
    parser.add_argument("--data-root")
    parser.add_argument("--stage", choices=("discover", "acquire", "inspect", "pilot"), default="pilot")
    parser.add_argument("--metadata-json", help="Authorized Dataverse Native API JSON export for this exact DOI")
    parser.add_argument("--file-id", type=int, action="append", default=[], dest="file_ids")
    parser.add_argument("--import-dir", help="Directory with authorized original filenames, verified against metadata checksums")
    parser.add_argument("--max-download-bytes", type=int)
    parser.add_argument("--max-disk-bytes", type=int)
    parser.add_argument("--extract", action="store_true")
    args = parser.parse_args()
    try:
        result = run(**vars(args))
        print(json.dumps({"status": result["status"], "bytes": result["total_downloaded_bytes"], "downloaded_files": sum(bool(f.get("downloaded_bytes")) for f in result["files"]), "metadata_files": len(result["files"])}))
    except AcquisitionError as error:
        parser.exit(2, f"BLOCKED: {error}\nNo real model or real-data demo was produced. See DOWNLOAD_MANIFEST.json and BLOCKED.json.\n")
