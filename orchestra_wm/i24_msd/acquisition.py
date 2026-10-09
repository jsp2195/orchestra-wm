"""Bounded, public Dataverse downloads. No guessed file IDs or credential handling.

Completion means checksum-verified file bytes, NOT verified traffic scenarios.
Partial transfers restart; Range is intentionally unused rather than assumed.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile

from . import DOI, DATASET_PAGE

HOST = "https://dataverse.harvard.edu"
METADATA_URL = HOST + "/api/datasets/:persistentId/?" + urllib.parse.urlencode({"persistentId": "doi:" + DOI})
USER_AGENT = "ORCHESTRA-I24-MSD public research acquisition/1.0"
CHUNK = 1024 * 1024


class AcquisitionError(RuntimeError):
    pass


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def hashes(path):
    sha, md5 = hashlib.sha256(), hashlib.md5()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK), b""):
            sha.update(chunk)
            md5.update(chunk)
    return {"SHA-256": sha.hexdigest(), "MD5": md5.hexdigest()}


def public_url(url):
    """Persist destination host/path, never signed redirect query parameters."""
    p = urllib.parse.urlsplit(url)
    return urllib.parse.urlunsplit((p.scheme, p.netloc, p.path, "", ""))


def open_public(url, attempts=3):
    if not url.startswith("https://"):
        raise AcquisitionError("Only HTTPS public downloads are supported")
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept-Encoding": "identity"})
            response = urllib.request.urlopen(request, timeout=45)
            final = urllib.parse.urlsplit(response.geturl())
            if final.scheme != "https" or any(word in final.path.lower() for word in ("login", "signin", "oauth")):
                response.close()
                raise AcquisitionError("Public request redirected to authentication or insecure transport")
            return response
        except urllib.error.HTTPError as error:
            # No retry of access denials; no login/token fallbacks.
            if error.code not in (429, 500, 502, 503, 504) or attempt == attempts - 1:
                raise AcquisitionError(f"HTTP {error.code}: {public_url(url)}") from error
            wait = error.headers.get("Retry-After", "")
            if wait.isdigit() and int(wait) > 30:
                raise AcquisitionError(f"Server requested Retry-After {wait}s; retry later") from error
            time.sleep(int(wait) if wait.isdigit() else 2 ** attempt)
        except urllib.error.URLError as error:
            if "403" in str(error) or attempt == attempts - 1:
                raise AcquisitionError(str(error)) from error
            time.sleep(2 ** attempt)


def reject_error_body(prefix, content_type, *, allow_html=False):
    start = prefix.lstrip().lower()
    if not allow_html and ("html" in content_type.lower() or start.startswith((b"<!doctype html", b"<html"))):
        raise AcquisitionError("HTML response is not trajectory data (possible login/error page)")
    if start.startswith(b'{"status":"error"') or start.startswith(b'{"status": "error"'):
        raise AcquisitionError("Dataverse returned an error object instead of file bytes")


def fetch_metadata(destination):
    with open_public(METADATA_URL) as response:
        data = response.read(16 * CHUNK + 1)
        if len(data) > 16 * CHUNK:
            raise AcquisitionError("Metadata exceeded 16 MiB budget")
        reject_error_body(data[:4096], response.headers.get("Content-Type", ""))
    try:
        payload = json.loads(data)
        manifest = metadata_manifest(payload)
    except (ValueError, KeyError, TypeError) as error:
        raise AcquisitionError(f"Invalid Dataverse metadata: {error}") from error
    atomic_json(destination, payload)
    return manifest


def metadata_manifest(payload):
    if payload.get("status") != "OK":
        raise AcquisitionError("Dataverse metadata did not report OK")
    data = payload["data"]
    actual = "/".join((str(data.get("authority", "")), str(data.get("identifier", ""))))
    if actual != DOI:
        raise AcquisitionError(f"Wrong dataset DOI: {actual}")
    version = data["latestVersion"]
    if version.get("versionState") != "RELEASED":
        raise AcquisitionError("Only a public RELEASED dataset version may be acquired")
    files = []
    for entry in version.get("files", []):
        f = entry["dataFile"]
        name = f["filename"]
        if Path(name).name != name or name in ("", ".", "..") or "\\" in name:
            raise AcquisitionError("Unsafe source filename")
        size, file_id = int(f["filesize"]), int(f["id"])
        if size <= 0 or file_id <= 0:
            raise AcquisitionError("Invalid metadata size/file ID")
        restricted = bool(entry.get("restricted", f.get("restricted", False)))
        files.append({"id": file_id, "name": name, "description": entry.get("description", f.get("description")),
                      "directory_label": entry.get("directoryLabel"), "declared_bytes": size,
                      "content_type": f.get("contentType"), "storage_format": f.get("originalFileFormat", f.get("contentType")),
                      "checksum": f.get("checksum"), "restricted": restricted,
                      "accessibility": "RESTRICTED" if restricted else "PUBLIC_METADATA_ONLY",
                      "url": f"{HOST}/api/access/datafile/{file_id}", "downloaded_bytes": 0,
                      "split": None, "split_note": "Unverified; not inferred from a filename"})
    if len({f["id"] for f in files}) != len(files):
        raise AcquisitionError("Duplicate Dataverse file IDs")
    return {"doi": DOI, "dataset_page": DATASET_PAGE, "dataset_id": data.get("id"),
            "version": {k: version.get(k) for k in ("id", "versionNumber", "versionMinorNumber", "versionState", "releaseTime")},
            "license": version.get("license"), "terms_of_use": version.get("termsOfUse"),
            "citation_metadata": version.get("metadataBlocks", {}).get("citation"),
            "status": "METADATA_VISIBLE", "files": files, "total_downloaded_bytes": 0,
            "real_scenarios_parsed": 0, "metadata_url": METADATA_URL}


def disk_usage(root):
    return sum(p.stat().st_size for p in Path(root).rglob("*") if p.is_file())


def check_budget(root, additional, max_disk_bytes):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if additional < 0 or max_disk_bytes <= 0 or disk_usage(root) + additional > max_disk_bytes:
        raise AcquisitionError("Disk budget exceeded; select smaller files or explicitly raise --max-disk-bytes")
    if shutil.disk_usage(root).free < additional + 256 * CHUNK:
        raise AcquisitionError("Insufficient free disk; preserving 256 MiB headroom")


def verify_file(path, item):
    path = Path(path)
    if path.stat().st_size != item["declared_bytes"]:
        raise AcquisitionError("File byte count differs from Dataverse metadata")
    with path.open("rb") as stream:
        reject_error_body(stream.read(4096), item.get("content_type") or "")
    measured = hashes(path)
    checksum = item.get("checksum")
    if not checksum:
        raise AcquisitionError("Source checksum unavailable; authenticity requires a verified release checksum")
    algorithm = checksum["type"].upper().replace("SHA256", "SHA-256")
    if algorithm not in measured:
        raise AcquisitionError(f"Unsupported source checksum: {algorithm}")
    if measured[algorithm].lower() != checksum["value"].lower():
        raise AcquisitionError("Source checksum mismatch")
    return measured


def acquire(manifest, raw_root, ids, max_download_bytes, max_disk_bytes, import_dir=None, on_progress=None):
    """Import or download complete selected files; persist progress after each file.

    Existing files are rehashed. A crash never turns a .part into a complete shard.
    Selection must come from verified metadata; no automatic guessed train split.
    """
    raw_root = Path(raw_root)
    if raw_root.is_symlink():
        raise AcquisitionError("Raw root must not be a symlink")
    entries = {f["id"]: f for f in manifest["files"]}
    if not ids or len(set(ids)) != len(ids) or any(i not in entries for i in ids):
        raise AcquisitionError("Select unique actual file IDs listed by metadata")
    selected = [entries[i] for i in ids]
    if any(f["restricted"] for f in selected):
        raise AcquisitionError("Restricted file: use the publisher's authorization flow; public downloader will not access it")
    if max_download_bytes <= 0 or sum(f["declared_bytes"] for f in selected) > max_download_bytes:
        raise AcquisitionError("Selected complete files exceed --max-download-bytes")
    check_budget(raw_root, 0, max_disk_bytes)
    for item in selected:
        dest = raw_root / f'{item["id"]}-{item["name"]}'
        part = dest.with_name(dest.name + ".part")
        if dest.is_symlink() or part.is_symlink():
            raise AcquisitionError("Refusing symlink download destination")
        if not dest.exists():
            # Resume is whole-file restart. Never trust HTTP Range without evidence.
            if part.exists():
                part.unlink()
            check_budget(raw_root, item["declared_bytes"], max_disk_bytes)
            source = None
            try:
                if import_dir:
                    source_path = Path(import_dir) / item["name"]
                    if source_path.is_symlink():
                        raise AcquisitionError("Imported files must be regular files")
                    source = source_path.open("rb")
                    if source_path.stat().st_size != item["declared_bytes"]:
                        raise AcquisitionError("Imported file size mismatch")
                else:
                    # Reconstruct endpoint from metadata ID, never a manifest-supplied host.
                    source = open_public(f'{HOST}/api/access/datafile/{item["id"]}')
                    if source.status != 200:
                        raise AcquisitionError(f"Expected complete HTTP 200 file, received {source.status}")
                    length = source.headers.get("Content-Length")
                    if length is not None and int(length) != item["declared_bytes"]:
                        raise AcquisitionError("Content-Length differs from metadata")
                    if "html" in source.headers.get("Content-Type", "").lower():
                        raise AcquisitionError("HTML authentication/error response, not public file bytes")
                written = 0
                with part.open("xb") as target:
                    while True:
                        chunk = source.read(min(CHUNK, item["declared_bytes"] - written + 1))
                        if not chunk:
                            break
                        if not written:
                            reject_error_body(chunk[:4096], "")
                        written += len(chunk)
                        if written > item["declared_bytes"]:
                            raise AcquisitionError("Transfer exceeded declared size")
                        target.write(chunk)
                    target.flush()
                    os.fsync(target.fileno())
                measured = verify_file(part, item)
                os.replace(part, dest)
            except Exception:
                part.unlink(missing_ok=True)
                raise
            finally:
                if source is not None:
                    source.close()
        else:
            measured = verify_file(dest, item)
        item.update(local_path=str(dest), hashes=measured, downloaded_bytes=dest.stat().st_size,
                    accessibility="BYTES_VERIFIED_NOT_PARSED", acquisition="manual_import" if import_dir else "public_api")
        manifest["total_downloaded_bytes"] = sum(f["downloaded_bytes"] for f in manifest["files"])
        manifest["status"] = "BYTES_VERIFIED_NOT_PARSED"
        if on_progress:
            on_progress(manifest)
    return manifest


def safe_extract(archive, destination, max_disk_bytes, max_members=10000, selected_names=None):
    """Atomic archive extraction; refuse traversal, links, special files and bombs."""
    archive, destination = Path(archive), Path(destination)
    if destination.exists():
        raise AcquisitionError("Extraction destination already exists; reuse inspected manifest or choose a new directory")
    stage = destination.with_name(destination.name + ".partial")
    if stage.exists():
        raise AcquisitionError("Interrupted extraction exists; inspect it before removing")
    if zipfile.is_zipfile(archive):
        reader = zipfile.ZipFile(archive)
        entries = [(m.filename, m.file_size, m.is_dir(), (m.external_attr >> 16) & 0o170000, m) for m in reader.infolist()]
        open_member = reader.open
    elif tarfile.is_tarfile(archive):
        reader = tarfile.open(archive)
        entries = [(m.name, m.size, m.isdir(), 0 if m.isfile() or m.isdir() else -1, m) for m in reader.getmembers()]
        open_member = reader.extractfile
    else:
        raise AcquisitionError("Not a supported ZIP/TAR archive")
    try:
        if len(entries) > max_members:
            raise AcquisitionError("Archive member count budget exceeded")
        names = set()
        for name, size, directory, kind, _ in entries:
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or "\\" in name or kind not in (0, 0o100000, 0o040000):
                raise AcquisitionError("Unsafe archive member (path, link or special file)")
            if str(path) in names or not path.parts or size < 0:
                raise AcquisitionError("Duplicate/invalid archive member")
            names.add(str(path))
        if selected_names is not None:
            if not selected_names or not set(selected_names).issubset(names):
                raise AcquisitionError("Extraction selection contains missing archive members")
            entries = [e for e in entries if e[0] in selected_names]
        total = sum(e[1] for e in entries)
        check_budget(archive.parent, total, max_disk_bytes)
        stage.mkdir(parents=True)
        extracted = []
        for name, size, directory, _, member in entries:
            dest = stage / name
            if directory:
                dest.mkdir(parents=True, exist_ok=True)
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            written = 0
            with open_member(member) as source, dest.open("xb") as target:
                while chunk := source.read(min(CHUNK, size - written + 1)):
                    written += len(chunk)
                    if written > size:
                        raise AcquisitionError("Archive member exceeds declared size")
                    target.write(chunk)
            if written != size:
                raise AcquisitionError("Truncated archive member")
            extracted.append({"path": name, "bytes": size, "hashes": hashes(dest)})
        atomic_json(stage / "EXTRACTION_MANIFEST.json", {"archive_sha256": hashes(archive)["SHA-256"], "files": extracted})
        os.replace(stage, destination)
        return extracted
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    finally:
        reader.close()
