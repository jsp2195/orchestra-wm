"""Inspect native bytes without guessing their semantic trajectory schema.

The official webpage specifies nine-second clips, but not their encoding. Until
release metadata AND real records are available, conversion to RoadsideScene is
deliberately gated. A TFRecord container is not proof of a Waymo Scenario proto.
"""
from __future__ import annotations

import json
from pathlib import Path
import struct
import tarfile
import zipfile

from .acquisition import AcquisitionError, hashes, verify_file


class SchemaUnverified(AcquisitionError):
    pass


def _crc_table():
    table = []
    for n in range(256):
        for _ in range(8):
            n = (n >> 1) ^ (0x82F63B78 if n & 1 else 0)
        table.append(n)
    return tuple(table)


CRC_TABLE = _crc_table()


def crc32c(data):
    crc = 0xFFFFFFFF
    for byte in data:
        crc = CRC_TABLE[(crc ^ byte) & 255] ^ (crc >> 8)
    return crc ^ 0xFFFFFFFF


def masked_crc(data):
    crc = crc32c(data)
    return (((crc >> 15) | (crc << 17)) + 0xA282EAD8) & 0xFFFFFFFF


def tfrecords(path, max_records=8, max_record_bytes=16 * 1024**2):
    """Verify record framing + CRC32C; return opaque payloads, never infer proto."""
    with Path(path).open("rb") as stream:
        for _ in range(max_records):
            header = stream.read(12)
            if not header:
                return
            if len(header) != 12:
                raise AcquisitionError("Truncated TFRecord header")
            length, expected = struct.unpack("<QI", header)
            if masked_crc(header[:8]) != expected:
                raise AcquisitionError("Invalid TFRecord length CRC32C")
            if length > max_record_bytes:
                raise AcquisitionError("TFRecord exceeds per-record inspection budget")
            payload, crc = stream.read(length), stream.read(4)
            if len(payload) != length or len(crc) != 4:
                raise AcquisitionError("Truncated TFRecord payload")
            if masked_crc(payload) != struct.unpack("<I", crc)[0]:
                raise AcquisitionError("Invalid TFRecord data CRC32C")
            yield payload


def protobuf_wire_fields(payload, max_fields=100):
    """Structural inspection only: numbers/wire types, no guessed field names."""
    pos, result = 0, []

    def varint():
        nonlocal pos
        value = 0
        for shift in range(0, 70, 7):
            if pos >= len(payload):
                raise AcquisitionError("Truncated protobuf varint")
            byte = payload[pos]
            pos += 1
            value |= (byte & 127) << shift
            if byte < 128:
                return value
        raise AcquisitionError("Oversized protobuf varint")

    while pos < len(payload) and len(result) < max_fields:
        tag = varint()
        field, wire = tag >> 3, tag & 7
        if not 0 < field < 2**29:
            raise AcquisitionError("Invalid protobuf field number")
        if wire == 0:
            varint()
        elif wire in (1, 5):
            pos += 8 if wire == 1 else 4
        elif wire == 2:
            length = varint()
            pos += length
        else:
            raise AcquisitionError("Unsupported protobuf wire type for bounded inspection")
        if pos > len(payload):
            raise AcquisitionError("Truncated protobuf field")
        result.append({"field_number": field, "wire_type": wire})
    return {"fields": result, "complete": pos == len(payload), "message_type": "UNKNOWN"}


def describe(value, depth=0):
    """Bound output size; do not dump trajectories or untrusted textual fields."""
    if depth > 3:
        return {"type": type(value).__name__}
    if isinstance(value, dict):
        return {str(k): describe(v, depth + 1) for k, v in list(value.items())[:50]}
    if isinstance(value, list):
        return {"type": "list", "length": len(value), "first": describe(value[0], depth + 1) if value else None}
    return {"type": type(value).__name__}


def inspect_file(path, max_records=8):
    path = Path(path)
    report = {"path": str(path), "bytes": path.stat().st_size, "sha256": hashes(path)["SHA-256"],
              "semantic_schema": "UNVERIFIED", "real_scenarios_parsed": 0,
              "units": None, "native_dt_seconds": None, "cutoff": None}
    with path.open("rb") as stream:
        prefix = stream.read(512)
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            report.update(container="ZIP", member_count=len(members),
                          members=[{"name": m.filename, "uncompressed_bytes": m.file_size} for m in members[:50]])
    elif tarfile.is_tarfile(path):
        with tarfile.open(path, "r|*") as archive:
            report.update(container="TAR", members=[])
            for index, member in enumerate(archive):
                if index >= 50:
                    break
                report["members"].append({"name": member.name, "uncompressed_bytes": member.size})
    elif prefix[:4] == b"PAR1":
        import pyarrow.parquet as pq
        file = pq.ParquetFile(path)
        report.update(container="PARQUET", rows=file.metadata.num_rows, arrow_schema=str(file.schema_arrow))
    elif prefix.lstrip().startswith((b"[", b"{")):
        import ijson
        # Event inspection is streaming even for a single enormous record.
        events = []
        with path.open("rb") as stream:
            for index, (key, event, value) in enumerate(ijson.parse(stream)):
                if index >= 200:
                    break
                events.append({"path": key[:200], "event": event})
        report.update(container="JSON", first_events=events)
    elif len(prefix) >= 12 and masked_crc(prefix[:8]) == struct.unpack("<I", prefix[8:12])[0]:
        messages = []
        for record in tfrecords(path, max_records):
            try:
                wire = protobuf_wire_fields(record)
            except AcquisitionError as error:
                wire = {"error": str(error), "message_type": "UNKNOWN"}
            messages.append({"payload_bytes": len(record), "wire": wire})
        report.update(container="TFRECORD", records_crc_verified=len(messages), messages=messages,
                      note="CRC framing does not identify tf.train.Example versus Scenario or another proto")
    else:
        report.update(container="UNKNOWN", note="No unsafe pickle/object deserialization; inspect official schema first")
    return report


class I24MSDAdapter:
    """Separate acquisition/inspection boundary for the genuine scenario release.

    This is an explicitly INCOMPLETE semantic adapter, not a fallback parser.
    `scenes()` must stay blocked until authentic records, units, timing and maps
    are verified and an evidence-pinned conversion is implemented and tested.
    """

    def __init__(self, manifest):
        from . import DOI
        if manifest.get("doi") != DOI:
            raise AcquisitionError("I24MSDAdapter requires the I24-MSD DOI; continuous I24 is a different source")
        self.manifest = manifest

    def inspect(self):
        reports = []
        for entry in self.manifest.get("files", []):
            if not entry.get("downloaded_bytes"):
                continue
            path = Path(entry["local_path"])
            measured = verify_file(path, entry)
            if entry.get("hashes") != measured:
                raise AcquisitionError("Acquired content changed since manifest creation")
            reports.append(inspect_file(path))
        if not reports:
            raise AcquisitionError("No checksum-verified I24-MSD file bytes available")
        return reports

    def scenes(self):
        self.inspect()
        raise SchemaUnverified(
            "I24-MSD semantic schema remains unverified. Inspect actual records plus release codebook, "
            "then implement/test the evidence-pinned RoadsideScene mapping. Training is blocked; "
            "a .tfrecord suffix or successful download is not a verified multi-agent scene."
        )
