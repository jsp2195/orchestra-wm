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
    """Harvard release Scenario-proto adapter, verified on CRC-checked records.

    Native 91 x 0.1 s records supply a current_time_index=10 cutoff. The wire
    schema is the vendored Waymo Scenario subset, NOT tf.train.Example. No Waymo
    data, autonomous ego controls, or continuous-I24 units are implied.
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

    def scenes(self, native_files=None):
        if not native_files:
            self.inspect()
            raise SchemaUnverified("I24-MSD semantic schema remains unverified for these inputs; supply verified extracted Scenario TFRecords")
        scenes = []
        for path in native_files:
            for raw in tfrecords(path, max_records=10000):
                scene = self.record(raw, str(path))
                if scene is not None:
                    scenes.append(scene)
        return scenes

    def record(self, raw, source_file, kind="REAL_I24"):
        import hashlib
        import numpy as np
        from orchestra_wm.i24.schema import RoadsideScene, RoadMap
        from .protos.scenario_pb2 import Scenario
        proto = Scenario.FromString(raw)
        original = len(proto.SerializeToString())
        proto.DiscardUnknownFields()
        if original != len(proto.SerializeToString()):
            raise SchemaUnverified("Unknown protobuf fields: inspect the release schema before conversion")
        time = np.array(proto.timestamps_seconds, dtype=np.float64)
        if len(time) != 91 or not np.allclose(np.diff(time), .1, atol=1e-5):
            raise SchemaUnverified("Unexpected native timing; do not silently reuse the 91-step protocol")
        if not proto.HasField("current_time_index") or proto.current_time_index != 10:
            raise SchemaUnverified("Unexpected source cutoff")
        if not proto.scenario_id or not 1 <= len(proto.tracks) <= 32:
            raise SchemaUnverified("Missing scenario ID or unexpected vehicle count")
        ids = tuple(str(t.id) for t in proto.tracks)
        if len(set(ids)) != len(ids):
            raise SchemaUnverified("Duplicate track identity")
        n = len(ids)
        xy = np.zeros((91, n, 2), dtype=np.float64)
        velocity = np.zeros_like(xy)
        size = np.zeros_like(xy)
        heading = np.zeros((91, n), dtype=np.float32)
        valid = np.zeros((91, n), dtype=bool)
        required = ("center_x", "center_y", "velocity_x", "velocity_y", "length", "width", "heading", "valid")
        for j, track in enumerate(proto.tracks):
            if len(track.states) != 91 or not track.HasField("id"):
                raise SchemaUnverified("State/time or identity contract mismatch")
            for i, state in enumerate(track.states):
                if not state.valid:
                    continue
                if not all(state.HasField(k) for k in required):
                    raise SchemaUnverified("Missing required native kinematics/dimensions; no fabricated fields")
                xy[i, j] = state.center_x, state.center_y
                velocity[i, j] = state.velocity_x, state.velocity_y
                size[i, j] = state.length, state.width
                heading[i, j] = state.heading
                valid[i, j] = True
        if not np.isfinite(xy).all() or not np.isfinite(velocity).all() or np.any(size[valid] <= 0):
            raise SchemaUnverified("Nonfinite kinematics or invalid dimensions")
        history_valid = valid[:11]
        if history_valid.any() is False or not history_valid.any():
            return None
        # Eligibility depends ONLY on the observed history, never forecast labels.
        if history_valid.any(0).sum() < 2:
            return None
        sign = -1. if np.median(velocity[:11, :, 0][history_valid]) < 0 else 1.
        origin = float(np.median(xy[:11, :, 0][history_valid]))
        lane_centers, source_map = [], []
        for feature in proto.map_features:
            typ = feature.WhichOneof("feature_data")
            value = getattr(feature, typ) if typ else None
            if value is not None and hasattr(value, "polyline"):
                points = [[p.x, p.y, p.z] for p in value.polyline]
                source_map.append({"id": str(feature.id), "type": typ, "polyline_xyz_m": points})
                if typ == "lane" and len(points) > 1:
                    lane_centers.append(float(np.median(np.array(points)[:, 1])))
        lane_centers = sorted(set(round(v, 4) for v in lane_centers))
        if len(lane_centers) < 2:
            raise SchemaUnverified("Insufficient native static lane geometry")
        gaps = np.diff(lane_centers)
        width = float(np.median(gaps[gaps > 1]))
        if not 2 < width < 6:
            raise SchemaUnverified("Lane geometry does not support the metres interpretation")
        # Numerical field extent only; never label these bounds physical road edges.
        road = RoadMap(-1000., 1000., tuple(lane_centers), width, bins=8, confidence=0.,
                       coordinate_frame="s=travel_sign*(native center_x-origin), d=native center_y; metres")
        values = np.zeros((91, n, 8), dtype=np.float32)
        values[:, :, 0] = sign * (xy[:, :, 0] - origin)
        values[:, :, 1] = xy[:, :, 1]
        values[:, :, 2] = sign * velocity[:, :, 0]
        values[:, :, 3] = velocity[:, :, 1]
        values[:, :, 6:] = size
        for i in range(1, 91):
            usable = valid[i] & valid[i-1]
            values[i, usable, 4:6] = (values[i, usable, 2:4] - values[i-1, usable, 2:4]) / (time[i] - time[i-1])
        values[~valid] = 0
        selected = np.arange(0, 91, 2)  # causal decimation; no interpolation across cutoff
        provenance = {"kind": kind, "dataset": "I24-MSD", "doi": self.manifest["doi"],
                      "source_hash": hashlib.sha256(raw).hexdigest(), "source_file": source_file,
                      "kinematics": "source SI center positions/velocity; backward finite-difference acceleration",
                      "native_dt_seconds": float(np.median(np.diff(time))), "native_steps": 91,
                      "source_cutoff_index": 10, "context_steps": 6, "target_dt_seconds": .2,
                      "x_origin_m": origin, "travel_sign": sign, "source_heading_rad": heading[selected].tolist(),
                      "source_map": source_map, "source_date": "2022-11-24",
                      "confidence_available": False, "macro_census_valid": False,
                      "physical_boundary_known": False, "source_split": None,
                      "mask_semantics": "source valid = existence/usable track, EMULATED observation masks separate; confidence zero means unavailable"}
        return RoadsideScene(proto.scenario_id, "I24-MSD/2022-11-24", provenance, time[selected], ids,
                             values[selected], valid[selected], valid[selected].copy(),
                             np.zeros_like(valid[selected], dtype=np.float32), road,
                             np.array([t.object_type for t in proto.tracks]), ids).validate()
