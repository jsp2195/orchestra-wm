# Phase 4C private artifact recovery

Git recovers code, aggregate metrics and figures. It does not recover the private
canonical data, checkpoints or exact prediction arrays. Current-worker filesystem
persistence in a future task is not guaranteed. Retain the supplied private Drive
manifests and original Phase 4B recovery bundles outside the worker.

The original bundles remain unchanged:

| Bundle | SHA256 |
|---|---|
| original_pilot_recovery.tar.gz | `2087b14b8b92b2c0b22ba034b1ee8925597a2d55ede4897bca8ca6baf35d3e8d` |
| expanded_prepared_recovery.tar.gz | `f5002c23bc752124adcb7432df1e081a301be920be947cae4c6644e0b07234dd` |

New complete-recovery packages copy existing experiment bytes without regenerating
data or inference. The main package contains its checkpoints/optimizer/RNG states,
curves, validation and development predictions, metrics, diagnostics and gallery.
The joint package contains all three joint models and both evaluated partitions,
curves, gallery, private interactive payload, scientific report and protocol.
Small checkpoint-only and standalone HTML backups are also available separately.

| Package | Parts | Exact archive bytes | SHA256 |
|---|---:|---:|---|
| matched complete | 12 | 297777348 | `14e8ce96bf8777975b52d7327b05c09f9e3631ac4cfea3c97ef7866d911bed48` |
| joint complete | 28 | 682736269 | `e229b780d0538031f0d9f3fdc92840dc458815555ddd7c458174252b0c2e5a1a` |

Parts are at most 24 MiB. The private manifests contain ordered names, exact byte
ranges, per-part hashes and verified Drive IDs. Accept a remote package only with
`COMPLETE_AUTHENTICATED_READBACK_RECONSTRUCTION_SHA256_VERIFIED` in its manifest.
The public preservation receipt records verification hashes without disclosing
private Drive IDs. Owner-only permission metadata is checked during upload.

Download every named part from the private manifest, retaining its Drive filename.
If Drive downloads a collection ZIP, first unpack that collection into an empty
directory. These commands assume a POSIX shell and original Drive filenames:

```sh
# Run in the directory containing the downloaded parts; outputs must not exist.
test ! -e matched_recovery.tar.gz && cat ORCHESTRA_phase4c_matched_complete_recovery.tar.gz.part??? > matched_recovery.tar.gz
test ! -e joint_recovery.tar.gz && cat ORCHESTRA_phase4c_joint_complete_recovery.tar.gz.part??? > joint_recovery.tar.gz
sha256sum matched_recovery.tar.gz joint_recovery.tar.gz
```

The full hashes must exactly equal the table above. Never extract an archive that
fails validation. Check every part against its manifest if the concatenated hash
does not match; do not discard otherwise verified parts. On macOS use
`shasum -a 256` instead of `sha256sum`.

Restore archives only into a new recovery directory first. Review their relative
paths and match source/data/config/checkpoint fingerprints before copying missing
files into the checkout. Do not overwrite a newer run or existing source data.
The private original preservation receipt belongs at
`work/phase4b/private_backup_receipt.json`; it is included in the separately
verified main checkpoint-only backup. The receipt is evidence of authenticated
readback, not a substitute for the actual restored dataset and its matching hashes.

The [scientific report](../outputs/i24_phase4c/PHASE4C_REPORT.md) contains exact
completed-run verification/resume commands. Use the same `/workspace/orchestra-wm`
root because recorded source fingerprints include paths. Do not edit fingerprints
to accommodate a changed model, configuration or training budget. Continued
training requires a new namespace and explicit checkpoint warm-start provenance.

The standalone HTML files are private because they embed trajectory arrays.
Download and open them locally in a browser; Drive's preview does not host their
JavaScript application. Browser verification compared the payloads with exact
evaluated arrays and exercised all advertised controls.
