# Phase 4B preservation and secure recovery

The six original checkpoints and all source archives remain in place. The
source-hashed inventory is `outputs/i24_phase4b/audit/preservation.json`.
`work/phase4b/original_pilot_recovery.tar.gz` contains the original prepared
data, optimizer/RNG checkpoints, saved evaluations and reports; every tar member
is checked against the inventory. This is a **local recovery copy**, not a durable
remote backup. The overlay survived this resumed task; future-task persistence
is not guaranteed. No persistent volume or writable private store is configured.

Before substantial new training, copy this compact bundle and the preservation
manifest to an approved private durable store and verify its SHA256 after reading
it back. Keep the original source ZIPs until explicitly authorized to remove
them after successful backup. Git recovers code/reports only. A reproduced model
is not a recovered checkpoint.

Secure setup for an approved Google Drive artifact folder:

1. In your own browser, create/select a **private** folder approved for these
   restricted research artifacts. Do not change sharing to public.
2. Authorize a separate OAuth client/token with `drive.file` scope for files
   created by that application. Use the application's authorized folder picker
   to grant access to the artifact folder if required. Keep the source-data token
   read-only. Obtain and refresh tokens outside chat.
3. In the cloud environment's secure network-secret settings, bind the separate
   credential as `GOOGLE_DRIVE_ARTIFACT_TOKEN`, with substitution permitted only
   for `www.googleapis.com`. Save and publish the environment. Never paste it
   into chat, commands, repository files or logs.
4. Provide only the non-secret approved destination folder ID. Verify current
   capability and a small authenticated upload/readback before transferring the
   bundle using resumable Drive uploads. Record returned file ID, size and MD5,
   and verify SHA256 on authenticated readback. Do not claim backup before that.

An approved durable private bucket/volume with scoped write identity is also
suitable. No upload was attempted using the existing read-only token.

The metadata-only Drive listing returned HTTP401 once during Phase 4B. To restore
source access, refresh `GOOGLE_DRIVE_ACCESS_TOKEN` in those secure settings and
publish. No unsuccessful request was repeated, and no archive was redownloaded.

Current-worker verification:

```sh
cd /workspace/orchestra-wm
uv run python scripts/preserve_i24_phase4b.py
```

On a fresh worker, recover the private bundle first, verify its recorded SHA256,
inspect tar member paths, and restore into an empty checkout of the recorded
commit. Never overwrite surviving artifacts. The large archive remains separately
recoverable through its authenticated Drive ID and recorded hashes if absent.
