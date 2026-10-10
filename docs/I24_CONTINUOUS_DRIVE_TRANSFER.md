# Authorized private Google Drive transfer — Phase 4

## Status and one required authorization step

The user supplied these original-I24 archive references. They have **not yet been independently accessed from this cloud environment**:

| User-declared file | Drive ID | User-declared bytes |
|---|---|---:|
| 11-21-2022.zip | 1fbWITAB78jWrm0mQr_O4TX52aRxDMpLB | 6,248,968,002 |
| auxiliary_information.zip | 16zYgDwdljmgyXrYsjPIkL-vw2_K9PsTn | 4,917,485 |

Folder ID: `1DwPNI6tVZqzYEtiI381gyDboe7KVbCFF`. `raw.zip` is explicitly excluded (separate Zenodo source).

Google Drive plugin installation was user-confirmed but returned `completed:false`; no Drive tool became callable. `rclone v1.60.1-DEV` is installed but has no default configuration. The injected Google application-credential file is empty JSON; no usable account authorization was established. Cloud environment secret metadata contains no bound Drive token. Do not assume ChatGPT's Drive authorization is inherited by the shell. An unauthenticated request to the documented Google Drive v3 file metadata endpoint returned HTTP403; its body/headers were not logged and this is not proof of private-file access.

**Required action:** in this cloud environment's secure settings, supply the saved **`GOOGLE_DRIVE_ACCESS_TOKEN`** personal secret with a newly issued Google OAuth access token authorized for these files and read-only Drive scope `https://www.googleapis.com/auth/drive.readonly`, using your approved Google OAuth client/credential workflow. Save the configuration and publish the environment as requested by the onboarding UI. The secret is restricted to **`www.googleapis.com`**; the existing domain allowlist is preserved. Do not paste the value in chat, source files or shell commands. No service-account private key, account password, refresh token or public sharing is requested. If your approved credential workflow cannot issue that access token, complete the pending Drive connector setup instead; binary download support would still need verification. Never make the folder public.

The draft save is confirmed; runtime secret delivery and authenticated binary transfer are **unverified**. A Google login alone is not this binding. Access tokens expire; refresh/reissue through the same secure setting if necessary. The script deliberately does not persist or refresh credentials.

## Budget and executable transfer

Before transfer the workspace had 28,761,337,856 bytes free. Both archives total **6,253,885,487 bytes**, leaving **12,507,452,369 bytes** for extraction/scratch after a 10,000,000,000-byte free-space reserve. This is a capacity calculation, not a measured download. The user explicitly requested the 6.25 GB archive after identifying the older 3 GB cap. The new **archive transfer** budget is narrowly bounded at 6,300,000,000 bytes; the existing extracted-source pilot cap is unchanged. Never extract the full archive blindly. Read its member inventory and choose a bounded source subset before creating acquisition metadata; a later justified limit change may be necessary based on real member sizes.

```sh
cd /workspace/orchestra-wm
uv run python scripts/acquire_i24_continuous_drive.py \
  --data-root data/i24_continuous/raw \
  --max-download-bytes 6300000000
```

This exact command was executed and exited **2** because the secure token binding is missing. No bytes downloaded. With authorization it will:

1. Fetch metadata for only the two supplied IDs via the documented Drive v3 API. Check exact names/sizes, binary MIME type, `canDownload` and Drive MD5.
2. Preserve partial files and an immutable metadata/version stamp. Resume only if the server returns HTTP206 with the exact expected Content-Range. Reject unexpected redirects without forwarding authorization; a legitimate new redirect host requires explicit review/configuration.
3. Stream to ignored `.part` files, enforcing byte limits and the disk reserve. Retry metadata/open requests on transient429/5xx with bounded backoff. Interrupted body transfers preserve partial bytes and resume on the next invocation; they are not marked complete.
4. Verify final byte counts, MD5 and SHA256; inspect ZIP member paths and compressed/uncompressed sizes without extraction. Reject traversal and symlinks. Atomically rename the verified archive. Existing verified archives are hashed/reused, never redownloaded or deleted.
5. Write ignored `data/i24_continuous/raw/DRIVE_ARCHIVE_MANIFEST.json`. A successful transfer proves archive integrity, **not source release, corrected timestamps, continuity, units, authorized portal provenance or real model training**. Those require inspected member contents. ZIP CRCs are inventoried but per-member decompression/CRC validation occurs only when selected members are subsequently extracted.

Official API routes used: `GET https://www.googleapis.com/drive/v3/files/{fileId}?fields=...` and the same resource with `alt=media`. No undocumented endpoints, credential scraping or public-download bypasses. No archive contents or private signed URLs are published.

After verified transfer, continue source/schema inspection, bounded extraction, source QC, split registration and actual pipeline implementation. The current import command remains a gate, not a training implementation. All training/evaluation/GIFs remain BLOCKED until authentic continuous records pass review. Preserve Harvard data and all earlier results.
