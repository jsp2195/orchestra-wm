# Colab handoff validation

Starting research commit: `2cf2330533f6630425116b636f1d7ba878862653`.
Notebook implementation pin: `72286edda0ca9a3bb620f9251bac0522869e1710`.
The notebook is output-free and uses Colab's built-in Drive mount; it does not
read OAuth secret variables or call the source/preservation Drive HTTP adapters.
The pinned implementation is a descendant of the requested research commit.

The final regression run passed **143 tests, 8 skipped** in 17.52 seconds:

```bash
uv run --no-sync pytest --import-mode=importlib -q
```

Behavioral verification includes authentic ZIP-format fixtures with both nested
integrity checks and original MD5 rejection, Drive shortcut traversal without
cycles, archive reconstruction by ordered part hashes, non-overwriting extraction,
malicious member rejection, corrupted-backup rejection, and enforcement of the
10 GB reserve before writing. Interrupted native conversion matches the original
frozen converter's scene selection, values, masks and field arrays. A real small
ResidualFields numerical fixture reproduces uninterrupted weights, optimizer
state, RNG states and training rows after interruption between last/best writes.
These fixtures do not constitute real-data training or evaluation.

Promotion tests reject a single learned model failing the persistence comparison,
missing outcome rows, an unresolved stability heuristic, changed selected weights,
or non-Nov25 outcomes. Source-date routing refuses reingestion of completed Nov21/22.
Fresh workspace initialization, immutable code locking, preserved interrupted
caches, notebook JSON/Python syntax, empty cell outputs and the implementation pin
were checked. The repository is public, so its Colab checkout needs no GitHub secret.

All **863** files tracked at the starting commit were checked byte-for-byte; only
an append to `.gitignore` differs. No original scientific implementation, report,
protocol, completed-data manifest, seed, date assignment or recorded result changed.
Only new adapters, notebook, documentation, safe source identity metadata and tests
are committed. Private source IDs, datasets, arrays, credentials and weights are
not part of the new Git changes.

**Not executed here:** the user's interactive Colab authorization, mounted real ZIP
reads in Colab, full multi-date training, validation and independent forecasts.
The notebook runs those stages after authorization and records their actual status
in private Drive. It makes no new scientific outcome claim. See
`docs/I24_PHASE4E_COLAB.md` for recovery, preservation limits and frozen controls.
