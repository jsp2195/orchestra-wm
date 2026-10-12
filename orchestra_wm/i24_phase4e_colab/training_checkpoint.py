"""Keep complete model/optimizer/RNG checkpoints on Drive and verified snapshots."""
import io
from pathlib import Path
import torch
from .storage import atomic_bytes, snapshot

BACKUP_ROOT = None


def backup_run(run):
    if BACKUP_ROOT is None:
        raise RuntimeError('Mounted private backup root must be configured')
    return snapshot([run], BACKUP_ROOT, Path(run).name)


def atomic_checkpoint(path, payload):
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    atomic_bytes(path, buffer.getvalue())
    # Snapshot after best.pt is saved as well, so an improved pair is consistent.
    if Path(path).name == 'best.pt' or (payload['step'] % 100 == 0 and payload.get('best_step') != payload['step']):
        backup_run(Path(path).parent.parent)
