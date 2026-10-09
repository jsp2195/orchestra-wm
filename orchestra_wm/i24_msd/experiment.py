"""Bounded real-clip pilot using the preserved passive Phase-3 model.

Macro feedback is disabled: curated vehicle selections are not roadway censuses.
All inference states recur through prior imagination, never future observations.
"""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from orchestra_wm.i24.data import observations
from orchestra_wm.i24.model import RoadsideWorldModel, observe_history
from orchestra_wm.i24.training import atomic_checkpoint
from .acquisition import AcquisitionError, atomic_json, hashes


class MSDWorldModel(RoadsideWorldModel):
    def observe(self, state, observation, observation_mask, dt):
        # No population-density information is inferred from curated agent counts.
        observation = {**observation, 'fields': torch.zeros_like(observation['fields']),
                       'field_support': torch.zeros_like(observation['field_support'])}
        result = super().observe(state, observation, observation_mask, dt)
        return replace(result, fields=torch.zeros_like(result.fields), field_memory=torch.zeros_like(result.field_memory))

    def _transition(self, state, generator=None, posterior_target=None):
        result, prediction = super()._transition(state, generator, posterior_target)
        # No verified observation footprint exists. Fixed-cohort existence is not
        # inferred from the arbitrary numerical field extent or future labels.
        result = replace(result, valid=state.valid & state.seen,
                         fields=torch.zeros_like(result.fields), field_memory=torch.zeros_like(result.field_memory))
        prediction['valid'] = result.valid
        return result, prediction


def split_scenes(scenes, max_scenes=96, seed=7301):
    """Group all source-shared identities before subsampling; never split siblings."""
    if len({s.scene_id for s in scenes}) != len(scenes):
        raise AcquisitionError('Duplicate scenario identifiers')
    if len({s.provenance['source_hash'] for s in scenes}) != len(scenes):
        raise AcquisitionError('Duplicate scenario bytes')
    parent = list(range(len(scenes)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    identity = {}
    for i, scene in enumerate(scenes):
        for track in scene.track_ids:
            key = (scene.session_id, track)
            if key in identity:
                parent[find(i)] = find(identity[key])
            identity[key] = i
    groups = {}
    for i, scene in enumerate(scenes): groups.setdefault(find(i), []).append(scene)
    ranked = sorted(groups.values(), key=lambda group: hashlib.sha256((str(seed)+','.join(sorted(s.scene_id for s in group))).encode()).hexdigest())
    if len(ranked) < 6: raise AcquisitionError('Too few disjoint source-track groups')
    # Fixed partition allocation, before any training/test prediction metrics.
    bounds = (int(.6*len(ranked)), int(.8*len(ranked)))
    splits = {'train': [], 'validation': [], 'test': []}
    for i, group in enumerate(ranked):
        name = 'train' if i < bounds[0] else 'validation' if i < bounds[1] else 'test'
        group_id = hashlib.sha256(','.join(sorted(s.scene_id for s in group)).encode()).hexdigest()[:16]
        for scene in group:
            scene.provenance['overlap_group'] = group_id
        splits[name].extend(group)
    for name, frac in [('train', .6), ('validation', .2), ('test', .2)]:
        splits[name] = splits[name][:max(2, int(max_scenes*frac))]
        if not splits[name]: raise AcquisitionError('Empty split')
    ids = {name: {t for s in group for t in s.track_ids} for name, group in splits.items()}
    for x, y in [('train','validation'),('train','test'),('validation','test')]:
        if ids[x] & ids[y]: raise AcquisitionError('Source track overlap across splits')
    # Additional geometry fingerprint: reject identical observed histories with
    # renamed IDs. This is a guard, not proof against all near-overlap sources.
    signatures = {}
    for name, group in splits.items():
        for s in group:
            absolute = s.values[:6, :, :2].astype(np.float64).copy()
            absolute[:,:,0] = absolute[:,:,0]*s.provenance['travel_sign']+s.provenance['x_origin_m']
            sig = hashlib.sha256(np.round(absolute, 2).tobytes()+s.existence[:6].tobytes()).hexdigest()
            if sig in signatures and signatures[sig] != name: raise AcquisitionError('Duplicate observed geometry across splits')
            signatures[sig] = name
    return splits


def load_model(path):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    cfg = ckpt['config']
    model = MSDWorldModel(cfg['model_dim'], cfg['latent_dim'], ckpt['variant'], .2)
    model.load_state_dict(ckpt['model']); model.eval()
    return model, ckpt


def objective(model, scene, rng, horizon=10, validation=False):
    regime = 'dense' if validation else ['dense','dense','sparse50','outage'][int(rng.integers(4))]
    try:
        obs, cohort = observations(scene, 6, regime, int(rng.integers(100000)))
    except ValueError as error:
        if 'No historical visible tracks' not in str(error): raise
        obs, cohort = observations(scene, 6, 'dense', 0)  # no usable conditioning: dense training fallback
    state = observe_history(model, obs, scene.road)
    targets = torch.tensor(scene.values[6:6+horizon,cohort], dtype=torch.float32)[None]
    valid = torch.tensor(scene.existence[6:6+horizon,cohort])[None]
    gen = torch.Generator().manual_seed(int(rng.integers(2**30)))
    total = state.agents.new_zeros(())
    for t in range(horizon):
        previous = state
        state, prior = model.imagine_step(previous, generator=gen)
        mask = valid[:,t,:,None]; denom = mask.sum().clamp_min(1)
        # Position/speed scales physical, fixed before evaluating the test set.
        scale = torch.tensor([10.,3.,10.,3.])
        rollout = ((((prior['agents'][:,:,:4]-targets[:,t,:,:4])/scale)**2)*mask).sum()/denom
        if model.variant != 'deterministic':
            _, post = model._transition(previous, generator=gen, posterior_target=targets[:,t])
            acceleration = (targets[:,t,:,2:4]-previous.agents[:,:,2:4])/.2
            nll = ((.5*((acceleration-post['acc_mean'])/post['acc_std']).square()+post['acc_std'].log())*mask).sum()/denom
            rollout = rollout + .001*nll + .001*post['kl']
        total = total + rollout
    return total/horizon


def train(cfg, splits, out, fingerprint):
    torch.set_num_threads(cfg['threads'])
    out = Path(out); (out/'checkpoints').mkdir(exist_ok=True)
    metadata, all_rows = [], []
    begun = time.perf_counter()
    training_scenes = [s for s in splits['train'] if s.existence[6:16, s.detection[:6].any(0)].any()]
    if not training_scenes: raise AcquisitionError('No valid future training targets')
    for seed in cfg['training_seeds']:
        for variant in cfg['variants']:
            torch.manual_seed(seed); rng = np.random.default_rng(seed)
            model = MSDWorldModel(cfg['model_dim'], cfg['latent_dim'], variant, .2)
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.get('learning_rate', .001))
            path = out/'checkpoints'/f'{variant}-{seed}.pt'
            start, elapsed, rows = 0, 0., []
            if path.exists():
                loaded = torch.load(path, map_location='cpu', weights_only=False)
                if loaded['fingerprint'] != fingerprint: raise AcquisitionError('Checkpoint fingerprint changed; use new output')
                model.load_state_dict(loaded['model']); optimizer.load_state_dict(loaded['optimizer'])
                start, elapsed, rows = loaded['step'], loaded['elapsed'], loaded['rows']
                rng.bit_generator.state = loaded['numpy_rng']; torch.set_rng_state(loaded['torch_rng'])
            clock = time.perf_counter()
            for step in range(start+1, cfg['training_steps']+1):
                if time.perf_counter()-begun > cfg['max_wall_seconds']: raise AcquisitionError('Training wall budget reached; rerun to resume checkpoint')
                model.train(); optimizer.zero_grad()
                loss = objective(model, training_scenes[int(rng.integers(len(training_scenes)))], rng)
                if not torch.isfinite(loss): raise AcquisitionError('Nonfinite training loss')
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step()
                if step == 1 or step % 25 == 0 or step == cfg['training_steps']:
                    model.eval(); vrng = np.random.default_rng(991)
                    with torch.no_grad(): val = float(np.mean([float(objective(model, s, vrng, validation=True)) for s in splits['validation'][:6]]))
                    if not np.isfinite(val): raise AcquisitionError('Nonfinite validation loss')
                    rows.append({'variant':variant,'seed':seed,'step':step,'train_loss':float(loss.detach()),'validation_loss':val})
                    atomic_checkpoint(path, {'model':model.state_dict(),'optimizer':optimizer.state_dict(),'variant':variant,'seed':seed,
                        'config':cfg,'step':step,'fingerprint':fingerprint,'source_kind':'REAL_I24_MSD','elapsed':elapsed+time.perf_counter()-clock,
                        'rows':rows,'numpy_rng':rng.bit_generator.state,'torch_rng':torch.get_rng_state()})
                    print(f'{variant} step={step} loss={float(loss.detach()):.4f} validation={val:.4f}', flush=True)
            saved = torch.load(path, map_location='cpu', weights_only=False)
            all_rows.extend(rows)
            metadata.append({'variant':variant,'seed':seed,'parameters':sum(p.numel() for p in model.parameters()),'updates':saved['step'],
                             'seconds':saved['elapsed'],'checkpoint_sha256':hashes(path)['SHA-256'],'validation_loss':rows[-1]['validation_loss']})
    pd.DataFrame(all_rows).to_csv(out/'training.csv', index=False)
    atomic_json(out/'training_metadata.json', metadata)
    return metadata
