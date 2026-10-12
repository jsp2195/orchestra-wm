from pathlib import Path
import numpy as np
import pytest
import torch
from orchestra_wm.i24_phase4e_colab import training, training_checkpoint


def test_actual_optimizer_rng_and_best_pair_resume_after_disconnect(tmp_path,monkeypatch):
    repository=Path.cwd()
    (tmp_path/'orchestra_wm').symlink_to(repository/'orchestra_wm',target_is_directory=True)
    monkeypatch.chdir(tmp_path)
    cache=Path('cache');cache.mkdir();(cache/'manifest.json').write_text('{}')
    torch.manual_seed(32);k=2
    item=dict(aggregate=torch.randn(25,k,15),micro=torch.randn(25,k,8),
              reference=torch.tensor([[12.,20.,864.],[15.,15.,810.]]),positions=torch.rand(k,2),
              adjacency=torch.ones(k,k)/k,target=torch.ones(100,k,3)*10,support=torch.ones(100,k,dtype=torch.bool))
    items=[(dict(split='train',number=0,regime=r),item) for r in ['dense','sparse50','outage']]
    items.append((dict(split='validation',number=0,regime='dense'),item))
    monkeypatch.setattr(training,'load_cache',lambda p:({},items))
    monkeypatch.setattr(training,'verify_preservation',lambda *args:None)  # Explicit small numerical fixture only.
    monkeypatch.setattr(training_checkpoint,'backup_run',lambda *args:None)
    training.train(cache,'uninterrupted',seed=101,max_updates=3)
    original=training.atomic_checkpoint
    def disconnect(path,state):
        original(path,state)
        if Path(path).name=='last.pt' and state['step']==1:raise RuntimeError('disconnect before best.pt')
    monkeypatch.setattr(training,'atomic_checkpoint',disconnect)
    with pytest.raises(RuntimeError,match='disconnect'):training.train(cache,'resumed',seed=101,max_updates=3)
    assert not Path('resumed/checkpoints/best.pt').exists()
    monkeypatch.setattr(training,'atomic_checkpoint',original)
    training.train(cache,'resumed',seed=101,max_updates=3)
    first=torch.load('uninterrupted/checkpoints/last.pt',weights_only=False)
    second=torch.load('resumed/checkpoints/last.pt',weights_only=False)
    for variant in ['full','aggregate']:
        for name,value in first['models'][variant].items():assert torch.equal(value,second['models'][variant][name])
        for index,state in first['optimizers'][variant]['state'].items():
            for name,value in state.items():assert torch.equal(value,second['optimizers'][variant]['state'][index][name])
    assert first['numpy_rng']==second['numpy_rng']
    assert torch.equal(first['torch_rng'],second['torch_rng'])
    assert first['rows']==second['rows']
    assert torch.load('resumed/checkpoints/best.pt',weights_only=False)['step']==first['best_step']
