import numpy as np
from orchestra_wm.i24_continuous.evaluation import energy_score


def test_energy_score_uses_joint_target_and_offdiagonal_sample_pairs():
    assert energy_score(np.array([[0.,0.],[0.,0.]]),np.array([0.,0.]))==0
    assert np.isclose(energy_score(np.array([[0.,0.],[2.,0.]]),np.array([1.,0.])),0)
    assert np.isclose(energy_score(np.array([[2.,2.]]),np.array([0.,0.])),2)
    assert energy_score(np.empty((2,0)),np.empty(0)) is None
